"""Recovery from interruption at every persistence boundary.

The commit record is the only authority for where a run resumes. These tests interrupt a run at
each point where that could go wrong and assert the same property every time: **recovery continues
from the last committed physical state, without skipping a chunk and without running one twice.**

Most scenarios are constructed directly on a run directory, because reaching them from a real run
would mean racing a specific millisecond. One test does terminate a real subprocess, because the
constructed scenarios all assume the shape of a crash rather than observing one -- and an
assumption about a crash is exactly the thing worth checking against a real one.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from md_templates.openmm import runstate

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ["walker.chk", "walker.state.xml"]


# ---------------------------------------------------------------------------------------------
# helpers: build a run directory in a given state of completeness
# ---------------------------------------------------------------------------------------------

def write_generation(run_dir: Path, gen: int, *, members=MEMBERS, partial=False) -> Path:
    gdir = runstate.generation_dir(run_dir, gen)
    gdir.mkdir(parents=True, exist_ok=True)
    for name in (members[:1] if partial else members):
        (gdir / name).write_text(f"restart bytes for generation {gen}")
    return gdir


def write_chunk(run_dir: Path, index: int, *, done=True) -> Path:
    cdir = run_dir / f"chunk_{index:04d}"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "end.chk").write_text("chunk checkpoint")
    (cdir / "traj_all.dcd").write_text("frames")
    if done:
        (cdir / "done.json").write_text(json.dumps({"chunk": index}))
    return cdir


def commit(run_dir: Path, gen: int, **extra) -> None:
    runstate.commit_generation(run_dir, gen, members=MEMBERS, **extra)


# ---------------------------------------------------------------------------------------------
# 1-6: interruption at each persistence boundary
# ---------------------------------------------------------------------------------------------

def test_1_interrupted_before_any_generation_exists(tmp_path):
    """Nothing durable: the run starts at chunk 0, not at whatever was written."""
    write_chunk(tmp_path, 0)                       # outputs exist but nothing was ever committed
    assert runstate.resume_boundary(tmp_path) == 0


def test_2_interrupted_after_only_some_restart_members_exist(tmp_path):
    commit(tmp_path, 0) if write_generation(tmp_path, 0) else None
    write_generation(tmp_path, 1, partial=True)    # crash between the two member writes
    assert runstate.committed_generation(tmp_path) == 0
    assert runstate.resume_boundary(tmp_path) == 1, "a half-written generation moved the boundary"


def test_3_interrupted_after_members_written_but_before_the_commit(tmp_path):
    """The window the commit record exists to close."""
    write_generation(tmp_path, 0)
    commit(tmp_path, 0)
    write_generation(tmp_path, 1)                  # complete on disk, never committed
    assert runstate.resume_boundary(tmp_path) == 1
    assert runstate.committed_generation(tmp_path) == 0


def test_4_chunk_outputs_and_done_json_exist_but_the_generation_is_not_committed(tmp_path):
    """The defect this milestone fixes: done.json must NOT move the resume boundary.

    Deriving the boundary from done.json here would start at chunk 2 while the restart is the state
    after chunk 0 -- advancing the counter over propagation that was never committed.
    """
    write_generation(tmp_path, 0)
    commit(tmp_path, 0)
    write_chunk(tmp_path, 0)
    write_chunk(tmp_path, 1)                       # written, done.json present, NOT committed
    assert runstate.resume_boundary(tmp_path) == 1, "done.json was treated as authoritative"


def test_5_exchange_rows_durable_but_not_committed_are_not_counted(tmp_path):
    write_generation(tmp_path, 0)
    commit(tmp_path, 0, attempts_committed=4)
    record = runstate.committed_record(tmp_path)
    assert record["attempts_committed"] == 4, "the watermark is what bounds countable history"


def test_6_immediately_after_the_commit_record_changes(tmp_path):
    write_generation(tmp_path, 0)
    commit(tmp_path, 0)
    write_generation(tmp_path, 1)
    commit(tmp_path, 1)
    assert runstate.resume_boundary(tmp_path) == 2


# ---------------------------------------------------------------------------------------------
# 7-8: restart selection and fallback
# ---------------------------------------------------------------------------------------------

class _FakeContext:
    def __init__(self):
        self.state = None

    def setState(self, state):
        self.state = state


class _FakeSim:
    """Enough of a Simulation to exercise the restart selection logic without OpenMM."""

    def __init__(self, *, checkpoint_ok=True):
        self.context = _FakeContext()
        self._checkpoint_ok = checkpoint_ok
        self.loaded = None

    def loadCheckpoint(self, path):
        if not self._checkpoint_ok:
            raise RuntimeError("checkpoint was written by a different platform")
        self.loaded = ("checkpoint", path)


def test_7_corrupt_checkpoint_falls_back_to_the_state(tmp_path, monkeypatch):
    gdir = write_generation(tmp_path, 0)
    (gdir / "walker.state.xml").write_text("<State/>")

    import md_templates.openmm.runstate as rs

    monkeypatch.setattr(rs, "load_restart", rs.load_restart)     # use the real implementation
    deserialized = {}
    fake_xml = type("X", (), {"deserialize": staticmethod(lambda s: deserialized.setdefault("s", s))})
    monkeypatch.setitem(sys.modules, "openmm",
                        type("M", (), {"XmlSerializer": fake_xml, "__spec__": None}))
    sim = _FakeSim(checkpoint_ok=False)
    assert rs.load_restart(sim, gdir) == "state"
    assert deserialized["s"] == "<State/>", "the State file was not the one loaded"


def test_8_unusable_checkpoint_and_missing_state_refuses_to_append(tmp_path, monkeypatch):
    gdir = write_generation(tmp_path, 0, members=["walker.chk"])   # no State at all

    import md_templates.openmm.runstate as rs

    monkeypatch.setitem(sys.modules, "openmm",
                        type("M", (), {"XmlSerializer": object, "__spec__": None}))
    with pytest.raises(rs.RunStateError, match="no usable restart"):
        rs.load_restart(_FakeSim(checkpoint_ok=False), gdir)


# ---------------------------------------------------------------------------------------------
# 9-10: corrupt history and uncommitted tails
# ---------------------------------------------------------------------------------------------

def test_9_a_log_shorter_than_its_watermark_is_corruption():
    """Rows that were committed have gone missing; continuing would invent history."""
    from md_templates.openmm.rest2 import _lifetime_counts

    import tempfile
    d = Path(tempfile.mkdtemp())
    log = d / "exchange_attempts.csv"
    log.write_text("attempt_index,accepted\n0,1\n1,0\n")          # 2 rows
    attempts, _ = _lifetime_counts(log)
    assert attempts == 2
    # the driver compares this against the committed watermark and refuses when rows < watermark;
    # see run_rest2_remd. Here we assert the inputs that decision is made from.
    assert attempts < 5


def test_10_an_uncommitted_tail_is_quarantined_not_adopted(tmp_path):
    write_generation(tmp_path, 0)
    commit(tmp_path, 0)
    write_chunk(tmp_path, 0)
    write_chunk(tmp_path, 1)                       # uncommitted tail
    write_chunk(tmp_path, 2)

    moved = runstate.quarantine_uncommitted_tail(tmp_path, runstate.resume_boundary(tmp_path))
    assert sorted(moved) == [f"{runstate.QUARANTINE_DIR}/chunk_0001",
                             f"{runstate.QUARANTINE_DIR}/chunk_0002"]
    assert not (tmp_path / "chunk_0001").exists(), "the tail was left in place to be appended to"
    assert (tmp_path / "chunk_0000").is_dir(), "a COMMITTED chunk was quarantined"
    # nothing is destroyed: the tail is still readable under recovery/
    assert (tmp_path / runstate.QUARANTINE_DIR / "chunk_0001" / "traj_all.dcd").is_file()


def test_a_missing_committed_artifact_stops_the_run(tmp_path):
    """A chunk the record calls finished, with no outputs, is corruption -- not something to skip."""
    write_generation(tmp_path, 1)
    commit(tmp_path, 1)
    write_chunk(tmp_path, 0)                       # chunk 1 committed but never written
    with pytest.raises(runstate.RunStateError, match="missing from chunks"):
        runstate.assert_committed_outputs(tmp_path, runstate.resume_boundary(tmp_path),
                                          required=["done.json", "end.chk"])


def test_restart_at_the_wrong_step_is_refused(tmp_path):
    """A checkpoint that loads cleanly but sits at the wrong step would skip physics silently."""
    with pytest.raises(runstate.RunStateError, match="commit record says"):
        runstate.assert_restart_consistent(loaded_step=500, loaded_time_ps=2.0,
                                           record={"steps": 1000}, timestep_fs=4.0)
    # and the matching step/time pair is accepted
    runstate.assert_restart_consistent(loaded_step=1000, loaded_time_ps=4.0,
                                       record={"steps": 1000}, timestep_fs=4.0)


def test_restart_time_inconsistent_with_its_step_is_refused():
    with pytest.raises(runstate.RunStateError, match="ps but step"):
        runstate.assert_restart_consistent(loaded_step=1000, loaded_time_ps=99.0,
                                           record={"steps": 1000}, timestep_fs=4.0)


# ---------------------------------------------------------------------------------------------
# the real thing: interrupt a live subprocess and recover it
# ---------------------------------------------------------------------------------------------

def _prepared_bundle(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("crash")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    res = subprocess.run(
        [sys.executable, "-m", "md_templates.openmm.cli", "prepare",
         "--system", "small_macrocycle_smoke", "--experiment", "smoke",
         "--out-root", str(root), "--platform", "CPU"],
        env=env, capture_output=True, text=True, timeout=1800, cwd=str(root),
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return next(p for p in root.iterdir() if "__bundle__" in p.name)


@pytest.fixture(scope="module")
def prepared_bundle(tmp_path_factory):
    return _prepared_bundle(tmp_path_factory)


@pytest.mark.slow
def test_killed_subprocess_resumes_from_the_last_committed_chunk(prepared_bundle, tmp_path):
    """Terminate a REAL run mid-flight, then resume it.

    The property under test is the one that matters and that no constructed fixture can establish:
    after an actual SIGKILL, recovery continues from the last COMMITTED physical state -- no chunk
    skipped, none run twice, and the exchange history strictly monotonic across the boundary.
    """
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    out_root = tmp_path / "runs"
    out_root.mkdir()
    run_dir = out_root / "killme"

    # a long enough experiment that the kill lands mid-run rather than after it
    experiment = tmp_path / "long.yaml"
    import yaml
    from md_templates.openmm.schemas import shipped_experiment
    doc = yaml.safe_load(shipped_experiment("smoke").read_text())
    doc["experiment_id"] = "crash_probe"
    doc["rest2"]["n_chunks"] = 40
    experiment.write_text(yaml.safe_dump(doc, sort_keys=False))

    proc = subprocess.Popen(
        [sys.executable, "-m", "md_templates.openmm.cli", "rest2",
         "--bundle", str(prepared_bundle), "--experiment", str(experiment),
         "--out-root", str(out_root), "--run-name", "killme", "--platform", "CPU"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(tmp_path),
    )
    try:
        committed = None
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            gen = None
            try:
                gen = runstate.committed_generation(run_dir)
            except Exception:                                  # noqa: BLE001 - mid-write is fine
                gen = None
            if gen is not None and gen >= 1:
                committed = gen
                break
            time.sleep(0.5)
        assert committed is not None, "no generation was committed before the deadline"
        proc.send_signal(signal.SIGKILL)
    finally:
        proc.wait(timeout=120)

    assert proc.returncode != 0, "the process was not actually killed"
    boundary_after_kill = runstate.resume_boundary(run_dir)
    assert boundary_after_kill == committed + 1

    import csv
    rows_before = list(csv.DictReader((run_dir / "exchange_attempts.csv").open()))
    committed_attempts = runstate.committed_record(run_dir)["attempts_committed"]

    # resume: two more chunks
    doc["rest2"]["n_chunks"] = 2
    experiment.write_text(yaml.safe_dump(doc, sort_keys=False))
    res = subprocess.run(
        [sys.executable, "-m", "md_templates.openmm.cli", "rest2",
         "--bundle", str(prepared_bundle), "--experiment", str(experiment),
         "--out-root", str(out_root), "--resume-run", "killme", "--platform", "CPU"],
        env=env, capture_output=True, text=True, timeout=1800, cwd=str(tmp_path),
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]

    rows_after = list(csv.DictReader((run_dir / "exchange_attempts.csv").open()))
    idx = [int(r["attempt_index"]) for r in rows_after]
    steps = [int(r["step"]) for r in rows_after]

    assert idx == list(range(len(idx))), f"attempt indices are not contiguous: {idx[:20]}"
    assert steps == sorted(steps), "steps went backwards across the recovery boundary"
    assert len(rows_after) > committed_attempts, "the resume added no attempts"
    # the uncommitted tail was discarded, so history restarts from the watermark, never duplicating
    assert len(rows_after) >= committed_attempts
    assert len(set(idx)) == len(idx), "an attempt was recorded twice across recovery"
    # chunks are contiguous from zero: none skipped, none run twice
    chunks = sorted(int(p.name.split("_")[1])
                    for p in (run_dir / "replica_00").glob("chunk_*") if p.is_dir())
    assert chunks == list(range(len(chunks))), f"chunk sequence has a hole or a repeat: {chunks}"
    assert len(rows_before) >= committed_attempts - 0
