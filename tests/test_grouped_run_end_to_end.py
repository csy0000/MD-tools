"""One real, complete grouped REST2 run, small enough to be a unit test.

WHY THIS EXISTS
    Every other test of the driver reads its source or binds one of its methods to a stub. That
    caught a great deal, and it missed two things it structurally cannot catch: a name in the
    completion path that is never bound, and an attribute used in the frame event that nothing
    ever creates. Both are invisible until the loop actually runs.

    So this runs it. Two states, a handful of exchanges, alanine dipeptide in vacuum on the CPU
    platform -- a few seconds -- through the real replica executor with a real group file.
    It is the cheapest thing that executes `_loop`, the frame commit, the completion record and
    the summary at once.

PLATFORM_POLICY_EXEMPTION: deliberately CPU and deliberately tiny. This tests the plumbing that
carries the numbers, not the numbers; the scientific validation is the ALA production run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import EXCHANGES, TAUS    # noqa: F401 - re-exported for readers

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"
#: The checkout's `src`, for subprocesses. NOT the package directory: putting that on
#: PYTHONPATH would let `remd/statistics.py` shadow the standard library.
SRC = Path(__file__).resolve().parents[1] / "src"
netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("openmm")

from md_tools.remd import amber_trajectory as amber


def _run(work, *extra):
    environment = dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", str(len(TAUS)),
         "-x", "exchange.nc", "-r", "restart.json", "--checkpoint", "checkpoint.nc",
         "-o", "run.out", "--rem", "rem.log", *extra],
        cwd=str(work), capture_output=True, text=True, timeout=1800, env=environment)


@pytest.fixture(scope="module")
def completed(prepared):
    """The finished run, plus its report. `-o` is where a stage's output goes, not the terminal."""
    work, n_atoms, _ = prepared
    result = _run(work)
    assert result.returncode == 0, (result.stdout[-4000:] + result.stderr[-4000:])
    return work, n_atoms, (work / "run.out").read_text(encoding="utf-8")


# --- the run finishes at all ----------------------------------------------------------------------

@pytest.mark.slow
def test_a_grouped_run_completes(completed):
    _, _, report = completed
    assert "run_status: completed" in report


@pytest.mark.slow
def test_the_completion_manifest_is_written(completed):
    work, _, _ = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    assert record["run_status"] == "completed"
    assert record["exchanges_committed"] == EXCHANGES


# --- section 3/5: the state trajectories exist and are Amber files ----------------------------------

@pytest.mark.slow
def test_one_trajectory_per_state_is_written(completed):
    """The defect this test was written for: `self.trajectories` was used by the frame event and
    never created, so a real run raised AttributeError at the first frame."""
    work, _, _ = completed
    for index in range(len(TAUS)):
        assert (work / amber.state_trajectory_name(index)).is_file()
    assert not (work / f"remd{len(TAUS)}.nc").exists()


@pytest.mark.slow
def test_each_state_trajectory_records_its_own_state_and_tau(completed):
    work, n_atoms, _ = completed
    for index, tau in enumerate(TAUS):
        seen = amber.read_frames(work / amber.state_trajectory_name(index))
        assert seen["conventions"] == "AMBER"
        assert seen["state_index"] == index
        assert seen["tau"] == pytest.approx(tau)
        assert seen["n_atoms"] == n_atoms


@pytest.mark.slow
def test_every_state_holds_the_committed_number_of_frames(completed):
    """The marker in the authoritative record counts these rows, so they must agree with it."""
    work, _, _ = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    expected = record["whole_frames"]
    counts = {index: amber.read_frames(work / amber.state_trajectory_name(index))["n_frames"]
              for index in range(len(TAUS))}
    assert set(counts.values()) == {expected}, counts


@pytest.mark.slow
def test_the_state_trajectories_agree_frame_for_frame_on_time(completed):
    work, _, _ = completed
    times = {index: tuple(amber.read_frames(
        work / amber.state_trajectory_name(index))["times_ps"]) for index in range(len(TAUS))}
    assert len(set(times.values())) == 1, times


@pytest.mark.slow
def test_cpptraj_reads_the_state_trajectories(completed):
    """Amber-compatible means the installed cpptraj reads it, not that we like our own bytes."""
    import shutil
    cpptraj = shutil.which("cpptraj")
    if cpptraj is None:
        pytest.skip("cpptraj is not installed in this environment")
    work, n_atoms, _ = completed
    script = work / "read.in"
    script.write_text(f"parm {work / 'topology.pdb'}\n"
                      f"trajin {work / amber.state_trajectory_name(0)}\n"
                      f"trajout {work / 'roundtrip.nc'}\ngo\nquit\n", encoding="utf-8")
    result = subprocess.run([cpptraj, "-i", str(script)], capture_output=True, text=True,
                            cwd=str(work), timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert (work / "roundtrip.nc").is_file()


# --- section 8: the completion report ---------------------------------------------------------------

@pytest.mark.slow
def test_the_completion_report_is_persisted_and_printed(completed):
    """The other defect: the record builder called a name it had not imported, and no executed
    test reached it."""
    work, _, report = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    persisted = record["completion_report"]
    assert [p["state_pair"] for p in persisted["by_neighbouring_pair"]] == [[0, 1]]
    assert persisted["overall"]["proposed"] == persisted["by_neighbouring_pair"][0]["proposed"]
    assert "NEIGHBOURING-PAIR acceptance" in report
    assert "state 0 <-> state 1" in report


@pytest.mark.slow
def test_no_convergence_diagnostic_survives_into_the_record_or_the_terminal(completed):
    work, _, report = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    assert "round_trips" not in record
    assert "round trip" not in report.lower()


@pytest.mark.slow
def test_the_mapping_integrity_check_is_recorded(completed):
    work, _, _ = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    assert record["mapping_integrity"]["every_row_is_a_permutation"] is True
    assert record["mapping_integrity"]["offending_rows"] == []


# --- section 4: the resolved per-state records ------------------------------------------------------

@pytest.mark.slow
def test_the_resolved_record_names_every_state_trajectory(completed):
    work, _, _ = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    states = record["states"]
    assert [s["index"] for s in states] == list(range(len(TAUS)))
    assert [s["trajectory"] for s in states] == [amber.state_trajectory_name(i)
                                                 for i in range(len(TAUS))]
    assert [s["tau"] for s in states] == TAUS
    for state in states:
        assert state["effective_temperature_k"] == pytest.approx(
            300.0 / (1.0 - state["tau"]) ** 2)


@pytest.mark.slow
def test_every_named_state_trajectory_actually_exists(completed):
    """The record names the files; the files are there. A record that named a file it had not
    written would be the worst of both."""
    work, _, _ = completed
    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    for state in record["states"]:
        assert (work / state["trajectory"]).is_file()


# --- section 6: the REM log -------------------------------------------------------------------------

@pytest.mark.slow
def test_the_rem_log_is_written_with_a_row_per_state_per_exchange(completed):
    work, _, _ = completed
    text = (work / "rem.log").read_text(encoding="utf-8")
    assert text.startswith("#")
    blocks = text.count("# exchange ")
    assert blocks == EXCHANGES, f"{blocks} exchange blocks for {EXCHANGES} exchanges"


# --- continuation: the state trajectories are reopened, not restarted -------------------------------

@pytest.mark.slow
def test_a_resumed_run_continues_the_same_state_trajectories(prepared, tmp_path_factory):
    """`--resume` after an interruption must append to `remd*.nc`, not refuse or replace them.

    The set is reopened at the committed-frame marker, which is what makes the uncommitted rows
    left by a crash harmless: they are overwritten from the marker onward.
    """
    import shutil

    source, n_atoms, _ = prepared
    work = tmp_path_factory.mktemp("resumed")
    for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py", "ladder.group"):
        shutil.copy(source / name, work / name)

    first = _run(work)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    before = amber.read_frames(work / amber.state_trajectory_name(0))["n_frames"]

    # Extend the finished run: the same files carry on rather than a second set appearing.
    second = _run(work, "--extend", str(EXCHANGES))
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]

    after = {index: amber.read_frames(work / amber.state_trajectory_name(index))["n_frames"]
             for index in range(len(TAUS))}
    assert set(after.values()) == {before * 2}, (before, after)
    assert not (work / "remd0.nc.1").exists()

    record = json.loads((work / "restart.json").read_text(encoding="utf-8"))
    assert set(after.values()) == {record["whole_frames"]}


@pytest.mark.slow
def test_a_fresh_run_refuses_to_write_into_an_existing_set(prepared, tmp_path_factory):
    """Without --resume or --extend, existing state trajectories are somebody else's data."""
    import shutil

    source, _, _ = prepared
    work = tmp_path_factory.mktemp("collide")
    for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py", "ladder.group"):
        shutil.copy(source / name, work / name)
    assert _run(work).returncode == 0

    for name in ("exchange.nc", "restart.json", "checkpoint.nc", "exchange.runstate.json",
                 "exchange.solute.nc"):
        (work / name).unlink(missing_ok=True)
    before = (work / "whole_state0_prod1.nc").read_bytes()

    result = _run(work)
    assert result.returncode != 0
    assert (work / "whole_state0_prod1.nc").read_bytes() == before, "the refused run still wrote into the file"
