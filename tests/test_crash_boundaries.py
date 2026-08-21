"""The five commit boundaries, each interrupted deliberately rather than by timing.

A time-based `kill -9` proves that *some* interruption is survivable. It cannot say which boundary
it hit, and the five boundaries fail differently: one leaves trajectory frames past the last commit,
one leaves streams that end cleanly with nothing saved for them, one leaves half a restart pair, one
leaves a complete restart pair that no commit points at, and one leaves a commit whose cache has not
caught up. A test that cannot name the boundary it exercised has not covered any of them.

`faults.crash_point` makes the boundary selectable from the environment and is inert otherwise. The
child is a real subprocess killed with `os._exit`, so no `finally` block, `atexit` handler or buffer
flush runs -- the same conditions a `kill -9` produces, and the conditions these guarantees must
hold under.

The existing timing-based kill test in `test_cmd_continuity_and_crash.py` is kept as additional
coverage. It exercises arrival at a boundary this file cannot name, which is worth having.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
INPUT_GEN = REPO_ROOT / "MD_input_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
               / "ace_ala_nme.pdb")

STREAMS = ("cMD_1_all_atoms.dcd", "cMD_1_selected_atoms.dcd", "cMD_1.log")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


def _tiny_config() -> dict:
    return {
        "profile": "implicit-md-peptide-v1",
        "protocol": {
            "integrator": {"kind": "langevin-middle", "timestep": "2 fs",
                           "temperature": "300 K", "friction": "1 /ps"},
            "equilibration": {"protocol": "simple", "minimize_max_iterations": 100,
                              "restrained": "1 ps"},
            "production": {"method": "md", "duration_per_segment": "2 ps"},
        },
        "randomness": {"master_seed": 20260821},
        "execution": {"platform": "CPU", "precision": "mixed",
                      "reporting": {"all_atom": "1 ps", "solute": "0.2 ps"}},
    }


@pytest.fixture(scope="module")
def committed_project(tmp_path_factory):
    """A project with exactly one committed cMD segment, ready to be interrupted."""
    bundle = tmp_path_factory.mktemp("cb_bundle") / "bundle"
    system_config = REPO_ROOT / "test" / "ala" / "cMD" / "implicit" / "system_config.json"
    prepared = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(bundle),
                    "--config", str(system_config))
    if prepared.returncode != 0:
        pytest.skip(f"implicit preparation unavailable: {prepared.stderr[-300:]}")

    root = tmp_path_factory.mktemp("cb_project")
    config = root / "md.json"
    config.write_text(json.dumps(_tiny_config()))
    project = root / "run"
    assert _run(INPUT_GEN, "--system", str(bundle / "system_manifest.json"),
                "-o", str(project), "--config", str(config)).returncode == 0

    for stage in ("min", "eq", "cMD_1"):
        result = subprocess.run([str(project / stage / f"{stage}.sh")], capture_output=True,
                                text=True, cwd=str(project / stage))
        assert result.returncode == 0, f"{stage}: {result.stderr[-500:]}"
    return project


def _fresh(committed_project, tmp_path) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(committed_project, copy)
    return copy


def _segment(project: Path, crash_at: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if crash_at:
        from md_templates.openmm.faults import FAULT_ENV

        env[FAULT_ENV] = crash_at
    else:
        from md_templates.openmm.faults import FAULT_ENV

        env.pop(FAULT_ENV, None)
    return subprocess.run([str(project / "cMD_1" / "cMD_1.sh")], capture_output=True, text=True,
                          cwd=str(project / "cMD_1"), env=env)


def _committed(project: Path) -> dict:
    matches = glob.glob(str(project / "cMD_1" / "run" / "**" / "committed.json"), recursive=True)
    return json.loads(Path(matches[0]).read_text()) if matches else {}


def _run_state(project: Path) -> dict:
    path = project / "cMD_1" / "run" / "run_state.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _streams(project: Path) -> dict:
    stage = project / "cMD_1"
    return {name: (stage / name).read_bytes() for name in STREAMS if (stage / name).is_file()}


def _log_rows(project: Path) -> list[list[float]]:
    text = (project / "cMD_1" / "cMD_1.log").read_text()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.lstrip("-")[:1].isdigit():
            continue
        rows.append([float(v) for v in line.split(",")])
    return rows


def _assert_history_is_sound(record: dict, expected_generations: int) -> None:
    """Monotonic, contiguous, one record per generation, starting at step 0."""
    history = record["invocation_history"]
    assert len(history) == expected_generations, history
    assert record["invocations_completed"] == expected_generations
    assert history[0]["started_absolute_step"] == 0
    for entry in history:
        assert entry["started_absolute_step"] < entry["ended_absolute_step"], entry
    for earlier, later in zip(history, history[1:]):
        assert earlier["ended_absolute_step"] == later["started_absolute_step"], (earlier, later)
    steps = [entry["ended_absolute_step"] for entry in history]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), steps
    assert record["absolute_step"] == steps[-1]


BOUNDARIES_BEFORE_COMMIT = [
    "before_reporters_close",
    "after_reporters_close",
    "after_checkpoint_member",
    "after_restart_members",
]


def test_the_hooks_are_inert_unless_armed():
    """The whole mechanism must cost nothing and do nothing in a normal run."""
    from md_templates.openmm.faults import CRASH_POINTS, FAULT_ENV, crash_point

    assert FAULT_ENV not in os.environ or os.environ[FAULT_ENV] == ""
    for point in CRASH_POINTS:
        crash_point(point)          # would kill this process if it were armed
    assert len(CRASH_POINTS) == 5


@pytest.mark.slow
@pytest.mark.parametrize("boundary", BOUNDARIES_BEFORE_COMMIT)
def test_a_crash_before_the_commit_keeps_the_last_committed_physics(
        committed_project, tmp_path, boundary):
    """Boundaries 1-4: nothing was committed, so generation 1 must remain authoritative."""
    project = _fresh(committed_project, tmp_path)
    before_record = _committed(project)
    before_streams = _streams(project)
    assert before_record["invocations_completed"] == 1

    crashed = _segment(project, crash_at=boundary)
    assert crashed.returncode != 0, "the injected crash did not stop the process"
    assert boundary in (crashed.stderr + crashed.stdout)

    after = _committed(project)
    assert after["invocations_completed"] == 1, (
        f"{boundary}: an uncommitted segment was counted as committed")
    assert after["absolute_step"] == before_record["absolute_step"]
    assert after["continuity_hash"] == before_record["continuity_hash"]
    _assert_history_is_sound(after, 1)


@pytest.mark.slow
@pytest.mark.parametrize("boundary", BOUNDARIES_BEFORE_COMMIT)
def test_retrying_after_a_pre_commit_crash_creates_exactly_one_generation(
        committed_project, tmp_path, boundary):
    """The retry must add one segment, not zero and not two, whatever the crash left behind."""
    project = _fresh(committed_project, tmp_path)
    first = _committed(project)

    assert _segment(project, crash_at=boundary).returncode != 0
    retry = _segment(project)
    assert retry.returncode == 0, retry.stdout[-800:] + retry.stderr[-800:]

    record = _committed(project)
    _assert_history_is_sound(record, 2)
    assert record["absolute_step"] == 2 * first["absolute_step"], (
        "the retried segment did not advance by exactly one segment length")

    rows = _log_rows(project)
    steps = [row[0] for row in rows]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), (
        f"{boundary}: the state log has a repeated or out-of-order row: {steps}")

    from md_templates.openmm import dcdtail

    for name, watermark_key in (("cMD_1_all_atoms.dcd", "all_atom"),
                                ("cMD_1_selected_atoms.dcd", "selected_atoms")):
        scan = dcdtail.frame_offsets(project / "cMD_1" / name)
        assert scan["trailing_bytes"] == 0, f"{boundary}: {name} kept an uncommitted tail"
        assert scan["complete_frames"] == record["watermarks"][watermark_key], (
            f"{boundary}: {name} holds {scan['complete_frames']} frames, watermark says "
            f"{record['watermarks'][watermark_key]}")


@pytest.mark.slow
def test_a_crash_before_reporters_close_leaves_a_tail_that_is_recovered(
        committed_project, tmp_path):
    """Boundary 1 specifically: this is the one that leaves frames past the last commit.

    Asserted separately from the shared retry test because if the tail were NOT produced, that test
    would still pass -- it would simply be recovering nothing.
    """
    from md_templates.openmm import dcdtail

    project = _fresh(committed_project, tmp_path)
    committed_frames = _committed(project)["watermarks"]["selected_atoms"]

    assert _segment(project, crash_at="before_reporters_close").returncode != 0
    scan = dcdtail.frame_offsets(project / "cMD_1" / "cMD_1_selected_atoms.dcd")
    assert scan["complete_frames"] > committed_frames, (
        "boundary 1 produced no uncommitted frames, so the recovery it exists to test is untested")

    assert _segment(project).returncode == 0
    record = _committed(project)
    _assert_history_is_sound(record, 2)
    after = dcdtail.frame_offsets(project / "cMD_1" / "cMD_1_selected_atoms.dcd")
    assert after["trailing_bytes"] == 0
    assert after["complete_frames"] == record["watermarks"]["selected_atoms"]


@pytest.mark.slow
def test_a_crash_between_the_restart_members_leaves_half_a_pair_and_still_recovers(
        committed_project, tmp_path):
    """Boundary 3: the checkpoint exists, the State does not, and no commit points at either."""
    project = _fresh(committed_project, tmp_path)
    assert _segment(project, crash_at="after_checkpoint_member").returncode != 0

    restart_root = project / "cMD_1" / "run" / "restart"
    orphan = sorted(p for p in restart_root.glob("gen_*") if p.is_dir())[-1]
    members = {p.name for p in orphan.iterdir()}
    assert any(m.endswith(".chk") for m in members), members
    assert not any(m.endswith(".xml") for m in members), (
        f"boundary 3 wrote the State as well, so half-a-pair was never produced: {members}")

    assert _committed(project)["invocations_completed"] == 1
    assert _segment(project).returncode == 0
    _assert_history_is_sound(_committed(project), 2)


@pytest.mark.slow
def test_a_crash_after_the_atomic_commit_keeps_the_generation_and_rebuilds_the_cache(
        committed_project, tmp_path):
    """Boundary 5: committed.json is the authority, and run_state.json is only a cache of it.

    The commit succeeded, so the physics must be kept. What is stale is the cache, and the next
    invocation must reconstruct it from the commit rather than treating the commit as suspect or
    re-running the segment that already landed.
    """
    project = _fresh(committed_project, tmp_path)
    assert _segment(project, crash_at="after_atomic_commit").returncode != 0

    record = _committed(project)
    assert record["invocations_completed"] == 2, (
        "the commit completed before the crash, so its generation must be retained")
    _assert_history_is_sound(record, 2)

    cached = len((_run_state(project).get("invocations") or []))
    assert cached < record["invocations_completed"], (
        "the cache was already up to date, so the staleness this test exists for did not occur")

    assert _segment(project).returncode == 0
    final = _committed(project)
    _assert_history_is_sound(final, 3)
    assert final["absolute_step"] == 3 * (record["absolute_step"] // 2)

    reconciled = _run_state(project).get("invocations") or []
    assert len(reconciled) >= 1
    assert all(entry.get("reconciled_from") == "committed.json" for entry in reconciled), (
        "the rebuilt cache does not record that it was reconstructed from the commit")


@pytest.mark.slow
@pytest.mark.parametrize("boundary", ["before_reporters_close", "after_restart_members",
                                      "after_atomic_commit"])
def test_committed_frames_are_never_rewritten_by_a_crash_and_retry(
        committed_project, tmp_path, boundary):
    """Whatever happens, the frames generation 1 committed must survive byte for byte."""
    project = _fresh(committed_project, tmp_path)
    before = _streams(project)
    prefixes = {name: len(data) for name, data in before.items()}

    assert _segment(project, crash_at=boundary).returncode != 0
    assert _segment(project).returncode == 0

    after = _streams(project)
    for name, original in before.items():
        # the log is rewritten atomically on recovery, so compare its committed rows, not bytes
        if name.endswith(".dcd"):
            assert after[name][92:prefixes[name]] == original[92:], (
                f"{boundary}: {name} altered frames that were already committed")
    rows = _log_rows(project)
    assert rows == sorted(rows, key=lambda r: r[0])
