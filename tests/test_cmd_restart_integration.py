"""A cMD stage survives a crash at every boundary of its checkpoint transaction.

THE DEFECT THIS EXISTS FOR

The stage used to checkpoint like this:

    simulation.saveCheckpoint(path)
    Path(path + ".json").write_text(json.dumps({"steps_done": ...}))

Two writes, in order, to two names that mean nothing apart. A crash between them -- or a
filesystem that reorders them, which is the ordinary case without an fsync -- leaves a NEW Context
checkpoint beside an OLD `steps_done`. The resume then continues from step 3000 believing it is at
2000: it re-emits frames that already exist, and reports a stage length that was never run.
Nothing raises. The trajectory is simply wrong, and it looks complete.

It is the same defect the AIS path checkpoint had, and it now uses the same transaction, so these
tests are the cMD half of what `test_ais_recovery_integration.py` proves for a switching path.

WHAT IS ASSERTED

For a crash at each of the five commit boundaries and at the trajectory-frame boundaries:

  the committed pointer still names a COMPLETE generation, and only that one is followed;
  the trajectory is cut back to the count the checkpoint vouches for, never left long;
  the resumed stage produces the same final state as an uninterrupted reference, byte for byte;
  no step appears twice and none is missing.

The last is the one that matters. Every other property can hold while the numbers are wrong.

These run the GENERATED stage script as a subprocess, because that is what a person runs, and
because a crash has to be a real process exit rather than an exception caught in-process.

PLATFORM_POLICY_EXEMPTION: these run on whatever `machine.openmm.platform` resolves to, which on
this machine is CUDA. Nothing here asserts a platform: the object under test is the ordering of
writes to disk, which is identical on every platform. The CUDA lane proves CUDA.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

from md_tools.openmm.checkpoint import (BOUNDARIES, FAULT_AFTER_ENVIRONMENT,  # noqa: E402
                                        FAULT_ENVIRONMENT, POINTER_NAME, read_committed)

#: ONE worker for this module. Its module-scoped fixture builds a system and runs
#: dynamics; scattered across workers it is built once per worker that draws a test.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("cmd-restart")]

#: Short enough to run many times, long enough that several checkpoints are committed before the
#: crash: 40 steps at a checkpoint every 10 is four commits.
PRODUCTION_STEPS = 40
CHECKPOINT_EVERY = 10
FRAME_EVERY = 5


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A built implicit ALA system and a generated cMD project, made once."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cmd-restart")

    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 10,
                   "production_steps": PRODUCTION_STEPS},
        "reporting": {"crd_printout_solute": FRAME_EVERY, "info_printout": FRAME_EVERY,
                      "checkpoint_printout": CHECKPOINT_EVERY}}), encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(root / "project"),
                                 "--config", str(root / "cMD.config")],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _user_config(directory: Path) -> dict[str, str]:
    path = directory / "user.config"
    path.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    return {"MD_TOOLS_CONFIG": str(path)}


def _run_stage(project_root: Path, work: Path, *, environment=None, timeout=900, extra=()):
    """The production stage, in its own directory, as a subprocess."""
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(_user_config(work))
    base.update(environment or {})
    return subprocess.run(
        [sys.executable, str(project_root / "project" / "cMD.py"),
         "-p", str(project_root / "built.pdb"), "-s", str(project_root / "built.xml"),
         "-odir", str(work), *extra],
        cwd=work, capture_output=True, text=True, timeout=timeout, env=base)


def _checkpoints(work: Path) -> Path:
    return work / "cMD.checkpoints"


def _frames(path: Path) -> int:
    from md_tools.openmm.trajectory import count_frames

    return count_frames(path) if path.is_file() else 0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def reference(project, tmp_path_factory):
    """One uninterrupted run of the stage, to compare every recovered run against."""
    work = tmp_path_factory.mktemp("reference")
    done = _run_stage(project, work)
    assert done.returncode == 0, done.stdout + done.stderr
    return work, _digest(work / "cMD.xml"), _frames(work / "solute_prod1.nc")


# --- the transaction ------------------------------------------------------------------------------

def test_the_uninterrupted_stage_commits_through_the_pointer(project, reference):
    """The baseline, and a guard on every test below: there IS a committed generation to find."""
    work, _restart, frames = reference
    assert (_checkpoints(work) / POINTER_NAME).is_file(), "no committed pointer was written"
    committed = read_committed(_checkpoints(work))
    assert committed is not None
    assert committed["state"]["steps_done"] == PRODUCTION_STEPS
    assert committed["state"]["stage"] == "cMD"
    assert committed["state"]["streams"]["trajectory"] == frames
    assert frames > 1, "a one-frame trajectory cannot demonstrate a truncation"


def test_the_committed_state_binds_the_identity_a_resume_has_to_agree_about(project, reference):
    """Not only the fingerprint: the stage, the seed, the timestep, the platform and the counts."""
    work, _restart, _frames = reference
    state = read_committed(_checkpoints(work))["state"]
    for field in ("fingerprint", "steps_done", "stage", "seed", "timestep_fs", "ensemble",
                  "platform", "streams"):
        assert field in state, f"{field} is not bound into the committed checkpoint"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_crash_at_a_commit_boundary_resumes_to_the_same_result(boundary, project, reference,
                                                                 tmp_path):
    """Crash at each of the five, resume, and land on the uninterrupted final state exactly.

    `FAULT_AFTER=1` so a complete generation always precedes the crash: the interesting case is
    resuming from a good generation while a newer, half-written one lies beside it. A crash at the
    very first commit -- nothing committed, the stage restarts from the beginning -- is covered by
    `test_checkpoint_transaction.py`.
    """
    reference_work, reference_restart, reference_frames = reference
    work = tmp_path / f"crash-{boundary}"
    work.mkdir()

    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: boundary,
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0, f"{boundary}: the injected crash did not stop the stage"

    committed = read_committed(_checkpoints(work))
    assert committed is not None, f"{boundary}: no generation was committed before the crash"
    vouched = int(committed["state"]["steps_done"])
    assert 0 < vouched < PRODUCTION_STEPS, f"{boundary}: crashed at {vouched} steps"

    resumed = _run_stage(project, work)
    assert resumed.returncode == 0, f"{boundary}:\n{resumed.stdout}{resumed.stderr}"

    assert _digest(work / "cMD.xml") == reference_restart, (
        f"{boundary}: the recovered stage produced a different final state from an "
        f"uninterrupted run of the same stage with the same seed")
    assert _frames(work / "solute_prod1.nc") == reference_frames, (
        f"{boundary}: {_frames(work / 'solute_prod1.nc')} frames against the reference's "
        f"{reference_frames}")


def test_a_trajectory_longer_than_its_checkpoint_is_cut_back_not_kept(project, reference,
                                                                      tmp_path):
    """The sharp case, constructed directly.

    Frames are flushed as they are written, so after a crash the trajectory is routinely LONGER
    than the checkpoint that describes it. Keeping the extra frames would put the coordinates
    permanently ahead of the step count, and every frame after that point would be attributed to
    the wrong step -- silently, in a file that opens and reads perfectly.
    """
    reference_work, reference_restart, reference_frames = reference
    work = tmp_path / "long-trajectory"
    work.mkdir()

    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "after-checkpoint-write",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0

    committed = read_committed(_checkpoints(work))
    vouched = int(committed["state"]["streams"]["trajectory"])
    on_disk = _frames(work / "solute_prod1.nc")
    assert on_disk >= vouched, "frames went backwards before recovery"

    resumed = _run_stage(project, work)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _frames(work / "solute_prod1.nc") == reference_frames
    assert _digest(work / "cMD.xml") == reference_restart


def test_recovery_never_reads_the_newest_generation_on_disk(project, reference, tmp_path):
    """An uncommitted newer generation is debris from a crash, and must be ignored.

    "The newest file wins" is the single most natural way to write this and the single most wrong:
    the newest file is exactly what a crash leaves behind.
    """
    reference_work, reference_restart, reference_frames = reference
    work = tmp_path / "uncommitted-newer"
    work.mkdir()
    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "before-pointer-replace",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0

    directory = _checkpoints(work)
    committed = read_committed(directory)
    # The generations live under `checkpoints/`; the pointer is the only file at the top.
    generations = sorted(int(p.stem.split("_")[1])
                         for p in (directory / "checkpoints").glob("generation_*.chk"))
    assert max(generations) > committed["generation"], (
        "this test needs an uncommitted newer generation on disk to be ignoring one")

    resumed = _run_stage(project, work)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _digest(work / "cMD.xml") == reference_restart


def test_a_checkpoint_from_a_different_configuration_is_refused(project, reference, tmp_path):
    """The fingerprint is checked before anything is loaded: right numbers, wrong simulation."""
    work = tmp_path / "foreign"
    work.mkdir()
    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0

    sidecar = Path(read_committed(_checkpoints(work))["sidecar"])
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["state"]["fingerprint"] = "a different configuration entirely"
    sidecar.write_text(json.dumps(document), encoding="utf-8")

    refused = _run_stage(project, work)
    assert refused.returncode != 0
    assert "fingerprint mismatch" in (refused.stdout + refused.stderr), refused.stderr


def test_a_completed_log_whose_outputs_are_gone_is_not_treated_as_done(project, reference,
                                                                       tmp_path):
    """"status: completed" is a field in a file, and a field in a file is not a finished run.

    The stage used to return 0 on the strength of that field alone, BEFORE the preflight ran. A
    log left behind by a scratch filesystem that was cleaned, or a partial copy, then ended the
    command successfully -- and the next stage continued from a restart that does not exist.
    """
    work = tmp_path / "hollow"
    work.mkdir()
    done = _run_stage(project, work)
    assert done.returncode == 0, done.stdout + done.stderr

    (work / "cMD.xml").unlink()
    again = _run_stage(project, work)
    assert again.returncode != 0, again.stdout + again.stderr
    message = again.stdout + again.stderr
    assert "cMD.xml" in message and "missing" in message, message


def test_a_completed_log_from_another_configuration_is_not_treated_as_done(project, tmp_path):
    """The same field, in a log that describes a run set up differently.

    Done by changing the CONFIGURATION rather than by editing the log: what has to be caught is a
    `resolved.config` that was edited after a run finished, and a log left in place describing
    the run that is no longer the one this script would produce. Editing the log instead would
    test the same comparison from the wrong side.
    """
    work = tmp_path / "foreign-log"
    work.mkdir()
    done = _run_stage(project, work)
    assert done.returncode == 0, done.stdout + done.stderr
    from md_tools.build.record import read_record

    assert read_record(work / "cMD.log")["status"] == "completed"

    # A copy of the project whose resolved settings differ in something that is NOT extendable.
    altered = tmp_path / "altered-project"
    shutil.copytree(project / "project", altered)
    config = altered / "resolved.config"
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["dynamics"]["temperature_K"] = float(document["dynamics"]["temperature_K"]) + 10.0
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(_user_config(work))
    again = subprocess.run(
        [sys.executable, str(altered / "cMD.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-odir", str(work)],
        cwd=work, capture_output=True, text=True, timeout=900, env=base)
    assert again.returncode != 0, again.stdout + again.stderr
    assert "different configuration" in (again.stdout + again.stderr), again.stderr


# --- the resume contract, and what --overwrite means ---------------------------------------------

def test_an_interrupted_stage_resumes_without_being_asked_to(project, reference, tmp_path):
    """THE contract: interrupted resumes automatically, `--resume` is not required.

    The alternative -- requiring `--resume` -- means an interrupted stage that is simply re-run
    silently starts over and discards committed work, which is the failure the checkpoint exists
    to prevent. `--resume` is accepted so the four protocols take the same flags, and it changes
    nothing here.
    """
    reference_work, reference_restart, reference_frames = reference
    work = tmp_path / "auto-resume"
    work.mkdir()
    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0
    committed = read_committed(_checkpoints(work))
    assert 0 < int(committed["state"]["steps_done"]) < PRODUCTION_STEPS

    # No `--resume` anywhere.
    resumed = _run_stage(project, work)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Resume" in (work / "cMD.log").read_text(encoding="utf-8")
    assert _digest(work / "cMD.xml") == reference_restart


def test_resume_is_refused_by_name_for_cmd(project, reference, tmp_path):
    """`--resume` is not a cMD flag; it is refused by name rather than accepted and ignored.

    An interrupted stage continues from its committed checkpoint automatically -- that is the
    whole cMD contract -- so a flag that pretends to control it, and does nothing, is worse than
    one refused: it lets a caller believe they asked for something. Re-running the SAME command
    WITHOUT the flag is how a crashed run continues.
    """
    reference_work, reference_restart, _frames = reference
    work = tmp_path / "explicit-resume"
    work.mkdir()
    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0

    refused = _run_stage(project, work, extra=["--resume"])
    assert refused.returncode != 0, refused.stdout + refused.stderr
    message = refused.stdout + refused.stderr
    assert "--resume is not a cMD flag" in message, message[-1500:]

    resumed = _run_stage(project, work)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _digest(work / "cMD.xml") == reference_restart


def test_overwrite_starts_clean_and_does_not_load_an_old_checkpoint(project, reference, tmp_path):
    """`--overwrite` that resumed from the generations it was asked to replace is not overwrite.

    It produced a run that was half the old one -- and reported success, because every count and
    every hash was internally consistent with a run nobody asked for.
    """
    reference_work, reference_restart, reference_frames = reference
    work = tmp_path / "overwrite-clean"
    work.mkdir()
    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0
    assert read_committed(_checkpoints(work)) is not None

    fresh = _run_stage(project, work, extra=["--overwrite"])
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    log = (work / "cMD.log").read_text(encoding="utf-8")
    assert "Resume" not in log, "--overwrite continued from the checkpoint it was replacing"
    # A clean run of the same stage lands where the uninterrupted reference does.
    assert _digest(work / "cMD.xml") == reference_restart
    assert _frames(work / "solute_prod1.nc") == reference_frames


def test_the_checkpoint_commits_all_three_appendable_streams(project, reference):
    """DCD, state CSV and phase-space NetCDF, not just the trajectory.

    Committing one meant a resume truncated that one and appended to the other two, leaving three
    streams describing three different instants in files that read perfectly.
    """
    work, _restart, _frames = reference
    streams = read_committed(_checkpoints(work))["state"]["streams"]
    assert "trajectory" in streams, streams
    assert "state_csv" in streams, (
        f"the state CSV is not committed; a resume cannot truncate it: {streams}")
    # The phase-space stream is only present when the stage was configured to write one.
    if (work / "cMD.phase_space.nc").is_file():
        assert "phase_space" in streams, streams


def test_a_resume_truncates_the_state_csv_as_well_as_the_trajectory(project, reference, tmp_path):
    """The sharp case for the CSV: it is flushed per row and outlives the checkpoint."""
    reference_work, reference_restart, reference_frames = reference
    work = tmp_path / "csv-truncation"
    work.mkdir()
    crashed = _run_stage(project, work, environment={FAULT_ENVIRONMENT: "after-checkpoint-write",
                                                     FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0

    committed = read_committed(_checkpoints(work))["state"]["streams"]

    # `mdout.csv`, NOT `cMD.csv`. The production state table was renamed to match what Amber
    # calls it, and this test kept the old name -- so `csv_path.is_file()` was false every time
    # and the guard below turned into an unconditional skip. The test did not become flaky; it
    # stopped running, and reported that as a property of the stage ("this stage wrote no state
    # CSV") rather than as a stale filename here.
    #
    # So the premise is ASSERTED now. If the production stage ever stops committing a state CSV,
    # that is a defect in the thing under test and this must fail, not opt out.
    csv_path = work / "mdout.csv"
    assert csv_path.is_file(), (
        f"the production stage wrote no state table; {sorted(p.name for p in work.iterdir())}")
    assert "state_csv" in committed, (
        f"the state CSV is not a committed stream, so a resume cannot truncate it: {committed}")

    with csv_path.open(encoding="utf-8") as handle:
        before = max(sum(1 for _ in handle) - 1, 0)
    assert before >= committed["state_csv"], "the CSV went backwards before recovery"

    resumed = _run_stage(project, work)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    with (reference_work / "mdout.csv").open(encoding="utf-8") as handle:
        expected = max(sum(1 for _ in handle) - 1, 0)
    with csv_path.open(encoding="utf-8") as handle:
        after = max(sum(1 for _ in handle) - 1, 0)
    assert after == expected, (
        f"the recovered state CSV holds {after} rows against the uninterrupted run's {expected}")


def test_resume_cannot_be_used_to_bypass_the_collision_check(project, reference, tmp_path):
    """`--resume` cannot excuse existing outputs, because it is refused before that check runs.

    Only a valid committed checkpoint short of the step count excuses them, and that is a fact
    about the directory rather than a claim on the command line. `--resume` is refused BY NAME
    for cMD before the collision check is even reached, which forecloses it as a bypass a
    different way than the flag being accepted and simply not matching would.
    """
    work = tmp_path / "no-checkpoint"
    work.mkdir()
    done = _run_stage(project, work)
    assert done.returncode == 0, done.stdout + done.stderr

    # Outputs present, log gone, checkpoints gone: not completed, not interrupted.
    import shutil

    (work / "cMD.log").unlink()
    shutil.rmtree(_checkpoints(work))

    refused = _run_stage(project, work, extra=["--resume"])
    assert refused.returncode != 0, refused.stdout + refused.stderr
    message = refused.stdout + refused.stderr
    assert "--resume is not a cMD flag" in message, message[-1500:]

    # The collision check itself, reached without the flag in the way.
    refused_again = _run_stage(project, work)
    assert refused_again.returncode != 0, refused_again.stdout + refused_again.stderr
    message = refused_again.stdout + refused_again.stderr
    assert "already exist" in message and "--overwrite" in message, message[-1500:]
