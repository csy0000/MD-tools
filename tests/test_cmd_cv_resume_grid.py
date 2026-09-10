"""After a resume, the absolute step grid must be exactly the grid an uninterrupted run writes.

WHAT WAS WRONG

    OpenMM's `loadCheckpoint` RESTORES the Context's step count: a checkpoint taken at step 25
    comes back with `currentStep == 25`. The reporters nevertheless added the already-completed
    count a second time -- `CVReporter(..., step_offset=done)` and the same pattern in
    `PhaseSpaceReporter` -- so the first observation after a resume was labelled roughly
    `2N + interval` instead of `N + interval`.

    Every step label, every `time_ps` derived from it, and every row's place on the grid were
    wrong for the whole continued run, in files that are otherwise perfectly well formed: right
    header, right column count, plausible monotonic numbers. Nothing downstream could notice.

    The tests here compare an interrupted-then-resumed run against an uninterrupted one and
    require the step grids to be equal. That is the only assertion that catches this, because
    every self-consistency check the damaged output could be given, it passes.

PLATFORM_POLICY_EXEMPTION: the stage runs under `--cpu` through the generated script. What is
under test is which integer labels a row, which is identical on every platform.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""

#: 40 production steps, CV every 5, checkpoint every 10, trajectory every 20.
PRODUCTION = 40
CV_EVERY = 5
CHECKPOINT_EVERY = 10
TRAJECTORY_EVERY = 20


def _project(root: Path, *, tau: float = 0.0, phase_space: int = 0) -> Path:
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    document = {
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": PRODUCTION},
        "reporting": {"crd_printout_solute": TRAJECTORY_EVERY, "info_printout": TRAJECTORY_EVERY,
                      "checkpoint_printout": CHECKPOINT_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        # A pinned seed, so an interrupted run and an uninterrupted one are the same trajectory
        # and the comparison is about LABELS rather than about dynamics.
        "dynamics": {"seed": 20260904, "tau": tau, "phase_space_printout": phase_space},
    }
    (root / "cMD.config").write_text(yaml.safe_dump(document), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./cMD", "--config", str(root / "cMD.config"),
               "--all-in-one"],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    return _project(tmp_path_factory.mktemp("cv-resume"))


@pytest.fixture(scope="module")
def phase_space_project(tmp_path_factory):
    """A fixed-tau project, which is the only shape that writes a phase-space stream."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    return _project(tmp_path_factory.mktemp("ps-resume"), tau=0.5, phase_space=CV_EVERY)


def _run(project: Path, destination: Path, *extra, environment=None, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base.update(environment or {})
    done = subprocess.run(
        [sys.executable, str(project / "cMD" / "md.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "cMD", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


def _cv_rows(destination: Path):
    found = sorted(destination.rglob("*.cv.csv"))
    assert len(found) == 1, f"expected one dynamics series, got {found}"
    lines = found[0].read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    return found[0], header, [dict(zip(header, line.split(","))) for line in lines[1:]]


EXPECTED_STEPS = list(range(0, PRODUCTION + 1, CV_EVERY))


def _commits_before_production(project: Path) -> int:
    """How many checkpoint commits the chain performs before production starts.

    EVERY stage commits a final generation, including the minimisation and the zero-length
    equilibration stages. `MD_TOOLS_CHECKPOINT_FAULT_AFTER` counts boundary crossings globally, so
    a fault meant for the production stage has to let those through first. Derived from the plan
    rather than written as a literal: a change to the stage chain would otherwise silently move
    the crash into a different stage, and the test would still "pass" while testing nothing.
    """
    from md_tools.build.md import resolve_md_config, stage_plan

    plan = stage_plan(resolve_md_config(project / "cMD.config"))
    before = [stage for stage in plan if int(stage.get("steps") or 0) == 0]
    assert len(before) == len(plan) - 1, (
        f"expected exactly one dynamics stage in this chain, got plan {[s['name'] for s in plan]}")
    return len(before)


def test_an_uninterrupted_run_writes_the_declared_grid(project, tmp_path):
    """The reference. Without it the resume comparison could pass with both sides wrong."""
    destination = tmp_path / "clean"
    _run(project, destination)
    _path, _header, rows = _cv_rows(destination)
    assert [int(r["step"]) for r in rows] == EXPECTED_STEPS


@pytest.mark.parametrize("committed_step", [10, 20, 30])
def test_a_resumed_run_writes_the_same_absolute_grid(project, tmp_path, committed_step):
    """THE regression. Interrupt at a NONZERO committed generation, resume, compare grids.

    Parametrised over WHERE the interruption lands, because the defect scaled with the completed
    step count: the further in the crash, the further out the labels. A crash at step 30 under
    the old arithmetic labelled the next observation 65, past the 40-step budget entirely.

    `after-pointer-replace` fires once the generation is committed, so the checkpoint that comes
    back is the one whose commit crashed -- the fault lets `allowed` boundaries through and
    raises on the next.
    """
    from md_tools.openmm.checkpoint import (FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT,
                                            read_committed)

    destination = tmp_path / f"resumed-{committed_step}"
    allowed = _commits_before_production(project) + committed_step // CHECKPOINT_EVERY - 1
    crashed = _run(project, destination, expect=1,
                   environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                FAULT_AFTER_ENVIRONMENT: str(allowed)})
    assert crashed.returncode != 0

    tree = sorted(destination.rglob("*.checkpoints"))
    assert tree, f"no checkpoint tree under {destination}"
    done = int(read_committed(tree[0])["state"]["steps_done"])
    assert done == committed_step, (
        f"the fixture aimed at a commit at step {committed_step} and landed at {done}; the test "
        f"would not be exercising the interruption it claims to")
    assert 0 < done < PRODUCTION, "the interruption must be at a nonzero, non-final step"

    _run(project, destination)

    _path, _header, rows = _cv_rows(destination)
    steps = [int(r["step"]) for r in rows]
    assert steps == EXPECTED_STEPS, (
        f"a resume from step {done} produced {steps}, not {EXPECTED_STEPS}. A label beyond "
        f"{PRODUCTION} means the completed steps were counted twice")
    assert len(steps) == len(set(steps)), "the resume duplicated an observation"
    assert max(steps) == PRODUCTION, "an observation was written past the requested final step"


def test_the_times_follow_the_absolute_steps_after_a_resume(project, tmp_path):
    """`time_ps` is derived from the step, so a doubled step silently doubles the time axis."""
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "times"
    _run(project, destination, expect=1,
         environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                      FAULT_AFTER_ENVIRONMENT: str(_commits_before_production(project) + 1)})
    _run(project, destination)

    _path, _header, rows = _cv_rows(destination)
    timestep_fs = 2.0
    for row in rows:
        expected = int(row["step"]) * timestep_fs / 1000.0
        assert abs(float(row["time_ps"]) - expected) < 1e-9, row


def test_trajectory_frame_indices_stay_global_across_a_resume(project, tmp_path):
    """A continued trajectory is one file, so its frame indices keep counting."""
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "frames"
    _run(project, destination, expect=1,
         environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                      FAULT_AFTER_ENVIRONMENT: str(_commits_before_production(project) + 1)})
    _run(project, destination)

    _path, _header, rows = _cv_rows(destination)
    named = {int(r["step"]): r["trajectory_frame_index"] for r in rows}
    # Frames are written every 20 steps; the first lands at step 20.
    assert named[5] == "" and named[15] == "", named
    assert named[20] == "0", named
    assert named[40] == "1", named

    import mdtraj

    trajectory = sorted(destination.rglob("*.dcd"))
    assert trajectory, "no trajectory was written"
    frames = mdtraj.load(str(trajectory[0]), top=str(project / "built.pdb"))
    assert frames.n_frames == 2, (
        f"a resumed run wrote {frames.n_frames} frames; a frame index in the CV series names a "
        f"frame that must exist in the continued trajectory")


def test_phase_space_steps_are_absolute_after_a_resume(phase_space_project, tmp_path):
    """The same defect, in the stream a reservoir is built from.

    A reservoir frame mislabelled by its own run's completed step count makes the reservoir's
    time axis disagree with the trajectory that produced it, which is how a window selection
    silently draws from the wrong part of the run.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT
    from md_tools.md.phase_space import PhaseSpaceReader

    destination = tmp_path / "ps"
    _run(phase_space_project, destination, expect=1,
         environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                      FAULT_AFTER_ENVIRONMENT:
                          str(_commits_before_production(phase_space_project) + 1)})
    _run(phase_space_project, destination)

    found = sorted(destination.rglob("*.phase_space.nc"))
    assert found, f"no phase-space stream under {destination}"
    with PhaseSpaceReader(found[0]) as reader:
        steps = [int(s) for s in reader.steps()]
    assert steps == steps_sorted(steps), "the phase-space steps are not increasing"
    assert max(steps) <= PRODUCTION, (
        f"a phase-space frame is labelled step {max(steps)}, past the requested "
        f"{PRODUCTION}: the completed steps were counted twice")
    assert len(steps) == len(set(steps)), "a phase-space step was written twice"


def steps_sorted(values):
    return sorted(values)
