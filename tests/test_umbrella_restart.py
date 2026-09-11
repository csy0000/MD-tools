"""An interrupted umbrella window must resume STILL BIASED.

The failure this guards against hides completely: a resumed window that came back unbiased writes
a collective-variable series that merely broadens, which is indistinguishable -- by eye or by any
downstream check -- from a window that was softer than intended. Nothing errors and the run
reports completion.

WHAT THIS FILE DOES AND DOES NOT PROVE, because the distinction was measured rather than assumed.

These tests pass with the resume path's `set_strength` call REMOVED. That is not a weakness in
them; it is a fact about OpenMM. `Context.setState` restores global parameters along with
positions and velocities, so a resumed window comes back biased through the checkpoint whether or
not the stage runner reapplies the restraint. The call is kept as defence against a future resume
path that rebuilds a Context without setState, and the code says so.

So what is pinned here is the PROPERTY -- a resumed umbrella window is still biased, and its
series is continuous across the boundary -- not any particular mechanism for it. A test asserting
the mechanism would have been a test of OpenMM, and it would have passed for a reason its name
did not describe.

PLATFORM_POLICY_EXEMPTION: implicit ALA, whatever platform is available, a few hundred steps.
What is under test is whether a parameter survives a resume, which is platform-independent; the
biasing force's CUDA behaviour is `test_torsion_restraint_cuda.py`.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT, read_committed

from .conftest import ALA_PDB, REPO_ROOT


CLI = [sys.executable, "-c",
       "import sys; from md_tools.cli.md_openmm import main; sys.argv[0]='md-openmm'; main()"]

PRODUCTION_STEPS = 600
CV_EVERY = 50
CHECKPOINT_EVERY = 100
CENTRE_DEG = -60.0
FORCE_CONSTANT = 800.0        # stiff, so an unbiased stretch is obvious rather than arguable


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A built implicit ALA system and a generated umbrella project, made once."""
    if not ALA_PDB.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("umbrella-restart")

    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA_PDB), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800, env=_environment())
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(yaml.safe_dump({
        "schema_version": 1,
        "collective_variables": [
            {"name": "phi_ALA", "type": "torsion", "atom_indices": [4, 6, 8, 14]}]}),
        encoding="utf-8")
    (root / "umbrella.yaml").write_text(yaml.safe_dump({
        "schema_version": 1,
        "restraints": [{"cv": "phi_ALA", "form": "harmonic",
                        "centre_deg": CENTRE_DEG, "force_constant": FORCE_CONSTANT}]}),
        encoding="utf-8")
    (root / "u.config").write_text(yaml.safe_dump({
        "protocol": "umbrella", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": CV_EVERY,
                   "production_steps": PRODUCTION_STEPS},
        "reporting": {"crd_printout_solute": CHECKPOINT_EVERY,
                      "info_printout": CHECKPOINT_EVERY,
                      "checkpoint_printout": CHECKPOINT_EVERY},
        "collective_variables": {"file": "cv.yaml", "interval_steps": CV_EVERY},
        "umbrella": {"file": "umbrella.yaml"}}), encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(root / "project"),
                                 "--config", str(root / "u.config")],
                          cwd=root, capture_output=True, text=True, timeout=600,
                          env=_environment())
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _environment(work: Path | None = None, extra=None) -> dict[str, str]:
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    if work is not None:
        path = work / "user.config"
        path.write_text(yaml.safe_dump(
            {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
        base["MD_TOOLS_CONFIG"] = str(path)
    base.update(extra or {})
    return base


def _platform_flags():
    """`--cpu` where there is no CUDA, which is what this file's exemption already claims.

    The docstring says "whatever platform is available", and the tests then asked for none -- so
    the runtime applied its default, CUDA, and refused on a machine without it: "CUDA is required
    and this OpenMM build does not provide it". That is the correct refusal; the test was simply
    not asking for what it said it wanted, and it turned CI's non-GPU lane red.
    """
    from openmm import Platform

    available = {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}
    return () if "CUDA" in available else ("--cpu",)


def _run(project_root: Path, work: Path, *, extra_env=None, extra=()):
    return subprocess.run(
        [sys.executable, str(project_root / "project" / "umbrella.py"),
         "-p", str(project_root / "built.pdb"), "-s", str(project_root / "built.xml"),
         "-odir", str(work), *_platform_flags(), *extra],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, extra_env))


def _phi_series(work: Path) -> list[tuple[int, float]]:
    path = work / "umbrella.cv.csv"
    assert path.is_file(), f"no collective-variable series at {path}"
    with path.open(newline="") as handle:
        return [(int(row["step"]), float(row["phi_ALA"])) for row in csv.DictReader(handle)]


def _offset(degrees: float) -> float:
    """How far from the restraint centre, on the circle."""
    return abs(((degrees - CENTRE_DEG + 180.0) % 360.0) - 180.0)


def test_an_interrupted_window_resumes_still_biased(project, tmp_path):
    """Crash mid-production, resume, and check the restraint is back ON.

    The assertion is on the collective variable AFTER the resume point, because that is the only
    place the failure shows: everything before it was written by the biased first process.
    """
    work = tmp_path / "interrupted"
    work.mkdir()

    crashed = _run(project, work, extra_env={FAULT_ENVIRONMENT: "after-checkpoint-sync",
                                             FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0, "the injected crash did not stop the stage"

    committed = read_committed(work / "umbrella.checkpoints")
    assert committed is not None, "nothing was committed before the crash"
    resume_step = int(committed["state"]["steps_done"])
    assert 0 < resume_step < PRODUCTION_STEPS, f"crashed at {resume_step} steps"

    resumed = _run(project, work)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr

    after = [(step, phi) for step, phi in _phi_series(work) if step > resume_step]
    assert after, f"the resumed stretch wrote no observations after step {resume_step}"

    worst = max(_offset(phi) for _, phi in after)
    assert worst < 40.0, (
        f"after resuming at step {resume_step}, phi reached {worst:.1f} deg from the restraint "
        f"centre ({CENTRE_DEG}); a k = {FORCE_CONSTANT} restraint does not permit that, so the "
        f"bias was not reapplied on the resume path")


def test_the_restraint_holds_across_the_whole_uninterrupted_window(project, tmp_path):
    """The control. Without it the test above could pass on a window that never moves anyway."""
    work = tmp_path / "uninterrupted"
    work.mkdir()
    done = _run(project, work)
    assert done.returncode == 0, done.stdout + done.stderr

    series = _phi_series(work)
    assert len(series) > 1
    worst = max(_offset(phi) for step, phi in series if step > 0)
    assert worst < 40.0, f"the restraint does not hold phi even uninterrupted: {worst:.1f} deg"


def test_the_resumed_window_reports_a_continuous_series(project, tmp_path):
    """No duplicated and no missing observation across the resume boundary.

    A series that repeats a step, or skips one, breaks every reweighting that assumes uniform
    spacing -- and none of them can detect it.
    """
    work = tmp_path / "continuity"
    work.mkdir()
    crashed = _run(project, work, extra_env={FAULT_ENVIRONMENT: "after-checkpoint-sync",
                                             FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0
    assert _run(project, work).returncode == 0

    steps = [step for step, _ in _phi_series(work)]
    assert steps == sorted(steps), "the series is not in step order"
    assert len(steps) == len(set(steps)), f"duplicated steps across the resume: {steps}"
    gaps = {b - a for a, b in zip(steps, steps[1:])}
    assert gaps <= {CV_EVERY}, f"irregular spacing across the resume boundary: {sorted(gaps)}"
