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
    # INTO `build/`, which is where the dataset layout keeps the System every run on it shares.
    # `build-md` validates the chain it generates against that System now, so a `built.xml` at
    # the root is a System nothing can find.
    (root / "build").mkdir(exist_ok=True)
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA_PDB), "-os", "build/built.xml",
               "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
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
        # A REAL minimisation, because the `min` stage below now actually runs. This count is the
        # min stage's; a production stage never consults it.
        "stages": {"minimization_iterations": 1000, "restrained_nvt_steps": CV_EVERY,
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

    # MINIMISE FIRST, and continue the window from that state (below, through `-c`).
    #
    # This file used to launch `umbrella.py` straight off `built.xml` with no `-c`, skipping the
    # generated chain entirely. That is not a shortcut, it is a different experiment: the
    # deposited geometry has
    # phi at -180 exactly -- an eclipsed dipeptide -- while the window is centred at -60 with a
    # stiff 800 kJ/mol/rad^2 bias. Production therefore opened 120 degrees from its own restraint,
    # on a structure nothing had relaxed, and the strain became kinetic energy: 5723 K by step 100,
    # then `Particle coordinate is NaN`.
    #
    # It was NOT a platform defect, though it looked exactly like one. The identical inputs were
    # measured dying on CPU and surviving on CUDA -- and, in an isolated probe of the same System,
    # dying on CUDA and surviving on CPU. Whichever platform happened to live was still integrating
    # at 4525 K and testing nothing, so a `gpu` marker would have buried the problem rather than
    # described it. Minimised and equilibrated first, the bias comes on from a relaxed 300 K
    # structure and the window is steady on both platforms.
    #
    # MINIMISATION ONLY, not the three equilibration stages, and the reason is the step counter.
    # It is ABSOLUTE across a chain, so equilibrating first would open production at step 10050
    # (50 + 5000 + 5000) and every assertion below that reads a committed step against
    # PRODUCTION_STEPS would be comparing an absolute number with a window length -- measured:
    # "crashed at 10100 steps, assert 10100 < 600", and the resumes were then refused outright
    # because a directory committed at 10100 no longer reads as 600 steps interrupted. `min`
    # performs no dynamics, contributes no steps, and leaves production starting at zero, which is
    # what these tests are written against. Relaxing the structure is the whole of what was needed.
    stage = subprocess.run(
        CLI + ["md-run", "-i", "input/min.in", "-p", "build/built.pdb", "-s", "build/built.xml",
               "-odir", "min", *_platform_flags()],
        cwd=root, capture_output=True, text=True, timeout=1800, env=_environment(root))
    assert stage.returncode == 0, f"min failed:\n{stage.stdout}{stage.stderr}"
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
    """`--cpu` unless a CUDA DEVICE is actually usable.

    WHICH PLATFORMS ARE REGISTERED IS A DIFFERENT QUESTION FROM WHETHER A DEVICE EXISTS, and this
    helper used to ask the first while promising the second. The CUDA platform stays registered
    when `CUDA_VISIBLE_DEVICES=""` -- there is simply nothing behind it -- so the old check found
    "CUDA" in the platform list, returned no flag, the runtime applied its CUDA default, and the
    stage died with `CUDA_ERROR_NO_DEVICE`. Every GPU-hidden run of this file went red while the
    file's own docstring promised "--cpu where there is no CUDA", and three sessions reported it
    as a mystery failure before it was diagnosed.

    So probe the device rather than the registration: build the smallest possible Context on CUDA
    and fall back to `--cpu` if that cannot be done, for any reason.
    """
    try:
        from openmm import Context, Platform, System, VerletIntegrator, unit

        system = System()
        system.addParticle(1.0 * unit.amu)
        context = Context(system, VerletIntegrator(0.001 * unit.picosecond),
                          Platform.getPlatformByName("CUDA"))
        del context
    except Exception:
        return ("--cpu",)
    return ()


def _run(project_root: Path, work: Path, *, extra_env=None, extra=()):
    return subprocess.run(
        [sys.executable, str(project_root / "project" / "umbrella.py"),
         "-p", str(project_root / "build" / "built.pdb"),
         "-s", str(project_root / "build" / "built.xml"),
         # CONTINUE FROM THE MINIMISED STATE. See the fixture for what running this stage off the
         # unrelaxed build instead used to do, and why the chain stops at `min`.
         "-c", str(project_root / "min" / "min.xml"),
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
