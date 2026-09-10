"""A REMD CV row's named frame holds exactly the configuration it was measured on, across resume.

TWO DEFECTS, BOTH INVISIBLE IN THE OUTPUT

    ALIGNMENT. CV is observed before the coincident state-trajectory frame is written -- that
    ordering is the pre-exchange convention and is deliberate -- but the row was given
    `state["frame_index"]`, which at that moment still holds the PREVIOUS frame's index. On the
    first coincident event that is `-1`; after that it points one frame behind. Every value was
    therefore attributed to a configuration it was not measured on, in a file whose columns,
    ranges and monotonicity are all perfectly ordinary.

    CONTINUATION. `_continue` never reopened the series, so a resumed ladder wrote no CV rows at
    all after the interruption. The run completed, every other stream continued correctly, and
    the CV files simply stopped at the crash.

The only assertion that catches the first is to load each named frame and recompute the torsion
independently. The only assertion that catches the second is to compare a resumed run's series
against an uninterrupted one.

PLATFORM_POLICY_EXEMPTION: the ladder runs under `--cpu`. What is under test is which frame a row
names and which rows exist, which is bookkeeping and identical on every platform.
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

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

QUARTET = [4, 6, 8, 14]
CV_YAML = f"""\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: {QUARTET}
"""

STATES = 3
EXCHANGE_EVERY = 10
EXCHANGES = 4
CV_EVERY = 5
TOTAL = EXCHANGE_EVERY * EXCHANGES


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("remd-align")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        # The state trajectory is written every exchange interval, so CV (every 5) and frames
        # (every 10) coincide at 10, 20, 30, 40 and not at 5, 15, 25, 35.
        #
        # `crd_printout_whole` is the one that matters here and it must be SET: a CV row names a
        # frame of the WHOLE state trajectory, and the key defaults to 0. Written as
        # `info_printout` -- the state table -- the ladder wrote one whole frame in the entire
        # run and every CV row named nothing, which reads as an alignment failure rather than as
        # a stream that was never asked for.
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY,
                      "crd_printout_whole": EXCHANGE_EVERY,
                      "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./REST2", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "built.pdb"))
    system = XmlSerializer.deserialize((root / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return root


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
        [sys.executable, str(project / "REST2" / "REST2.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "REST2", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("remd-align-run") / "run"
    _run(project, destination)
    return destination


def test_every_named_frame_holds_the_configuration_the_row_was_measured_on(completed, project):
    """THE alignment test: load `remdN.nc[k]` and recompute the torsion independently.

    A row naming a frame one behind still produces a plausible in-range number, so nothing short
    of recomputing from the named frame can distinguish a correct file from a misaligned one.
    """
    mdtraj = pytest.importorskip("mdtraj")

    checked = 0
    for index in range(STATES):
        rows = _rows(completed / f"remd{index}.cv.csv")
        frames = mdtraj.load(str(completed / f"whole_state{index}_prod1.nc"),
                             top=str(project / "built.pdb"))
        for row in rows:
            named = row["trajectory_frame_index"]
            if named == "":
                continue
            assert named != "-1", "-1 is not a frame index and must never be written"
            frame = int(named)
            assert 0 <= frame < frames.n_frames, (
                f"state {index} step {row['step']} names frame {frame}, and the trajectory holds "
                f"{frames.n_frames}")
            expected = math.degrees(float(
                mdtraj.compute_dihedrals(frames[frame], [QUARTET])[0][0]))
            reported = float(row["phi"])
            difference = abs((reported - expected + 180.0) % 360.0 - 180.0)
            assert difference < 1e-2, (
                f"state {index} step {row['step']} reports {reported} and frame {frame} holds "
                f"{expected}: the row names a frame it was not measured on")
            checked += 1
    assert checked >= STATES * 2, f"only {checked} aligned row(s) were available to check"


def test_rows_off_the_frame_cadence_carry_no_index(completed):
    """Frames every 10, CVs every 5: the odd multiples can never name one."""
    rows = _rows(completed / "remd0.cv.csv")
    by_step = {int(r["step"]): r["trajectory_frame_index"] for r in rows}
    for step in (0, 5, 15, 25, 35):
        assert by_step[step] == "", (
            f"no state trajectory frame is written at step {step}, so the field must be empty; "
            f"got {by_step[step]!r}")


def test_a_state_the_exchange_moved_names_no_frame_at_that_step(completed):
    """THE case that makes this a per-state decision rather than a per-step one.

    The state trajectory frame is written from the POST-exchange occupant; the CV row describes
    the PRE-exchange one. When a swap is accepted, those are different configurations, and the
    only honest answer is an empty field -- "no frame in this file holds what this row measured".

    So at a coincident step, a state names the frame if and only if its walker did not change.
    Proved from the files themselves: the walker occupying a state at the NEXT observation is the
    post-exchange occupant, so a row whose walker differs from the following row's is a row whose
    state was moved by the exchange at that step.
    """
    frame_steps = {10, 20, 30, 40}
    moved_and_empty = coincident = 0
    for index in range(STATES):
        rows = _rows(completed / f"remd{index}.cv.csv")
        for row, following in zip(rows, rows[1:]):
            step = int(row["step"])
            if step not in frame_steps:
                continue
            coincident += 1
            swapped = int(row["walker_index"]) != int(following["walker_index"])
            named = row["trajectory_frame_index"]
            if swapped:
                assert named == "", (
                    f"state {index} step {step}: the exchange moved this state's walker "
                    f"({row['walker_index']} -> {following['walker_index']}), so the frame holds "
                    f"a different configuration, but the row names {named!r}")
                moved_and_empty += 1
            else:
                assert named != "" and named != "-1", (
                    f"state {index} step {step}: the walker did not change, so the frame holds "
                    f"exactly this configuration and must be named; got {named!r}")
    assert coincident, "no coincident CV/frame step was exercised"
    assert moved_and_empty, (
        "no accepted exchange landed on a frame step in this run, so the per-state rule was "
        "never exercised; the test would pass on a build that names the frame unconditionally")


def test_a_named_frame_index_is_never_negative(completed):
    """-1 is not a frame index. Writing it invites a reader to index from the end of the file."""
    for index in range(STATES):
        for row in _rows(completed / f"remd{index}.cv.csv"):
            named = row["trajectory_frame_index"]
            if named != "":
                assert int(named) >= 0, f"state {index}: {named}"


@pytest.mark.parametrize("boundary", ["after-cv-row", "before-checkpoint", "after-checkpoint"])
def test_a_resumed_ladder_writes_the_same_cv_series_as_an_uninterrupted_one(
        project, tmp_path, boundary):
    """THE continuation test. `_continue` used to leave the series closed and write nothing.

    Interrupted on both sides of a CV row and on both sides of a checkpoint commit, because those
    are the windows the committed count exists to close: a crash after a row reaches the disk but
    before the checkpoint vouching for it leaves a file LONGER than the bookkeeping, and a resume
    that appended blindly would duplicate the overlap.
    """
    reference = tmp_path / f"reference-{boundary}"
    _run(project, reference)
    expected = {index: _rows(reference / f"remd{index}.cv.csv") for index in range(STATES)}

    destination = tmp_path / f"resumed-{boundary}"
    crashed = _run(project, destination, expect=1,
                   environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                                "MD_TOOLS_FAIL_LADDER_AT": boundary,
                                "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    assert crashed.returncode != 0, crashed.stdout[-2000:] + crashed.stderr[-2000:]
    partial = _rows(destination / "remd0.cv.csv")
    assert partial, "the interruption left no CV rows, so the resume has nothing to continue"
    assert len(partial) < len(expected[0]), "the interruption did not stop the run early"

    _run(project, destination, "--resume")

    for index in range(STATES):
        rows = _rows(destination / f"remd{index}.cv.csv")
        steps = [int(r["step"]) for r in rows]
        assert steps == [int(r["step"]) for r in expected[index]], (
            f"state {index} at {boundary}: a resumed ladder produced {steps}")
        assert len(steps) == len(set(steps)), f"state {index}: a step was written twice"
        assert steps[0] == 0 and steps[-1] == TOTAL


def test_a_checkpoint_without_a_committed_cv_count_refuses_continuation(project, tmp_path):
    """A legacy checkpoint cannot say which rows are durable, so it must refuse, not guess.

    Guessing from the file length is exactly the guess the count exists to avoid: rows written
    after the last commit are the ones a resume must discard.
    """
    destination = tmp_path / "legacy"
    _run(project, destination, expect=1,
         environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                      "MD_TOOLS_FAIL_LADDER_AT": "after-checkpoint",
                      "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})

    # Strip the committed count, as a build that reported CVs without binding them would leave it.
    import netCDF4

    checkpoint = sorted(destination.glob("*.chk.nc")) or sorted(destination.rglob("*checkpoint*"))
    assert checkpoint, f"no checkpoint under {destination}"
    dataset = netCDF4.Dataset(str(checkpoint[0]), "a")
    try:
        import json

        extra = json.loads(getattr(dataset, "extra_json", "{}"))
        extra.pop("cv_rows", None)
        dataset.extra_json = json.dumps(extra, sort_keys=True)
    finally:
        dataset.close()

    refused = _run(project, destination, "--resume", expect=1)
    # The refusal is written into the rank's own report, which `executor.main` captures as the
    # run's record; only a pointer to it reaches the launcher's stdout.
    reports = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                        for path in sorted(destination.glob("REST2.out*")))
    message = (refused.stdout + refused.stderr + reports).lower()
    assert "collective-variable" in message, message[-3000:]
    assert "committed" in message, message[-3000:]
