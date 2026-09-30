"""A CV series is aligned to its own stage, in a run where the stage does not start at step 0.

WHY THIS FILE EXISTS ALONGSIDE test_cmd_cv_output.py

    That file already recomputes every reported torsion from the frame it claims, with MDTraj.
    It passes, and it cannot fail: its fixture sets every equilibration length to 0 and runs the
    production script on its own, so `simulation.currentStep` begins at 0 and the absolute step
    equals the stage-relative one. Every alignment it asserts is an identity in that fixture.

    A real run is the whole chain. `min`, `eq_1`, `eq_2`, `eq_3` and the production stage each
    continue from the previous stage's restart, which carries the Context's step count, so every
    stage after the first begins at a non-zero absolute step. That is the configuration in which
    an offset can be wrong, and nothing exercised it.

WHAT IS ASSERTED, AND WHY IT IS CONVENTION-INDEPENDENT

    The docs settle the step convention ("Steps are absolute", docs/basics/collective-variables.md)
    but these assertions do not depend on that choice. Whichever convention holds, one series
    cannot use two of them:

      the spacing between consecutive observations is the cadence, everywhere. `cv/schedule.py`
      refuses an interval that does not divide the stage length precisely so that no gap of a
      different size exists, "because every downstream time-series analysis assumes uniform
      spacing and none of them can detect the violation". A first gap of a different width is
      that violation, arriving by another route;

      the series spans exactly the stage's step count, so the final row is the step the stage's
      result is quoted at rather than some later number;

      a `trajectory_frame_index` names a frame that EXISTS in the file it indexes, and holds the
      configuration the row measured. An index past the end is unindexable; an index inside the
      file but off by a fixed amount is worse, because it reads cleanly and reports another
      configuration's torsion under this row's step.

PLATFORM_POLICY_EXEMPTION: the stages run under `--cpu` through the generated scripts, exactly as
test_cmd_cv_output.py does. What is under test is the CSV's step axis and frame alignment, which
is platform-independent; this is not offered as CUDA evidence for anything.
"""
from __future__ import annotations

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

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
  - name: psi
    type: torsion
    atom_indices: [6, 8, 14, 16]
"""

#: The cadence, and the stage lengths it divides. Small enough to run on a CPU in seconds, and
#: chosen so that the production stage is longer than the trajectory cadence -- otherwise no row
#: would carry a frame index and the alignment check would have nothing to check.
INTERVAL = 5
EQUILIBRATION_STEPS = 10
PRODUCTION_STEPS = 40
TRAJECTORY_INTERVAL = 20

#: Each stage's own length, by the key its series is filed under. `min` writes no series: a
#: minimiser's iterations are not dynamics.
STAGE_STEPS = {
    "eq_1": EQUILIBRATION_STEPS,
    "eq_2": EQUILIBRATION_STEPS,
    "eq_3": EQUILIBRATION_STEPS,
    "cMD": PRODUCTION_STEPS,
}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """The whole generated chain, run in order, exactly as `run.sh` runs it."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cv-stage-alignment")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        # Every equilibration length is NON-ZERO. That is the entire point of this fixture: it is
        # what makes each later stage begin at an absolute step that is not 0.
        "stages": {"minimization_iterations": 5,
                   "restrained_nvt_steps": EQUILIBRATION_STEPS,
                   "restrained_npt_steps": EQUILIBRATION_STEPS,
                   "unrestrained_npt_steps": EQUILIBRATION_STEPS,
                   "production_steps": PRODUCTION_STEPS},
        "reporting": {"crd_printout_solute": TRAJECTORY_INTERVAL,
                      "info_printout": TRAJECTORY_INTERVAL,
                      "checkpoint_printout": TRAJECTORY_INTERVAL},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": INTERVAL},
    }), encoding="utf-8")
    generated = subprocess.run(
        CLI + ["build-md", "-odir", "./cMD-run1", "--config", str(root / "cMD.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    run = root / "cMD-run1"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    environment["MD_TOOLS_CONFIG"] = str(user)

    topology = str(root / "build" / "built.pdb")
    system = str(root / "build" / "built.xml")
    # The same order, restarts and output directories `run.sh` uses. Driving the generated scripts
    # directly keeps the test independent of whichever `md-openmm` happens to be on PATH.
    chain = (
        ("../min/min.py", None, "../min"),
        ("eq/eq_1.py", "../min/min.xml", "eq"),
        ("eq/eq_2.py", "eq/eq_1.xml", "eq"),
        ("eq/eq_3.py", "eq/eq_2.xml", "eq"),
        ("cMD.py", "eq/eq_3.xml", "."),
    )
    for script, restart, odir in chain:
        command = [sys.executable, str(run / script), "-p", topology, "-s", system,
                   "-odir", odir, "--cpu"]
        if restart:
            command += ["-c", restart]
        done = subprocess.run(command, cwd=run, capture_output=True, text=True,
                              timeout=1800, env=environment)
        assert done.returncode == 0, (
            f"{script} failed\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}")
    return run


def _series(run: Path, key: str):
    """One stage's CV series, by the key it is filed under, as a header and a list of rows."""
    found = sorted(run.rglob(f"{key}.cv.csv"))
    assert len(found) == 1, f"expected exactly one {key}.cv.csv under {run}, got {found}"
    lines = found[0].read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    return found[0], header, [dict(zip(header, line.split(","))) for line in lines[1:]]


def test_minimisation_writes_no_series(completed):
    """A minimiser's iterations have no timestep, so a `time_ps` for them would be a fiction."""
    assert not sorted(completed.rglob("min.cv.csv"))


@pytest.mark.parametrize("key", sorted(STAGE_STEPS))
def test_the_observations_are_uniformly_spaced(completed, key):
    """No gap of a different width, which is what the divisibility refusal exists to guarantee."""
    _path, _header, rows = _series(completed, key)
    steps = [int(row["step"]) for row in rows]
    gaps = sorted({second - first for first, second in zip(steps, steps[1:])})
    assert gaps == [INTERVAL], (
        f"{key}: observations are spaced {gaps}, not a uniform {INTERVAL} steps: {steps}")


@pytest.mark.parametrize("key", sorted(STAGE_STEPS))
def test_the_series_spans_exactly_the_stage(completed, key):
    """The last row is the step the stage's own result is quoted at."""
    _path, _header, rows = _series(completed, key)
    steps = [int(row["step"]) for row in rows]
    assert steps[-1] - steps[0] == STAGE_STEPS[key], (
        f"{key}: the series spans {steps[-1] - steps[0]} steps for a stage of "
        f"{STAGE_STEPS[key]}: first {steps[0]}, last {steps[-1]}")
    assert len(steps) == STAGE_STEPS[key] // INTERVAL + 1, steps


@pytest.mark.parametrize("key", sorted(STAGE_STEPS))
def test_every_frame_index_names_a_frame_that_exists(completed, key):
    """An index past the end of the file it indexes is unindexable."""
    netCDF4 = pytest.importorskip("netCDF4")
    _path, _header, rows = _series(completed, key)
    claimed = [int(row["trajectory_frame_index"]) for row in rows
               if row["trajectory_frame_index"]]
    trajectory = sorted(completed.rglob(f"solute_{key}.nc")) or \
        sorted(completed.rglob("solute_prod1.nc"))
    assert len(trajectory) == 1, f"{key}: expected one solute trajectory, got {trajectory}"
    with netCDF4.Dataset(str(trajectory[0])) as dataset:
        frames = dataset.dimensions["frame"].size
    out_of_range = [index for index in claimed if not 0 <= index < frames]
    assert not out_of_range, (
        f"{key}: {trajectory[0].name} holds {frames} frame(s), valid indices "
        f"0..{frames - 1}, but the series names {out_of_range}")


def test_every_commit_agrees_with_the_cv_prefix_it_was_written_with(completed):
    """A generation's `steps_done` and its committed CV prefix describe the same progress.

    Both are written in ONE transaction, deliberately, "so the prefix a resume trusts and the
    coordinates it restores were committed together". They are therefore two statements about the
    same instant and must agree: `rows` observations at a cadence of `INTERVAL`, counting the
    initial one, means `(rows - 1) * INTERVAL` steps of THIS stage have been integrated.

    This is what the resume arithmetic depends on. `remaining = steps - done` reads `steps_done`
    as this stage's own progress, so a commit that records an absolute step instead makes a
    resumed stage integrate fewer steps than it was asked for -- and report completion.
    """
    import json

    generations = sorted((completed / "cMD.checkpoints" / "checkpoints").glob("generation_*.json"))
    assert generations, "the stage committed no checkpoint generation to check"
    disagreed = []
    for generation in generations:
        state = json.loads(generation.read_text(encoding="utf-8"))["state"]
        rows = ((state.get("cv_prefix") or {}).get("rows")) or 0
        if not rows:
            continue
        implied = (int(rows) - 1) * INTERVAL
        recorded = int(state["steps_done"])
        if recorded != implied:
            disagreed.append(
                f"{generation.name}: steps_done = {recorded}, but its committed CV prefix of "
                f"{rows} row(s) at an interval of {INTERVAL} vouches for {implied} step(s)")
    assert not disagreed, (
        "a commit's step count and its own CV prefix disagree:\n  " + "\n  ".join(disagreed))


@pytest.mark.parametrize("key", sorted(STAGE_STEPS))
def test_no_commit_claims_more_steps_than_the_stage_has(completed, key):
    """`remaining = steps - done` is negative once `done` exceeds the stage's length."""
    import json

    directory = next((path for path in completed.rglob(f"{key}.checkpoints")), None)
    if directory is None:
        pytest.skip(f"{key} kept no checkpoint directory")
    overrun = []
    for generation in sorted((directory / "checkpoints").glob("generation_*.json")):
        recorded = int(json.loads(generation.read_text(encoding="utf-8"))["state"]["steps_done"])
        if recorded > STAGE_STEPS[key]:
            overrun.append(f"{generation.name}: steps_done = {recorded}")
    assert not overrun, (
        f"{key} is {STAGE_STEPS[key]} steps long, but " + ", ".join(overrun))


def test_a_checkpoint_without_the_step_convention_refuses_and_touches_nothing(completed, tmp_path):
    """An old-convention record is refused, in the READ-ONLY phase, before any log is opened.

    `steps_done` alone cannot say what it counts from, so a record lacking `absolute_step` is
    ambiguous and is refused rather than guessed. Two things are asserted, and the second is the
    one that is easy to get wrong: the refusal must not replace the prior run's `.out` or its
    machine record with a `status: failed` account of a run that never started. Completion is
    read from a machine record, which is exactly why a rejected attempt may not become one.
    """
    import json
    import shutil

    from md_tools.run.continuation import ContinuationError, validate_stage_continuation

    tree = tmp_path / "run"
    shutil.copytree(completed, tree, symlinks=True)
    checkpoints = tree / "cMD.checkpoints"
    if not (checkpoints / "current_checkpoint.json").is_file():
        pytest.skip("the stage kept no committed checkpoint to age")

    # Age every generation to the old convention by dropping the key, exactly as a record from
    # before it existed would look.
    aged = 0
    for generation in (checkpoints / "checkpoints").glob("generation_*.json"):
        document = json.loads(generation.read_text(encoding="utf-8"))
        if document["state"].pop("absolute_step", None) is not None:
            generation.write_text(json.dumps(document), encoding="utf-8")
            aged += 1
    assert aged, "no generation carried `absolute_step`, so there was nothing to age"

    guarded = {path: path.read_bytes() for path in sorted(tree.glob("cMD.*"))
               if path.is_file() and path.suffix in {".out", ".log", ".csv"}}
    assert guarded, "nothing to protect: the run wrote no .out, .log or .csv"

    # THE RUNTIME'S ROUTE: it knows its own checkpoint path.
    with pytest.raises(ContinuationError) as refusal:
        validate_stage_continuation(tree, stage={}, definition=None, own_checkpoints=checkpoints)
    assert "0.6.3 or earlier" in str(refusal.value), str(refusal.value)
    assert "--overwrite" in str(refusal.value)

    # THE PUBLIC SURFACE'S ROUTE: it has the stage name, and the generation records one.
    import json as _json

    recorded = _json.loads(
        sorted((checkpoints / "checkpoints").glob("generation_*.json"))[-1]
        .read_text(encoding="utf-8"))["state"].get("stage")
    if recorded:
        with pytest.raises(ContinuationError):
            validate_stage_continuation(tree, stage={}, definition=None, stage_name=recorded)

    # AND A NEIGHBOUR'S RECORD IS NOT THIS STAGE'S BUSINESS: the equilibration stages share one
    # `-odir`, so an aged tree must not refuse a stage that never reads it.
    validate_stage_continuation(tree, stage={}, definition=None,
                                own_checkpoints=tree / "nothing-of-mine.checkpoints")

    for path, before in guarded.items():
        assert path.read_bytes() == before, f"the refusal modified {path.name}"


def test_overwrite_does_not_validate_the_generations_it_replaces(completed, tmp_path):
    """`--overwrite` starts CLEAN, so it must not refuse over data it is about to discard."""
    import json
    import shutil

    from md_tools.run.continuation import validate_stage_continuation

    tree = tmp_path / "run"
    shutil.copytree(completed, tree, symlinks=True)
    for generation in (tree / "cMD.checkpoints" / "checkpoints").glob("generation_*.json"):
        document = json.loads(generation.read_text(encoding="utf-8"))
        document["state"].pop("absolute_step", None)
        generation.write_text(json.dumps(document), encoding="utf-8")

    validate_stage_continuation(tree, stage={}, definition=None, overwrite=True)


def test_the_reported_torsion_is_the_claimed_frames_torsion(completed):
    """THE test: recompute each aligned row's torsions from the frame it names, independently.

    An index that is inside the file but off by a fixed amount passes every structural check and
    reports another configuration's torsion under this row's step.
    """
    mdtraj = pytest.importorskip("mdtraj")
    _path, _header, rows = _series(completed, "cMD")
    trajectory = sorted(completed.rglob("solute_prod1.nc"))
    assert len(trajectory) == 1, trajectory
    # `build/` is a SIBLING of the run, reached through the symlink the generated directory
    # carries: it belongs to the system, and every run on it shares one copy.
    topology = completed / "build" / "built.pdb"
    assert topology.is_file(), f"no built.pdb to read the trajectory against at {topology}"
    frames = mdtraj.load(str(trajectory[0]), top=str(topology))

    checked = 0
    for row in rows:
        if not row["trajectory_frame_index"]:
            continue
        index = int(row["trajectory_frame_index"])
        assert 0 <= index < len(frames), (
            f"step {row['step']} names frame {index} of {len(frames)}")
        for name, quartet in (("phi", [4, 6, 8, 14]), ("psi", [6, 8, 14, 16])):
            expected = math.degrees(float(
                mdtraj.compute_dihedrals(frames[index], [quartet])[0][0]))
            reported = float(row[name])
            difference = abs((reported - expected + 180.0) % 360.0 - 180.0)
            assert difference < 1e-2, (name, row["step"], index, reported, expected)
        checked += 1
    assert checked >= 2, f"only {checked} row(s) could be checked against a stored frame"
