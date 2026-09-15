"""A shorter prefix that looks coherent must not quietly discard committed samples.

THE RISK

    A ladder checkpoint records three things about its collective-variable series: the absolute
    `step` the dynamics reached, `extra.cv_rows`, and a per-state `cv_prefix` carrying each
    state's row count, digest and cost. The validation added earlier reconciles the last two
    against each other -- the block count against every per-state entry, and each cost against
    its own rows -- but never against the FIRST.

    So a prefix claiming fewer rows than the checkpoint's own progress implies is internally
    perfect: every digest matches the shorter prefix, every cost matches the shorter count, every
    state agrees. The continuation accepts it, truncates each series to that shorter length, and
    then resumes dynamics from the recorded step. The rows between the two are committed
    scientific samples, and they are gone -- while the series that remains has a silent gap
    between its last row and the step the run continues from.

    That is data loss with no error, and it is the one shape of corruption that every check in
    place was blind to, because each check was consistent with the others.

WHERE THE EXPECTATION COMES FROM

    Not from `cv_rows`, which is the counter under test. From the checkpoint's committed step and
    the configured observation grid: a ladder observes step 0 and then every `interval_steps`, so
    a series whose first committed row is at `first` and whose run reached `step` holds
    `(step - first) / interval + 1` rows. Reading `first` from the validated committed prefix
    rather than assuming zero is what makes this correct for an extension, whose series begins at
    its own first step rather than at the origin.

    Checkpoint cadence, CV cadence and frame cadence are independent. The reconciliation uses
    only the CV interval, and the committed step is required to sit on that grid because the
    ladder commits its checkpoints at exchange boundaries which the CV cadence divides.

PLATFORM_POLICY_EXEMPTION: `--cpu`. What is under test is which rows survive a resume -- file
arithmetic, identical on every platform. The same resume runs on real CUDA in
`test_cv_cuda_lanes.py`.
"""
from __future__ import annotations

import json
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

STATES = 2
EXCHANGE_EVERY = 10
EXCHANGES = 6
CV_EVERY = 5
TOTAL = EXCHANGE_EVERY * EXCHANGES
#: Derived by hand from the schedule: step 0 and every 5 steps to 60.
COMPLETE_ROWS = TOTAL // CV_EVERY + 1

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


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ladder-progress")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        # `crd_printout_whole` must be set: a ladder's CV rows and its progress reconciliation
        # both count frames of the WHOLE state trajectory, and the key defaults to 0.
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY,
                      "crd_printout_whole": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260907},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./REST2-run1", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=900).returncode == 0

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    context = openmm.Context(system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                             openmm.Platform.getPlatformByName("Reference"))
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
    base["OPENMM_CPU_THREADS"] = "1"
    done = subprocess.run(
        [sys.executable, str(project / "REST2-run1" / "REST2.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "REST2-run1", capture_output=True, text=True, timeout=1800,
        env={**base, **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint(destination: Path):
    from md_tools.remd import storage

    return storage.ReplicaCheckpoint(destination / "REST2_checkpoint.nc").read()


def _interrupted(project, destination):
    """A genuine interruption, leaving a committed prefix and an uncommitted tail."""
    crashed = _run(project, destination, expect=1,
                   environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                                "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                                "MD_TOOLS_FAIL_PROPAGATION_AFTER": "3"})
    assert crashed.returncode != 0
    return _checkpoint(destination)


def _expected_rows(checkpoint, first_step=0):
    """From committed progress and the configured grid -- never from `cv_rows`."""
    step = int(checkpoint["step"])
    assert step % CV_EVERY == 0, (
        f"the committed step {step} is off the {CV_EVERY}-step CV grid; this schedule's exchange "
        f"boundaries are multiples of the CV cadence, so the expectation would be a guess")
    return (step - first_step) // CV_EVERY + 1


def _rewrite_prefix(destination: Path, rows: int, definition_names=("phi", "psi")):
    """Shrink the committed prefix to `rows`, leaving every other field CONSISTENT with it.

    Digests, per-state counts, the block count and every cost are recomputed for the shorter
    prefix, so nothing internal disagrees. That is the whole point: the record is coherent, and
    only its disagreement with the checkpoint's own progress reveals it.
    """
    import netCDF4

    from md_tools.cv import prefix as cv_prefix
    from md_tools.cv.cost import CVCost, cost_record

    dataset = netCDF4.Dataset(str(destination / "REST2_checkpoint.nc"), "a")
    try:
        extra = json.loads(dataset.extra_json)
        block = extra["cv_prefix"]
        scope = CVCost(observations=rows, evaluations=rows * len(definition_names),
                       wall_seconds=0.001)
        total = CVCost()
        for entry in block["states"]:
            index = int(entry["state_index"])
            series = destination / f"cv_state{index}.csv"
            entry["rows"] = rows
            entry["prefix_sha256"] = cv_prefix.prefix_digest(series, rows)
            entry["cost"] = cost_record(scope, scope, rows=rows)
            total = total.plus(scope)
        block["rows"] = rows
        aggregate = cost_record(total, total, rows=rows * len(block["states"]))
        aggregate["aggregation"] = "sum over thermodynamic states"
        aggregate["per_state"] = [
            {**cost_record(scope, scope, rows=rows), "state_index": int(e["state_index"])}
            for e in block["states"]]
        block["cost"] = aggregate
        extra["cv_rows"] = rows
        dataset.extra_json = json.dumps(extra, sort_keys=True)
    finally:
        dataset.close()


# --- the control: a genuine interrupted resume keeps every committed sample -------------------

def test_a_valid_interrupted_resume_retains_every_committed_row(project, tmp_path):
    """The control, and the arithmetic the defect case depends on.

    Without this, a refusal-everywhere implementation would satisfy the case below.
    """
    reference = tmp_path / "reference"
    _run(project, reference)
    want = {i: _rows(reference / f"cv_state{i}.csv") for i in range(STATES)}
    assert len(want[0]) == COMPLETE_ROWS

    destination = tmp_path / "resumed"
    checkpoint = _interrupted(project, destination)
    committed = _expected_rows(checkpoint)
    assert 0 < committed < COMPLETE_ROWS, (
        f"the interruption committed {committed} of {COMPLETE_ROWS} rows; it must land mid-run")

    # The uncommitted tail a crash leaves behind is legitimately present and longer.
    on_disk = len(_rows(destination / "cv_state0.csv"))
    assert on_disk >= committed

    _run(project, destination, "--resume")
    for index in range(STATES):
        got = _rows(destination / f"cv_state{index}.csv")
        assert len(got) == COMPLETE_ROWS
        assert [r["step"] for r in got] == [r["step"] for r in want[index]]
        assert got == want[index], f"state {index} differs from the uninterrupted reference"


# --- the defect: a coherent shorter prefix must not silently discard committed rows -----------

def test_a_prefix_shorter_than_committed_progress_is_refused(project, tmp_path):
    """THE data-loss case.

    Every field of the record agrees with every other; only the checkpoint's own progress says
    otherwise. Accepting it truncates committed samples and then resumes dynamics from a later
    step, leaving a gap no later reader can detect.
    """
    destination = tmp_path / "shortened"
    checkpoint = _interrupted(project, destination)
    honest = _expected_rows(checkpoint)
    assert honest >= 3, "need room to shorten the prefix by more than one row"

    before = {i: _rows(destination / f"cv_state{i}.csv") for i in range(STATES)}
    _rewrite_prefix(destination, honest - 2)

    refused = _run(project, destination, "--resume", expect=1)
    assert refused.returncode != 0, refused.stdout[-3000:]

    combined = refused.stdout + refused.stderr + \
        (destination / "REST2.out").read_text(encoding="utf-8", errors="replace")
    # The reconciliation specifically, not merely "something went wrong": before the fix this run
    # also ended nonzero, but only from the completion check at the END, with the rows already
    # gone and a whole segment of dynamics recomputed on top of the gap.
    assert "observation grid" in combined, combined[-3000:]
    assert f"exactly {honest} " in combined, combined[-3000:]

    # And the committed samples are still on disk: the refusal happened before any truncation.
    for index in range(STATES):
        assert _rows(destination / f"cv_state{index}.csv") == before[index], (
            f"state {index} was truncated by a run that then refused")


def test_a_prefix_longer_than_committed_progress_is_refused(project, tmp_path):
    """The other direction: a prefix vouching for rows the run never reached."""
    destination = tmp_path / "lengthened"
    checkpoint = _interrupted(project, destination)
    honest = _expected_rows(checkpoint)
    on_disk = len(_rows(destination / "cv_state0.csv"))
    if on_disk <= honest:
        pytest.skip("this interruption left no uncommitted tail to over-claim")

    before = {i: _rows(destination / f"cv_state{i}.csv") for i in range(STATES)}
    _rewrite_prefix(destination, honest + 1)
    refused = _run(project, destination, "--resume", expect=1)
    combined = refused.stdout + refused.stderr + \
        (destination / "REST2.out").read_text(encoding="utf-8", errors="replace")
    assert "observation grid" in combined, combined[-3000:]
    for index in range(STATES):
        assert _rows(destination / f"cv_state{index}.csv") == before[index]
