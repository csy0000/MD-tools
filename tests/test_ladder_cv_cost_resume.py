"""REST2 and rREST2 CV cost across TWO consecutive interruptions.

WHY TWO

    One resume passes whether the accumulation is "restored prefix + this segment" (correct),
    "just the restored prefix", or "just this segment" -- the last two are wrong and a single
    interruption cannot tell them apart, because with one prior segment the three answers can
    coincide. The second interruption separates them: only the correct rule reproduces the
    uninterrupted reference.

WHY TWO TORSIONS

    With a single CV, `cv_observations` and `cv_evaluations` are numerically equal, so a counter
    that increments once per reporter call passes. That is exactly how the original misnomer
    survived a full test suite. Every definition here has two.

WHY ONE CPU THREAD

    Comparing a resumed CV series value-by-value against an uninterrupted reference requires the
    two runs to be the same trajectory, and OpenMM's CPU platform is only reproducible at a fixed
    thread count: its force reductions are summed in thread-completion order, so the same seed on
    the same machine gives a different trajectory when the pool size differs. Two replicate ladder
    runs measurably diverge by the first observation with the default pool and agree to the last
    digit with `OPENMM_CPU_THREADS=1`. This is a property of the platform, not of the ladder --
    nothing here pins a thread count in production.

PLATFORM_POLICY_EXEMPTION: the ladders run under `--cpu`. What is under test is counter
arithmetic across invocations, which is identical on every platform; the same runtimes are
exercised on real CUDA in `test_cv_cuda_lanes.py` and under MPI in `test_cv_mpi_cuda_lanes.py`.
"""
from __future__ import annotations

import csv
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
N_CV = 2
EXPECTED_ROWS = TOTAL // CV_EVERY + 1

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


def _reservoir(root: Path, *, frames=6, tau_max=0.5):
    """Reuse the pre-refresh test's builder so both files agree what a reservoir is."""
    from tests.test_rrest2_pre_refresh_cv import _reservoir as build

    build(root, frames=frames, tau_max=tau_max)


@pytest.fixture(scope="module", params=["REST2", "rREST2"])
def project(request, tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    protocol = request.param
    root = tmp_path_factory.mktemp(f"cost-{protocol}")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")

    document = {
        "protocol": protocol, "solvent": "implicit",
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
        "dynamics": {"seed": 20260904},
    }
    if protocol == "rREST2":
        _reservoir(root)
        document["reservoir"] = {"enabled": True, "path": "../reservoir.nc",
                                 "refresh_interval_exchanges": 1, "velocities": "inherit"}
    (root / f"{protocol}.config").write_text(yaml.safe_dump(document), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", f"./{protocol}", "--config", str(root / f"{protocol}.config")],
        cwd=root, capture_output=True, text=True, timeout=900)
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
    return root, protocol


def _run(project, destination: Path, *extra, environment=None, expect=0):
    root, protocol = project
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base["OPENMM_CPU_THREADS"] = "1"     # see WHY ONE CPU THREAD
    done = subprocess.run(
        [sys.executable, str(root / protocol / f"{protocol}.py"),
         "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
         "-c", str(root / "initial_state.xml"), "-odir", str(destination), "--cpu", *extra],
        cwd=root / protocol, capture_output=True, text=True, timeout=1800,
        env={**base, **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _cost(destination: Path):
    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    block = manifest["collective_variables"]
    assert block is not None, "the manifest records no CV series"
    return block["cost"], manifest


def test_a_fresh_ladder_reports_scalar_evaluations_and_equal_scopes(project, tmp_path):
    """The reference, and the assertion a call-counter cannot satisfy."""
    destination = tmp_path / "fresh"
    _run(project, destination)
    cost, _manifest = _cost(destination)

    assert cost["cumulative"]["cv_observations"] == STATES * EXPECTED_ROWS
    assert cost["cumulative"]["cv_evaluations"] == STATES * EXPECTED_ROWS * N_CV, (
        "each observation of a two-torsion definition is two scalar evaluations, per state")
    assert cost["segment"] == cost["cumulative"], "a fresh run's scopes are equal"
    assert cost["aggregation"] == "sum over thermodynamic states"
    assert len(cost["per_state"]) == STATES, "per-state records make the total auditable"
    assert sum(e["cumulative"]["cv_observations"] for e in cost["per_state"]) \
        == cost["cumulative"]["cv_observations"]


def test_cost_and_series_survive_two_interruptions(project, tmp_path):
    """THE regression: two consecutive interruptions, compared against an uninterrupted run."""
    reference = tmp_path / "reference"
    _run(project, reference)
    ref_cost, _ = _cost(reference)
    ref_series = {index: _rows(reference / f"remd{index}.cv.csv") for index in range(STATES)}

    destination = tmp_path / "resumed"
    first = _run(project, destination, expect=1,
                 environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                              "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                              "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    assert first.returncode != 0
    second = _run(project, destination, "--resume", expect=1,
                  environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                               "MD_TOOLS_FAIL_LADDER_AT": "after-checkpoint",
                               "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    assert second.returncode != 0
    _run(project, destination, "--resume")

    cost, _ = _cost(destination)
    assert cost["cumulative"] == ref_cost["cumulative"] or (
        cost["cumulative"]["cv_observations"] == ref_cost["cumulative"]["cv_observations"]
        and cost["cumulative"]["cv_evaluations"] == ref_cost["cumulative"]["cv_evaluations"]), (
        f"after two resumes the cumulative cost is {cost['cumulative']} against the "
        f"uninterrupted {ref_cost['cumulative']}: earlier work was lost or double counted")
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"], (
        "the segment equals the cumulative, so earlier segments were not carried")
    assert cost["segment"]["cv_observations"] > 0
    assert cost["cumulative"]["wall_seconds"] >= cost["segment"]["wall_seconds"] >= 0.0

    for index in range(STATES):
        got = _rows(destination / f"remd{index}.cv.csv")
        want = ref_series[index]
        assert [r["step"] for r in got] == [r["step"] for r in want], f"state {index} step grid"
        assert [r["walker_index"] for r in got] == [r["walker_index"] for r in want]
        # THE VALUES, not only the grid. A resume that restores coordinates but not the
        # integrator's pseudo-random stream produces a correct, statistically exact and
        # COMPLETELY DIFFERENT trajectory from the resume point on -- which every check above
        # still passes. Only this one refuses it.
        for column in ("phi", "psi"):
            assert [r[column] for r in got] == [r[column] for r in want], (
                f"state {index} {column} diverges from the uninterrupted reference: the "
                f"continuation did not resume the trajectory, it started a new one")
        assert [r["trajectory_frame_index"] for r in got] \
            == [r["trajectory_frame_index"] for r in want], f"state {index} frame references"
        assert len(got) == EXPECTED_ROWS
        assert len({r["step"] for r in got}) == EXPECTED_ROWS, "a step was written twice"


def test_re_entering_a_completed_ladder_evaluates_nothing_further(project, tmp_path):
    """Verification and CSV reads are not CV work and must not be counted as such.

    Re-entry is `--resume`: a plain re-run refuses on the outputs it would overwrite, which is
    the collision check doing its job and says nothing about CV accounting. `--resume` on a
    finished ladder is the path that actually reads the committed series back, verifies it, and
    must then evaluate nothing.
    """
    destination = tmp_path / "reentry"
    _run(project, destination)
    before, _ = _cost(destination)
    rows_before = {index: _rows(destination / f"remd{index}.cv.csv") for index in range(STATES)}

    _run(project, destination, "--resume")
    after, _ = _cost(destination)
    for index in range(STATES):
        assert _rows(destination / f"remd{index}.cv.csv") == rows_before[index], (
            f"state {index}: re-entry rewrote a committed series")
    assert after["cumulative"]["cv_evaluations"] == before["cumulative"]["cv_evaluations"], (
        "re-entering a completed ladder evaluated collective variables again")
    assert after["segment"]["cv_evaluations"] == 0, (
        "the re-entering invocation performed no CV work, so its segment must be empty")


def test_the_record_says_how_many_cpu_threads_produced_it(project, tmp_path):
    """Whether two runs are comparable must be readable from the record.

    OpenMM's CPU platform sums force reductions in thread-completion order, so it reproduces
    itself only at a fixed pool size: the same ladder, same seed, same inputs, same machine,
    diverges by the first observation when the pool differs. That is a property of the platform
    and nothing here can change it -- but a record that does not say what the pool was leaves a
    reader comparing two runs with no way to tell a real divergence from a different thread
    count, which is exactly how this was first mistaken for a defect in the ladder.
    """
    destination = tmp_path / "threads"
    _run(project, destination)
    execution = json.loads(
        (destination / "restart.json").read_text(encoding="utf-8"))["execution"]
    assert execution["platform"] == "CPU", execution["platform"]
    assert execution["cpu_threads"] == 1, (
        f"the run was launched with OPENMM_CPU_THREADS=1 and the record says "
        f"{execution.get('cpu_threads')!r}")
