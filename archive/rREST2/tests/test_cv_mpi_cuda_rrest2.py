"""rREST2 with CV reporting under a REAL multi-rank launcher, on real CUDA.

WHY A SEPARATE FILE

    `test_cv_mpi_cuda_lanes.py` runs REST2 and only REST2. Its name and the evidence report
    described it as the MPI+CUDA CV lane, which overstated what had actually been executed:
    rREST2's reservoir refresh -- the one part of the ladder that REPLACES a walker's
    configuration mid-run, and the part whose CV semantics were most recently corrected -- had
    never been exercised on a device under MPI. The pre-refresh semantics were proven serially
    and under `--cpu`.

    That gap matters specifically here. A refresh replaces the top rung's coordinates between
    the CV row being measured and the trajectory frame being written, and the deep snapshot that
    keeps the two apart is taken on the ROOT rank while every rank propagates. Whether that
    still holds when the rungs are spread across ranks and devices is not something a serial
    CPU run can answer.

WHAT IS FORCED, NOT HOPED FOR

    `refresh_interval_exchanges: 1` refreshes at every exchange, so the event is deterministic;
    a test that merely hoped a refresh would occur would pass vacuously on the run where none
    did. `test_a_refresh_actually_happened` states that plainly before anything else is claimed.

Both supported velocity policies are covered: `inherit` installs the recorded momentum
("stored", the default) and `resample` draws fresh Maxwell momenta. Both go through the same
configuration-replacement path.

Every subprocess call carries a timeout, and for the failure cases the timeout IS the
assertion: a hang is the failure being tested for.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_rrest2_pre_refresh_cv import QUARTET, _build, _refresh_events, _reservoir, _torsion

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

LAUNCH_TIMEOUT = 1200

#: The ladder width IS the rank count: `owned_states` refuses any world size that is neither 1
#: nor exactly the number of states, so a three-rung ladder is a three-rank launch.
STATES = 3
EXCHANGE_EVERY = 10
EXCHANGES = 4
CV_EVERY = 5
N_CV = 2
EXPECTED_STEPS = list(range(0, EXCHANGE_EVERY * EXCHANGES + 1, CV_EVERY))

#: TWO torsions, per the accounting rule this whole change exists to enforce: with one, a
#: reporter-call counter and a scalar-evaluation counter are numerically identical and the
#: cost assertions below would pass against either.
PSI = [6, 8, 14, 16]
CV_YAML = f"""\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: {QUARTET}
  - name: psi
    type: torsion
    atom_indices: {PSI}
"""


def _require_mpi_and_cuda():
    import openmm

    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail("no CUDA platform is available; a CPU run is not CUDA evidence")


@pytest.fixture(scope="module", params=["stored", "maxwell"])
def project(request, tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_mpi_and_cuda()
    policy = {"stored": "inherit", "maxwell": "resample"}[request.param]
    root = tmp_path_factory.mktemp(f"rrest2-mpi-cuda-{request.param}")
    _build(root)
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    _reservoir(root)
    (root / "rREST2.config").write_text(yaml.safe_dump({
        "protocol": "rREST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        "reservoir": {"enabled": True, "path": "../reservoir.nc",
                      "refresh_interval_exchanges": 1, "velocities": policy},
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY,
                      "crd_printout_whole": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./rREST2-run1", "--config", str(root / "rREST2.config")],
        cwd=root, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    context = openmm.Context(system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                             openmm.Platform.getPlatformByName("CUDA"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return root


def _environment(root: Path, **extra):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base.update(extra)
    return base


def _launch(project: Path, destination: Path, *extra, ranks=STATES, environment=None, expect=0):
    """No `--cpu`. If CUDA is unavailable the run must fail rather than quietly use the CPU."""
    done = subprocess.run(
        ["mpirun", "-n", str(ranks), sys.executable,
         str(project / "rREST2-run1" / "rREST2.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-ng", str(ranks), "-odir", str(destination), *extra],
        cwd=project / "rREST2-run1", capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(project, **(environment or {})))
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _reports(destination: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                     for path in sorted(destination.rglob("rREST2.out*")))


def _cost(destination: Path):
    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    block = manifest["collective_variables"]
    assert block is not None, "the manifest records no CV series"
    return block["cost"]


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    """One uninterrupted multi-rank CUDA run, shared by the read-only assertions."""
    destination = tmp_path_factory.mktemp("rrest2-mpi-run") / "run"
    _launch(project, destination)
    return destination


def test_the_ladder_ran_on_cuda_across_ranks_with_cv_for_every_state(completed):
    manifest = json.loads((completed / "restart.json").read_text(encoding="utf-8"))
    execution = manifest.get("execution") or {}
    assert execution.get("platform") == "CUDA", execution
    assert int(execution.get("mpi_size") or 0) == STATES, execution
    assert manifest["collective_variables"] is not None

    for index in range(STATES):
        rows = _rows(completed / f"cv_state{index}.csv")
        assert [int(r["step"]) for r in rows] == EXPECTED_STEPS
        assert {"phi", "psi"} <= set(rows[0]), "both torsions must be reported"

    # Each rank recorded its own placement. A multi-rank launch quietly sharing one device would
    # still finish, and would not be the placement the run claims.
    reports = _reports(completed)
    assert "CUDA" in reports, reports[-2000:]
    assert any(f"rank {rank}" in reports for rank in range(STATES)), reports[-2000:]


def test_a_refresh_actually_happened(completed):
    """Every refresh assertion below is vacuous without one, so this states it plainly."""
    assert _refresh_events(completed), (
        "no accepted reservoir refresh was recorded under MPI, so the pre-refresh semantics "
        "were never exercised on a device and the tests below prove nothing")


def test_the_refreshed_row_holds_the_propagated_configuration(completed, project):
    """THE regression, now across ranks: the row must not hold the reservoir sample.

    The refreshed state's CV row is measured BEFORE the replacement. If it matched the reservoir
    sample instead, a row labelled pre-exchange would carry a coordinate that state never
    propagated -- and under MPI the snapshot that keeps them apart is taken on root while the
    owning rank is the one that propagated it.
    """
    from md_tools.md.phase_space import PhaseSpaceReader

    events = _refresh_events(completed)
    assert events, "no refresh to check"
    with PhaseSpaceReader(project / "reservoir.nc") as reader:
        samples = {index: _torsion(reader.frame(index)[0])
                   for index in {frame for _e, _s, frame in events}}

    checked = 0
    for exchange_index, state_index, frame in events:
        step = (exchange_index + 1) * EXCHANGE_EVERY
        rows = {int(r["step"]): r for r in _rows(completed / f"cv_state{state_index}.csv")}
        if step not in rows:
            continue
        reported = float(rows[step]["phi"])
        difference = abs((reported - samples[frame] + 180.0) % 360.0 - 180.0)
        assert difference > 1.0, (
            f"state {state_index} step {step} reports {reported} and the reservoir sample it "
            f"was refreshed from has {samples[frame]}: the row was evaluated after the "
            f"replacement")
        checked += 1
    assert checked, "no refreshed state had a CV row at its refresh step"


def test_the_refreshed_state_names_no_trajectory_frame(completed):
    """The frame written at that step holds the reservoir sample; this row does not."""
    events = _refresh_events(completed)
    assert events, "no refresh to check"
    checked = 0
    for exchange_index, state_index, _frame in events:
        step = (exchange_index + 1) * EXCHANGE_EVERY
        rows = {int(r["step"]): r for r in _rows(completed / f"cv_state{state_index}.csv")}
        if step not in rows:
            continue
        named = rows[step]["trajectory_frame_index"]
        assert named == "", (
            f"state {state_index} step {step} names frame {named!r}, but that frame holds the "
            f"reservoir sample and this row holds the propagated configuration")
        checked += 1
    assert checked, "no refreshed state had a CV row at its refresh step"


def test_unaffected_states_still_name_the_correct_post_exchange_frame(completed, project):
    """The fix must not simply blank the column: untouched states keep working references."""
    mdtraj = pytest.importorskip("mdtraj")

    refreshed_at = {((e + 1) * EXCHANGE_EVERY, s) for e, s, _f in _refresh_events(completed)}
    checked = 0
    for state_index in range(STATES):
        frames = mdtraj.load(str(completed / f"whole_state{state_index}_prod1.nc"),
                             top=str(project / "build" / "built.pdb"))
        for row in _rows(completed / f"cv_state{state_index}.csv"):
            named = row["trajectory_frame_index"]
            if named == "":
                continue
            assert (int(row["step"]), state_index) not in refreshed_at
            expected = math.degrees(float(
                mdtraj.compute_dihedrals(frames[int(named)], [QUARTET])[0][0]))
            reported = float(row["phi"])
            assert abs((reported - expected + 180.0) % 360.0 - 180.0) < 1e-2, (
                f"state {state_index} step {row['step']} names frame {named} but was measured "
                f"elsewhere")
            checked += 1
    assert checked, "no state retained a frame reference, so the column was over-blanked"


def test_the_fresh_run_reports_scalar_evaluations_per_state(completed):
    cost = _cost(completed)
    rows = len(EXPECTED_STEPS)
    assert cost["cumulative"]["cv_observations"] == STATES * rows
    assert cost["cumulative"]["cv_evaluations"] == STATES * rows * N_CV, (
        "each observation of a two-torsion definition is two scalar evaluations, per state")
    assert cost["segment"] == cost["cumulative"], "a fresh run's scopes are equal"
    assert len(cost["per_state"]) == STATES


def test_interruption_and_resume_through_mpi_reproduce_the_reference(project, tmp_path):
    """Interrupt a multi-rank CUDA rREST2 run and resume it, against an uninterrupted reference.

    The comparison is by VALUE, not only by step grid: a continuation that restored coordinates
    but not each rung's integrator stream would write a different trajectory onto a grid that
    still lines up perfectly.
    """
    reference = tmp_path / "reference"
    _launch(project, reference)
    want = {index: _rows(reference / f"cv_state{index}.csv") for index in range(STATES)}
    reference_cost = _cost(reference)

    destination = tmp_path / "resumed"
    crashed = _launch(project, destination, expect=1,
                      environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                                   "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                                   "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    assert crashed.returncode != 0
    _launch(project, destination, "--resume")

    cost = _cost(destination)
    assert cost["cumulative"]["cv_observations"] == reference_cost["cumulative"]["cv_observations"]
    assert cost["cumulative"]["cv_evaluations"] == reference_cost["cumulative"]["cv_evaluations"]
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"], (
        "the segment equals the cumulative, so earlier segments were not carried across the "
        "interruption")

    for index in range(STATES):
        got = _rows(destination / f"cv_state{index}.csv")
        assert [r["step"] for r in got] == [r["step"] for r in want[index]]
        assert len({r["step"] for r in got}) == len(got), "a step was written twice"
        for column in ("phi", "psi", "walker_index"):
            assert [r[column] for r in got] == [r[column] for r in want[index]], (
                f"state {index} {column} diverges from the uninterrupted reference")


#: Which (rank, boundary) pairs a THREE-rank rREST2 launch actually reaches. The table is not a
#: product, for the same reason it is not one in `test_cv_mpi_cuda_lanes.py`: the CV rows, the
#: checkpoint and the reservoir refresh are all written by ROOT while every rank propagates, so
#: arming those boundaries on a non-root rank injects nothing at all -- the run completes and the
#: test would sit in the suite proving the opposite of what it claims. What a non-root rank does
#: during a refresh step is PROPAGATE, so that is where its failure is injected.
REACHABLE = [
    (0, "after-cv-row"),
    (0, "after-checkpoint"),
    (0, "propagation"),
    (1, "propagation"),
    (2, "propagation"),
]


@pytest.mark.parametrize("rank, boundary", REACHABLE,
                         ids=[f"rank{r}-{b}" for r, b in REACHABLE])
def test_a_rank_local_failure_stops_the_whole_rrest2_communicator(project, tmp_path, rank,
                                                                  boundary):
    """A rank that raises alone must stop the job, not hang it. The timeout IS the assertion.

    rREST2 had continuation coverage under MPI and no injected-failure coverage: a rank dying
    mid-refresh left the others blocked in the next collective, and a job that hangs burns its
    allocation and reports nothing. A test that waited forever could not tell that from a job
    that is merely slow, so every launch here carries a timeout.
    """
    destination = tmp_path / f"fail-{boundary}-{rank}"
    done = _launch(project, destination, expect=1,
                   environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": str(rank),
                                "MD_TOOLS_FAIL_LADDER_AT": boundary,
                                "MD_TOOLS_FAIL_PROPAGATION_AFTER": "1"})
    assert done.returncode != 0, done.stdout[-2000:] + done.stderr[-2000:]

    combined = done.stdout + done.stderr + _reports(destination)
    assert f"rank {rank}" in combined, combined[-3000:]
    # No completion manifest may survive, and no rank may claim the run finished.
    assert not (destination / "restart.json").exists(), (
        "a failed launch left a completion manifest")
    assert "run_status: completed" not in combined
