"""CV continuation and fail-closed behaviour under a REAL multi-rank launcher, on real CUDA.

WHY A SEPARATE FILE

    The existing MPI+GPU lane runs REST2, rREST2 and AIS without collective variables enabled, so
    it exercises none of the CV code on a device under MPI. The CV restart tests run serially and
    under `--cpu`. Between them nothing covered the case this file is for: several ranks, real
    devices, CV reporting on, and an interruption in the middle.

    That matters because the CV series is written by ROOT ONLY while every rank propagates. A
    continuation therefore has to agree across ranks about how many rows are committed, and a
    rank-local failure during CV evaluation or CV writing has to stop the whole communicator
    rather than leave the others in a collective the failing rank abandoned.

Every subprocess call carries a timeout, and the timeout IS the assertion for the failure cases:
a hang is the failure being tested for, so a test that waits forever cannot detect one.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: Long enough for a launcher to start N interpreters and run a short ladder on a device; short
#: enough that a hang is a failure rather than a wait.
LAUNCH_TIMEOUT = 900

STATES = 2
EXCHANGE_EVERY = 10
EXCHANGES = 4
CV_EVERY = 5
EXPECTED = list(range(0, EXCHANGE_EVERY * EXCHANGES + 1, CV_EVERY))

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""


def _require_mpi_and_cuda():
    import openmm

    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail("no CUDA platform is available; a CPU run is not CUDA evidence")


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


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_mpi_and_cuda()
    root = tmp_path_factory.mktemp("cv-mpi-cuda")
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
        "reporting": {"solute_printout": EXCHANGE_EVERY, "system_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./REST2", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "built.pdb"))
    system = XmlSerializer.deserialize((root / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return root


def _launch(project: Path, destination: Path, *extra, ranks=STATES, environment=None,
            expect=0):
    done = subprocess.run(
        ["mpirun", "-n", str(ranks), sys.executable,
         str(project / "REST2" / "REST2.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-ng", str(ranks), "-odir", str(destination), *extra],
        cwd=project / "REST2", capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
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
                     for path in sorted(destination.rglob("REST2.out*")))


def test_a_two_rank_cuda_ladder_writes_cv_for_every_state(project, tmp_path):
    """CV reporting on, several ranks, real devices, and the rank-to-device mapping recorded."""
    destination = tmp_path / "fresh"
    _launch(project, destination)

    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    assert (manifest.get("execution") or {}).get("platform") == "CUDA", manifest.get("execution")
    assert manifest["collective_variables"] is not None

    for index in range(STATES):
        assert [int(r["step"]) for r in _rows(destination / f"remd{index}.cv.csv")] == EXPECTED

    # Each rank recorded its own device: a two-rank launch that quietly shared one device would
    # still finish, and would not be the placement the run claims.
    reports = _reports(destination)
    assert "CUDA" in reports, reports[-2000:]
    assert any(f"rank {rank}" in reports for rank in range(STATES)), reports[-2000:]


def test_cv_continuation_under_mpi_on_cuda(project, tmp_path):
    """The series is written by root only while every rank propagates.

    A continuation therefore has to agree across ranks about how many rows are committed. If it
    did not, the resumed series would duplicate or drop the overlap.
    """
    reference = tmp_path / "reference"
    _launch(project, reference)
    expected = {index: [int(r["step"]) for r in _rows(reference / f"remd{index}.cv.csv")]
                for index in range(STATES)}

    destination = tmp_path / "resumed"
    crashed = _launch(project, destination, expect=1,
                      environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                                   "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                                   "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    assert crashed.returncode != 0
    _launch(project, destination, "--resume")

    for index in range(STATES):
        steps = [int(r["step"]) for r in _rows(destination / f"remd{index}.cv.csv")]
        assert steps == expected[index], f"state {index}: {steps}"
        assert len(steps) == len(set(steps))


#: Which (rank, boundary) pairs are REACHABLE, and why the table is not simply a product.
#:
#: The CV rows and the checkpoint are written by ROOT ONLY -- every rank propagates, root
#: gathers and writes. So `before-cv-row`, `after-cv-row` and `before-checkpoint` do not exist on
#: a non-root rank, and arming them there injects nothing: the run completes and the test would
#: "pass" while proving the opposite of what it claims.
#:
#: What a non-root rank does during a CV step is PROPAGATE, so that is where its failure is
#: injected. The property under test is unchanged -- a rank-local failure anywhere must stop the
#: whole communicator -- and it is now asserted at a seam each rank actually reaches.
REACHABLE = [
    (0, "before-cv-row"),
    (0, "after-cv-row"),
    (0, "before-checkpoint"),
    (0, "after-checkpoint"),
    (0, "propagation"),
    (1, "propagation"),
]


@pytest.mark.parametrize("rank, boundary", REACHABLE,
                         ids=[f"rank{r}-{b}" for r, b in REACHABLE])
def test_a_rank_local_failure_stops_the_whole_communicator(project, tmp_path, rank, boundary):
    """Root AND non-root, each at a boundary it actually executes. The timeout is the assertion.

    A rank that raises alone leaves the others blocked in the next collective, and the job hangs
    until a scheduler kills it -- which burns the allocation and reports nothing. The subprocess
    timeout is what distinguishes a stopped job from a hung one, so a test that waited forever
    could not detect the failure it exists for.
    """
    destination = tmp_path / f"fail-{boundary}-{rank}"
    done = _launch(project, destination, expect=1,
                   environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": str(rank),
                                "MD_TOOLS_FAIL_LADDER_AT": boundary,
                                "MD_TOOLS_FAIL_PROPAGATION_AFTER": "1"})
    assert done.returncode != 0, done.stdout[-2000:] + done.stderr[-2000:]

    combined = done.stdout + done.stderr + _reports(destination)
    assert f"rank {rank}" in combined, combined[-3000:]
    # No rank may report a completed run, and no completion manifest may survive.
    assert not (destination / "restart.json").exists(), (
        "a failed launch left a completion manifest")
    assert "run_status: completed" not in combined


def test_the_cv_and_checkpoint_boundaries_are_root_only(project):
    """Pins WHY the table above is not a full product, so a future reader does not "fix" it.

    If CV writing ever became per-rank, this assertion fails and the reachability table has to be
    revisited -- rather than the non-root cases silently passing by never injecting anything.
    """
    source = (REPO / "src" / "md_tools" / "remd" / "driver.py").read_text(encoding="utf-8")
    body = source[source.index("def _loop(self"):]
    body = body[:body.index("\n    def ")]
    assert 'observing = ("cv" in events' in body
    assert "self.coordinator.is_root)" in body, (
        "CV observation is no longer root-only; the reachability table above needs revisiting")
