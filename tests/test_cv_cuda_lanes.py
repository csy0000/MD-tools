"""Collective-variable reporting, on real CUDA devices, with CV actually enabled.

WHY THIS FILE EXISTS

    The coverage matrix cited `test_ais_cv_output.py` as CUDA evidence for the AIS CV path. That
    file invokes `--cpu`. The cMD and REST2 CV restart tests do the same, and the AIS cases in
    the MPI lane do not enable collective variables at all. So the CV runtimes had no CUDA
    evidence whatsoever, while the published matrix said they did -- which is worse than a gap,
    because a gap invites work and a false entry closes the question.

    Every test here runs WITHOUT `--cpu` and asserts the recorded platform is CUDA. A test that
    cannot get a device fails; it does not fall back, and it is not marked xfail. That is the
    point: a CPU run proves nothing about the platform and must never be counted as if it did.

The bookkeeping these runtimes perform -- grids, alignment, prefixes, manifests -- is asserted
exhaustively in the CPU files, which are fast and can afford the matrix. What is asserted HERE is
that the same code paths execute on a device and produce the same scientific structure there.
"""
from __future__ import annotations

import csv
import json
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

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

QUARTET = [4, 6, 8, 14]
PSI_QUARTET = [6, 8, 14, 16]
#: TWO torsions, deliberately. With one, `cv_observations` and `cv_evaluations` are numerically
#: identical, so a counter that increments once per reporter call satisfies every cost assertion
#: below -- which is exactly how that misnomer survived. The lane costs nothing extra for the
#: second torsion and can no longer be passed by a call counter.
CV_YAML = f"""\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: {QUARTET}
  - name: psi
    type: torsion
    atom_indices: {PSI_QUARTET}
"""
N_CV = 2


def _require_cuda():
    """A device, or a failure. Never a substitution."""
    import openmm

    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail("no CUDA platform is available; a CPU run is not CUDA evidence")


def _environment(root: Path):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    return base


def _build(root: Path):
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")


def _generate(root: Path, name: str, document: dict, *extra):
    (root / f"{name}.config").write_text(yaml.safe_dump(document), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", f"./{name}", "--config", str(root / f"{name}.config"), *extra],
        cwd=root, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    return root / name


def _initial_state(root: Path):
    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "built.pdb"))
    system = XmlSerializer.deserialize((root / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    # On CUDA, like everything else in this file. A serialised State is platform-independent, so
    # the Reference platform would do -- but a file whose entire purpose is CUDA evidence should
    # not have to carry a platform-policy exemption naming another platform. The guard that
    # objected is the same one that keeps a CPU run from being cited as CUDA evidence, and it is
    # better satisfied than excused.
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    path = root / "initial_state.xml"
    path.write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return path


def _assert_cuda(record, where):
    platform = record.get("platform") or {}
    name = platform.get("name") if isinstance(platform, dict) else platform
    assert name == "CUDA", f"{where} ran on {name!r}, so it is not CUDA evidence"
    return platform


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- 1 and 2: cMD CV, and fixed-tau phase space plus CV, fresh and resumed ---------------------

@pytest.fixture(scope="module")
def cmd_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_cuda()
    root = tmp_path_factory.mktemp("cuda-cv-cmd")
    _build(root)
    return root


def _cmd_config(root: Path, *, tau=0.0, phase_space=0):
    return {
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 40},
        "reporting": {"solute_printout": 20, "system_printout": 20, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904, "tau": tau, "phase_space_printout": phase_space},
    }


def _run_cmd(root, scripts, destination, *, environment=None, expect=0):
    done = subprocess.run(
        [sys.executable, str(scripts / "md.py"),
         "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
         "-odir", str(destination)],
        cwd=scripts, capture_output=True, text=True, timeout=2400,
        env={**_environment(root), **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


EXPECTED = list(range(0, 41, 5))


def test_cmd_cv_runs_on_cuda_and_writes_the_declared_grid(cmd_project, tmp_path):
    from md_tools.build.record import read_record

    scripts = _generate(cmd_project, "cudaA", _cmd_config(cmd_project), "--all-in-one")
    destination = tmp_path / "fresh"
    _run_cmd(cmd_project, scripts, destination)

    _assert_cuda(read_record(destination / "cMD.log"), "cMD")
    series = sorted(destination.rglob("*.cv.csv"))
    assert len(series) == 1
    assert [int(r["step"]) for r in _rows(series[0])] == EXPECTED


def test_cmd_cv_resume_on_cuda_reproduces_the_grid_and_cost(cmd_project, tmp_path):
    """The absolute-step convention, on a device."""
    from md_tools.build.record import read_record
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    scripts = _generate(cmd_project, "cudaB", _cmd_config(cmd_project), "--all-in-one")
    destination = tmp_path / "resumed"
    crashed = _run_cmd(cmd_project, scripts, destination, expect=1,
                       environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                    FAULT_AFTER_ENVIRONMENT: "6"})
    assert crashed.returncode != 0
    _run_cmd(cmd_project, scripts, destination)

    record = read_record(destination / "cMD.log")
    _assert_cuda(record, "cMD resume")
    series = sorted(destination.rglob("*.cv.csv"))[0]
    assert [int(r["step"]) for r in _rows(series)] == EXPECTED
    cost = record["collective_variable_cost"]
    assert cost["cv_rows"] == len(EXPECTED)
    # EXACT, and in both scopes. `>= len(EXPECTED)` was satisfied by a counter that had lost a
    # segment as readily as by one that had carried it.
    assert cost["cumulative"]["cv_observations"] == len(EXPECTED)
    assert cost["cumulative"]["cv_evaluations"] == len(EXPECTED) * N_CV, (
        "an observation of a two-torsion definition is two scalar evaluations")
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"], (
        "this run was interrupted, so its final segment is a strict subset of the cumulative")
    assert cost["cumulative"]["wall_seconds"] >= cost["segment"]["wall_seconds"] >= 0.0


def test_fixed_tau_phase_space_and_cv_resume_on_cuda(cmd_project, tmp_path):
    """The stream a reservoir is built from, plus CV, both on the absolute grid."""
    from md_tools.build.record import read_record
    from md_tools.md.phase_space import PhaseSpaceReader
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    scripts = _generate(cmd_project, "cudaC",
                        _cmd_config(cmd_project, tau=0.5, phase_space=5), "--all-in-one")
    destination = tmp_path / "ps"
    _run_cmd(cmd_project, scripts, destination, expect=1,
             environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                          FAULT_AFTER_ENVIRONMENT: "6"})
    _run_cmd(cmd_project, scripts, destination)

    _assert_cuda(read_record(destination / "cMD.log"), "fixed-tau cMD")
    stream = sorted(destination.rglob("*.phase_space.nc"))
    assert stream, "no phase-space stream was written"
    with PhaseSpaceReader(stream[0]) as reader:
        steps = [int(s) for s in reader.steps()]
    assert steps == sorted(set(steps)) and max(steps) <= 40, steps
    series = sorted(destination.rglob("*.cv.csv"))[0]
    assert [int(r["step"]) for r in _rows(series)] == EXPECTED


# --- 3 and 4: REST2 and rREST2 on CUDA --------------------------------------------------------

def _ladder_config(root: Path, *, reservoir=False, states=3):
    document = {
        "protocol": "rREST2" if reservoir else "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": states, "exchange_interval_steps": 10,
                  "number_of_exchanges": 4},
        "reporting": {"solute_printout": 10, "system_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }
    if reservoir:
        document["reservoir"] = {"enabled": True, "path": "../reservoir.nc",
                                 "refresh_interval_exchanges": 1, "velocities": "inherit"}
    return document


def _run_ladder(root, scripts, destination, name, *extra, environment=None, expect=0):
    done = subprocess.run(
        [sys.executable, str(scripts / f"{name}.py"),
         "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
         "-c", str(root / "initial_state.xml"), "-odir", str(destination), *extra],
        cwd=scripts, capture_output=True, text=True, timeout=2400,
        env={**_environment(root), **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


@pytest.fixture(scope="module")
def ladder_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_cuda()
    root = tmp_path_factory.mktemp("cuda-cv-ladder")
    _build(root)
    _initial_state(root)
    return root


def test_rest2_cv_runs_on_cuda_fresh_and_resumed(ladder_project, tmp_path):
    scripts = _generate(ladder_project, "REST2", _ladder_config(ladder_project))

    fresh = tmp_path / "fresh"
    _run_ladder(ladder_project, scripts, fresh, "REST2")
    manifest = json.loads((fresh / "restart.json").read_text(encoding="utf-8"))
    assert (manifest.get("execution") or {}).get("platform") == "CUDA", manifest.get("execution")
    assert manifest["collective_variables"] is not None
    for index in range(3):
        assert [int(r["step"]) for r in _rows(fresh / f"remd{index}.cv.csv")] == EXPECTED

    resumed = tmp_path / "resumed"
    _run_ladder(ladder_project, scripts, resumed, "REST2", expect=1,
                environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                             "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                             "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    _run_ladder(ladder_project, scripts, resumed, "REST2", "--resume")
    for index in range(3):
        assert [int(r["step"]) for r in _rows(resumed / f"remd{index}.cv.csv")] == EXPECTED


def test_rrest2_cv_on_cuda_holds_the_pre_refresh_configuration(ladder_project, tmp_path):
    """The scientific regression, on a device: the row must not hold the reservoir sample."""
    from md_tools.md.phase_space import PhaseSpaceReader
    from md_tools.remd import storage

    # Reuse the CPU file's reservoir builder: one definition of what a distinguishable
    # reservoir is, so the two lanes cannot drift apart about it.
    from tests.test_rrest2_pre_refresh_cv import _reservoir
    from md_tools.cv import torsion_degrees

    _reservoir(ladder_project)
    scripts = _generate(ladder_project, "rREST2",
                        _ladder_config(ladder_project, reservoir=True))
    destination = tmp_path / "rrest2"
    _run_ladder(ladder_project, scripts, destination, "rREST2")

    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    assert (manifest.get("execution") or {}).get("platform") == "CUDA", manifest.get("execution")

    reporter = storage.ReplicaReporter(destination / "rREST2.nc", mode="r")
    try:
        events = reporter.reservoir_events()
    finally:
        reporter.close()
    accepted = [(i, int(r[0]), int(r[1])) for i, r in enumerate(events)
                if int(r[3]) == 1 and int(r[0]) >= 0]
    assert accepted, "no reservoir refresh was accepted on CUDA"

    with PhaseSpaceReader(ladder_project / "reservoir.nc") as reader:
        samples = {frame: torsion_degrees(reader.frame(frame)[0], QUARTET)
                   for _e, _s, frame in accepted}

    checked = 0
    for exchange_index, state_index, frame in accepted:
        step = (exchange_index + 1) * 10
        rows = {int(r["step"]): r for r in _rows(destination / f"remd{state_index}.cv.csv")}
        if step not in rows:
            continue
        reported = float(rows[step]["phi"])
        difference = abs((reported - samples[frame] + 180.0) % 360.0 - 180.0)
        assert difference > 1.0, (
            f"state {state_index} step {step} on CUDA reports {reported}, and the reservoir "
            f"sample has {samples[frame]}: the row holds the post-refresh coordinate")
        assert rows[step]["trajectory_frame_index"] == "", (
            "a refreshed state named a frame holding the reservoir sample")
        checked += 1
    assert checked, "no refreshed state had a CV row at its refresh step"


# --- 5: AIS on CUDA with the three-group decomposition ----------------------------------------

@pytest.fixture(scope="module")
def ais_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_cuda()
    root = tmp_path_factory.mktemp("cuda-cv-ais")
    _build(root)
    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    _generate(root, "AIS", {
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "observation_interval_steps": 10, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": 10, "system_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    })
    return root


def _run_ais(root, destination, *extra, environment=None, expect=0):
    done = subprocess.run(
        [sys.executable, str(root / "AIS" / "AIS.py"),
         "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
         "-source-traj", str(root / "source.dcd"), "-odir", str(destination), *extra],
        cwd=root / "AIS", capture_output=True, text=True, timeout=2400,
        env={**_environment(root), **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


GROUPS = ("non_scaled", "sqrt_scaled", "lin_scaled")


def test_ais_cv_and_decomposition_on_cuda_fresh_and_resumed(ais_project, tmp_path):
    """The lane the matrix wrongly attributed to a `--cpu` file."""
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    fresh = tmp_path / "fresh"
    _run_ais(ais_project, fresh)
    record = json.loads((fresh / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert record["platform"] == "CUDA", record["platform"]
    assert record["cv_rows"] == 20 // 5 + 1
    steps = [int(r["protocol_step"]) for r in _rows(fresh / "path_0000" / "cv.csv")]
    assert steps == list(range(0, 21, 5))

    observations = _rows(fresh / "path_0000" / "observations.csv")
    for row in observations:
        total = float(row["cumulative_work_kj_mol"])
        parts = sum(float(row[f"total_work_{group}_kj_mol"]) for group in GROUPS)
        assert abs(total - parts) <= max(1e-3, 1e-6 * abs(total)), row

    resumed = tmp_path / "resumed"
    _run_ais(ais_project, resumed, expect=1,
             environment={FAULT_ENVIRONMENT: "after-work-row", FAULT_AFTER_ENVIRONMENT: "1"})
    _run_ais(ais_project, resumed, "--resume")
    again = json.loads((resumed / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert again["platform"] == "CUDA"
    assert again["cv_rows"] == record["cv_rows"]
    assert [int(r["protocol_step"]) for r in _rows(resumed / "path_0000" / "cv.csv")] == steps
