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
import shutil
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
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")


def _generate(root: Path, name: str, document: dict, *extra):
    """Generate one run into a SYSTEM ROOT OF ITS OWN, and return the run directory.

    `build/`, `min/` and `input/` belong to the SYSTEM and are shared by every run on it -- and
    the sharing is enforced: `input/min.in` is refused if a second configuration resolves it
    differently. The variants here are deliberately DIFFERENT experiments -- `cudaC` runs at
    `tau = 0.5` with phase-space output, the ladders differ in protocol and in reservoir -- so
    generating them into one root made the second one refuse with

        build-md: input/min.in already exists and is not what this configuration resolves to.

    Each gets its own root, seeded from the module's built system. `build/` is copied rather than
    rebuilt: the physics is identical and `build-top` is the slow part.

    The RELATIVE inputs a configuration names -- `../reservoir.nc` for rREST2, `../source.dcd`
    for AIS -- resolve against the run directory, so they are copied in as well. `cv.yaml` is
    named by absolute path and needs no copy; it is carried anyway so a root is self-contained.
    """
    system = root / f"system-{name}"
    shutil.copytree(root / "build", system / "build")
    for helper in ("cv.yaml", "initial_state.xml", "reservoir.nc", "source.dcd"):
        source = root / helper
        if source.is_file():
            shutil.copy2(source, system / helper)

    tau = float((document.get("dynamics") or {}).get("tau") or 0.0)
    if tau > 0.0 and document.get("protocol") == "cMD":
        # A hot stage runs on its SAVED scaled state and scales nothing itself (step 3).
        from .conftest import make_scaled_state

        make_scaled_state(system, tau=tau)
    (system / f"{name}.config").write_text(yaml.safe_dump(document), encoding="utf-8")
    if document.get("protocol") == "REST2":
        # A ladder integrates SAVED scaled states (0.5.4); `build-md` refuses without them.
        from .conftest import make_states_for

        make_states_for(system, system / f"{name}.config")
    done = subprocess.run(
        CLI + ["build-md", "-odir", f"./{name}-run1",
               "--config", str(system / f"{name}.config"), *extra],
        cwd=system, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    run = system / f"{name}-run1"
    if document.get("protocol") == "REST2" and (system / "initial_state.xml").is_file():
        # WHERE EVERY GROUP LINE CONTINUES FROM, `-c eq/eq_3.xml` beside the group file: a ladder
        # reads its coordinates only from its group file, so the CUDA-made starting state goes
        # there instead of onto the command line.
        (run / "eq").mkdir(exist_ok=True)
        shutil.copy2(system / "initial_state.xml", run / "eq" / "eq_3.xml")
    return run


def _initial_state(root: Path):
    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
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
    (root / "build").mkdir(exist_ok=True)
    _build(root)
    return root


def _cmd_config(root: Path, *, tau=0.0, phase_space=0):
    return {
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 40},
        "reporting": {"crd_printout_solute": 20, "crd_printout_whole": 20,
                      "info_printout": 20, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904, "tau": tau, "phase_space_printout": phase_space},
    }


def _run_cmd(scripts, destination, *, environment=None, expect=0):
    # THE VARIANT'S OWN ROOT, which is the run directory's parent. Taking the module
    # fixture's root would read `build/` from a system this run was not generated
    # against, now that each variant has a root of its own.
    root = Path(scripts).parent
    # A hot variant's production runs on its saved state; an unscaled one on the built System.
    state = root / "build" / "cMD" / "system_state0.xml"
    system = state if state.is_file() else root / "build" / "built.xml"
    done = subprocess.run(
        # The production stage, named for the protocol: the retired `--all-in-one` md.py ran the
        # whole chain, and every equilibration length in `_cmd_config` is 0.
        [sys.executable, str(scripts / "cMD.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(system),
         "-odir", str(destination)],
        cwd=scripts, capture_output=True, text=True, timeout=2400,
        env={**_environment(root), **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


EXPECTED = list(range(0, 41, 5))

#: Checkpoint commits `_run_cmd`'s process lets through before the injected crash: the one at step
#: 10, so production dies committing step 20 (commits every 10 of 40 steps). RE-DERIVED for the split
#: layout: this was "6" when the retired `--all-in-one` md.py ran the minimisation and three
#: zero-length equilibration stages in the same process, each committing a final generation. With
#: production launched alone, 6 is past its last commit and the run finished instead of crashing.
COMMITS_BEFORE_THE_CRASH = "1"


def test_cmd_cv_runs_on_cuda_and_writes_the_declared_grid(cmd_project, tmp_path):
    from md_tools.build.record import read_record

    scripts = _generate(cmd_project, "cudaA", _cmd_config(cmd_project))
    destination = tmp_path / "fresh"
    _run_cmd(scripts, destination)

    _assert_cuda(read_record(destination / "cMD.log"), "cMD")
    series = sorted(destination.rglob("*.cv.csv"))
    assert len(series) == 1
    assert [int(r["step"]) for r in _rows(series[0])] == EXPECTED


def test_cmd_cv_resume_on_cuda_reproduces_the_grid_and_cost(cmd_project, tmp_path):
    """The absolute-step convention, on a device."""
    from md_tools.build.record import read_record
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    scripts = _generate(cmd_project, "cudaB", _cmd_config(cmd_project))
    destination = tmp_path / "resumed"
    crashed = _run_cmd(scripts, destination, expect=1,
                       environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                    FAULT_AFTER_ENVIRONMENT: COMMITS_BEFORE_THE_CRASH})
    assert crashed.returncode != 0
    _run_cmd(scripts, destination)

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
                        _cmd_config(cmd_project, tau=0.5, phase_space=5))
    destination = tmp_path / "ps"
    _run_cmd(scripts, destination, expect=1,
             environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                          FAULT_AFTER_ENVIRONMENT: COMMITS_BEFORE_THE_CRASH})
    _run_cmd(scripts, destination)

    _assert_cuda(read_record(destination / "cMD.log"), "fixed-tau cMD")
    stream = sorted(destination.rglob("*.phase_space.nc"))
    assert stream, "no phase-space stream was written"
    with PhaseSpaceReader(stream[0]) as reader:
        steps = [int(s) for s in reader.steps()]
    assert steps == sorted(set(steps)) and max(steps) <= 40, steps
    series = sorted(destination.rglob("*.cv.csv"))[0]
    assert [int(r["step"]) for r in _rows(series)] == EXPECTED


# --- 3: REST2 on CUDA --------------------------------------------------------

def _ladder_config(root: Path, *, states=3):
    document = {
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": states, "exchange_interval_steps": 10,
                  "number_of_exchanges": 4},
        "reporting": {"crd_printout_solute": 10, "crd_printout_whole": 10,
                      "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }
    return document


def _ladder_directory(scripts, tmp_path, label):
    """A fresh copy of the generated run directory for one ladder, which runs INTO itself.

    The group file names `-i _protocol.py` relative to itself and the runtime writes that helper
    into `-odir`, so the two must be one directory (any other `-odir` is refused) -- `run.sh`
    launches with `-odir .` in the run directory. A copy per case, as a sibling of the generated
    one so the group file's `../build/` paths still resolve.
    """
    destination = Path(scripts).parent / f"{tmp_path.name}-{label}"
    shutil.copytree(scripts, destination)
    return destination


def _run_ladder(scripts, destination, name, *extra, environment=None, expect=0):
    """`destination` is a run directory from `_ladder_directory`; the ladder is launched in it.

    NO -s and no -c: a ladder reads both only from its group file (0.5.4), whose lines name the
    saved scaled states under `build/REST2/` and the starting state in `eq/eq_3.xml`.
    """
    root = Path(scripts).parent
    done = subprocess.run(
        [sys.executable, str(destination / f"{name}.py"),
         "-p", str(root / "build" / "built.pdb"), "--groupfile", "remd_groupfile.1",
         "-odir", ".", *extra],
        cwd=destination, capture_output=True, text=True, timeout=2400,
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
    (root / "build").mkdir(exist_ok=True)
    _build(root)
    _initial_state(root)
    return root


def test_rest2_cv_runs_on_cuda_fresh_and_resumed(ladder_project, tmp_path):
    scripts = _generate(ladder_project, "REST2", _ladder_config(ladder_project))

    fresh = _ladder_directory(scripts, tmp_path, "fresh")
    _run_ladder(scripts, fresh, "REST2")
    manifest = json.loads((fresh / "restart.json").read_text(encoding="utf-8"))
    assert (manifest.get("execution") or {}).get("platform") == "CUDA", manifest.get("execution")
    assert manifest["collective_variables"] is not None
    for index in range(3):
        assert [int(r["step"]) for r in _rows(fresh / f"cv_state{index}.csv")] == EXPECTED

    resumed = _ladder_directory(scripts, tmp_path, "resumed")
    _run_ladder(scripts, resumed, "REST2", expect=1,
                environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                             "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                             "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    _run_ladder(scripts, resumed, "REST2", "--resume")
    for index in range(3):
        assert [int(r["step"]) for r in _rows(resumed / f"cv_state{index}.csv")] == EXPECTED


# --- 5: AIS on CUDA, two end states --------------------------------------------------------------

def write_scaled_v0(root: Path, tau: float = 0.5) -> Path:
    """`build/V0.xml`: the REST2 state at `tau` of `build/built.xml`. V1 is `built.xml` itself.

    The pair the retired single-topology AIS switched along tau, as the two files the two-state
    AIS transforms between -- a parameter-only edit of the same particles.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2.hamiltonian import build_scaled_system

    base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    solute = solute_atom_indices(PDBFile(str(root / "build" / "built.pdb")).topology)
    target = root / "build" / "V0.xml"
    target.write_text(XmlSerializer.serialize(build_scaled_system(base, solute, float(tau))),
                      encoding="utf-8")
    return target


def committed_instant(directory: Path) -> dict | None:
    """What an interrupted AIS path's committed generation vouches for, read BEFORE a resume.

    The state record, and the exact prefix of every appendable stream it counts: observation
    rows, CV rows, state rows and staged frames. None when nothing was committed.
    """
    from md_tools.openmm.checkpoint import read_committed

    committed = read_committed(directory) if directory.is_dir() else None
    if committed is None:
        return None
    state = committed["state"]

    def prefix(name, count):
        path = directory / name
        return _rows(path)[:int(count)] if path.is_file() else []

    frames = None
    staged = directory / "frames.partial.nc"
    if staged.is_file() and int(state["frames"]) > 0:
        import mdtraj

        with mdtraj.formats.NetCDFTrajectoryFile(str(staged)) as handle:
            frames = handle.read()[0][:int(state["frames"])].copy()
    return {"state": state, "observations": prefix("observations.csv", state["work_rows"]),
            "cv": prefix("cv.csv", state["cv_rows"]),
            "system": prefix("system.csv", state["state_rows"]), "frames": frames}


def assert_restored_and_complete(run: Path, path_index: int, instant, *, switching_steps: int):
    """A path finished on CUDA is COMPLETE AND VALID, and continued EXACTLY from `instant`.

    Not row-for-row equality with an uninterrupted CUDA run past the resume point: the two-state
    mixing force's inner Contexts keep atom-ordering state no checkpoint captures, so on CUDA a
    resume continues the same switching process from the same committed instant as a new
    realisation (decided 2026-09-16). Everything the committed generation vouches for is
    asserted exactly: the stream prefixes byte for byte, lambda at the committed update, the
    cumulative work carried across the boundary, and counters that only grow from it.
    """
    import numpy

    directory = run / f"path_{path_index:04d}"
    completion = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    assert completion["status"] == "completed"
    assert completion["platform"] == "CUDA", completion["platform"]

    # -- complete and valid, on its own terms -----------------------------------------------------
    observations = _rows(directory / "observations.csv")
    assert len(observations) == int(completion["observations"])
    steps = [int(row["protocol_step"]) for row in observations]
    assert steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == switching_steps, steps
    for row in observations:
        assert abs(float(row["lambda"]) - int(row["protocol_step"]) / switching_steps) < 1e-12, row
    assert float(observations[0]["cumulative_work_kj_mol"]) == 0.0
    running = 0.0
    for row in observations[1:]:
        running += float(row["incremental_work_kj_mol"])
        total = float(row["cumulative_work_kj_mol"])
        assert abs(running - total) <= 1e-9 * max(1.0, abs(total)), (path_index, row)
    assert float(completion["total_work_kj_mol"]) == float(
        observations[-1]["cumulative_work_kj_mol"])
    cv_path = directory / "cv.csv"
    if cv_path.is_file():
        cv_steps = [int(row["protocol_step"]) for row in _rows(cv_path)]
        assert len(cv_steps) == len(set(cv_steps)) == int(completion["cv_rows"]), cv_steps

    if instant is None:
        return completion

    # -- restored exactly ----------------------------------------------------------------------------
    state = instant["state"]
    assert completion["resumed"] is True
    for name, prefix in (("observations.csv", instant["observations"]),
                         ("cv.csv", instant["cv"]), ("system.csv", instant["system"])):
        if prefix:
            assert _rows(directory / name)[:len(prefix)] == prefix, (
                f"path {path_index}: a committed row of {name} changed across the resume")
    assert abs(float(state["lambda"]) - int(state["protocol_step"]) / switching_steps) < 1e-12
    carried = (float(state["cumulative_work_kj_mol"])
               - float(state["work_since_last_observation_kj_mol"]))
    committed_rows = instant["observations"]
    if committed_rows:
        last = float(committed_rows[-1]["cumulative_work_kj_mol"])
        assert abs(carried - last) <= 1e-9 * max(1.0, abs(last)), (state, committed_rows[-1])
        if len(observations) > len(committed_rows):
            following = observations[len(committed_rows)]
            restart = (float(following["cumulative_work_kj_mol"])
                       - float(following["incremental_work_kj_mol"]))
            assert abs(restart - last) <= 1e-9 * max(1.0, abs(last)), (
                f"path {path_index}: the work after the resume does not start from the committed "
                f"cumulative {last}")
    if instant["frames"] is not None:
        import mdtraj

        with mdtraj.formats.NetCDFTrajectoryFile(
                str(run / completion["trajectory"])) as handle:
            published = handle.read()[0]
        assert numpy.array_equal(published[:len(instant["frames"])], instant["frames"]), (
            f"path {path_index}: a committed frame changed across the resume")
    before, after = state["evaluation_counters"], completion["evaluation_counters"]
    for name in ("work_derivative_evaluations", "observation_potential_energy_evaluations",
                 "other_useful_energy_evaluations", "parameter_updates"):
        assert int(after[name]) >= int(before[name]), (path_index, name, before, after)
    return completion


@pytest.fixture(scope="module")
def ais_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_cuda()
    root = tmp_path_factory.mktemp("cuda-cv-ais")
    (root / "build").mkdir(exist_ok=True)
    _build(root)
    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    write_scaled_v0(root)
    _generate(root, "AIS", {
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "observation_interval_steps": 10, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "crd_printout_whole": 10,
                      "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    })
    # THE VARIANT'S ROOT, not the module's. `_generate` gives each variant a system root of its
    # own -- `build/`, `min/` and `input/` are shared per SYSTEM and refuse a second, different
    # configuration -- so `AIS-run1`, `build/` and `source.dcd` all live under `system-AIS/`, and
    # `_run_ais` resolves every one of them from what this returns.
    return root / "system-AIS"


def _run_ais(root, destination, *extra, environment=None, expect=0):
    done = subprocess.run(
        [sys.executable, str(root / "AIS-run1" / "AIS.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "V0.xml"),
         "-p2", str(root / "build" / "built.pdb"), "-s2", str(root / "build" / "built.xml"),
         "-source-traj", str(root / "source.dcd"), "-odir", str(destination), *extra],
        cwd=root / "AIS-run1", capture_output=True, text=True, timeout=2400,
        env={**_environment(root), **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


def test_ais_cv_and_two_state_on_cuda_fresh_and_resumed(ais_project, tmp_path):
    """The lane the matrix wrongly attributed to a `--cpu` file -- with the two-state Hamiltonian.

    Fresh: CV rows on their grid, and on every frame-aligned observation the mixture
    `(1 - lambda) V0 + lambda V1` equal to the direct potential the CUDA Context evaluated, at the
    precision the run used. Resumed: exact restoration of the committed generation, and a
    complete valid path -- not row-for-row equality with the fresh run, which CUDA does not give.
    """
    from md_tools.ais.two_state import identity_tolerance
    from md_tools.build.record import read_record
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    fresh = tmp_path / "fresh"
    _run_ais(ais_project, fresh)
    record = assert_restored_and_complete(fresh, 0, None, switching_steps=20)
    assert record["cv_rows"] == 20 // 5 + 1
    steps = [int(r["protocol_step"]) for r in _rows(fresh / "path_0000" / "cv.csv")]
    assert steps == list(range(0, 21, 5))

    precision = read_record(fresh / "AIS.log")["acceleration"].get("cuda_precision") or "mixed"
    aligned = 0
    for row in _rows(fresh / "path_0000" / "observations.csv"):
        if row["coordinate_frame_index"] == "":
            assert row["potential_direct_kj_mol"] == "", row
            continue
        lam = float(row["lambda"])
        v0, v1 = float(row["potential_v0_kj_mol"]), float(row["potential_v1_kj_mol"])
        direct = float(row["potential_direct_kj_mol"])
        allowed = identity_tolerance(max(abs(v0), abs(v1), abs(direct)), precision=precision)
        assert abs((1.0 - lam) * v0 + lam * v1 - direct) <= allowed, row
        assert v0 != v1, "V0 and V1 agree at a saved coordinate: the pair switches nothing"
        aligned += 1
    assert aligned >= 2, f"only {aligned} frame-aligned observation(s)"

    resumed = tmp_path / "resumed"
    _run_ais(ais_project, resumed, expect=1,
             environment={FAULT_ENVIRONMENT: "after-work-row", FAULT_AFTER_ENVIRONMENT: "1"})
    instant = committed_instant(resumed / "path_0000")
    _run_ais(ais_project, resumed, "--resume")
    again = assert_restored_and_complete(resumed, 0, instant, switching_steps=20)
    assert again["cv_rows"] == record["cv_rows"]
    assert [int(r["protocol_step"]) for r in _rows(resumed / "path_0000" / "cv.csv")] == steps


# --- 5: MULTIPLY resumed, all four protocols, on real CUDA -------------------------------------
#
# The tests above interrupt each protocol ONCE. One resume passes whether the accumulation is
# "restored prefix + this segment" (correct), "just the prefix", or "just this segment" -- with a
# single prior segment the three answers can coincide. It also cannot show a continuation that
# rebuilds its state correctly the first time and loses it the second. Two consecutive
# interruptions separate them, and the requirement is for that depth on a DEVICE, not only under
# `--cpu`, because the restore path being exercised here restores an OpenMM context checkpoint
# whose contents are platform-specific.

def test_cmd_cv_survives_two_interruptions_on_cuda(cmd_project, tmp_path):
    from md_tools.build.record import read_record
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    scripts = _generate(cmd_project, "cudaD", _cmd_config(cmd_project))

    reference = tmp_path / "reference"
    _run_cmd(scripts, reference)
    want = (sorted(reference.rglob("*.cv.csv"))[0]).read_text(encoding="utf-8")

    # Both counts come from what the runs actually do, not from a guess. `_run_cmd` launches the
    # PRODUCTION stage alone, which commits every 10 steps -- every second CV row. 0 crashes on its
    # first commit, step 10, with 3 of the 9 rows committed, leaving three production commits still
    # to come -- room for a SECOND interruption. The resumed invocation's first commit is step 20,
    # and 0 crashes there too, with 5 rows.
    #
    # RE-DERIVED for the split layout. This was ("4", "0") while the retired `--all-in-one` md.py
    # ran the minimisation and equilibration stages in the same process, whose three generations
    # the first count had to let through; launched alone, 4 was past step 30 and the two crashes
    # left [9, 9] -- which the assertion below caught.
    #
    # Crashing first at step 30 leaves exactly one commit remaining, so the resumed run necessarily
    # wrote every remaining row before its fault could fire: the series was already complete and
    # the third invocation then correctly refused to re-run a finished run. The assertion below is
    # what caught that, and it stays, so a future edit cannot quietly reduce this back to a single
    # interruption.
    destination = tmp_path / "twice"
    committed = []
    for allowed in ("0", "0"):
        crashed = _run_cmd(scripts, destination, expect=1,
                           environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                        FAULT_AFTER_ENVIRONMENT: allowed})
        assert crashed.returncode != 0
        committed.append(len(_rows(sorted(destination.rglob("*.cv.csv"))[0])))
    assert committed[0] < committed[1] < len(EXPECTED), (
        f"the two interruptions left {committed} rows of {len(EXPECTED)}: they must land at "
        f"different, incomplete points or nothing about repeated carrying is exercised")
    _run_cmd(scripts, destination)

    record = read_record(destination / "cMD.log")
    _assert_cuda(record, "cMD twice resumed")
    got = (sorted(destination.rglob("*.cv.csv"))[0]).read_text(encoding="utf-8")
    assert got == want, (
        "after two interruptions on CUDA the series differs from the uninterrupted reference")

    cost = record["collective_variable_cost"]
    assert cost["cumulative"]["cv_observations"] == len(EXPECTED)
    assert cost["cumulative"]["cv_evaluations"] == len(EXPECTED) * N_CV
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"], (
        "the final segment equals the cumulative, so earlier segments were not carried")


def _ladder_twice(scripts, name, tmp_path):
    """Fresh reference, then two interruptions and completion. Returns (reference, resumed)."""
    reference = _ladder_directory(scripts, tmp_path, f"{name}-reference")
    _run_ladder(scripts, reference, name)

    resumed = _ladder_directory(scripts, tmp_path, f"{name}-twice")
    _run_ladder(scripts, resumed, name, expect=1,
                environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                             "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                             "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    _run_ladder(scripts, resumed, name, "--resume", expect=1,
                environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                             "MD_TOOLS_FAIL_LADDER_AT": "after-checkpoint",
                             "MD_TOOLS_FAIL_PROPAGATION_AFTER": "1"})
    _run_ladder(scripts, resumed, name, "--resume")
    return reference, resumed


def _assert_ladder_matches(reference: Path, resumed: Path, states=3):
    for index in range(states):
        want = _rows(reference / f"cv_state{index}.csv")
        got = _rows(resumed / f"cv_state{index}.csv")
        assert [int(r["step"]) for r in got] == EXPECTED
        assert len({r["step"] for r in got}) == len(got), "a step was written twice"
        for column in ("phi", "psi", "walker_index", "trajectory_frame_index"):
            assert [r[column] for r in got] == [r[column] for r in want], (
                f"state {index} {column} diverges from the uninterrupted reference after two "
                f"interruptions on CUDA")


def test_rest2_cv_survives_two_interruptions_on_cuda(ladder_project, tmp_path):
    scripts = _generate(ladder_project, "REST2twice", _ladder_config(ladder_project))
    reference, resumed = _ladder_twice(scripts, "REST2", tmp_path)
    _assert_ladder_matches(reference, resumed)

    cost = json.loads(
        (resumed / "restart.json").read_text(encoding="utf-8"))["collective_variables"]["cost"]
    assert cost["cumulative"]["cv_evaluations"] == 3 * len(EXPECTED) * N_CV
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"]


def test_ais_cv_survives_two_interruptions_on_cuda(ais_project, tmp_path):
    """Two interruptions, each resumed from EXACTLY its committed generation, on a device.

    Past each resume point a CUDA continuation is a new realisation of the same switching process
    (the two-state mixing force's inner Contexts are not checkpointed), so the path is not
    compared with an uninterrupted reference row for row. Each committed instant is captured
    before the resume that continues it, and every stream prefix it vouches for must survive
    both later invocations unchanged.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "ais-reference"
    _run_ais(ais_project, reference)
    want = _rows(reference / "path_0000" / "cv.csv")

    # Both crashes land AFTER a committed generation, so each resume has a prefix to carry. A
    # crash before the path's first commit is a different case -- the path restarts from its
    # source frame, and its final segment then legitimately equals its cumulative, which would
    # make the carrying assertion below vacuous rather than failing loudly.
    #
    # The counts are chosen for that. The first crash lands after path 0's step-10 row, past its
    # step-10 commit. The resuming invocation then writes path 0's last row (1), finishes it, and
    # writes path 1's rows at steps 0 and 10 (2, 3) -- so the second crash lands in PATH 1, past
    # its own step-10 commit. The count used to be 1, which crashed path 1 at step 0 with nothing
    # committed, and the "second carried prefix" was a fresh restart that the old row-for-row
    # comparison could not tell apart.
    resumed = tmp_path / "ais-twice"
    paths = 2
    instants = []
    for attempt, allowed in enumerate(("2", "3")):
        _run_ais(ais_project, resumed, *(("--resume",) if attempt else ()), expect=1,
                 environment={FAULT_ENVIRONMENT: "after-work-row",
                              FAULT_AFTER_ENVIRONMENT: allowed})
        # Captured for EVERY path: the second interruption may land in a later path than the
        # first, once the first path has finished inside the resuming invocation.
        instants += [(index, committed_instant(resumed / f"path_{index:04d}"))
                     for index in range(paths)
                     if not (resumed / f"path_{index:04d}" / "completed.json").is_file()]
    carried = [(index, instant) for index, instant in instants if instant is not None]
    assert len(carried) >= 2, (
        f"only {len(carried)} committed generation(s) were interrupted, so a second carried "
        f"prefix was never exercised")
    _run_ais(ais_project, resumed, "--resume")

    # Every committed instant, not only the last: a prefix carried by the first resume must also
    # survive the second.
    for index, instant in carried:
        assert_restored_and_complete(resumed, index, instant, switching_steps=20)
    for index in range(paths):
        assert_restored_and_complete(resumed, index, None, switching_steps=20)
    record = json.loads(
        (resumed / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    got = _rows(resumed / "path_0000" / "cv.csv")
    assert [r["protocol_step"] for r in got] == [r["protocol_step"] for r in want], (
        "after two interruptions on CUDA the CV grid differs from the uninterrupted reference")

    cost = record["collective_variable_cost"]
    assert cost["cumulative"]["cv_evaluations"] == len(want) * N_CV
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"]
