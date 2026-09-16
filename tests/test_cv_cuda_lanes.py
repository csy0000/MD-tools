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
    done = subprocess.run(
        CLI + ["build-md", "-odir", f"./{name}-run1",
               "--config", str(system / f"{name}.config"), *extra],
        cwd=system, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    return system / f"{name}-run1"


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
                                    FAULT_AFTER_ENVIRONMENT: "6"})
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
                          FAULT_AFTER_ENVIRONMENT: "6"})
    _run_cmd(scripts, destination)

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
        "reporting": {"crd_printout_solute": 10, "crd_printout_whole": 10,
                      "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }
    if reservoir:
        document["reservoir"] = {"enabled": True, "path": "../reservoir.nc",
                                 "refresh_interval_exchanges": 1, "velocities": "inherit"}
    return document


def _run_ladder(scripts, destination, name, *extra, environment=None, expect=0):
    root = Path(scripts).parent
    done = subprocess.run(
        [sys.executable, str(scripts / f"{name}.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
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
    (root / "build").mkdir(exist_ok=True)
    _build(root)
    _initial_state(root)
    return root


def test_rest2_cv_runs_on_cuda_fresh_and_resumed(ladder_project, tmp_path):
    scripts = _generate(ladder_project, "REST2", _ladder_config(ladder_project))

    fresh = tmp_path / "fresh"
    _run_ladder(scripts, fresh, "REST2")
    manifest = json.loads((fresh / "restart.json").read_text(encoding="utf-8"))
    assert (manifest.get("execution") or {}).get("platform") == "CUDA", manifest.get("execution")
    assert manifest["collective_variables"] is not None
    for index in range(3):
        assert [int(r["step"]) for r in _rows(fresh / f"cv_state{index}.csv")] == EXPECTED

    resumed = tmp_path / "resumed"
    _run_ladder(scripts, resumed, "REST2", expect=1,
                environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                             "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                             "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    _run_ladder(scripts, resumed, "REST2", "--resume")
    for index in range(3):
        assert [int(r["step"]) for r in _rows(resumed / f"cv_state{index}.csv")] == EXPECTED


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
    _run_ladder(scripts, destination, "rREST2")

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
        rows = {int(r["step"]): r for r in _rows(destination / f"cv_state{state_index}.csv")}
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
    (root / "build").mkdir(exist_ok=True)
    _build(root)
    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    _generate(root, "AIS", {
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "work_measurement": "components",
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
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
         "-source-traj", str(root / "source.dcd"), "-odir", str(destination), *extra],
        cwd=root / "AIS-run1", capture_output=True, text=True, timeout=2400,
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

    # Both counts come from what the runs actually do, not from a guess. The first invocation
    # runs the whole chain, and its first three generations belong to the minimisation and
    # equilibration stages; production then commits every 10 steps, which is every second CV
    # row. 4 crashes at step 10 with 3 of the 9 rows committed, leaving three production commits
    # still to come -- room for a SECOND interruption. The resumed invocation skips every
    # finished stage, so its first commit is already inside production and 0 crashes there.
    #
    # Crashing first at 6 (step 30, the count the single-resume test above uses) leaves exactly
    # one commit remaining, so the resumed run necessarily wrote every remaining row before its
    # fault could fire: the series was already complete and the third invocation then correctly
    # refused to re-run a finished run. The assertion below is what caught that, and it stays,
    # so a future edit cannot quietly reduce this back to a single interruption.
    destination = tmp_path / "twice"
    committed = []
    for allowed in ("4", "0"):
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
    reference = tmp_path / f"{name}-reference"
    _run_ladder(scripts, reference, name)

    resumed = tmp_path / f"{name}-twice"
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


def test_rrest2_cv_survives_two_interruptions_on_cuda(ladder_project, tmp_path):
    """rREST2 twice resumed on a device: a refresh REPLACES a walker mid-run.

    The one ladder path where a continuation has to reproduce not just its own dynamics but the
    reservoir draws that displaced them, and it had no resume coverage on CUDA at all.
    """
    from tests.test_rrest2_pre_refresh_cv import _reservoir

    _reservoir(ladder_project)
    scripts = _generate(ladder_project, "rREST2twice",
                        _ladder_config(ladder_project, reservoir=True))
    reference, resumed = _ladder_twice(scripts, "rREST2", tmp_path)
    _assert_ladder_matches(reference, resumed)

    from md_tools.remd import storage

    def _draws(directory):
        reporter = storage.ReplicaReporter(directory / "rREST2.nc", mode="r")
        try:
            return [(int(r[0]), int(r[1])) for r in reporter.reservoir_events()
                    if int(r[3]) == 1 and int(r[0]) >= 0]
        finally:
            reporter.close()

    want, got = _draws(reference), _draws(resumed)
    assert want, "no reservoir refresh was accepted, so nothing about refresh was exercised"
    assert got == want, (
        "the resumed run drew different reservoir frames: the refresh stream was not restored")


def test_ais_cv_survives_two_interruptions_on_cuda(ais_project, tmp_path):
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "ais-reference"
    _run_ais(ais_project, reference)
    want = _rows(reference / "path_0000" / "cv.csv")

    # Both crashes land AFTER a committed generation, so each resume has a prefix to carry. A
    # crash before the path's first commit is a different case -- the path restarts from its
    # source frame, and its final segment then legitimately equals its cumulative, which would
    # make the carrying assertion below vacuous rather than failing loudly.
    resumed = tmp_path / "ais-twice"
    _run_ais(ais_project, resumed, expect=1,
             environment={FAULT_ENVIRONMENT: "after-work-row", FAULT_AFTER_ENVIRONMENT: "2"})
    _run_ais(ais_project, resumed, "--resume", expect=1,
             environment={FAULT_ENVIRONMENT: "after-work-row", FAULT_AFTER_ENVIRONMENT: "1"})
    _run_ais(ais_project, resumed, "--resume")

    record = json.loads(
        (resumed / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert record["platform"] == "CUDA", record["platform"]
    assert _rows(resumed / "path_0000" / "cv.csv") == want, (
        "after two interruptions on CUDA the path differs from the uninterrupted reference")

    cost = record["collective_variable_cost"]
    assert cost["cumulative"]["cv_evaluations"] == len(want) * N_CV
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"]
