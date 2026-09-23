"""Selective REST2 end to end (0.6.1 S1-F): a 0.6.0 ladder continues, a selective one is refused.

The evidence S0 asked for before landing the integration patches (6b09108, 6e855ed, 75d6ea1,
8b3b634), end to end rather than on a helper:

  (a) a ladder RUN BY 0.6.0 CODE -- `e524e0e`, exported from this repository's history, not a
      dictionary built here -- is extended in place under this checkout in legacy mode, through
      `ReplicaRun.compare_identity`. Its restart.json records a v2 identity, and the test asserts
      that too, so the continuation is known to cross the format change;
  (b) the same 0.6.0 run is REFUSED BY NAME when its states are rebuilt with a selective region;
  (c) a selective ladder run by this checkout exports a REST2 bundle whose `verify_rungs.py`
      passes, and fails once the recorded region is altered; the same for a selective fixed-tau
      state and `verify_state.py`.

Every ladder here is NVT on ACE-ALA-NME in a periodic box with no water, which an NVT ladder
integrates happily and a barostat would collapse. `--cpu`, CUDA hidden.

PLATFORM_POLICY_EXEMPTION: the ladders run under `--cpu` only to produce real 0.6.0 and 0.6.1 run
records to continue and export; what is tested is identity and reconstruction, not dynamics.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = ["-m", "md_tools.cli.md_openmm"]
#: The 0.6.0 source the legacy record is produced with: `dev-0.6.0` carrying the 20260918
#: instruction, whose `src/md_tools` is the 0.6.0 package.
V060 = "e524e0e"

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("selective-rest2-integration")]

STATES, EXCHANGE_EVERY, EXCHANGES = 3, 10, 2


def _v060_source(tmp_path_factory) -> Path:
    found = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{V060}^{{commit}}"],
                           capture_output=True)
    if found.returncode != 0:
        pytest.skip(f"commit {V060} (0.6.0) is not in this checkout's history, so no 0.6.0 run "
                    f"record can be produced; this evidence needs a clone with history")
    archive = subprocess.run(["git", "-C", str(REPO), "archive", V060, "src"],
                             capture_output=True, check=True).stdout
    root = tmp_path_factory.mktemp("v060")
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(root, filter="data")
    return root / "src"


def _env(root: Path, src: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OPENMM_CPU_THREADS"] = "2"
    user = root / "user.config"
    if not user.is_file():
        user.write_text(yaml.safe_dump({"schema_version": "1.0",
                                        "user": {"person_id": "t", "name": "T"}}),
                        encoding="utf-8")
    env["MD_TOOLS_CONFIG"] = str(user)
    return env


def _run(args, *, cwd, env, ok=True, timeout=1800):
    done = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout)
    if ok:
        assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _scaler(root: Path, env, text: str, *, overwrite=False):
    build = root / "build"
    (build / "scaler.config").write_text(text, encoding="utf-8")
    _run(CLI + ["build-top", "--rest2-scaler", "-s", "build/built.xml", "-p", "build/built.pdb",
                "--config", "build/scaler.config"] + (["--overwrite"] if overwrite else []),
         cwd=root, env=env)


def _project(root: Path, env):
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "explicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "dynamics": {"seed": 20260919},
    }), encoding="utf-8")
    _run(CLI + ["build-md", "-odir", "./REST2-run1", "--config", str(root / "REST2.config")],
         cwd=root, env=env)


def _initial_state(root: Path):
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
        context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)),
        encoding="utf-8")


def _ladder(root: Path, env, run: Path, *extra, ok=True):
    from .conftest import ladder_group_file

    return _run([str(root / "REST2-run1" / "REST2.py"), "-p", str(root / "build" / "built.pdb"),
                 "--groupfile", str(ladder_group_file(root, run)), "-odir", str(run), "--cpu",
                 *extra], cwd=root / "REST2-run1", env=env, ok=ok)


def _dataset(root: Path):
    from .conftest import make_dataset_root

    make_dataset_root(root, solvent="explicit")


LEGACY_SCALER = f"method: REST2\nschedule:\n  n_states: {STATES}\n  tau_max: 0.5\n"
SELECTIVE_SCALER = LEGACY_SCALER + 'backbone_scaling_list: ":2"\n'


@pytest.fixture(scope="module")
def v060_ladder(tmp_path_factory):
    """A complete ladder written entirely by 0.6.0 code: states, project and run."""
    src = _v060_source(tmp_path_factory)
    root = tmp_path_factory.mktemp("legacy")
    env = _env(root, src)
    _dataset(root)
    _scaler(root, env, LEGACY_SCALER)
    _project(root, env)
    _initial_state(root)
    run = root / "run"
    _ladder(root, env, run)
    return root, run


def _copy(v060_ladder, tmp_path) -> tuple[Path, Path]:
    root, _run_dir = v060_ladder
    copy = tmp_path / "copy"
    shutil.copytree(root, copy, symlinks=True)
    return copy, copy / "run"


def _restart_identity(run: Path) -> dict:
    return json.loads((run / "restart.json").read_text(encoding="utf-8"))["scientific_identity"]


def test_the_record_really_is_0_6_0s(v060_ladder):
    _root, run = v060_ladder
    assert _restart_identity(run)["hamiltonian"]["format"] == "md-tools-hamiltonian-identity/v2"


def test_a_0_6_0_ladder_extends_in_place_under_0_6_1_in_legacy_mode(v060_ladder, tmp_path):
    root, run = _copy(v060_ladder, tmp_path)
    env = _env(root, REPO / "src")
    done = _ladder(root, env, run, "--extend", "1")
    assert done.returncode == 0
    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["resumed_from_step"] == EXCHANGES * EXCHANGE_EVERY, "it CONTINUED 0.6.0's run"
    assert manifest["steps_completed"] == (EXCHANGES + 1) * EXCHANGE_EVERY
    assert manifest["exchanges_committed"] == EXCHANGES + 1


def _selective_copy(v060_ladder, tmp_path):
    root, run = _copy(v060_ladder, tmp_path)
    env = _env(root, REPO / "src")
    _scaler(root, env, SELECTIVE_SCALER, overwrite=True)
    record = yaml.safe_load((root / "build" / "REST2" / "scaler.yaml").read_text(encoding="utf-8"))
    assert record["selection"]["selection_mode"] == "explicit"
    return root, run, env


def _snapshot(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def test_an_in_place_continuation_onto_selective_states_is_refused_read_only(v060_ladder,
                                                                             tmp_path):
    """In place, the FIRST guard to fire is the content-addressed `solute.yaml`: it records every
    state's sha256, so states rebuilt under a new selection can never be continued in place."""
    root, run, env = _selective_copy(v060_ladder, tmp_path)
    before = _snapshot(run)
    done = _ladder(root, env, run, "--extend", "1", ok=False)
    assert done.returncode != 0
    assert "solute.yaml already exists and was generated from different content" in done.stderr
    assert _snapshot(run) == before, "a refused continuation writes nothing"


def test_an_extension_of_the_0_6_0_run_onto_selective_states_is_refused_before_anything_exists(
        v060_ladder, tmp_path):
    """Out of place, the states the group file names are not the ones the parent's solute.yaml
    records. That is refused READ-ONLY (83c4a98, released in 0.6.0): on stderr, naming the
    states, before the extension's -odir holds anything, and with the parent untouched."""
    root, parent, env = _selective_copy(v060_ladder, tmp_path)
    before = _snapshot(parent)
    extension = root / "extended"
    done = _ladder(root, env, extension, "--extend", "1", "--extend-from", str(parent), ok=False)
    assert done.returncode != 0
    assert "names other saved states than" in done.stderr, done.stderr[-4000:]
    # State 0 is tau = 0, the unscaled System, identical under ANY selection; only the hot
    # states differ, and the refusal names exactly those.
    assert "state 0: parent " not in done.stderr
    for state in range(1, STATES):
        assert f"state {state}: parent " in done.stderr, done.stderr[-4000:]
    assert "Nothing was written" in done.stderr
    assert not extension.exists() or not any(extension.iterdir()), (
        f"the refused extension left {sorted(p.name for p in extension.iterdir())}")
    assert _snapshot(parent) == before


def test_the_same_extension_onto_legacy_states_is_accepted(v060_ladder, tmp_path):
    """The control for the test above: out of place, legacy, the v2 parent extends."""
    root, parent = _copy(v060_ladder, tmp_path)
    env = _env(root, REPO / "src")
    extension = root / "extended"
    _ladder(root, env, extension, "--extend", "1", "--extend-from", str(parent))
    manifest = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"


# --- (c) exports -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def selective_ladder(tmp_path_factory):
    """A selective ladder run the way `run.sh` runs one -- a stage into `eq/`, then `md-run` under
    mpirun, one rank per state -- so `export-reference` finds what it looks for."""
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH; the exported ladder needs one rank per state")
    root = tmp_path_factory.mktemp("selective")
    env = _env(root, REPO / "src")
    # A REAL build: export-reference proves every input against the build-top record, and a
    # stand-in System has none. Explicit TIP3P around ACE-ALA-NME.
    (root / "sys.config").write_text("solvent:\n  padding_nm: 0.8\n", encoding="utf-8")
    _run(CLI + ["build-top", "-i", str(REPO / "tests" / "data" / "ALA.pdb"),
                "-os", "build/built.xml", "-op", "build/built.pdb", "-log", "build/built.log",
                "--config", str(root / "sys.config")], cwd=root, env=env)
    _scaler(root, env, SELECTIVE_SCALER)
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "explicit",
        "stages": {"minimization_iterations": 0, "restrained_nvt_steps": 20,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "dynamics": {"seed": 20260919},
    }), encoding="utf-8")
    _run(CLI + ["build-md", "-odir", "./run-run1", "--config", str(root / "REST2.config")],
         cwd=root, env=env)
    run = root / "run-run1"
    _run(CLI + ["md-run", "-i", "../input/eq_1.in", "-p", "../build/built.pdb",
                "-s", "../build/built.xml", "-r", "eq/eq_3.xml", "-chk", "eq/eq.chk",
                "-o", "eq/eq.out", "-log", "eq/eq.log", "-odir", "eq", "--cpu"],
         cwd=run, env=env)
    done = subprocess.run(
        ["mpirun", "-n", str(STATES), sys.executable, *CLI, "md-run", "-ng", str(STATES),
         "-i", "../input/REST2.in", "-p", "../build/built.pdb", "--groupfile",
         "remd_groupfile.1", "-x", "REST2.nc", "-r", "restart.json", "-o", "REST2.out",
         "-log", "REST2.log", "--cpu"],
        cwd=run, env=env, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root, run, env


def _bundle(selective_ladder, tmp_path) -> Path:
    root, run, env = selective_ladder
    bundle = tmp_path / "bundle"
    _run(CLI + ["export-reference", "-idata", str(run), "-odir", str(bundle), "--stage", "REST2"],
         cwd=root, env=env)
    return bundle


def test_a_selective_ladder_records_its_region_in_the_identity(selective_ladder):
    _root, run, _env_ = selective_ladder
    hamiltonian = _restart_identity(run)["hamiltonian"]
    assert hamiltonian["format"] == "md-tools-hamiltonian-identity/v3"
    assert hamiltonian["selection_mode"] == "explicit"


def test_a_selective_rest2_bundle_verifies_and_an_altered_region_does_not(selective_ladder,
                                                                          tmp_path):
    bundle = _bundle(selective_ladder, tmp_path)
    derivation = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))["derivation"]
    assert derivation["selection_mode"] == "explicit"
    assert derivation["torsion_central_bonds"] is not None
    good = _run(["verify_rungs.py"], cwd=bundle, env=selective_ladder[2])
    assert "rebuild identically" in good.stdout

    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    provenance["derivation"]["scaled_atom_indices"] = \
        provenance["derivation"]["scaled_atom_indices"][:-1]
    (bundle / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    bad = _run(["verify_rungs.py"], cwd=bundle, env=selective_ladder[2], ok=False)
    assert bad.returncode == 1 and "DIFFERS" in bad.stdout


@pytest.fixture(scope="module")
def selective_stage(tmp_path_factory):
    """A selective FIXED-TAU cMD run on its saved state, exported by `export-reference`."""
    root = tmp_path_factory.mktemp("selective-stage")
    env = _env(root, REPO / "src")
    (root / "sys.config").write_text("solvent:\n  padding_nm: 0.8\n", encoding="utf-8")
    _run(CLI + ["build-top", "-i", str(REPO / "tests" / "data" / "ALA.pdb"),
                "-os", "build/built.xml", "-op", "build/built.pdb", "-log", "build/built.log",
                "--config", str(root / "sys.config")], cwd=root, env=env)
    _scaler(root, env, "method: cMD\nschedule:\n  n_states: 1\n  tau_min: 0.5\n  tau_max: 0.5\n"
                       'backbone_scaling_list: ":2"\nsidechain_scaling_list: ":2"\n')
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 7, "tau": 0.5},
        "stages": {"minimization_iterations": 0, "restrained_nvt_steps": 20,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 20},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                      "checkpoint_printout": 20},
    }), encoding="utf-8")
    _run(CLI + ["build-md", "-odir", "./run-run1", "--config", str(root / "cMD.config")],
         cwd=root, env=env)
    run = root / "run-run1"
    for source, odir, parent in (("../input/eq_1.in", "eq", None),
                                 ("../input/cMD.in", ".", "eq/eq_1.xml")):
        argv = CLI + ["md-run", "-i", source, "-p", "../build/built.pdb",
                      "-s", "../build/cMD/system_state0.xml", "-odir", odir, "--cpu"]
        _run(argv + (["-c", parent] if parent else []), cwd=run, env=env)
    bundle = root / "bundle"
    _run(CLI + ["export-reference", "-idata", str(run), "-odir", str(bundle)], cwd=root, env=env)
    return bundle, env


def test_a_selective_stage_bundle_verifies_and_an_altered_region_does_not(selective_stage,
                                                                          tmp_path):
    bundle, env = selective_stage
    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)
    record = yaml.safe_load((copy / "input" / "scaler.yaml").read_text(encoding="utf-8"))
    assert record["selection"]["selection_mode"] == "explicit"
    good = _run(["input/verify_state.py"], cwd=copy, env=env)
    assert "identical" in good.stdout

    record["scaler_arguments"]["cmap_terms"] = []
    record["scaler_arguments"]["torsion_central_bonds"] = \
        record["scaler_arguments"]["torsion_central_bonds"][1:]
    (copy / "input" / "scaler.yaml").write_text(yaml.safe_dump(record), encoding="utf-8")
    bad = _run(["input/verify_state.py"], cwd=copy, env=env, ok=False)
    assert bad.returncode == 1 and "DIFFERS" in bad.stdout, bad.stdout + bad.stderr


# --- direct exchange energies at stored coordinates ------------------------------------------------

#: Tolerance on a recomputed reduced potential, in kT, CALIBRATED before the comparison was made
#: final. The CPU platform (and CUDA mixed) holds positions in single precision: re-evaluating one
#: stored configuration on a fresh Context of the same platform reproduces the ladder's u only to
#: that precision. Measured on this fixture (2026-09-19, CPU platform): rounding the stored float64
#: coordinates to float32 moves u by 0.002 kT, and the recomputed values differ from the recorded
#: ones by at most 0.0077 kT absolute and 0.0055 kT in the cross-Hamiltonian differences. 0.05 kT
#: is ~10x that floor. The quantity the test must resolve is a WRONG Hamiltonian, which misses by
#: 8.4 kT (the region's nonbonded set without its torsions and CMAP) to 32 kT (the whole solute)
#: here, and the test checks it does. (A first tolerance of 1e-3 kT was a guess made before this
#: calibration, for a double-precision recomputation; it is not the platform's precision.)
EXCHANGE_ENERGY_TOLERANCE_KT = 0.05


def stored_exchange_energies(run: Path):
    """`(u[k], positions[walker], box[walker], k)` of the last committed exchange, in float64:
    the ladder checkpoint stores each walker's coordinates after exchange k, and an exchange moves
    no coordinate, so they are the configurations `u[k][i][w]` was evaluated at."""
    import netCDF4
    import numpy as np

    checkpoint = netCDF4.Dataset(run / "REST2_checkpoint.nc")
    k = int(checkpoint.getncattr("exchange_index"))
    positions = np.array(checkpoint["positions"][:], dtype=float)
    box = np.array(checkpoint["box"][:], dtype=float)
    checkpoint.close()
    analysis = netCDF4.Dataset(run / "REST2.nc")
    u = np.array(analysis["u"][k], dtype=float)
    analysis.close()
    return u, positions, box, k


def reduced_potentials(systems, positions, box, *, platform: str):
    """u_i(x_w) for every System i and stored walker w, in kT at 300 K, on `platform`."""
    import numpy as np
    import openmm
    from openmm import unit

    kt = (unit.MOLAR_GAS_CONSTANT_R * 300.0 * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
    out = np.zeros((len(systems), len(positions)))
    for i, system in enumerate(systems):
        context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName(platform))
        for w in range(len(positions)):
            context.setPeriodicBoxVectors(*box[w])
            context.setPositions(positions[w])
            out[i, w] = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole) / kt
        del context
    return out


def exchange_energy_discrepancy(run: Path, systems, *, platform: str):
    """Worst |recomputed - recorded| over u and over u_i - u_0 (per walker), in kT."""
    import numpy as np

    u, positions, box, _k = stored_exchange_energies(run)
    mine = reduced_potentials(systems, positions, box, platform=platform)
    return (float(np.abs(mine - u).max()),
            float(np.abs((mine - mine[0]) - (u - u[0])).max()), mine.size)


def _saved_states(root: Path, n: int):
    from openmm import XmlSerializer

    return [XmlSerializer.deserialize((root / "build" / "REST2" / f"system_state{i}.xml")
                                      .read_text(encoding="utf-8")) for i in range(n)]


def _whole_solute_states(root: Path, n: int):
    """The WRONG Hamiltonian for a selective ladder: the same taus over the whole solute."""
    from openmm import XmlSerializer

    from md_tools.rest2.hamiltonian import build_scaled_system

    record = yaml.safe_load((root / "build" / "REST2" / "scaler.yaml").read_text(encoding="utf-8"))
    base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    return [build_scaled_system(base, record["solute"]["atom_indices"], state["tau"],
                                excluded_bonds=[tuple(b) for b in
                                                record["unscaled_torsions"]["unscaled_central_bonds"]])
            for state in record["states"][:n]]


def test_a_selective_ladders_exchange_energies_are_its_saved_states_energies(selective_ladder):
    """u[k][i][w], recorded by the ladder, against the saved state i evaluated here at walker w's
    stored coordinates, on the platform the ladder ran on (CPU)."""
    root, run, _env_ = selective_ladder
    absolute, cross, compared = exchange_energy_discrepancy(run, _saved_states(root, STATES),
                                                            platform="CPU")
    print(f"exchange energies: {compared} values, worst |u| {absolute:.3g} kT, worst "
          f"cross-Hamiltonian {cross:.3g} kT")
    assert compared == STATES * STATES
    assert absolute < EXCHANGE_ENERGY_TOLERANCE_KT and cross < EXCHANGE_ENERGY_TOLERANCE_KT
    # ...and the check can fail: the whole-solute Hamiltonian at the same taus is caught.
    _, wrong, _ = exchange_energy_discrepancy(run, _whole_solute_states(root, STATES),
                                              platform="CPU")
    assert wrong > 20 * EXCHANGE_ENERGY_TOLERANCE_KT, wrong


# --- with identical states, a selection cannot refuse on its own -------------------------------------

@pytest.fixture(scope="module")
def selective_parent(tmp_path_factory):
    """A completed SELECTIVE ladder run by this checkout: the parent the next test extends."""
    root = tmp_path_factory.mktemp("selective-parent")
    env = _env(root, REPO / "src")
    _dataset(root)
    _scaler(root, env, SELECTIVE_SCALER)
    _project(root, env)
    _initial_state(root)
    run = root / "run"
    _ladder(root, env, run)
    return root, run


def test_with_identical_states_a_selection_cannot_refuse_on_its_own(selective_parent, tmp_path):
    """The identity hashes the Hamiltonian, not its provenance (S0 ruling, 323779f). Rebuild the
    states with the SAME region spelled differently: every state file is byte-identical, the
    record's provenance differs, and the out-of-place extension must be ACCEPTED. The
    hamiltonian refusal of `compare_identity` on the selection alone is then unreachable: the
    states agree, so the only thing left to differ is provenance, and provenance is not identity."""
    source_root, _ = selective_parent
    root = tmp_path / "copy"
    shutil.copytree(source_root, root, symlinks=True)
    parent = root / "run"
    env = _env(root, REPO / "src")
    states = root / "build" / "REST2"
    before_states = {p.name: p.read_bytes() for p in states.glob("system_state*.xml")}
    before_record = yaml.safe_load((states / "scaler.yaml").read_text(encoding="utf-8"))

    _scaler(root, env, LEGACY_SCALER + 'backbone_scaling_list: ":2-2"\n', overwrite=True)
    after_record = yaml.safe_load((states / "scaler.yaml").read_text(encoding="utf-8"))
    assert {p.name: p.read_bytes() for p in states.glob("system_state*.xml")} == before_states
    assert after_record["selection"]["masks"]["backbone"] == ":2-2"
    assert after_record["selection_provenance_sha256"] != \
        before_record["selection_provenance_sha256"], "the record's provenance DID change"
    assert after_record["selection_sha256"] == before_record["selection_sha256"]

    extension = root / "extended"
    _ladder(root, env, extension, "--extend", "1", "--extend-from", str(parent))
    manifest = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["scientific_identity"]["hamiltonian"] == \
        _restart_identity(parent)["hamiltonian"]
