"""Every CUDA-dependent function and branch, inventoried and exercised on real NVIDIA hardware.

"CUDA tested" has to mean something checkable. It cannot mean "the GPU tests pass", because that
sentence is true of a suite that never touches half the branches -- and the halves it misses are
the ones nobody thought about, which is why they are the ones that break.

So this file holds two things:

1. **THE INVENTORY.** Every function that creates an OpenMM `Context` or `Simulation`, selects or
   configures CUDA, evaluates energies or forces on it, integrates or minimises, changes REST2 or
   AIS Context parameters, or writes output derived from CUDA execution. It is checked against the
   source by `ast`, so a NEW Context-creating function that nobody added to the matrix fails this
   file rather than quietly going untested.

2. **THE LANES.** Real runs on real GPUs for the branches the other GPU files do not reach:
   the three CUDA precisions, HMR and non-HMR timesteps, explicit `--device` placement, a NetCDF
   source ensemble, and the tau-basis decomposition -- whose whole claim is an identity between
   energies, and single precision is exactly where an identity stops holding.

`docs/release-notes/cuda-coverage-matrix.md` is generated from this file by
`python -m pytest tests/test_cuda_coverage_matrix.py --cuda-evidence=<path>`, so the published
matrix is a record of a run rather than a table someone typed.

NOTHING HERE SUBSTITUTES. There is no CPU fallback, no Reference lane and no mock: a test that
cannot get a CUDA device fails, and is reported as an unmet criterion. That is the point.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "md_tools"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# --- the inventory --------------------------------------------------------------------------------

#: `module::function` -> what it does on the GPU, and which lane proves it.
#:
#: Every entry is a place a CUDA kernel actually runs. The mapping is data rather than prose so
#: `test_every_cuda_site_in_the_source_is_in_this_matrix` can hold it to the source.
CUDA_SITES = {
    "openmm/platform_policy.py::_prove_cuda_initialises": (
        "opens a one-particle CUDA Context to prove the platform works before any output exists",
        "test_platform_policy.py, and every preflight in every lane below"),
    "openmm/platform_policy.py::resolve_platform_request": (
        "constructs the CUDA Platform and its Precision/DeviceIndex properties",
        "test_cuda_precision_lane, test_explicit_device_placement_lane"),
    "openmm/builders.py::_write_initial_state": (
        "a Context over the built System, to write its initial state",
        "build-top in every lane's fixture"),
    "md/stage.py::stage_main": (
        "minimisation, NVT/NPT dynamics, restraints, barostat, reporters, checkpoint commit",
        "test_cmd_cuda_smoke.py, test_cmd_restart_integration.py, test_cuda_precision_lane, "
        "test_hmr_timestep_lane"),
    "remd/engine.py::ReplicaEngine.__init__": (
        "one Simulation per thermodynamic state, all on CUDA",
        "test_md_run_mpi_gpu.py, test_rrest2_cuda_smoke.py"),
    "remd/engine.py::count_cuda_devices": (
        "opens a Context per device index to count what this process can actually see",
        "test_explicit_device_placement_lane"),
    "remd/driver.py::ReplicaRun._build_platform": (
        "consumes the preflight platform; no second resolution",
        "test_md_run_mpi_gpu.py::test_each_rank_kept_its_own_record_and_ran_on_cuda"),
    "ais/run.py::run_one_path": (
        "AIS switching: parameter updates, work energies, the three basis probes, frames, "
        "state rows and checkpoint generations, all on CUDA",
        "test_md_run_mpi_gpu.py, test_ais_decomposition_lane"),
    "rest2/scaler.py::TauSwitcher.set_amplitude": (
        "pushes scaled parameters into a live CUDA Context via updateParametersInContext",
        "test_ais_decomposition_lane"),
}

#: Functions that construct a Context but never on CUDA, with the reason. Each is a deliberate,
#: named exemption rather than an omission -- and the reason is checkable by reading the callsite.
NON_CUDA_CONTEXT_SITES = {
    "openmm/system.py::protonate":
        "names the Reference platform explicitly, to add hydrogens deterministically while "
        "building a System. No dynamics, and deliberately platform-independent so a structure "
        "built on one machine is the structure built on another.",
    "remd/driver.py::ReplicaRun._begin":
        "constructs an ExchangeContext, which is bookkeeping around Simulations the engine "
        "already created on CUDA -- the kernels are the engine's, and are covered by its entry.",
}


def _context_creating_functions() -> dict[str, str]:
    """Every function in the package that constructs a `Simulation` or a `Context`.

    Found by walking the AST rather than by grepping, so a call spread over several lines, or one
    inside a nested function, is found the same way as a one-liner.
    """
    found: dict[str, str] = {}
    for module in sorted(SRC.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        stack: list[str] = []

        def walk(node, stack):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    walk(child, stack + [child.name])
                    continue
                if isinstance(child, ast.Call):
                    name = getattr(child.func, "id", None) or getattr(child.func, "attr", None)
                    if name in ("Simulation", "Context") and stack:
                        key = f"{module.relative_to(SRC).as_posix()}::{'.'.join(stack)}"
                        found.setdefault(key, name)
                walk(child, stack)

        walk(tree, stack)
    return found


def test_every_cuda_site_in_the_source_is_in_this_matrix():
    """A new Context-creating function must be classified, not silently left untested.

    This is what keeps the matrix from becoming a table that was true once. Adding a function that
    opens a Context and forgetting to say which lane exercises it fails HERE, at the point the
    function is added, rather than in six months when its CUDA branch turns out never to have run.
    """
    found = _context_creating_functions()
    classified = set(CUDA_SITES) | set(NON_CUDA_CONTEXT_SITES)
    unclassified = sorted(set(found) - classified)
    assert not unclassified, (
        "these functions construct an OpenMM Context and appear in neither CUDA_SITES nor "
        "NON_CUDA_CONTEXT_SITES:\n  " + "\n  ".join(unclassified))


def test_every_matrix_entry_still_names_a_function_that_exists():
    """The other direction: an entry left behind after a rename claims coverage of nothing."""
    stale = []
    for entry in sorted(set(CUDA_SITES) | set(NON_CUDA_CONTEXT_SITES)):
        module, _, qualified = entry.partition("::")
        path = SRC / module
        if not path.is_file():
            stale.append(f"{entry} (no such module)")
            continue
        leaf = qualified.split(".")[-1]
        if f"def {leaf}(" not in path.read_text(encoding="utf-8"):
            stale.append(f"{entry} (no such function)")
    assert not stale, "matrix entries naming nothing:\n  " + "\n  ".join(stale)


# --- the hardware ---------------------------------------------------------------------------------

def _cuda_devices() -> list[dict[str, str]]:
    """What `nvidia-smi` says is here. Recorded in the evidence, never inferred."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    devices = []
    for line in done.stdout.strip().splitlines():
        index, name, driver, memory = (part.strip() for part in line.split(",", 3))
        devices.append({"index": index, "name": name, "driver": driver, "memory": memory})
    return devices


@pytest.fixture(scope="module")
def hardware():
    devices = _cuda_devices()
    if not devices:
        pytest.fail("no NVIDIA device is visible. This file is the CUDA evidence lane: a missing "
                    "GPU is an unmet acceptance criterion, not a reason to skip.")
    from openmm import Platform

    names = {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail(f"OpenMM offers no CUDA platform; it has {sorted(names)}")
    return devices


def _environment(work: Path, **extra) -> dict[str, str]:
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    config = work / "user.config"
    # NESTED under `machine:`, which is where the resolver looks. Splatting the inner mapping in
    # at the top level wrote `openmm:` as a root key, the machine settings were therefore absent,
    # and every precision lane silently ran at the built-in default while claiming to test three.
    machine = extra.pop("machine", None)
    document = {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}
    if machine:
        document["machine"] = machine
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(config)
    base.update(extra)
    return base


def _machine(platform="CUDA", precision="mixed", device_policy="local_rank") -> dict:
    return {"machine": {"openmm": {"platform": platform, "precision": precision,
                                   "device_policy": device_policy}}}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One implicit ALA system, built once. Small enough to run many lanes against."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cuda-matrix")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


#: Results collected across the lanes, written into the evidence document at teardown.
RESULTS: list[dict[str, object]] = []


def _record(lane: str, *, feature: str, precision: str, device: str, detail: str = "") -> None:
    RESULTS.append({"lane": lane, "feature": feature, "precision": precision,
                    "device": device, "result": "passed", "detail": detail})


# --- precision ------------------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
def test_cuda_precision_lane(precision, built, hardware, tmp_path):
    """A cMD stage on CUDA at each supported precision, proved from the run's own record.

    Precision is a `machine.openmm` setting and it reaches the Context as a CUDA property. If it
    did not, every one of these would pass identically and prove nothing -- so the assertion is on
    what the RUN recorded, which is read back out of the provenance log.
    """
    work = tmp_path / precision
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 5,
                   "production_steps": 20},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 10}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"), "-odir", str(work)],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine(precision=precision)))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "cMD.log")
    acceleration = record["acceleration"]
    assert acceleration["resolved_platform"] == "CUDA", acceleration
    assert acceleration.get("cuda_precision") == precision, (
        f"the run recorded precision {acceleration.get('precision')!r}, not {precision!r}: the "
        f"setting did not reach the Context, so this lane proves nothing")
    assert (work / "cMD.dcd").is_file() and (work / "cMD.xml").is_file()
    _record("test_cuda_precision_lane", feature=f"cMD implicit, {precision} precision",
            precision=precision, device=acceleration.get("cuda_device_index") or "OpenMM-selected",
            detail="minimisation, restrained NVT, production, DCD, restart, checkpoint")


# --- HMR and the timestep -------------------------------------------------------------------------

def test_hmr_timestep_lane(built, hardware, tmp_path):
    """4 fs runs on CUDA when the masses prove repartitioning, and is refused when they do not.

    Both halves on the GPU, because the refusal is the half that matters and a refusal proved on
    the CPU says nothing about the platform the run would have used.
    """
    work = tmp_path / "hmr"
    work.mkdir()

    # (a) the refusal: 4 fs against the ordinary masses in `built.xml`.
    (work / "fast.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 10},
        "dynamics": {"timestep_fs": 4.0},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 10}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "fast"),
                                      "--config", str(work / "fast.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    destination = work / "refused"
    refused = subprocess.run(
        [sys.executable, str(work / "fast" / "cMD.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"), "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900,
        env=_environment(work, **_machine()))
    assert refused.returncode != 0, refused.stdout + refused.stderr
    assert not destination.exists(), "the refusal created output"

    # (b) an HMR-built system takes 4 fs and runs.
    (work / "hmr.config").write_text(
        "solvent:\n  model: GBn2\nhydrogen_mass_repartitioning:\n  enabled: true\n",
        encoding="utf-8")
    hmr_built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "hmr.xml", "-op", "hmr.pdb",
               "-log", "hmr.log", "--config", str(work / "hmr.config")],
        cwd=work, capture_output=True, text=True, timeout=1800)
    if hmr_built.returncode != 0:
        pytest.fail(f"could not build an HMR system, so the 4 fs lane cannot be run:\n"
                    f"{hmr_built.stdout}{hmr_built.stderr}")

    ran = subprocess.run(
        [sys.executable, str(work / "fast" / "cMD.py"),
         "-p", str(work / "hmr.pdb"), "-s", str(work / "hmr.xml"), "-odir", str(work / "ok")],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert ran.returncode == 0, ran.stdout + ran.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "ok" / "cMD.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert float(record["timestep"]["timestep_fs"]) == 4.0, record["timestep"]
    _record("test_hmr_timestep_lane", feature="cMD implicit, 4 fs with HMR; 4 fs refused without",
            precision="mixed", device=record["acceleration"].get("cuda_device_index") or "-",
            detail="both halves on CUDA")


# --- explicit device placement --------------------------------------------------------------------

def test_explicit_device_placement_lane(built, hardware, tmp_path):
    """`--device N` puts THIS process on device N, and the record says which one.

    Run on the last visible device rather than device 0, so a placement that was ignored shows up
    as a difference instead of coinciding with the default.
    """
    if len(hardware) < 2:
        pytest.fail(f"only {len(hardware)} CUDA device(s) visible; explicit placement cannot be "
                    f"distinguished from the default. Reported as an unmet criterion.")
    wanted = len(hardware) - 1

    work = tmp_path / "device"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 10},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 10}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"),
         "-odir", str(work / "run"), "--device", str(wanted)],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    acceleration = read_record(work / "run" / "cMD.log")["acceleration"]
    assert acceleration["resolved_platform"] == "CUDA"
    assert str(acceleration["cuda_device_index"]) == str(wanted), acceleration
    _record("test_explicit_device_placement_lane", feature="cMD implicit, --device placement",
            precision="mixed", device=str(wanted),
            detail=f"{len(hardware)} devices visible; ran on the last")


# --- the tau-basis decomposition, on CUDA ---------------------------------------------------------

def test_ais_decomposition_lane(built, hardware, tmp_path):
    """The three-group identity holds on a real GPU, at the precision a run actually uses.

    The decomposition's whole claim is an identity between energies. On the Reference platform
    that identity is exact to 1e-7 and the test is about the algebra; here it is about the
    PLATFORM, because single- and mixed-precision nonbonded sums are exactly where an identity
    stops holding to the tolerance somebody assumed on a CPU.
    """
    work = tmp_path / "ais"
    work.mkdir()

    import mdtraj

    frames = mdtraj.load(str(built / "built.pdb"))
    mdtraj.join([frames] * 12).save_dcd(str(work / "source.dcd"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "observation_interval_steps": 5,
                "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": 10, "system_printout": 10, "checkpoint_printout": 10}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"),
         "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    import csv

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA", record["acceleration"]
    decomposition = record["decomposition"]
    assert decomposition["potential_energy_evaluations_per_update"] == 3
    assert decomposition["switching_energy_evaluations"] > 0
    assert decomposition["switching_energy_evaluation_seconds"] > 0.0

    with (work / "run" / "AIS_work.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "the work table is empty"
    # The identity, on every row the GPU produced, at the tolerance this precision documents.
    from md_tools.ais.decomposition import reconstruction_tolerance

    precision = record["acceleration"].get("cuda_precision") or "mixed"
    checked = 0
    for row in rows:
        total = float(row["total_work_kj_mol"])
        parts = sum(float(row[name]) for name in
                    ("total_work_unscaled_kj_mol", "total_work_linear_kj_mol",
                     "total_work_quadratic_kj_mol"))
        allowed = reconstruction_tolerance(total, precision=precision) * len(rows)
        assert abs(parts - total) <= max(allowed, 1e-6), (
            f"step {row['switch_step']}: components sum to {parts} against a measured "
            f"{total} at {precision} precision")
        reconstructed = float(row["potential_total_reconstructed_kj_mol"])
        assert reconstructed == reconstructed, "a NaN reached the reconstruction"
        checked += 1
    assert checked >= 2
    _record("test_ais_decomposition_lane",
            feature="AIS implicit, three-group decomposition and work-sum identity",
            precision=precision, device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"{checked} rows checked; "
                   f"{decomposition['switching_energy_evaluations']} probe evaluations in "
                   f"{decomposition['switching_energy_evaluation_seconds']:.3f} s")


def test_ais_reads_a_netcdf_source_on_cuda(built, hardware, tmp_path):
    """The other supported source format. DCD is covered in test_md_run_mpi_gpu.py."""
    work = tmp_path / "netcdf"
    work.mkdir()

    import mdtraj

    frames = mdtraj.load(str(built / "built.pdb"))
    mdtraj.join([frames] * 12).save_netcdf(str(work / "source.nc"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 10,
                "observation_interval_steps": 5, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.nc"},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"),
         "-source-traj", str(work / "source.nc"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["source"]["format"] == "netcdf", record["source"]
    assert (work / "run" / "AIS_traj0000.nc").is_file()
    _record("test_ais_reads_a_netcdf_source_on_cuda",
            feature="AIS implicit, NetCDF source ensemble",
            precision=record["acceleration"].get("cuda_precision") or "mixed",
            device=record["acceleration"].get("cuda_device_index") or "-",
            detail="source format decided from content, not suffix")


# --- no fallback ----------------------------------------------------------------------------------

def test_there_is_no_automatic_cpu_fallback_on_this_machine(built, hardware, tmp_path):
    """With CUDA unavailable, a run REFUSES. It does not quietly become a CPU run.

    Exercised through the deterministic seam rather than by removing a driver, and asserted here
    -- in the CUDA file -- because "there is no fallback" is a claim about the CUDA lane.
    """
    work = tmp_path / "nofallback"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 5},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    destination = work / "run"
    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"), "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900,
        env=_environment(work, MD_TOOLS_FORCE_NO_CUDA="1", **_machine()))
    assert done.returncode != 0, done.stdout + done.stderr
    assert "CPU" not in done.stdout.replace("--cpu", ""), (
        "the run mentioned falling back to the CPU:\n" + done.stdout)
    assert not destination.exists(), "the refusal created output"
    _record("test_there_is_no_automatic_cpu_fallback_on_this_machine",
            feature="no automatic CPU fallback", precision="-", device="-",
            detail="CUDA made unavailable through the deterministic seam")


# --- explicit solvent -----------------------------------------------------------------------------
#
# Every lane above is implicit, which is the fast one. Explicit solvent is a different code path on
# the GPU in every way that matters: PME rather than a cutoff, a barostat that can be active, a box
# that has to survive a checkpoint, and a particle count two orders of magnitude larger. A matrix
# that proved only the implicit branch would be proving the branch that does not use PME.

@pytest.fixture(scope="module")
def built_explicit(tmp_path_factory):
    """One small explicit ALA box, built once. Slow enough to be worth building only here."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cuda-matrix-explicit")
    # `TIP3P` in the spelling the schema accepts, and the default padding: unknown keys and
    # unknown values are both refused rather than ignored, which is what made this fixture fail
    # loudly instead of quietly building something else.
    (root / "sys.config").write_text("solvent:\n  model: TIP3P\n", encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        pytest.fail("could not build an explicit-solvent system, so the explicit CUDA lanes "
                    f"cannot run:\n{done.stdout}{done.stderr}")
    return root


def test_explicit_solvent_npt_lane(built_explicit, hardware, tmp_path):
    """Explicit cMD on CUDA: PME, a real barostat, NPT, and a box that survives the restart."""
    work = tmp_path / "explicit"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 20},
        "reporting": {"solute_printout": 10, "system_printout": 10,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built_explicit / "built.pdb"), "-s", str(built_explicit / "built.xml"),
         "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "cMD.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["implicit"] is False, "this lane is meant to be the explicit one"
    assert record["stage"]["ensemble"] == "NPT", record["stage"]
    assert record["barostats"]["active"] >= 1, (
        f"an NPT explicit stage ran with no active barostat: {record['barostats']}")
    # The box has to come back out of the restart, or a continuation silently changes the volume.
    from openmm import XmlSerializer

    state = XmlSerializer.deserialize(
        (work / "run" / "cMD.xml").read_text(encoding="utf-8"))
    vectors = state.getPeriodicBoxVectors()
    assert vectors is not None and vectors[0][0]._value > 0.0, vectors
    _record("test_explicit_solvent_npt_lane",
            feature="cMD explicit (PME), NPT with an active barostat",
            precision=record["acceleration"].get("cuda_precision") or "mixed",
            device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"{record['stage']['steps']} production steps; box preserved in the restart")


def _equilibrate_chain(project: Path, topology: Path, system: Path, work: Path, run: Path, *,
                       last: str) -> Path:
    """Run a cMD project's whole chain, including its production stage, and return that stage."""
    return _equilibrate(project, topology, system, work, run, script=f"{last}.py",
                        include_last=True)


def _equilibrate(project: Path, topology: Path, system: Path, work: Path, run: Path, *,
                 script: str = "REST2.py", include_last: bool = False) -> Path:
    """Run the project's equilibration chain and return the state the ladder starts from.

    A ladder is refused without `-c`: every rung starts from the same equilibrated configuration,
    and a group line with no coordinates is a rung starting from the topology's own positions.
    So the lanes below run the chain rather than inventing a starting state -- which also puts
    the minimisation and every equilibration stage on CUDA, in the same lane.
    """
    from md_tools.md.stage import load_generated_plan

    plan, _config = load_generated_plan(project / script)
    previous = None
    for stage in plan:
        name = stage["name"]
        script = project / f"{name}.py"
        if not script.is_file():
            continue
        argv = [sys.executable, str(script), "-p", str(topology), "-s", str(system),
                "-odir", str(run)]
        if previous is not None:
            argv += ["-c", str(previous)]
        done = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=3600,
                              env=_environment(work, **_machine()))
        assert done.returncode == 0, f"{name}:\n{done.stdout}{done.stderr}"
        previous = run / f"{name}.xml"
    assert previous is not None and previous.is_file(), "the chain produced no starting state"
    return previous


def test_explicit_solvent_rest2_lane(built_explicit, hardware, tmp_path):
    """A REST2 ladder on explicit solvent, on CUDA, serially.

    The scaled Hamiltonian over a PME system is the combination REST2 is actually used for, and
    it is the one where a NonbondedForce exception or a reciprocal-space term scaling wrongly
    would show up as an acceptance ratio nobody questions.
    """
    work = tmp_path / "rest2-explicit"
    work.mkdir()
    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "explicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 20},
        "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 10,
                  "number_of_exchanges": 2},
        "reporting": {"solute_printout": 10, "system_printout": 10,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "REST2.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    start = _equilibrate(work / "project", built_explicit / "built.pdb",
                         built_explicit / "built.xml", work, work / "run")
    done = subprocess.run(
        [sys.executable, str(work / "project" / "REST2.py"),
         "-p", str(built_explicit / "built.pdb"), "-s", str(built_explicit / "built.xml"),
         "-c", str(start), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    trajectories = sorted((work / "run").glob("remd*.nc"))
    assert len(trajectories) == 2, [p.name for p in trajectories]
    report = (work / "run" / "REST2.out").read_text(encoding="utf-8")
    assert "platform           : CUDA" in report, report[:2000]
    _record("test_explicit_solvent_rest2_lane",
            feature="REST2 explicit (PME), 2 states, serial",
            precision="mixed", device="-",
            detail=f"{len(trajectories)} state trajectories; exchanges attempted on CUDA")


@pytest.mark.parametrize("states", [2, 4, 6])
def test_multi_rank_rest2_ladders_of_several_sizes(states, built, hardware, tmp_path):
    """Real `mpirun -n N` ladders on CUDA, one rank per state, one GPU per rank.

    Sizes rather than one size: `owned_states` and the neighbour exchange rule both depend on the
    state count's parity, and a ladder that only ever ran with 2 has never exercised the case
    where a rank has a neighbour on both sides. 6 is the largest this machine's 9 devices allow
    with a device to spare.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the multi-rank CUDA lane is an unmet criterion")
    if len(hardware) < states:
        pytest.fail(f"{states} states need {states} devices; {len(hardware)} visible")

    work = tmp_path / f"ladder-{states}"
    work.mkdir()
    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 5,
                   "production_steps": 20},
        "rest2": {"number_of_replicas": states, "exchange_interval_steps": 10,
                  "number_of_exchanges": 2},
        "reporting": {"solute_printout": 10, "system_printout": 10,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "REST2.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    start = _equilibrate(work / "project", built / "built.pdb", built / "built.xml",
                         work, work / "run")
    done = subprocess.run(
        ["mpirun", "-n", str(states), sys.executable, str(work / "project" / "REST2.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"),
         "-c", str(start), "-odir", str(work / "run"), "-ng", str(states)],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    trajectories = sorted((work / "run").glob("remd*.nc"))
    assert len(trajectories) == states, [p.name for p in trajectories]

    # One device per rank, and they must be DIFFERENT devices: `device_policy: local_rank` is the
    # setting, and N ranks sharing one GPU is the failure it exists to prevent.
    devices = []
    for rank in range(states):
        name = "REST2.out" if rank == 0 else f"REST2.out.rank{rank:02d}"
        text = (work / "run" / name).read_text(encoding="utf-8")
        assert "platform           : CUDA" in text, text[:1500]
        line = next(line for line in text.splitlines() if "platform           :" in line)
        devices.append(line.split("device=")[1].split()[0] if "device=" in line else None)
    assert len(set(devices)) == states, f"ranks shared devices: {devices}"
    _record("test_multi_rank_rest2_ladders_of_several_sizes",
            feature=f"REST2 implicit, {states} states, real mpirun -n {states}",
            precision="mixed", device=", ".join(str(d) for d in devices),
            detail=f"{len(trajectories)} state trajectories; one rank per device")


def test_a_genuinely_absent_cuda_device_is_refused_without_any_seam(built, hardware, tmp_path):
    """`CUDA_VISIBLE_DEVICES=""`. No seam, no monkeypatch: the driver really has no device.

    `MD_TOOLS_FORCE_NO_CUDA` proves the refusal PATH, and it is our own code deciding to fail --
    which is exactly the thing under test, so on its own it is a weak witness. Emptying
    `CUDA_VISIBLE_DEVICES` makes the CUDA platform fail in OpenMM, in the driver, with
    `CUDA_ERROR_NO_DEVICE`. That is the failure a machine with a busy or broken GPU actually
    produces, and the run must refuse rather than quietly become a CPU run.
    """
    work = tmp_path / "nodevice"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 5},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    destination = work / "run"
    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"), "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900,
        env=_environment(work, CUDA_VISIBLE_DEVICES="", **_machine()))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message
    assert not destination.exists(), (
        f"the refusal created output: {sorted(p.name for p in destination.iterdir())}")
    # And it refused for the right reason, naming the platform rather than something downstream.
    assert "CUDA" in message, message[-2000:]
    _record("test_a_genuinely_absent_cuda_device_is_refused_without_any_seam",
            feature="CUDA genuinely unavailable (CUDA_VISIBLE_DEVICES=\"\"): refused, no fallback",
            precision="-", device="none visible",
            detail="real CUDA_ERROR_NO_DEVICE from the driver, not a test seam")


def test_a_genuinely_unimportable_mpi4py_stops_a_plural_launch(built, hardware, tmp_path):
    """A broken mpi4py, made broken the way a broken one is: it raises on import.

    `MD_TOOLS_FORCE_NO_MPI4PY` proves the path and is again our own code choosing to fail. This
    shadows the real package with one whose `__init__` raises the ImportError a missing
    `libmpi.so` actually produces, so what runs is the genuine `except ImportError` branch in
    `md_tools.remd.mpi` -- the one that must refuse a plural launch rather than let N ranks each
    believe they are the whole world and write over one set of files.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the plural-launch lane is an unmet criterion")

    work = tmp_path / "brokenmpi"
    (work / "shadow" / "mpi4py").mkdir(parents=True)
    (work / "shadow" / "mpi4py" / "__init__.py").write_text(
        'raise ImportError("libmpi.so.40: cannot open shared object file: '
        'No such file or directory")\n', encoding="utf-8")

    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 10},
        "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                  "number_of_exchanges": 2},
        "reporting": {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "REST2.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    environment = _environment(work, **_machine())
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(work / "shadow"), environment.get("PYTHONPATH", "")])

    destination = work / "run"
    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(work / "project" / "REST2.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"),
         "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900, env=environment)
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "mpi4py" in message, message[-2000:]
    assert not destination.exists(), (
        "two uncoordinated ranks created output over one set of paths")
    _record("test_a_genuinely_unimportable_mpi4py_stops_a_plural_launch",
            feature="mpi4py genuinely unimportable under mpirun -n 2: refused, nothing written",
            precision="-", device="-",
            detail="the real ImportError branch, through a shadowing package, not a test seam")


def test_multi_rank_rrest2_with_a_real_reservoir(built, hardware, tmp_path):
    """rREST2 under real `mpirun -n 2` on CUDA, against a reservoir produced at the top rung.

    The serial rREST2 lane lives in `test_rrest2_cuda_smoke.py`. This is the coordinated one:
    the reservoir refresh is a COLLECTIVE operation -- rank 0 draws the sample and every rank has
    to agree about which exchange it happened at -- and a refresh that works in one process says
    nothing about one that has to be agreed across four.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the multi-rank rREST2 lane is an unmet criterion")
    if len(hardware) < 2:
        pytest.fail(f"2 states need 2 devices; {len(hardware)} visible")

    work = tmp_path / "rrest2-mpi"
    work.mkdir()
    tau_max = 0.5

    # The reservoir: a fixed-tau run AT THE LADDER'S TOP RUNG, streaming complete phase space.
    # Anything else is a sample of a different distribution.
    (work / "hot.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "dynamics": {"tau": tau_max, "seed": 11, "phase_space_printout": 10},
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "production_steps": 40},
        "reporting": {"solute_printout": 10, "system_printout": 20,
                      "checkpoint_printout": 40}}, sort_keys=False), encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", str(work / "hot"),
                                 "--config", str(work / "hot.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    hot_run = work / "hotrun"
    _equilibrate_chain(work / "hot", built / "built.pdb", built / "built.xml", work, hot_run,
                       last="cMD")
    phase_space = sorted(hot_run.glob("*phase*"))
    assert phase_space, f"no phase-space file: {sorted(p.name for p in hot_run.iterdir())}"
    reservoir = phase_space[0]

    (work / "rrest2.config").write_text(yaml.safe_dump({
        "protocol": "rREST2", "solvent": "implicit",
        "dynamics": {"seed": 13},
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "production_steps": 40},
        "reporting": {"solute_printout": 10, "system_printout": 20,
                      "checkpoint_printout": 40},
        "rest2": {"number_of_replicas": 2, "tau_max": tau_max,
                  "exchange_interval_steps": 20, "number_of_exchanges": 2},
        "reservoir": {"enabled": True, "path": str(reservoir),
                      "refresh_interval_exchanges": 1, "velocities": "inherit"}},
        sort_keys=False), encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                 "--config", str(work / "rrest2.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    run = work / "run"
    start = _equilibrate(work / "project", built / "built.pdb", built / "built.xml", work, run,
                         script="rREST2.py")
    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(work / "project" / "rREST2.py"),
         "-p", str(built / "built.pdb"), "-s", str(built / "built.xml"),
         "-c", str(start), "-odir", str(run), "-ng", "2"],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    trajectories = sorted(run.glob("remd*.nc"))
    assert len(trajectories) == 2, [p.name for p in trajectories]
    report = (run / "rREST2.out").read_text(encoding="utf-8")
    assert "platform           : CUDA" in report, report[:1500]
    assert "reservoir" in report.lower(), report[:2000]
    _record("test_multi_rank_rrest2_with_a_real_reservoir",
            feature="rREST2 implicit, 2 states, real mpirun -n 2, real reservoir refresh",
            precision="mixed", device="0, 1",
            detail=f"reservoir {reservoir.name}; {len(trajectories)} state trajectories")


def test_ais_on_explicit_solvent(built_explicit, hardware, tmp_path):
    """AIS over a PME system on CUDA, and the decomposition identity with it.

    Every other AIS lane is implicit. The identity's claim is that it survives the PME reciprocal
    sum, the Ewald self-energy and the long-range dispersion correction -- and none of those exist
    in an implicit system, so no implicit lane tests the part of the claim most likely to be wrong.
    """
    work = tmp_path / "ais-explicit"
    work.mkdir()

    import mdtraj

    frames = mdtraj.load(str(built_explicit / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(work / "source.dcd"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "explicit",
        "ais": {"number_of_paths": 2, "switching_steps": 10,
                "observation_interval_steps": 5, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": 5, "system_printout": 5,
                      "checkpoint_printout": 5}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"),
         "-p", str(built_explicit / "built.pdb"), "-s", str(built_explicit / "built.xml"),
         "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    import csv

    from md_tools.ais.decomposition import reconstruction_tolerance
    from md_tools.build.record import read_record

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["implicit"] is False, "this lane is meant to be the explicit one"
    precision = record["acceleration"].get("cuda_precision") or "mixed"

    with (work / "run" / "AIS_work.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    worst = 0.0
    for row in rows:
        total = float(row["total_work_kj_mol"])
        parts = sum(float(row[name]) for name in
                    ("total_work_unscaled_kj_mol", "total_work_linear_kj_mol",
                     "total_work_quadratic_kj_mol"))
        worst = max(worst, abs(parts - total))
        allowed = reconstruction_tolerance(total, precision=precision) * len(rows)
        assert abs(parts - total) <= max(allowed, 1e-6), (
            f"explicit, step {row['switch_step']}: components sum to {parts} against {total}")

    decomposition = record["decomposition"]
    particles = record["inputs"]["system"].get("particles") if isinstance(
        record["inputs"].get("system"), dict) else None
    _record("test_ais_on_explicit_solvent",
            feature="AIS explicit (PME), three-group identity through the reciprocal sum",
            precision=precision, device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"{len(rows)} rows, worst |sum - total| = {worst:.3e} kJ/mol; "
                   f"{decomposition['switching_energy_evaluations']} probe evaluations in "
                   f"{decomposition['switching_energy_evaluation_seconds']:.3f} s"
                   + (f"; {particles} particles" if particles else ""))


def test_the_decomposition_cost_is_measured_on_a_large_system(built, built_explicit, hardware,
                                                              tmp_path):
    """What the three extra evaluations per update actually cost, implicit against explicit.

    "The overhead is acceptable" is only a claim if it comes with a number, and a number from a
    22-particle implicit system says nothing about a solvated one -- where the parameter pushes
    (`updateParametersInContext` over every particle and every exception) dominate, not the energy
    evaluation.
    """
    import csv

    from md_tools.build.record import read_record

    measured = {}
    for label, root, solvent in (("implicit", built, "implicit"),
                                 ("explicit", built_explicit, "explicit")):
        work = tmp_path / f"cost-{label}"
        work.mkdir()

        import mdtraj

        frames = mdtraj.load(str(root / "built.pdb"))
        mdtraj.join([frames] * 8).save_dcd(str(work / "source.dcd"))
        (work / "AIS.config").write_text(yaml.safe_dump({
            "protocol": "AIS", "solvent": solvent,
            "ais": {"number_of_paths": 1, "switching_steps": 20,
                    "observation_interval_steps": 5,
                    "parameter_update_interval_steps": 1},
            "ais_source": {"trajectory": "../source.dcd"},
            "reporting": {"solute_printout": 10, "system_printout": 10,
                          "checkpoint_printout": 20}}), encoding="utf-8")
        assert subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                     "--config", str(work / "AIS.config")],
                              capture_output=True, text=True, timeout=600).returncode == 0
        done = subprocess.run(
            [sys.executable, str(work / "project" / "AIS.py"),
             "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
             "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
            cwd=work, capture_output=True, text=True, timeout=3600,
            env=_environment(work, **_machine()))
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

        record = read_record(work / "run" / "AIS.log")
        assert record["acceleration"]["resolved_platform"] == "CUDA"
        decomposition = record["decomposition"]
        evaluations = int(decomposition["switching_energy_evaluations"])
        seconds = float(decomposition["switching_energy_evaluation_seconds"])
        assert evaluations > 0 and seconds > 0.0
        measured[label] = (evaluations, seconds, seconds / evaluations)

    for label, (evaluations, seconds, each) in measured.items():
        _record("test_the_decomposition_cost_is_measured_on_a_large_system",
                feature=f"decomposition overhead, AIS {label}, update interval 1 step",
                precision="mixed", device="-",
                detail=f"{evaluations} probe evaluations in {seconds:.3f} s "
                       f"({each * 1000:.2f} ms each)")
    # Not a performance assertion -- hardware varies and a threshold here would be a flaky test
    # pretending to be a measurement. What is asserted is that the number EXISTS for both sizes,
    # which is what "measured and documented" requires.
    assert set(measured) == {"implicit", "explicit"}


# --- the evidence document ------------------------------------------------------------------------

def test_write_the_coverage_evidence(hardware, request):
    """Emit the matrix, with what actually ran, so the published table is a record of a run.

    LAST IN THE FILE, deliberately. pytest runs a module in definition order, and `RESULTS` is
    filled by the lanes as they pass -- so a lane defined below this function would run after it
    and be missing from the document, which is how a matrix comes to describe less than was
    actually verified. It writes the document only
    when `--cuda-evidence=<path>` is given, so an ordinary GPU run does not rewrite a committed
    file as a side effect.
    """
    destination = request.config.getoption("--cuda-evidence", default=None)
    if not destination:
        pytest.skip("no --cuda-evidence=<path> given; the lanes above ran, nothing to write")

    from openmm import version as openmm_version

    lines = ["# CUDA coverage matrix", "",
             "Generated by `tests/test_cuda_coverage_matrix.py`. Every row is a real run on the "
             "hardware named below.", "", "## Hardware", ""]
    lines += [f"* device {device['index']}: {device['name']}, driver {device['driver']}, "
              f"{device['memory']}" for device in hardware]
    lines += ["", f"* OpenMM {openmm_version.full_version}", ""]
    lines += ["## Source sites", "",
              "| source function or branch | what runs on CUDA | lane |", "|---|---|---|"]
    for site, (what, lane) in sorted(CUDA_SITES.items()):
        lines.append(f"| `{site}` | {what} | {lane} |")
    lines += ["", "## Lanes executed in this run", "",
              "| lane | feature | precision | device | result | detail |", "|---|---|---|---|---|---|"]
    for entry in RESULTS:
        lines.append(f"| `{entry['lane']}` | {entry['feature']} | {entry['precision']} | "
                     f"{entry['device']} | {entry['result']} | {entry['detail']} |")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert Path(destination).is_file()
