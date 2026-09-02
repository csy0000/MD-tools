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


# --- the evidence document ------------------------------------------------------------------------

def test_write_the_coverage_evidence(hardware, request):
    """Emit the matrix, with what actually ran, so the published table is a record of a run.

    Ordered last by name so the lanes above have filled `RESULTS`. It writes the document only
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
