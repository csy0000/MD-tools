"""Shared fixtures. Deliberately few: these tests check user-visible behaviour, not internals."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALA_PDB = Path(__file__).resolve().parent / "data" / "ALA.pdb"


def pytest_addoption(parser):
    parser.addoption(
        "--error-on-skip", action="store_true", default=False,
        help="turn every skip into a failure. Release validation uses this: a scientific smoke "
             "test that skipped because a dependency was missing is a test that did not run, and "
             "reporting that as a pass with a note is how a broken release ships.")
    parser.addoption(
        "--cuda-evidence", action="store", default=None, metavar="PATH",
        help="write the CUDA coverage matrix, with the lanes that actually ran, to PATH. Omitted "
             "in an ordinary run so a GPU suite does not rewrite a committed document as a side "
             "effect of passing.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.config.getoption("--error-on-skip"):
        report.outcome = "failed"
        report.longrepr = (f"{item.nodeid} skipped, and --error-on-skip forbids skips here:\n"
                           f"{report.longrepr}")


#: The `src/` of the checkout these tests belong to. Both this process and every subprocess it
#: spawns are pinned to it.
CHECKOUT_SRC = REPO_ROOT / "src"


def _pin_this_process_to_the_checkout() -> None:
    """Make `md_tools` resolve to THIS checkout, in this process and in everything it spawns.

    conftest is imported before any test module, so this runs before the suite's first
    `from md_tools...`. Two halves, and both are needed:

      * `sys.path`, for in-process imports. Without it, a machine carrying an editable install of
        another checkout runs every in-process test against that other code -- which surfaces as
        the BASE commit's defects failing tests the branch under test has already fixed.
      * `os.environ["PYTHONPATH"]`, for subprocesses. Tests spawn `python -m
        md_tools.cli.md_openmm` to generate projects, and a subprocess resolves the INSTALLED
        package regardless of what this process imported. Setting the variable here rather than
        passing `env=` per call means every call site is covered, including ones written later --
        several tests spawn their own subprocesses without going through `run_cli`.

    This is not hypothetical. A full GPU suite once came back green having generated every project
    with a different clone's code, and that result was reported as evidence on a pull request.
    """
    entry = str(CHECKOUT_SRC)
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    for name in [n for n in sys.modules if n == "md_tools" or n.startswith("md_tools.")]:
        del sys.modules[name]
    importlib.invalidate_caches()

    inherited = os.environ.get("PYTHONPATH")
    parts = [q for q in (inherited or "").split(os.pathsep) if q and q != entry]
    os.environ["PYTHONPATH"] = os.pathsep.join([entry, *parts])


_pin_this_process_to_the_checkout()




#: A 3 nm cube, enough to make the System periodic. Nothing is solvated into it.
_ANGSTROM_BOX = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0))


def make_dataset_root(root: Path, *, solvent: str = "implicit") -> Path:
    """A dataset root with `build/built.{xml,pdb}`, built WITHOUT tleap. Returns `root`.

    *solvent* must match the projects the caller is about to generate: `build-md` validates the
    whole chain against this System, so an explicit project needs `solvent="explicit"` and an
    implicit one the default.

    WHY THIS EXISTS. Scaling moved to build time: `build-md` for a ladder now deserialises
    `../build/built.xml`, scales it to every rung and serialises the result, so a REST2 or rREST2
    generation cannot be produced from an empty directory any more. Every fixture that used to do
    exactly that needs a built System.

    WHY NOT `build-top`. The honest route needs `tleap` from AmberTools and takes long enough that
    the fixtures using it carry 1800 s timeouts. Most tests that generate a project only INSPECT
    THE GENERATED TEXT -- which files exist, what run.sh types, whether an `.in` round-trips --
    and for those the System's provenance is not the variable. `openmm.app.ForceField` gives a
    real, fully parameterised, serialisable System in ~0.1 s with no external executable.

    WHAT THIS IS THEREFORE NOT. It is not what `build-top` produces, so it must not be used to
    assert anything ABOUT `build-top`, about a System's provenance, or as evidence for a run's
    scientific content. A test whose subject is the built System itself uses `build-top` --
    `test_direct_runtime_preflight.py` and `test_consumer_contract_gates.py` still do, and they
    keep their tleap dependency deliberately.

    It is nonetheless a genuine exercise of the rung path: ALA.pdb is ACE-ALA-NME, so
    `classify_unscaled_torsions` finds two ordinary amide omega and protects the torsions around them
    rather than finding nothing to do.
    """
    from openmm import XmlSerializer, app

    build = Path(root) / "build"
    build.mkdir(parents=True, exist_ok=True)

    # NEVER OVERWRITE AN EXISTING BUILT SYSTEM, and this is a correctness guard rather than an
    # optimisation. Several modules call the real `build-top` into `build/` AND reach this helper
    # through a shared fixture; writing unconditionally would replace a tleap-built System with
    # this hand-parameterised stand-in, and the tests that followed would integrate a DIFFERENT
    # Hamiltonian while every path still resolved. That is a silent substitution of scientific
    # content, which is exactly what a fixture must not do.
    if (build / "built.xml").is_file() and (build / "built.pdb").is_file():
        return Path(root)

    pdb = app.PDBFile(str(ALA_PDB))
    if solvent == "explicit":
        # PERIODIC, which is the property `build-md` now reads. Generation validates the whole
        # chain against the built System, so an explicit project -- whose equilibration is NPT --
        # is refused against the implicit System below: there is no volume to control. A test
        # that generates explicit text therefore needs a System with a box.
        #
        # A BOX, NOT A SOLVATED SYSTEM. No water is added: filling a box costs seconds per
        # fixture and none of the tests using this inspect the solvent. What they need is a
        # System that is periodic and barostattable, which this is. Everything in the docstring
        # above about what this helper may not stand for applies here with more force.
        #
        # SO DO NOT INTEGRATE IT. An empty periodic box under a barostat collapses immediately --
        # "the periodic box size has decreased to less than twice the nonbonded cutoff" -- so a
        # test that actually RUNS an explicit stage needs either real solvation or, where the
        # solvent is beside the point, the implicit System below. This is for generation-time
        # assertions: which files exist, what run.sh types, whether an `.in` round-trips.
        pdb.topology.setPeriodicBoxVectors(_ANGSTROM_BOX)
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         constraints=app.HBonds, rigidWater=True)
    else:
        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds, rigidWater=True)
    (build / "built.xml").write_text(XmlSerializer.serialize(system), encoding="utf-8")
    with open(build / "built.pdb", "w") as handle:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, handle)
    return Path(root)


@pytest.fixture
def dataset_root(tmp_path):
    """A dataset root for one test. See `make_dataset_root`."""
    return make_dataset_root(tmp_path)


def run_cli(module: str, *args, cwd: Path | None = None):
    """Invoke an entry point the way a user does. Pinned via os.environ, see above.

    There is no longer any routing here. The shim that mapped `sys-config`, `sys-gen` and `md-gen`
    onto the generator API existed only while AIS still needed that route; AIS is a `build-md`
    protocol now, the generators are deleted, and a test that names a retired subcommand should
    fail exactly as a user's shell would.
    """
    return subprocess.run([sys.executable, "-m", f"md_tools.cli.{module}", *args],
                          capture_output=True, text=True, cwd=str(cwd or REPO_ROOT))








@pytest.fixture
def md_openmm(tmp_path):
    def call(*args, cwd=None):
        return run_cli("md_openmm", *args, cwd=cwd or tmp_path)
    return call


def template_module(name: str):
    """Import one of the modules that is copied into a generated project.

    The generated scripts import these by filename beside themselves, so there is no package to
    import them from. Loading them by path is how a test exercises the same code the run does.
    """
    import importlib.util
    import sys

    path = REPO_ROOT / "src" / "md_tools" / "remd" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_template_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dcd_header(path: Path) -> dict:
    """Frame and atom counts straight out of a DCD header.

    No DCD reader is a declared dependency of this repository, and adding one to assert a frame
    count would be a dependency bought for a test. The header carries both numbers, and the atom
    count is the part that matters: a solute-subset trajectory read against the whole-system
    topology is the mistake being guarded against.
    """
    import struct

    raw = Path(path).read_bytes()
    frames = struct.unpack("<i", raw[8:12])[0]
    interval = struct.unpack("<i", raw[16:20])[0]
    offset = 4 + 84 + 4
    title_bytes = struct.unpack("<i", raw[offset:offset + 4])[0]
    offset += 4 + title_bytes + 4
    atoms = struct.unpack("<i", raw[offset + 4:offset + 8])[0]
    return {"frames": frames, "interval": interval, "atoms": atoms}


#: A smoke protocol: picoseconds of dynamics and a box as small as the cutoff allows. What these
#: sizes test is the stage chain and the bookkeeping, not any scientific quantity.


# `tiny_project` is gone with `openmm/sysgen.py` and `openmm/mdgen.py`, which it called
# directly. It had no callers left, so it was scaffolding that could only have failed. Tests
# that need a real project build one through the public commands.


def run_stage(directory: Path, script: str = "run.py", platform: str | None = None,
              extra_env: dict | None = None):
    """Run one generated stage the way a user does: on CUDA.

    MD runs on a GPU. A minimisation, an equilibration, a production run or a restart validated on
    the CPU says nothing about the platform the work is actually done on, and the CPU and CUDA
    paths differ in exactly the places these tests exist to check -- device assignment, context
    creation, and which replica lands where. `MD_PLATFORM` is left unset so the generated script
    resolves CUDA itself, which is also the resolution being tested.
    """
    import os
    import subprocess
    import sys as _sys

    environment = dict(os.environ)
    environment.pop("MD_PLATFORM", None)
    if platform:
        environment["MD_PLATFORM"] = platform
    if extra_env:
        environment.update(extra_env)
    return subprocess.run([_sys.executable, script], cwd=str(directory), capture_output=True,
                          text=True, env=environment, timeout=1800)


def cuda_is_available() -> bool:
    """Whether this machine can run the MD tests at all."""
    try:
        from openmm import Platform

        names = {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}
    except Exception:
        return False
    if "CUDA" not in names:
        return False
    try:
        import openmm

        system = openmm.System()
        system.addParticle(1.0)
        context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                                 Platform.getPlatformByName("CUDA"))
        context.setPositions([(0, 0, 0)])
        context.getState(getEnergy=True)
        del context
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Fail, do not silently pass, when a GPU test is collected on a machine without one.

    `-m "not gpu"` is the supported way to run the rest of the suite on a CPU-only machine. What
    is not supported is a green suite on such a machine that looks like it validated the runtime.
    """
    if not any("gpu" in item.keywords for item in items):
        return
    if cuda_is_available():
        return
    skip = pytest.mark.skip(reason="no working CUDA platform; MD tests validate nothing on CPU. "
                                   "Run them on the GPU machine, or deselect with -m 'not gpu'.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


# --- spreading the suite over every GPU ---------------------------------------------------------
#
# `select_device_for_rank` gives a SINGLE process "the first visible device" -- device 0. That is
# right for one run and wrong for a parallel suite: under `-n 24` every serial CUDA test built its
# Context on device 0 while devices 1-8 sat idle. Only MPI ladders spread, because `local_rank`
# hands rank i device i. The symptom was device 0 pinned near its memory ceiling while the rest of
# the machine was free, and at least one example test skipping itself with "this machine is too
# loaded to demonstrate the resume in 45 s".
#
# CUDA_VISIBLE_DEVICES is the lever, and it RENUMBERS: a worker that can see only physical device 5
# calls it device 0, so "the first visible device" becomes the device this worker was given. The
# runtime needs no change -- it already asks the driver what is visible.
#
# Two classes, because they want opposite things:
#
#   one device   an ordinary CUDA test. One worker, one GPU, round-robin over all nine.
#   many devices a REMD ladder or an AIS launch, which runs one rank per state and needs at least
#                as many visible devices as ranks. These are kept OFF DEVICE 0: it is the RTX
#                A5000 and the others are RTX 3080s, so a ladder spanning it would give one rung
#                different throughput from the rest.
#
#: Modules whose tests launch more than one rank. Listed here, in one auditable place, rather than
#: marked across a dozen files -- and rather than guessed from the node id, which would silently
#: stop matching the first time a file is renamed.
MULTI_RANK_MODULES = frozenset({
    "test_cv_mpi_cuda_ais.py", "test_cv_mpi_cuda_lanes.py", "test_cv_mpi_cuda_rrest2.py",
    "test_md_run_mpi_gpu.py", "test_rrest2_cuda_smoke.py",
    "test_examples_getting_started.py", "test_mpi_fail_closed.py",
    "test_driver_fail_closed.py", "test_regression_preflight_task.py",
    "test_runtime_contract_matrix.py", "test_own_replica_exchange.py",
    "test_rest2_equilibration_per_tau_cuda.py",
    "test_hprest2_gpu_evidence.py",
})

#: Modules that must see the machine EXACTLY as it is, and get no assignment at all.
#:
#: `test_cuda_coverage_matrix.py` is the record of what ran on what. It asks the DRIVER what
#: hardware exists -- `nvidia-smi` reports every physical GPU and ignores CUDA_VISIBLE_DEVICES --
#: and then both places a run on the last device BY PHYSICAL INDEX and asserts that an N-rank
#: ladder occupies N DISTINCT devices. Either kind of assignment breaks it, in opposite ways:
#: hiding device 0 left nine devices in its table and eight visible, so `--device 8` came back
#: "Illegal value for DeviceIndex: 8"; giving it one device instead put every rank of every
#: ladder on that one, and it reported "ranks shared devices: ['0', '0', '0', '0']". Both times
#: the test was right and the assignment was wrong.
UNASSIGNED_MODULES = frozenset({"test_cuda_coverage_matrix.py"})

#: Physical device the ladders must not touch. See above.
RESERVED_FOR_SERIAL_ONLY = 0


def _worker_number() -> int:
    """This xdist worker's index, or 0 when the suite runs in one process."""
    name = os.environ.get("PYTEST_XDIST_WORKER", "")
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def _machine_devices():
    """Every physical CUDA device, as the driver reports it. Asked ONCE and remembered.

    Snapshotted before any assignment below, and never re-read from the environment: this hook
    WRITES CUDA_VISIBLE_DEVICES, so a later read would see one worker's own slice and shrink the
    pool on every subsequent test until every worker believed the machine had a single GPU.
    """
    global _MACHINE_DEVICES
    if _MACHINE_DEVICES is None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
            _MACHINE_DEVICES = ([line.strip() for line in out.stdout.splitlines() if line.strip()]
                                if out.returncode == 0 else [])
        except (OSError, subprocess.SubprocessError):
            _MACHINE_DEVICES = []
    return _MACHINE_DEVICES


#: Filled by the first call to `_machine_devices`.
_MACHINE_DEVICES = None

#: What CUDA_VISIBLE_DEVICES held before the suite touched it. A value set from outside is an
#: instruction -- someone confining this run to particular cards meant it -- so the assignment
#: below stands down entirely rather than widening it.
_INHERITED_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


def pytest_runtest_setup(item):
    """Give this test its own slice of the machine, before it or any subprocess it spawns starts.

    Set per TEST rather than per worker: a worker runs both classes over its lifetime, so a single
    assignment at startup would either starve the ladders of devices or keep the ordinary tests
    off device 0 for no reason.

    RESET FIRST, EVERY TEST. The assignment is written into this worker's `os.environ` and nothing
    took it back, so a test that received no assignment -- a non-GPU test, or a module in
    `UNASSIGNED_MODULES` -- inherited whatever the previous test on the same worker had been given.
    `test_cuda_coverage_matrix.py`, the one module that must see the machine as it is, therefore
    saw ONE card whenever an ordinary CUDA test had run before it on that worker: its four-state
    ladder reported "4 states need 4 devices; 1 visible" on a machine with nine free GPUs, and
    before its guard counted visible devices it ran all four ranks on that one card instead.
    Which tests passed depended on how xdist distributed them.
    """
    if _INHERITED_VISIBLE_DEVICES is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _INHERITED_VISIBLE_DEVICES

    if "gpu" not in item.keywords or _INHERITED_VISIBLE_DEVICES:
        return

    module = Path(str(item.fspath)).name
    if module in UNASSIGNED_MODULES:
        return

    devices = _machine_devices()
    if len(devices) <= 1:
        return

    worker = _worker_number()
    if module in MULTI_RANK_MODULES:
        pool = [d for d in devices if d != str(RESERVED_FOR_SERIAL_ONLY)]
        if not pool:
            return
        # Rotated per worker so two workers running ladders at once do not both start at device 1.
        start = worker % len(pool)
        order = pool[start:] + pool[:start]
    else:
        order = [devices[worker % len(devices)]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(order)


# --- a tiny complete REST2 ladder, shared by the end-to-end and extension suites ----------------
#
# Two states, alanine dipeptide in vacuum, CPU, a few seconds. Small enough to be a unit test and
# real enough to execute the driver's loop, which nothing that reads source can do.

TAUS = [0.0, 0.3]
EXCHANGES = 6


PROTOCOL = f"""
from md_tools.remd.facade import REST2Protocol

protocol = REST2Protocol(
    tau={TAUS!r},
    temperature_k=300.0,
    timestep_fs=1.0,
    exchange_interval_ps=0.05,
    whole_output_interval_ps=0.05,
    solute_output_interval_ps=0.05,
    checkpoint_interval_ps=0.15,
    number_of_exchanges={EXCHANGES},
    platform="CPU",
)
"""


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """Alanine dipeptide in vacuum, serialised the way a generated stage would leave it."""
    from openmm import XmlSerializer, unit
    from openmm.app import ForceField, PDBFile

    work = tmp_path_factory.mktemp("grouped")
    pdb = PDBFile(str(ALA_PDB))
    field = ForceField("amber14-all.xml")
    system = field.createSystem(pdb.topology, constraints=None, removeCMMotion=False)

    (work / "system.xml").write_text(XmlSerializer.serialize(system), encoding="utf-8")
    with open(work / "topology.pdb", "w") as handle:
        PDBFile.writeFile(pdb.topology, pdb.positions, handle)

    # The coordinates a stage hands on are a serialised State, not a PDB: they carry velocities,
    # which a continuation needs and a PDB cannot hold.
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import Simulation

    simulation = Simulation(pdb.topology, system,
                            LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond,
                                                     1.0 * unit.femtosecond),
                            Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=50)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260831)
    opened = simulation.context.getState(getPositions=True, getVelocities=True)
    (work / "coordinates.xml").write_text(XmlSerializer.serialize(opened), encoding="utf-8")
    (work / "protocol.py").write_text(PROTOCOL, encoding="utf-8")

    lines = []
    for index in range(len(TAUS)):
        lines.append(f"-i protocol.py -p topology.pdb -s system.xml -c coordinates.xml "
                     f"--group-index {index}")
    (work / "ladder.group").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return work, pdb.topology.getNumAtoms(), unit


