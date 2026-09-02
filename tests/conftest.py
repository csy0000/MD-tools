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


