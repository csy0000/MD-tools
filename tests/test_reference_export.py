"""A reference bundle must run without this package, and must agree with the engine.

PLATFORM_POLICY_EXEMPTION: a few hundred steps of implicit ALA on whatever platform is available.
What is under test is whether the exported script builds the SAME Context as the engine, which is
platform-independent; the CUDA behaviour of the stage itself is the GPU-marked files.

WHY AN EQUIVALENCE TEST AND NOT JUST A SMOKE TEST

    The exported `run.py` is a SECOND implementation of what `md_tools.md` does -- the integrator,
    the reporters, the loop. Second implementations drift, and this one drifts silently: a bundle
    that builds a subtly different Context still runs, still writes a trajectory, and is wrong in
    a way nobody notices until it is compared against the data it claims to reproduce.

    So the test does that comparison. Same System, same seed, same platform: OpenMM is
    deterministic, so the two must agree to the last decimal, not merely look similar.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A real, finished cMD run -- short, implicit, CPU -- and the bundle exported from it."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("reference")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0

    (root / "cMD.config").write_text(
        "protocol: cMD\nsolvent: implicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 7}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 0, restrained_npt_steps: 0,"
        " unrestrained_npt_steps: 0, production_steps: 500}\n"
        "reporting: {crd_printout_solute: 100, info_printout: 100, checkpoint_printout: 500}\n",
        encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", "./run", "--config", str(root / "cMD.config")],
                          cwd=root, capture_output=True, text=True, timeout=600).returncode == 0
    done = subprocess.run(
        CLI + ["md-run", "-i", "cMD.in", "-p", "../built.pdb", "-s", "../built.xml",
               "-odir", ".", "--cpu"],
        cwd=root / "run", capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    from md_tools.reference import export_reference

    bundle = root / "bundle"
    export_reference(root / "run", bundle)
    return root, bundle


def test_the_bundle_holds_the_system_that_was_integrated(finished):
    """Not a rebuilt equivalent: the same bytes, checked against the digest the run recorded."""
    import hashlib

    root, bundle = finished
    for name, original in (("system.xml", "built.xml"), ("topology.pdb", "built.pdb")):
        mine = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((root / original).read_bytes()).hexdigest()
        assert mine == theirs, f"{name} is not the file the run integrated"


def test_the_bundle_never_imports_md_tools(finished):
    """The point of the bundle. Enforced by blocking the import, not by reading the source."""
    _root, bundle = finished
    blocker = bundle / "_blocked.py"
    blocker.write_text(
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'md_tools':\n"
        "            raise ImportError('bundle imported md_tools: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "sys.argv = ['run.py', '--steps', '100', '--platform', 'CPU']\n"
        "runpy.run_path('run.py', run_name='__main__')\n", encoding="utf-8")
    done = subprocess.run([sys.executable, str(blocker)], cwd=bundle,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "imported md_tools" not in done.stderr


def test_the_exported_runner_reproduces_the_engine_exactly(finished):
    """Same System, seed, platform and ONE THREAD -- so the positions must match exactly.

    This is what stops the exported script drifting from the engine. A bundle that builds a
    slightly different Context runs perfectly and reproduces nothing, and only a comparison
    against the engine shows it.

    `Threads: 1` is what makes "exactly" available rather than approximate. Multi-threaded CPU
    sums forces in whatever order threads finish, so two identically-built Contexts still
    diverge -- measured at 2.3e-08 nm after 200 steps, which is rounding amplified by Langevin
    dynamics and says nothing about whether the setups agree. Pinning the thread count removes
    the one source of difference that is not this bundle's doing, so the assertion can be
    equality rather than a tolerance nobody could justify.

    This is what stops the exported script drifting from the engine. A bundle that builds a
    slightly different Context runs perfectly and reproduces nothing, and only a comparison
    against the engine shows it.
    """
    import numpy as np
    from openmm import Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    _root, bundle = finished
    settings = json.loads((bundle / "settings.json").read_text(encoding="utf-8"))

    # See the docstring: one thread, so the comparison can be exact.
    PROPS = {"Threads": "1"}

    def positions(build):
        simulation = build()
        simulation.context.setVelocitiesToTemperature(
            settings["temperature_K"] * unit.kelvin, settings["seed"])
        simulation.step(200)
        return simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value

    def by_hand():
        from openmm import LangevinMiddleIntegrator

        pdb = PDBFile(str(bundle / "topology.pdb"))
        system = XmlSerializer.deserialize((bundle / "system.xml").read_text(encoding="utf-8"))
        integrator = LangevinMiddleIntegrator(
            settings["temperature_K"] * unit.kelvin,
            settings["friction_per_ps"] / unit.picosecond,
            settings["timestep_fs"] * unit.femtosecond)
        integrator.setRandomNumberSeed(settings["seed"])
        simulation = Simulation(pdb.topology, system, integrator,
                                Platform.getPlatformByName("CPU"), PROPS)
        simulation.context.setPositions(pdb.positions)
        return simulation

    sys.path.insert(0, str(bundle))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_bundle_run", bundle / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mine = positions(lambda: module.build("CPU", PROPS))
    finally:
        sys.path.remove(str(bundle))
    theirs = positions(by_hand)

    assert np.array_equal(mine, theirs), (
        "the exported runner and a hand-built Context diverge from the same System, seed and "
        "platform, so the bundle does not reproduce what it describes")
