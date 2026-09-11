"""A reference bundle must run without this package, and must reproduce the engine's own run.

PLATFORM_POLICY_EXEMPTION: a few hundred steps of implicit ALA on the CPU. What is under test is
whether the exported script integrates the SAME Hamiltonian from the SAME state as the engine,
which is platform-independent; the CUDA behaviour of the stage itself is the GPU-marked files.

WHY THE COMPARISON IS AGAINST THE ENGINE AND NOT AGAINST A CONTEXT BUILT HERE

    The exported `run.py` is a SECOND implementation of what `md_tools.md` does -- the System
    preparation, the integrator, the seed, the starting state, the loop. Second implementations
    drift, and this one drifts silently: a bundle that builds a different Context still runs,
    still writes a trajectory, and is wrong in a way nobody notices until it is compared against
    the data it claims to reproduce.

    An earlier version of this test compared the bundle against a Context built in the test file,
    which shared the bundle's assumptions and therefore agreed with all three of its mistakes:
    it exported the BUILD System instead of the scaled one the stage integrates, used the config
    seed instead of the derived one, and started from the topology's coordinates instead of the
    state the stage continued from. At tau = 0.5 the first of those alone is a different
    Hamiltonian. Only the engine's own output is an independent answer.

    `OPENMM_CPU_THREADS=1` on both sides is what makes "reproduce" mean equality rather than a
    tolerance nobody could justify: multi-threaded CPU sums forces in whatever order threads
    finish, and two identically-built Contexts diverge by ~1e-08 nm over a couple of hundred
    steps for reasons that have nothing to do with this bundle.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

#: Both sides of the comparison, single-threaded. See the module docstring.
ONE_THREAD = {**os.environ, "OPENMM_CPU_THREADS": "1"}

PRODUCTION_STEPS = 400

#: ONE worker for this module. Its module-scoped fixture builds a system and runs
#: dynamics; scattered across workers it is built once per worker that draws a test.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("reference-export")]


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A real, finished, SCALED cMD run that continues from an equilibration state.

    tau is deliberately non-zero and there is a real `-c`: those are the two things the export
    has to get right and the two an unscaled first-stage run cannot exercise.
    """
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
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 7, tau: 0.5}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 200, restrained_npt_steps: 0,"
        f" unrestrained_npt_steps: 0, production_steps: {PRODUCTION_STEPS}}}\n"
        "reporting: {crd_printout_solute: 100, info_printout: 100, checkpoint_printout: 1000}\n",
        encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", "./run", "--config", str(root / "cMD.config")],
                          cwd=root, capture_output=True, text=True, timeout=600).returncode == 0

    run = root / "run"
    for stage, parent in (("eq_nvt_posres", None), ("cMD", "eq_nvt_posres.xml")):
        argv = CLI + ["md-run", "-i", f"{stage}.in", "-p", "../built.pdb", "-s", "../built.xml",
                      "-r", f"{stage}.xml", "-chk", f"{stage}.chk",
                      "-o", f"{stage}.out", "-log", f"{stage}.log", "-odir", ".", "--cpu"]
        if parent:
            argv += ["-c", parent]
        done = subprocess.run(argv, cwd=run, capture_output=True, text=True, timeout=1800,
                              env=ONE_THREAD)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    from md_tools.reference import export_reference

    bundle = root / "bundle"
    manifest = export_reference(run, bundle)
    return root, run, bundle, manifest


def test_input_holds_what_the_user_supplied_each_one_proven(finished):
    """The structure by digest, build-top's configuration by resolution, build-md's by contract."""
    import hashlib

    root, run, bundle, _manifest = finished
    inputs = bundle / "input"
    assert sorted(p.name for p in inputs.iterdir()) == [
        "ALA.pdb", "README.md", "build-md.config", "build-top.config", "build_settings.json",
        "build_system.py", "built.pdb", "built.xml", "cMD.in", "eq_nvt_posres.in",
        "structure.leap"]
    assert (inputs / "ALA.pdb").read_bytes() == ALA.read_bytes()
    assert (inputs / "build-top.config").read_bytes() == (root / "sys.config").read_bytes()
    assert (inputs / "build-md.config").read_bytes() == (run / "resolved.config").read_bytes()
    for name in ("built.xml", "built.pdb"):
        assert (inputs / name).read_bytes() == (root / name).read_bytes()
    for name in ("cMD.in", "eq_nvt_posres.in"):
        assert (inputs / name).read_bytes() == (run / name).read_bytes()
    recorded = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))["inputs"]
    assert recorded["structure"]["sha256"] == hashlib.sha256(ALA.read_bytes()).hexdigest()
    assert "resolves to the build-top record" in recorded["build_top_config"]["verified"]
    assert [entry["file"] for entry in recorded["stage_inputs"]] == [
        "input/eq_nvt_posres.in", "input/cMD.in"], "the stage inputs, in the order they ran"
    assert recorded["structure_origin"]["sequence"] == ["ACE", "ALA", "NME"]
    readme = (inputs / "README.md").read_text(encoding="utf-8")
    assert "md-openmm build-top -i input/ALA.pdb --config input/build-top.config" in readme
    assert ("md-openmm md-run -i input/eq_nvt_posres.in -p input/built.pdb -s input/built.xml"
            in readme)
    assert "md-openmm md-run -i input/cMD.in -p input/built.pdb -s input/built.xml" in readme
    assert "python input/build_system.py --out rebuilt" in readme
    assert "sequence `{ ACE ALA NME }`" in readme
    assert str(root) not in readme and sys.executable not in readme, \
        "a path from the machine the run was on reached the bundle"


def test_export_reference_works_from_inside_the_run_directory_with_a_dot(finished, tmp_path):
    """`-idata .` used to be refused: the built System is found "beside or above" the run
    directory, and a bare `.` has no parents to search."""
    _root, run, bundle, _manifest = finished
    done = subprocess.run(CLI + ["export-reference", "-idata", ".", "-odir",
                                 str(tmp_path / "dot-bundle")],
                          cwd=run, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    for name in ("system.xml", "topology.pdb", "start.xml", "input/built.xml"):
        assert (tmp_path / "dot-bundle" / name).read_bytes() == (bundle / name).read_bytes(), name


def test_the_system_rebuilds_from_the_structure_without_md_tools(finished, tmp_path):
    """Route 2 of input/README.md: openmm-env and the bundle, nothing of md-tools, same bytes."""
    import shutil

    _root, _run, bundle, _manifest = finished
    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)
    blocker = (
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'md_tools':\n"
        "            raise ImportError('bundle imported md_tools: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "sys.argv = ['input/build_system.py', '--out', 'rebuilt']\n"
        "runpy.run_path('input/build_system.py', run_name='__main__')\n")
    (copy / "_blocked.py").write_text(blocker, encoding="utf-8")
    done = subprocess.run([sys.executable, "_blocked.py"], cwd=copy, capture_output=True,
                          text=True, timeout=600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert (copy / "rebuilt" / "built.xml").read_bytes() == (copy / "input" / "built.xml").read_bytes()


def _copy_of_the_run(finished, tmp_path):
    """The finished tree, copied, so a test can damage it without touching the shared fixture."""
    import shutil

    root, _run, _bundle, _manifest = finished
    copy = tmp_path / "copy"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("bundle"))
    return copy


def test_a_configuration_edited_after_the_build_is_not_passed_off_as_the_one_used(finished,
                                                                                 tmp_path):
    """The campaign case: a file with the right name that no longer describes the build."""
    from md_tools.reference import export_reference

    copy = _copy_of_the_run(finished, tmp_path)
    # The build ran with absolute paths, so the copied record still names the fixture's own
    # `sys.config`. Point it at the copy's, then change the copy's: same name, same place the
    # record says, different content -- and the shared fixture is never touched.
    log = copy / "built.log"
    log.write_text(log.read_text(encoding="utf-8").replace(str(finished[0]), str(copy)),
                   encoding="utf-8")
    (copy / "sys.config").write_text("solvent:\n  model: OBC2\n", encoding="utf-8")
    manifest = export_reference(copy / "run", tmp_path / "bundle")
    inputs = tmp_path / "bundle" / "input"
    assert not (inputs / "build-top.config").exists()
    assert (inputs / "build-top.resolved.yaml").is_file()
    assert manifest["provenance"]["inputs"]["build_top_config"]["verified"] is False


def test_a_structure_that_is_not_the_one_recorded_refuses_before_writing(finished, tmp_path):
    """input/ must never hold a file that merely shares its name with the one used."""
    from md_tools.reference import export_reference

    copy = _copy_of_the_run(finished, tmp_path)
    log = copy / "built.log"
    text = log.read_text(encoding="utf-8")
    import hashlib

    real = hashlib.sha256(ALA.read_bytes()).hexdigest()
    assert real in text
    log.write_text(text.replace(real, "0" * 64), encoding="utf-8")
    target = tmp_path / "bundle"
    with pytest.raises(FileNotFoundError, match="the structure build-top read"):
        export_reference(copy / "run", target)
    assert not target.exists(), "a refused export wrote files"


def test_the_bundle_holds_the_system_the_stage_integrated_not_the_one_it_was_built_from(finished):
    """At tau = 0.5 the build System is a DIFFERENT Hamiltonian, and exporting it is the bug.

    Between `built.xml` and the Context the engine scales the solute, adds the positional
    restraint force and, for explicit solvent, the barostat. A bundle carrying `built.xml` would
    run happily and sample the unscaled ensemble.
    """
    root, run, bundle, _manifest = finished
    exported = (bundle / "system.xml").read_text(encoding="utf-8")
    assert exported != (root / "built.xml").read_text(encoding="utf-8"), \
        "the bundle carries the BUILD System; at tau > 0 that is not what was integrated"

    from md_tools.run.preflight import _prepare_stage, load_inputs
    from md_tools.build.record import read_record
    from openmm import XmlSerializer

    block = read_record(run / "cMD.log")["stage"]
    loaded = load_inputs(root / "built.pdb", root / "built.xml")
    prepared = _prepare_stage(loaded, stage=block, name=block["name"], where="test")
    assert exported == XmlSerializer.serialize(prepared["prepared_system"])


def test_the_bundle_starts_from_the_state_the_stage_continued_from(finished):
    """Not from the topology's coordinates, which are the BUILT structure.

    Starting there would re-run an unequilibrated system and present the result as a
    reproduction of a production segment.
    """
    import hashlib

    _root, run, bundle, _manifest = finished
    assert (bundle / "start.xml").is_file(), "the bundle has no starting state"
    assert hashlib.sha256((bundle / "start.xml").read_bytes()).hexdigest() == \
        hashlib.sha256((run / "eq_nvt_posres.xml").read_bytes()).hexdigest()


def test_the_exported_seed_is_the_one_the_integrator_actually_used(finished):
    """`derive_seed(config_seed, stage_name)`, not the number written in the config.

    The two differ, and using the config value gives a bundle that is reproducible, plausible,
    and not this run.
    """
    from md_tools.md._stages import derive_seed

    _root, _run, bundle, _manifest = finished
    settings = json.loads((bundle / "settings.json").read_text(encoding="utf-8"))
    assert settings["config_seed"] == 7
    assert settings["seed"] == derive_seed(7, "cMD")
    assert settings["seed"] != settings["config_seed"]


def test_the_bundle_never_imports_md_tools(finished):
    """The point of the bundle. Enforced by blocking the import, not by reading the source."""
    _root, _run, bundle, _manifest = finished
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


def test_the_bundle_reproduces_the_engines_own_final_state(finished):
    """The assertion the whole exporter exists for.

    The engine ran `PRODUCTION_STEPS` steps and serialised where it ended up. The bundle runs the
    same number of steps from the same start, and the two end in the SAME place -- same System,
    same seed, same platform, one thread each.
    """
    import numpy as np
    from openmm import XmlSerializer

    _root, run, bundle, _manifest = finished
    done = subprocess.run([sys.executable, "run.py", "--steps", str(PRODUCTION_STEPS),
                           "--platform", "CPU"],
                          cwd=bundle, capture_output=True, text=True, timeout=1800,
                          env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]

    def final(path):
        state = XmlSerializer.deserialize(Path(path).read_text(encoding="utf-8"))
        return state.getPositions(asNumpy=True)._value

    theirs = final(run / "cMD.xml")
    mine = final(bundle / "final.xml")

    assert np.array_equal(mine, theirs), (
        "the exported bundle and the engine end in different places from the same System, seed, "
        "start and platform, so the bundle does not reproduce the run it describes")


def test_a_ladder_record_is_refused_before_anything_is_written(finished, tmp_path):
    """A refusal that leaves files behind is not a refusal.

    Asked for `--stage REST2`, the exporter used to reach `block["name"]` on a record whose block
    is called `ladder` rather than `stage`, report the bare KeyError key -- `export-reference:
    'name'` -- and leave a directory holding a topology behind it. Someone reading that directory
    has a partial bundle and a message that names nothing.

    The record type is the discriminator and it is checked first: `md-stage:cMD` is exportable,
    `md-replica:REST2` is not, and the difference is the control flow the runner drives.
    """
    _root, run, _bundle, _manifest = finished

    ladder = tmp_path / "ladder"
    ladder.mkdir()
    (ladder / "REST2.log").write_text(
        (run / "cMD.log").read_text(encoding="utf-8")
        .replace("record_type: md-stage:cMD", "record_type: md-replica:REST2"),
        encoding="utf-8")

    from md_tools.reference import export_reference

    out = tmp_path / "bundle-that-must-not-appear"
    with pytest.raises(ValueError) as refusal:
        export_reference(ladder, out, stage="REST2")
    assert "md-replica:REST2" in str(refusal.value)
    assert not out.exists(), "the refusal created a directory"
