"""A REST2 bundle must run without md_tools, and must reproduce the ladder exchange for exchange.

PLATFORM_POLICY_EXEMPTION: a four-rung ALA ladder, ten exchanges of fifty steps, on the CPU. What
is under test is whether the exported bundle makes the SAME decisions as the engine, which is
platform-independent.

WHY THE DECISIONS AND NOT THE ACCEPTANCE RATE

    A runner that swaps the wrong pairs still produces a plausible acceptance rate. The quantity
    that cannot be accidentally right is the state-to-walker mapping after every exchange:
    matching it requires the per-rung integrator seeds (`seed + 977*rung`), all four reduced
    potentials, the odd/even sweep phase, the acceptance criterion and the RNG substream to be
    identical. Any one of them wrong and the mapping diverges within two or three exchanges.

    That is the check the cMD export did not have, and it is why the cMD export shipped carrying
    the BUILD System instead of the integrated one -- its first test compared the bundle against
    a Context built in the test file, which shared all of the bundle's assumptions.

WHY BYTE-IDENTITY IS ASSERTED AS WELL

    The bundle does not reimplement the exchange; it carries md_tools' own modules, copied. So
    the strongest available statement is not "these agree" but "these are the same bytes", and
    that is what makes drift impossible rather than merely detectable.
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

ONE_THREAD = {**os.environ, "OPENMM_CPU_THREADS": "1"}
RUNGS, EXCHANGES, INTERVAL = 4, 10, 50

#: ONE worker for this module. Its module-scoped fixture builds a system and runs
#: dynamics; scattered across workers it is built once per worker that draws a test.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("reference-export-rest2")]


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """A real four-rung ladder through the engine, and the bundle exported from it."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if subprocess.run(["which", "mpirun"], capture_output=True).returncode != 0:
        pytest.skip("no mpirun on PATH; a ladder needs one rank per rung")

    root = tmp_path_factory.mktemp("rest2reference")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0

    (root / "REST2.config").write_text(
        "protocol: REST2\nsolvent: implicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 7}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 100, restrained_npt_steps: 0,"
        " unrestrained_npt_steps: 0, production_steps: 500}\n"
        "reporting: {crd_printout_solute: 50, info_printout: 50, checkpoint_printout: 500}\n"
        f"rest2: {{number_of_replicas: {RUNGS}, tau_max: 0.5, "
        f"exchange_interval_steps: {INTERVAL}, number_of_exchanges: {EXCHANGES}, "
        "state_trajectory: true, rem_log: true, neighbour_acceptance_report: true}\n",
        encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", "./run", "--config",
                                 str(root / "REST2.config")],
                          cwd=root, capture_output=True, text=True, timeout=600).returncode == 0

    run = root / "run"
    equilibration = subprocess.run(
        CLI + ["md-run", "-i", "eq_nvt_posres.in", "-p", "../built.pdb", "-s", "../built.xml",
               "-r", "eq.xml", "-chk", "eq.chk", "-o", "eq.out", "-log", "eq.log",
               "-odir", ".", "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=1800, env=ONE_THREAD)
    assert equilibration.returncode == 0, equilibration.stdout[-3000:] + equilibration.stderr[-3000:]

    done = subprocess.run(
        ["mpirun", "-n", str(RUNGS), *CLI, "md-run", "-ng", str(RUNGS), "-i", "REST2.in",
         "-p", "../built.pdb", "-s", "../built.xml", "-c", "eq.xml", "-x", "REST2.nc",
         "-r", "restart.json", "-o", "REST2.out", "-log", "REST2.log", "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=3600, env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    from md_tools.reference import export_rest2_reference

    bundle = root / "bundle"
    manifest = export_rest2_reference(run, bundle, stage="REST2")
    return run, bundle, manifest


def test_the_vendored_modules_are_byte_identical_to_the_packages_own(ladder):
    """Not "equivalent": the same bytes. Drift is then impossible rather than detectable."""
    _run, bundle, manifest = ladder
    source = Path(__import__("md_tools.remd", fromlist=["__file__"]).__file__).parent
    for name in manifest["vendored"]:
        mine = hashlib.sha256((bundle / "ladder" / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((source / name).read_bytes()).hexdigest()
        assert mine == theirs, f"ladder/{name} is not md_tools/remd/{name}"


def test_the_bundle_names_the_engine_that_ran_not_the_one_that_exported_it(ladder, tmp_path,
                                                                            monkeypatch):
    """Two commits, two questions: which engine ran the ladder, and where ladder/ was copied from.

    A run that recorded no commit used to be given the EXPORTER's, so a bundle named a commit the
    ladder never ran on, beside the run's own version string and timestamps. The record is
    doctored here because in a checkout the two commits coincide and the conflation is invisible.
    """
    import copy

    from md_tools import __version__
    from md_tools.build.record import read_record, source_commit
    from md_tools.reference import rest2_export

    run, _bundle, _manifest = ladder
    recorded = read_record(run / "REST2.log")

    def export_with(commit, name):
        def doctored(run_dir, stage):
            record = copy.deepcopy(recorded)
            record.setdefault("environment", {})["md_tools_commit"] = commit
            return record
        monkeypatch.setattr(rest2_export, "_ladder_record", doctored)
        out = tmp_path / name
        return out, rest2_export.export_rest2_reference(run, out, stage="REST2")["provenance"]

    ran = "0" * 40                       # a commit this checkout is certainly not at
    out, provenance = export_with(ran, "recorded")
    assert provenance["md_tools_commit"] == ran
    assert provenance["ladder_modules_from"] == {"md_tools_version": __version__,
                                                 "md_tools_commit": source_commit()}
    assert ran not in (out / "ladder" / "__init__.py").read_text(encoding="utf-8")

    _out, provenance = export_with(None, "unrecorded")
    assert provenance["md_tools_commit"] is None, "a commit the run never recorded was invented"


def _blocked(script: str, *args: str) -> str:
    """A runner that raises on any md_tools import, then executes `script` with `args`."""
    return ("import sys, runpy\n"
            "class B:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'md_tools':\n"
            "            raise ImportError('bundle imported md_tools: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            f"sys.argv = [{script!r}, *{list(args)!r}]\n"
            f"runpy.run_path({script!r}, run_name='__main__')\n")


def test_the_bundled_scaling_module_is_the_packages_own(ladder):
    """The rung construction travels as the same bytes, like the exchange modules beside it."""
    _run, bundle, manifest = ladder
    source = Path(__import__("md_tools.rest2", fromlist=["__file__"]).__file__).parent
    name = manifest["scaling_module"]
    assert (bundle / "ladder" / name).read_bytes() == (source / name).read_bytes()


def test_every_rung_rebuilds_from_rung_zero_without_md_tools(ladder, tmp_path):
    """How the scaled Systems were derived, checked with OpenMM alone."""
    import shutil

    _run, bundle, _manifest = ladder
    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)
    (copy / "_blocked.py").write_text(_blocked("verify_rungs.py"), encoding="utf-8")
    done = subprocess.run([sys.executable, "_blocked.py"], cwd=copy, capture_output=True,
                          text=True, timeout=600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"all {RUNGS} rungs rebuild identically from rung 0" in done.stdout, done.stdout


def test_a_tampered_rung_is_named(ladder, tmp_path):
    """The check must be able to fail: change one torsion in one rung and it has to say which."""
    import shutil

    from openmm import PeriodicTorsionForce, XmlSerializer

    _run, bundle, _manifest = ladder
    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)
    path = copy / "system_rung2.xml"
    system = XmlSerializer.deserialize(path.read_text(encoding="utf-8"))
    torsions = next(system.getForce(i) for i in range(system.getNumForces())
                    if isinstance(system.getForce(i), PeriodicTorsionForce))
    parameters = list(torsions.getTorsionParameters(1))
    parameters[6] = parameters[6] * 1.01
    torsions.setTorsionParameters(1, *parameters)
    path.write_text(XmlSerializer.serialize(system), encoding="utf-8")
    done = subprocess.run([sys.executable, "verify_rungs.py"], cwd=copy, capture_output=True,
                          text=True, timeout=600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "rung 2" in done.stdout and "DIFFERS" in done.stdout, done.stdout
    assert "rung(s) [2]" in done.stdout, done.stdout


def test_the_derivation_names_the_solute_and_the_omega_bonds(ladder):
    """Every input `build_scaled_system` takes besides tau, so a reader can execute the rules."""
    import json

    _run, bundle, _manifest = ladder
    derivation = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))["derivation"]
    assert derivation["solute_atom_indices"] == list(range(22))      # capped ALA, implicit
    assert len(derivation["excluded_bonds"]) == 2                     # ACE-ALA and ALA-NME amides
    assert derivation["check"] == "python verify_rungs.py"


def test_input_holds_what_the_user_supplied_each_one_proven(ladder):
    """The structure by digest, build-top's configuration by resolution, build-md's by contract."""
    import hashlib
    import json

    run, bundle, _manifest = ladder
    root = run.parent
    inputs = bundle / "input"
    assert (inputs / "ALA.pdb").read_bytes() == ALA.read_bytes()
    assert (inputs / "build-top.config").read_bytes() == (root / "sys.config").read_bytes()
    assert (inputs / "build-md.config").read_bytes() == (run / "resolved.config").read_bytes()
    assert (inputs / "README.md").is_file()
    recorded = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))["inputs"]
    assert recorded["structure"]["sha256"] == hashlib.sha256(ALA.read_bytes()).hexdigest()
    assert recorded["build_top_config"]["file"] == "input/build-top.config"
    assert "resolves to the build-top record" in recorded["build_top_config"]["verified"]


def test_the_bundle_never_imports_md_tools(ladder):
    """Enforced by blocking the import, not by reading the source."""
    _run, bundle, _manifest = ladder
    blocker = bundle / "_blocked.py"
    blocker.write_text(
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'md_tools':\n"
        "            raise ImportError('bundle imported md_tools: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.argv = ['run.py', '--exchanges', '{EXCHANGES}', '--platform', 'CPU']\n"
        "runpy.run_path('run.py', run_name='__main__')\n", encoding="utf-8")
    done = subprocess.run([sys.executable, str(blocker)], cwd=bundle, capture_output=True,
                          text=True, timeout=3600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "imported md_tools" not in done.stderr


def test_the_bundle_reproduces_the_engines_ladder_exchange_for_exchange(ladder):
    """The assertion this exporter exists for. See the module docstring for why it is the mapping."""
    run, bundle, _manifest = ladder
    assert (bundle / "exchange.csv").is_file(), (
        "run the bundle first; test_the_bundle_never_imports_md_tools does that")

    engine = {}
    with (run / "exchange.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            engine.setdefault(int(row["exchange"]), {})[int(row["state"])] = int(row["walker"])
    engine_mapping = {k: [v[s] for s in sorted(v)] for k, v in engine.items()}

    mapping = list(range(RUNGS))
    mine = {}
    with (bundle / "exchange.csv").open(encoding="utf-8") as handle:
        proposals = {}
        for row in csv.DictReader(handle):
            proposals.setdefault(int(row["exchange"]), []).append(
                (int(row["state_i"]), int(row["state_j"]), int(row["accepted"])))
    for attempt in sorted(proposals):
        for state_i, state_j, accepted in proposals[attempt]:
            if accepted:
                mapping[state_i], mapping[state_j] = mapping[state_j], mapping[state_i]
        mine[attempt] = list(mapping)

    assert engine_mapping, "the engine wrote no exchange record"
    differing = [k for k in sorted(engine_mapping) if engine_mapping[k] != mine.get(k)]
    assert not differing, (
        "the bundle and the engine disagree about which walkers occupy which states after "
        f"exchange(s) {differing}:\n"
        + "\n".join(f"  {k}: engine {engine_mapping[k]}  bundle {mine.get(k)}"
                    for k in differing[:5]))
