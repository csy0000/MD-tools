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
    return _engine_ladder_and_bundle(tmp_path_factory.mktemp("rest2reference"),
                                     equilibration_steps=0)


@pytest.fixture(scope="module")
def equilibrated_ladder(tmp_path_factory):
    """The same ladder with per-state equilibration: each rung relaxes under ITS OWN Hamiltonian,
    from the shared equilibrated start, before the first exchange."""
    return _engine_ladder_and_bundle(tmp_path_factory.mktemp("rest2reference-eq"),
                                     equilibration_steps=EQUILIBRATION_STEPS)


@pytest.fixture(scope="module")
def per_tau_ladder(tmp_path_factory):
    """`rest2.equilibration_per_tau`: every rung runs the equilibration stages under its own tau
    from `min.xml`, then `equilibration_steps`, then exchanges."""
    return _engine_ladder_and_bundle(tmp_path_factory.mktemp("rest2reference-per-tau"),
                                     equilibration_steps=EQUILIBRATION_STEPS, per_tau=True)


#: Long enough that a hot rung relaxed for this many steps is clearly not the shared start.
EQUILIBRATION_STEPS = 200


def _engine_ladder_and_bundle(root, *, equilibration_steps, per_tau=False):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if subprocess.run(["which", "mpirun"], capture_output=True).returncode != 0:
        pytest.skip("no mpirun on PATH; a ladder needs one rank per rung")

    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0

    # Per-tau: two per-rung stages (restrained, then free), and a real minimisation, since the
    # ladder then starts from `min.xml` rather than from an equilibrated state.
    stages = ("stages: {minimization_iterations: 50, restrained_nvt_steps: 100, "
              "restrained_npt_steps: 0, unrestrained_npt_steps: 100, production_steps: 500}\n"
              if per_tau else
              "stages: {minimization_iterations: 0, restrained_nvt_steps: 100, "
              "restrained_npt_steps: 0, unrestrained_npt_steps: 0, production_steps: 500}\n")
    (root / "REST2.config").write_text(
        "protocol: REST2\nsolvent: implicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 7}\n"
        + stages +
        "reporting: {crd_printout_solute: 50, info_printout: 50, checkpoint_printout: 500}\n"
        f"rest2: {{number_of_replicas: {RUNGS}, tau_max: 0.5, "
        f"equilibration_steps: {equilibration_steps}, "
        + ("equilibration_per_tau: true, " if per_tau else "") +
        f"exchange_interval_steps: {INTERVAL}, number_of_exchanges: {EXCHANGES}, "
        "state_trajectory: true, rem_log: true, neighbour_acceptance_report: true}\n",
        encoding="utf-8")
    # A ladder integrates SAVED scaled states (0.5.4): `build-md` refuses to generate one until
    # `build/REST2/` exists, as `md-openmm build-top --rest2-scaler` writes it.
    from .conftest import make_states_for

    make_states_for(root, root / "REST2.config")
    assert subprocess.run(CLI + ["build-md", "-odir", "./run-run1", "--config",
                                 str(root / "REST2.config")],
                          cwd=root, capture_output=True, text=True, timeout=600).returncode == 0

    run = root / "run-run1"
    # The tau = 0 chain: minimisation alone under per-tau equilibration (implicit solvent), the
    # restrained NVT stage otherwise. Either way its end state is `eq.xml`, the ladder's `-c`.
    # THE PREPARATION STAGE WRITES INTO ITS OWN `-odir`, never the run root.
    #
    # `-odir .` made `md-run` compare the shared input against the RUN's `resolved.config`, which
    # says `protocol: REST2` -- while `min.in` and `eq_1.in` deliberately carry no protocol at
    # all, so they resolve to the schema default and the comparison refused with
    # `protocol: was 'REST2', now 'cMD'`. That is the cost of method-neutral preparation inputs,
    # and the answer is the one `run.sh` uses: minimisation into `../min`, equilibration into
    # `eq/`, each beside its own declaration. The explicit names move with it so the ladder's
    # `-c` still finds this state.
    #
    # THE END STATE IS NAMED WHAT THE GROUP FILE CONTINUES FROM. A ladder reads its inputs only from
    # `remd_groupfile.1` (0.5.4), whose lines name `-c eq/eq_3.xml`, or `-c ../min/min.xml` under
    # per-tau equilibration; the shortened chain here writes that file directly.
    stage_odir, start = ("../min", "../min/min.xml") if per_tau else ("eq", "eq/eq_3.xml")
    equilibration = subprocess.run(
        CLI + ["md-run", "-i", "../input/min.in" if per_tau else "../input/eq_1.in",
               "-p", "../build/built.pdb", "-s", "../build/built.xml",
               "-r", start, "-chk", f"{stage_odir}/eq.chk",
               "-o", f"{stage_odir}/eq.out", "-log", f"{stage_odir}/eq.log",
               "-odir", stage_odir, "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=1800, env=ONE_THREAD)
    assert equilibration.returncode == 0, equilibration.stdout[-3000:] + equilibration.stderr[-3000:]

    done = subprocess.run(
        ["mpirun", "-n", str(RUNGS), *CLI, "md-run", "-ng", str(RUNGS), "-i", "../input/REST2.in",
         "-p", "../build/built.pdb", "--groupfile", "remd_groupfile.1", "-x", "REST2.nc",
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
    assert f"all {RUNGS} rungs rebuild identically from the built System" in done.stdout, done.stdout


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
    names = {p.name for p in inputs.iterdir()}
    # `eq_1.in`, not `eq_nvt_posres.in`: the bundle keeps the name of the SHARED file the stage
    # read, so it records which input was read rather than which stage read it.
    assert {"eq_1.in", "REST2.in", "built.xml", "built.pdb", "build_system.py",
            "build_settings.json", "structure.leap"} <= names
    assert (inputs / "REST2.in").read_bytes() == (root / "input" / "REST2.in").read_bytes()
    readme = (inputs / "README.md").read_text(encoding="utf-8")
    # A 0.5.4 ladder reads -s (and -c) only from its group file, so the command names the group
    # file and no System; the states are made by the scaler, whose command route 4 now carries.
    ladder_line = next(line for line in readme.splitlines()
                       if f"mpirun -n {RUNGS} md-openmm md-run -ng {RUNGS}" in line)
    assert "-i input/REST2.in -p input/built.pdb --groupfile remd_groupfile.1" in ladder_line
    assert " -s " not in ladder_line and " -c " not in ladder_line, ladder_line
    assert ("md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb "
            "--config input/scaler.config") in readme, readme
    assert (inputs / "scaler.config").is_file()
    assert "python verify_rungs.py" in readme
    assert str(root) not in readme


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
    _assert_same_mapping(run, bundle)


def test_a_ladder_that_equilibrates_each_state_is_reproduced_too(equilibrated_ladder):
    """`equilibration_steps` relaxes every rung under its own Hamiltonian before the first exchange.

    The runner used to skip it, so a bundle of such a ladder started its hot rungs from the shared
    start instead of from where the engine's equilibration left them -- a different run that still
    completes and still reports a plausible acceptance rate. Only ladders with the setting at 0,
    the default and the only case the first test covered, were reproduced.
    """
    run, bundle, _manifest = equilibrated_ladder
    (bundle / "_blocked.py").write_text(
        _blocked("run.py", "--exchanges", str(EXCHANGES), "--platform", "CPU"), encoding="utf-8")
    done = subprocess.run([sys.executable, "_blocked.py"], cwd=bundle, capture_output=True,
                          text=True, timeout=3600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    _assert_same_mapping(run, bundle)


def test_a_ladder_that_equilibrates_every_rung_under_its_own_tau_is_reproduced_too(per_tau_ladder):
    """`rest2.equilibration_per_tau`: the bundle runs the same per-rung stages with the same code.

    Every rung starts its exchanges from where ITS OWN restrained and free stages left it, then
    from `equilibration_steps` on top. A runner that skipped the stages, seeded them differently,
    restrained different atoms or set the restraint before the configuration would start the hot
    rungs somewhere else and diverge within the first few close exchanges.
    """
    import json

    run, bundle, manifest = per_tau_ladder
    assert "rung_equilibration.py" in manifest["vendored"]
    settings = json.loads((bundle / "settings.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in settings["per_tau_equilibration"]] == ["eq_nvt_posres",
                                                                      "eq_nvt_free"]
    assert (run / "per_tau_equilibration.json").is_file()
    (bundle / "_blocked.py").write_text(
        _blocked("run.py", "--exchanges", str(EXCHANGES), "--platform", "CPU"), encoding="utf-8")
    done = subprocess.run([sys.executable, "_blocked.py"], cwd=bundle, capture_output=True,
                          text=True, timeout=3600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "# per-tau equilibration" in done.stdout, done.stdout[-2000:]
    _assert_same_mapping(run, bundle)


def test_a_ladder_without_per_tau_equilibration_bundles_an_empty_plan(ladder):
    """Off is recorded as off, and the runner then does exactly what it did before."""
    import json

    _run, bundle, _manifest = ladder
    assert json.loads((bundle / "settings.json").read_text(
        encoding="utf-8"))["per_tau_equilibration"] == []


def _assert_same_mapping(run, bundle):
    """The engine's state-to-walker mapping after every exchange, against the bundle's."""
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
