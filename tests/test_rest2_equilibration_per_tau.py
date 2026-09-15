"""`rest2.equilibration_per_tau`: the equilibration stages on every rung, under its own tau.

THE SETTING

    Off by default. When on, the tau = 0 chain stops at minimisation (implicit solvent) or after
    its NPT stages (explicit solvent, which fix the box every rung shares), and every rung -- tau = 0
    included -- then runs eq_nvt_posres, eq_nvt_posres_2 and eq_nvt_free under its own tau at
    fixed volume, then `equilibration_steps`, then the first exchange.

WHAT IS ASSERTED

    * Off leaves every generated script, `run.sh` and the `_protocol.py` helper byte-identical to
      what build-md wrote before the setting existed, and the scientific identity without the
      key; each `.in` and `resolved.config` gains exactly one line. A change to any of those is a
      change to every existing ladder directory, whose helpers are content-addressed.
    * Refused by name for a protocol with no ladder, and when there is nothing to run.
    * The per-rung stages ARE the fixed-volume stages a scaled run gets: the same stage dicts.
    * The driver runs them stage by stage across its rungs and stops between stages on an
      interruption, recording a phase that `--resume` refuses read-only.
    * A real implicit ladder leaves each rung at a state an independent reconstruction -- plain
      OpenMM, the documented seed formula and restraint -- reproduces bit for bit, distinct per
      rung; an explicit ladder keeps the box its NPT stages fixed on every rung.
PLATFORM_POLICY_EXEMPTION: the CPU ladders here compare the engine's per-tau end states with an
independent reconstruction bit for bit, which needs OPENMM_CPU_THREADS=1 on both sides. The CUDA
evidence, one rank per rung, is `test_rest2_equilibration_per_tau_cuda.py`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "md_tools"
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ONE_THREAD = {**os.environ, "OPENMM_CPU_THREADS": "1"}
RUNGS = 4
STAGES = ("eq_nvt_posres", "eq_nvt_posres_2", "eq_nvt_free")

#: One worker for the module-scoped ladders below; see tests/test_reference_export_rest2.py.
pytestmark = [pytest.mark.xdist_group("rest2-per-tau")]


def _resolve(tmp_path, document):
    from md_tools.build.md import resolve_md_config

    path = tmp_path / "per_tau.config"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return resolve_md_config(path)


def _build_md(root, document, odir="md_script"):
    path = root / f"{odir}.config"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", odir, "--config", str(path)], cwd=root,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return root / odir


def _documented_seed(base, *parts):
    """`derive_seed`, written out from its definition rather than imported."""
    value = int(base)
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = (value * 1000003 + byte) & 0xFFFFFFFF
    return (value % (2 ** 31 - 1)) or 1


# --- resolution -----------------------------------------------------------------------------------

def test_it_is_off_by_default_and_then_plans_nothing(tmp_path):
    from md_tools.build.md import per_tau_equilibration_stages

    resolved = _resolve(tmp_path, {"protocol": "REST2"})
    assert resolved["rest2"]["equilibration_per_tau"] is False
    assert per_tau_equilibration_stages(resolved) == []


@pytest.mark.parametrize("document", [
    {"protocol": "cMD"},
    {"protocol": "umbrella", "umbrella": {"file": "umbrella.yaml"},
     "collective_variables": {"file": "cv.yaml", "interval_steps": 10}},
    {"protocol": "AIS", "ais_source": {"trajectory": "source.nc"}},
], ids=["cMD", "umbrella", "AIS"])
def test_a_protocol_without_a_ladder_is_refused_by_name(tmp_path, document):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError) as refusal:
        _resolve(tmp_path, dict(document, rest2={"equilibration_per_tau": True}))
    message = str(refusal.value)
    assert "rest2.equilibration_per_tau" in message and document["protocol"] in message, message


def test_a_ladder_with_nothing_to_run_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="all 0"):
        _resolve(tmp_path, {"protocol": "REST2", "rest2": {"equilibration_per_tau": True},
                            "stages": {"restrained_nvt_steps": 0, "restrained_npt_steps": 0,
                                       "unrestrained_npt_steps": 0}})


def test_the_rung_stages_are_the_stages_a_scaled_run_gets(tmp_path):
    """"The same equilibration" is the same stage dicts, not a second description of them."""
    from md_tools.build.md import per_tau_equilibration_stages, stage_plan

    stages = {"restrained_nvt_steps": 300, "restrained_npt_steps": 200,
              "unrestrained_npt_steps": 100}
    dynamics = {"restraint_kcal_per_mol_A2": 2.5}
    ladder = _resolve(tmp_path, {"protocol": "REST2", "solvent": "explicit", "stages": stages,
                                 "dynamics": dynamics, "rest2": {"equilibration_per_tau": True}})
    scaled = _resolve(tmp_path, {"protocol": "cMD", "solvent": "explicit", "stages": stages,
                                 "dynamics": dict(dynamics, tau=0.3)})
    expected = [{"name": stage["name"], "steps": stage["steps"],
                 "restraint_kcal_per_mol_A2": stage["restraint_kcal_per_mol_A2"]}
                for stage in stage_plan(scaled) if stage["name"].startswith("eq_")]
    assert [stage["name"] for stage in expected] == list(STAGES)
    assert per_tau_equilibration_stages(ladder) == expected
    assert [stage["restraint_kcal_per_mol_A2"] for stage in expected] == [2.5, 2.5, 0.0]


def test_a_stage_of_zero_steps_is_left_out(tmp_path):
    from md_tools.build.md import per_tau_equilibration_stages

    resolved = _resolve(tmp_path, {"protocol": "rREST2", "rest2": {"equilibration_per_tau": True},
                                   "reservoir": {"enabled": True, "path": "reservoir.nc"},
                                   "stages": {"restrained_npt_steps": 0}})
    assert [stage["name"] for stage in per_tau_equilibration_stages(resolved)] == [
        "eq_nvt_posres", "eq_nvt_free"]


def test_the_stage_names_agree_between_the_resolver_and_the_runtime():
    from md_tools.build.md import PER_TAU_STAGE_NAMES as resolver
    from md_tools.remd.rung_equilibration import PER_TAU_STAGE_NAMES as runtime

    assert resolver == runtime == STAGES


# --- what build-md writes ---------------------------------------------------------------------------

def test_under_implicit_solvent_the_tau_zero_chain_is_minimisation_alone(tmp_path):
    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": "implicit",
                               "rest2": {"equilibration_per_tau": True}})
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["REST2.in", "REST2.py", "build-md.log", "min.in", "min.py", "resolved.config", "run.sh"])
    assert "-c min.xml" in (out / "run.sh").read_text(encoding="utf-8")
    assert "per-tau equilibration" in (out / "build-md.log").read_text(encoding="utf-8")


def test_under_explicit_solvent_the_npt_stages_still_fix_the_box_first(tmp_path):
    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": "explicit",
                               "rest2": {"equilibration_per_tau": True}})
    assert {"eq_nvt_posres.in", "eq_npt_posres.in", "eq_npt_free.in"} <= {
        p.name for p in out.iterdir()}
    assert "-c eq_npt_free.xml" in (out / "run.sh").read_text(encoding="utf-8")


def test_every_generated_input_resolves_back_to_its_resolved_config(tmp_path):
    from md_tools.build.md import resolve_md_config
    from md_tools.run.inputs import parse_run_input

    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": "implicit",
                               "rest2": {"equilibration_per_tau": True}})
    expected = resolve_md_config(out / "resolved.config")
    assert expected["rest2"]["equilibration_per_tau"] is True
    for path in sorted(out.glob("*.in")):
        parsed = parse_run_input(path)
        for block in ("rest2", "stages", "dynamics"):
            assert parsed.resolved[block] == expected[block], (path.name, block)


#: What `build-md` generates for a default REST2 ladder, by sha256. Originally captured BEFORE
#: `rest2.equilibration_per_tau` existed, which is what makes it evidence: with the setting off,
#: every script, `run.sh` and the protocol helper must still be these bytes, and each `.in` and
#: `resolved.config` these bytes once the one added line is taken out again.
#:
#: THESE DIGESTS MOVE WHEN THE SCHEMA GAINS A FIELD, and that is expected rather than a
#: regression: a generated `.in` and `resolved.config` list every resolved key, so one new field
#: rewrites eleven of the files below. `stages.number_of_segments` did exactly that -- the six
#: `.in`/`resolved.config` entries per solvent changed and the `.py` files and `run.sh` did not,
#: which is itself the shape a schema-only change should have.
#:
#: HOW TO REFRESH THEM HONESTLY. Do not just paste the new hashes: that absorbs any unintended
#: drift sitting beside the intended change. Strip the NEW field's line out of the freshly
#: generated text and check the PINNED digest comes back. If it does, that line is the only
#: difference and the snapshot can be updated. If it does not, something else moved as well, and
#: that is the thing to look at rather than to overwrite.
BEFORE = {
    "explicit": {
        "eq_npt_free.in": "7f069da94d44b80f072b3ed2080907a525a4bb1c4a62bfc448f79ef0f9ddf25a",
        "eq_npt_free.py": "6daed6d160528e34e730c67997fb015667e2a3fe5cf80677be6b3f1e68b3ccad",
        "eq_npt_posres.in": "81b550eb7e420aeffb77c47875f002cda7ae896d31364426f97bff1772812f55",
        "eq_npt_posres.py": "a9a19c6c5e8f839a7a51e81a1ec554f89655bd04d581bc0c1babaa6aa07e6562",
        "eq_nvt_posres.in": "1368cf3488975818bf3ec89f0baa3c5661cdefa5f78b209ee737745e1cf70e7c",
        "eq_nvt_posres.py": "b5a332209934c06cbc1fe47f8cb780933bd672cf13de18f2d76214bb6cf89019",
        "min.in": "f4e6932b1baf9328279ee5526d89948cb1f10f9009f0752e205bf9c4ff453b6a",
        "min.py": "c85b0c6bfd43627551e84f9fec6a0e76db4d16f089a640053d78fba522d9d601",
        "resolved.config": "6f2fec67bd208c7510d537405614a064b225baf4d693757170319806b8714093",
        "REST2.in": "1ae0792348eb5a20accdbb5296bab7eb57a68e23c40ee2889c33318879ac4e96",
        "REST2.py": "3e039fcc9c24d68ebadeda2c73c88b47c628011583e55b75430d80e4cd1c2f87",
        "run.sh": "700b5d0b4013c2c48faba7de66905ac9c62e84122d4f5e3505bb0add98234ead",
    },
    "implicit": {
        "eq_nvt_free.in": "e752e9d786765ee03bbfeb1bc005857bd18b8a7d191560fafb61fb1873ecef15",
        "eq_nvt_free.py": "63f242cd9e3c1bf26ae98ff86e95792d76bbe4d94d06653bdaa6c1b140628f62",
        "eq_nvt_posres_2.in": "1962fa101d2bc707b32c9727f5386b0344f7e2d908f640fe4e205f81d62a806e",
        "eq_nvt_posres_2.py": "3054435667e24ebc079e4ecae1f1ddbc3de854035a6ee440f1a61e4f95c2cd35",
        "eq_nvt_posres.in": "0bb523155922ccb4ff64252aa7d0b86ee6e2bb79198d58fde37c983b58d80955",
        "eq_nvt_posres.py": "b5a332209934c06cbc1fe47f8cb780933bd672cf13de18f2d76214bb6cf89019",
        "min.in": "f156fce24799290387ddb192ffef350420fa05d714cff2ac69ebaef751279bee",
        "min.py": "c85b0c6bfd43627551e84f9fec6a0e76db4d16f089a640053d78fba522d9d601",
        "resolved.config": "51ebec05d57459fa2916373958a75bc720d7939bd684681a9aa113cc53c11c94",
        "REST2.in": "38b63fe39dd4afc13a496cfac2f5cdc78b10be3e2bd5e2e39e2c2f7fcc74999e",
        "REST2.py": "3e039fcc9c24d68ebadeda2c73c88b47c628011583e55b75430d80e4cd1c2f87",
        "run.sh": "56855931c369e433e8b2111ff9309aea6d564e892194d3720a0cf5a5ec861d14",
    },
}
#: The `_protocol.py` a default four-state ladder materialises at 2 fs, before this setting.
PROTOCOL_HELPER_BEFORE = "806d23669d38b89fa27af10b37250f5ee72e598ad124e71ec62723fa37a0bc91"


@pytest.mark.parametrize("solvent", ["explicit", "implicit"])
def test_off_leaves_every_generated_file_as_it_was(tmp_path, solvent):
    out = _build_md(tmp_path, {"protocol": "REST2", "solvent": solvent})
    assert {p.name for p in out.iterdir()} - {"build-md.log"} == set(BEFORE[solvent])
    for name, digest in BEFORE[solvent].items():
        text = (out / name).read_text(encoding="utf-8")
        if name.endswith(".in") or name == "resolved.config":
            added = [line for line in text.splitlines() if "equilibration_per_tau" in line]
            assert len(added) == 1 and "false" in added[0], (name, added)
            text = "".join(line for line in text.splitlines(keepends=True)
                           if "equilibration_per_tau" not in line)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, name


@pytest.mark.parametrize("solvent", ["explicit", "implicit"])
def test_off_leaves_the_protocol_helper_and_the_identity_as_they_were(tmp_path, solvent):
    from md_tools.remd.generated import ladder_from_resolved, protocol_file_text

    ladder = ladder_from_resolved(_resolve(tmp_path, {"protocol": "REST2", "solvent": solvent}),
                                  "REST2")
    assert "per_tau_equilibration" not in ladder
    ladder["dynamics"]["timestep_fs"] = 2.0     # what the runtime resolves `auto` to here
    text = protocol_file_text(ladder)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PROTOCOL_HELPER_BEFORE
    namespace = {}
    exec(text, namespace)
    assert "per_tau_equilibration" not in namespace["protocol"].describe()


def test_on_the_stages_reach_the_protocol_and_its_identity(tmp_path):
    from md_tools.build.md import per_tau_equilibration_stages
    from md_tools.remd.generated import ladder_from_resolved, protocol_file_text

    resolved = _resolve(tmp_path, {"protocol": "REST2", "solvent": "implicit",
                                   "rest2": {"equilibration_per_tau": True}})
    ladder = ladder_from_resolved(resolved, "REST2")
    assert ladder["per_tau_equilibration"] == per_tau_equilibration_stages(resolved)
    ladder["dynamics"]["timestep_fs"] = 2.0
    namespace = {}
    exec(protocol_file_text(ladder), namespace)
    protocol = namespace["protocol"]
    assert protocol.per_tau_equilibration == ladder["per_tau_equilibration"]
    assert protocol.describe()["per_tau_equilibration"]["stages"] == ladder["per_tau_equilibration"]


@pytest.mark.parametrize("plan, words", [
    ([{"name": "eq_npt_free", "steps": 10}], "not one of"),
    ([{"name": "eq_nvt_posres", "steps": 0}], "positive integer"),
    ([{"name": "eq_nvt_free", "steps": 5}, {"name": "eq_nvt_posres", "steps": 5}], "order"),
    ([{"name": "eq_nvt_posres", "steps": 5, "restraint_kcal_per_mol_A2": -1.0}], ">= 0"),
])
def test_a_malformed_plan_is_refused_by_the_protocol(plan, words):
    from md_tools.remd.protocol import ProtocolError, REST2Protocol

    with pytest.raises(ProtocolError, match=words):
        REST2Protocol(tau=[0.0, 0.5], temperature_k=300.0, timestep_fs=2.0,
                      exchange_interval_ps=0.1, number_of_exchanges=2,
                      per_tau_equilibration=plan)


def test_the_handoff_files_are_owned_outputs_only_when_the_setting_is_on(tmp_path):
    from md_tools.run.preflight import _ladder_inventory

    common = dict(protocol="REST2", replicas=3, output=tmp_path / "REST2.out",
                  log=tmp_path / "REST2.log", trajectory=tmp_path / "REST2.nc",
                  restart=tmp_path / "restart.json",
                  checkpoint=tmp_path / "REST2_checkpoint.nc", groupfile=None)
    off = _ladder_inventory(**common).roles
    on = _ladder_inventory(**common, per_tau=True).roles
    assert set(on) - set(off) == {"per_tau_state_0", "per_tau_state_1", "per_tau_state_2",
                                  "per_tau_record"}
    assert on["per_tau_state_2"].name == "per_tau_state2.xml"


# --- the runtime module -------------------------------------------------------------------------------

def test_the_rung_equilibration_module_imports_nothing_from_md_tools():
    """It is copied into reference bundles, which run with md_tools absent."""
    tree = ast.parse((SRC / "remd" / "rung_equilibration.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in {"openmm", "numpy"} for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.module in {"engine", "core"}, node.module
            else:
                assert node.module.split(".")[0] in {"openmm", "numpy"}, node.module


def test_the_seeds_are_the_documented_derivation_and_all_distinct():
    from md_tools.md._stages import derive_seed as stage_chain_seed
    from md_tools.remd.rung_equilibration import derive_seed, stage_seed

    assert stage_chain_seed is derive_seed
    seeds = {(stage, index): stage_seed(7, stage, index) for stage in STAGES for index in range(8)}
    for (stage, index), seed in seeds.items():
        assert seed == _documented_seed(7, stage, f"state{index}")
    assert len(set(seeds.values())) == len(seeds)


class _Interruption:
    requested = False
    signal = None


class _Coordinator:
    rank, size, is_root = 0, 1, True

    @staticmethod
    def any_true(value):
        return bool(value)

    def barrier(self):
        pass


def _stub_driver(tmp_path, monkeypatch, *, interrupt_after=None):
    """A driver with its OpenMM work stubbed out: only the orchestration is under test here."""
    from md_tools.remd import rung_equilibration
    from md_tools.remd.driver import ReplicaRun

    calls = []
    interruption = _Interruption()

    def fake_stage(restrained, configuration, stage, **settings):
        calls.append((stage["name"], settings["state_index"]))
        if interrupt_after is not None and len(calls) >= interrupt_after:
            interruption.requested = True
        return configuration, {"stage": stage["name"], "seed": 1}, "<State/>"

    monkeypatch.setattr(rung_equilibration, "run_stage", fake_stage)
    monkeypatch.setattr(rung_equilibration, "restrained_clone", lambda system, *_: system)
    start = tmp_path / "min.xml"
    start.write_text("<State/>", encoding="utf-8")

    driver = ReplicaRun.__new__(ReplicaRun)
    driver.protocol = SimpleNamespace(
        per_tau_equilibration=[
            {"name": "eq_nvt_posres", "steps": 10, "restraint_kcal_per_mol_A2": 1.0},
            {"name": "eq_nvt_free", "steps": 10, "restraint_kcal_per_mol_A2": 0.0}],
        random_seed=7, temperature_k=300.0, friction_per_ps=1.0, timestep_fs=2.0,
        constraint_tolerance=1e-8, total_steps=100, n_states=2, tau=[0.0, 0.5])
    driver.files = SimpleNamespace(topology=str(ALA), trajectory=str(tmp_path / "REST2.nc"),
                                   coordinates=str(start))
    driver.owned = [0, 1]
    driver.coordinator = _Coordinator()
    driver.solute_indices = list(range(22))
    driver._interruption = interruption
    driver._platform, driver._properties, driver._fault_crossings = None, {}, {}
    driver.trajectories = driver.solute_trajectories = driver.reporter = None
    driver.engine = SimpleNamespace(set_configuration=lambda index, configuration: None)
    driver._gather_configurations = lambda state: [SimpleNamespace(box=None)] * 2
    return driver, calls, interruption


def test_every_rung_runs_a_stage_before_any_rung_runs_the_next(tmp_path, monkeypatch):
    driver, calls, _interruption = _stub_driver(tmp_path, monkeypatch)
    state = {"configurations": ["walker0", "walker1"], "state_to_walker": [0, 1]}
    assert driver._equilibrate_per_tau(state, systems=["rung0", "rung1"]) is False
    assert calls == [("eq_nvt_posres", 0), ("eq_nvt_posres", 1),
                     ("eq_nvt_free", 0), ("eq_nvt_free", 1)]
    record = json.loads((tmp_path / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    assert [entry["file"] for entry in record["states"]] == ["per_tau_state0.xml",
                                                             "per_tau_state1.xml"]
    assert [len(entry["stages"]) for entry in record["states"]] == [2, 2]


def test_an_interruption_stops_between_stages_and_names_the_phase(tmp_path, monkeypatch):
    """Every rung finishes the stage it is in; none starts the next. The record says where."""
    from md_tools.remd.storage import read_run_state

    driver, calls, _interruption = _stub_driver(tmp_path, monkeypatch, interrupt_after=2)
    state = {"configurations": ["walker0", "walker1"], "state_to_walker": [0, 1]}
    assert driver._equilibrate_per_tau(state, systems=["rung0", "rung1"]) is True
    assert calls == [("eq_nvt_posres", 0), ("eq_nvt_posres", 1)]
    assert state["per_tau_stage"] == "eq_nvt_free"
    assert not (tmp_path / "per_tau_equilibration.json").exists()

    result = driver._record_per_tau_interruption(state, identity={})
    assert result["run_status"] == "interrupted"
    recorded = read_run_state(tmp_path / "REST2.nc")
    assert recorded["status"] == "interrupted"
    assert recorded["phase"] == "per_tau_equilibration"
    assert recorded["stage"] == "eq_nvt_free"
    assert "--overwrite" in recorded["note"]


# --- real ladders, CPU ----------------------------------------------------------------------------------

def _build_top(root, config_text):
    (root / "sys.config").write_text(config_text, encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


def _stage(run, name, parent=None):
    argv = CLI + ["md-run", "-i", f"{name}.in", "-p", "../built.pdb", "-s", "../built.xml",
                  "-r", f"{name}.xml", "-chk", f"{name}.chk", "-o", f"{name}.out",
                  "-log", f"{name}.log", "-odir", ".", "--cpu"]
    if parent:
        argv += ["-c", parent]
    done = subprocess.run(argv, cwd=run, capture_output=True, text=True, timeout=1800,
                          env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


def _ladder_argv(start, *extra):
    return ["mpirun", "-n", str(RUNGS), *CLI, "md-run", "-ng", str(RUNGS), "-i", "REST2.in",
            "-p", "../built.pdb", "-s", "../built.xml", "-c", start, "-x", "REST2.nc",
            "-r", "restart.json", "-o", "REST2.out", "-log", "REST2.log", "--cpu", *extra]


def _needs_mpirun():
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH; a ladder needs one rank per rung")


IMPLICIT = {
    "protocol": "REST2", "solvent": "implicit",
    "dynamics": {"timestep_fs": 2.0, "seed": 11},
    "stages": {"minimization_iterations": 50, "restrained_nvt_steps": 100,
               "restrained_npt_steps": 0, "unrestrained_npt_steps": 100, "production_steps": 0},
    "reporting": {"crd_printout_solute": 50, "info_printout": 50, "checkpoint_printout": 100},
    "rest2": {"number_of_replicas": RUNGS, "tau_max": 0.5, "exchange_interval_steps": 50,
              "number_of_exchanges": 4, "equilibration_steps": 50,
              "equilibration_per_tau": True},
}


@pytest.fixture(scope="module")
def implicit_ladder(tmp_path_factory):
    _needs_mpirun()
    root = tmp_path_factory.mktemp("per-tau-implicit")
    _build_top(root, "solvent:\n  model: GBn2\n")
    run = _build_md(root, IMPLICIT, odir="run")
    _stage(run, "min")
    done = subprocess.run(_ladder_argv("min.xml"), cwd=run, capture_output=True, text=True,
                          timeout=3600, env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root, run, done.stdout


def _positions(path):
    from openmm import XmlSerializer, unit

    state = XmlSerializer.deserialize(Path(path).read_text(encoding="utf-8"))
    return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.slow
def test_every_rung_ends_its_equilibration_at_its_own_state(implicit_ladder):
    _root, run, _stdout = implicit_ladder
    record = json.loads((run / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    assert [stage["name"] for stage in record["stages"]] == ["eq_nvt_posres", "eq_nvt_free"]
    assert [entry["state_index"] for entry in record["states"]] == list(range(RUNGS))
    seeds = [stage["seed"] for entry in record["states"] for stage in entry["stages"]]
    assert len(set(seeds)) == len(seeds) == 2 * RUNGS

    start = _positions(run / "min.xml")
    ends = []
    for entry in record["states"]:
        assert _sha256(run / entry["file"]) == entry["sha256"]
        ends.append(_positions(run / entry["file"]))
    for index, end in enumerate(ends):
        assert not np.array_equal(end, start), f"rung {index} never left the starting state"
        for other in ends[index + 1:]:
            assert not np.array_equal(end, other), "two rungs ended in the same configuration"

    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["per_tau_equilibration"]["states"] == record["states"]
    assert manifest["per_tau_equilibration"]["record_sha256"] == _sha256(
        run / "per_tau_equilibration.json")


@pytest.mark.slow
def test_a_rungs_end_state_is_what_the_documented_procedure_gives(implicit_ladder):
    """Rebuilt with plain OpenMM: the rung's scaling, the documented restraint and seed formula.

    Not the module the driver calls -- that would compare the code with itself. Only the rung's
    scaled System comes from md_tools, because the scaling is not what is under test here.
    """
    from openmm import (Context, CustomExternalForce, LangevinMiddleIntegrator, Platform,
                        XmlSerializer, unit)
    from openmm.app import PDBFile

    from md_tools.rest2 import build_scaled_system

    root, run, _stdout = implicit_ladder
    solute = yaml.safe_load((run / "solute.yaml").read_text(encoding="utf-8"))
    atoms = list(range(int(solute["n_solute_atoms"])))
    excluded = [tuple(bond) for bond in solute["rest2"]["omega_excluded_bonds"]]
    record = json.loads((run / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    rung = 2
    tau = record["states"][rung]["tau"]
    system = build_scaled_system(
        XmlSerializer.deserialize((root / "built.xml").read_text(encoding="utf-8")),
        atoms, tau, excluded_bonds=excluded)

    force = CustomExternalForce("0.5*restraint_k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("restraint_k", 0.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    reference = PDBFile(str(root / "built.pdb")).positions.value_in_unit(unit.nanometer)
    for index in atoms:
        force.addParticle(index, list(reference[index]))
    system.addForce(force)

    start = XmlSerializer.deserialize((run / "min.xml").read_text(encoding="utf-8"))
    positions = start.getPositions(asNumpy=True)
    velocities = start.getVelocities(asNumpy=True)
    for name, strength in (("eq_nvt_posres", 1.0), ("eq_nvt_free", 0.0)):
        integrator = LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond,
                                              2.0 * unit.femtosecond)
        integrator.setConstraintTolerance(1e-8)
        integrator.setRandomNumberSeed(_documented_seed(11, name, f"state{rung}"))
        context = Context(system, integrator, Platform.getPlatformByName("CPU"),
                          {"Threads": "1"})
        context.setPositions(positions)
        context.setVelocities(velocities)
        context.setParameter("restraint_k", strength * 418.4)
        integrator.step(100)
        state = context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        del context, integrator

    engine = _positions(run / f"per_tau_state{rung}.xml")
    assert np.array_equal(np.asarray(positions.value_in_unit(unit.nanometer)), engine), (
        "the engine's per-tau end state is not what the documented procedure produces; largest "
        f"difference {np.abs(np.asarray(positions.value_in_unit(unit.nanometer)) - engine).max()}"
        " nm")


@pytest.mark.slow
def test_the_ladder_says_what_it_did(implicit_ladder):
    from md_tools.build.record import read_record

    _root, run, stdout = implicit_ladder
    assert "per_tau_equilibration=[" in (run / "_protocol.py").read_text(encoding="utf-8")
    record = read_record(run / "REST2.log")
    assert [s["name"] for s in record["ladder"]["per_tau_equilibration"]] == [
        "eq_nvt_posres", "eq_nvt_free"]
    readable = stdout + "".join(p.read_text(encoding="utf-8", errors="replace")
                                for p in run.glob("REST2*.out"))
    assert "# per-tau equilibration" in readable


def _tree(path):
    return {p.relative_to(path).as_posix(): _sha256(p)
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


@pytest.mark.slow
def test_resume_after_an_interruption_during_it_is_refused_and_writes_nothing(implicit_ladder,
                                                                              tmp_path):
    """No exchange step was taken, so there is no checkpoint: say so, and touch nothing."""
    root, _run, _stdout = implicit_ladder
    copy = tmp_path / "root"
    shutil.copytree(root, copy)
    run = copy / "run"
    state_path = run / "REST2.runstate.json"
    document = json.loads(state_path.read_text(encoding="utf-8"))
    document.update(status="interrupted", phase="per_tau_equilibration", stage="eq_nvt_free")
    state_path.write_text(json.dumps(document), encoding="utf-8")
    before = _tree(copy)

    done = subprocess.run(_ladder_argv("min.xml", "--resume"), cwd=run, capture_output=True,
                          text=True, timeout=1800, env=ONE_THREAD)
    assert done.returncode != 0, done.stdout[-2000:]
    assert "per-tau equilibration" in done.stderr and "--overwrite" in done.stderr, done.stderr
    assert _tree(copy) == before, "a refused continuation changed the directory"


@pytest.mark.slow
def test_a_failure_during_it_stops_the_whole_ladder(implicit_ladder, tmp_path):
    """A rank that fails in per-tau equilibration aborts every rank; nothing claims completion."""
    root, _run, _stdout = implicit_ladder
    for name in ("built.xml", "built.pdb", "built.log"):
        shutil.copy2(root / name, tmp_path / name)
    run = _build_md(tmp_path, IMPLICIT, odir="run")
    _stage(run, "min")
    done = subprocess.run(
        _ladder_argv("min.xml"), cwd=run, capture_output=True, text=True, timeout=1800,
        env={**ONE_THREAD, "MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "2",
             "MD_TOOLS_FAIL_LADDER_AT": "per-tau-equilibration"})
    assert done.returncode != 0
    assert "per-tau-equilibration" in done.stdout + done.stderr
    assert not (run / "restart.json").exists()
    assert not (run / "per_tau_equilibration.json").exists()


@pytest.fixture(scope="module")
def explicit_ladder(tmp_path_factory):
    _needs_mpirun()
    root = tmp_path_factory.mktemp("per-tau-explicit")
    _build_top(root, "solvent:\n  model: TIP3P\n")
    run = _build_md(root, {
        "protocol": "REST2", "solvent": "explicit",
        "dynamics": {"timestep_fs": 2.0, "seed": 5},
        "stages": {"minimization_iterations": 50, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
                   "production_steps": 0},
        "reporting": {"crd_printout_solute": 50, "info_printout": 50,
                      "checkpoint_printout": 100},
        "rest2": {"number_of_replicas": RUNGS, "tau_max": 0.5, "exchange_interval_steps": 50,
                  "number_of_exchanges": 2, "equilibration_per_tau": True},
    }, odir="run")
    parent = None
    for name in ("min", "eq_nvt_posres", "eq_npt_posres", "eq_npt_free"):
        _stage(run, name, parent)
        parent = f"{name}.xml"
    done = subprocess.run(_ladder_argv("eq_npt_free.xml"), cwd=run, capture_output=True,
                          text=True, timeout=3600, env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root, run


@pytest.mark.slow
def test_explicit_rungs_share_the_box_the_npt_stages_fixed(explicit_ladder):
    from openmm import XmlSerializer, unit

    _root, run = explicit_ladder

    def box(path):
        state = XmlSerializer.deserialize(Path(path).read_text(encoding="utf-8"))
        return np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))

    fixed = box(run / "eq_npt_free.xml")
    record = json.loads((run / "per_tau_equilibration.json").read_text(encoding="utf-8"))
    assert record["box_shared"] is True
    assert [s["name"] for s in record["stages"]] == list(STAGES)
    for entry in record["states"]:
        assert np.array_equal(box(run / entry["file"]), fixed), entry["file"]

    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    # The rungs the ladder propagated carry neither the restraint nor a barostat.
    forces = json.dumps(manifest["hamiltonian"])
    assert "CustomExternalForce" not in forces and "Barostat" not in forces, forces

