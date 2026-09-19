"""The ladder's direct Python API: caller-supplied rungs are declared, recorded, never re-derived.

Shared contract §3, decided by the user 2026-09-19. `md-run` reads every rung from a saved state;
a DIRECT caller (hpREST2's OPES-REST2 builds its own `LadderPreflight` and `ReplicaRun`) now has
to declare `rung_source='caller-supplied'` with a reason, and every rung's origin is recorded by
canonical digest against the saved states: `saved-state <i> (verified)` or `caller-modified`.

The OPES-shaped fixture is three saved-state rungs plus one AUXILIARY rung: a copy of the hot rung
carrying one added `CustomCompoundBondForce` with a `Continuous1DFunction` table, in a free force
group. It must run with the declaration alone, and come out as three verified rungs and one
caller-modified rung.

PLATFORM_POLICY_EXEMPTION: the direct ladders run under `--cpu` (platform CPU requested
explicitly) to produce real run records; what is tested is declaration, provenance and identity,
not dynamics.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

openmm = pytest.importorskip("openmm")
from openmm import XmlSerializer                                               # noqa: E402

from md_tools.rest2 import rungs as rung_rules                                  # noqa: E402
from md_tools.rest2.identity import system_fingerprint                          # noqa: E402
from md_tools.run.preflight import LadderPreflight, PreflightError              # noqa: E402

from .test_selective_rest2_integration import (REPO, _env, _initial_state, _project,  # noqa: E402
                                               _run, _scaler)

STATES = 3
LEGACY_SCALER = f"method: REST2\nschedule:\n  n_states: {STATES}\n  tau_max: 0.5\n"


# --- the auxiliary rung ----------------------------------------------------------------------------

def auxiliary(hot, *, table=None, group=31):
    """The hot rung plus a bias: hpREST2's shape. The force sits in its own free group."""
    system = copy.deepcopy(hot)
    values = list(table if table is not None else
                  [0.5 * math.sin(2 * math.pi * k / 24) for k in range(25)])
    force = openmm.CustomCompoundBondForce(4, "0.1*bias(dihedral(p1,p2,p3,p4))")
    force.addTabulatedFunction("bias", openmm.Continuous1DFunction(values, -math.pi, math.pi,
                                                                    True))
    force.addBond([4, 6, 8, 14], [])
    force.setForceGroup(group)
    system.addForce(force)
    return system


# --- fixtures: saved states on disk ----------------------------------------------------------------

def _states(root: Path, scaler_text: str) -> Path:
    from .conftest import make_dataset_root

    make_dataset_root(root, solvent="explicit")
    _scaler(root, _env(root, REPO / "src"), scaler_text)
    return root / "build" / "REST2"


def _saved(states_dir: Path):
    return [XmlSerializer.deserialize((states_dir / f"system_state{i}.xml").read_text(
        encoding="utf-8")) for i in range(STATES)]


@pytest.fixture(scope="module")
def states(tmp_path_factory):
    return _states(tmp_path_factory.mktemp("rungs"), LEGACY_SCALER)


@pytest.fixture(scope="module")
def hot_only_states(tmp_path_factory):
    """tau_min > 0: no saved state is the built System, so routes (b) and (c) are distinct."""
    return _states(tmp_path_factory.mktemp("rungs-hot"),
                   f"method: REST2\nschedule:\n  n_states: {STATES}\n  tau_min: 0.1\n"
                   f"  tau_max: 0.5\n")


# --- 1. the declaration, at construction -----------------------------------------------------------

def _plan(**fields):
    base = dict(coordination=None, machine={}, acceleration=None, device_index=None,
                device_policy="local_rank", device_policy_detail="")
    base.update(fields)
    return LadderPreflight(**base)


def test_undeclared_supplied_rungs_are_refused_by_name(states):
    with pytest.raises(PreflightError, match="without saying where they come from"):
        _plan(rung_systems=tuple(_saved(states)))


def test_a_caller_cannot_claim_saved_states(states):
    with pytest.raises(PreflightError, match="cannot be claimed by a caller"):
        _plan(rung_systems=tuple(_saved(states)), rung_source="saved-states",
              rung_source_reason="trust me")


def test_caller_supplied_needs_a_reason(states):
    with pytest.raises(PreflightError, match="non-empty rung_source_reason"):
        _plan(rung_systems=tuple(_saved(states)), rung_source="caller-supplied",
              rung_source_reason="  ")


def test_a_declared_plan_is_accepted_at_construction(states):
    plan = _plan(rung_systems=tuple(_saved(states)), rung_source="caller-supplied",
                 rung_source_reason="three saved states, unmodified")
    assert plan.rung_source == "caller-supplied"


def test_an_unknown_source_is_refused(states):
    with pytest.raises(PreflightError, match="is not one of"):
        _plan(rung_systems=tuple(_saved(states)), rung_source="derived",
              rung_source_reason="x")


# --- 2. origins, structure and where the saved states are found ------------------------------------

def test_three_saved_rungs_verify_and_the_auxiliary_is_caller_modified(states):
    systems = _saved(states)
    rungs = systems + [auxiliary(systems[-1])]
    digests, _taus, reference = rung_rules.saved_state_digests(states / "scaler.yaml")
    rung_rules.check_rung_structure(rungs, reference)
    origins = [entry["origin"] for entry in rung_rules.rung_origins(rungs, digests)]
    assert origins == ["saved-state 0 (verified)", "saved-state 1 (verified)",
                       "saved-state 2 (verified)", "caller-modified"]


def test_a_rung_of_another_system_is_refused(states):
    systems = _saved(states)
    heavier = copy.deepcopy(systems[1])
    heavier.setParticleMass(0, heavier.getParticleMass(0) * 2)
    constrained = copy.deepcopy(systems[1])
    constrained.addConstraint(0, 5, 0.3)
    extra = copy.deepcopy(systems[1])
    extra.addParticle(1.0)
    reference = rung_rules.saved_state_digests(states / "scaler.yaml")[2]
    for rung, words in ((heavier, "masses"), (constrained, "constraints"), (extra, "particles")):
        with pytest.raises(rung_rules.RungProvenanceError, match=words):
            rung_rules.check_rung_structure([systems[0], rung], reference)


def test_the_record_is_found_by_each_route(states, hot_only_states, tmp_path):
    built = states.parent / "built.xml"
    # (c) the built System, with build/REST2/scaler.yaml naming it as its source
    assert rung_rules.find_saved_states_record(system_path=built) == states / "scaler.yaml"
    # (b) a saved state itself -- on a ladder with tau_min > 0, where no state IS built.xml
    hot_state = hot_only_states / "system_state0.xml"
    assert hot_state.read_bytes() != (hot_only_states.parent / "built.xml").read_bytes()
    assert rung_rules.find_saved_states_record(system_path=hot_state) == \
        hot_only_states / "scaler.yaml"
    # (a) an explicit record wins over the others
    assert rung_rules.find_saved_states_record(
        explicit=hot_only_states / "scaler.yaml", system_path=built) == \
        hot_only_states / "scaler.yaml"
    # (c) refuses a scaler.yaml whose source is another System
    other = tmp_path / "built.xml"
    other.write_text(XmlSerializer.serialize(_saved(states)[1]), encoding="utf-8")
    (tmp_path / "REST2").symlink_to(states)
    assert rung_rules.find_saved_states_record(system_path=other) is None


# --- 3. digest stability (hpREST2's note) ----------------------------------------------------------

def test_an_auxiliary_digest_survives_a_round_trip_and_a_regenerated_table(states):
    hot = _saved(states)[-1]
    first = auxiliary(hot)
    round_tripped = XmlSerializer.deserialize(XmlSerializer.serialize(first))
    assert system_fingerprint(round_tripped) == system_fingerprint(first)
    regenerated = auxiliary(hot, table=[float(repr(v)) for v in
                                        [0.5 * math.sin(2 * math.pi * k / 24) for k in range(25)]])
    assert system_fingerprint(regenerated) == system_fingerprint(first)
    changed_values = [0.5 * math.sin(2 * math.pi * k / 24) for k in range(25)]
    changed_values[7] += 1e-9
    assert system_fingerprint(auxiliary(hot, table=changed_values)) != system_fingerprint(first)


# --- 4. the direct ladder, end to end (CPU) --------------------------------------------------------

@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A ladder project whose preflight gives a real, sealed saved-state plan to start from."""
    root = tmp_path_factory.mktemp("direct")
    env = _env(root, REPO / "src")
    _states(root, LEGACY_SCALER)
    _project(root, env)
    _initial_state(root)
    return root


def _preflight_plan(root: Path, out: Path):
    from md_tools.run.preflight import preflight_ladder

    from .conftest import ladder_group_file

    ladder = {"protocol": "REST2", "n_states": STATES, "tau_max": 0.5,
              "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 1,
                           "friction_per_ps": 1.0},
              "exchange_interval_steps": 10, "number_of_exchanges": 2}
    return preflight_ladder(
        topology=str(root / "build" / "built.pdb"), system=None,
        groupfile=str(ladder_group_file(root, out)), replicas=STATES,
        output=out / "REST2.out", log=out / "REST2.log", trajectory=out / "REST2.nc",
        cpu=True, protocol="REST2", timestep_fs=2.0, ladder=ladder, out_dir=out, tau=0.5)


def _direct_run(root: Path, out: Path, plan, *, taus, system_path=None, extend=0):
    """What hpREST2's driver does: its own protocol and plan, `ReplicaRun` constructed directly."""
    from openmm.app import PDBFile

    from md_tools.remd.driver import ReplicaRun
    from md_tools.remd.protocol import REST2Protocol

    os.environ["MD_TOOLS_CONFIG"] = str(root / "user.config")
    # hpREST2's `opes_rest2_protocol`: an auxiliary rung shares the hot tau, which REST2Protocol
    # refuses because in a plain ladder tau identifies a rung. It is constructed with distinct
    # placeholders and the true list installed after, exactly as hpREST2 does it.
    placeholder = [i * 0.9 / len(taus) for i in range(len(taus))]
    protocol = REST2Protocol(tau=placeholder, temperature_k=300.0, timestep_fs=2.0,
                             exchange_interval_ps=0.02, number_of_exchanges=2, random_seed=5,
                             platform="CPU")
    protocol.tau = [float(t) for t in taus]
    out.mkdir(parents=True, exist_ok=True)
    files = SimpleNamespace(
        topology=str(root / "build" / "built.pdb"),
        system=str(system_path or root / "build" / "built.xml"),
        coordinates=str(root / "initial_state.xml"), trajectory=str(out / "REST2.nc"),
        restart=str(out / "restart.json"), checkpoint=str(out / "REST2_checkpoint.nc"),
        output=str(out / "REST2.out"), rem=None)
    base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    run = ReplicaRun(protocol=protocol, files=files, base_system=base,
                     topology=PDBFile(files.topology).topology, solute_indices=list(range(22)),
                     platform="CPU", explicit_cpu=True, prepared=plan)
    return run.run(extend=extend)


def _declared(plan, rungs, reason="OPES-like: three saved-state rungs, unmodified, plus the hot "
                                  "rung carrying a tabulated dihedral bias"):
    # No saved_states_record, like hpREST2's plan: the record is found by route (c), from
    # files.system = build/built.xml.
    return dataclasses.replace(plan, rung_systems=tuple(rungs), rung_source="caller-supplied",
                               rung_source_reason=reason, rung_origins=(), saved_state_seal=None,
                               saved_states_record=None,
                               replicas=len(rungs), tau_list=tuple([0.0, 0.25, 0.5, 0.5][
                                   :len(rungs)]))


def test_a_plan_with_no_rungs_is_refused_before_any_output(project, tmp_path):
    from md_tools.remd.driver import DriverError

    out = tmp_path / "none"
    plan = _preflight_plan(project, out)
    empty = dataclasses.replace(plan, rung_systems=(), rung_source="", rung_origins=(),
                                saved_state_seal=None)
    with pytest.raises(DriverError, match="not re-derived at run time"):
        _direct_run(project, out, empty, taus=[0.0, 0.25, 0.5])
    assert not out.exists() or not any(out.iterdir())


def test_a_saved_state_plan_with_swapped_rungs_is_refused(project, tmp_path):
    from md_tools.remd.driver import DriverError

    out = tmp_path / "swapped"
    plan = _preflight_plan(project, out)
    swapped = dataclasses.replace(plan, rung_systems=tuple(reversed(plan.rung_systems)))
    with pytest.raises(DriverError, match="not the Systems its preflight verified"):
        _direct_run(project, out, swapped, taus=[0.0, 0.25, 0.5])


def test_the_opes_shaped_ladder_runs_with_the_declaration_alone(project, tmp_path):
    out = tmp_path / "opes"
    plan = _preflight_plan(project, out)
    rungs = list(plan.rung_systems) + [auxiliary(plan.rung_systems[-1])]
    record = _direct_run(project, out, _declared(plan, rungs), taus=[0.0, 0.25, 0.5, 0.5])
    assert record["run_status"] == "completed"
    entry = json.loads((out / "restart.json").read_text(encoding="utf-8"))[
        "scientific_identity"]["rungs"]
    assert entry["rung_source"] == "caller-supplied"
    assert "tabulated dihedral bias" in entry["rung_source_reason"]
    assert entry["saved_states_record"] == "scaler.yaml"
    assert [r["origin"] for r in entry["rungs"]] == [
        "saved-state 0 (verified)", "saved-state 1 (verified)", "saved-state 2 (verified)",
        "caller-modified"]
    assert entry["rungs"][3]["system_digest"] == system_fingerprint(rungs[3])


def test_the_saved_state_path_records_verified_origins(project, tmp_path):
    """md-run's own path (sealed plan), through the same driver: every rung verified."""
    out = tmp_path / "saved"
    plan = _preflight_plan(project, out)
    assert plan.rung_source == "saved-states"
    record = _direct_run(project, out, plan, taus=[0.0, 0.25, 0.5])
    assert record["run_status"] == "completed"
    entry = json.loads((out / "restart.json").read_text(encoding="utf-8"))[
        "scientific_identity"]["rungs"]
    assert [r["origin"] for r in entry["rungs"]] == [
        f"saved-state {i} (verified)" for i in range(STATES)]
    assert rung_rules.all_verified_saved_states(entry)


# --- 5. the resume pair ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def finished_opes(project, tmp_path_factory):
    out = tmp_path_factory.mktemp("opes-finished") / "run"
    plan = _preflight_plan(project, out)
    rungs = list(plan.rung_systems) + [auxiliary(plan.rung_systems[-1])]
    assert _direct_run(project, out, _declared(plan, rungs),
                       taus=[0.0, 0.25, 0.5, 0.5])["run_status"] == "completed"
    return out


def _copy_run(finished, tmp_path):
    import shutil

    out = tmp_path / "run"
    shutil.copytree(finished, out)
    return out


def test_the_same_rungs_after_a_process_restart_extend(project, finished_opes, tmp_path):
    """A NEW process rebuilds the auxiliary from identical float64 values: same digests."""
    out = _copy_run(finished_opes, tmp_path)
    plan = _preflight_plan(project, out)
    rungs = list(plan.rung_systems) + [auxiliary(
        XmlSerializer.deserialize(XmlSerializer.serialize(plan.rung_systems[-1])))]
    record = _direct_run(project, out, _declared(plan, rungs), taus=[0.0, 0.25, 0.5, 0.5],
                         extend=1)
    assert record["run_status"] == "completed"


@pytest.mark.parametrize("change", ["table", "declaration"])
def test_other_rungs_or_another_declaration_are_refused_on_extension(project, finished_opes,
                                                                     tmp_path, change):
    from md_tools.remd.driver import IdentityError

    out = _copy_run(finished_opes, tmp_path)
    plan = _preflight_plan(project, out)
    table = [0.5 * math.sin(2 * math.pi * k / 24) for k in range(25)]
    reason = None
    if change == "table":
        table[7] += 1e-6
    else:
        reason = "a different account of the same rungs"
    rungs = list(plan.rung_systems) + [auxiliary(plan.rung_systems[-1], table=table)]
    declared = _declared(plan, rungs, **({"reason": reason} if reason else {}))
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(IdentityError, match="rungs"):
        _direct_run(project, out, declared, taus=[0.0, 0.25, 0.5, 0.5], extend=1)
    assert json.loads((out / "restart.json").read_text(encoding="utf-8")) == \
        json.loads(before["restart.json"])


def test_a_declared_rung_of_another_system_is_refused_by_the_driver(project, tmp_path):
    from md_tools.remd.driver import DriverError

    out = tmp_path / "heavier"
    plan = _preflight_plan(project, out)
    heavier = copy.deepcopy(plan.rung_systems[-1])
    heavier.setParticleMass(0, heavier.getParticleMass(0) * 2)
    rungs = list(plan.rung_systems) + [heavier]
    with pytest.raises(DriverError, match="masses"):
        _direct_run(project, out, _declared(plan, rungs), taus=[0.0, 0.25, 0.5, 0.5])
    assert not out.exists() or not any(out.iterdir())


def test_caller_supplied_rungs_with_no_findable_saved_states_are_refused(project, tmp_path):
    from md_tools.remd.driver import DriverError

    out = tmp_path / "unfindable"
    plan = _preflight_plan(project, out)
    stray = tmp_path / "elsewhere.xml"
    stray.write_text((project / "build" / "built.xml").read_text(encoding="utf-8"),
                     encoding="utf-8")
    with pytest.raises(DriverError, match="none were found"):
        _direct_run(project, out, _declared(plan, list(plan.rung_systems)),
                    taus=[0.0, 0.25, 0.5], system_path=stray)
