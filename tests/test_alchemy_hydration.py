"""S4: legs of a thermodynamic cycle on S2's real plans and S3's Amber18 Hamiltonian.

Fixture `alchemy-endpoints/1`: ethane -> chloroethane, hybrid. The fast tests here check the leg
layer (`md_tools.alchemy.campaign`) on the vacuum leg with the CPU platform; the M2 campaign
itself (acceptance rows M2.0-M2.6) is slow and its claimed values are CUDA runs.
"""
from __future__ import annotations

import json
import math
import os

import pytest

os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")
openmm = pytest.importorskip("openmm")
from openmm import app  # noqa: E402

from tests import alchemy_fixtures as af  # noqa: E402
from md_tools.alchemy.campaign import (analyze_leg, combine_repeats, prepare_leg,  # noqa: E402
                                       read_leg, run_leg)
from md_tools.alchemy.cycles import Leg  # noqa: E402
from md_tools.alchemy.paths import linear_path  # noqa: E402
from md_tools.alchemy.windows import WindowError, WindowSettings, window_paths  # noqa: E402

NAMES = ["lambda_bonded", "lambda_electrostatics", "lambda_sterics"]


def _vacuum(a, b):
    from md_tools.alchemy.hamiltonian import from_plan
    from md_tools.alchemy.topology import build_topology_plan
    plan = build_topology_plan(a, b, af.core_map(a, b),
                               af.vacuum_environment(a, constraints=app.HBonds), mode="hybrid")
    return plan, from_plan(plan)


@pytest.fixture(scope="module")
def vacuum_plan():
    a, b = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    return _vacuum(a, b)


def _prepare(tmp_path, plan, h, s_values=(0.0, 0.5, 1.0)):
    path = linear_path(NAMES, endpoint_a="ethane", endpoint_b="chloroethane")
    return prepare_leg(tmp_path / "leg", plan=plan, hamiltonian=h, path=path, s_values=s_values,
                       temperature_k=300.0, pressure_bar=None, environment="vacuum",
                       endpoint_a="ethane", endpoint_b="chloroethane", scheme="amber18-hybrid")


def test_a_leg_records_what_rebuilds_it(tmp_path, vacuum_plan):
    plan, h = vacuum_plan
    record = _prepare(tmp_path, plan, h)
    assert record["plan_sha256"] == plan.sha256
    back, path, states = read_leg(tmp_path / "leg")
    assert back == record and [s.state_id for s in states] == ["w000", "w001", "w002"]
    assert path.components == tuple(sorted(NAMES))
    with pytest.raises(WindowError, match="prepared once"):
        _prepare(tmp_path, plan, h)


def test_a_leg_refuses_another_hamiltonian(tmp_path, vacuum_plan):
    plan, h = vacuum_plan
    _prepare(tmp_path, plan, h)
    a, b = af.package(af.ETHANE), af.package(af.ETHANOL)
    _, other = _vacuum(a, b)
    with pytest.raises(WindowError, match="not the one"):
        run_leg(tmp_path / "leg", hamiltonian=other, settings=WindowSettings(
            steps=100, report_interval=50, checkpoint_interval=100), cpu=True)
    assert not (tmp_path / "leg" / "windows").exists()


def test_a_short_vacuum_leg_runs_and_analyses(tmp_path, vacuum_plan):
    """Real S3 Hamiltonian, real plan, real propagation (CPU, --cpu): the plumbing, not a number."""
    plan, h = vacuum_plan
    _prepare(tmp_path, plan, h)
    st = WindowSettings(steps=2000, report_interval=100, checkpoint_interval=1000,
                        equilibration_steps=200, timestep_fs=2.0, seed=3, minimize_iterations=200)
    res = run_leg(tmp_path / "leg", hamiltonian=h, settings=st, cpu=True)
    assert [r["rows"] for r in res] == [21, 21, 21]
    record = json.loads(window_paths(tmp_path / "leg" / "windows" / "r1", "w001")["record"]
                        .read_text())
    assert set(record["context_parameters"]["w001"]) > set(NAMES)   # S3's derived parameters
    leg, analysis = analyze_leg(tmp_path / "leg")
    assert leg.environment == "vacuum" and leg.estimator == "MBAR"
    assert (leg.endpoint_a, leg.endpoint_b) == ("ethane", "chloroethane")
    assert set(analysis["estimates"]) == {"EXP_forward", "EXP_reverse", "BAR", "MBAR", "TI"}
    assert math.isfinite(leg.delta_g_kj_mol)


def test_combining_repeats_is_inverse_variance():
    legs = [Leg(f"r{i}", "vacuum", "A", "B", v, s, 300.0, "MBAR", "x")
            for i, (v, s) in enumerate([(1.0, 0.1), (2.0, 0.2)])]
    c = combine_repeats(legs)
    assert c.delta_g_kj_mol == pytest.approx((1 / 0.01 + 2 / 0.04) / (1 / 0.01 + 1 / 0.04))
    assert c.sigma_kj_mol == pytest.approx((1 / (1 / 0.01 + 1 / 0.04)) ** 0.5)
    with pytest.raises(WindowError, match="different legs"):
        combine_repeats([legs[0], Leg("x", "solvent", "A", "B", 1, 0.1, 300.0, "MBAR", "x")])


def test_the_two_legs_are_matched_before_any_sample_is_read(tmp_path):
    from md_tools.alchemy.campaign import matched_leg_report
    from md_tools.alchemy.hamiltonian import from_plan
    from md_tools.alchemy.topology import TopologyError, build_topology_plan
    a, b = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    path = linear_path(NAMES, endpoint_a="ethane", endpoint_b="chloroethane")

    def leg(name, env, environment, pressure):
        plan = build_topology_plan(a, b, af.core_map(a, b), env, mode="hybrid")
        prepare_leg(tmp_path / name, plan=plan, hamiltonian=from_plan(plan), path=path,
                    s_values=(0.0, 1.0), temperature_k=300.0, pressure_bar=pressure,
                    environment=environment, endpoint_a="ethane", endpoint_b="chloroethane",
                    scheme="amber18-hybrid")
        return plan

    vac = leg("vac", af.vacuum_environment(a, constraints=app.HBonds), "vacuum", None)
    leg("solv", af.water_environment(), "solvent", 1.01325)
    leg("vac_free", af.vacuum_environment(a), "vacuum", None)       # no constraints
    report = matched_leg_report(tmp_path / "vac", tmp_path / "solv")
    assert report["ligand_hamiltonian_sha256"] == vac.record["ligand_hamiltonian_sha256"]
    assert json.loads((tmp_path / "solv" / "leg.json").read_text())[
        "ligand_hamiltonian_sha256"] == report["ligand_hamiltonian_sha256"]
    with pytest.raises(TopologyError, match="constraint_policy"):
        matched_leg_report(tmp_path / "vac_free", tmp_path / "solv")


def test_a_vacuum_plan_makes_a_vacuum_window_and_nothing_else_does(tmp_path):
    """S0's three rules at plan level: the permission is derived from the plan's solvation, the
    window is labelled from it, and a caller's contrary flag is refused before any output."""
    from md_tools.alchemy.hamiltonian import from_plan
    from md_tools.alchemy.topology import Environment, build_topology_plan
    from md_tools.ligands.mapping import LigandSelector
    a, b = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    v = af.FIXTURE_ROOT.parent / "vacuum-v1"
    env = Environment.from_files(v / "built.xml", v / "built.pdb", LigandSelector(resname="ETA"),
                                 record=v / "built.log")
    plan = build_topology_plan(a, b, af.core_map(a, b), env, mode="hybrid")
    assert plan.record["environment"]["solvation"] == "vacuum"
    h = from_plan(plan)
    path = linear_path(NAMES, endpoint_a="ethane", endpoint_b="chloroethane")
    prepare_leg(tmp_path / "leg", plan=plan, hamiltonian=h, path=path, s_values=(0.0, 1.0),
                temperature_k=300.0, pressure_bar=None, environment="vacuum",
                endpoint_a="ethane", endpoint_b="chloroethane", scheme="amber18-hybrid")
    assert read_leg(tmp_path / "leg")[0]["solvation"] == "vacuum"
    st = WindowSettings(steps=200, report_interval=100, checkpoint_interval=200, seed=1,
                        minimize_iterations=50)
    with pytest.raises(WindowError, match="disagrees with the plan"):
        run_leg(tmp_path / "leg", hamiltonian=h, settings=st, windows=["w000"], cpu=True,
                vacuum_leg=False)
    assert not (tmp_path / "leg" / "windows").exists()
    run_leg(tmp_path / "leg", hamiltonian=h, settings=st, windows=["w000"], cpu=True)
    record = json.loads(window_paths(tmp_path / "leg" / "windows" / "r1", "w000")["record"]
                        .read_text())
    assert record["solvation"] == "vacuum"
