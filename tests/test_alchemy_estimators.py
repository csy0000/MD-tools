"""S4: alchemical sample records and the EXP / BAR / MBAR / TI estimators, on frozen fixtures.

The fixture is S4's analytic harmonic model (`tests/data/alchemy_s4/harmonic_model.py`), which
has an exact free energy -- not the shared-contracts section 7 fixture, which does not exist yet.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from md_tools.alchemy import estimators as est
from md_tools.alchemy.paths import AlchemicalPath, Knot, PathError, linear_path, staged_path
from md_tools.alchemy.samples import (SampleRecordError, SampleSet, concatenate, kt_kj_mol,
                                      window_states, BAR_NM3_TO_KJ_MOL, KJ_PER_KCAL)

DATA = Path(__file__).resolve().parent / "data" / "alchemy_s4"


def _model():
    spec = importlib.util.spec_from_file_location("s4_harmonic_model", DATA / "harmonic_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("s4_harmonic_model", mod)
    spec.loader.exec_module(mod)
    return mod


H = _model()


@pytest.fixture(scope="module")
def frozen():
    return SampleSet.read_json(H.FIXTURE_FILE)


@pytest.fixture(scope="module")
def analysis(frozen):
    return est.analyze(frozen)


# ---------------------------------------------------------------------- the frozen fixture
def test_frozen_fixture_is_what_the_generator_writes(frozen):
    """The fixture is versioned: regeneration must reproduce it, or the version moves."""
    again = H.generate()
    assert again.to_record()["samples"]["sample_id"] == frozen.to_record()["samples"]["sample_id"]
    np.testing.assert_allclose(again.potential_kj_mol, frozen.potential_kj_mol, rtol=0, atol=1e-9)
    for c in frozen.derivatives:
        np.testing.assert_allclose(again.derivatives[c], frozen.derivatives[c], atol=1e-9)
    assert frozen.provenance["fixture_version"] == H.FIXTURE_VERSION


def test_frozen_fixture_sits_on_the_shared_lambda_grid(frozen):
    assert [st.s for st in frozen.states] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert frozen.counts() == {f"w{i:03d}": H.SAMPLES_PER_WINDOW for i in range(5)}


def test_cross_state_potentials_are_the_model(frozen):
    """Every column is the model potential at that state, independently recomputed from x."""
    u = frozen.potential_kj_mol
    i0 = frozen.from_state("w000")                    # le = 0, ls = 0: mu = 0, K = K0
    sum_x2 = 2.0 * u[i0, 0] / H.K0                     # sum x^2, from U at w000
    sum_x = -frozen.derivatives["lambda_electrostatics"][i0] / (H.K0 * H.A)  # sum x
    np.testing.assert_allclose(frozen.derivatives["lambda_sterics"][i0],
                               0.5 * (H.K1 - H.K0) * sum_x2)
    # w004 (le = 1, ls = 1): 1/2 K1 sum (x - A)^2 + C, expanded
    np.testing.assert_allclose(
        u[i0, 4], 0.5 * H.K1 * (sum_x2 - 2 * H.A * sum_x + H.D * H.A ** 2) + H.C, rtol=1e-10)
    # w002 (le = 1, ls = 0)
    np.testing.assert_allclose(
        u[i0, 2], 0.5 * H.K0 * (sum_x2 - 2 * H.A * sum_x + H.D * H.A ** 2) + H.C, rtol=1e-10)


@pytest.mark.parametrize("name", ["MBAR", "BAR", "EXP_forward", "EXP_reverse"])
def test_estimators_recover_the_exact_free_energy(analysis, frozen, name):
    exact = frozen.provenance["exact_delta_f_kJ_mol"] / KJ_PER_KCAL
    e = analysis["estimates"][name]
    gate = est.agreement_gate(e["delta_g_kcal_mol"], e["sigma_kcal_mol"], exact)
    assert gate["verdict"] == "PASS", (name, gate)


def test_ti_passes_only_with_its_integration_uncertainty(analysis, frozen):
    """Trapezoidal TI over 1/K has a real discretisation bias on this grid (~0.56 kJ/mol). The
    quadrature indicator must see it, and the gate must include it -- as the aims require."""
    exact = frozen.provenance["exact_delta_f_kJ_mol"]
    ti = analysis["estimates"]["TI"]
    quad = ti["diagnostics"]["quadrature_discrepancy_kJ_mol"]
    assert quad > 0.2
    # the analytic trapezoid bias on this grid, from the model alone
    kt = frozen.kt
    k = [H.stiffness(ls) for ls in (0.0, 0.5, 1.0)]
    f = [0.5 * (H.K1 - H.K0) * H.D * kt / kk for kk in k]
    trap = 0.25 * (f[0] + 2 * f[1] + f[2])
    bias = trap - 0.5 * H.D * kt * math.log(H.K1 / H.K0)
    assert ti["delta_g_kJ_mol"] - exact == pytest.approx(bias, abs=3 * ti["sigma_kJ_mol"])
    gate = est.agreement_gate(ti["delta_g_kcal_mol"], ti["sigma_kcal_mol"], exact / KJ_PER_KCAL,
                              integration_sigma_kcal=quad / KJ_PER_KCAL)
    assert gate["verdict"] == "PASS", gate


def test_mbar_reports_overlap(analysis):
    d = analysis["estimates"]["MBAR"]["diagnostics"]
    o = np.array(d["overlap_matrix"])
    assert o.shape == (5, 5)
    np.testing.assert_allclose(o.sum(axis=1), 1.0, atol=1e-8)
    assert d["min_neighbour_overlap"] == pytest.approx(min(o[i, i + 1] for i in range(4)))
    assert d["poor_overlap"] is False
    assert set(d["statistical_inefficiency"]) == {f"w{i:03d}" for i in range(5)}


# ---------------------------------------------------------------------- independent references
def test_bar_matches_pymbar(frozen):
    pymbar = pytest.importorskip("pymbar")
    u = frozen.reduced_potential()
    a, b = frozen.from_state("w001"), frozen.from_state("w002")
    w_f, w_r = u[a, 2] - u[a, 1], u[b, 1] - u[b, 2]
    ours, sig = est.bar_from_work(w_f, w_r)
    ref = pymbar.other_estimators.bar(w_f, w_r)
    assert ours == pytest.approx(ref["Delta_f"], abs=1e-8)
    assert sig == pytest.approx(ref["dDelta_f"], rel=1e-6)


def test_statistical_inefficiency_matches_pymbar(frozen):
    ts = pytest.importorskip("pymbar.timeseries")
    y = frozen.derivatives["lambda_electrostatics"][frozen.from_state("w001")]
    assert est.statistical_inefficiency(y) == pytest.approx(
        ts.statistical_inefficiency(y, fast=False), rel=1e-10)
    # AR(1), rho = 0.8, linear observable: g = (1 + rho)/(1 - rho) = 9 in expectation
    assert 5 < est.statistical_inefficiency(y) < 14


def test_exp_matches_the_closed_form_for_a_gaussian_work_distribution():
    """For w ~ N(m, s^2), -ln<e^-w> = m - s^2/2."""
    rng = np.random.default_rng(1)
    m, s = 3.0, 1.2
    w = rng.normal(m, s, 400000)
    df = -est._logmeanexp(-w)
    assert df == pytest.approx(m - s * s / 2, abs=0.02)


def test_exp_is_stable_at_huge_work():
    """A naive mean(exp(-w)) returns 0 here and -ln gives inf."""
    w = np.array([2000.0, 2001.0, 2002.5])
    assert np.mean(np.exp(-w)) == 0.0
    df = -est._logmeanexp(-w)
    assert math.isfinite(df) and 2000.0 < df < 2001.0


# ---------------------------------------------------------------------- TI and the chain rule
def _deterministic(path, s_values, means, *, pressure=None):
    """Two identical samples per window whose derivatives equal given per-window values."""
    states = window_states(path, s_values, temperature_k=300.0, pressure_bar=pressure)
    n = 2 * len(states)
    origin = [st.state_id for st in states for _ in range(2)]
    ders = {c: np.array([means[st.state_id][c] for st in states for _ in range(2)])
            for c in path.components}
    return SampleSet(states, path, [f"{o}:{i}" for i, o in enumerate(origin)], origin,
                     list(range(n)), np.zeros((n, len(states))),
                     None if pressure is None else np.full(n, 27.0), ders)


def test_ti_uses_one_sided_slopes_at_a_knot():
    """Staged path, dU/d le = 2, dU/d ls = 6 everywhere. Exact integral = 2 + 6 = 8 (each
    component goes 0 -> 1 once). Any single dU/ds value at the knot gives something else."""
    p = staged_path([("lambda_electrostatics", 0.5), ("lambda_sterics", 1.0)],
                    endpoint_a="A", endpoint_b="B")
    s = [0.0, 0.25, 0.5, 0.75, 1.0]
    means = {f"w{i:03d}": {"lambda_electrostatics": 2.0, "lambda_sterics": 6.0} for i in range(5)}
    ti = est.ti_estimate(_deterministic(p, s, means))
    assert ti.delta_g_kj_mol == pytest.approx(8.0, abs=1e-12)
    # the naive integrand, one value per window with the RIGHT-hand slope at the knot
    naive = [2 * 2, 2 * 2, 6 * 2, 6 * 2, 6 * 2]
    assert np.trapezoid(naive, s) != pytest.approx(8.0, abs=0.5)


def test_ti_refuses_windows_that_skip_a_knot():
    p = staged_path([("lambda_electrostatics", 0.5), ("lambda_sterics", 1.0)],
                    endpoint_a="A", endpoint_b="B")
    with pytest.raises(PathError, match="knot"):
        window_states(p, [0.0, 0.4, 0.6, 1.0], temperature_k=300.0)


def test_ti_refuses_a_missing_derivative_component(frozen):
    partial = SampleSet(frozen.states, frozen.path, frozen.sample_id, frozen.origin_state,
                        frozen.step, frozen.potential_kj_mol, None,
                        {"lambda_sterics": frozen.derivatives["lambda_sterics"]})
    with pytest.raises(est.EstimatorError, match="missing component is a missing term"):
        est.ti_estimate(partial)


def test_endpoint_difference_is_not_the_derivative(frozen):
    """On this nonlinear path mean(U_B - U_A) at s=0 is not <dU/ds> at s=0: the AIS identity does
    not transfer, and the estimators never use it."""
    idx = frozen.from_state("w000")
    diff = np.mean(frozen.potential_kj_mol[idx, 4] - frozen.potential_kj_mol[idx, 0])
    slope = 2.0 * np.mean(frozen.derivatives["lambda_electrostatics"][idx])  # dle/ds = 2
    assert abs(diff - slope) > 5.0


# ---------------------------------------------------------------------- provenance
def test_estimates_do_not_depend_on_sample_order(frozen):
    """Samples arriving in any order -- as from workers finishing out of turn -- give the same
    answer, because they are grouped by origin state, not by position or process."""
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(frozen.sample_id))
    shuffled = frozen.subset(perm)
    for f in (est.bar_path, est.ti_estimate):
        assert f(shuffled).delta_f_kt == pytest.approx(f(frozen).delta_f_kt, abs=1e-10)


def test_per_window_records_concatenate_in_any_order(frozen):
    parts = [frozen.subset(frozen.from_state(s)) for s in frozen.state_ids]
    a = concatenate(parts[::-1])
    assert est.bar_path(a).delta_f_kt == pytest.approx(est.bar_path(frozen).delta_f_kt, abs=1e-10)


def test_a_rank_field_is_refused(frozen):
    record = frozen.to_record()
    record["samples"]["rank"] = [0] * len(frozen.sample_id)
    with pytest.raises(SampleRecordError, match="ORIGIN STATE"):
        SampleSet.from_record(record)


def test_undefined_origin_state_is_refused(frozen):
    record = frozen.to_record()
    record["samples"]["origin_state"][0] = "w099"
    with pytest.raises(SampleRecordError, match="does not define"):
        SampleSet.from_record(record)


def test_json_round_trip(frozen, tmp_path):
    frozen.write_json(tmp_path / "s.json")
    back = SampleSet.read_json(tmp_path / "s.json")
    np.testing.assert_array_equal(back.potential_kj_mol, frozen.potential_kj_mol)
    assert back.path.digest() == frozen.path.digest()


# ---------------------------------------------------------------------- ensembles and units
def test_npt_reduced_potential_carries_pv():
    s = H.generate(samples_per_window=20, pressure_bar=1.01325)
    u = s.reduced_potential()
    pv = 1.01325 * BAR_NM3_TO_KJ_MOL * s.volume_nm3 / s.kt
    np.testing.assert_allclose(u, s.potential_kj_mol / s.kt + pv[:, None])


def test_npt_without_volume_is_refused():
    s = H.generate(samples_per_window=5, pressure_bar=1.0)
    with pytest.raises(SampleRecordError, match="pV term would be silently omitted"):
        SampleSet(s.states, s.path, s.sample_id, s.origin_state, s.step, s.potential_kj_mol,
                  None, s.derivatives)


def test_nvt_with_volume_is_refused():
    s = H.generate(samples_per_window=5)
    with pytest.raises(SampleRecordError, match="NVT"):
        SampleSet(s.states, s.path, s.sample_id, s.origin_state, s.step, s.potential_kj_mol,
                  np.ones(len(s.sample_id)), s.derivatives)


def test_boltzmann_constant_is_openmms():
    unit = pytest.importorskip("openmm.unit")
    r = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    assert kt_kj_mol(1.0) == pytest.approx(r, rel=1e-12)
    pv = (1.0 * unit.bar * 1.0 * unit.nanometer ** 3 * unit.AVOGADRO_CONSTANT_NA)
    assert BAR_NM3_TO_KJ_MOL == pytest.approx(pv.value_in_unit(unit.kilojoule_per_mole), rel=1e-9)


def test_non_finite_potential_is_refused(frozen):
    pot = frozen.potential_kj_mol.copy()
    pot[3, 2] = np.inf
    with pytest.raises(SampleRecordError, match="non-finite"):
        SampleSet(frozen.states, frozen.path, frozen.sample_id, frozen.origin_state, frozen.step,
                  pot, None, frozen.derivatives)


# ---------------------------------------------------------------------- names and paths
@pytest.mark.parametrize("bad", ["lambda", "tau", "Lambda_sterics", "sterics"])
def test_component_names_that_mean_something_else_are_refused(bad):
    with pytest.raises(PathError):
        linear_path([bad], endpoint_a="A", endpoint_b="B")


def test_state_components_must_be_the_paths(frozen):
    record = frozen.to_record()
    record["states"][1]["components"]["lambda_sterics"] = 0.3
    with pytest.raises(SampleRecordError, match="the path puts"):
        SampleSet.from_record(record)


def test_path_record_round_trip_and_digest():
    p = staged_path([("lambda_electrostatics", 0.4), ("lambda_sterics", 1.0)],
                    endpoint_a="coupled", endpoint_b="decoupled")
    q = AlchemicalPath.from_record(json.loads(json.dumps(p.to_record())))
    assert q.digest() == p.digest()
    assert p.components_at(0.2) == {"lambda_electrostatics": 0.5, "lambda_sterics": 0.0}
    assert [s.slopes for s in p.segments] == [
        {"lambda_electrostatics": 2.5, "lambda_sterics": 0.0},
        {"lambda_electrostatics": 0.0, "lambda_sterics": pytest.approx(1 / 0.6)}]


def test_path_must_run_zero_to_one():
    with pytest.raises(PathError, match="s = 0 to s = 1"):
        AlchemicalPath((Knot(0.0, {"lambda_x": 0.0}), Knot(0.8, {"lambda_x": 1.0})),
                       endpoint_a="A", endpoint_b="B")


# ---------------------------------------------------------------------- the gate itself
def test_gate_verdicts():
    assert est.agreement_gate(1.0, 0.05, 1.1)["verdict"] == "PASS"
    assert est.agreement_gate(1.0, 0.05, 1.2)["verdict"] == "FAIL"          # > 3 sigma
    assert est.agreement_gate(1.0, 0.24, 1.55)["verdict"] == "FAIL"         # > 0.5 kcal/mol
    assert est.agreement_gate(1.0, 0.4, 1.1)["verdict"] == "INCONCLUSIVE"   # sigma too large
    assert est.agreement_gate(1.0, 0.0, 1.0 + 1e-9)["verdict"] == "FAIL"   # exact, no floor
    assert est.agreement_gate(1.0, 0.0, 1.0 + 1e-9,
                              numerical_floor_kcal=1e-6)["verdict"] == "PASS"
    assert est.agreement_gate(1.0, 0.0, 1.6, numerical_floor_kcal=5.0)["verdict"] == "FAIL"
