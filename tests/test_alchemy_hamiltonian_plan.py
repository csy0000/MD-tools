"""The softcore Hamiltonian over S2's real topology plans (fixture `alchemy-endpoints/1`).

Real AM1-BCC packages -- ethane, chloroethane, ethanol -- in TIP3P under PME with the dispersion
correction on, combined by `md_tools.alchemy.topology.build_topology_plan` (hybrid). What only a
real plan can show:

* the plan's dummy-removed junction terms are accepted, and nothing else about a unique atom's
  bonded terms differs;
* under the default boundary rule the Hamiltonian's end states ARE the plan's end-state Systems,
  energy for energy, dispersion correction included -- both groups here have no internal pair,
  so no internal ghost term exists to account for;
* the derivatives are the finite-difference limits of the energy, and the Amber18 path is not
  linear (the AIS identity dU/dlambda = U1 - U0 fails);
* S4's window layer accepts the Hamiltonian for a path over its three components.

Reference platform, float64: fixed-coordinate evaluation, nothing propagated.
"""
from __future__ import annotations

import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_fixtures as af  # noqa: E402
from md_tools.alchemy.hamiltonian import build_hamiltonian, from_plan  # noqa: E402
from md_tools.alchemy.softcore import SoftcoreSettings  # noqa: E402

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")
#: Tolerance fixed before comparing: S2's end-state recovery holds these Systems to 1e-7 kJ/mol
#: per force class on Reference (handoffs/S2.md), and the Hamiltonian adds float64 arithmetic
#: over |E| ~ 1e4, ~1e-11 relative.
ENDPOINT_TOL = 1e-7


def _state(v):
    return dict(zip(NAMES, v))


def _context(system, x):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(x)
    return c


def _energy(system, x):
    return _context(system, x).getState(getEnergy=True).getPotentialEnergy()._value


@pytest.fixture(scope="module", params=["chloroethane", "ethanol"])
def plan(request):
    from md_tools.alchemy.topology import build_topology_plan
    a = af.package(af.ETHANE)
    b = af.package({"chloroethane": af.CHLOROETHANE, "ethanol": af.ETHANOL}[request.param])
    return build_topology_plan(a, b, af.core_map(a, b), af.water_environment(), mode="hybrid")


def test_the_end_states_are_the_plans_systems(plan):
    x = plan.positions_nm
    h = from_plan(plan)
    c = _context(h.system, x)
    assert any(f["differing_terms"] for f in h.record["bonded_mixed_forces"])   # dummy-removed
    assert h.record["nonbonded"]["use_dispersion_correction"] is True
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(_energy(plan.system_a, x), abs=ENDPOINT_TOL)
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(_energy(plan.system_b, x), abs=ENDPOINT_TOL)


def test_the_amber18_boundary_rule_changes_the_end_states_and_says_so(plan):
    """`unscaled` keeps a unique atom's 1-4s with the core at full strength: the end states are
    then NOT the plan's Systems, by exactly those 1-4s."""
    x = plan.positions_nm
    h = build_hamiltonian(plan.system_a, plan.system_b, plan.a_only, plan.b_only,
                          settings=SoftcoreSettings(sc_boundary_14="unscaled"))
    assert h.record["softcore"]["sc_boundary_14"] == "unscaled"
    c = _context(h.system, x)
    assert abs(h.energy(c, _state((0, 0, 0))) - _energy(plan.system_a, x)) > 1e-3
    assert abs(h.energy(c, _state((1, 1, 1))) - _energy(plan.system_b, x)) > 1e-3


def _fd(f, lam, h):
    if lam - h < 0:
        return (-3 * f(lam) + 4 * f(lam + h) - f(lam + 2 * h)) / (2 * h)
    if lam + h > 1:
        return (3 * f(lam) - 4 * f(lam - h) + f(lam - 2 * h)) / (2 * h)
    return (f(lam + h) - f(lam - h)) / (2 * h)


#: OpenMM re-integrates a custom long-range correction by quadrature whenever lambda_sterics
#: changes; the dispersion carrier's energy departs from exact linearity by up to 4.9e-8 kJ/mol
#: (measured over 41 lambdas on the chloroethane plan). Its derivative is the exact slope. So the
#: carrier is checked on its own, with a finite-difference tolerance derived from that noise:
#: a one-sided 3-point difference at h = 1e-2 amplifies it by (3 + 4 + 1) / (2 h) = 400.
DISPERSION_ENERGY_NOISE = 5e-8
DISPERSION = "dispersion"


def test_derivatives_are_the_limit_of_the_energy_and_the_path_is_not_linear(plan):
    from md_tools.alchemy.hamiltonian import FORCE_GROUPS
    x = plan.positions_nm
    h = from_plan(plan)
    c = _context(h.system, x)
    rest = set(FORCE_GROUPS.values()) - {FORCE_GROUPS[DISPERSION]}

    def energy(t, groups):
        h.set_state(c, _state(t))
        return c.getState(getEnergy=True, groups=groups).getPotentialEnergy()._value
    for base in (0.0, 0.5, 1.0):
        parts = h.derivative_components(c, _state((base,) * 3))
        for k, name in enumerate(NAMES):
            def f(v, k=k, groups=rest):
                t = [base] * 3
                t[k] = v
                return energy(t, groups)
            want = sum(v for g, v in parts[name].items() if g != DISPERSION)
            coarse, fine = _fd(f, base, 1e-2), _fd(f, base, 1e-3)
            richardson = (100 * fine - coarse) / 99
            assert abs(want - richardson) < 2e-5 * max(1.0, abs(want)), (base, name, want, richardson)
        def g(v):
            return energy([base, v, base], {FORCE_GROUPS[DISPERSION]})
        slope = parts["lambda_sterics"][DISPERSION]
        assert abs(slope - _fd(g, base, 1e-2)) < 400 * DISPERSION_ENERGY_NOISE, (base, slope)
    # Along a linear path dU/dlambda is the same at every lambda (AIS's U1 - U0). Here it is not.
    # (Comparing the midpoint slope with the secant would be no test: a curved U(lambda) can
    # have its midpoint slope equal to its secant, and at these coordinates it nearly does.)
    slope_0 = sum(h.derivatives(c, _state((0, 0, 0))).values())
    slope_1 = sum(h.derivatives(c, _state((1, 1, 1))).values())
    assert abs(slope_1 - slope_0) > 1.0, (slope_0, slope_1)


def test_the_dispersion_slope_is_the_end_states_own(plan):
    """d/dlambda_sterics of the dispersion term is D_B - D_A, with D_X the correction OpenMM applies
    to the plan's end-state System X -- measured with the correction on and off."""
    x = plan.positions_nm

    def correction(system):
        s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
        on = _energy(s, x)
        for f in s.getForces():
            if isinstance(f, openmm.NonbondedForce):
                f.setUseDispersionCorrection(False)
        return on - _energy(s, x)
    h = from_plan(plan)
    c = _context(h.system, x)
    for base in (0.0, 0.5, 1.0):
        slope = h.derivative_components(c, _state((base,) * 3))["lambda_sterics"][DISPERSION]
        assert slope == pytest.approx(correction(plan.system_b) - correction(plan.system_a), abs=1e-7)


def test_the_window_layer_accepts_it_for_a_three_component_path(plan):
    from md_tools.alchemy.paths import AlchemicalPath, Knot
    from md_tools.alchemy.windows import check_hamiltonian_matches_path
    h = from_plan(plan)
    path = AlchemicalPath(knots=(Knot(0.0, _state((0, 0, 0))), Knot(1.0, _state((1, 1, 1)))),
                          endpoint_a="ethane", endpoint_b="the substituted molecule")
    check_hamiltonian_matches_path(h, path)
