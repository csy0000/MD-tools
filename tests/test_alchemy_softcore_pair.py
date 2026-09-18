"""One softcore pair, checked against Amber18 manual eqs. 21.5-21.7 written out independently.

The reference is sympy, typed from the manual -- not a call into `md_tools.alchemy`, and not a
copy of its expression strings. Energy, force and the complete dU/dlambda of each component are
compared on the Reference platform (double precision), for the disappearing direction (A-only
particle, V0 form) and the appearing direction (B-only particle, V1 form: lambda <-> 1 - lambda).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
sympy = pytest.importorskip("sympy")

from md_tools.alchemy.hamiltonian import (AlchemicalHamiltonianError,  # noqa: E402
                                          build_hamiltonian)
from md_tools.alchemy.softcore import SoftcoreError, SoftcoreSettings  # noqa: E402

# The Coulomb constant NonbondedForce uses, kJ nm mol^-1 e^-2; `test_coulomb_constant_is_openmms`
# measures it, so the reference below does not borrow the module's copy.
K_COULOMB = 138.93545764438198
ALPHA = 0.5            # scalpha, dimensionless
BETA = 12.0 * 0.01     # scbeta, 12 A^2 in nm^2

QC, SC, EC = 0.4, 0.32, 0.65     # common particle (end state A charge)
QU, SU, EU = -0.4, 0.29, 0.40    # unique particle, in the end state where it is physical


def _reference(direction: str):
    """(U, dU/dr, dU/dlam_e, dU/dlam_s) as numpy callables of (r, lam_e, lam_s)."""
    r, le, ls = sympy.symbols("r lambda_e lambda_s", positive=True)
    sigma = (SC + SU) / 2
    eps = sympy.sqrt(EC * EU)
    if direction == "disappearing":           # eq. 21.5 and 21.7 as printed
        w_e, l_e, w_s, l_s = 1 - le, le, 1 - ls, ls
    else:                                     # eq. 21.6; 21.7 with lambda <-> 1 - lambda
        w_e, l_e, w_s, l_s = le, 1 - le, ls, 1 - ls
    lj = 4 * eps * w_s * (1 / (ALPHA * l_s + (r / sigma) ** 6) ** 2 - 1 / (ALPHA * l_s + (r / sigma) ** 6))
    coul = w_e * K_COULOMB * QC * QU / sympy.sqrt(BETA * l_e + r ** 2)
    u = lj + coul
    f = lambda expr: sympy.lambdify((r, le, ls), expr, "numpy")  # noqa: E731
    return f(u), f(sympy.diff(u, r)), f(sympy.diff(u, le)), f(sympy.diff(u, ls))


def _pair_systems(direction: str):
    """Particle 0 common, particle 1 unique. Net charge 0 in both end states."""
    def system(q0, unique_physical):
        s = openmm.System()
        s.addParticle(12.0)
        s.addParticle(12.0)
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        nb.addParticle(q0, SC, EC)
        if unique_physical:
            nb.addParticle(QU, SU, EU)
        else:
            nb.addParticle(0.0, SU, 0.0)
        s.addForce(nb)
        return s
    physical, dummy = system(QC, True), system(QC + QU, False)
    if direction == "disappearing":
        return physical, dummy, {1}, set()
    return dummy, physical, set(), {1}


def _context(system):
    return openmm.Context(system, openmm.VerletIntegrator(0.001),
                          openmm.Platform.getPlatformByName("Reference"))


def test_coulomb_constant_is_openmms():
    s = openmm.System()
    s.addParticle(1.0)
    s.addParticle(1.0)
    nb = openmm.NonbondedForce()
    nb.addParticle(1.0, 0.1, 0.0)
    nb.addParticle(1.0, 0.1, 0.0)
    s.addForce(nb)
    c = _context(s)
    c.setPositions([[0, 0, 0], [1, 0, 0]])
    assert c.getState(getEnergy=True).getPotentialEnergy()._value == pytest.approx(K_COULOMB, rel=1e-12)


@pytest.mark.parametrize("direction", ["disappearing", "appearing"])
def test_pair_energy_force_and_derivatives(direction):
    u, du_dr, du_dle, du_dls = _reference(direction)
    sa, sb, a_only, b_only = _pair_systems(direction)
    h = build_hamiltonian(sa, sb, a_only, b_only)
    c = _context(h.system)
    lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    worst = {"energy": 0.0, "force": 0.0, "dle": 0.0, "dls": 0.0}
    for r in (0.02, 0.1, 0.25, 0.31, 0.5, 1.2):
        c.setPositions([[0, 0, 0], [r, 0, 0]])
        for le, ls in itertools.product(lambdas, lambdas):
            state = {"lambda_electrostatics": le, "lambda_sterics": ls, "lambda_bonded": 0.0}
            h.set_state(c, state)
            st = c.getState(getEnergy=True, getForces=True)
            energy = st.getPotentialEnergy()._value
            force_on_1 = st.getForces(asNumpy=True)._value[1]
            d = h.derivatives(c, state)
            ref_e, ref_f = float(u(r, le, ls)), -float(du_dr(r, le, ls))
            ref_dle, ref_dls = float(du_dle(r, le, ls)), float(du_dls(r, le, ls))
            for key, got, want in (("energy", energy, ref_e), ("force", force_on_1[0], ref_f),
                                   ("dle", d["lambda_electrostatics"], ref_dle),
                                   ("dls", d["lambda_sterics"], ref_dls)):
                err = abs(got - want) / max(1.0, abs(want))
                worst[key] = max(worst[key], err)
                assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (key, r, le, ls, got, want)
            assert abs(force_on_1[1]) < 1e-12 and abs(force_on_1[2]) < 1e-12
            assert d["lambda_bonded"] == 0.0
    assert max(worst.values()) < 1e-9, worst


def test_physical_end_points_of_the_pair():
    """lambda = 0: plain Coulomb + LJ. lambda = 1: nothing, for the disappearing particle."""
    sa, sb, a_only, b_only = _pair_systems("disappearing")
    h = build_hamiltonian(sa, sb, a_only, b_only)
    c = _context(h.system)
    r = 0.37
    c.setPositions([[0, 0, 0], [r, 0, 0]])
    sigma, eps = (SC + SU) / 2, np.sqrt(EC * EU)
    plain = K_COULOMB * QC * QU / r + 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    zero = {"lambda_electrostatics": 0.0, "lambda_sterics": 0.0, "lambda_bonded": 0.0}
    one = {"lambda_electrostatics": 1.0, "lambda_sterics": 1.0, "lambda_bonded": 1.0}
    assert h.energy(c, zero) == pytest.approx(plain, rel=1e-12)
    assert h.energy(c, one) == pytest.approx(0.0, abs=1e-12)


def test_overlap_is_finite_away_from_the_physical_end():
    """The point of softcore: r -> 0 stays finite once lambda > 0 (and only then)."""
    sa, sb, a_only, b_only = _pair_systems("disappearing")
    h = build_hamiltonian(sa, sb, a_only, b_only)
    c = _context(h.system)
    c.setPositions([[0, 0, 0], [1e-5, 0, 0]])
    state = {"lambda_electrostatics": 0.5, "lambda_sterics": 0.5, "lambda_bonded": 0.0}
    energy = h.energy(c, state)
    u, *_ = _reference("disappearing")
    assert np.isfinite(energy)
    # NonbondedForce's plain q q / r and the softcore delta cancel here: 1/1e-5 against O(1).
    # Double precision keeps ~1e-11 relative of 1e7 -> 1e-4 absolute is the honest bound.
    assert energy == pytest.approx(float(u(1e-5, 0.5, 0.5)), abs=1e-4)


@pytest.mark.parametrize("name", ["gapsys", "beutler", "openfe", "smoothstep", "Gapsys"])
def test_other_potentials_are_refused_by_name(name):
    with pytest.raises(SoftcoreError, match="refused"):
        SoftcoreSettings(softcore_function=name)


def test_unknown_function_and_bad_values_are_refused():
    with pytest.raises(SoftcoreError, match="not known"):
        SoftcoreSettings(softcore_function="amber18-smooth")
    with pytest.raises(SoftcoreError, match="> 0"):
        SoftcoreSettings(scalpha=0.0)
    with pytest.raises(SoftcoreError, match="unknown"):
        SoftcoreSettings.from_mapping({"sc": True, "scgamma": 1.0})


def test_defaults_are_the_md_tools_defaults():
    s = SoftcoreSettings()
    assert (s.sc, s.softcore_function, s.scalpha, s.scbeta) == (True, "amber18", 0.5, 12.0)
    assert s.scbeta_nm2 == pytest.approx(0.12)


def test_sc_false_refuses_a_disappearing_particle():
    sa, sb, a_only, b_only = _pair_systems("disappearing")
    with pytest.raises(AlchemicalHamiltonianError, match="sc is false"):
        build_hamiltonian(sa, sb, a_only, b_only, settings=SoftcoreSettings(sc=False))
