"""The Amber18 softcore Hamiltonian on S3's miniature end-state pair, vacuum and PME.

Compared with `alchemy_s3_fixture.reference_energy`, an independent numpy implementation of the
Amber18 rules (explicit pairs, plain Ewald with an explicit k-sum, bonded energies from the force
parameters) -- not another call into `md_tools.alchemy`.

TOLERANCES, fixed from a calibration run BEFORE the Hamiltonian was compared (2026-09-19, OpenMM
8.6 Reference platform, double precision), and asserted by `test_reference_calibration`:

    vacuum: the reference reproduces OpenMM's energy of each ordinary end-state System to 1e-13
            kJ/mol -> Hamiltonian tolerance 1e-8 kJ/mol absolute.
    PME, ewaldErrorTolerance 1e-6: 2.9e-5 kJ/mol for either end state (PME against exact Ewald)
            -> Hamiltonian tolerance 1e-4 kJ/mol absolute, about 3.5x the calibrated error.

A tolerance here is never widened after a failure; the implementation is fixed instead.

PLATFORM_POLICY_EXEMPTION: this file compares single-point energies, forces and derivatives with an
independent numpy implementation (explicit pairs, an explicit Ewald k-sum), and those comparisons
hold to 1e-8 kJ/mol, which single precision cannot carry -- Reference is the right platform for
them, not a substitute for CUDA. One test does propagate: the NPT check runs 40 steps with a
MonteCarloBarostat to move the box, and its subject is that nothing box-dependent is cached (PME
parameters, the dispersion coefficients), not the quality of any dynamics; it compares the running
Context against a fresh one at the same box and coordinates. The device evidence for this
Hamiltonian is test_alchemy_hamiltonian_cuda.py and test_alchemy_hamiltonian_cuda_dynamics.py, and
NPT ON CUDA IS NOT YET COVERED THERE -- recorded as a gap in handoffs/S3.md, not claimed here.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_s3_fixture as fx  # noqa: E402
from md_tools.alchemy.hamiltonian import (FORCE_GROUPS,  # noqa: E402
                                          AlchemicalHamiltonianError, build_hamiltonian)
from md_tools.alchemy.softcore import SoftcoreSettings  # noqa: E402

TOL = {False: 1e-8, True: 1e-4}
BOX, CUTOFF = 2.4, 0.9
NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")

DIAGONAL = [(v, v, v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
OFF_DIAGONAL = [(1.0, 0.5, 0.3), (0.0, 0.6, 1.0), (0.4, 1.0, 0.0), (0.8, 0.2, 0.5)]


def _state(t):
    return dict(zip(NAMES, t))


def _context(system, x):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(x)
    return c


def _kappa(system, x):
    c = _context(system, x)
    nb = next(f for f in c.getSystem().getForces() if type(f).__name__ == "NonbondedForce")
    return nb.getPMEParametersInContext(c)[0]


def _setup(periodic, boundary_14="scaled", **components):
    sa, sb, a, b, x = fx.build(periodic, **components)
    h = build_hamiltonian(sa, sb, a, b, settings=SoftcoreSettings(sc_boundary_14=boundary_14))
    ref_kw = dict(kappa=_kappa(sa, x), cutoff=CUTOFF, box=BOX) if periodic else {}
    ref_kw["boundary_14"] = boundary_14
    return sa, sb, a, b, x, h, _context(h.system, x), ref_kw


@pytest.mark.parametrize("periodic", [False, True], ids=["vacuum", "pme"])
def test_reference_calibration(periodic):
    sa, sb, _a, _b, x = fx.build(periodic)
    for s in (sa, sb):
        kw = dict(kappa=_kappa(s, x), cutoff=CUTOFF, box=BOX) if periodic else {}
        got = _context(s, x).getState(getEnergy=True).getPotentialEnergy()._value
        assert abs(got - fx.plain_energy(s, x, **kw)) < {False: 1e-11, True: 5e-5}[periodic]


@pytest.mark.parametrize("boundary_14", ["scaled", "unscaled"])
@pytest.mark.parametrize("periodic", [False, True], ids=["vacuum", "pme"])
@pytest.mark.parametrize("variant", ["full", "charges_only", "lj_only", "bonded_only", "tail",
                                     "vsite"])
def test_energy_against_the_reference(periodic, variant, boundary_14):
    """Per component, by construction: each variant zeroes the other term families.

    `vsite` makes the waters 4-site, with a charged M virtual site: an environment virtual site,
    which the Hamiltonian carries unchanged (S2's OPC complex fixture has 180 of them).
    `tail` grows the appearing group into a five-atom chain, so the region-internal rules --
    non-excluded internal pairs and internal 1-4s unscaled, and removed from the weighted Ewald
    sum -- have something to act on.
    """
    components = {"full": {}, "charges_only": {"lj": False, "bonded": False},
                  "lj_only": {"charges": False, "bonded": False},
                  "bonded_only": {"charges": False, "lj": False}, "tail": {"tail": True},
                  "vsite": {"vsite": True}}[variant]
    sa, sb, a, b, x, h, c, kw = _setup(periodic, boundary_14, **components)
    worst = 0.0
    for t in DIAGONAL + OFF_DIAGONAL:
        got = h.energy(c, _state(t))
        want = fx.reference_energy(sa, sb, a, b, x, _state(t), **kw)
        err = abs(got - want["total"])
        worst = max(worst, err)
        assert err < TOL[periodic], (variant, t, got, want)
    print(f"\n{'pme' if periodic else 'vacuum'} {variant} {boundary_14}: worst |E - E_ref| = {worst:.3e} kJ/mol")


@pytest.mark.parametrize("boundary_14", ["scaled", "unscaled"])
@pytest.mark.parametrize("tail", [False, True], ids=["one-atom", "tail"])
@pytest.mark.parametrize("periodic", [False, True], ids=["vacuum", "pme"])
def test_physical_end_states(periodic, tail, boundary_14):
    """Under the default rule U(0) IS System A and U(1) IS System B, energy for energy: the plan's
    dummy end already keeps a group's own internal terms physical (contract section 4), which is
    exactly what the Hamiltonian keeps at every lambda.

    Under `unscaled` (the Amber18 manual rule) the other region's 1-4s with the core also stay at
    full strength, so the end state is the System plus those, summed here by hand from the end
    state where the region is physical.
    """
    sa, sb, a, b, x, h, c, kw = _setup(periodic, boundary_14, tail=tail)

    def boundary_14s(system, region):
        nb = next(f for f in system.getForces() if type(f).__name__ == "NonbondedForce")
        total = 0.0
        for k in range(nb.getNumExceptions()):
            i, j, qq, sg, ep = nb.getExceptionParameters(k)
            if (i in region) != (j in region):
                d = x[j] - x[i]
                if periodic:
                    d -= BOX * np.round(d / BOX)
                r = float(np.linalg.norm(d))
                total += fx.K_COULOMB * qq._value / r + 4 * ep._value * (
                    (sg._value / r) ** 12 - (sg._value / r) ** 6)
        return total

    e_a = _context(sa, x).getState(getEnergy=True).getPotentialEnergy()._value
    e_b = _context(sb, x).getState(getEnergy=True).getPotentialEnergy()._value
    ghost_a = ghost_b = 0.0
    if boundary_14 == "unscaled":
        ghost_a, ghost_b = boundary_14s(sa, a), boundary_14s(sb, b)
        assert ghost_a != 0.0 and ghost_b != 0.0
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(e_a + ghost_b, abs=TOL[periodic])
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(e_b + ghost_a, abs=TOL[periodic])


def _richardson(estimates):
    """Richardson extrapolation of each ADJACENT pair of estimates, steps a factor 10 apart, both
    O(h^2) (central, and the second-order one-sided form): (100 D(h/10) - D(h)) / 99 cancels the
    h^2 term. Every pair, not only the smallest steps: near a clash the (1e-3, 1e-4) pair is the
    informative one, and with a noise floor the (1e-2, 1e-3) pair. Reported beside the raw
    differences."""
    return [(100.0 * fine - coarse) / 99.0 for coarse, fine in zip(estimates, estimates[1:])]


def _fd(f, lam, h):
    """Central in the interior; second-order one-sided at an end point."""
    if lam - h < 0:
        return (-3 * f(lam) + 4 * f(lam + h) - f(lam + 2 * h)) / (2 * h)
    if lam + h > 1:
        return (3 * f(lam) - 4 * f(lam - h) + f(lam - 2 * h)) / (2 * h)
    return (f(lam + h) - f(lam - h)) / (2 * h)


@pytest.mark.parametrize("periodic, tail, boundary_14, dispersion", [
    (False, False, "scaled", False), (False, True, "scaled", False),
    (False, True, "unscaled", False), (True, False, "scaled", False),
    (True, True, "scaled", True)],
    ids=["vacuum", "vacuum-tail", "vacuum-tail-unscaled14", "pme", "pme-tail-dispersion"])
def test_derivatives_against_finite_differences(periodic, tail, boundary_14, dispersion):
    """Every component, at the five lambdas, against finite differences of TWO energies.

    * of this Hamiltonian's own energy (the definition: dU/dlambda_k is the limit of this), and
    * of the independent reference's energy -- which catches a term the Hamiltonian's energy and
      derivative both omit, which differencing its own energy never could.

    Several step sizes; the error must shrink with h until round-off, and the best must be within
    tolerance. The table is printed (`-s`) as the convergence report.
    """
    sa, sb, a, b, x, h, c, kw = _setup(periodic, boundary_14, tail=tail, dispersion=dispersion)
    # With the dispersion correction on, the dispersion carrier (group `dispersion`) is left OUT
    # of both sides here: the independent reference has no tail term, and OpenMM re-integrates
    # the carrier's long-range correction by quadrature at every lambda, which leaves ~5e-8 kJ/mol
    # of non-linearity in its energy. Its slope is held to the end states' own corrections to 1e-8
    # by `test_dispersion_mixes_the_end_states_own_corrections` instead.
    groups = set(FORCE_GROUPS.values()) - ({FORCE_GROUPS["dispersion"]} if dispersion else set())
    # The tail's first atom starts 0.22 nm from the Cl-, so at lambda_sterics = 1 the derivative
    # is ~2.4e4 kJ/mol and the O(h^2) truncation needs h = 1e-4 to fall under 1e-5 relative, in
    # PME as in vacuum.
    steps = (1e-2, 1e-3, 1e-4)
    report = []
    for base in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = _state((base, base, base))
        parts = h.derivative_components(c, state)
        analytic = {p: sum(v for g, v in parts[p].items() if FORCE_GROUPS[g] in groups)
                    for p in NAMES}
        for k, name in enumerate(NAMES):
            def own(v, k=k):
                t = [base] * 3
                t[k] = v
                h.set_state(c, _state(t))
                return c.getState(getEnergy=True, groups=groups).getPotentialEnergy()._value

            def ref(v, k=k):
                t = [base] * 3
                t[k] = v
                return fx.reference_energy(sa, sb, a, b, x, _state(t), **kw)["total"]
            own_fd = [_fd(own, base, s) for s in steps]
            ref_fd = [_fd(ref, base, s) for s in steps]
            own_err = [abs(analytic[name] - d) for d in own_fd + _richardson(own_fd)]
            ref_err = [abs(analytic[name] - d) for d in ref_fd + _richardson(ref_fd)]
            report.append((base, name, analytic[name], own_err, ref_err))
            scale = max(1.0, abs(analytic[name]))
            # truncation O(h^2) at 1e-2 must fall by >= ~10x by 1e-3 unless already at round-off
            assert own_err[1] <= max(own_err[0] / 10, 1e-7 * scale), (base, name, own_err)
            assert min(own_err) < {False: 1e-6, True: 1e-5}[periodic] * scale, (base, name, own_err)
            assert min(ref_err) < {False: 1e-6, True: 3e-4}[periodic] * scale, (base, name, ref_err)
    print(f"\n{'pme' if periodic else 'vacuum'}{' tail' if tail else ''}: lambda, component, dU/dlambda, "
          "|err| vs own FD, then vs reference FD: h = 1e-2, 1e-3, 1e-4, Richardson(1e-2,1e-3), (1e-3,1e-4)")
    for base, name, value, own_err, ref_err in report:
        print(f"  {base:4.2f} {name:22s} {value:14.6f}  "
              + " ".join(f"{e:9.2e}" for e in own_err) + "  |  "
              + " ".join(f"{e:9.2e}" for e in ref_err))


def test_the_ais_identity_does_not_hold_here():
    """dU/dlambda on the one-step path is NOT U(1) - U(0): the softcore path is not linear."""
    sa, sb, a, b, x, h, c, kw = _setup(False)
    d = sum(h.derivatives(c, _state((0.5, 0.5, 0.5))).values())
    secant = h.energy(c, _state((1, 1, 1))) - h.energy(c, _state((0, 0, 0)))
    assert abs(d - secant) > 1.0, (d, secant)


def test_sc_false_is_linear_mixing_when_nothing_appears():
    """With no unique particle, sc: false is allowed, and there -- only there -- U is linear."""
    import openmm as mm

    def pair(q0, q1, s0):
        s = mm.System()
        s.addParticle(1.0); s.addParticle(1.0)
        nb = mm.NonbondedForce()
        nb.addParticle(q0, s0, 0.5); nb.addParticle(q1, 0.3, 0.4)
        s.addForce(nb)
        return s
    sa, sb = pair(0.3, -0.3, 0.3), pair(0.1, -0.1, 0.35)
    h = build_hamiltonian(sa, sb, set(), set(), settings=SoftcoreSettings(sc=False))
    c = _context(h.system, np.array([[0, 0, 0], [0.33, 0, 0]]))
    u0, u1 = h.energy(c, _state((0, 0, 0))), h.energy(c, _state((1, 1, 1)))
    for v in (0.25, 0.5, 0.75):
        assert h.energy(c, _state((v, v, v))) == pytest.approx((1 - v) * u0 + v * u1, abs=1e-10)
        assert sum(h.derivatives(c, _state((v, v, v))).values()) == pytest.approx(u1 - u0, abs=1e-9)


def test_system_and_state_reconstruct_the_energy_without_the_original_context():
    sa, sb, a, b, x, h, c, kw = _setup(True)
    xml = openmm.XmlSerializer.serialize(h.system)
    json.dumps(h.record)                     # the provenance record is plain data
    rebuilt = _context(openmm.XmlSerializer.deserialize(xml), x)
    for t in DIAGONAL + OFF_DIAGONAL:
        state = _state(t)
        for name, value in h.context_parameters(state).items():
            rebuilt.setParameter(name, value)
        assert rebuilt.getState(getEnergy=True).getPotentialEnergy()._value == \
            pytest.approx(h.energy(c, state), abs=1e-9)


def test_a_context_from_another_system_is_refused():
    sa, sb, a, b, x, h, c, kw = _setup(False)
    with pytest.raises(AlchemicalHamiltonianError, match="lacks"):
        h.set_state(_context(sa, x), _state((0.5, 0.5, 0.5)))
    with pytest.raises(AlchemicalHamiltonianError, match="lambda_restraints"):
        h.context_parameters({**_state((0, 0, 0)), "lambda_restraints": 0.0})


def _nb(system):
    return next(f for f in system.getForces() if type(f).__name__ == "NonbondedForce")


@pytest.mark.parametrize("defect, message", [
    ("dummy_charged", "is a dummy in System A"),
    ("bonded_on_softcore_differs", "touches a softcore particle"),
    ("common_loses_lj", "has Lennard-Jones in one end state only"),
    ("net_charge_changes", "net charge changes"),
    ("cutoff_periodic", "not supported"),
])
def test_invalid_end_state_pairs_are_refused_by_name(defect, message):
    sa, sb, a, b, x = fx.build(False)
    if defect == "dummy_charged":
        _nb(sa).setParticleParameters(8, 0.1, 0.3, 0.0)
        _nb(sa).setParticleParameters(9, -1.1, 0.44, 0.418)       # keep the net charge
    elif defect == "bonded_on_softcore_differs":
        bonds = next(f for f in sb.getForces() if type(f).__name__ == "HarmonicBondForce")
        bonds.setBondParameters(6, 0, 7, 0.109, 2.0e5)
    elif defect == "common_loses_lj":
        _nb(sb).setParticleParameters(1, -0.27, 0.34, 0.0)
    elif defect == "net_charge_changes":
        _nb(sb).setParticleParameters(9, -0.9, 0.44, 0.418)
    elif defect == "cutoff_periodic":
        for s in (sa, sb):
            s.setDefaultPeriodicBoxVectors([3, 0, 0], [0, 3, 0], [0, 0, 3])
            _nb(s).setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    with pytest.raises(AlchemicalHamiltonianError, match=message):
        build_hamiltonian(sa, sb, a, b)


def test_record_names_the_rules_and_the_end_states():
    sa, sb, a, b, x, h, c, kw = _setup(True)
    r = h.record
    assert r["schema"] == "md-tools-alchemical-hamiltonian/1"
    assert r["softcore"] == {"sc": True, "softcore_function": "amber18", "scalpha": 0.5,
                             "scbeta_angstrom2": 12.0, "scbeta_nm2": 0.12,
                             "sc_boundary_14": "scaled",
                             "sc_boundary_14_amber": "gti_add_sc = 1 (pmemd 20+ default)",
                             "sc_boundary_14_default":
                                 "scaled -- user-confirmed 2026-09-19"}
    assert r["rules"]["boundary_14"] == "scaled"
    assert r["particles"]["a_only"] == [7] and r["particles"]["b_only"] == [8]
    assert r["particles"]["common_lj_changing"] == [0]
    assert r["nonbonded"]["kappa_nm_inv"] == pytest.approx(kw["kappa"], rel=1e-12)
    assert {m["type"] for m in r["bonded_mixed_forces"]} == {
        "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"}


# ------------------------------------------------------------------------------------------------
# The dispersion correction
# ------------------------------------------------------------------------------------------------

def test_dispersion_coefficient_is_openmms():
    """The written-out coefficient equals OpenMM's own correction on a probe with no pair inside
    the cutoff, where the energy IS the correction. Several classes, a repeated class, eps = 0."""
    from md_tools.alchemy.hamiltonian import _dispersion_coefficient
    rng = np.random.default_rng(7)
    sigma = [0.3, 0.3, 0.35, 0.28, 0.3, 0.4]
    epsilon = [0.5, 0.5, 0.2, 0.0, 0.5, 0.9]
    box, cutoff = 12.0, 1.0
    s = openmm.System()
    s.setDefaultPeriodicBoxVectors([box, 0, 0], [0, box, 0], [0, 0, box])
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(cutoff)
    nb.setUseDispersionCorrection(True)
    for sg, ep in zip(sigma, epsilon):
        s.addParticle(1.0)
        nb.addParticle(0.0, sg, ep)
    s.addForce(nb)
    x = np.array([[2.0 * k, 0.0, 0.0] for k in range(len(sigma))]) + rng.uniform(0, 0.1, (len(sigma), 3))
    measured = _context(s, x).getState(getEnergy=True).getPotentialEnergy()._value * box ** 3
    assert _dispersion_coefficient(sigma, epsilon, cutoff) == pytest.approx(measured, rel=1e-12)


@pytest.mark.parametrize("scale", [1.0, 1.08], ids=["built-box", "expanded-box"])
def test_dispersion_mixes_the_end_states_own_corrections(scale):
    """U_disp(lambda) = (1 - lam_s) D_A(V) + lam_s D_B(V), with D_X the correction OpenMM applies to
    end-state System X, at the built box and at an expanded one (it is a 1/V term, as NPT needs)."""
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    h = build_hamiltonian(sa, sb, a, b)
    x = x * scale
    vectors = [np.array(v._value) * scale for v in sa.getDefaultPeriodicBoxVectors()]

    def energy(system, state=None, correction=True):
        s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
        for f in s.getForces():
            if isinstance(f, openmm.NonbondedForce):
                f.setUseDispersionCorrection(correction)
            if isinstance(f, openmm.CustomNonbondedForce) and not correction:
                f.setUseLongRangeCorrection(False)
        c = _context(s, x)
        c.setPeriodicBoxVectors(*vectors)
        if state is not None:
            for name, value in h.context_parameters(state).items():
                c.setParameter(name, value)
        return c.getState(getEnergy=True).getPotentialEnergy()._value

    d_a = energy(sa) - energy(sa, correction=False)
    d_b = energy(sb) - energy(sb, correction=False)
    assert abs(d_a - d_b) > 1e-3             # the end states' corrections really differ
    for t in DIAGONAL + OFF_DIAGONAL:
        state = _state(t)
        mixed = energy(h.system, state) - energy(h.system, state, correction=False)
        assert mixed == pytest.approx((1 - t[1]) * d_a + t[1] * d_b, abs=1e-8), t
    c = _context(h.system, x)
    c.setPeriodicBoxVectors(*vectors)
    parts = h.derivative_components(c, _state((0.5, 0.5, 0.5)))
    assert parts["lambda_sterics"]["dispersion"] == pytest.approx(d_b - d_a, abs=1e-8)
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(energy(sa), abs=1e-7)
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(energy(sb), abs=1e-7)


def test_dispersion_with_a_switching_function_is_refused():
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    for s in (sa, sb):
        _nb(s).setUseSwitchingFunction(True)
        _nb(s).setSwitchingDistance(0.8)
    with pytest.raises(AlchemicalHamiltonianError, match="switching function"):
        build_hamiltonian(sa, sb, a, b)


def test_openmm_long_range_corrections_are_not_additive_over_subsets():
    """Why the dispersion term is built from whole end states. A canary on OpenMM's behaviour
    (8.6): if it changes, the construction's reasoning must be revisited, not just this number.

    * NonbondedForce, and CustomNonbondedForce without groups, count N (N + 1) / 2 pairs with the
      N self pairs included, normalised to N^2 -- not the distinct-pair tail.
    * CustomNonbondedForce WITH interaction groups returns exactly N / (N + 1) times the
      distinct-pair tail of the group pairs, N being every particle in the System.
    So neither kind of correction, summed over a split of the particles into subsets, gives the
    correction of the whole; a Hamiltonian that moves LJ out of NonbondedForce subset by subset
    shifts both end states' energies by O(1/N) of their tail.
    """
    rng = np.random.default_rng(3)
    n, box, rc = 7, 3.0, 1.0
    sig, eps = rng.uniform(0.25, 0.4, n), rng.uniform(0.1, 0.8, n)
    x = rng.uniform(0, box, (n, 3))

    def tail(i, j):
        s, e = 0.5 * (sig[i] + sig[j]), math.sqrt(eps[i] * eps[j])
        return 16 * math.pi * e * (s ** 12 / (9 * rc ** 9) - s ** 6 / (3 * rc ** 3)) / box ** 3

    def correction(make):
        out = []
        for on in (True, False):
            s = openmm.System()
            for _ in range(n):
                s.addParticle(1.0)
            s.setDefaultPeriodicBoxVectors([box, 0, 0], [0, box, 0], [0, 0, box])
            s.addForce(make(on))
            out.append(_context(s, x).getState(getEnergy=True).getPotentialEnergy()._value)
        return out[0] - out[1]

    def custom(groups):
        def make(on):
            f = openmm.CustomNonbondedForce("4*e*((s/r)^12-(s/r)^6); s=0.5*(s1+s2); e=sqrt(e1*e2)")
            f.addPerParticleParameter("s")
            f.addPerParticleParameter("e")
            for i in range(n):
                f.addParticle([sig[i], eps[i]])
            f.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            f.setCutoffDistance(rc)
            f.setUseLongRangeCorrection(on)
            for g in groups:
                f.addInteractionGroup(*g)
            return f
        return make

    def nonbonded(on):
        f = openmm.NonbondedForce()
        f.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        f.setCutoffDistance(rc)
        f.setUseDispersionCorrection(on)
        for i in range(n):
            f.addParticle(0.0, sig[i], eps[i])
        return f

    distinct = sum(tail(i, j) for i, j in itertools.combinations(range(n), 2))
    with_self = distinct + sum(tail(i, i) for i in range(n))
    assert correction(nonbonded) == pytest.approx(n / (n + 1) * with_self, rel=1e-6)
    assert correction(custom([])) == pytest.approx(n / (n + 1) * with_self, rel=1e-6)
    group = ([0, 1, 2], [3, 4, 5, 6])
    group_tail = sum(tail(i, j) for i in group[0] for j in group[1])
    assert correction(custom([group])) == pytest.approx(n / (n + 1) * group_tail, rel=1e-6)
    assert abs(n / (n + 1) * with_self - distinct) > 1e-5 * abs(distinct)


def _exclusion_sets(system):
    """Per nonbonded force, the set of excluded pairs as the CUDA platform compares them."""
    out = []
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            out.append((type(f).__name__, f.getForceGroup(), frozenset(
                tuple(sorted(f.getExceptionParameters(k)[:2])) for k in range(f.getNumExceptions()))))
        elif isinstance(f, openmm.CustomNonbondedForce):
            out.append((type(f).__name__, f.getForceGroup(), frozenset(
                tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions()))))
    return out


@pytest.mark.parametrize("tail", [False, True], ids=["one-atom", "tail"])
@pytest.mark.parametrize("periodic", [False, True], ids=["vacuum", "pme"])
def test_every_nonbonded_force_carries_one_exclusion_set(periodic, tail):
    """The CUDA platform refuses a Context whose nonbonded forces differ in their exclusions ("All
    Forces must have identical exceptions"); Reference and CPU do not check. The first CUDA lane
    (2026-09-19) failed on exactly this for a unique group with internal pairs, which every
    Reference test had passed. This holds the invariant where CPU can see it."""
    sa, sb, a, b, x = fx.build(periodic, tail=tail, dispersion=periodic)
    h = build_hamiltonian(sa, sb, a, b)
    sets = _exclusion_sets(h.system)
    assert len(sets) >= 3
    distinct = {s for _, _, s in sets}
    assert len(distinct) == 1, [(name, group, len(s)) for name, group, s in sets]


#: Reference, float64. Steps halve; the Richardson estimate cancels the h^2 truncation, leaving
#: round-off ~1e-16 |E| / h ~ 1e-9 kJ/mol/nm here. A component passes when its Richardson error
#: is below 1e-6 x max(1, |F|) -- fixed before the first run of this test.
FD_STEPS_REFERENCE = (1e-4, 5e-5, 2.5e-5)


def _fd_ok(rows):
    return all(r["richardson_error"] <= 1e-6 * max(1.0, abs(r["force"])) for r in rows)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_forces_are_the_gradient_of_the_energy_at_the_clash_geometry(lam):
    """Every softcore atom and the Cl- (0.22 nm from T1), every component, three steps: the force
    is -dE/dx of the Hamiltonian's total energy, and the error converges as h^2."""
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    c = _context(h.system, x)
    h.set_state(c, _state((lam,) * 3))
    rows = fx.fd_force_check(c, [7, 8, 9, 12, 13, 14, 15], FD_STEPS_REFERENCE)
    worst = max(rows, key=lambda r: r["richardson_error"] / max(1.0, abs(r["force"])))
    print(f"\nlambda {lam}: worst component atom {worst['atom']}{worst['component']} "
          f"F = {worst['force']:.4e}, errors {['%.1e' % e for e in worst['errors']]}, "
          f"Richardson {worst['richardson_error']:.1e} kJ/mol/nm")
    assert _fd_ok(rows), [r for r in rows if not _fd_ok([r])][:3]


@pytest.mark.parametrize("kind, size", [("step", 0.2), ("kink", 50.0)])
def test_the_force_energy_check_has_power(kind, size):
    """The same check, on the same geometry, FAILS for an energy step and for a slope kink placed
    at the Cl- - T1 distance -- the non-smoothness a force/energy check exists to catch."""
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    r0 = float(np.linalg.norm(x[12] - x[9])) + 1e-5            # inside every stencil
    broken = _context(fx.defective_system(h, "softcore_b_lj", r0, kind, size), x)
    h.set_state(broken, _state((1.0, 1.0, 1.0)))
    rows = fx.fd_force_check(broken, [9, 12], FD_STEPS_REFERENCE)
    worst = max(r["richardson_error"] for r in rows)
    print(f"\n{kind}: worst Richardson error {worst:.2e} kJ/mol/nm")
    assert not _fd_ok(rows), kind


def test_environment_virtual_sites_are_carried_and_softcore_ones_refused():
    """Environment virtual sites (4-site water) are copied exactly and the end states are still
    the Systems; a virtual site built from a softcore particle is refused by name."""
    sa, sb, a, b, x = fx.build(True, vsite=True)
    h = build_hamiltonian(sa, sb, a, b)
    sites = [i for i in range(sa.getNumParticles()) if sa.isVirtualSite(i)]
    assert sites and all(h.system.isVirtualSite(i) for i in sites)
    c = _context(h.system, x)
    e_a = _context(sa, x).getState(getEnergy=True).getPotentialEnergy()._value
    e_b = _context(sb, x).getState(getEnergy=True).getPotentialEnergy()._value
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(e_a, abs=TOL[True])
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(e_b, abs=TOL[True])
    for s in (sa, sb):                       # an M site built on the A-only particle
        s.setVirtualSite(sites[0], openmm.TwoParticleAverageSite(7, 0, 0.5, 0.5))
    with pytest.raises(AlchemicalHamiltonianError, match="softcore particle"):
        build_hamiltonian(sa, sb, a, b)


def test_a_barostat_moving_the_box_leaves_nothing_stale():
    """NPT: with a MonteCarloBarostat in both end states (copied once, as an identical force), a
    run whose box moves gives, at every lambda, the energy a fresh Context reports at that box and
    those coordinates. Nothing box-dependent is cached: PME alpha and grid are fixed at build, the
    softcore delta's kappa is a constant, both dispersion pieces are C/V at the current volume."""
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    for s in (sa, sb):
        s.addForce(openmm.MonteCarloBarostat(1.0, 300.0, 1))
    h = build_hamiltonian(sa, sb, a, b)
    live = openmm.Context(h.system, openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.0005),
                          openmm.Platform.getPlatformByName("Reference"))
    live.setPositions(x)
    h.set_state(live, _state((0.5, 0.5, 0.5)))
    openmm.LocalEnergyMinimizer.minimize(live, 10.0, 100)
    start = live.getState(getEnergy=True).getPotentialEnergy()._value
    v0 = live.getState().getPeriodicBoxVolume()._value
    live.getIntegrator().step(40)
    st = live.getState(getPositions=True, getEnergy=True)
    energy = st.getPotentialEnergy()._value
    assert np.isfinite(energy) and abs(energy) < abs(start) + 5000.0, (start, energy)
    assert st.getPeriodicBoxVolume()._value != v0, "the barostat never moved the box"
    box = st.getPeriodicBoxVectors()
    pos = st.getPositions(asNumpy=True)._value

    def fresh_energy(t, vectors):
        fresh = _context(h.system, pos)
        fresh.setPeriodicBoxVectors(*vectors)
        return h.energy(fresh, _state(t))
    stale = [v._value for v in sa.getDefaultPeriodicBoxVectors()]
    for t in DIAGONAL + OFF_DIAGONAL:
        assert h.energy(live, _state(t)) == pytest.approx(fresh_energy(t, box), abs=1e-8), t
        # the check can fail: the same comparison against a Context left at the ORIGINAL box
        assert abs(h.energy(live, _state(t)) - fresh_energy(t, stale)) > 1e-4, t


@pytest.mark.parametrize("dispersion", [False, True], ids=["no-lrc", "lrc"])
@pytest.mark.parametrize("tail", [False, True], ids=["no-internal-pairs", "internal-pairs"])
def test_the_decoupling_shape_an_abfe_leg_needs(dispersion, tail):
    """S2's proposed ABFE `decoupling` mode, measured rather than assumed: the whole ligand is the
    unique region, nothing appears, and there is no mapped core.

    The questions it answers: a unique group bonded to nothing in the environment (no junction);
    the ligand's internal terms physical at BOTH ends, so the decoupled end is a physical molecule
    in vacuum inside the box; and the end states still reproduced, dispersion correction included.
    """
    sa, _sb, _a, _b, x = fx.build(True, dispersion=dispersion, tail=tail)
    # the whole ligand: the molecule, and the five-atom chain when the fixture carries it. A small
    # ligand has NO non-excluded internal pair (everything is within three bonds); the tail gives
    # it 1-5 pairs. Both are legitimate ABFE shapes, so both are tested.
    ligand = sorted(set(range(9)) | (set(range(12, 16)) if tail else set()))
    da, db = fx.decoupling_pair(sa, ligand)
    h = build_hamiltonian(da, db, ligand, set())
    assert h.record["particles"]["b_only"] == []
    assert (h.record["plan_internal_pairs_checked"] > 0) is tail
    c = _context(h.system, x)
    e_a = _context(da, x).getState(getEnergy=True).getPotentialEnergy()._value
    e_b = _context(db, x).getState(getEnergy=True).getPotentialEnergy()._value
    assert abs(e_a - e_b) > 1.0                        # decoupling is a real change (~10 kJ/mol
    #                                                    here: a small, mostly non-polar ligand)
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(e_a, abs=1e-8)
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(e_b, abs=1e-8)
    # the ligand's intramolecular energy is lambda-independent: the unscaled group
    internal = FORCE_GROUPS["softcore_internal"]
    energies = {v: h.energy_components(c, _state((v, v, v)))["softcore_internal"]
                for v in (0.0, 0.5, 1.0)}
    assert len(set(round(e, 9) for e in energies.values())) == 1, energies
    # and the derivative is still the finite-difference limit at the decoupled end
    parts = h.derivative_components(c, _state((1.0, 1.0, 1.0)))
    groups = set(FORCE_GROUPS.values()) - {FORCE_GROUPS["dispersion"]}

    def energy(t):
        h.set_state(c, _state(t))
        return c.getState(getEnergy=True, groups=groups).getPotentialEnergy()._value
    for k, name in enumerate(NAMES):
        want = sum(v for g, v in parts[name].items() if g != "dispersion")
        fds = [_fd(lambda v, k=k: energy([1.0 if i != k else v for i in range(3)]), 1.0, step)
               for step in (1e-2, 1e-3, 1e-4)]
        best = min(abs(want - r) for r in _richardson(fds))
        assert best < 2e-5 * max(1.0, abs(want)), (name, want, fds)


def _junction_removed(system, atoms):
    """A copy of `system` with the angle on `atoms` at force constant 0. Passed the DUMMY end's
    System, this is exactly a plan's `dummy-removed` junction term (the builder refuses the
    removal at the physical end, which is how this test found its own first mistake)."""
    s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for f in s.getForces():
        if isinstance(f, openmm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                i, j, l, t0, kk = f.getAngleParameters(k)
                if (i, j, l) == atoms:
                    f.setAngleParameters(k, i, j, l, t0, 0.0)
                    return s
    raise AssertionError(f"no angle on {atoms}")


def test_the_bonded_integrand_has_an_expectation_and_warns():
    """dU/dlambda_bonded SHOULD carry nothing from terms touching a unique atom.

    This is the check that did not exist on 2026-09-19, when a probe printed 660.77 kJ/mol for S2's
    chloroethane plan and it was read as "large but plausible". It was the whole bonded integrand
    coming from junction terms switched between zero and full strength, and it took S4's M2 gates to
    make the consequence visible. A diagnostic without an expectation cannot warn anybody.
    """
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    c = _context(h.system, x)
    state = _state((0.0, 0.0, 0.0))

    # this fixture's terms on a unique atom are identical at both ends, which is the expectation
    split = fx.bonded_derivative_split(sa, sb, set(a) | set(b), x, box=BOX)
    assert split["unique_touching"] == 0.0, split
    assert split["warning"] is None
    analytic = h.derivative_components(c, state)["lambda_bonded"]["bonded_mixed"]
    assert analytic == pytest.approx(split["total"], abs=1e-9)
    assert analytic == pytest.approx(split["core"], abs=1e-9)

    # and a plan that removes a junction term at the dummy end puts that energy on the lambda path
    # FB (particle 8) appears, so its dummy end is System A: that is where the plan would zero it
    removed = _junction_removed(sa, (1, 0, 8))          # C1-C0-FB, the appearing atom's junction
    split = fx.bonded_derivative_split(removed, sb, set(a) | set(b), x, box=BOX)
    assert abs(split["unique_touching"]) > fx.BONDED_JUNCTION_WARNING_KJ
    assert split["warning"] is not None and "on the lambda path" in split["warning"]
    h2 = build_hamiltonian(removed, sb, a, b)
    c2 = _context(h2.system, x)
    moved = h2.derivative_components(c2, state)["lambda_bonded"]["bonded_mixed"]
    assert moved == pytest.approx(split["total"], abs=1e-9)
    assert abs(moved - analytic) == pytest.approx(abs(split["unique_touching"]), abs=1e-9)
    print(f"\nbonded integrand: identical junctions {analytic:.3f} kJ/mol (core only); "
          f"one dummy-removed junction angle {moved:.3f}, of which "
          f"{split['unique_touching']:.3f} is the junction term")
