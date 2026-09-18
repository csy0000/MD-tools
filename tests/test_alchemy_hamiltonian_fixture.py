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
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_s3_fixture as fx  # noqa: E402
from md_tools.alchemy.hamiltonian import (AlchemicalHamiltonianError,  # noqa: E402
                                          build_hamiltonian)
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
@pytest.mark.parametrize("variant", ["full", "charges_only", "lj_only", "bonded_only", "tail"])
def test_energy_against_the_reference(periodic, variant, boundary_14):
    """Per component, by construction: each variant zeroes the other term families.

    `tail` grows the appearing group into a five-atom chain, so the region-internal rules --
    non-excluded internal pairs and internal 1-4s unscaled, and removed from the weighted Ewald
    sum -- have something to act on.
    """
    components = {"full": {}, "charges_only": {"lj": False, "bonded": False},
                  "lj_only": {"charges": False, "bonded": False},
                  "bonded_only": {"charges": False, "lj": False}, "tail": {"tail": True}}[variant]
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
    """U(0) is System A plus B's unscaled ghost terms; U(1) is System B plus A's.

    The ghost is exactly what stays at full strength for the region that is not there, summed by
    hand from the other end state: its internal non-excluded pairs and internal exceptions, and
    -- only under the Amber18 rule `unscaled` -- its 1-4s with the core. Under `scaled` a
    one-atom region leaves no ghost at all: U(0) IS System A.
    """
    sa, sb, a, b, x, h, c, kw = _setup(periodic, boundary_14, tail=tail)

    def ghost(system, region):
        """Exceptions touching the region, and non-excluded pairs inside it, as vacuum terms."""
        nb = next(f for f in system.getForces() if type(f).__name__ == "NonbondedForce")

        def r_of(i, j):
            d = x[j] - x[i]
            if periodic:
                d -= BOX * np.round(d / BOX)
            return float(np.linalg.norm(d))

        def pair(r, qq, sg, ep):
            return fx.K_COULOMB * qq / r + 4 * ep * ((sg / r) ** 12 - (sg / r) ** 6)
        total, excepted = 0.0, set()
        for k in range(nb.getNumExceptions()):
            i, j, qq, sg, ep = nb.getExceptionParameters(k)
            excepted.add((min(i, j), max(i, j)))
            inside = i in region and j in region
            if inside or ((i in region or j in region) and boundary_14 == "unscaled"):
                total += pair(r_of(i, j), qq._value, sg._value, ep._value)
        for i, j in itertools.combinations(sorted(region), 2):
            if (i, j) not in excepted:
                qi, si, ei = (v._value for v in nb.getParticleParameters(i))
                qj, sj, ej = (v._value for v in nb.getParticleParameters(j))
                total += pair(r_of(i, j), qi * qj, 0.5 * (si + sj), math.sqrt(ei * ej))
        return total

    e_a = _context(sa, x).getState(getEnergy=True).getPotentialEnergy()._value
    e_b = _context(sb, x).getState(getEnergy=True).getPotentialEnergy()._value
    g_a, g_b = ghost(sa, a), ghost(sb, b)
    if boundary_14 == "unscaled":
        assert g_a != 0.0 and g_b != 0.0
    elif not tail:
        assert g_a == 0.0 and g_b == 0.0
    else:
        assert g_a == 0.0 and g_b != 0.0         # only the tail has an inside
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(e_a + g_b, abs=TOL[periodic])
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(e_b + g_a, abs=TOL[periodic])


def _richardson(estimates):
    """Richardson extrapolation of each ADJACENT pair of estimates, steps a factor 10 apart, both
    O(h^2) (central, and the second-order one-sided form): (100 D(h/10) - D(h)) / 99 cancels the
    h^2 term. Every pair, not only the smallest steps: with the dispersion correction on, OpenMM
    re-integrates its long-range correction numerically whenever lambda_sterics changes, which
    puts ~5e-9 kJ/mol of noise in the energy, so at h = 1e-4 a difference is noise-limited and
    the (1e-2, 1e-3) pair is the informative one. Reported beside the raw differences."""
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
    # With the dispersion correction on, the independent reference has no tail term; its
    # finite difference is then compared after adding the exact linear dispersion slope, which
    # `test_dispersion_mixes_the_end_states_own_corrections` pins to OpenMM's own corrections.
    slope = 0.0
    if dispersion:
        parts = h.derivative_components(c, _state((0.5, 0.5, 0.5)))
        slope = parts["lambda_sterics"]["dispersion"]
        assert slope != 0.0
    # The tail's first atom starts 0.22 nm from the Cl-, so at lambda_sterics = 1 the derivative
    # is ~2.4e4 kJ/mol and the O(h^2) truncation needs h = 1e-4 to fall under 1e-5 relative, in
    # PME as in vacuum.
    steps = (1e-2, 1e-3, 1e-4)
    report = []
    for base in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = _state((base, base, base))
        analytic = h.derivatives(c, state)
        for k, name in enumerate(NAMES):
            def own(v, k=k):
                t = [base] * 3
                t[k] = v
                return h.energy(c, _state(t))

            def ref(v, k=k):
                t = [base] * 3
                t[k] = v
                return fx.reference_energy(sa, sb, a, b, x, _state(t), **kw)["total"]
            own_fd = [_fd(own, base, s) for s in steps]
            ref_fd = [_fd(ref, base, s) + (slope if name == "lambda_sterics" else 0.0)
                      for s in steps]
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
                                 "scaled -- PROVISIONAL, the user's decision is pending (2026-09-19)"}
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
