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
    """Richardson extrapolation of the last two estimates, steps a factor 10 apart, both O(h^2)
    (central, and the second-order one-sided form): (100 D(h/10) - D(h)) / 99 cancels the h^2
    term. It is reported beside the raw differences, never instead of them."""
    return (100.0 * estimates[-1] - estimates[-2]) / 99.0


def _fd(f, lam, h):
    """Central in the interior; second-order one-sided at an end point."""
    if lam - h < 0:
        return (-3 * f(lam) + 4 * f(lam + h) - f(lam + 2 * h)) / (2 * h)
    if lam + h > 1:
        return (3 * f(lam) - 4 * f(lam - h) + f(lam - 2 * h)) / (2 * h)
    return (f(lam + h) - f(lam - h)) / (2 * h)


@pytest.mark.parametrize("periodic, tail, boundary_14", [
    (False, False, "scaled"), (False, True, "scaled"), (False, True, "unscaled"),
    (True, False, "scaled"), (True, True, "scaled")],
    ids=["vacuum", "vacuum-tail", "vacuum-tail-unscaled14", "pme", "pme-tail"])
def test_derivatives_against_finite_differences(periodic, tail, boundary_14):
    """Every component, at the five lambdas, against finite differences of TWO energies.

    * of this Hamiltonian's own energy (the definition: dU/dlambda_k is the limit of this), and
    * of the independent reference's energy -- which catches a term the Hamiltonian's energy and
      derivative both omit, which differencing its own energy never could.

    Several step sizes; the error must shrink with h until round-off, and the best must be within
    tolerance. The table is printed (`-s`) as the convergence report.
    """
    sa, sb, a, b, x, h, c, kw = _setup(periodic, boundary_14, tail=tail)
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
            ref_fd = [_fd(ref, base, s) for s in steps]
            own_err = [abs(analytic[name] - d) for d in own_fd] + [abs(analytic[name] - _richardson(own_fd))]
            ref_err = [abs(analytic[name] - d) for d in ref_fd] + [abs(analytic[name] - _richardson(ref_fd))]
            report.append((base, name, analytic[name], own_err, ref_err))
            scale = max(1.0, abs(analytic[name]))
            # truncation O(h^2) at 1e-2 must fall by >= ~10x by 1e-3 unless already at round-off
            assert own_err[1] <= max(own_err[0] / 10, 1e-7 * scale), (base, name, own_err)
            assert min(own_err) < {False: 1e-6, True: 1e-5}[periodic] * scale, (base, name, own_err)
            assert min(ref_err) < {False: 1e-6, True: 3e-4}[periodic] * scale, (base, name, ref_err)
    print(f"\n{'pme' if periodic else 'vacuum'}{' tail' if tail else ''}: lambda, component, dU/dlambda, "
          "|err| vs own FD, then vs reference FD, at h = 1e-2, 1e-3, 1e-4, then Richardson(1e-3, 1e-4)")
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
                             "sc_boundary_14_amber": "gti_add_sc = 1 (pmemd 20+ default)"}
    assert r["rules"]["boundary_14"] == "scaled"
    assert r["particles"]["a_only"] == [7] and r["particles"]["b_only"] == [8]
    assert r["particles"]["common_lj_changing"] == [0]
    assert r["nonbonded"]["kappa_nm_inv"] == pytest.approx(kw["kappa"], rel=1e-12)
    assert {m["type"] for m in r["bonded_mixed_forces"]} == {
        "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"}
