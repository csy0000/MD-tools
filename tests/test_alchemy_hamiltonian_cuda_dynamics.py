"""The softcore Hamiltonian propagated on CUDA: its forces are the gradient of its energy, and a
state change reaches a live device Context.

Deliberately short. What is under test is that the device integrates THIS Hamiltonian -- forces
consistent with the energy it reports, at an interior lambda where softcore is active and at the
end states -- not the quality of any sampling. Only CUDA is named here; the fixed-coordinate
comparison with Reference is `test_alchemy_hamiltonian_cuda.py`.

DESELECTED on a machine without CUDA (the `gpu` marker), never skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_s3_fixture as fx  # noqa: E402
from md_tools.alchemy.hamiltonian import build_hamiltonian  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")


def _cuda_context(system, x, integrator, precision="double"):
    c = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"),
                       {"Precision": precision})
    assert c.getPlatform().getName() == "CUDA"
    c.setPositions(x)
    return c


def _nve_excursion(system, x, set_state):
    """Max |E_total - E_total(0)| over 2000 x 0.2 fs of velocity Verlet, double precision, after
    the same minimisation and the same seeded velocities."""
    c = _cuda_context(system, x, openmm.VerletIntegrator(0.0002))
    set_state(c)
    openmm.LocalEnergyMinimizer.minimize(c, 10.0, 200)
    c.setVelocitiesToTemperature(300.0, 20260919)

    def total():
        st = c.getState(getEnergy=True)
        return st.getPotentialEnergy()._value + st.getKineticEnergy()._value
    e0 = total()
    energies = []
    for _ in range(20):
        c.getIntegrator().step(100)
        energies.append(total())
    assert np.all(np.isfinite(energies))
    return max(abs(e - e0) for e in energies)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_nve_conserves_the_softcore_hamiltonian_on_cuda(lam):
    """A SMOKE TEST, NOT EVIDENCE: velocity Verlet in double precision stays finite and its
    total-energy excursion stays within 2 x that of the plain end-state System A.

    It is not evidence of force/energy consistency. On this fixture NVE was shown to have no power
    (CPU, 0.05 fs, 100 fs): a deliberate 20% energy step injected into softcore_b_lj gave the same
    excursion as the correct Hamiltonian and as System A (0.071 vs 0.072 kJ/mol at lambda 0.5).
    That job is test_alchemy_hamiltonian_cuda.py::test_cuda_forces_are_the_gradient_of_the_energy.

    CALIBRATED, and why it changed (disclosed in handoffs/S3.md). The first form used an absolute
    bound, 5e-4 x KE + 0.05 kJ/mol, never calibrated. The second CUDA run failed it at ~0.5 kJ/mol
    for every lambda, and a CPU diagnostic then showed the plain System A -- no alchemical force at
    all -- excursing by 0.495 kJ/mol under the same protocol (0.22 at 0.1 fs): the stiff flexible
    O-H bonds at 0.2 fs, not the softcore forces. Forces that were not the gradient of the reported
    energy would show up as an excursion beyond System A's; the bound is 2 x System A's + 0.05
    kJ/mol, written down before the next run.
    """
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    plain = _nve_excursion(sa, x, lambda c: None)
    state = dict(zip(NAMES, (lam, lam, lam)))
    softcore = _nve_excursion(h.system, x, lambda c: h.set_state(c, state))
    print(f"\nNVE lambda={lam}: Hamiltonian {softcore:.3f}, plain System A {plain:.3f} kJ/mol")
    assert softcore <= 2 * plain + 0.05, (lam, softcore, plain)


def test_set_state_reaches_a_live_cuda_context():
    """After dynamics, moving the state on the running Context gives the energy a freshly made
    Context reports at that state and those coordinates."""
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    live = _cuda_context(h.system, x, openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001), "mixed")
    h.set_state(live, dict(zip(NAMES, (0.5, 0.5, 0.5))))
    openmm.LocalEnergyMinimizer.minimize(live, 10.0, 200)
    live.getIntegrator().step(500)
    positions = live.getState(getPositions=True).getPositions(asNumpy=True)._value
    assert np.all(np.isfinite(positions))
    for t in ((0.0, 0.0, 0.0), (0.8, 0.3, 0.6), (1.0, 1.0, 1.0)):
        state = dict(zip(NAMES, t))
        moved = h.energy(live, state)
        fresh = _cuda_context(h.system, positions, openmm.VerletIntegrator(0.001), "mixed")
        assert moved == pytest.approx(h.energy(fresh, state), rel=1e-6, abs=1e-3), t


@pytest.mark.parametrize("precision", ["mixed", "double"])
def test_npt_on_cuda_leaves_nothing_stale(precision):
    """NPT on the device: a MonteCarloBarostat moves the box during a run, and the Hamiltonian's
    energy at every state still equals a fresh Context's at that box and those coordinates.

    The Reference twin (test_alchemy_hamiltonian_fixture.py::test_a_barostat_moving_the_box_leaves
    _nothing_stale) checks the same property in float64 and catches it on any machine; it is not
    CUDA evidence, and NPT on CUDA was an uncovered row until this test. What only the device can
    show: the PME grid, the dispersion coefficients and the softcore delta's kappa are fixed at
    build, and CUDA rebuilds none of them when the barostat rescales the box.

    THE FIXTURE IS THE ONE WITHOUT THE CLASH, and the first run of this test is why. With the tail
    fixture -- an appearing atom 0.22 nm from the Cl- -- 500 steps at 0.5 fs under a barostat blew
    up to 1e15 kJ/mol, and the comparison then "passed" in mixed precision only because an absolute
    tolerance is meaningless at that magnitude. The subject here is whether anything box-dependent
    goes stale, not whether an overlapped start survives; the clash geometry is covered by the
    energy, force and derivative rows, which evaluate it rather than integrate it. So: no clash, a
    sanity gate on the energy before any comparison, and -- since 2026-09-30 -- no absolute or
    relative tolerance at all, but a RATIO against the staleness signal the test itself computes.
    The power check runs FIRST and asserts that signal exists, because a ratio fails OPEN where an
    absolute bound fails closed: a tiny gap beats a tiny signal, so a fixture that quietly stopped
    producing one would pass forever.
    """
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    for s in (sa, sb):
        s.addForce(openmm.MonteCarloBarostat(1.0, 300.0, 5))
    # EVERY STOCHASTIC SOURCE IS SEEDED, and the barostat too. Until 2026-09-29 none of them was:
    # OpenMM reads an unset seed as "choose randomly", so this test compared at a DIFFERENT
    # configuration and a different box on every run, and S0 found it failing intermittently on
    # card 8 -- once in five runs, by 3.9e-4 against a 3.2e-4 bound, a 1.2x miss. A test with no
    # fixed input cannot be evidence in either direction: "it passed" and "it failed" were both
    # true of it, and at ~20% it is frequent enough to reach a release gate and be dismissed as a
    # flake, rare enough that four clean runs would have closed the question. The NVE smoke test
    # in this file seeded its velocities from the day it was written; this one did not.
    #
    # THE SEEDS WERE VERIFIED TO FIX THE INPUT, and the verification itself nearly misled me:
    # on the CPU platform at OPENMM_CPU_THREADS=2 two seeded runs still diverged by 2.29
    # kJ/mol and 0.41 nm^3, which reads exactly like seeding that did not take. At ONE thread
    # every stage is bit-identical -- minimisation, 10 steps, 10 steps with the barostat, and
    # the full 500 steps: dE = dV = 0.000e+00. The residue was the CPU platform summing force
    # reductions in thread-completion order (`md_tools.remd.driver` records the same), with
    # 500 steps of Langevin dynamics amplifying it.
    #
    # AND THE SEEDS ARE NOT ENOUGH ON CUDA, measured by S0 on card 7: three replications of this
    # setup diverged ALREADY AFTER MINIMISATION -- -1847.52 / -1833.69 / -1861.23 kJ/mol -- before
    # a single seeded step. `LocalEnergyMinimizer` takes no seed and is not reproducible there, in
    # mixed OR double, while single-point energies and forces over 50 fresh Context pairs are
    # bit-identical. So minimisation is REMOVED from this setup rather than seeded: what this test
    # asks is whether anything box-dependent goes stale, and that question never needed a
    # minimised start. The seeds stay because they make the rest of the input fixed and cost
    # nothing.
    integrator = openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.0005)
    integrator.setRandomNumberSeed(20260929)
    for s in (sa, sb):
        for force in s.getForces():
            if isinstance(force, openmm.MonteCarloBarostat):
                force.setRandomNumberSeed(20260929)
    h = build_hamiltonian(sa, sb, a, b)
    live = _cuda_context(h.system, x, integrator, precision)
    state = dict(zip(NAMES, (0.5, 0.5, 0.5)))
    h.set_state(live, state)
    live.setVelocitiesToTemperature(300.0, 20260929)
    start = live.getState(getEnergy=True).getPotentialEnergy()._value
    v0 = live.getState().getPeriodicBoxVolume()._value
    live.getIntegrator().step(500)
    st = live.getState(getPositions=True, getEnergy=True)
    energy = st.getPotentialEnergy()._value
    assert np.isfinite(energy) and abs(energy) < abs(start) + 5000.0, (precision, start, energy)
    assert st.getPeriodicBoxVolume()._value != v0, "the barostat never moved the box"
    box, pos = st.getPeriodicBoxVectors(), st.getPositions(asNumpy=True)._value
    # NO ABSOLUTE TOLERANCE. This test used `abs=rel*|E| + 1e-6` with rel = 1e-6 at mixed, and S0
    # measured that bound sitting directly on top of the noise it was bounding: a 3.9e-4 kJ/mol
    # gap against a 3.2e-4 bound, failing once in five runs on CUDA. Widening it would have needed
    # a number nobody had measured on THIS fixture, and borrowing windows.py's 2e-5 -- taken from
    # a 1038-atom solvated system -- would have been the same error in the other direction.
    #
    # The test does not need one. It already computes the signal it is looking for: the same
    # comparison against a Context left at the ORIGINAL box. So the assertion is a RATIO --
    # whatever the correct comparison costs on this device at this precision, staleness must cost
    # hugely more -- and it calibrates itself on every platform, in the same run, with no constant
    # to go stale. Measured: on Reference the correct comparison is 0.000e+00 against a 3.26e+01
    # staleness signal; on CUDA mixed S0 measured 3.9e-4 against the same ~3e+01, a ratio of ~1e5.
    # RATIO_FLOOR is 1e3, two orders below the smallest ratio either platform has shown.
    ratio_floor = 1e3

    def fresh_energy(t, vectors):
        fresh = _cuda_context(h.system, pos, openmm.VerletIntegrator(0.001), precision)
        fresh.setPeriodicBoxVectors(*vectors)
        return h.energy(fresh, dict(zip(NAMES, t)))
    stale = [v._value for v in sa.getDefaultPeriodicBoxVectors()]
    for t in ((0.0, 0.0, 0.0), (0.25, 0.5, 0.75), (1.0, 1.0, 1.0)):
        moved = h.energy(live, dict(zip(NAMES, t)))
        gap = abs(moved - fresh_energy(t, box))              # nothing should be stale here
        signal = abs(moved - fresh_energy(t, stale))         # everything box-dependent IS stale here
        # the power check first, so a fixture that stopped producing a signal cannot pass quietly
        assert signal > 1.0, (precision, t, "the stale-box comparison produced no signal, so this "
                                            "test could not have failed either way", signal)
        assert gap * ratio_floor < signal, (precision, t, moved, gap, signal, gap / signal)
