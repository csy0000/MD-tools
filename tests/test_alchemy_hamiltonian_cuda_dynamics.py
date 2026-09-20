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
    sanity gate on the energy before any comparison, and a relative tolerance.
    """
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    for s in (sa, sb):
        s.addForce(openmm.MonteCarloBarostat(1.0, 300.0, 5))
    h = build_hamiltonian(sa, sb, a, b)
    live = _cuda_context(h.system, x, openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.0005), precision)
    state = dict(zip(NAMES, (0.5, 0.5, 0.5)))
    h.set_state(live, state)
    openmm.LocalEnergyMinimizer.minimize(live, 1.0, 2000)
    start = live.getState(getEnergy=True).getPotentialEnergy()._value
    v0 = live.getState().getPeriodicBoxVolume()._value
    live.getIntegrator().step(500)
    st = live.getState(getPositions=True, getEnergy=True)
    energy = st.getPotentialEnergy()._value
    assert np.isfinite(energy) and abs(energy) < abs(start) + 5000.0, (precision, start, energy)
    assert st.getPeriodicBoxVolume()._value != v0, "the barostat never moved the box"
    box, pos = st.getPeriodicBoxVectors(), st.getPositions(asNumpy=True)._value
    rel = {"mixed": 1e-6, "double": 1e-10}[precision]

    def fresh_energy(t, vectors):
        fresh = _cuda_context(h.system, pos, openmm.VerletIntegrator(0.001), precision)
        fresh.setPeriodicBoxVectors(*vectors)
        return h.energy(fresh, dict(zip(NAMES, t)))
    stale = [v._value for v in sa.getDefaultPeriodicBoxVectors()]
    for t in ((0.0, 0.0, 0.0), (0.25, 0.5, 0.75), (1.0, 1.0, 1.0)):
        moved = h.energy(live, dict(zip(NAMES, t)))
        again = fresh_energy(t, box)
        tol = rel * max(1.0, abs(again)) + 1e-6
        assert moved == pytest.approx(again, abs=tol), (precision, t, moved, again, tol)
        # the check can fail: the same comparison against a Context left at the ORIGINAL box
        assert abs(moved - fresh_energy(t, stale)) > 100 * tol, (precision, t)
