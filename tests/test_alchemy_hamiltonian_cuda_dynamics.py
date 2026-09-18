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


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_nve_conserves_the_softcore_hamiltonian_on_cuda(lam):
    """Velocity Verlet in double precision: total energy drift over 2000 x 0.2 fs steps stays
    within 0.05% of the kinetic energy scale. Forces that were not the gradient of the reported
    energy -- a missing derivative term in a custom expression, a delta not cancelling its
    NonbondedForce term -- show up as drift."""
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    c = _cuda_context(h.system, x, openmm.VerletIntegrator(0.0002))
    state = dict(zip(NAMES, (lam, lam, lam)))
    h.set_state(c, state)
    openmm.LocalEnergyMinimizer.minimize(c, 10.0, 200)
    c.setVelocitiesToTemperature(300.0, 20260919)

    def total():
        st = c.getState(getEnergy=True)
        return st.getPotentialEnergy()._value + st.getKineticEnergy()._value, st.getKineticEnergy()._value
    e0, k0 = total()
    energies = []
    for _ in range(20):
        c.getIntegrator().step(100)
        energies.append(total()[0])
    drift = max(abs(e - e0) for e in energies)
    assert np.all(np.isfinite(energies))
    assert drift < 5e-4 * k0 + 0.05, (lam, drift, k0)


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
