"""What an exchange does to the sampler state, on two real Contexts.

The arithmetic tests establish that the acceptance probability is right. They say nothing about the
part that actually moves the sampler: on acceptance, positions, box vectors and velocities must move
together as one complete state; on rejection, all three must be restored exactly. A criterion that
computes the right probability and then swaps two of the three quantities samples nothing anyone can
name, and every energy in the log would still look plausible.

Reference platform and double precision throughout, because "restored exactly" is the claim -- on
CUDA in mixed precision a round trip through a State is not bitwise, and the test would be asserting
the platform's tolerance rather than the code's correctness.
"""

from __future__ import annotations

import math

import pytest

openmm = pytest.importorskip("openmm")
import numpy as np  # noqa: E402
from openmm import unit  # noqa: E402
from openmm import app  # noqa: E402

from md_templates.openmm.rest2 import attempt_rest2_exchange  # noqa: E402


class _FixedRng:
    """A random source with a scripted value, so acceptance is decided by the test, not by luck."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def _make(charge, box_nm, seed, sigma=0.3):
    """A three-particle periodic System whose energy depends on `charge` and `sigma`."""
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*(np.eye(3) * box_nm) * unit.nanometer)
    for _ in range(3):
        system.addParticle(12.0 * unit.amu)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.9 * unit.nanometer)
    for _ in range(3):
        force.addParticle(charge, sigma, 0.5)
    system.addForce(force)

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("UNL", chain)
    for _ in range(3):
        topology.addAtom("C", app.element.carbon, residue)

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
    simulation = app.Simulation(
        topology, system, integrator, openmm.Platform.getPlatformByName("Reference"))
    simulation.context.setPositions(
        np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]) * unit.nanometer)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
    return simulation


def _snapshot(sim):
    state = sim.context.getState(getPositions=True, getVelocities=True)
    return (np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
            np.array(state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond)),
            np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)))


# -------------------------------------------------------------------------------------------
def test_an_accepted_exchange_swaps_positions_box_and_velocities_together():
    """All three, or the replicas end up holding a state that never existed."""
    a = _make(0.0, 2.4, seed=11)          # identical Hamiltonians -> delta = 0 -> log_accept = 0
    b = _make(0.0, 2.6, seed=22)          # different boxes, so a swap is observable
    before_a, before_b = _snapshot(a), _snapshot(b)

    result = attempt_rest2_exchange(a, b, beta0=0.4, rng=_FixedRng(0.0), pressure_bar=1.0)
    assert result["accepted"] is True, result["log_acceptance"]

    after_a, after_b = _snapshot(a), _snapshot(b)
    np.testing.assert_allclose(after_a[0], before_b[0], atol=1e-9)   # positions crossed
    np.testing.assert_allclose(after_b[0], before_a[0], atol=1e-9)
    np.testing.assert_allclose(after_a[1], before_b[1], atol=1e-9)   # velocities crossed
    np.testing.assert_allclose(after_b[1], before_a[1], atol=1e-9)
    np.testing.assert_allclose(after_a[2], before_b[2], atol=1e-9)   # box crossed
    np.testing.assert_allclose(after_b[2], before_a[2], atol=1e-9)


def test_a_rejected_exchange_restores_all_three_exactly():
    """Rejection must be a no-op. A partial restore leaves a replica mid-swap."""
    # The Hamiltonians must genuinely DIFFER, or the exchange is always favourable: with identical
    # Hamiltonians E_ij == E_jj and E_ji == E_ii, so delta is exactly zero whatever the geometry
    # and the move is accepted. Two earlier attempts failed here for that reason.
    a = _make(0.0, 2.4, seed=33, sigma=0.30)
    b = _make(0.0, 2.6, seed=44, sigma=0.75)
    # Replica i carries a near-overlapping pair. Under j's much larger sigma that configuration is
    # an enormous repulsion, so E_ji dominates and the exchange is refused outright.
    a.context.setPositions(
        np.array([[0.0, 0.0, 0.0], [0.09, 0.0, 0.0], [0.0, 0.4, 0.0]]) * unit.nanometer)
    before_a, before_b = _snapshot(a), _snapshot(b)

    # rng returns ~1, so log(rng) ~ 0, which is never below a strongly negative log_accept
    result = attempt_rest2_exchange(a, b, beta0=0.4, rng=_FixedRng(0.999999), pressure_bar=1.0)
    assert result["accepted"] is False, result["log_acceptance"]
    assert result["log_acceptance"] < 0.0

    after_a, after_b = _snapshot(a), _snapshot(b)
    for before, after, label in ((before_a, after_a, "i"), (before_b, after_b, "j")):
        np.testing.assert_allclose(after[0], before[0], atol=1e-10, err_msg=f"positions {label}")
        np.testing.assert_allclose(after[1], before[1], atol=1e-10, err_msg=f"velocities {label}")
        np.testing.assert_allclose(after[2], before[2], atol=1e-10, err_msg=f"box {label}")


def test_the_recorded_energies_come_from_the_cross_evaluation():
    """E_ij and E_ji must be each configuration evaluated under the OTHER Hamiltonian."""
    a = _make(0.0, 2.4, seed=55, sigma=0.30)
    b = _make(0.0, 2.4, seed=66, sigma=0.55)
    result = attempt_rest2_exchange(a, b, beta0=0.4, rng=_FixedRng(0.999999), pressure_bar=1.0)
    for key in ("energy_i_on_i_kj_mol", "energy_j_on_j_kj_mol",
                "energy_i_on_j_kj_mol", "energy_j_on_i_kj_mol"):
        assert math.isfinite(result[key]), key
    # different Hamiltonians, so the cross terms cannot both equal the diagonal ones
    assert result["energy_i_on_j_kj_mol"] != pytest.approx(result["energy_j_on_j_kj_mol"])


def test_velocities_are_not_rescaled_on_acceptance():
    """REST2 replicas share one physical temperature; rescaling would inject unaccounted energy."""
    a = _make(0.0, 2.4, seed=77)
    b = _make(0.0, 2.4, seed=88)
    before_a, before_b = _snapshot(a), _snapshot(b)
    ke_before = {round(float(np.sum(before_a[1] ** 2)), 9),
                 round(float(np.sum(before_b[1] ** 2)), 9)}

    result = attempt_rest2_exchange(a, b, beta0=0.4, rng=_FixedRng(0.0), pressure_bar=1.0)
    assert result["accepted"] is True

    after_a, after_b = _snapshot(a), _snapshot(b)
    ke_after = {round(float(np.sum(after_a[1] ** 2)), 9),
                round(float(np.sum(after_b[1] ** 2)), 9)}
    assert ke_after == ke_before, "the velocity magnitudes changed; they were rescaled, not swapped"


def test_a_nonperiodic_exchange_records_no_volume_and_no_pv():
    """Implicit solvent has no box, so there is no pV term to compute -- not a zero one."""
    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0 * unit.amu)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for _ in range(3):
        force.addParticle(0.0, 0.3, 0.5)
    system.addForce(force)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("UNL", chain)
    for _ in range(3):
        topology.addAtom("C", app.element.carbon, residue)

    sims = []
    for seed in (101, 202):
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
        sim = app.Simulation(topology, system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
        sim.context.setPositions(
            np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]) * unit.nanometer)
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
        sims.append(sim)

    result = attempt_rest2_exchange(sims[0], sims[1], beta0=0.4, rng=_FixedRng(0.0),
                                    pressure_bar=1.0)
    assert result["periodic"] is False
    assert result["volume_i_nm3"] is None and result["volume_j_nm3"] is None
    assert result["pv_i_kj_mol"] is None and result["pv_j_kj_mol"] is None
    assert result["pressure_i_bar"] is None, "a pressure cannot apply without a volume"
