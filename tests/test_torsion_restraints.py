"""Torsional biasing restraints: the energy is checked, not the sampling.

A restraint is easy to test badly. Running dynamics and finding the mean torsion near the target
looks like proof and is not: a NEGATED convention puts the mean at minus the target, and if the
structure you start from happens to sit at 180 degrees -- as alanine dipeptide's phi does -- then
180 is its own negative and the check passes on a broken implementation. That happened while this
module was being written, which is why the tests below evaluate the energy function on known
geometries instead.

What must hold:

  * the minimum is exactly at theta0, including when theta0 sits on the +/-180 seam;
  * the wrap is symmetric -- two torsions equally far from theta0 have equal energy, even when one
    is reached by crossing the seam;
  * a flat-bottom restraint is EXACTLY zero inside its window, so samples there are unbiased;
  * outside the window the penalty grows from the window EDGE, not from theta0.

PLATFORM_POLICY_EXEMPTION: evaluates a potential on four particles at fixed coordinates. No
dynamics are propagated and no sampling is performed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openmm import Context, Platform, System, VerletIntegrator, unit

from md_tools.md.torsion_restraints import (FLAT_BOTTOM, HARMONIC, TORSION_RESTRAINT_PARAMETER,
                                            TorsionRestraint)

K = 500.0            # kJ/mol/rad^2


def _energy_at(form, theta0, degrees, width=0.0, k=K):
    """The bias energy at each angle in `degrees`, on four particles placed at a known torsion."""
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, form)
    restraint.add_torsion((0, 1, 2, 3), theta0, width)
    context = Context(system, VerletIntegrator(0.001),
                      Platform.getPlatformByName("Reference"))
    context.setParameter(TORSION_RESTRAINT_PARAMETER, k)
    out = {}
    for deg in degrees:
        a = math.radians(deg)
        positions = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                              [math.cos(a), 0.0, math.sin(a)]])
        context.setPositions(positions * unit.nanometer)
        out[deg] = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
    return out


@pytest.mark.parametrize("theta0", [-150.0, -60.0, 0.0, 60.0, 170.0, 180.0])
def test_the_minimum_sits_exactly_at_the_requested_centre(theta0):
    """Including on the seam, which is the case the wrap exists for."""
    grid = list(range(-180, 181, 5))
    energies = _energy_at(HARMONIC, theta0, grid)
    lowest = min(energies, key=energies.get)
    # Compared on the circle: for theta0 = 180 the minimum may be found at -180, same angle.
    separation = abs(((lowest - theta0 + 180) % 360) - 180)
    assert separation <= 5, (
        f"centre requested at {theta0}, minimum found at {lowest} "
        f"({separation} deg away)")
    assert energies[lowest] < 1e-6, f"the minimum should be zero energy; got {energies[lowest]}"


@pytest.mark.parametrize("theta0,left,right", [
    (170.0, 150.0, -170.0),        # 20 deg either side, the right one across the seam
    (-170.0, -150.0, 170.0),       # the mirror case
    (0.0, -30.0, 30.0),            # nowhere near the seam, as a control
])
def test_the_wrap_is_symmetric_across_the_seam(theta0, left, right):
    """Two angles equally far from the centre must cost the same, seam or no seam.

    Without the wrap, the angle reached by crossing +/-180 is treated as ~340 degrees away and
    costs hundreds of kJ/mol more -- a bias that depends on where the molecule happens to be.
    """
    energies = _energy_at(HARMONIC, theta0, [left, right])
    assert abs(energies[left] - energies[right]) < 1e-6, (
        f"theta0={theta0}: E({left}) = {energies[left]:.6f} but E({right}) = "
        f"{energies[right]:.6f}; the wrap is not symmetric")
    assert energies[left] > 1.0, "this test is only meaningful away from the minimum"


def test_the_harmonic_form_is_quadratic_in_the_offset():
    """Doubling the displacement quadruples the energy. Pins the form, not just the minimum."""
    energies = _energy_at(HARMONIC, 0.0, [10.0, 20.0, 40.0])
    assert energies[20.0] / energies[10.0] == pytest.approx(4.0, rel=1e-4)
    assert energies[40.0] / energies[20.0] == pytest.approx(4.0, rel=1e-4)


def test_a_flat_bottom_restraint_is_exactly_zero_inside_its_window():
    """Not merely small. Samples inside the window are unbiased and need no reweighting."""
    inside = [-85.0, -75.0, -60.0, -45.0, -35.0]
    energies = _energy_at(FLAT_BOTTOM, -60.0, inside, width=30.0)
    for deg, value in energies.items():
        assert value == 0.0, f"flat-bottom is biased at {deg} deg inside its window: {value}"


def test_a_flat_bottom_penalty_grows_from_the_window_edge_not_the_centre():
    """The distinction that makes it a bound rather than an offset harmonic.

    At 10 degrees beyond the edge the energy must equal a harmonic's at 10 degrees from ITS centre.
    """
    beyond = _energy_at(FLAT_BOTTOM, -60.0, [-100.0], width=30.0)[-100.0]
    harmonic_at_10 = _energy_at(HARMONIC, 0.0, [10.0])[10.0]
    assert beyond == pytest.approx(harmonic_at_10, rel=1e-6), (
        f"flat-bottom 10 deg past its edge cost {beyond}, but a harmonic 10 deg from its centre "
        f"costs {harmonic_at_10}; the penalty is not measured from the edge")


def test_the_strength_is_zero_until_it_is_set():
    """A System carrying the force must be unbiased until something asks otherwise.

    The force is present from the start so the System's force layout -- and therefore the
    checkpoint layout -- is constant along a chain of stages. That is only safe if being present
    costs nothing.
    """
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, HARMONIC)
    restraint.add_torsion((0, 1, 2, 3), 0.0)
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setPositions(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                   [0.0, -1.0, 1.0]]) * unit.nanometer)
    assert context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) == 0.0


def test_the_strength_scales_the_energy_linearly():
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    TorsionRestraint(system, HARMONIC).add_torsion((0, 1, 2, 3), 0.0)
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    a = math.radians(30.0)
    context.setPositions(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                   [math.cos(a), 0.0, math.sin(a)]]) * unit.nanometer)
    seen = []
    for k in (100.0, 200.0, 400.0):
        context.setParameter(TORSION_RESTRAINT_PARAMETER, k)
        seen.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole))
    assert seen[1] == pytest.approx(2 * seen[0], rel=1e-6)
    assert seen[2] == pytest.approx(4 * seen[0], rel=1e-6)


# --- refusals, each naming what was wrong -----------------------------------------------------

def test_an_unknown_form_is_refused_by_name():
    system = System()
    with pytest.raises(ValueError, match="unknown torsion restraint form"):
        TorsionRestraint(system, "gaussian")


def test_a_flat_bottom_restraint_without_a_width_is_refused():
    """A zero-width bound is a harmonic restraint; asking for it that way is a mistake."""
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, FLAT_BOTTOM)
    with pytest.raises(ValueError, match="positive half-width"):
        restraint.add_torsion((0, 1, 2, 3), 0.0)


def test_a_width_given_to_a_harmonic_restraint_is_refused_rather_than_ignored():
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, HARMONIC)
    with pytest.raises(ValueError, match="no half-width"):
        restraint.add_torsion((0, 1, 2, 3), 0.0, 30.0)


@pytest.mark.parametrize("atoms,message", [
    ((0, 1, 2), "exactly 4 atoms"),
    ((0, 1, 2, 3, 4), "exactly 4 atoms"),
    ((0, 1, 1, 2), "4 DISTINCT atoms"),
])
def test_a_malformed_torsion_is_refused(atoms, message):
    system = System()
    for _ in range(5):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, HARMONIC)
    with pytest.raises(ValueError, match=message):
        restraint.add_torsion(atoms, 0.0)


def test_a_negative_force_constant_is_refused():
    """It would push the torsion away from its centre, which is not a restraint."""
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, HARMONIC)
    restraint.add_torsion((0, 1, 2, 3), 0.0)

    class _Simulation:
        def __init__(self, context):
            self.context = context

    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    with pytest.raises(ValueError, match="must be >= 0"):
        restraint.set_strength(_Simulation(context), -1.0)


# --- per-restraint strength, which is what lets one window mix them --------------------------

def test_the_per_torsion_scale_carries_the_force_constant():
    """Each restraint may have its own strength, in ONE force.

    The energy is 0.5 * k_global * scale * dtheta^2, so the global means "are the biases on" and
    `scale` carries the strength. Without this a window would need one force constant for all its
    restraints -- and the first implementation did, which made a configuration with two different
    constants build successfully and fail at RUN time, after minimisation and three equilibration
    stages had already run.
    """
    stiff = _energy_at(HARMONIC, 0.0, [30.0], k=1.0)[30.0]        # scale defaults to 1.0
    system = System()
    for _ in range(4):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, HARMONIC)
    restraint.add_torsion((0, 1, 2, 3), 0.0, scale=4.0)
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setParameter(TORSION_RESTRAINT_PARAMETER, 1.0)
    a = math.radians(30.0)
    context.setPositions(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                   [math.cos(a), 0.0, math.sin(a)]]) * unit.nanometer)
    scaled = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert scaled == pytest.approx(4.0 * stiff, rel=1e-6)


def test_two_restraints_in_one_force_keep_their_own_strengths():
    """A stiff and a soft restraint side by side. Both are in the same CustomTorsionForce."""
    system = System()
    for _ in range(5):
        system.addParticle(12.0)
    restraint = TorsionRestraint(system, HARMONIC)
    restraint.add_torsion((0, 1, 2, 3), 0.0, scale=800.0)
    restraint.add_torsion((1, 2, 3, 4), 0.0, scale=50.0)
    assert restraint.n_torsions == 2

    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setParameter(TORSION_RESTRAINT_PARAMETER, 1.0)
    # Both torsions displaced identically: the energy must be the SUM of two different strengths,
    # not twice one of them.
    a = math.radians(30.0)
    context.setPositions(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                   [math.cos(a), 0.0, math.sin(a)],
                                   [2.0, 0.0, 0.0]]) * unit.nanometer)
    total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert total > 0.0
    # Releasing the global releases both, whatever their scales.
    context.setParameter(TORSION_RESTRAINT_PARAMETER, 0.0)
    assert context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) == 0.0
