"""Distance and angle restraints, checked BY ENERGY against the form they claim.

A restraint is not testable by sampling. A window whose force constant is wrong by a factor, or
whose centre is interpreted in the wrong units, produces a trajectory that looks entirely healthy
-- values in range, under the right column heading, with a mean somewhere near where it was asked
to be. The only thing that settles it is evaluating the potential at a known geometry and
comparing with the arithmetic the form promises, which is what this file does. It is the same
argument `tests/test_torsion_restraints.py` makes for the wrapped torsion form.

PLATFORM_POLICY_EXEMPTION: single-point energies on the Reference platform. Nothing is
integrated, no sampling is claimed, and this is not evidence for any platform's dynamics.
"""
from __future__ import annotations

import math

import pytest

openmm = pytest.importorskip("openmm")
from openmm import Context, System, VerletIntegrator, unit                       # noqa: E402
from openmm import Platform                                                      # noqa: E402

from md_tools.md.coordinate_restraints import (ANGLE_RESTRAINT_PARAMETER,        # noqa: E402
                                               AngleRestraint,
                                               DISTANCE_RESTRAINT_PARAMETER,
                                               DistanceRestraint)

REFERENCE = Platform.getPlatformByName("Reference")


def _system(n=4):
    system = System()
    for _ in range(n):
        system.addParticle(12.0)
    return system


def _energy(system, positions, parameters=None):
    context = Context(system, VerletIntegrator(0.001 * unit.picosecond), REFERENCE)
    context.setPositions(positions)
    for name, value in (parameters or {}).items():
        context.setParameter(name, value)
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_a_distance_restraint_is_off_until_it_is_switched_on():
    """A System carrying the force must be unbiased until something asks otherwise.

    Otherwise the bias would act during the equilibration that precedes the window.
    """
    system = _system(2)
    restraint = DistanceRestraint(system)
    restraint.add_distance([0, 1], centre_nm=1.0, scale=100.0)
    # Two atoms 2 nm apart: 1 nm from a centre at 1.0, so a live restraint would be plainly
    # nonzero and an inert one is exactly zero.
    positions = [[0, 0, 0], [2.0, 0, 0]] * unit.nanometer
    assert _energy(system, positions) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("separation, centre, k", [
    (1.5, 1.0, 100.0),
    (0.8, 1.0, 250.0),
    (1.0, 1.0, 500.0),       # exactly at the centre: zero
    (3.0, 0.5, 40.0),
])
def test_a_harmonic_distance_restraint_matches_half_k_dr_squared(separation, centre, k):
    system = _system(2)
    restraint = DistanceRestraint(system)
    restraint.add_distance([0, 1], centre_nm=centre, scale=k)
    positions = [[0, 0, 0], [separation, 0, 0]] * unit.nanometer
    expected = 0.5 * k * (separation - centre) ** 2
    energy = _energy(system, positions, {DISTANCE_RESTRAINT_PARAMETER: 1.0})
    assert energy == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("separation, inside", [
    (1.00, True), (1.15, True), (1.19, True),     # within 0.2 nm of 1.0
    (1.25, False), (0.70, False),
])
def test_a_flat_bottom_distance_restraint_is_exactly_zero_inside_its_window(separation, inside):
    """Exactly zero, not merely small: samples inside need no reweighting at all."""
    system = _system(2)
    restraint = DistanceRestraint(system, form="flat_bottom")
    restraint.add_distance([0, 1], centre_nm=1.0, half_width_nm=0.2, scale=300.0)
    positions = [[0, 0, 0], [separation, 0, 0]] * unit.nanometer
    energy = _energy(system, positions, {DISTANCE_RESTRAINT_PARAMETER: 1.0})
    if inside:
        assert energy == pytest.approx(0.0, abs=1e-12)
    else:
        edge = abs(separation - 1.0) - 0.2
        assert energy == pytest.approx(0.5 * 300.0 * edge ** 2, rel=1e-9)


@pytest.mark.parametrize("degrees, centre, k", [
    (90.0, 90.0, 100.0),
    (60.0, 90.0, 100.0),
    (120.0, 90.0, 250.0),
])
def test_a_harmonic_angle_restraint_matches_half_k_dtheta_squared_in_radians(degrees, centre, k):
    """The centre is written in degrees and the arithmetic happens in radians.

    A form that squared a difference of DEGREES would be wrong by (180/pi)^2 -- a factor of 3283
    -- which is the kind of error that still samples, just in a window far tighter than anyone
    asked for.
    """
    system = _system(3)
    restraint = AngleRestraint(system)
    restraint.add_angle([0, 1, 2], centre_degrees=centre, scale=k)
    # Vertex at atom 1, arms of unit length, opening angle `degrees`.
    radians = math.radians(degrees)
    positions = [[1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [math.cos(radians), math.sin(radians), 0.0]] * unit.nanometer
    expected = 0.5 * k * (radians - math.radians(centre)) ** 2
    energy = _energy(system, positions, {ANGLE_RESTRAINT_PARAMETER: 1.0})
    assert energy == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_an_angle_centre_outside_zero_to_180_is_refused():
    """An angle between three atoms is unsigned and bounded; it does not wrap onto a geometry."""
    system = _system(3)
    restraint = AngleRestraint(system)
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        restraint.add_angle([0, 1, 2], centre_degrees=200.0, scale=10.0)


def test_a_negative_distance_centre_is_refused():
    """Squaring it would centre the window on its absolute value without saying so."""
    system = _system(2)
    restraint = DistanceRestraint(system)
    with pytest.raises(ValueError, match="must not be negative"):
        restraint.add_distance([0, 1], centre_nm=-1.0, scale=10.0)


@pytest.mark.parametrize("builder, adder, atoms", [
    (DistanceRestraint, "add_distance", [0, 0]),
    (AngleRestraint, "add_angle", [0, 1, 1]),
])
def test_repeated_atoms_are_refused(builder, adder, atoms):
    """A term over a repeated atom has a zero or undefined coordinate, not a weak one."""
    system = _system(3)
    restraint = builder(system)
    with pytest.raises(ValueError, match="distinct"):
        getattr(restraint, adder)(atoms, 1.0)


def test_the_two_kinds_carry_distinct_global_parameters():
    """One switch each, so releasing a distance bias cannot release an orientational one.

    A Boresch restraint scans the distance while HOLDING the five orientational terms, so the two
    groups must be independently switchable. Sharing one parameter would make that inexpressible.
    """
    assert DISTANCE_RESTRAINT_PARAMETER != ANGLE_RESTRAINT_PARAMETER
    from md_tools.md.torsion_restraints import TORSION_RESTRAINT_PARAMETER

    assert len({DISTANCE_RESTRAINT_PARAMETER, ANGLE_RESTRAINT_PARAMETER,
                TORSION_RESTRAINT_PARAMETER}) == 3


def test_the_forms_the_configuration_accepts_are_the_forms_the_runtime_builds():
    """Pins the two lists together, as the torsion module's test does."""
    from md_tools.md.coordinate_restraints import RESTRAINT_FORMS as RUNTIME
    from md_tools.umbrella.definition import RESTRAINT_FORMS as CONFIGURED

    assert tuple(CONFIGURED) == tuple(RUNTIME)
