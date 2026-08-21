"""Positional restraints must measure the distance their System actually has.

Finding 1. `_add_positional_restraints` used `periodicdistance` for every solvent mode. For an
explicit periodic box that is right: an atom that wanders across a face is measured to its own
reference point rather than to an image of it. For an implicit nonperiodic System it is wrong twice
over -- the restraint depends on OpenMM's *default* box vectors, and the restraint Force itself
reports periodic boundary use, so a protocol that claims to have no box acquires one through its own
restraint.

These tests interrogate the **Force and the System the Context will use**, never a boolean captured
before the Force was added. A test that asserted `results["periodic"] is False` would pass with the
defect fully present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _two_particle_system(*, periodic: bool):
    """The smallest System that can carry a restraint, with and without a box.

    A NonbondedForce decides periodicity: with `CutoffPeriodic` the System reports periodic
    boundary use, with `NoCutoff` it does not. Built here rather than loaded so the test states
    exactly what it is testing.
    """
    import openmm
    from openmm import app, unit

    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0 * unit.dalton)
    nonbonded = openmm.NonbondedForce()
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    nonbonded.setNonbondedMethod(
        openmm.NonbondedForce.CutoffPeriodic if periodic else openmm.NonbondedForce.NoCutoff)
    if periodic:
        nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    system.addForce(nonbonded)
    if periodic:
        system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 2.0).tolist())

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    for name in ("C1", "C2"):
        topology.addAtom(name, app.element.carbon, residue)
    if periodic:
        topology.setPeriodicBoxVectors((np.eye(3) * 2.0).tolist())
    return system, topology


def _restrained(periodic: bool, positions_nm):
    from md_templates.openmm.equilibration import _add_positional_restraints

    system, topology = _two_particle_system(periodic=periodic)
    index, atoms, convention = _add_positional_restraints(
        system, topology, "solute", np.asarray(positions_nm))
    return system, topology, index, atoms, convention


def _energy(system, positions_nm, *, k=1000.0, box=None):
    import openmm
    from openmm import unit

    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(np.asarray(positions_nm) * unit.nanometer)
    context.setParameter("k_restraint", k)
    value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del context, integrator
    return value


# ---------------------------------------------------------------------------------------------
# the implicit case
# ---------------------------------------------------------------------------------------------

def test_an_implicit_restraint_force_does_not_use_periodic_boundaries():
    """Asked of the FORCE, which is what the defect actually made periodic."""
    from openmm import CustomExternalForce

    system, _, index, _, convention = _restrained(False, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    force = system.getForce(index)
    assert isinstance(force, CustomExternalForce)
    assert force.usesPeriodicBoundaryConditions() is False, (
        "the implicit restraint Force reports periodic boundary use; it is measuring minimum-image "
        "distance in a System that has no box")
    assert "periodicdistance" not in force.getEnergyFunction()
    assert convention == "cartesian (nonperiodic)"


def test_the_implicit_system_is_still_nonperiodic_after_the_restraint_is_added():
    """The System is what the Context is built from, so it is the System that must stay honest."""
    system, _, _, _, _ = _restrained(False, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert system.usesPeriodicBoundaryConditions() is False


def test_arbitrary_box_vectors_do_not_change_the_implicit_restraint_energy():
    """The sharpest test: if the restraint reads box vectors, changing them changes the energy.

    A nonperiodic Context still accepts box vectors -- OpenMM keeps a default set -- so this
    distinguishes "does not have a box" from "does not USE one".
    """
    reference = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    displaced = [[0.9, 0.0, 0.0], [0.5, 0.0, 0.0]]
    system, _, _, _, _ = _restrained(False, reference)

    small = (np.eye(3) * 1.0).tolist()
    large = (np.eye(3) * 5.0).tolist()
    energy_small = _energy(system, displaced, box=small)
    energy_large = _energy(system, displaced, box=large)

    assert energy_small == pytest.approx(energy_large, rel=1e-12), (
        f"the implicit restraint energy moved from {energy_small} to {energy_large} when only the "
        "box vectors changed, so it is reading a box it should not have")
    # and it is the plain Cartesian value: k * 0.9^2 for the displaced particle
    assert energy_small == pytest.approx(1000.0 * 0.9 ** 2, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# the explicit case must be unchanged
# ---------------------------------------------------------------------------------------------

def test_an_explicit_restraint_still_uses_minimum_image_distance():
    from openmm import CustomExternalForce

    system, _, index, _, convention = _restrained(True, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    force = system.getForce(index)
    assert isinstance(force, CustomExternalForce)
    assert "periodicdistance" in force.getEnergyFunction()
    assert convention == "minimum-image (periodicdistance)"


def test_an_explicit_restraint_measures_across_a_box_face():
    """An atom just across the face is 0.1 nm from its reference, not 1.9 nm.

    This is the behaviour minimum-image distance exists for, and it is why the explicit branch must
    keep it: without it, an atom that wraps is dragged the width of the box.
    """
    box = (np.eye(3) * 2.0).tolist()
    reference = [[0.05, 0.0, 0.0], [0.5, 0.0, 0.0]]
    across = [[1.95, 0.0, 0.0], [0.5, 0.0, 0.0]]          # 0.1 nm away through the face
    system, _, _, _, _ = _restrained(True, reference)

    energy = _energy(system, across, box=box)
    assert energy == pytest.approx(1000.0 * 0.1 ** 2, rel=1e-6), (
        f"expected the minimum-image distance 0.1 nm, got an energy of {energy} kJ/mol, which is "
        f"{np.sqrt(energy / 1000.0):.3f} nm")


def test_the_two_conventions_disagree_where_it_matters():
    """If they ever agree on this geometry, the branch is not doing anything."""
    box = (np.eye(3) * 2.0).tolist()
    reference = [[0.05, 0.0, 0.0], [0.5, 0.0, 0.0]]
    across = [[1.95, 0.0, 0.0], [0.5, 0.0, 0.0]]

    periodic_system, _, _, _, _ = _restrained(True, reference)
    cartesian_system, _, _, _, _ = _restrained(False, reference)

    periodic_energy = _energy(periodic_system, across, box=box)
    cartesian_energy = _energy(cartesian_system, across, box=box)
    assert periodic_energy == pytest.approx(1000.0 * 0.1 ** 2, rel=1e-6)
    assert cartesian_energy == pytest.approx(1000.0 * 1.9 ** 2, rel=1e-6)
    assert cartesian_energy > periodic_energy * 100
