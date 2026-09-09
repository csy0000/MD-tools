"""The torsional restraint on CUDA: it biases the dynamics, and releasing it costs nothing.

`test_torsion_restraints.py` checks the energy function on the Reference platform, which is where
the functional form belongs. This file checks the thing that only a real device can show: that
`set_strength` reaches a live CUDA Context and changes what the integrator does.

Kept deliberately short. What is under test is that the parameter arrives and the bias acts, not
the quality of any sampling -- 4 ps holds a torsion in place and would tell us nothing about a
free-energy profile, which is analysis and lives outside this repository anyway.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, PDBFile, Simulation

from md_tools.md.torsion_restraints import (FLAT_BOTTOM, HARMONIC, TORSION_RESTRAINT_PARAMETER,
                                            TorsionRestraint)

from .conftest import ALA_PDB

pytestmark = pytest.mark.skipif(
    "CUDA" not in [Platform.getPlatform(i).getName()
                   for i in range(Platform.getNumPlatforms())],
    reason="no CUDA platform")

K = 500.0                    # kJ/mol/rad^2


def _phi_indices(topology):
    atoms = list(topology.atoms())

    def index(residue, name):
        return next(a.index for a in atoms
                    if a.residue.name == residue and a.name == name)
    return (index("ACE", "C"), index("ALA", "N"), index("ALA", "CA"), index("ALA", "C"))


def _dihedral(positions, quad):
    """IUPAC sign convention, verified against mdtraj in the prototype for this module.

    Checking a dihedral only at the starting structure would prove nothing here: alanine
    dipeptide's phi starts at 180 degrees, which is its own negative.
    """
    p = np.asarray(positions.value_in_unit(unit.nanometer))
    b0, b1, b2 = p[quad[1]] - p[quad[0]], p[quad[2]] - p[quad[1]], p[quad[3]] - p[quad[2]]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1))
    return -math.degrees(math.atan2(np.dot(m, n2), np.dot(n1, n2)))


def _simulation(form, centre, width=0.0, seed=11):
    pdb = PDBFile(str(ALA_PDB))
    field = ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")
    system = field.createSystem(pdb.topology, constraints=HBonds)
    phi = _phi_indices(pdb.topology)
    restraint = TorsionRestraint(system, form)
    restraint.add_torsion(phi, centre, width)
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                          0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    simulation = Simulation(pdb.topology, system, integrator,
                            Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    simulation.context.setPositions(pdb.positions)
    return simulation, restraint, phi, pdb


def _sample(simulation, phi, blocks=20, steps=100):
    seen = []
    for _ in range(blocks):
        simulation.step(steps)
        seen.append(_dihedral(simulation.context.getState(getPositions=True).getPositions(), phi))
    return np.array(seen)


def _circular_mean(degrees):
    radians = np.radians(degrees)
    return np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))


def test_set_strength_reaches_a_live_cuda_context_and_holds_the_torsion():
    """The restraint must actually bias dynamics on the device, not merely be present."""
    simulation, restraint, phi, pdb = _simulation(HARMONIC, -60.0)
    restraint.set_strength(simulation, K)
    assert simulation.context.getParameter(TORSION_RESTRAINT_PARAMETER) == K

    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 11)
    simulation.step(2000)
    held = _sample(simulation, phi)

    offset = (_circular_mean(held) + 60.0 + 180.0) % 360.0 - 180.0
    assert abs(offset) < 15.0, (
        f"phi restrained to -60 deg settled at {_circular_mean(held):.1f} deg on CUDA")
    # And it is genuinely held: a free alanine dipeptide phi wanders far more than this.
    assert held.std() < 15.0, f"restrained phi has sd {held.std():.1f} deg; it is not held"


def test_releasing_the_restraint_removes_the_bias_without_removing_the_force():
    """0.0 releases it. The Force stays, so the System's layout -- and the checkpoint layout --
    is unchanged along a chain of stages."""
    simulation, restraint, phi, pdb = _simulation(HARMONIC, -60.0)
    before = simulation.context.getSystem().getNumForces()

    restraint.set_strength(simulation, K)
    simulation.minimizeEnergy(maxIterations=200)
    biased = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    restraint.set_strength(simulation, 0.0)
    released = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    assert simulation.context.getSystem().getNumForces() == before
    assert simulation.context.getParameter(TORSION_RESTRAINT_PARAMETER) == 0.0
    # At the biased minimum the restraint is near its own minimum too, so the two energies are
    # close; what must hold is that releasing never RAISES the energy.
    assert released <= biased


def test_a_flat_bottom_restraint_explores_more_than_a_harmonic_one_on_cuda():
    """The forms differ in what they permit, and the difference must survive onto the device."""
    tight, restraint, phi, _ = _simulation(HARMONIC, -60.0)
    restraint.set_strength(tight, K)
    tight.minimizeEnergy(maxIterations=200)
    tight.context.setVelocitiesToTemperature(300 * unit.kelvin, 11)
    tight.step(2000)
    narrow = _sample(tight, phi)

    loose, loose_restraint, phi2, _ = _simulation(FLAT_BOTTOM, -60.0, width=30.0)
    loose_restraint.set_strength(loose, K)
    loose.minimizeEnergy(maxIterations=200)
    loose.context.setVelocitiesToTemperature(300 * unit.kelvin, 11)
    loose.step(2000)
    wide = _sample(loose, phi2)

    assert wide.std() > narrow.std(), (
        f"flat-bottom sd {wide.std():.1f} is not wider than harmonic sd {narrow.std():.1f}")
