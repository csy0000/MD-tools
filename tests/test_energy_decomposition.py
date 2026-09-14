"""The post-hoc decomposition adds up, term by term, the way Amber's `mdout` prints it.

THE CHECK THAT MATTERS IS THE SUM. Any individual term looks plausible on its own -- a missed force,
a double-counted exception, a pair that landed in two columns at once -- and only the comparison
against the undecomposed total exposes it. So that is what is asserted here, on a system carrying
every shape the splitter has to handle at once: bonds, angles, torsions, a positional restraint,
PME electrostatics, Lennard-Jones, and 1-4 exceptions.

WHY THE SPLIT IS EXACT AND NOT A FIT. A nonbonded energy is additive in its two physical halves
because they depend on disjoint parameters: `U_nb(q, eps) = U_elec(q) + U_LJ(eps)`. Zeroing the
epsilons leaves electrostatics with its reciprocal sum and self-energy attached, both functions of
charge alone; zeroing the charges leaves Lennard-Jones with the dispersion correction attached, a
function of epsilon alone. The 1-4 terms come apart the same way, because an exception REPLACES its
pair's standard interaction -- so a copy carrying only exceptions is the 1-4 contribution and
nothing else, and a copy with the exceptions zeroed has those pairs absent rather than
double-counted.

AND THE PRODUCTION SYSTEM IS NEVER TOUCHED, which is the reason this is post-hoc at all: force
groups are part of a serialised System, so regrouping or restructuring the one a run integrates
would change `system_sha256` and invalidate every checkpoint fingerprint in flight.

PLATFORM_POLICY_EXEMPTION: single-point energies on a small synthetic System, on the CPU platform.
Nothing is propagated -- what is under test is whether a decomposition sums to its own total, which
is a property of the arithmetic and not of the device.
"""
from __future__ import annotations

import openmm
import pytest
from openmm import XmlSerializer, unit

from md_tools.openmm.decomposition import (AMBER_TERM_ORDER, decompose,
                                           decomposition_systems)

BOX_NM = 4.0
CUTOFF_NM = 1.0


@pytest.fixture
def system_and_positions():
    """Eight charged, dispersive particles with bonded terms, a restraint and 1-4 exceptions."""
    system = openmm.System()
    for _ in range(8):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(BOX_NM, 0, 0) * unit.nanometer,
        openmm.Vec3(0, BOX_NM, 0) * unit.nanometer,
        openmm.Vec3(0, 0, BOX_NM) * unit.nanometer)

    bonds = openmm.HarmonicBondForce()
    for i in range(7):
        bonds.addBond(i, i + 1, 0.15, 1000.0)
    system.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    for i in range(6):
        angles.addAngle(i, i + 1, i + 2, 2.0, 100.0)
    system.addForce(angles)

    torsions = openmm.PeriodicTorsionForce()
    for i in range(5):
        torsions.addTorsion(i, i + 1, i + 2, i + 3, 2, 0.0, 8.0)
    system.addForce(torsions)

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(CUTOFF_NM * unit.nanometer)
    nonbonded.setUseDispersionCorrection(True)
    for index in range(8):
        # Alternating signs so the system is neutral and the electrostatics is not trivial.
        nonbonded.addParticle(0.4 if index % 2 == 0 else -0.4, 0.32, 0.55)
    # 1-2 and 1-3 excluded outright; 1-4 pairs SCALED, which is what Amber prints separately.
    for i in range(7):
        nonbonded.addException(i, i + 1, 0.0, 0.32, 0.0)
    for i in range(6):
        nonbonded.addException(i, i + 2, 0.0, 0.32, 0.0)
    for i in range(5):
        nonbonded.addException(i, i + 3, -0.4 * 0.4 / 1.2, 0.32, 0.55 / 2.0)
    system.addForce(nonbonded)

    restraint = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 100.0)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    for index in range(3):
        restraint.addParticle(index, [0.1 * index, 0.2, 0.3])
    system.addForce(restraint)

    # Machinery that carries no potential energy, present so the splitter must ignore it.
    system.addForce(openmm.CMMotionRemover())
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))

    positions = [openmm.Vec3(0.1 + 0.16 * i, 0.2 + 0.02 * i, 0.3) for i in range(8)]
    return system, positions


def test_the_terms_sum_to_the_undecomposed_total(system_and_positions):
    """THE ONE THAT CATCHES EVERYTHING ELSE."""
    system, positions = system_and_positions
    result = decompose(system, positions,
                       box_vectors=system.getDefaultPeriodicBoxVectors())
    assert result["exact"], (
        f"residual {result['residual_kj_mol']:.6e} kJ/mol\n"
        f"terms: {result['terms']}\nsum {result['sum_kj_mol']} vs total {result['total_kj_mol']}")
    assert result["sum_kj_mol"] == pytest.approx(result["total_kj_mol"], abs=1e-3)


def test_every_amber_term_this_system_has_is_reported(system_and_positions):
    system, positions = system_and_positions
    result = decompose(system, positions,
                       box_vectors=system.getDefaultPeriodicBoxVectors())
    assert result["order"] == [
        "BOND", "ANGLE", "DIHED", "1-4 NB", "1-4 EEL", "VDWAALS", "EELEC", "RESTRAINT"]
    # Amber's own column order, so two engines read down one column.
    assert list(result["terms"]) == [t for t in AMBER_TERM_ORDER if t in result["terms"]]


def test_the_terms_are_individually_non_trivial(system_and_positions):
    """A decomposition of zeros would also sum correctly.

    Each term must actually carry energy, or the sum test above could pass while the splitter
    silently zeroed something.
    """
    system, positions = system_and_positions
    result = decompose(system, positions,
                       box_vectors=system.getDefaultPeriodicBoxVectors())
    for term in ("BOND", "ANGLE", "DIHED", "VDWAALS", "EELEC", "RESTRAINT", "1-4 EEL", "1-4 NB"):
        assert result["terms"][term] != 0.0, f"{term} came out exactly zero"


def test_the_production_system_is_not_modified(system_and_positions):
    """The whole reason this is post-hoc. A changed System is a changed `system_sha256`."""
    system, positions = system_and_positions
    before = XmlSerializer.serialize(system)
    groups_before = [f.getForceGroup() for f in system.getForces()]

    decompose(system, positions, box_vectors=system.getDefaultPeriodicBoxVectors())

    assert XmlSerializer.serialize(system) == before, "the integrated System was mutated"
    assert [f.getForceGroup() for f in system.getForces()] == groups_before


def test_prepared_systems_can_be_reused_across_frames(system_and_positions):
    """Building the copies is the expensive part; a trajectory sweep must pay it once."""
    system, positions = system_and_positions
    prepared = decomposition_systems(system)
    box = system.getDefaultPeriodicBoxVectors()

    first = decompose(system, positions, box_vectors=box, systems=prepared)
    moved = [openmm.Vec3(p.x + 0.01, p.y, p.z) for p in positions]
    second = decompose(system, moved, box_vectors=box, systems=prepared)

    assert first["exact"] and second["exact"]
    assert first["terms"]["EELEC"] != second["terms"]["EELEC"], (
        "a reused system must still see the new coordinates")


def test_a_system_without_a_restraint_reports_no_restraint_term():
    """An absent term is omitted, not reported as zero: those are different facts."""
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    system.addForce(bonds)

    result = decompose(system, [openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(0.2, 0.0, 0.0)])
    assert "RESTRAINT" not in result["terms"]
    assert result["exact"], result["residual_kj_mol"]
