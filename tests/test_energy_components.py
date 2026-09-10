"""`energy_components.csv`: the attribution Amber's `mdout` gives and one total cannot.

`mdout.csv` reports a single potential energy, so a drift in it cannot be attributed -- an Amber
user reading `EEL` or `EGB` out of an `mdout` to work out what a build did wrong has nothing to
read. This is that decomposition, per force group, at the state table's own cadence.

PLATFORM_POLICY_EXEMPTION: single-point energies on a three-particle synthetic System, on the
Reference platform so the arithmetic is exact and machine-independent. Nothing is propagated --
there is no integration step in this file -- and what is under test is whether a decomposition
adds up, which is a property of the sum and not of the device that computed the terms.

The groups are read off the System AS BUILT and nothing here reassigns them: a force group is
part of the serialised System, so changing one would change `system_sha256` and make every run in
flight unresumable for the sake of a diagnostic.
"""
from __future__ import annotations

import csv

import pytest

from md_tools.md._stages import energy_components_name


def test_the_file_is_named_for_its_stage_and_segment():
    assert energy_components_name("cMD") == "energy_components.csv"
    assert energy_components_name("cMD", 3) == "energy_components_prod3.csv"
    assert energy_components_name("eq_nvt_free") == "energy_components_eq_nvt_free.csv"


@pytest.fixture
def reported(tmp_path):
    """One report from a real System, with the components and the total taken together."""
    import openmm
    from openmm import unit
    from openmm.app import Simulation

    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    bonds.setForceGroup(0)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 2.0, 100.0)
    angles.setForceGroup(1)
    nonbonded = openmm.NonbondedForce()
    for _ in range(3):
        nonbonded.addParticle(0.1, 0.3, 0.5)
    nonbonded.setForceGroup(11)
    for force in (bonds, angles, nonbonded):
        system.addForce(force)

    integrator = openmm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond,
                                                 0.002 * unit.picoseconds)
    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("X", chain)
    for _ in range(3):
        topology.addAtom("C", openmm.app.Element.getBySymbol("C"), residue)
    simulation = Simulation(topology, system, integrator,
                            openmm.Platform.getPlatformByName("Reference"))
    simulation.context.setPositions([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.3, 0.0]])

    from md_tools.md.stage import _EnergyComponentsReporter

    path = tmp_path / "energy_components.csv"
    reporter = _EnergyComponentsReporter(path, 1, system)
    reporter.report(simulation, simulation.context.getState(getEnergy=True))
    reporter._handle.flush()

    total = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows, total


def test_the_components_sum_to_the_total_potential_energy(reported):
    """THE CHECK THAT MATTERS. A decomposition that does not add up is worse than none.

    Anything else -- a missed force, a group counted twice, a group silently dropped -- shows up
    here and nowhere else, because every individual number would still look plausible.
    """
    rows, total = reported
    values = [float(value) for value in rows[1][2:]]
    assert sum(values) == pytest.approx(total, rel=1e-9, abs=1e-9)


def test_each_group_is_labelled_by_the_forces_in_it(reported):
    """Sharing a group is stated, not hidden behind a prettier name."""
    rows, _ = reported
    header = rows[0]
    assert header[0] == '#"Step"'
    labels = [column for column in header[2:]]
    assert any("HarmonicBondForce" in label for label in labels)
    assert any("HarmonicAngleForce" in label for label in labels)
    assert any("NonbondedForce" in label for label in labels)


def test_a_massless_bookkeeping_force_contributes_no_column(tmp_path):
    """`CMMotionRemover` has no energy; a column of zeros for it is noise, not information."""
    import openmm

    system = openmm.System()
    system.addParticle(12.0)
    remover = openmm.CMMotionRemover()
    remover.setForceGroup(0)
    system.addForce(remover)

    from md_tools.md.stage import _EnergyComponentsReporter

    path = tmp_path / "energy_components.csv"
    reporter = _EnergyComponentsReporter(path, 1, system)
    reporter._handle.flush()
    assert path.read_text(encoding="utf-8").strip() == '#"Step","Time (ps)"'


# -- averages and RMS fluctuations, as Amber prints at the foot of an mdout ---------------------

def test_the_statistics_are_the_mean_and_rms_fluctuation_of_each_column(tmp_path):
    """RMS FLUCTUATION, not standard error: sqrt(<x^2> - <x>^2), the quantity Amber reports."""
    import math

    from md_tools.md.stage import _state_table_statistics

    path = tmp_path / "mdout.csv"
    path.write_text('#"Step","Time (ps)","Potential Energy (kJ/mole)","Temperature (K)"\n'
                    "1,0.1,-10.0,290.0\n"
                    "2,0.2,-14.0,310.0\n", encoding="utf-8")
    stats = {name: (mean, rms) for name, mean, rms, _ in _state_table_statistics(path)}

    assert stats["Potential Energy (kJ/mole)"][0] == pytest.approx(-12.0)
    assert stats["Potential Energy (kJ/mole)"][1] == pytest.approx(2.0)
    assert stats["Temperature (K)"][0] == pytest.approx(300.0)
    assert stats["Temperature (K)"][1] == pytest.approx(10.0)
    assert "Step" not in stats and "Time (ps)" not in stats, (
        "Step and Time are the independent variable; averaging them says nothing")


def test_statistics_over_a_table_with_no_rows_are_empty_rather_than_a_division_by_zero(tmp_path):
    from md_tools.md.stage import _state_table_statistics

    path = tmp_path / "mdout.csv"
    path.write_text('#"Step","Time (ps)","Temperature (K)"\n', encoding="utf-8")
    assert _state_table_statistics(path) == []
    assert _state_table_statistics(tmp_path / "absent.csv") == []
