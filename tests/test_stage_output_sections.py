"""The `.out` states what ran: the census, the method, and the resolved selections.

WHY THIS EXISTS. Amber's `mdout` opens with a topology census and a full echo of every resolved
control variable, which is why an Amber run can very nearly be reconstructed from its own output.
MD-tools' `.out` carried the inputs, the step count and the platform, and nothing else: the cutoff,
the PME treatment, the constraint count, the net charge and the size of the restrained selection
all existed -- in `resolved.config`, in the machine record, in `solute.yaml` -- and a reader had to
open three files to assemble them.

These are the two helpers behind those sections. They are pure functions of a topology and a
serialised System, and they are tested here rather than through a stage run because what is under
test is arithmetic over a System, not anything a device computes.

DERIVED, NEVER CONFIGURED. Every number comes from the System as serialised. A configuration that
asked for a 1.0 nm cutoff and a System carrying 0.8 nm are different runs, and only one of them
integrates -- so a `.out` that echoed the configuration would be describing the wrong one.

PLATFORM_POLICY_EXEMPTION: no Context, no propagation, no energy evaluation. Synthetic Systems
only.
"""
from __future__ import annotations

import openmm
import pytest
from openmm import unit

from md_tools.md.stage import _method_summary, _system_census


def _topology(residues):
    """A topology with the named residues, one atom each unless a count is given."""
    topology = openmm.app.Topology()
    chain = topology.addChain()
    carbon = openmm.app.Element.getBySymbol("C")
    for name, count in residues:
        for _ in range(count):
            residue = topology.addResidue(name, chain)
            topology.addAtom("C", carbon, residue)
    return topology


def _system(n_particles, *, charges=(), constraints=0, remover=False, box=None):
    system = openmm.System()
    for _ in range(n_particles):
        system.addParticle(12.0)
    if charges:
        nonbonded = openmm.NonbondedForce()
        for charge in charges:
            nonbonded.addParticle(charge, 0.3, 0.5)
        system.addForce(nonbonded)
    for index in range(constraints):
        system.addConstraint(index, index + 1, 0.1)
    if remover:
        system.addForce(openmm.CMMotionRemover())
    if box is not None:
        system.setDefaultPeriodicBoxVectors(*[
            openmm.Vec3(*(box[i] if j == i else 0.0 for j in range(3))) * unit.nanometer
            for i in range(3)])
    return system


# -- the census ---------------------------------------------------------------------------------

def test_the_census_counts_atoms_and_residues_by_name():
    topology = _topology([("HOH", 3), ("ALA", 1), ("NA+", 2)])
    census = _system_census(topology, _system(6))
    assert census["atoms"] == 6
    assert census["residues"] == {"HOH": 3, "ALA": 1, "NA+": 2}
    # Most numerous first, so the solvent does not hide the solute at the end of a long line.
    assert census["residues_summary"].startswith("6 (HOH 3,")


def test_the_census_sums_the_net_charge():
    """THE CHEAPEST SETUP CHECK THERE IS.

    Amber prints `Sum of charges from parm topology file` and says out loud when it forces
    neutrality. A system that is not neutral when it was meant to be produces a puzzling energy
    much later and nothing earlier says why.
    """
    census = _system_census(_topology([("X", 3)]), _system(3, charges=(0.5, -0.25, -0.25)))
    assert census["net_charge_e"] == pytest.approx(0.0, abs=1e-12)

    charged = _system_census(_topology([("X", 2)]), _system(2, charges=(1.0, 0.0)))
    assert charged["net_charge_e"] == pytest.approx(1.0)


def test_degrees_of_freedom_subtract_constraints_and_the_motion_remover():
    """The count behind every reported temperature.

    3N, less one per constraint, less three for a centre-of-mass remover. Amber's own step-0
    temperature looks wrong until you know this number, which is exactly why it belongs in the
    output rather than in a reader's head.
    """
    plain = _system_census(_topology([("X", 10)]), _system(10))
    assert plain["degrees_of_freedom"] == 30

    constrained = _system_census(_topology([("X", 10)]), _system(10, constraints=4))
    assert constrained["degrees_of_freedom"] == 26
    assert constrained["constraints"] == 4

    with_remover = _system_census(_topology([("X", 10)]),
                                  _system(10, constraints=4, remover=True))
    assert with_remover["degrees_of_freedom"] == 23


def test_the_box_is_reported_only_when_the_system_is_actually_periodic():
    """AN OPENMM SYSTEM DEFAULTS TO A 2 nm CUBE, so the vectors are never absent.

    Asking `getDefaultPeriodicBoxVectors` therefore cannot answer "is there a box". An implicit
    System would report `2.000 x 2.000 x 2.000 nm` and a volume of 8 nm^3 -- both invented, and
    printed with the same confidence as a measurement. Periodicity is decided by the nonbonded
    method, which is what makes the box load-bearing in the first place.
    """
    explicit = _system(2, charges=(0.0, 0.0), box=(2.0, 3.0, 4.0))
    nonbonded = [f for f in explicit.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    census = _system_census(_topology([("X", 2)]), explicit)
    assert census["box_nm"] == "2.000 x 3.000 x 4.000 nm"
    assert census["volume_nm3"] == pytest.approx(24.0)

    # Implicit solvent: a non-periodic method, so no box and no volume however the vectors read.
    implicit = _system(2, charges=(0.0, 0.0))
    [f for f in implicit.getForces()
     if isinstance(f, openmm.NonbondedForce)][0].setNonbondedMethod(
        openmm.NonbondedForce.NoCutoff)
    census = _system_census(_topology([("X", 2)]), implicit)
    assert census["box_nm"] is None
    assert census["volume_nm3"] is None

    # And a System with no NonbondedForce at all is not periodic either.
    bare = _system_census(_topology([("X", 2)]), _system(2, box=(2.0, 2.0, 2.0)))
    assert bare["box_nm"] is None


# -- the method ---------------------------------------------------------------------------------

def test_the_method_reports_the_nonbonded_treatment_and_cutoff():
    system = _system(2, charges=(0.0, 0.0), box=(3.0, 3.0, 3.0))
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    summary = _method_summary(system)
    assert "PME" in summary["nonbonded"]
    assert "cutoff 0.9 nm" in summary["nonbonded"]
    assert "Ewald tolerance" in summary["nonbonded"]


def test_the_method_reports_the_dispersion_correction_because_it_changes_the_energy():
    """It is also the reason REST2 cannot switch tau by parameter offsets under PME.

    OpenMM computes the tail from the particles' stored epsilons and does not apply parameter
    offsets to it, so a run carrying the correction takes the re-upload path. A reader asking why
    a ladder is slower than expected should be able to see this in the output.
    """
    system = _system(2, charges=(0.0, 0.0), box=(3.0, 3.0, 3.0))
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setUseDispersionCorrection(True)
    assert _method_summary(system)["dispersion correction"] == "on"
    nonbonded.setUseDispersionCorrection(False)
    assert _method_summary(system)["dispersion correction"] == "off"


def test_the_method_counts_the_exceptions_that_hold_the_1_4_terms():
    """The 1-4 pairs live INSIDE NonbondedForce as exceptions, which is why no force group can
    separate `1-4 EEL` from `EELEC`. Stating the count says how much is in there."""
    system = _system(4, charges=(0.0, 0.0, 0.0, 0.0))
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nonbonded.addException(0, 1, 0.0, 0.3, 0.0)
    nonbonded.addException(2, 3, 0.0, 0.3, 0.0)
    assert _method_summary(system)["exceptions (1-4 pairs)"] == 2


def test_the_method_names_the_barostat_in_the_system_or_says_none():
    """`ntb`/`ntp` in Amber's control block. A barostat PRESENT is not the same as one ACTIVE --
    the stage line reports the second; this reports the first."""
    without = _system(2, charges=(0.0, 0.0))
    assert _method_summary(without)["barostat in system"] == "none"

    with_barostat = _system(2, charges=(0.0, 0.0), box=(3.0, 3.0, 3.0))
    with_barostat.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))
    assert "MonteCarloBarostat" in _method_summary(with_barostat)["barostat in system"]


def test_the_method_lists_every_force_so_a_missing_term_is_visible():
    system = _system(2, charges=(0.0, 0.0), remover=True)
    system.addForce(openmm.HarmonicBondForce())
    forces = _method_summary(system)["forces"]
    for name in ("NonbondedForce", "CMMotionRemover", "HarmonicBondForce"):
        assert name in forces


def test_a_system_with_no_nonbonded_force_still_summarises():
    """An implicit or bonded-only System must not make the section raise."""
    summary = _method_summary(_system(2, constraints=1))
    assert summary["constraints"] == "1 bond(s)"
    assert "nonbonded" not in summary


# -- the single-sample guard --------------------------------------------------------------------

def test_a_single_report_yields_a_zero_fluctuation_that_must_not_be_printed_as_one():
    """`sqrt(<x^2> - <x>^2)` over one sample is exactly 0, which reads as "it did not move".

    The statistics helper returns the sample count precisely so the caller can tell the two apart,
    and a stage shorter than `state_interval_steps` produces exactly one row -- the common case,
    not the corner one.
    """
    import inspect

    from md_tools.md import stage as stage_module
    from md_tools.md.stage import _state_table_statistics

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mdout.csv"
        path.write_text('#"Step","Time (ps)","Temperature (K)"\n1,0.1,300.0\n', encoding="utf-8")
        statistics = _state_table_statistics(path)

    assert len(statistics) == 1
    _, mean, fluctuation, count = statistics[0]
    assert mean == pytest.approx(300.0)
    assert fluctuation == 0.0
    assert count == 1, "the count is what distinguishes 'did not move' from 'nothing to compare'"

    # The caller must branch on that count. Asserted against the source because the rendering
    # happens inside `stage_main`, which needs a Context to reach -- the same approach
    # `test_own_replica_exchange` uses for the write-ordering guarantee.
    source = inspect.getsource(stage_module)
    assert "single sample" in source
    assert "if count > 1" in source
