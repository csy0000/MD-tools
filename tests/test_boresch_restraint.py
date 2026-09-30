"""The Boresch restraint: the six coordinates, and every anchor choice that cannot define them.

WHY THE ANCHOR CHECKS MATTER MORE THAN THE ARITHMETIC

A Boresch restraint holds all six rigid-body degrees of freedom of a ligand relative to a
receptor, so that a dissociation profile describes ONE pathway. Restrain the distance alone and
the ligand orbits the site at fixed separation; two windows at the same separation then sample
different exit routes and the profile is an average over routes nobody chose. Nothing in the
output says so.

The failure modes checked here are the silent ones. A near-collinear anchor triple makes its
dihedral undefined -- it swings widely for an unmeasurable change in geometry -- and a restrained
angle near 0 or 180 degrees cannot be held, because the angle's gradient with respect to its
atoms vanishes there. Both produce windows that run to completion and sample something.

PLATFORM_POLICY_EXEMPTION: pure geometry on arrays, and single-point energies on the Reference
platform. Nothing is integrated and no platform's dynamics are claimed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from md_tools.umbrella import (BORESCH_TERMS, BoreschAnchors, BoreschError,
                               boresch_cv_document, boresch_window_document, check_anchors,
                               measure_anchors, standard_state_note)

#: A well-conditioned arrangement: the receptor kinked in x-y, the ligand offset in y and z, so
#: no triple is near collinear and both restrained angles sit well inside [20, 160].
GOOD = np.array([
    [0.0, 0.0, 0.0],      # R1
    [-0.4, 0.15, 0.0],    # R2
    [-0.8, 0.0, 0.25],    # R3
    [0.5, 0.35, 0.0],     # L1
    [0.9, 0.55, 0.2],     # L2
    [1.3, 0.3, 0.45],     # L3
])
ANCHORS = BoreschAnchors(receptor=(0, 1, 2), ligand=(3, 4, 5))


def test_the_six_terms_span_all_six_rigid_body_degrees_of_freedom():
    """One distance, two angles, three dihedrals -- three translational and three rotational."""
    kinds = [kind for _name, kind, _slots in BORESCH_TERMS]
    assert kinds.count("distance") == 1
    assert kinds.count("angle") == 2
    assert kinds.count("torsion") == 3
    assert len(BORESCH_TERMS) == 6


def test_each_term_spans_the_anchors_the_definition_requires():
    """The atom sets are the Boresch definition's, not a plausible rearrangement of it."""
    assert ANCHORS.atoms_for("r") == (0, 3)              # R1 - L1
    assert ANCHORS.atoms_for("thA") == (1, 0, 3)         # R2 - R1 - L1
    assert ANCHORS.atoms_for("thB") == (0, 3, 4)         # R1 - L1 - L2
    assert ANCHORS.atoms_for("phA") == (2, 1, 0, 3)      # R3 - R2 - R1 - L1
    assert ANCHORS.atoms_for("phB") == (1, 0, 3, 4)      # R2 - R1 - L1 - L2
    assert ANCHORS.atoms_for("phC") == (0, 3, 4, 5)      # R1 - L1 - L2 - L3
    # Every term must involve BOTH molecules, or it restrains one of them to itself and holds
    # nothing about the pose.
    for name, _kind, _slots in BORESCH_TERMS:
        atoms = set(ANCHORS.atoms_for(name))
        assert atoms & set(ANCHORS.receptor), name
        assert atoms & set(ANCHORS.ligand), name


def test_a_repeated_anchor_is_refused_before_any_geometry_is_measured():
    """It collapses a coordinate onto a constant, which restrains nothing while still appearing."""
    with pytest.raises(BoreschError, match="distinct"):
        BoreschAnchors(receptor=(0, 1, 2), ligand=(2, 4, 5))


def test_well_conditioned_anchors_are_accepted_and_measured():
    """The discriminating half: these guards must PASS on a reasonable choice."""
    measured = check_anchors(GOOD, ANCHORS)
    assert set(measured) == {name for name, _k, _s in BORESCH_TERMS}
    assert measured["r"] == pytest.approx(np.linalg.norm(GOOD[3] - GOOD[0]), rel=1e-12)
    assert 20.0 < measured["thA"] < 160.0
    assert 20.0 < measured["thB"] < 160.0


@pytest.mark.parametrize("label, mutate, fragment", [
    ("all six on a line",
     lambda p: np.array([[0, 0, 0], [-0.4, 0, 0], [-0.8, 0, 0],
                         [0.5, 0, 0], [0.9, 0, 0], [1.3, 0, 0]], float),
     "collinear"),
    ("thA straightened",
     lambda p: np.vstack([p[0], [-0.4, 0.0, 0.0], p[2], [0.5, 0.0, 0.0], p[4], p[5]]),
     "thA"),
    ("R1 and L1 coincident",
     lambda p: np.vstack([p[0], p[1], p[2], p[0], p[4], p[5]]),
     "coincide"),
])
def test_ill_conditioned_anchors_are_refused_with_an_actionable_message(label, mutate, fragment):
    """And refused by THIS function, not by the geometry it calls.

    `torsion_degrees` refuses a collinear dihedral correctly, with a message about atom indices
    and an undefined dihedral -- the right refusal from a geometry function and the wrong one to
    hand somebody choosing anchors, because it names no anchor, no margin and no remedy. The
    conditioning is therefore checked BEFORE the dihedrals are measured; measured first, this
    whole check was unreachable for exactly the input it exists to diagnose.
    """
    with pytest.raises(BoreschError) as refusal:
        check_anchors(mutate(GOOD), ANCHORS)
    message = str(refusal.value)
    assert fragment in message, message
    assert "do not define a Boresch restraint" in message
    # It says why this is refused now rather than later, because every window shares the choice.
    assert "every window in a profile shares this choice" in message.lower()


def test_the_generated_cv_document_resolves_through_the_real_parser():
    """The six coordinates must be a cv.yaml this build actually accepts."""
    import yaml

    from md_tools.cv import parse_cv_definition

    document = boresch_cv_document(ANCHORS)
    definition = parse_cv_definition(yaml.safe_dump(document), source="boresch",
                                     particles=len(GOOD))
    assert definition.names == ("boresch_r", "boresch_thA", "boresch_thB",
                                "boresch_phA", "boresch_phB", "boresch_phC")
    assert definition.kinds == ("distance", "angle", "angle", "torsion", "torsion", "torsion")
    # A mixed definition must not claim one unit for the whole file.
    body = definition.resolved()
    assert "units" not in body
    columns = {c["name"]: c for c in body["collective_variables"]}
    assert columns["boresch_r"]["units"] == "nanometers"
    assert "wrapping" not in columns["boresch_r"]
    assert columns["boresch_phA"]["wrapping"] == "[-180, 180)"


def test_a_window_scans_the_distance_and_holds_the_other_five_as_measured():
    """The five held terms ARE the pathway; they must carry the bound pose's own values."""
    measured = check_anchors(GOOD, ANCHORS)
    window = boresch_window_document(measured, centre_nm=1.2)
    by_cv = {r["cv"]: r for r in window["restraints"]}
    assert len(by_cv) == 6

    # The scanned coordinate sits where the window asked, in nanometres.
    assert by_cv["boresch_r"]["centre_nm"] == pytest.approx(1.2)
    assert "centre_deg" not in by_cv["boresch_r"]

    # The held terms carry the MEASURED reference values, in degrees.
    for term in ("thA", "thB", "phA", "phB", "phC"):
        entry = by_cv[f"boresch_{term}"]
        assert entry["centre_deg"] == pytest.approx(measured[term], abs=1e-6)
        assert "centre_nm" not in entry


def test_the_window_resolves_against_its_own_cv_document():
    """End to end through the real loaders: the restraint must find every name it restrains."""
    import yaml

    from md_tools.cv import parse_cv_definition
    from md_tools.umbrella.definition import load_umbrella_definition

    measured = check_anchors(GOOD, ANCHORS)
    definition = parse_cv_definition(yaml.safe_dump(boresch_cv_document(ANCHORS)),
                                     source="boresch", particles=len(GOOD))

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "umbrella.yaml"
        path.write_text(yaml.safe_dump(boresch_window_document(measured, centre_nm=1.2)),
                        encoding="utf-8")
        restraints = load_umbrella_definition(path, definition)

    assert len(restraints) == 6
    scanned = [r for r in restraints if r.kind == "distance"]
    assert len(scanned) == 1 and scanned[0].units == "nanometers"
    assert scanned[0].centre == pytest.approx(1.2)
    assert scanned[0].centre_deg is None, "a distance has no centre in degrees"
    held = [r for r in restraints if r.kind != "distance"]
    assert len(held) == 5 and all(r.units == "degrees" for r in held)


def test_the_dissociation_coordinate_cannot_also_be_held():
    """Scanning r and holding r at its bound value are contradictory instructions."""
    measured = check_anchors(GOOD, ANCHORS)
    with pytest.raises(BoreschError, match="dissociation coordinate"):
        boresch_window_document(measured, centre_nm=1.2, hold=("r", "thA"))


def test_a_held_term_with_no_measured_reference_is_refused():
    """A reference pose is measured, never chosen: typed values describe a pose never occupied."""
    measured = check_anchors(GOOD, ANCHORS)
    del measured["phC"]
    with pytest.raises(BoreschError, match="no measured reference"):
        boresch_window_document(measured, centre_nm=1.2)


def test_the_standard_state_note_states_what_is_missing_rather_than_computing_it():
    """A number quoted here would be used as though this module had validated it. It has not."""
    note = standard_state_note(10000.0, 100.0, temperature_k=300.0)
    assert "NOT a binding free energy" in note
    assert "standard-state correction" in note
    # The rms excursions it does quote are checkable arithmetic: sqrt(kT/k).
    kt = 0.0083144621 * 300.0
    assert f"{math.sqrt(kt / 10000.0):.4f}" in note


def test_the_six_restraints_build_into_a_system_and_bias_it():
    """All six terms must be constructible together, with each kind's own switch."""
    openmm = pytest.importorskip("openmm")
    from openmm import Context, Platform, System, VerletIntegrator, unit

    from md_tools.md.coordinate_restraints import (ANGLE_RESTRAINT_PARAMETER, AngleRestraint,
                                                   DISTANCE_RESTRAINT_PARAMETER,
                                                   DistanceRestraint)
    from md_tools.md.torsion_restraints import TORSION_RESTRAINT_PARAMETER, TorsionRestraint

    measured = check_anchors(GOOD, ANCHORS)
    system = System()
    for _ in range(len(GOOD)):
        system.addParticle(12.0)

    distance = DistanceRestraint(system)
    distance.add_distance(ANCHORS.atoms_for("r"), centre_nm=1.2, scale=10000.0)
    angles = AngleRestraint(system)
    for term in ("thA", "thB"):
        angles.add_angle(ANCHORS.atoms_for(term), measured[term], scale=100.0)
    torsions = TorsionRestraint(system)
    for term in ("phA", "phB", "phC"):
        torsions.add_torsion(ANCHORS.atoms_for(term), measured[term], scale=100.0)

    context = Context(system, VerletIntegrator(0.001 * unit.picosecond),
                      Platform.getPlatformByName("Reference"))
    context.setPositions(GOOD * unit.nanometer)

    # Everything off: the System carrying the forces is unbiased.
    assert context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) == pytest.approx(0.0, abs=1e-12)

    # The five HELD terms are at their measured values, so switching only those two parameters
    # on must cost nothing -- the pose is exactly where they hold it.
    context.setParameter(ANGLE_RESTRAINT_PARAMETER, 1.0)
    context.setParameter(TORSION_RESTRAINT_PARAMETER, 1.0)
    held_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert held_energy == pytest.approx(0.0, abs=1e-6), (
        "the held terms are centred on the measured pose, so they cost nothing there")

    # The scanned distance is centred 1.2 nm away from where the ligand is, so it must cost
    # exactly the harmonic amount -- and that is what drags the ligand outward, window by window.
    context.setParameter(DISTANCE_RESTRAINT_PARAMETER, 1.0)
    total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert total == pytest.approx(0.5 * 10000.0 * (measured["r"] - 1.2) ** 2, rel=1e-7)
