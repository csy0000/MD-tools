"""CMAP maps are shared lookup tables, so scaling one in place scales it for everyone.

A CMAP map is referenced by every torsion of the same residue type in the system, so "is this map
the solute's" is a question about its users and has three answers. Scaling a shared map in place
would scale the environment's energy too. Leaving it alone -- which is what this used to do -- is
the opposite error and just as wrong: a solute torsion sharing a map then gets NO scaling, and the
Hamiltonian quietly stops being the one the ladder claims.

PLATFORM_POLICY_EXEMPTION: force-object bookkeeping on synthetic CMAP forces. No dynamics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"
openmm = pytest.importorskip("openmm")
sys.path.insert(0, str(TEMPLATES))

import rest2_scaling as scaling                                    # noqa: E402

SOLUTE = set(range(6))
ENVIRONMENT = (6, 7, 8, 9, 10)


def _force(maps, torsions):
    """`maps` are constant energies; `torsions` are `(map, wholly_solute)`."""
    force = openmm.CMAPTorsionForce()
    for value in maps:
        force.addMap(2, [value] * 4)
    for map_index, solute in torsions:
        atoms = (0, 1, 2, 3, 1, 2, 3, 4) if solute else (6, 7, 8, 9, 7, 8, 9, 10)
        force.addTorsion(map_index, *atoms)
    return force


def _energies(force, index):
    return [float(e.value_in_unit(e.unit)) for e in force.getMapParameters(index)[1]]


# --- the four cases -----------------------------------------------------------------------------

def test_a_map_used_only_by_the_solute_is_scaled_in_place():
    force = _force([4.0], [(0, True)])
    duplicates = scaling.duplicate_shared_cmaps(force, SOLUTE)
    assert duplicates == {}, "an exclusively solute map needs no copy"
    scaling._scale_cmap(force, SOLUTE, 0.25, duplicates)
    assert force.getNumMaps() == 1
    assert _energies(force, 0) == [1.0] * 4


def test_a_map_used_only_by_the_environment_is_untouched():
    force = _force([4.0], [(0, False)])
    scaling._scale_cmap(force, SOLUTE, 0.25)
    assert force.getNumMaps() == 1, "no copy was needed"
    assert _energies(force, 0) == [4.0] * 4, "the environment's energy must not change"


def test_a_shared_map_is_duplicated_and_only_the_copy_is_scaled():
    force = _force([4.0], [(0, True), (0, False)])
    duplicates = scaling.duplicate_shared_cmaps(force, SOLUTE)

    assert duplicates == {1: 0}, "exactly one duplicate, of map 0"
    assert force.getNumMaps() == 2
    assert force.getTorsionParameters(0)[0] == 1, "the solute torsion points at the copy"
    assert force.getTorsionParameters(1)[0] == 0, "the environment torsion still points at the original"

    scaling._scale_cmap(force, SOLUTE, 0.25, duplicates)
    assert _energies(force, 0) == [4.0] * 4, "the original is left exactly as it was"
    assert _energies(force, 1) == [1.0] * 4, "the copy carries the scaling"


def test_many_solute_torsions_sharing_one_map_get_one_copy_between_them():
    force = _force([4.0], [(0, True), (0, True), (0, True), (0, False)])
    duplicates = scaling.duplicate_shared_cmaps(force, SOLUTE)

    assert len(duplicates) == 1, "one duplicate per MAP, not per torsion"
    assert force.getNumMaps() == 2
    assert [force.getTorsionParameters(i)[0] for i in range(4)] == [1, 1, 1, 0]

    scaling._scale_cmap(force, SOLUTE, 0.25, duplicates)
    assert _energies(force, 0) == [4.0] * 4
    assert _energies(force, 1) == [1.0] * 4


# --- the cold rung, and mixtures ----------------------------------------------------------------

def test_a_mixture_is_handled_map_by_map():
    #  0 shared, 1 solute-only, 2 environment-only
    force = _force([4.0, 8.0, 16.0],
                   [(0, True), (0, False), (1, True), (2, False)])
    duplicates = scaling.duplicate_shared_cmaps(force, SOLUTE)
    scaling._scale_cmap(force, SOLUTE, 0.25, duplicates)

    assert _energies(force, 0) == [4.0] * 4, "shared original untouched"
    assert _energies(force, 1) == [2.0] * 4, "solute-only scaled in place"
    assert _energies(force, 2) == [16.0] * 4, "environment-only untouched"
    assert _energies(force, 3) == [1.0] * 4, "the copy of the shared map"


def test_the_cold_rung_gains_no_duplicate_maps():
    """At tau = 0 the System must come back byte-equivalent to its source: no copies, no changes."""
    system = openmm.System()
    for _ in range(11):
        system.addParticle(12.0)
    system.addForce(_force([4.0], [(0, True), (0, False)]))
    scaled = scaling.build_scaled_system(system, SOLUTE, 0.0)
    force = [scaled.getForce(i) for i in range(scaled.getNumForces())][0]
    assert force.getNumMaps() == 1, "the cold rung must not gain a duplicate map"
    assert _energies(force, 0) == [4.0] * 4


def test_roles_name_all_three_populations():
    force = _force([1.0, 1.0, 1.0], [(0, True), (0, False), (1, True), (2, False)])
    roles = scaling.cmap_map_roles(force, SOLUTE)
    assert roles["shared"] == {0}
    assert roles["exclusive_solute"] == {1}
    assert roles["exclusive_other"] == {2}


# --- the live switching path --------------------------------------------------------------------

def test_repeated_switching_neither_accumulates_copies_nor_compounds_scaling():
    """The switcher rewrites map energies; it must never add a second copy, and each tau must be
    applied to the UNSCALED energies rather than on top of the previous tau."""
    system = openmm.System()
    for _ in range(11):
        system.addParticle(12.0)
    system.addForce(_force([4.0], [(0, True), (0, False)]))

    switcher = scaling.TauSwitcher(system, SOLUTE)
    prepared = switcher.prepared_system(0.0)
    force = [prepared.getForce(i) for i in range(prepared.getNumForces())][0]
    assert force.getNumMaps() == 2, "the copy is made once, when the System is prepared"

    reference = [switcher.base.getForce(i) for i in range(switcher.base.getNumForces())][0]
    duplicates = switcher._cmap_duplicates[0]
    assert duplicates == {1: 0}

    for tau, expected in ((0.5, 1.0), (0.25, 2.25), (0.5, 1.0)):
        scaling._restore_cmap(force, reference, duplicates)
        scaling._scale_cmap(force, SOLUTE, scaling.scaling_for_tau(tau)[0], duplicates)
        assert force.getNumMaps() == 2, "a switch added a map"
        assert _energies(force, 0) == [4.0] * 4, "the environment's map must never change"
        assert _energies(force, 1) == pytest.approx([expected] * 4), (
            f"tau={tau} must be applied to the unscaled energies, not composed on the previous")
