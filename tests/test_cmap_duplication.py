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

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"
openmm = pytest.importorskip("openmm")

# The internals of the scaling are what these tests exercise, so they name the
# module rather than the package facade -- `md_tools.rest2` exports the public
# API, and a test of `_scale_cmap` is not using the public API.
from md_tools.rest2 import scaler as scaling                                    # noqa: E402

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


# --- the omega exclusion, recorded as torsions rather than only as a bond -----------------------

def _torsion_system(torsions, n_atoms=8):
    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0)
    force = openmm.PeriodicTorsionForce()
    for (i, j, k, l) in torsions:
        force.addTorsion(i, j, k, l, 1, 0.0, 10.0)
    system.addForce(force)
    return system


def test_the_report_names_every_torsion_an_excluded_bond_protects():
    """A stored bond needs a force field to mean anything. The report says which torsion terms
    the exclusion actually left unscaled, against the System the ladder was built from."""
    system = _torsion_system([(0, 1, 2, 3), (1, 2, 3, 4), (2, 1, 2, 5), (0, 1, 6, 7)])
    report = scaling.torsion_exclusion_report(system, range(6), [(1, 2)])

    assert report["detector_version"] == scaling.OMEGA_DETECTOR_VERSION
    assert report["excluded_central_bonds"] == [[1, 2]]
    assert report["excluded_torsion_indices"] == {"1-2": [0, 2]}, (
        "both torsions about the excluded central bond, not just the first"
    )
    assert report["n_excluded_torsions"] == 2
    assert report["n_scaled_solute_torsions"] == 1, "the environment-reaching torsion is neither"


def test_the_report_uses_the_same_predicate_as_the_scaling():
    """If the report and the scaling could disagree, the record would describe a different
    exclusion than the one performed."""
    system = _torsion_system([(0, 1, 2, 3), (1, 2, 3, 4)])
    report = scaling.torsion_exclusion_report(system, range(6), [(1, 2)])
    scaled = scaling.build_scaled_system(system, range(6), 0.5, excluded_bonds=[(1, 2)])
    force = [scaled.getForce(i) for i in range(scaled.getNumForces())][0]

    protected = report["excluded_torsion_indices"]["1-2"]
    for index in range(force.getNumTorsions()):
        k_value = force.getTorsionParameters(index)[6]
        magnitude = float(k_value.value_in_unit(k_value.unit))
        if index in protected:
            assert magnitude == pytest.approx(10.0), "an excluded torsion must be unscaled"
        else:
            assert magnitude == pytest.approx(2.5), "a scaled solute torsion follows (1-tau)^2"


def test_no_exclusions_reports_an_empty_mapping_not_a_missing_one():
    system = _torsion_system([(0, 1, 2, 3)])
    report = scaling.torsion_exclusion_report(system, range(6), [])
    assert report["excluded_central_bonds"] == []
    assert report["excluded_torsion_indices"] == {}
    assert report["n_excluded_torsions"] == 0
    assert report["n_scaled_solute_torsions"] == 1
