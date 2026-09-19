"""The topology plan: endpoint recovery with every dummy, bonded and constraint term accounted.

A raw comparison of an endpoint System's energy with the physical endpoint's is NOT made here as a
pass criterion: the dummies' retained bonded terms are in one and not the other. Each test that
compares energies first asserts that the raw difference is large -- so it could not pass by
accident -- and then that the named accounting closes it per force class.

Energies run on OpenMM's Reference platform (float64). This is a construction check, not CUDA
evidence, and it claims nothing about a device.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("rdkit")

from tests.alchemy_fixtures import (CHLOROETHANE, ENERGY_TOL_KJ, ETHANE, ETHANOL,  # noqa: E402
                                    PACKAGES, core_map, independent_reference, package,
                                    vacuum_environment, water_environment)


@pytest.fixture(scope="module")
def eta():
    return package(ETHANE)


@pytest.fixture(scope="module")
def cle():
    return package(CHLOROETHANE)


@pytest.fixture(scope="module")
def eoh():
    return package(ETHANOL)


@pytest.fixture(scope="module")
def water():
    return water_environment()


def _build(a, b, amap, env, mode, **kwargs):
    from md_tools.alchemy.topology import build_topology_plan

    return build_topology_plan(a, b, amap, env, mode=mode, **kwargs)


@pytest.fixture(scope="module")
def hybrid(eta, cle, water):
    return _build(eta, cle, core_map(eta, cle), water, "hybrid")


def _accounting(plan, env, pkg, side, positions=None):
    from md_tools.alchemy.topology_recovery import endpoint_accounting

    reference, index = independent_reference(plan, env, pkg, side)
    return endpoint_accounting(plan, side, reference, index, positions_nm=positions)


def _assert_closes(accounting, *, raw_at_least):
    assert abs(accounting["raw_total_difference"]) > raw_at_least, (
        "the raw endpoint difference is small, so this comparison could pass without the "
        "accounting; the fixture no longer tests anything", accounting)
    worst = max(abs(v) for v in accounting["residual"].values())
    assert worst < ENERGY_TOL_KJ, accounting


# ------------------------------------------------------------------------------------------------
# the index space and the maps
# ------------------------------------------------------------------------------------------------
def test_hybrid_particle_sets_and_both_map_directions(hybrid, eta, cle, water):
    record = hybrid.record
    n_env = water.system.getNumParticles()
    assert hybrid.common == frozenset(range(7))            # C1 C2 H1..H5 of ethane, in place
    assert hybrid.a_only == frozenset({7})                 # ethane H6
    assert hybrid.b_only == frozenset({n_env})             # chloroethane Cl1, appended
    assert hybrid.system_a.getNumParticles() == hybrid.system_b.getNumParticles() == n_env + 1
    amap = record["atom_map"]
    assert amap["by_name"] == [[n, n] for n in ("C1", "C2", "H1", "H2", "H3", "H4", "H5")]
    for a, b in amap["a_to_b"].items():
        assert amap["b_to_a"][str(b)] == int(a)
    hyb_a = record["endpoints"]["A"]["hybrid_index_of_local_atom"]
    hyb_b = record["endpoints"]["B"]["hybrid_index_of_local_atom"]
    for a, b in amap["a_to_b"].items():
        assert hyb_a[int(a)] == hyb_b[b]
    assert hyb_b[cle.atom_names.index("Cl1")] == n_env


def test_environment_numbering_survives_combination(hybrid, water):
    numbering = hybrid.record["numbering"]
    assert numbering["source_residues_unchanged"] == water.topology.getNumResidues()
    source = list(water.topology.residues())
    combined = list(hybrid.topology.residues())
    for before, after in zip(source, combined):
        assert (before.index, before.name, before.id, [a.name for a in before.atoms()]) == \
               (after.index, after.name, after.id, [a.name for a in after.atoms()])
    appended = numbering["appended_residue"]
    assert appended["index_1based"] == len(source) + 1 and appended["name"] == "CLE"
    assert [a.index for a in combined[-1].atoms()] == sorted(hybrid.b_only)


def test_the_environment_particles_are_exactly_the_environment_at_A(hybrid, water):
    """System A IS the environment plus appended particles: nothing already there changed."""
    from openmm import XmlSerializer

    env = XmlSerializer.serialize(water.system)
    assert hybrid.record["environment"]["system_sha256"]
    for k, force in enumerate(water.system.getForces()):
        mine = hybrid.system_a.getForce(k)
        assert type(mine) is type(force)
    nb_env = next(f for f in water.system.getForces() if type(f).__name__ == "NonbondedForce")
    nb_a = next(f for f in hybrid.system_a.getForces() if type(f).__name__ == "NonbondedForce")
    for i in range(water.system.getNumParticles()):
        assert nb_a.getParticleParameters(i) == nb_env.getParticleParameters(i)
    for k in range(nb_env.getNumExceptions()):
        assert nb_a.getExceptionParameters(k) == nb_env.getExceptionParameters(k)
    assert env  # serialisable


# ------------------------------------------------------------------------------------------------
# energy: endpoint recovery against independently built physical endpoints
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["A", "B"])
def test_hybrid_endpoints_recover_the_physical_endpoints_term_by_term(hybrid, water, eta, cle,
                                                                      side):
    accounting = _accounting(hybrid, water, eta if side == "A" else cle, side)
    # At A the dummy Cl1 sits where the superposed conformer put it, so its retained bond, bend
    # and torsion are far from zero: the raw difference is tens of kJ/mol.
    _assert_closes(accounting, raw_at_least=1e-2)
    # The dispersion correction counts the dummy: a real, named term, larger than the tolerance.
    assert abs(accounting["dispersion_correction_shift"]) > 100 * ENERGY_TOL_KJ


@pytest.mark.parametrize("side", ["A", "B"])
def test_recovery_holds_away_from_the_built_coordinates(hybrid, water, eta, cle, side):
    rng = np.random.default_rng(7)
    x = hybrid.positions_nm + rng.normal(scale=0.01, size=hybrid.positions_nm.shape)
    accounting = _accounting(hybrid, water, eta if side == "A" else cle, side, positions=x)
    _assert_closes(accounting, raw_at_least=1e-2)


def test_dispersion_shift_is_required_to_close_the_accounting(hybrid, water, eta):
    """The guard can fail: without the dispersion term the nonbonded class does not close."""
    accounting = _accounting(hybrid, water, eta, "A")
    unaccounted = accounting["residual"]["NonbondedForce"] + \
        accounting["dispersion_correction_shift"]
    assert abs(unaccounted) > 100 * ENERGY_TOL_KJ


@pytest.mark.parametrize("side", ["A", "B"])
def test_dummies_interact_with_nothing(hybrid, side):
    """Move every dummy anywhere, onto a water if need be: only its own bonded energy changes."""
    from md_tools.alchemy.topology_recovery import _energies_by_class, dummy_energy

    dummies = sorted(hybrid.dummies(side))
    x0 = hybrid.positions_nm
    rng = np.random.default_rng(11)
    base = _energies_by_class(hybrid.system(side), x0)
    base_dummy = sum(dummy_energy(hybrid.record, side, x0).values())
    for trial in range(3):
        x = x0.copy()
        x[dummies] = x[8 + 3 * trial] + rng.normal(scale=0.05, size=(len(dummies), 3))
        moved = _energies_by_class(hybrid.system(side), x)
        moved_dummy = sum(dummy_energy(hybrid.record, side, x).values())
        # everything that changed is the dummies' own energy: retained bonded terms and, for a
        # group with internal pairs, its internal nonbonded energy -- nothing with the solvent
        changed = sum(moved[k] - base[k] for k in moved)
        assert abs(changed - (moved_dummy - base_dummy)) < ENERGY_TOL_KJ


# ------------------------------------------------------------------------------------------------
# separability of the dummy groups
# ------------------------------------------------------------------------------------------------
def test_junction_rule_makes_every_dummy_group_separable(hybrid):
    detail = next(c["detail"] for c in hybrid.record["checks"]
                  if c["check"] == "dummy-factorization")
    assert len(detail["groups"]) == 2
    for group in detail["groups"]:
        assert group["retained_terms_max_variation_kj_mol"] < 1e-9
        # Keeping every term (OpenFE's default) would couple the dummy to physical internal
        # coordinates: the same measurement moves by whole kJ/mol. This is what the check refuses.
        assert group["all_terms_max_variation_kj_mol"] > 1.0
        assert group["terms_removed"] > 0


def test_the_retained_frame_is_the_documented_one(hybrid, cle):
    group = next(g for g in hybrid.record["dummy_groups"] if g["dummy_at"] == "A")
    assert group["local_names"] == ["Cl1"]
    names = {h: n for n, h in zip(cle.atom_names,
                                  hybrid.record["endpoints"]["B"]["hybrid_index_of_local_atom"])}
    frame = [names[group["frame"][k]] for k in ("p1", "p2", "p3")]
    assert frame == ["C2", "C1", "H1"]      # heavy neighbour first, then lowest package index
    kept = [(s["atoms"], s["kind"]) for family in hybrid.record["terms"].values() for s in family
            if s["at_dummy_end"] == "dummy-retained" and group["atoms"][0] in s["atoms"]]
    kept_names = sorted((tuple(names[a] for a in atoms), kind) for atoms, kind in kept)
    assert kept_names == sorted([
        (("C2", "Cl1"), "bond"), (("C1", "C2", "Cl1"), "angle"),
        (("Cl1", "C2", "C1", "H1"), "proper"), (("Cl1", "C2", "C1", "H1"), "proper")]) or \
        kept_names == sorted([
            (("C2", "Cl1"), "bond"), (("C1", "C2", "Cl1"), "angle"),
            (("H1", "C1", "C2", "Cl1"), "proper"), (("H1", "C1", "C2", "Cl1"), "proper")])


def test_a_group_that_would_not_separate_is_refused(hybrid, monkeypatch):
    """Keep one removed coupling term, and the factorization check refuses the plan."""
    import copy

    from md_tools.alchemy import topology_recovery as recovery
    from md_tools.alchemy.topology import TopologyError

    record = copy.deepcopy(hybrid.record)
    for family in record["terms"].values():
        for slot in family:
            if slot["at_dummy_end"] == "dummy-removed":
                slot["at_dummy_end"] = "dummy-retained"
                slot["a" if slot["role"] == "b-only" else "b"] = \
                    slot["b" if slot["role"] == "b-only" else "a"]
                break
        else:
            continue
        break
    broken = type(hybrid)(record=record, system_a=hybrid.system_a, system_b=hybrid.system_b,
                          topology=hybrid.topology, positions_nm=hybrid.positions_nm)
    with pytest.raises(TopologyError, match="partition\\s+function would not cancel"):
        recovery.factorization_check(broken)


# ------------------------------------------------------------------------------------------------
# structure: missing, duplicated and omitted terms
# ------------------------------------------------------------------------------------------------
def test_term_accounting_reproduces_both_package_tables(hybrid, eta, cle):
    detail = next(c["detail"] for c in hybrid.record["checks"] if c["check"] == "term-accounting")
    tables = detail["endpoint_tables_reproduced"]
    # every angle and torsion of each package, and every bond that is not a constraint
    assert tables["A"]["angles"] == len(eta.table["angles"])
    assert tables["B"]["proper_torsions"] == len([r for r in cle.table["proper_torsions"]
                                                  if r[6] != 0.0])
    assert tables["B"]["exceptions"] == len(cle.table["exceptions"])


@pytest.mark.parametrize("damage", ["drop-b-term", "duplicate-a-term", "charge-on-dummy",
                                    "environment-term"])
def test_the_audit_catches_a_damaged_system(hybrid, eta, cle, water, damage):
    from openmm import XmlSerializer

    from md_tools.alchemy.topology_recovery import audit_plan
    from md_tools.alchemy.topology import TopologyError

    system_a = XmlSerializer.clone(hybrid.system_a)
    system_b = XmlSerializer.clone(hybrid.system_b)
    forces = {type(f).__name__: f for f in system_b.getForces()}
    if damage == "drop-b-term":
        slot = next(s for s in hybrid.record["terms"]["angles"] if s["role"] == "b-only")
        f = forces["HarmonicAngleForce"]
        i, j, k, theta, _ = f.getAngleParameters(slot["slot"])
        f.setAngleParameters(slot["slot"], i, j, k, theta, 0.0)
    elif damage == "duplicate-a-term":
        f = {type(f).__name__: f for f in system_a.getForces()}["HarmonicAngleForce"]
        slot = next(s for s in hybrid.record["terms"]["angles"] if s["role"] == "core")
        f.addAngle(*slot["atoms"], *slot["a"])
        f_b = forces["HarmonicAngleForce"]
        f_b.addAngle(*slot["atoms"], *slot["b"])
    elif damage == "charge-on-dummy":
        f = {type(f).__name__: f for f in system_a.getForces()}["NonbondedForce"]
        d = min(hybrid.b_only)
        _, s, _ = f.getParticleParameters(d)
        f.setParticleParameters(d, 0.1, s, 0.0)
    else:
        f = forces["NonbondedForce"]
        water_oxygen = 8          # the first water after the eight ethane atoms
        q, s, e = f.getParticleParameters(water_oxygen)
        f.setParticleParameters(water_oxygen, q, s, e * 1.01)
    broken = type(hybrid)(record=hybrid.record, system_a=system_a, system_b=system_b,
                          topology=hybrid.topology, positions_nm=hybrid.positions_nm)
    with pytest.raises(TopologyError):
        audit_plan(broken, eta, cle, water.system)


def test_every_dummy_exception_and_every_cross_pair_is_zero(hybrid):
    record = hybrid.record
    for slot in record["nonbonded"]["exclusions"]:
        assert slot["reason"] == "a-only-x-b-only"
        assert set(slot["atoms"]) & hybrid.a_only and set(slot["atoms"]) & hybrid.b_only
    assert len(record["nonbonded"]["exclusions"]) == len(hybrid.a_only) * len(hybrid.b_only)
    for slot in record["nonbonded"]["exceptions"]:
        if slot["role"] == "b-only":
            assert slot["a"][0] == 0.0 and slot["a"][2] == 0.0
        if slot["role"] == "a-only":
            assert slot["b"][0] == 0.0 and slot["b"][2] == 0.0


# ------------------------------------------------------------------------------------------------
# constraints
# ------------------------------------------------------------------------------------------------
def test_constraints_are_one_set_and_match_each_physical_endpoint(eta, eoh, water):
    """Ethane -> ethanol: the O-H of the appended dummy group is constrained under HBonds."""
    plan = _build(eta, eoh, core_map(eta, eoh), water, "hybrid")
    record = plan.record
    assert record["constraints"]["policy"] == "HBonds"
    hyb_b = record["endpoints"]["B"]["hybrid_index_of_local_atom"]
    o1, h6 = hyb_b[eoh.atom_names.index("O1")], hyb_b[eoh.atom_names.index("H6")]
    assert [c[:2] for c in record["constraints"]["appended"]] == [[o1, h6]]

    def pairs(system):
        rows = [system.getConstraintParameters(k) for k in range(system.getNumConstraints())]
        return {tuple(sorted(row[:2])): row[2] for row in rows}

    assert pairs(plan.system_a) == pairs(plan.system_b)
    for side, pkg in (("A", eta), ("B", eoh)):
        reference, index = independent_reference(plan, water, pkg, side)
        physical = {tuple(sorted((index[i], index[j]))): d
                    for (i, j), d in pairs(reference).items()}
        mine = pairs(plan.system(side))
        dummies = plan.dummies(side)
        # every constraint of the physical endpoint, at its length
        for key, d in physical.items():
            assert abs(mine[key].value_in_unit(d.unit) - d._value) < 1e-12
        # and the rest are wholly inside a dummy group plus its anchor
        extra = set(mine) - set(physical)
        groups = [set(g["atoms"]) | {g["anchor"]["p1"]} for g in record["dummy_groups"]
                  if g["dummy_at"] == side]
        assert extra and all(any(set(p) <= g for g in groups) for p in extra)
        assert all(set(p) & dummies for p in extra)


def test_ethanol_recovery_with_a_two_atom_dummy_group(eta, eoh, water):
    plan = _build(eta, eoh, core_map(eta, eoh), water, "hybrid")
    for side, pkg in (("A", eta), ("B", eoh)):
        _assert_closes(_accounting(plan, water, pkg, side), raw_at_least=1e-2)


def test_a_constraint_that_would_appear_along_the_path_is_refused(eta, cle, water):
    from md_tools.alchemy.topology import TopologyError

    amap = core_map(eta, cle, {"H6": "Cl1"})
    with pytest.raises(TopologyError, match="constrained at A and flexible at B"):
        _build(eta, cle, amap, water, "single")


# ------------------------------------------------------------------------------------------------
# single and dual
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("target,extra", [(CHLOROETHANE, {"H6": "Cl1"}), (ETHANOL, {"H6": "O1"})])
def test_single_topology_in_vacuum_recovers_both_endpoints(eta, target, extra):
    b = package(target)
    env = vacuum_environment(eta)
    plan = _build(eta, b, core_map(eta, b, extra), env, "single")
    assert not plan.a_only and len(plan.common) == 8
    for side, pkg in (("A", eta), ("B", b)):
        accounting = _accounting(plan, env, pkg, side)
        worst = max(abs(v) for v in accounting["residual"].values())
        assert worst < ENERGY_TOL_KJ, accounting
    if target == CHLOROETHANE:
        # every atom mapped: no dummies, and the H -> Cl particle keeps endpoint A's mass
        assert not plan.b_only and not plan.record["dummy_groups"]
        change = plan.record["particles"]["masses"]["core_mass_changes"]
        assert [c["particle"] for c in change] == [7]


def test_dual_topology_shares_no_particle_and_restrains_the_centroids(eta, cle, water):
    plan = _build(eta, cle, core_map(eta, cle), water, "dual")
    assert not plan.common
    assert plan.a_only == frozenset(range(8)) and len(plan.b_only) == 8
    assert len(plan.record["nonbonded"]["exclusions"]) == 64
    restraint = plan.record["restraint"]
    assert restraint["present_at"] == "both endpoints, identically"
    for side, pkg in (("A", eta), ("B", cle)):
        accounting = _accounting(plan, water, pkg, side)
        assert abs(accounting["restraint"]) >= 0.0
        worst = max(abs(v) for v in accounting["residual"].values())
        assert worst < ENERGY_TOL_KJ, accounting
        # the dummy ligand keeps every bonded term of its own
        assert sum(accounting["dummy"].values()) != 0.0


def test_dual_restraint_accounting_is_not_vacuous(eta, cle, water):
    plan = _build(eta, cle, core_map(eta, cle), water, "dual")
    x = plan.positions_nm.copy()
    x[sorted(plan.b_only)] += np.array([0.1, 0.0, 0.0])
    accounting = _accounting(plan, water, cle, "A", positions=x)
    assert accounting["restraint"] > 1.0
    assert abs(accounting["residual"]["CustomCentroidBondForce"]) < ENERGY_TOL_KJ


# ------------------------------------------------------------------------------------------------
# refusals about the environment
# ------------------------------------------------------------------------------------------------
def test_an_environment_holding_the_other_endpoint_is_refused(eta, cle, water):
    from md_tools.alchemy.topology import TopologyError

    with pytest.raises(TopologyError, match="not package"):
        _build(cle, eta, core_map(cle, eta), water, "hybrid")


def test_an_environment_whose_ligand_parameters_differ_is_refused(eta, cle, water):
    from openmm import XmlSerializer

    from md_tools.alchemy.topology import Environment, TopologyError

    system = XmlSerializer.clone(water.system)
    nb = next(f for f in system.getForces() if type(f).__name__ == "NonbondedForce")
    q, s, e = nb.getParticleParameters(0)
    nb.setParticleParameters(0, q * 1.01, s, e)
    damaged = Environment(system=system, topology=water.topology, positions_nm=water.positions_nm,
                          ligand=water.ligand)
    with pytest.raises(TopologyError, match="immutable"):
        _build(eta, cle, core_map(eta, cle), damaged, "hybrid")


def test_repartitioned_masses_are_refused(eta, cle, water):
    from openmm import XmlSerializer

    from md_tools.alchemy.topology import Environment, TopologyError

    system = XmlSerializer.clone(water.system)
    system.setParticleMass(2, 3.024)
    damaged = Environment(system=system, topology=water.topology, positions_nm=water.positions_nm,
                          ligand=water.ligand)
    with pytest.raises(TopologyError, match="repartitioning"):
        _build(eta, cle, core_map(eta, cle), damaged, "hybrid")


def test_separated_topology_is_refused_by_name(eta, cle, water):
    from md_tools.alchemy.topology import TopologyError

    with pytest.raises(TopologyError, match="separated topology is deferred"):
        _build(eta, cle, core_map(eta, cle), water, "separated")


# ------------------------------------------------------------------------------------------------
# provenance, serialisation and relocation
# ------------------------------------------------------------------------------------------------
def test_every_source_is_hashed(hybrid, eta, cle):
    record = hybrid.record
    for side, pkg in (("A", eta), ("B", cle)):
        endpoint = record["endpoints"][side]
        assert endpoint["package_sha256"] == pkg.package_sha256
        assert endpoint["parameter_digest"] == pkg.metadata["parameter_digest"]
        assert endpoint["chemical_state_digest"] == pkg.metadata["chemical_state"]["digest"]
        assert len(endpoint["ffxml_sha256"]) == len(endpoint["molecule_sdf_sha256"]) == 64
    env = record["environment"]
    for key in ("system_sha256", "topology_sha256", "positions_sha256", "system_file_sha256",
                "pdb_file_sha256"):
        assert len(env[key]) == 64
    assert len(record["atom_map"]["sha256"]) == 64
    assert len(record["coordinates"]["positions_sha256"]) == 64
    assert len(record["plan_sha256"]) == 64


def test_building_does_not_touch_the_packages(eta, cle, water):
    before = (eta.package_sha256, cle.package_sha256)
    _build(eta, cle, core_map(eta, cle), water, "hybrid")
    assert (package(ETHANE).package_sha256, package(CHLOROETHANE).package_sha256) == before


def test_the_plan_is_deterministic(eta, cle, water):
    one = _build(eta, cle, core_map(eta, cle), water, "hybrid")
    two = _build(eta, cle, core_map(eta, cle), water, "hybrid")
    assert one.sha256 == two.sha256


def test_a_written_plan_moves_and_reloads_with_identical_energies(hybrid, tmp_path):
    from md_tools.alchemy.topology_recovery import _energies_by_class
    from md_tools.alchemy.topology import load_plan

    written = hybrid.write(tmp_path / "plan")
    moved = tmp_path / "elsewhere" / "renamed"
    moved.parent.mkdir()
    os.rename(written, moved)
    loaded = load_plan(moved, package_roots=[PACKAGES])
    assert loaded.record == hybrid.record
    assert loaded.common == hybrid.common and loaded.b_only == hybrid.b_only
    assert np.array_equal(loaded.positions_nm, hybrid.positions_nm)
    for side in ("A", "B"):
        assert _energies_by_class(loaded.system(side), loaded.positions_nm) == \
               _energies_by_class(hybrid.system(side), hybrid.positions_nm)


@pytest.mark.parametrize("damage", ["system_b.xml", "plan.json", "extra-file"])
def test_a_modified_plan_does_not_load(hybrid, tmp_path, damage):
    from md_tools.alchemy.topology import TopologyError, load_plan

    written = hybrid.write(tmp_path / "plan")
    if damage == "system_b.xml":
        path = written / damage
        path.write_text(path.read_text().replace('q="0"', 'q="0.01"', 1))
    elif damage == "plan.json":
        document = json.loads((written / damage).read_text())
        document["particles"]["a_only"] = []
        (written / damage).write_text(json.dumps(document))
    else:
        (written / "notes.txt").write_text("x")
    with pytest.raises(TopologyError):
        load_plan(written)


def test_a_plan_is_never_written_over(hybrid, tmp_path):
    from md_tools.alchemy.topology import TopologyError

    (tmp_path / "plan").mkdir()
    with pytest.raises(TopologyError, match="exists"):
        hybrid.write(tmp_path / "plan")
    assert not any(p.name.startswith(".plan-") for p in tmp_path.iterdir())


def test_loading_against_other_packages_is_refused(hybrid, tmp_path, eta):
    from md_tools.alchemy.topology import TopologyError, load_plan
    from md_tools.ligands import PackageError

    written = hybrid.write(tmp_path / "plan")
    fake_root = tmp_path / "catalog"
    compound, parameter = CHLOROETHANE.split("/")
    shutil.copytree(PACKAGES / ETHANE, fake_root / compound / parameter)
    shutil.copytree(PACKAGES / ETHANE, fake_root / ETHANE)
    with pytest.raises((TopologyError, PackageError)):
        load_plan(written, package_roots=[fake_root])
    with pytest.raises(TopologyError, match="is in none of"):
        load_plan(written, package_roots=[tmp_path / "empty"])


def test_a_plan_from_packages_registered_in_a_temporary_catalog(tmp_path, monkeypatch, water):
    """"Registered endpoint parameters", with the machine's $MD_DATA never read.

    MD_DATA is SET to an empty temporary root (unsetting it would fall back to the user
    configuration). Both endpoints are registered there through the catalog's own write-once path
    and resolved back by reference, and the plan records exactly those identities.
    """
    from md_tools.ligands.catalog import default_catalog_root, register_package, resolve_package

    monkeypatch.setenv("MD_DATA", str(tmp_path / "md_data"))
    catalog = default_catalog_root()
    assert catalog == tmp_path / "md_data" / "parameters" / "ligands"
    for reference in (ETHANE, CHLOROETHANE):
        _, destination, new = register_package(PACKAGES / reference, catalog)
        assert new and destination == catalog / reference
    a = resolve_package(ETHANE, roots=[catalog])
    b = resolve_package(CHLOROETHANE, roots=[catalog])
    assert a.path.is_relative_to(tmp_path) and b.path.is_relative_to(tmp_path)
    plan = _build(a, b, core_map(a, b), water, "hybrid")
    assert plan.record["endpoints"]["A"]["reference"] == ETHANE
    assert plan.record["endpoints"]["B"]["package_sha256"] == package(CHLOROETHANE).package_sha256
    written = plan.write(tmp_path / "plan")
    from md_tools.alchemy.topology import load_plan

    assert load_plan(written, package_roots=[catalog]).sha256 == plan.sha256


def test_a_plan_rewritten_from_a_loaded_plan_is_the_same_plan(hybrid, tmp_path):
    """write -> load -> write -> load keeps combined.pdb byte-identical and its digest fixed.

    An OpenMM PDB write/read round trip reorders the bond between the ligand and the appended
    residue on every pass, so a plan that re-serialised its topology would flip digests forever.
    """
    from md_tools.alchemy.topology import load_plan
    from md_tools.rest2.selection import topology_digest

    first = load_plan(hybrid.write(tmp_path / "one"))
    second = load_plan(first.write(tmp_path / "two"))
    assert (tmp_path / "one" / "combined.pdb").read_bytes() == \
           (tmp_path / "two" / "combined.pdb").read_bytes()
    digest = hybrid.record["numbering"]["combined_topology_sha256"]
    assert topology_digest(hybrid.topology) == topology_digest(second.topology) == digest
    assert second.sha256 == hybrid.sha256


def test_recovery_has_no_platform_choice(hybrid, water, eta):
    """Recovery Contexts are Reference by construction: there is no second platform policy."""
    import inspect

    from md_tools.alchemy import topology_recovery as recovery

    for function in (recovery.endpoint_accounting, recovery.audit_plan,
                     recovery._energies_by_class, recovery._dispersion):
        assert "platform" not in inspect.signature(function).parameters, function.__name__
    assert _accounting(hybrid, water, eta, "A")["platform"] == recovery.RECOVERY_PLATFORM == \
        "Reference"


def _dummy_free_energy(record, positions, group, frame, include_removed, kt):
    """-kT ln Z of one single-atom dummy group, by quadrature over its position around P1.

    Z = integral of exp(-U_dummy/kT) d^3x over the dummy's position, U_dummy the bonded energy of
    every term touching it at its dummy endpoint (numpy, vectorised over the grid; the per-point
    `dummy_energy` it mirrors is checked against OpenMM in the recovery tests). Spherical
    coordinates about P1 in the (P1, P2, P3) frame, Jacobian r^2 sin(theta), the same grid for
    both conformations.
    """
    from md_tools.alchemy.topology_recovery import _frame

    (d,) = group["atoms"]
    side = group["dummy_at"].lower()
    other = "b" if side == "a" else "a"
    origin, axes = _frame(positions, frame["p1"], frame["p2"], frame["p3"])
    r, theta, phi = np.meshgrid(np.linspace(0.15, 0.21, 41),
                                np.linspace(1e-3, np.pi - 1e-3, 121),
                                np.linspace(-np.pi, np.pi, 144, endpoint=False), indexing="ij")
    local = np.stack([r * np.cos(theta), r * np.sin(theta) * np.cos(phi),
                      r * np.sin(theta) * np.sin(phi)], axis=-1)
    dummy = origin + local @ axes                      # (..., 3) positions of the dummy

    def at(atom):
        return dummy if atom == d else positions[atom]

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    energy = np.zeros(r.shape)
    for family, slots in record["terms"].items():
        for slot in slots:
            if d not in slot["atoms"]:
                continue
            if slot["at_dummy_end"] == "dummy-retained":
                params = slot[side]
            elif include_removed and slot["at_dummy_end"] == "dummy-removed":
                params = slot[other]
            else:
                continue
            p = [at(a) for a in slot["atoms"]]
            if family == "bonds":
                length = np.linalg.norm(p[0] - p[1], axis=-1)
                energy += 0.5 * params[1] * (length - params[0]) ** 2
            elif family == "angles":
                cos = np.sum(unit(p[0] - p[1]) * unit(p[2] - p[1]), axis=-1)
                energy += 0.5 * params[1] * (np.arccos(np.clip(cos, -1, 1)) - params[0]) ** 2
            else:
                b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
                b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
                v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
                w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
                angle = np.arctan2(np.sum(np.cross(b1, v) * w, axis=-1), np.sum(v * w, axis=-1))
                energy += params[1] * (1.0 + np.cos(slot["periodicity"] * angle - params[0]))
    weights = np.exp(-(energy - energy.min()) / kt) * r ** 2 * np.sin(theta)
    return energy.min() - kt * np.log(weights.sum())


def test_openfe_dummy_group_limitation_is_addressed_as_a_free_energy(hybrid):
    """OpenFE documents that keeping every dummy-core bonded term can bias the result, because the
    dummy partition function then depends on the physical conformation. Measured here as a free
    energy: -kT ln Z_dummy in two conformations of the physical core, differing only in core
    internal coordinates (an H-C-C bend and an H-C-C-H twist around the anchor).

    With the junction rule the dummy free energy is the same in both -- it cancels in any cycle.
    Keeping every term, as OpenFE does, it differs by a free energy a cycle would silently absorb.
    """
    record = hybrid.record
    group = next(g for g in record["dummy_groups"] if g["dummy_at"] == "A")   # Cl1 at endpoint A
    frame = group["frame"]
    kt = 2.494339  # kJ/mol at 300 K
    x1 = np.array(hybrid.positions_nm, dtype=float)
    x2 = x1.copy()
    hyb_a = record["endpoints"]["A"]["hybrid_index_of_local_atom"]
    c2, h4 = hyb_a[1], hyb_a[5]            # ethane C2 and H4, a physical neighbour of the anchor
    x2[h4] = x1[c2] + 1.08 * (x1[h4] - x1[c2]) + np.array([0.02, -0.015, 0.01])
    for h in (hyb_a[3], hyb_a[4]):         # H2, H3 on C1: move the torsion references
        x2[h] = x1[h] + np.array([0.01, 0.02, -0.01])
    kept = [_dummy_free_energy(record, x, group, frame, False, kt) for x in (x1, x2)]
    everything = [_dummy_free_energy(record, x, group, frame, True, kt) for x in (x1, x2)]
    print(f"dummy free energy, retained terms: {kept[0]:.6f} -> {kept[1]:.6f} kJ/mol; "
          f"every term: {everything[0]:.6f} -> {everything[1]:.6f} kJ/mol")
    assert abs(kept[1] - kept[0]) < 1e-9
    # the OpenFE-style set: a conformation-dependent dummy free energy, well above sampling noise
    assert abs(everything[1] - everything[0]) > 0.1, everything


# ------------------------------------------------------------------------------------------------
# a unique group's internal nonbonded interactions (S0 ruling: kept physical at the dummy end)
# ------------------------------------------------------------------------------------------------
def _pentane_plan(env):
    from tests.alchemy_fixtures import PENTANE

    a, b = package(ETHANE), package(PENTANE)
    return a, b, _build(a, b, core_map(a, b), env, "hybrid")


def _internal_by_hand(pkg, hyb, group_hyb, x):
    """The group's internal nonbonded energy straight from the PACKAGE table, not the plan.

    Every exception with both atoms in the group, and every other pair inside it with
    Lorentz-Berthelot parameters, as vacuum Coulomb + Lennard-Jones.
    """
    import itertools
    import math

    from md_tools.alchemy.topology import COULOMB_CONSTANT

    local = {h: i for i, h in enumerate(hyb)}
    members = sorted(local[h] for h in group_hyb)
    exceptions = {(r[0], r[1]): r[2:5] for r in pkg.table["exceptions"]}

    def pair(i, j, qq, sigma, epsilon):
        r = float(np.linalg.norm(x[hyb[i]] - x[hyb[j]]))
        return COULOMB_CONSTANT * qq / r + 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    total, n_exc, n_pairs = 0.0, 0, 0
    for i, j in itertools.combinations(members, 2):
        if (i, j) in exceptions:
            total += pair(i, j, *exceptions[(i, j)])
            n_exc += exceptions[(i, j)][0] != 0.0 or exceptions[(i, j)][2] != 0.0
        else:
            qi, si, ei = pkg.table["atoms"][i][2:5]
            qj, sj, ej = pkg.table["atoms"][j][2:5]
            total += pair(i, j, qi * qj, 0.5 * (si + sj), math.sqrt(ei * ej))
            n_pairs += 1
    return total, n_exc, n_pairs


def test_the_pentane_fixture_can_see_internal_nonbonded_terms(water):
    a, b, plan = _pentane_plan(water)
    group = next(g for g in plan.record["dummy_groups"] if g["dummy_at"] == "A")
    assert sorted(group["local_names"]) == sorted(
        ["C3", "C4", "C5", "H6", "H7", "H8", "H9", "H10", "H11", "H12"])
    hyb = plan.record["endpoints"]["B"]["hybrid_index_of_local_atom"]
    energy, n_exc, n_pairs = _internal_by_hand(b, hyb, group["atoms"], plan.positions_nm)
    assert n_exc > 0 and n_pairs > 0 and abs(energy) > 100 * ENERGY_TOL_KJ


def test_internal_terms_are_accounted_at_the_dummy_end_and_match_the_package(water):
    a, b, plan = _pentane_plan(water)
    accounting = _accounting(plan, water, a, "A")
    _assert_closes(accounting, raw_at_least=1e-2)
    group = next(g for g in plan.record["dummy_groups"] if g["dummy_at"] == "A")
    hyb = plan.record["endpoints"]["B"]["hybrid_index_of_local_atom"]
    by_hand, _, _ = _internal_by_hand(b, hyb, group["atoms"], plan.positions_nm)
    named = accounting["dummy"]["internal_exceptions"] + accounting["dummy"]["internal_pairs"]
    assert abs(named - by_hand) < ENERGY_TOL_KJ
    assert accounting["dummy"]["internal_pairs"] != 0.0


def test_annihilating_the_internal_terms_fails_the_endpoint_test(water):
    """The guard can fail: the S0-ruled endpoint is NOT reproduced by a fully annihilated dummy."""
    from openmm import NonbondedForce, XmlSerializer

    from md_tools.alchemy.topology_recovery import (_dispersion, _energies_by_class,
                                                    dummy_energy)

    a, b, plan = _pentane_plan(water)
    annihilated = XmlSerializer.clone(plan.system_a)
    internal = plan.record["nonbonded"]["unique_group_internal"]
    for force in annihilated.getForces():
        if isinstance(force, NonbondedForce):
            for slot in plan.record["nonbonded"]["exceptions"]:
                if slot["unique_group_internal"] and slot["role"] == "b-only":
                    i, j = slot["atoms"]
                    force.setExceptionParameters(slot["slot"], i, j, 0.0, slot["a"][1], 0.0)
        if force.getName() == internal["force"]:
            for k in range(force.getNumBonds()):
                i, j, params = force.getBondParameters(k)
                force.setBondParameters(k, i, j, [0.0, params[1], 0.0])
    reference, index = independent_reference(plan, water, a, "A")
    x = plan.positions_nm
    expected = (sum(_energies_by_class(reference, x[index]).values())
                + sum(dummy_energy(plan.record, "A", x).values())
                + _dispersion(plan.system_a, x) - _dispersion(reference, x[index]))
    kept = sum(_energies_by_class(plan.system_a, x).values())
    gone = sum(_energies_by_class(annihilated, x).values())
    assert abs(kept - expected) < ENERGY_TOL_KJ
    assert abs(gone - expected) > 100 * ENERGY_TOL_KJ


@pytest.mark.parametrize("side", ["A", "B"])
def test_pentane_recovers_both_endpoints_in_vacuum(side):
    from tests.alchemy_fixtures import PENTANE

    a, b = package(ETHANE), package(PENTANE)
    env = vacuum_environment(a)
    plan = _build(a, b, core_map(a, b), env, "hybrid")
    accounting = _accounting(plan, env, a if side == "A" else b, side)
    worst = max(abs(v) for v in accounting["residual"].values())
    assert worst < ENERGY_TOL_KJ, accounting


def test_internal_terms_do_not_break_separability(water):
    a, b, plan = _pentane_plan(water)
    detail = next(c["detail"] for c in plan.record["checks"]
                  if c["check"] == "dummy-factorization")
    propyl = next(g for g in detail["groups"] if g["dummy_at"] == "A")
    assert propyl["retained_terms_max_variation_kj_mol"] < 1e-9


def test_the_internal_pair_force_never_sees_a_box(water):
    """Identical in every leg is what makes it cancel: no periodic images, whatever the box."""
    a, b, plan = _pentane_plan(water)
    assert water.system.usesPeriodicBoundaryConditions()
    name = plan.record["nonbonded"]["unique_group_internal"]["force"]
    for system in (plan.system_a, plan.system_b):
        force = next(f for f in system.getForces() if f.getName() == name)
        assert not force.usesPeriodicBoundaryConditions()
