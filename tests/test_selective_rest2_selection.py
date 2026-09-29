"""Selective REST2 resolution (0.6.1 S1-A, S1-B, S1-D, S1-E): which atoms and which torsions.

Against the boundary fixture (`tests/selective_rest2_fixture.py`): two chains with a disulfide
between them, a proline-like omega, glycine, caps, and two copies of one ligand with a different
ligand between them. Each test names residues by their ONE-BASED TOPOLOGY index.

PLATFORM_POLICY_EXEMPTION: topology resolution and parameter bookkeeping; no Context is created.
"""
from __future__ import annotations

import pytest

from md_tools.rest2.regions import (EXCLUSIONS_FORMAT, SELECTION_POLICY, atom_category,
                                    central_bond_owner, cmap_decisions, exclusion_file_changes,
                                    explicit_selection, has_selectors, resolve_region)
from md_tools.rest2.selection import (EXPLICIT_MODE, SELECTION_FORMAT, ScalingSelection,
                                      SelectionError)

from .selective_rest2_fixture import RESIDUE, build_fixture


@pytest.fixture(scope="module")
def fx():
    return build_fixture()


def _classified(fx, region, tmp_path=None):
    """The torsion classification of exactly the residues the region touches, as the scaler does."""
    from md_tools.openmm.system import unscaled_torsions

    sdfs = {}
    wanted = set(region["classification_atoms"])
    names = {a.residue.name for a in fx.topology.atoms() if a.index in wanted}
    if tmp_path is not None:
        fx.write(tmp_path)
        sdfs = {name: tmp_path / f"{name}.sdf" for name in ("LGA", "LGB") if name in names}
    return unscaled_torsions(fx.topology, region["classification_atoms"], residue_sdfs=sdfs)


#: The package every LGA instance is recorded as, in a synthetic `ligand_mapping.json`.
LGA_PACKAGE = "LOCAL-ABCDEFGHIJKLMN/param_0123456789ab"


def mapping_for(fixture):
    """A `ligand_mapping.json`-shaped record naming each LGA/LGB residue's package."""
    instances = []
    for residue in fixture.topology.residues():
        if residue.name in ("LGA", "LGB"):
            parameter = "param_0123456789ab" if residue.name == "LGA" else "param_ba9876543210"
            instances.append({"residue_name": residue.name,
                              "package": {"compound_id": "LOCAL-ABCDEFGHIJKLMN",
                                          "parameter_id": parameter,
                                          "residue_name": residue.name},
                              "resolved": {"residue_index": residue.index}})
    return {"schema": "md-tools-ligand-mapping/1", "instances": instances}


def _select(fx, config, tmp_path=None, **kwargs):
    kwargs.setdefault("ligand_mapping", mapping_for(fx))
    region = resolve_region(fx.topology, config, config_dir=tmp_path, **kwargs)
    return region, explicit_selection(fx.topology, fx.system, region,
                                      _classified(fx, region, tmp_path))


def _bond(fx, a, b):
    return tuple(sorted((fx.atom(*a), fx.atom(*b))))


# --- membership ----------------------------------------------------------------------------------

def test_backbone_membership_covers_termini_glycine_proline_and_caps(fx):
    def names(number, category):
        return sorted(a.name for a in fx.residue(number).atoms() if atom_category(a) == category)

    assert names(RESIDUE["A.ALA"], "backbone") == ["C", "CA", "H", "HA", "N", "O"]
    assert names(RESIDUE["A.ALA"], "sidechain") == ["CB", "HB1", "HB2", "HB3"]
    assert names(RESIDUE["A.GLY"], "sidechain") == [], "glycine has no sidechain"
    assert names(RESIDUE["A.GLY"], "backbone") == ["C", "CA", "H", "HA2", "HA3", "N", "O"]
    assert names(RESIDUE["A.PRO"], "backbone") == ["C", "CA", "HA", "N", "O"]
    assert "CD" in names(RESIDUE["A.PRO"], "sidechain"), "proline's ring is sidechain"
    assert "SG" in names(RESIDUE["A.CYX"], "sidechain")
    assert names(RESIDUE["A.ACE"], "sidechain") == [] and names(RESIDUE["A.NME"], "sidechain") == []
    assert all(atom_category(a) is None for a in fx.residue(RESIDUE["LGA#1"]).atoms())


def test_central_bond_ownership_table(fx):
    atoms = list(fx.topology.atoms())

    def owner(a, b):
        return central_bond_owner(atoms[fx.atom(*a)], atoms[fx.atom(*b)])

    ser, phe, pro = RESIDUE["A.SER"], RESIDUE["A.PHE"], RESIDUE["A.PRO"]
    assert owner((ser, "N"), (ser, "CA")) == ((ser - 1, "backbone"),)          # phi
    assert owner((ser, "CA"), (ser, "C")) == ((ser - 1, "backbone"),)          # psi
    assert owner((ser, "CA"), (ser, "CB")) == ((ser - 1, "sidechain"),)        # chi1
    assert owner((phe, "C"), (pro, "N")) == ((phe - 1, "backbone"),), "omega_i is residue i's"
    assert owner((pro, "CD"), (pro, "N")) == ((pro - 1, "sidechain"),), "the proline ring"
    a7, b10 = RESIDUE["A.CYX"], RESIDUE["B.CYX"]
    assert owner((a7, "SG"), (b10, "SG")) == ((a7 - 1, "sidechain"), (b10 - 1, "sidechain"))
    lga = RESIDUE["LGA#1"]
    assert owner((lga, "C2"), (lga, "C3")) == ((lga - 1, "ligand"),)


# --- the two modes -------------------------------------------------------------------------------

def test_no_selector_is_the_legacy_mode_and_an_empty_mapping_is_not():
    assert not has_selectors({})
    assert not has_selectors({"method": "REST2", "backbone_scaling_list": None})
    assert has_selectors({"ligand_scaling_dict": {}})
    assert has_selectors({"sidechain_scaling_list": ":3"})


def test_an_omitted_category_means_none(fx):
    region, selection = _select(fx, {"backbone_scaling_list": ":2"})
    ala = {a.index for a in fx.residue(RESIDUE["A.ALA"]).atoms()}
    backbone = {a.index for a in fx.residue(RESIDUE["A.ALA"]).atoms()
                if atom_category(a) == "backbone"}
    assert set(selection.solute_atoms) == backbone, "the ALA sidechain was not asked for"
    assert not set(selection.solute_atoms) & (set(range(fx.system.getNumParticles())) - ala)
    assert selection.mode == EXPLICIT_MODE


@pytest.mark.parametrize("config, words", [
    ({"ligand_scaling_dict": {}}, "resolves to NOTHING"),
    ({"sidechain_scaling_list": f":{RESIDUE['A.GLY']}"}, "resolves to NOTHING"),
    ({"sidechain_scaling_list": f":{RESIDUE['A.ACE']}"}, "resolves to NOTHING"),
])
def test_an_explicit_region_that_heats_nothing_is_refused(fx, config, words):
    with pytest.raises(SelectionError, match=words):
        resolve_region(fx.topology, config)


# --- refusals --------------------------------------------------------------------------------------

@pytest.mark.parametrize("config, words", [
    ({"backbone_scaling_list": ":1897"}, None),
    ({"backbone_scaling_list": ":1898"}, "out of range"),
    ({"backbone_scaling_list": ":16"}, "solvent"),
    ({"backbone_scaling_list": ":13"}, "ligand_scaling_dict"),
    ({"sidechain_scaling_list": ":2@CA"}, "atom selector"),
    ({"ligand_scaling_dict": {"L01": {"mask": ":4"}}}, "not a ligand"),
    ({"ligand_scaling_dict": {"L01": {"mask": ":13,15"}}}, "ONE instance"),
    ({"ligand_scaling_dict": {"L01": "L01-exclusions.yaml"}}, "compact form"),
    ({"ligand_scaling_dict": {"L01": {"mask": ":13", "exclusions": "auto"}}}, "unknown key"),
    ({"ligand_scaling_dict": {"L01": {"torsion_exclusions": "auto"}}}, "`mask` is required"),
    ({"ligand_scaling_dict": {"L01": {"mask": ":13"}, "L02": {"mask": ":13"}}},
     "already selected"),
])
def test_refusals_name_what_could_not_be_resolved(fx, config, words):
    if words is None:           # the last residue is a water: refused as solvent, not out of range
        with pytest.raises(SelectionError, match="solvent"):
            resolve_region(fx.topology, config)
        return
    with pytest.raises(SelectionError, match=words):
        resolve_region(fx.topology, config)


def test_a_partial_region_on_an_implicit_system_is_refused(fx):
    with pytest.raises(SelectionError, match="implicit"):
        resolve_region(fx.topology, {"backbone_scaling_list": ":2"}, implicit=True)


# --- phi / psi / chi1 boundaries -------------------------------------------------------------------

def test_backbone_selection_takes_phi_and_psi_across_the_residue_boundary(fx):
    ser = RESIDUE["A.SER"]
    _, selection = _select(fx, {"backbone_scaling_list": f":{ser}"})
    bonds = set(selection.torsion_bonds)
    assert _bond(fx, (ser, "N"), (ser, "CA")) in bonds, "phi: C(i-1)-N-CA-C"
    assert _bond(fx, (ser, "CA"), (ser, "C")) in bonds, "psi: N-CA-C-N(i+1)"
    assert _bond(fx, (ser, "CA"), (ser, "CB")) not in bonds, "chi1 belongs to the sidechain"
    # The ordinary amide omega_i (SER C - PHE N) is owned here and PROTECTED.
    omega = _bond(fx, (ser, "C"), (ser + 1, "N"))
    assert omega in selection.excluded_bonds and omega not in bonds
    # The previous residue's C (an atom of phi) is not hot: torsion and nonbonded are separate.
    assert fx.atom(ser - 1, "C") not in selection.solute_atoms


def test_sidechain_selection_reaches_chi1_through_backbone_n_and_ca(fx):
    ser = RESIDUE["A.SER"]
    _, selection = _select(fx, {"sidechain_scaling_list": f":{ser}"})
    assert _bond(fx, (ser, "CA"), (ser, "CB")) in selection.torsion_bonds, "chi1"
    assert _bond(fx, (ser, "CB"), (ser, "OG")) in selection.torsion_bonds, "chi2"
    assert _bond(fx, (ser, "N"), (ser, "CA")) not in selection.torsion_bonds, "phi is backbone"
    assert fx.atom(ser, "N") not in selection.solute_atoms
    assert fx.atom(ser, "CA") not in selection.solute_atoms
    assert fx.atom(ser, "OG") in selection.solute_atoms


def test_proline_like_omega_is_scaled_by_its_owner_and_only_by_it(fx):
    phe, pro = RESIDUE["A.PHE"], RESIDUE["A.PRO"]
    omega = _bond(fx, (phe, "C"), (pro, "N"))
    _, owner = _select(fx, {"backbone_scaling_list": f":{phe}"})
    assert omega in owner.torsion_bonds and omega not in owner.excluded_bonds
    assert any("proline-like" in label["reason"] for label in owner.labels)
    _, other = _select(fx, {"backbone_scaling_list": f":{pro}"})
    assert omega not in other.torsion_bonds


def test_aromatic_ring_bonds_stay_protected_in_a_sidechain_selection(fx):
    phe = RESIDUE["A.PHE"]
    _, selection = _select(fx, {"sidechain_scaling_list": f":{phe}"})
    ring = _bond(fx, (phe, "CG"), (phe, "CD1"))
    assert ring in selection.excluded_bonds and ring not in selection.torsion_bonds
    assert _bond(fx, (phe, "CB"), (phe, "CG")) in selection.torsion_bonds, "chi2"


def test_a_disulfide_needs_both_sidechains(fx):
    a7, b10 = RESIDUE["A.CYX"], RESIDUE["B.CYX"]
    ss = _bond(fx, (a7, "SG"), (b10, "SG"))
    region, one = _select(fx, {"sidechain_scaling_list": f":{a7}"})
    assert ss not in one.torsion_bonds
    assert [e["bond"] for e in region["partially_owned_central_bonds"]] == [list(ss)]
    _, both = _select(fx, {"sidechain_scaling_list": f":{a7},{b10}"})
    assert ss in both.torsion_bonds


def test_numbering_runs_across_chains_without_a_reset(fx):
    region = resolve_region(fx.topology, {"backbone_scaling_list": ":8-9"})
    assert region["residue_map"]["8"]["residue_name"] == "NME"
    assert region["residue_map"]["8"]["chain"] == "A"
    assert region["residue_map"]["9"]["residue_name"] == "ACE"
    assert region["residue_map"]["9"]["chain"] == "B"
    assert region["residue_map"]["9"]["residue_id"] == "9", "author id, printed beside the index"


# --- ligand instances ----------------------------------------------------------------------------

def test_one_copy_of_a_compound_is_selected_and_its_twin_is_not(fx, tmp_path):
    first, twin = RESIDUE["LGA#1"], RESIDUE["LGA#2"]
    region, selection = _select(
        fx, {"ligand_scaling_dict": {"L01": {"mask": f":{first}", "torsion_exclusions": "auto"}}},
        tmp_path)
    hot = set(selection.solute_atoms)
    assert hot == {a.index for a in fx.residue(first).atoms()}
    assert not hot & {a.index for a in fx.residue(twin).atoms()}
    assert all(fx.atom(twin, "C2") not in b for b in selection.torsion_bonds)
    # Classified from bond orders: the C=C double bond and the amide C-N stay unscaled.
    assert _bond(fx, (first, "C1"), (first, "C2")) in selection.excluded_bonds
    assert _bond(fx, (first, "C3"), (first, "N1")) in selection.excluded_bonds
    assert _bond(fx, (first, "C2"), (first, "C3")) in selection.torsion_bonds
    assert _bond(fx, (first, "N1"), (first, "C4")) in selection.torsion_bonds
    instance = region["ligand_instances"][0]
    assert instance["label"] == "L01" and instance["topology_residue"] == first
    assert instance["residue_key"] == [fx.residue(first).chain.id, str(fx.residue(first).id), ""]


def test_two_distinct_ligands_are_selected_independently(fx, tmp_path):
    config = {"ligand_scaling_dict": {"A": {"mask": f":{RESIDUE['LGA#2']}"},
                                      "B": {"mask": f":{RESIDUE['LGB']}"}}}
    region, selection = _select(fx, config, tmp_path)
    hot = set(selection.solute_atoms)
    assert hot == ({a.index for a in fx.residue(RESIDUE["LGA#2"]).atoms()}
                   | {a.index for a in fx.residue(RESIDUE["LGB"]).atoms()})
    assert [i["label"] for i in region["ligand_instances"]] == ["A", "B"]


def _exclusions(path, pairs, residue_name="LGA", parameters=LGA_PACKAGE):
    import yaml

    document = {"format": EXCLUSIONS_FORMAT, "residue_name": residue_name,
                "central_bonds": [list(p) for p in pairs]}
    if parameters is not None:
        document["parameters"] = parameters
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_an_exclusion_file_is_mapped_by_package_local_names_and_saved_by_content(fx, tmp_path):
    first = RESIDUE["LGA#1"]
    _exclusions(tmp_path / "L01.yaml", [("C2", "C3")])
    config = {"ligand_scaling_dict": {"L01": {"mask": f":{first}",
                                              "torsion_exclusions": "L01.yaml"}}}
    region, selection = _select(fx, config, tmp_path)
    bond = _bond(fx, (first, "C2"), (first, "C3"))
    assert bond in selection.excluded_bonds and bond not in selection.torsion_bonds
    # ...and the file never removes an atom from the nonbonded set.
    assert fx.atom(first, "C2") in selection.solute_atoms
    block = region["ligand_instances"][0]["torsion_exclusions"]
    assert block["contents"] == (tmp_path / "L01.yaml").read_text(encoding="utf-8")
    assert block["central_bonds"] == [["C2", "C3"]]
    assert len(block["sha256"]) == 64


@pytest.mark.parametrize("pairs, residue_name, parameters, words", [
    ([("C1", "H1")], "LGA", LGA_PACKAGE, "not a central bond"),
    ([("C2", "C9")], "LGA", LGA_PACKAGE, "does not have"),
    ([("C2", "C3")], "LGB", LGA_PACKAGE, "written for residue"),
    ([("C2", "C3")], "LGA", None, "must name the parameter package"),
    ([("C2", "C3")], "LGA", "LOCAL-ABCDEFGHIJKLMN/param_ba9876543210", "mean nothing"),
])
def test_an_exclusion_file_that_protects_nothing_or_is_unbound_is_refused(
        fx, tmp_path, pairs, residue_name, parameters, words):
    _exclusions(tmp_path / "L01.yaml", pairs, residue_name, parameters)
    config = {"ligand_scaling_dict": {"L01": {"mask": f":{RESIDUE['LGA#1']}",
                                              "torsion_exclusions": "L01.yaml"}}}
    with pytest.raises(SelectionError, match=words):
        _select(fx, config, tmp_path)


def test_an_exclusion_file_on_an_instance_with_no_recorded_package_is_refused(fx, tmp_path):
    _exclusions(tmp_path / "L01.yaml", [("C2", "C3")])
    config = {"ligand_scaling_dict": {"L01": {"mask": f":{RESIDUE['LGA#1']}",
                                              "torsion_exclusions": "L01.yaml"}}}
    with pytest.raises(SelectionError, match="no recorded parameter package"):
        _select(fx, config, tmp_path, ligand_mapping=None)


def test_an_exclusion_on_a_bond_with_no_torsion_term_is_refused(fx, tmp_path):
    """A candidate bond (both ends have neighbours) whose force field gave it no proper term."""
    from openmm import PeriodicTorsionForce, XmlSerializer

    first = RESIDUE["LGA#1"]
    target = {fx.atom(first, "N1"), fx.atom(first, "C4")}
    stripped = XmlSerializer.deserialize(XmlSerializer.serialize(fx.system))
    force = next(f for f in stripped.getForces() if isinstance(f, PeriodicTorsionForce))
    for term in range(force.getNumTorsions()):
        i, j, k, l, n, phase, kv = force.getTorsionParameters(term)
        if {j, k} == target:
            force.setTorsionParameters(term, i, l, k, j, n, phase, kv)   # no longer across it
    _exclusions(tmp_path / "L01.yaml", [("N1", "C4")])
    config = {"ligand_scaling_dict": {"L01": {"mask": f":{first}",
                                              "torsion_exclusions": "L01.yaml"}}}
    region = resolve_region(fx.topology, config, config_dir=tmp_path,
                            ligand_mapping=mapping_for(fx))
    with pytest.raises(SelectionError, match="central bond of no proper torsion"):
        explicit_selection(fx.topology, stripped, region, _classified(fx, region, tmp_path))


def test_a_mutated_exclusion_file_is_detected_and_the_record_still_holds_what_ran(fx, tmp_path):
    first = RESIDUE["LGA#1"]
    path = _exclusions(tmp_path / "L01.yaml", [("C2", "C3")])
    config = {"ligand_scaling_dict": {"L01": {"mask": f":{first}",
                                              "torsion_exclusions": "L01.yaml"}}}
    _, before = _select(fx, config, tmp_path)
    document = before.to_document()
    assert exclusion_file_changes(document, config_dir=tmp_path) == []

    _exclusions(path, [("N1", "C4")])
    changes = exclusion_file_changes(document, config_dir=tmp_path)
    assert len(changes) == 1 and "L01" in changes[0]
    _, after = _select(fx, config, tmp_path)
    assert after.digest() != before.digest(), "a changed file is a different Hamiltonian"
    # The saved record reproduces the ORIGINAL exclusions without the file.
    restored = ScalingSelection.from_document(document)
    assert restored.as_scaler_arguments() == before.as_scaler_arguments()
    assert _bond(fx, (first, "C2"), (first, "C3")) in restored.excluded_bonds


def test_reordered_atoms_keep_the_named_exclusion_on_the_same_chemical_bond(tmp_path):
    straight, reversed_ = build_fixture(), build_fixture(reverse_lga_atoms=True)
    first = RESIDUE["LGA#1"]
    selections = []
    for fixture, where in ((straight, tmp_path / "a"), (reversed_, tmp_path / "b")):
        # The mapping record is per fixture: residue indices are the fixture's own.
        where.mkdir()
        _exclusions(where / "L01.yaml", [("C2", "C3")])
        config = {"ligand_scaling_dict": {"L01": {"mask": f":{first}",
                                                  "torsion_exclusions": "L01.yaml"}}}
        selections.append((fixture, _select(fixture, config, where)[1]))

    def by_name(fixture, selection, attribute):
        atoms = list(fixture.topology.atoms())
        return sorted(tuple(sorted(atoms[i].name for i in bond))
                      for bond in getattr(selection, attribute))

    (fa, sa), (fb, sb) = selections
    assert sa.torsion_bonds != sb.torsion_bonds, "the indices really did move"
    assert by_name(fa, sa, "torsion_bonds") == by_name(fb, sb, "torsion_bonds")
    assert by_name(fa, sa, "excluded_bonds") == by_name(fb, sb, "excluded_bonds")
    assert sa.topology_sha256 != sb.topology_sha256, "a record is bound to ITS topology"


def test_reordered_ligands_select_the_instance_the_mask_names(tmp_path):
    fixture = build_fixture(ligand_order=("LGB", "LGA", "LGA"))
    region = resolve_region(fixture.topology,
                            {"ligand_scaling_dict": {"L": {"mask": ":13"}}})
    assert region["residue_map"]["13"]["residue_name"] == "LGB"


# --- CMAP ----------------------------------------------------------------------------------------

def test_cmap_is_scaled_only_for_the_selected_residues_backbone(fx):
    ser = RESIDUE["A.SER"]
    _, selection = _select(fx, {"backbone_scaling_list": f":{ser}"})
    decisions = selection.to_document()["cmap_decisions"]
    scaled = [d for d in decisions if d["scaled"]]
    assert len(scaled) == 1
    assert scaled[0]["phi_central_bond"] == list(_bond(fx, (ser, "N"), (ser, "CA")))
    _, sidechain = _select(fx, {"sidechain_scaling_list": f":{ser}"})
    assert not any(d["scaled"] for d in sidechain.to_document()["cmap_decisions"])


def test_a_mixed_cmap_pair_is_left_unchanged_and_reported(fx):
    ser = RESIDUE["A.SER"]
    phi = _bond(fx, (ser, "N"), (ser, "CA"))
    decisions = cmap_decisions(fx.system, [phi])
    mixed = [d for d in decisions if d["reason"].startswith("MIXED")]
    assert len(mixed) == 1 and not mixed[0]["scaled"]
    assert "phi" in mixed[0]["reason"]


# --- the record ----------------------------------------------------------------------------------

def test_the_record_round_trips_and_carries_how_it_was_chosen(fx, tmp_path):
    config = {"backbone_scaling_list": ":2-4", "sidechain_scaling_list": ":3,11",
              "ligand_scaling_dict": {"L01": {"mask": ":13"}}}
    _, selection = _select(fx, config, tmp_path)
    path = selection.write(tmp_path / "selection.yaml")
    loaded = ScalingSelection.load(path, topology=fx.topology)
    assert loaded == selection
    document = loaded.to_document()
    assert document["format"] == SELECTION_FORMAT
    assert document["selection_mode"] == "explicit"
    assert document["policy"] == SELECTION_POLICY
    assert document["masks"]["backbone"] == ":2-4"
    assert document["masks"]["numbering"] == "topology-residue-index-1based"
    assert set(document["residue_map"]) == {"2", "3", "4", "11", "13"}
    assert document["residue_map"]["3"]["categories"] == ["backbone", "sidechain"]
    assert document["improper_policy"] == {"unscaled_impropers": True}


def test_a_record_is_refused_against_a_different_topology(fx, tmp_path):
    _, selection = _select(fx, {"backbone_scaling_list": ":2"})
    path = selection.write(tmp_path / "selection.yaml")
    other = build_fixture(ligand_order=("LGB", "LGA", "LGA"))
    with pytest.raises(SelectionError, match="topology_sha256"):
        ScalingSelection.load(path, topology=other.topology)


def test_a_version_1_record_is_read_as_the_legacy_selection(tmp_path):
    import yaml

    path = tmp_path / "solute.yaml"
    path.write_text(yaml.safe_dump({"format": "md-tools-solute-selection/1.0",
                                    "solute_atoms": [0, 1, 2], "unscaled_torsion_central_bonds":
                                    [[1, 2]]}), encoding="utf-8")
    loaded = ScalingSelection.load(path)
    assert loaded.mode == "legacy-full-solute"
    assert loaded.as_scaler_arguments()["torsion_central_bonds"] is None
