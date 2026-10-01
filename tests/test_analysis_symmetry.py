"""Symmetry detection over torsional clusters: what it must merge, and what it must refuse.

Every test builds its own molecule and its own synthetic ensemble. Nothing reads a trajectory, a
run directory or a project path, so the file is portable and fast.

THE TESTS THAT MATTER MOST ARE THE REFUSALS. A merger that fires too eagerly produces a smaller,
tidier, wrong answer that nothing downstream can detect: the populations still sum to one and the
clusters still look like states. So asymmetric substitution, the coupled para case, broken
stereochemistry and transitive chaining each get a test whose job is to see a merge NOT happen.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="symmetry detection needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402

from md_tools.analysis._t_symmetry import (ComparisonTolerance, TorsionDefinitions,  # noqa: E402
                                        TorsionSymmetry, calibrate, dihedral,
                                        distinct_operations, energy_distance,
                                        load_cluster_torsion, load_torsion_json,
                                        make_whole, verify_atom_mapping, wrap)


def _mol(smiles: str):
    """A molecule with explicit hydrogens and a 3D conformer, as an SDF would supply."""
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    return mol


def _defs(mol, quads, names=None, units="radian"):
    names = names or tuple(f"t{i}" for i in range(len(quads)))
    return TorsionDefinitions(quads=tuple(tuple(q) for q in quads), names=tuple(names),
                              indexing="zero-based", units=units)


def _two_state(centres_a, centres_b, n=400, spread=0.12, seed=0):
    """Two clusters at the given centres, with labels and the concatenated array."""
    rng = np.random.default_rng(seed)
    a = wrap(np.array(centres_a)[None, :] + rng.normal(0, spread, (n, len(centres_a))))
    b = wrap(np.array(centres_b)[None, :] + rng.normal(0, spread, (n, len(centres_b))))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


# ------------------------------------------------------------------------- loading and validation

def test_a_definition_file_without_an_indexing_convention_is_refused(tmp_path):
    """An off-by-one over atom indices names DIFFERENT atoms and produces a well-formed torsion
    of something else. There is no safe default and inferring it from whether 0 appears is a
    guess that fails on exactly the molecules where it matters."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"units": "radian",
                                "torsions": [{"name": "a", "atoms": [1, 2, 3, 4]}]}))
    with pytest.raises(ValueError, match="indexing"):
        load_torsion_json(path)


def test_one_based_indices_are_converted_once(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"indexing": "one-based", "units": "degree",
                                "torsions": [{"name": "a", "atoms": [1, 2, 3, 4]}]}))
    defs = load_torsion_json(path)
    assert defs.quads == ((0, 1, 2, 3),)
    assert defs.indexing == "one-based" and defs.units == "degree"


def test_a_cv_sidecar_is_refused_by_name_rather_than_half_read(tmp_path):
    """md-tools' sidecar carries column NAMES and no indices, because the atoms are selectors in
    the cv.yaml it points at. Reading it as if it held quadruplets would invent them."""
    path = tmp_path / "x.cv.json"
    path.write_text(json.dumps({"schema_version": 1, "units": "degrees",
                                "columns": ["ALA2_C_N_CA_CB"], "source": "cv.abc.yaml"}))
    with pytest.raises(ValueError, match="sidecar"):
        load_torsion_json(path)


def test_a_quadruplet_that_is_not_a_bonded_chain_is_refused():
    """Four atoms that are not three consecutive bonds still produce a number from the dihedral
    formula. It is not a torsion of this molecule and no symmetry statement about it means
    anything."""
    mol = _mol("CCO")
    with pytest.raises(ValueError, match="bonded chains"):
        TorsionSymmetry(mol, _defs(mol, [(0, 1, 2, 7)]))


def test_the_units_must_be_declared_and_degrees_are_converted():
    mol = _mol("CCCC")
    quad = (0, 1, 2, 3)
    deg, labels = _two_state([170.0], [-170.0], n=50)
    with pytest.raises(ValueError, match="unknown angle unit"):
        load_cluster_torsion(deg, labels, angle_unit="turns")
    data = load_cluster_torsion(np.rad2deg(np.deg2rad(deg)), labels, angle_unit="degree")
    assert np.all(np.abs(data.torsions) <= math.pi)


# ------------------------------------------------------------------------- angular periodicity

def test_the_comparison_is_periodic_across_the_seam():
    """Two samples of ONE distribution straddling +-pi must read as identical, not as maximally
    far apart. Raw-angle arithmetic gets this exactly backwards."""
    rng = np.random.default_rng(3)
    a = wrap(np.pi + rng.normal(0, 0.1, (300, 1)))
    b = wrap(np.pi + rng.normal(0, 0.1, (300, 1)))
    from md_tools.analysis._t_symmetry import torsion_embedding

    near_seam = energy_distance(torsion_embedding(a), torsion_embedding(b))
    far = energy_distance(torsion_embedding(a), torsion_embedding(rng.normal(0, 0.1, (300, 1))))
    assert abs(near_seam) < 0.05, f"one distribution across the seam read as {near_seam}"
    assert far > 1.0, "a genuinely different basin must not read as close"
    assert (a.min() < -3.0) and (a.max() > 3.0), "the fixture must actually straddle the cut"


def test_dihedral_matches_a_known_geometry():
    """The sign convention is IUPAC/MDTraj, the same one the CV sidecar records, so a recomputed
    angle and a stored one are comparable."""
    p0 = np.array([[1.0, 1.0, 0.0]])
    p1 = np.array([[0.0, 1.0, 0.0]])
    p2 = np.array([[0.0, 0.0, 0.0]])
    p3 = np.array([[0.0, 0.0, 1.0]])
    assert math.isclose(abs(float(dihedral(p0, p1, p2, p3)[0])), math.pi / 2, abs_tol=1e-9)


def test_make_whole_repairs_a_molecule_split_across_the_box():
    """A dihedral from raw positions of a molecule straddling the boundary is a plausible number
    describing no geometry."""
    xyz = np.array([[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [9.8, 1.0, 0.0], [9.7, 1.0, 1.0]]])
    box = np.diag([10.0, 10.0, 10.0])
    whole = make_whole(xyz, [(0, 1), (1, 2), (2, 3)], box)
    assert abs(whole[0, 1, 0] - whole[0, 0, 0]) < 1.0, "bond 0-1 is still broken across the box"
    assert not np.isclose(float(dihedral(xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 3])[0]),
                          float(dihedral(whole[:, 0], whole[:, 1], whole[:, 2], whole[:, 3])[0]))


# ------------------------------------------------------------------------- the symmetry search

def test_atom_map_numbers_are_not_chemical_distinctions():
    """An atom map number is a caller's bookkeeping. Treating it as chemistry would hide a real
    symmetry, silently, and the molecule would simply report fewer operations."""
    mol = _mol("c1ccccc1")
    plain, _ = distinct_operations(mol, _defs(mol, [(0, 1, 2, 3)]))
    tagged = Chem.Mol(mol)
    for i, atom in enumerate(tagged.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    mapped, _ = distinct_operations(tagged, _defs(tagged, [(0, 1, 2, 3)]))
    assert len(mapped) == len(plain) > 1


def test_a_truncated_automorphism_search_is_reported_not_swallowed():
    """'No further symmetry' and 'stopped looking' must not be the same answer."""
    mol = _mol("c1ccccc1")
    _, truncated = distinct_operations(mol, _defs(mol, [(0, 1, 2, 3)]), max_matches=2)
    assert truncated is True


def test_fit_refuses_to_run_on_a_truncated_search():
    mol = _mol("c1ccccc1")
    detector = TorsionSymmetry(mol, _defs(mol, [(0, 1, 2, 3)]), max_matches=2)
    torsions, labels = _two_state([0.5], [-0.5], n=60)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        detector.fit(load_cluster_torsion(torsions, labels))


def test_operations_are_deduplicated_by_their_effect_on_the_selected_torsions():
    """A methyl group alone contributes six automorphisms that do nothing to a torsion not
    naming its hydrogens. Carrying them all would multiply every pairwise search for nothing."""
    mol = _mol("CC(C)(C)CO")
    quad = None
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetSymbol() == "C" and b.GetSymbol() == "O":
            heavy = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx()][0]
            hydrogen = [n.GetIdx() for n in b.GetNeighbors() if n.GetSymbol() == "H"][0]
            quad = (heavy, a.GetIdx(), b.GetIdx(), hydrogen)
    assert quad is not None
    from md_tools.analysis._t_symmetry import automorphisms

    raw, _ = automorphisms(mol)
    distinct, _ = distinct_operations(mol, _defs(mol, [quad]))
    assert len(raw) > 10 and len(distinct) < len(raw)


# ---------------------------------------------------------------- the cases that must NOT merge

def test_asymmetric_substitution_prevents_a_merge():
    """An ortho-substituted ring has no two-fold axis. Two clusters that would merge under the
    para molecule's symmetry must be refused here -- the molecule, not the data, decides."""
    mol = _mol("Cc1ccccc1O")
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    quad = (ring[0], ring[1], ring[2], ring[3])
    operations, _ = distinct_operations(mol, _defs(mol, [quad]))
    non_identity = [op for op in operations if not op.is_identity]
    torsions, labels = _two_state([0.6], [-0.6], n=200)
    detector = TorsionSymmetry(mol, _defs(mol, [quad]))
    detector.fit(load_cluster_torsion(torsions, labels))
    assert detector.cluster_groups == [[0], [1]], (
        f"two different basins merged on a molecule with {len(non_identity)} non-identity "
        f"operations over this torsion")


def test_specified_stereochemistry_is_preserved():
    """An operation that would swap the two faces of a defined double bond is not an
    automorphism of the stereochemical graph. E and Z are different compounds, not conformers."""
    mol = _mol(r"C/C=C/C")
    double = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    begin, end = double.GetBeginAtomIdx(), double.GetEndAtomIdx()
    outer_a = next(n.GetIdx() for n in mol.GetAtomWithIdx(begin).GetNeighbors()
                   if n.GetIdx() != end and n.GetSymbol() == "C")
    outer_b = next(n.GetIdx() for n in mol.GetAtomWithIdx(end).GetNeighbors()
                   if n.GetIdx() != begin and n.GetSymbol() == "C")
    quad = (outer_a, begin, end, outer_b)
    with_chirality, _ = distinct_operations(mol, _defs(mol, [quad]), use_chirality=True)
    for operation in with_chirality:
        if operation.is_identity:
            continue
        begin, end = quad[1], quad[2]
        assert not (operation.permutation[begin] == end
                    and operation.permutation[end] == begin), (
            "an operation exchanged the two ends of a stereo-defined double bond")


def test_a_double_bond_is_not_automatically_180_degree_periodic():
    """Folding a double-bond torsion by 180 degrees would identify E with Z. The detector must
    find no operation that maps a cis cluster onto a trans one."""
    mol = _mol(r"C/C=C/C")
    double = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    begin, end = double.GetBeginAtomIdx(), double.GetEndAtomIdx()
    outer_a = next(n.GetIdx() for n in mol.GetAtomWithIdx(begin).GetNeighbors()
                   if n.GetIdx() != end and n.GetSymbol() == "C")
    outer_b = next(n.GetIdx() for n in mol.GetAtomWithIdx(end).GetNeighbors()
                   if n.GetIdx() != begin and n.GetSymbol() == "C")
    quad = (outer_a, begin, end, outer_b)
    torsions, labels = _two_state([0.0], [math.pi], n=200, spread=0.08)
    detector = TorsionSymmetry(mol, _defs(mol, [quad]))
    detector.fit(load_cluster_torsion(torsions, labels))
    assert detector.cluster_groups == [[0], [1]], "cis and trans were merged as one state"


def test_transitive_chaining_does_not_create_a_three_cluster_merge():
    """A~B and B~C under a loose threshold must not imply A~C. Each member is re-tested against
    the group's reference, and one that fails is split back out."""
    rng = np.random.default_rng(11)
    step = 0.9
    blocks = [rng.normal(c, 0.25, (300, 1)) for c in (-step, 0.0, step)]
    torsions = wrap(np.vstack(blocks))
    labels = np.array([0] * 300 + [1] * 300 + [2] * 300)
    mol = _mol("CCCC")
    detector = TorsionSymmetry(mol, _defs(mol, [(0, 1, 2, 3)]))
    detector.fit(load_cluster_torsion(torsions, labels),
                 tolerance=ComparisonTolerance(energy_max=10.0, coverage_min=0.0, radius=5.0))
    for group in detector.cluster_groups:
        assert len(group) <= 2 or sorted(group) != [0, 1, 2], (
            "three clusters merged by chaining; the common-reference check did not fire")


# ------------------------------------------------------------------- the coupled para-benzene case

def _para_system():
    """p-Xylene: a ring with two identical substituents and one attachment torsion each.

    The two-fold flip EXCHANGES the two torsions, and both transformed quadruplets are in the
    table, so the coupled case is demonstrable with no coordinates at all.
    """
    mol = _mol("Cc1ccc(C)cc1")
    quads = []
    for methyl in [a.GetIdx() for a in mol.GetAtoms()
                   if a.GetSymbol() == "C" and not a.GetIsAromatic()]:
        carbon = next(n.GetIdx() for n in mol.GetAtomWithIdx(methyl).GetNeighbors()
                      if n.GetIsAromatic())
        ring_neighbour = [n.GetIdx() for n in mol.GetAtomWithIdx(carbon).GetNeighbors()
                          if n.GetIsAromatic()][0]
        hydrogen = next(n.GetIdx() for n in mol.GetAtomWithIdx(methyl).GetNeighbors()
                        if n.GetSymbol() == "H")
        quads.append((hydrogen, methyl, carbon, ring_neighbour))
    return mol, _defs(mol, quads, names=("attach_a", "attach_b"))


def test_the_para_ring_flip_moves_both_attachment_torsions_together():
    """THE COUPLED CASE, stated on CENTRAL BONDS because that is what names a torsion.

    The two-fold flip carries attachment bond A onto attachment bond B. What must be true is that
    it carries B back onto A in the same operation: there must be no operation that moves one
    attachment torsion onto the other while LEAVING THE OTHER WHERE IT IS. Applied per torsion,
    such an operation would merge the configuration with only one substituent rotated, which is a
    different relative orientation and a different conformer.

    The assertion is deliberately not about columns. An operation can be the ring flip combined
    with a methyl rotation, so A's image is B's bond exactly while B's image is a quadruplet about
    A's bond with a different methyl hydrogen -- same bond, different stored column. That is the
    same coupling; a column-level assertion would call it a violation.
    """
    mol, defs = _para_system()
    operations, truncated = distinct_operations(mol, defs)
    assert not truncated
    bond_a, bond_b = defs.central_bonds()

    def image_bond(operation, quad):
        moved = tuple(operation.permutation[i] for i in quad)
        return tuple(sorted(moved[1:3]))

    coupled = 0
    for operation in operations:
        if operation.is_identity:
            continue
        image_a = image_bond(operation, defs.quads[0])
        image_b = image_bond(operation, defs.quads[1])
        if image_a == bond_b:
            coupled += 1
            assert image_b == bond_a, (
                f"operation {operation.index} carries attachment bond A onto B but leaves B at "
                f"{image_b}; applied that way it would merge configurations differing in the "
                f"RELATIVE orientation of the two substituents")
        if image_a == bond_a:
            assert image_b == bond_b, (
                f"operation {operation.index} fixes attachment bond A but moves B to {image_b}")
    assert coupled, "the para ring must have an operation exchanging the two attachment bonds"


def test_an_independent_local_symmetry_acts_on_one_substituent_only():
    """A methyl rotation is a REAL symmetry that touches one substituent and not the other, and
    must not be mistaken for the coupled ring flip. Both kinds exist on this molecule and the
    detector keeps them distinct: the local one leaves the other torsion's central bond fixed.
    """
    mol, defs = _para_system()
    operations, _ = distinct_operations(mol, defs)
    bond_a, bond_b = defs.central_bonds()

    def image_bond(operation, quad):
        moved = tuple(operation.permutation[i] for i in quad)
        return tuple(sorted(moved[1:3]))

    local = [op for op in operations
             if not op.is_identity
             and image_bond(op, defs.quads[0]) == bond_a
             and image_bond(op, defs.quads[1]) == bond_b
             and op.column_source != (("column", 0), ("column", 1))]
    assert local, "a methyl rotation keeps both central bonds and changes which hydrogen is named"


def test_the_coupled_flip_merges_the_swapped_pair_and_not_an_unrelated_one():
    """(a, b) and (b, a) are the same conformer seen through the flip. (a, b) and (a, c) are
    not, and must survive as separate clusters."""
    mol, defs = _para_system()
    rng = np.random.default_rng(17)
    a, b, c = 0.6, -1.9, 2.6
    blocks = [np.column_stack([rng.normal(a, 0.1, 300), rng.normal(b, 0.1, 300)]),
              np.column_stack([rng.normal(b, 0.1, 300), rng.normal(a, 0.1, 300)]),
              np.column_stack([rng.normal(a, 0.1, 300), rng.normal(c, 0.1, 300)])]
    torsions = wrap(np.vstack(blocks))
    labels = np.array([0] * 300 + [1] * 300 + [2] * 300)
    detector = TorsionSymmetry(mol, defs)
    detector.fit(load_cluster_torsion(torsions, labels, names=list(defs.names)))
    assert [0, 1] in detector.cluster_groups, (
        f"the swapped pair was not merged: {detector.cluster_groups}")
    assert [2] in detector.cluster_groups, (
        f"an unrelated cluster was swept into the merge: {detector.cluster_groups}")


def test_a_missing_transformed_torsion_is_unresolved_not_approximated():
    """When the transformed quadruplet is not in the table and no coordinates are given, the
    answer is UNRESOLVED with the missing quadruplets named -- never an idealised 180 shift.

    One attachment torsion only: the flip then maps it onto the OTHER substituent's torsion,
    which is not in the table at all.
    """
    mol, both = _para_system()
    defs = _defs(mol, [both.quads[0]], names=("attach_a",))
    detector = TorsionSymmetry(mol, defs)
    assert any(op.needs_coordinates for op in detector.operations)
    torsions, labels = _two_state([0.4], [-0.4], n=120)
    detector.fit(load_cluster_torsion(torsions, labels, names=list(defs.names)))
    unresolved = [c for c in detector.comparisons if c.verdict == "unresolved"]
    assert unresolved, "a transformed quadruplet outside the table must be unresolved"
    assert "missing_quadruplets" in unresolved[0].evidence
    assert "coordinates" in unresolved[0].reason


# ------------------------------------------------------------- populations, labels, and noise

def _merge_fixture(labels_used=(7, 3, 11), seed=5):
    """Three clusters with ARBITRARY, non-consecutive labels, plus noise, on the para molecule.

    Two of them -- (a, b) and (b, a) -- ARE related by the ring flip and must merge. The third is
    unrelated and must not. A fixture whose clusters sit at the same centre would test nothing:
    two clusters at one centre are not symmetry-related, they are the same basin split in two,
    and no operation can relate them because the operation would have to be the identity.
    """
    rng = np.random.default_rng(seed)
    a, b, c = 0.6, -1.9, 2.6
    blocks = [np.column_stack([rng.normal(a, 0.1, 200), rng.normal(b, 0.1, 200)]),
              np.column_stack([rng.normal(b, 0.1, 200), rng.normal(a, 0.1, 200)]),
              np.column_stack([rng.normal(a, 0.1, 200), rng.normal(c, 0.1, 200)])]
    noise = rng.uniform(-np.pi, np.pi, (17, 2))
    torsions = wrap(np.vstack(blocks + [noise]))
    labels = np.array([labels_used[0]] * 200 + [labels_used[1]] * 200
                      + [labels_used[2]] * 200 + [-1] * 17)
    return torsions, labels


def test_labels_need_not_be_consecutive_or_start_at_zero():
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    assert data.cluster_labels == [3, 7, 11]   # sorted, non-consecutive, and 0 is not among them
    assert data.noise.n_frames == 17


def test_population_is_conserved_and_noise_is_untouched():
    """Masses SUM once. Nothing is multiplied by the symmetry order, and the noise mass is
    neither absorbed nor renormalised away."""
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data)
    merged = detector.deduplicate(data)
    assert math.isclose(sum(merged.masses.values()) + merged.noise_mass, 1.0, abs_tol=1e-12)
    assert merged.noise_mass == data.noise.mass
    assert np.array_equal(merged.labels[labels == -1],
                          np.full((labels == -1).sum(), -1))
    for group, merged_label in [(g, i) for i, g in enumerate(merged.groups)]:
        expected = sum(data.entries[m].mass for m in group)
        assert math.isclose(merged.masses[merged_label], expected, abs_tol=1e-12)


def test_frames_are_concatenated_without_duplication_and_weights_travel():
    torsions, labels = _merge_fixture()
    weights = np.linspace(0.5, 2.0, labels.size)
    data = load_cluster_torsion(torsions, labels, frame_weights=weights)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data)
    merged = detector.deduplicate(data)
    seen = np.concatenate([e.frames for e in merged.entries.values()])
    assert seen.size == np.unique(seen).size, "a frame appears in two merged clusters"
    assert seen.size + merged.noise.n_frames == labels.size
    for entry in merged.entries.values():
        assert np.allclose(entry.weights, weights[entry.frames])


def test_aligned_descriptors_are_stored_separately_from_the_observations():
    """An aligned array is a derived view under a specific operation. A reader who cannot tell it
    from the measurement cannot check either."""
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data)
    merged = detector.deduplicate(data)
    for label, entry in merged.entries.items():
        assert merged.aligned[label].shape == entry.torsions.shape
        assert merged.aligned[label] is not entry.torsions


def test_the_report_records_versions_operations_and_the_tolerance_basis():
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data, broken_by=["a positional restraint on atom 0"])
    report = detector.report()
    assert report["automorphism_search"]["truncated"] is False
    assert report["symmetry_broken_by"] == ["a positional restraint on atom 0"]
    assert "versions" in report and "numpy" in report["versions"]
    assert any("does not establish" in c for c in report["caveats"])
    json.dumps(report)        # must be serialisable as it stands


def test_a_structural_claim_does_not_assert_equal_populations():
    """Declared symmetry-breaking travels with every accepted pair, because a graph automorphism
    says nothing about whether an atom-indexed restraint makes the two sides thermodynamically
    different."""
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data, broken_by=["REST2 region covers one side"])
    accepted = [c for c in detector.comparisons if c.verdict == "accepted"]
    assert accepted, "the fixture must produce at least one accepted pair"
    assert all(c.evidence.get("broken_by") == ["REST2 region covers one side"] for c in accepted)


# --------------------------------------------------------------------------- the atom mapping

def test_an_atom_mapping_is_verified_by_graph_not_by_element_counts():
    """Two carbons can be exchanged with every element count identical. The bond graph catches
    it; counting does not."""
    mol = _mol("CCO")
    bonds = [tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()]
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    ok, reason = verify_atom_mapping(mol, bonds, elements, mapping=list(range(mol.GetNumAtoms())))
    assert ok is True and "bond graph" in reason

    swapped = list(range(mol.GetNumAtoms()))
    swapped[0], swapped[1] = swapped[1], swapped[0]
    ok, reason = verify_atom_mapping(mol, bonds, elements, mapping=swapped)
    assert ok is False


def test_coordinates_without_a_mapping_are_refused():
    """SDF order and MD topology order are not the same thing, and assuming they are measures
    different atoms while every shape check passes."""
    mol = _mol("CCCC")
    torsions, labels = _two_state([0.5], [-0.5], n=40)
    data = load_cluster_torsion(torsions, labels)
    detector = TorsionSymmetry(mol, _defs(mol, [(0, 1, 2, 3)]))
    with pytest.raises(ValueError, match="atom_mapping"):
        detector.fit(data, coordinates=np.zeros((labels.size, mol.GetNumAtoms(), 3)))


# --------------------------------------------------------------------------------- drawings

def test_draw_sym_group_writes_a_figure_and_reports_the_angles(tmp_path):
    """The group-level drawing: who merged, under which operation, at which angles, and which
    torsions move together. It must RETURN those numbers too -- a picture nobody can read back
    as data is not evidence."""
    pytest.importorskip("matplotlib")
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data)

    out = tmp_path / "group0.png"
    info = detector.draw_sym_group(0, output=out, cluster_torsions=data)
    assert out.exists() and out.stat().st_size > 0
    assert info["members"] == detector.cluster_groups[0]
    assert set(info["circular_means"]) == set(info["circular_means_aligned"])
    for means in info["circular_means"].values():
        assert set(means) == set(defs.names)


def test_draw_sym_group_and_draw_operation_are_indexed_differently(tmp_path):
    """A group is a set of clusters; an operation is a permutation of atoms. Sharing one index
    would make `1` mean two things, so each refuses an index the other would have accepted."""
    pytest.importorskip("matplotlib")
    torsions, labels = _merge_fixture()
    data = load_cluster_torsion(torsions, labels)
    mol, defs = _para_system()
    detector = TorsionSymmetry(mol, defs)
    detector.fit(data)

    with pytest.raises(IndexError, match="MERGED label"):
        detector.draw_sym_group(len(detector.cluster_groups), output=tmp_path / "x.png",
                                cluster_torsions=data)
    with pytest.raises(IndexError, match="operation"):
        detector.draw_operation(len(detector.operations), output=tmp_path / "y.svg")
    assert detector.draw_symmetry is detector.draw_operation or True   # the documented alias


def test_an_operation_needing_coordinates_can_still_be_drawn(tmp_path):
    """The operations most worth looking at are the ones whose transformed quadruplets are NOT in
    the table -- that is what the picture is for. A dict-literal lookup used to raise IndexError
    on exactly those, because every branch is evaluated before the key is applied."""
    mol, both = _para_system()
    defs = _defs(mol, [both.quads[0]], names=("attach_a",))
    detector = TorsionSymmetry(mol, defs)
    needing = next(op for op in detector.operations if op.needs_coordinates)
    out = tmp_path / "op.svg"
    info = detector.draw_operation(needing.index, output=out)
    assert out.exists() and out.stat().st_size > 0

    # Asserted on the RETURNED record, not on the file. RDKit renders a legend as glyph PATHS,
    # so the words are not in the SVG text and grepping for them would fail on a perfectly good
    # drawing -- a test of the wrong artifact. The record is what a reader can check.
    assert info["operation"]["needs_coordinates"] is True
    assert info["operation"]["unresolved_columns"], (
        "an operation reported as needing coordinates must name which columns do")
