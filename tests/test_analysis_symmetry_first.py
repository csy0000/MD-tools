"""The symmetry-first clustering route: the vote rule, the quotient distance, and the contract.

Every test builds its own molecule and coordinates; nothing reads a trajectory or a project path.
Paracetamol is built from SMILES in exactly the atom order of the tutorial's SDF (heavy atoms
0-10, then hydrogens 11-19), so the indices here are the indices on the tutorial page.

Each test names the mistake it exists to catch, and the ones that matter most are shown able to
FAIL: a per-torsion fold that merges what must stay apart, an inverse applied where the operation
was meant, a margin read as a fraction.
"""
from __future__ import annotations

import importlib.util
import types

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402
from rdkit.Chem import AllChem, rdMolTransforms                               # noqa: E402

from md_tools.analysis._symmetry_metric import (                              # noqa: E402
    IncompleteSymmetryEnumeration, MissingCoordinatesError, SymmetricTorsionDescriptor,
    enumerate_symmetry, minimum_image, whole_molecule_coordinates)
from md_tools.analysis._t_symmetry import TorsionDefinitions, dihedral, wrap  # noqa: E402
from md_tools.analysis._torsions import (classify_to_clusters, embed,        # noqa: E402
                                         k_nearest_stable, vote)

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn (HDBSCAN) is an analysis-only dependency")

PARA_SMILES = "CC(=O)Nc1ccc(O)cc1"
PARA_TORSIONS = {"omega": (0, 1, 3, 4), "aryl": (1, 3, 4, 5), "phenol": (6, 7, 8, 17)}


# ----------------------------------------------------------------------------------- fixtures

def _mol(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _defs(quads, names=None):
    names = names or tuple(f"t{i}" for i in range(len(quads)))
    return TorsionDefinitions(quads=tuple(tuple(q) for q in quads), names=tuple(names),
                              indexing="zero-based", units="radian")


def _para():
    mol = _mol(PARA_SMILES)
    # the tutorial SDF's order, asserted rather than assumed
    assert [a.GetSymbol() for a in mol.GetAtoms()][:11] == list("CCONCCCCOCC")
    assert [n.GetIdx() for n in mol.GetAtomWithIdx(8).GetNeighbors()] == [7, 17]
    return mol, _defs(PARA_TORSIONS.values(), tuple(PARA_TORSIONS))


def _topology(mol):
    """Duck-typed mdtraj topology: what `check_atom_mapping` reads, and nothing else."""
    atoms = [types.SimpleNamespace(index=a.GetIdx(),
                                   element=types.SimpleNamespace(symbol=a.GetSymbol()))
             for a in mol.GetAtoms()]
    bonds = [(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    return types.SimpleNamespace(atoms=atoms, bonds=bonds, n_atoms=len(atoms))


def _para_frames(mol, settings, noise=0.01, seed=0):
    """One coordinate frame per (aryl, phenol) setting, in radians, with a little noise."""
    rng = np.random.default_rng(seed)
    out = []
    for aryl, phenol in settings:
        conf = Chem.Conformer(mol.GetConformer())
        rdMolTransforms.SetDihedralRad(conf, 1, 3, 4, 5, float(aryl))
        rdMolTransforms.SetDihedralRad(conf, 6, 7, 8, 17, float(phenol))
        out.append(conf.GetPositions())
    xyz = np.asarray(out)
    return xyz + rng.normal(0.0, noise, xyz.shape)


def _selected(xyz, quads):
    return np.column_stack([dihedral(*(xyz[:, i] for i in q)) for q in quads])


def _flip(enumeration):
    """The ring flip that fixes every methyl hydrogen: 5<->10, 6<->9."""
    for p in enumeration.permutations:
        if p[5] == 10 and p[6] == 9 and p[11:14] == (11, 12, 13):
            return p
    raise AssertionError("no ring flip found")


def _relabel(xyz, perm):
    """The SAME geometry under another naming: atom i of the result sits where atom perm[i] sat."""
    return xyz[:, list(perm), :]


def _closed(desc, xyz, mol):
    return desc.closed_angles(None, coordinates=xyz, atom_mapping=list(range(mol.GetNumAtoms())),
                              periodic=False, topology=_topology(mol))


# --------------------------------------------------------------------------------- the vote

def test_eighteen_of_twenty_assigns_and_seventeen_does_not():
    """The rule is a COUNT: n_winner / k >= 0.90 with equality accepted. 18/20 is exactly 0.9,
    which a strict `>` would refuse and float rounding could flip; 17/20 must never pass."""
    rows = np.array([[0] * 18 + [1] * 2, [0] * 17 + [1] * 3, [1] * 20, [0] * 10 + [1] * 10])
    v = vote(rows, 20, rule="fraction", threshold=0.90)
    assert v["labels"].tolist() == [0, -1, 1, -1]
    assert v["n_winner"].tolist() == [18, 17, 20, 10]
    assert v["vote_fraction"][0] == pytest.approx(0.9)


def test_a_reduced_denominator_is_used_and_reported():
    """With 10 eligible neighbours the denominator is 10, not 20: 9/10 assigns, 8/10 does not.
    Dividing by the nominal k would refuse every frame of a small training set."""
    rows = np.array([[0] * 9 + [1], [0] * 8 + [1] * 2])
    v = vote(rows, 10, rule="fraction", threshold=0.90)
    assert v["labels"].tolist() == [0, -1]
    assert v["k_effective"].tolist() == [10, 10]


def test_the_legacy_margin_is_a_different_statistic_which_is_why_it_is_not_reinterpreted():
    """18 of 20 against two votes is a fraction of 0.90 but a margin of 0.80: the same number
    means different things under the two rules, so a margin is refused, never converted."""
    rows = np.array([[0] * 18 + [1] * 2])
    assert vote(rows, 20, rule="fraction", threshold=0.9)["labels"][0] == 0
    assert vote(rows, 20, rule="legacy-margin", threshold=0.9)["labels"][0] == -1


def _line_fixture(n_a, n_b):
    """Query at 0; n_a class-0 frames then n_b class-1 frames at increasing distance, then far
    class-0 frames so the nearest 20 are exactly the first n_a + n_b."""
    near = np.concatenate([np.linspace(0.01, 0.18, n_a), np.linspace(0.19, 0.25, n_b)])
    far = np.full(30, 2.0)
    theta = np.concatenate([near, far])[:, None]
    labels = np.array([0] * n_a + [1] * n_b + [0] * 30)
    return theta, labels


def test_the_classifier_applies_the_count_end_to_end():
    theta, labels = _line_fixture(18, 2)
    out = classify_to_clusters(theta, labels, np.zeros((1, 1)))
    assert out["labels"][0] == 0 and out["n_winner"][0] == 18 and out["k_effective"][0] == 20
    theta, labels = _line_fixture(17, 3)
    out = classify_to_clusters(theta, labels, np.zeros((1, 1)))
    assert out["labels"][0] == -1 and out["n_winner"][0] == 17


def test_k_effective_falls_to_the_eligible_count_and_excludes_the_frame_itself():
    """Eleven training frames, each classified with ITSELF excluded: ten votes. A class-0 frame
    sees nine class-0 neighbours and one class-1: 9/10 = 0.90, assigned. With a second class-1
    frame it sees 8/10 and is not."""
    theta = np.concatenate([np.linspace(-0.05, 0.05, 10), [1.0]])[:, None]
    out = classify_to_clusters(theta, np.array([0] * 10 + [1]), theta, query_is_training=True)
    assert set(out["k_effective"].tolist()) == {10}
    assert out["labels"].tolist() == [0] * 11
    assert out["n_winner"][0] == 9 and out["vote_fraction"][0] == pytest.approx(0.9)
    theta = np.concatenate([np.linspace(-0.05, 0.05, 10), [1.0, 1.1]])[:, None]
    out = classify_to_clusters(theta, np.array([0] * 10 + [1, 1]), theta, query_is_training=True)
    assert set(out["k_effective"].tolist()) == {11}
    theta3 = np.concatenate([np.linspace(-0.05, 0.05, 9), [1.0, 1.1]])[:, None]
    out = classify_to_clusters(theta3, np.array([0] * 9 + [1, 1]), theta3, query_is_training=True)
    assert out["k_effective"][0] == 10 and out["n_winner"][0] == 8 and out["labels"][0] == -1


def test_neighbour_ties_are_broken_by_index_not_by_the_implementation():
    D = np.array([[0.1, 0.3, 0.3, 0.3, 0.2]])
    assert sorted(k_nearest_stable(D, 3)[0].tolist()) == [0, 1, 4]


def test_an_explicit_legacy_margin_is_refused_without_the_legacy_rule():
    theta, labels = _line_fixture(18, 2)
    with pytest.raises(TypeError, match="legacy-margin"):
        classify_to_clusters(theta, labels, np.zeros((1, 1)), min_vote_margin=0.9)
    out = classify_to_clusters(theta, labels, np.zeros((1, 1)), vote_rule="legacy-margin",
                               min_vote_margin=0.9)
    assert out["k"] == 15 and out["vote_rule"] == "legacy-margin"


def _historical_classify(theta_train, labels_train, theta_query, k=15, margin=0.9):
    """The pre-change classifier, transcribed: brute force, own label counted, margin gate."""
    Xtr, Xq = embed(theta_train, np.ones(theta_train.shape[1])), embed(theta_query,
                                                                      np.ones(theta_query.shape[1]))
    keep = labels_train >= 0
    Xtr, lab = Xtr[keep], labels_train[keep]
    kk = min(k, Xtr.shape[0])
    D2 = np.maximum(2.0 * theta_train.shape[1] - 2.0 * (Xq @ Xtr.T), 0.0)
    nn = np.argsort(D2, axis=1, kind="stable")[:, :kk]
    nl = lab[nn]
    classes = np.unique(lab)
    cnt = np.stack([(nl == c).sum(axis=1) for c in classes], axis=1)
    out = classes[np.argmax(cnt, axis=1)]
    srt = np.sort(cnt, axis=1)
    m = (srt[:, -1] - (srt[:, -2] if srt.shape[1] > 1 else 0)) / kk
    out = out.copy()
    out[m < margin] = -1
    return out


def test_the_legacy_rule_reproduces_the_historical_classifier():
    rng = np.random.default_rng(3)
    train = wrap(np.vstack([rng.normal(0, 0.4, (300, 2)), rng.normal(2.0, 0.4, (300, 2))]))
    labels = np.array([0] * 300 + [1] * 300)
    query = rng.uniform(-np.pi, np.pi, (500, 2))
    got = classify_to_clusters(train, labels, query, vote_rule="legacy-margin", use_tree=False)
    assert np.array_equal(got["labels"], _historical_classify(train, labels, query))


# -------------------------------------------------------------------------- the enumeration

def test_enumeration_is_whole_molecule_and_includes_identity_first():
    mol, _ = _para()
    e = enumerate_symmetry(mol)
    assert e.order == 12                                     # 3! methyl H x ring flip
    assert e.permutations[0] == tuple(range(mol.GetNumAtoms()))
    # NOT restricted to rings: the methyl-hydrogen permutations are there
    assert any(p[11:14] != (11, 12, 13) and p[5] == 5 for p in e.permutations)


def test_an_enumeration_of_exactly_the_cap_is_complete_and_one_over_is_refused():
    """The old test `len >= cap` called a complete search of exactly `cap` truncated. The matcher
    is now asked for cap + 1."""
    mol, defs = _para()
    assert enumerate_symmetry(mol, max_matches=12).order == 12
    with pytest.raises(IncompleteSymmetryEnumeration, match="INCOMPLETE"):
        enumerate_symmetry(mol, max_matches=11)
    with pytest.raises(IncompleteSymmetryEnumeration):
        SymmetricTorsionDescriptor(mol, defs, max_matches=4)


def test_atom_map_numbers_do_not_hide_a_symmetry():
    mol, _ = _para()
    mapped = Chem.Mol(mol)
    mapped.GetAtomWithIdx(5).SetAtomMapNum(1)
    assert enumerate_symmetry(mapped).order == 12


def test_specified_stereochemistry_is_preserved():
    """meso-tartaric acid (R,S): the end-swap is a graph automorphism but maps R onto S, so with
    specified stereochemistry only the identity survives. A diene with one E and one Z double
    bond likewise loses the end-swap. Without chirality both swaps are admitted."""
    from md_tools.analysis._symmetry_metric import _stereo_labels, _stereo_preserved

    meso = Chem.AddHs(Chem.MolFromSmiles("O=C(O)[C@H](O)[C@H](O)C(=O)O"))
    assert enumerate_symmetry(meso, use_chirality=True).order == 1
    assert enumerate_symmetry(meso, use_chirality=False).order == 2
    diene = Chem.AddHs(Chem.MolFromSmiles("Cl/C=C/CC/C=C\\Cl"))
    assert enumerate_symmetry(diene).order < enumerate_symmetry(diene, use_chirality=False).order
    # the CIP post-check refuses the swap on its own, independently of RDKit's matcher
    swap = next(p for p in enumerate_symmetry(meso, use_chirality=False).permutations
                if p[3] == 5)
    assert not _stereo_preserved(swap, _stereo_labels(meso))
    assert _stereo_preserved(tuple(range(meso.GetNumAtoms())), _stereo_labels(meso))


# ------------------------------------------------------------------------- the descriptor

def test_paracetamol_closure_adds_exactly_the_two_images_and_halves_their_weight():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    assert [d.quads[c] for c in d.added_columns] == [(1, 3, 4, 10), (9, 7, 8, 17)]
    assert d.column_weights.tolist() == [1.0, 0.5, 0.5, 0.5, 0.5]
    assert len(d.operations) == 2                         # 12 permutations, two distinct actions
    assert d.operations[0].n_atom_permutations == 6       # the methyl H perms collapse
    assert d.operations[1].sigma == (0, 3, 4, 1, 2)    # omega fixed; aryl<->aryl@, phenol<->phenol@
    assert d.isometry_verified and d.group_closure_verified


def test_unequal_weights_on_symmetry_equivalent_selections_are_refused():
    mol, _ = _para()
    defs = _defs([(1, 3, 4, 5), (1, 3, 4, 10)], ("a", "b"))
    with pytest.raises(ValueError, match="different metric weights"):
        SymmetricTorsionDescriptor(mol, defs, metric_weights=[1.0, 2.0])


def test_relabelling_is_a_column_permutation_measured_from_coordinates():
    """theta(p x) == theta(x)[:, sigma_p], with p x computed by RELABELLING the coordinates and
    every column recomputed from them. This is the identity the whole distance rests on."""
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _para_frames(mol, [(0.7, 0.1), (-2.0, 2.9), (2.5, -1.2)])
    flip = _flip(d.enumeration)
    th = _closed(d, xyz, mol)
    th_flipped = _closed(d, _relabel(xyz, flip), mol)
    assert np.allclose(wrap(th_flipped - d.transform(th, 1)), 0.0, atol=1e-12)
    # and the transformed aryl is NOT the stored aryl + pi: the outer atom really moved
    shift = wrap(th_flipped[:, 1] - th[:, 1] - np.pi)
    assert np.max(np.abs(shift)) > 1e-3


def test_a_three_cycle_is_applied_in_the_right_direction():
    """A non-self-inverse operation catches an inverse applied in place of the operation: for a
    two-fold flip the two are the same and no test of paracetamol can tell them apart."""
    mol = _mol("CCc1cc(CC)cc(CC)c1", seed=11)
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    quads = []
    for a in mol.GetAtoms():
        if a.GetIsAromatic() and any(n.GetSymbol() == "C" and not n.GetIsAromatic()
                                     for n in a.GetNeighbors()):
            ch2 = next(n for n in a.GetNeighbors() if not n.GetIsAromatic())
            ch3 = next(n.GetIdx() for n in ch2.GetNeighbors()
                       if n.GetSymbol() == "C" and n.GetIdx() != a.GetIdx())
            ortho = min(n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in ring)
            quads.append((ortho, a.GetIdx(), ch2.GetIdx(), ch3))
    d = SymmetricTorsionDescriptor(mol, _defs(quads))
    order3 = []
    for op in d.operations:
        s = np.array(op.sigma)
        if not op.is_identity and np.array_equal(s[s[s]], np.arange(s.size)):
            order3.append(op)
    assert order3, "the C3 rotation must act as a three-cycle on the descriptor"
    op = order3[0]
    inv = d.inverse(op.index)
    assert inv != op.index
    rng = np.random.default_rng(5)
    xyz = mol.GetConformer().GetPositions()[None] + rng.normal(0, 0.15, (4, mol.GetNumAtoms(), 3))
    th = _closed(d, xyz, mol)
    th_moved = _closed(d, _relabel(xyz, op.atom_permutation), mol)
    assert np.allclose(wrap(th_moved - d.transform(th, op.index)), 0.0, atol=1e-12)
    assert not np.allclose(wrap(th_moved - d.transform(th, inv)), 0.0, atol=1e-3)
    # composition agrees with relabelling twice
    other = order3[-1] if len(order3) > 1 else op
    th_twice = _closed(d, _relabel(_relabel(xyz, op.atom_permutation), other.atom_permutation), mol)
    assert np.allclose(wrap(th_twice - d.transform(th, d.compose(op.index, other.index))), 0.0,
                       atol=1e-12)


# ---------------------------------------------------------------------------- the distance

def _random_closed(d, n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (n, d.n_columns))


@pytest.mark.parametrize("smiles", [PARA_SMILES, "CCc1cc(CC)cc(CC)c1"])
def test_the_one_sided_minimum_equals_exhaustive_two_sided_enumeration(smiles):
    mol = _mol(smiles)
    if smiles == PARA_SMILES:
        defs = _defs(PARA_TORSIONS.values())
    else:
        defs = _defs([(3, 2, 1, 0), (3, 4, 5, 6)])        # two of the three ethyls
    d = SymmetricTorsionDescriptor(mol, defs)
    th = _random_closed(d, 40, seed=1)
    D = d.distance_matrix(d.embed(th[:20]), d.embed(th[20:]))
    for i in range(20):
        for j in range(20):
            exhaustive, (g, h) = d.exhaustive_distance(th[i], th[20 + j])
            assert D[i, j] == pytest.approx(exhaustive, abs=1e-9)
            direct = np.linalg.norm(d.embed(d.transform(th[i:i + 1], g))
                                    - d.embed(d.transform(th[20 + j:21 + j], h)))
            assert direct == pytest.approx(exhaustive, abs=1e-12)


def test_symmetry_related_configurations_are_at_zero_distance():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _para_frames(mol, [(0.6, 0.2), (-1.0, 3.0)])
    th = _closed(d, xyz, mol)
    th_rel = _closed(d, _relabel(xyz, _flip(d.enumeration)), mol)
    for i in range(2):
        dist, op = d.distance(th[i], th_rel[i])
        assert dist == pytest.approx(0.0, abs=1e-9) and op == 1
        # the plain Euclidean distance between the two descriptions is large
        assert np.linalg.norm(d.embed(th[i:i + 1]) - d.embed(th_rel[i:i + 1])) > 1.0


def test_the_distance_is_symmetric_and_invariant_to_relabelling_either_frame():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    flip = _flip(d.enumeration)
    rng = np.random.default_rng(2)
    xyz = _para_frames(mol, rng.uniform(-np.pi, np.pi, (12, 2)), noise=0.02)
    th, th_rel = _closed(d, xyz, mol), _closed(d, _relabel(xyz, flip), mol)
    X, Xr = d.embed(th), d.embed(th_rel)
    D = d.distance_matrix(X)
    assert np.array_equal(D, D.T) and np.all(np.diag(D) == 0.0)
    assert np.allclose(d.distance_matrix(Xr, X), D, atol=1e-7)      # relabel the first
    assert np.allclose(d.distance_matrix(X, Xr), D, atol=1e-7)      # relabel the second
    assert np.allclose(d.distance_matrix(X, X[:5]), D[:, :5], atol=1e-7)


def test_the_periodic_seam_costs_nothing():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    a = np.zeros(d.n_columns)
    b = np.zeros(d.n_columns)
    a[0], b[0] = np.pi - 0.01, -np.pi + 0.01
    dist, _ = d.distance(a, b)
    assert dist == pytest.approx(2 * np.sin(0.01), rel=1e-9)          # chord, weight 1


def test_minimising_operation_ties_go_to_the_earlier_operation():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    th = np.zeros(d.n_columns)                     # every column equal: both ops tie exactly
    assert d.distance(th, th) == (0.0, 0)
    assert d.exhaustive_distance(th, th)[1] == (0, 0)


def test_coupled_torsions_are_not_folded_independently():
    """Rotating ONLY the phenol by pi gives a different relative orientation of the two
    substituents. Folding each torsion by its own two-fold period calls the two the same; the
    symmetry-aware distance does not -- and it does merge the genuine relabelling."""
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    a = _para_frames(mol, [(0.6, 0.2)], noise=0.0)
    b = _para_frames(mol, [(0.6, 0.2 + np.pi)], noise=0.0)       # one substituent rotated
    c = _relabel(a, _flip(d.enumeration))                        # the same geometry, relabelled
    ta, tb, tc = (_closed(d, x, mol)[0] for x in (a, b, c))
    folded = embed(np.vstack([ta[:3], tb[:3]]), np.ones(3), [1, 2, 2])
    assert np.linalg.norm(folded[0] - folded[1]) < 1e-6             # the fold merges them
    assert d.distance(ta, tb)[0] > 0.9                              # the quotient does not
    assert d.distance(ta, tc)[0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------- coordinates and mapping

def test_missing_coordinates_name_the_quadruplets_to_recompute():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    with pytest.raises(MissingCoordinatesError, match=r"\[1, 3, 4, 10\]"):
        d.closed_angles(np.zeros((3, 3)))


def test_coordinates_need_an_explicit_validated_mapping_and_a_periodicity_statement():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _para_frames(mol, [(0.6, 0.2)])
    with pytest.raises(ValueError, match="no atom_mapping"):
        d.closed_angles(None, coordinates=xyz, periodic=False, topology=_topology(mol))
    with pytest.raises(ValueError, match="periodic=False"):
        d.closed_angles(None, coordinates=xyz, atom_mapping=list(range(20)),
                        topology=_topology(mol))
    swapped = list(range(20))
    swapped[5], swapped[6] = 6, 5                      # same elements, a moved bond
    with pytest.raises(ValueError, match="bond graph"):
        d.closed_angles(None, coordinates=xyz, atom_mapping=swapped, periodic=False,
                        topology=_topology(mol))


def test_a_graph_valid_mapping_that_disagrees_with_the_stored_table_is_refused():
    """The ring flip IS a graph automorphism, so the graph check passes it -- but the stored aryl
    column was measured to atom 5, and under the flipped mapping the same quadruplet is measured
    to atom 10. The geometric check catches what the graph cannot."""
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _para_frames(mol, [(0.6, 0.2), (-1.4, 2.0)])
    stored = _selected(xyz, PARA_TORSIONS.values())
    ok = d.closed_angles(stored, coordinates=xyz, atom_mapping=list(range(20)), periodic=False,
                         topology=_topology(mol))
    assert np.allclose(ok[:, :3], stored)
    with pytest.raises(ValueError, match="disagrees with the coordinates"):
        d.closed_angles(stored, coordinates=xyz, atom_mapping=list(_flip(d.enumeration)),
                        periodic=False, topology=_topology(mol))


def test_a_molecule_split_across_the_box_is_made_whole_first():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _para_frames(mol, [(0.6, 0.2), (-1.4, 2.0)])
    box = np.diag([3.0, 3.1, 3.2]) * 10.0
    broken = xyz.copy()
    broken[:, [5, 6, 15, 16], :] += box[0]                 # half the ring one box over
    broken[:, [9, 18], :] -= box[2]
    whole = d.closed_angles(None, coordinates=xyz, atom_mapping=list(range(20)), periodic=False,
                            topology=_topology(mol))
    fixed = d.closed_angles(None, coordinates=broken, atom_mapping=list(range(20)),
                            box_vectors=box, topology=_topology(mol))
    assert np.allclose(wrap(fixed - whole), 0.0, atol=1e-9)
    naive = d.closed_angles(None, coordinates=broken, atom_mapping=list(range(20)),
                            periodic=False, topology=_topology(mol))
    assert not np.allclose(wrap(naive - whole), 0.0, atol=1e-3)


def test_minimum_image_handles_a_reduced_triclinic_box():
    box = np.array([[[3.0, 0.0, 0.0], [1.0, 2.8, 0.0], [-1.2, 0.9, 2.5]]])
    true = np.array([[0.11, -0.07, 0.05]])
    shifted = true + box[0, 0] - 2 * box[0, 1] + box[0, 2]
    assert np.allclose(minimum_image(shifted, box), true)
    xyz = np.array([[[0.0, 0, 0], [0.15, 0, 0], [0.15, 0.15, 0]]])
    xyz[0, 2] += box[0, 2]
    whole = whole_molecule_coordinates(xyz, [(0, 1), (1, 2)], box)
    assert np.allclose(whole[0, 2], [0.15, 0.15, 0.0])


# ------------------------------------------------------------------------ the clustering route

def _para_ensemble(mol, per=150, seed=0, relabel_half=False):
    """Four basins: aryl in {0.6, 0.6 + pi} x phenol in {0.2, 0.2 + pi}."""
    rng = np.random.default_rng(seed)
    settings, truth = [], []
    for b, (a, p) in enumerate([(0.6, 0.2), (0.6 + np.pi, 0.2 + np.pi),
                                (0.6, 0.2 + np.pi), (0.6 + np.pi, 0.2)]):
        settings.append(np.column_stack([a + rng.normal(0, 0.12, per),
                                         p + rng.normal(0, 0.12, per)]))
        truth += [b] * per
    xyz = _para_frames(mol, np.vstack(settings), noise=0.01, seed=seed)
    return xyz, np.array(truth)


def _fit(mol, defs, xyz, **kw):
    from md_tools.analysis import t_hdbscan

    stored = _selected(xyz, PARA_TORSIONS.values())
    return t_hdbscan(stored, molecule=mol, torsion_definitions=defs, coordinates=xyz,
                     atom_mapping=list(range(20)), periodic=False, topology=_topology(mol),
                     **kw).fit()


@requires_sklearn
def test_symmetry_related_basins_share_a_cluster_during_the_fit():
    """Basins 0 and 1 are one conformer in two labellings, as are 2 and 3. The default route fits
    two clusters; the legacy distance fits four. Nothing is merged after the fit."""
    from md_tools.analysis import t_hdbscan

    mol, defs = _para()
    xyz, truth = _para_ensemble(mol)
    fit = _fit(mol, defs, xyz, mass_floor=0.05)
    assert fit.n_clusters_ == 2
    for a, b in ((0, 1), (2, 3)):
        la = np.unique(fit.labels_[(truth == a) & (fit.labels_ >= 0)])
        lb = np.unique(fit.labels_[(truth == b) & (fit.labels_ >= 0)])
        assert la.size == 1 and np.array_equal(la, lb)
    legacy = t_hdbscan(_selected(xyz, PARA_TORSIONS.values()), symmetry=False,
                       mass_floor=0.05).fit()
    assert legacy.n_clusters_ == 4


@requires_sklearn
def test_frames_weights_and_population_are_conserved():
    """Enumerations are descriptions, not observations: nothing is appended, every frame has
    one label, the weights are the caller's, and clusters + noise + ambiguous == 1."""
    mol, defs = _para()
    xyz, _ = _para_ensemble(mol, per=100)
    w = np.random.default_rng(4).uniform(0.2, 2.0, xyz.shape[0])
    fit = _fit(mol, defs, xyz, weights=w, mass_floor=0.05)
    s = fit.summary()
    assert fit.labels_.shape == (xyz.shape[0],) and s["n_fit_frames"] == xyz.shape[0]
    assert fit.X_.shape[0] == xyz.shape[0]
    assert np.array_equal(fit.weights_, w)
    total = sum(s["cluster_mass"].values()) + s["noise_mass"] + s["ambiguous_mass"]
    assert total == pytest.approx(1.0, abs=1e-12)
    for c, m in s["cluster_mass"].items():
        assert m == pytest.approx(w[fit.labels_ == c].sum() / w.sum())


@requires_sklearn
def test_one_distance_is_used_for_fit_density_and_vote(monkeypatch):
    """No Euclidean tree anywhere on the default route, HDBSCAN on the precomputed quotient, and
    a relabelled frame gets the label AND the density of the frame it relabels."""
    import sklearn.cluster
    import sklearn.neighbors

    seen = {}
    real = sklearn.cluster.HDBSCAN

    class Spy(real):
        def fit(self, X, y=None):
            seen["metric"] = self.metric
            seen["shape"] = np.shape(X)
            return super().fit(X, y)

    class NoTree:
        def __init__(self, *a, **k):
            raise AssertionError("a Euclidean neighbour tree was built on the symmetry route")

    monkeypatch.setattr(sklearn.cluster, "HDBSCAN", Spy)
    monkeypatch.setattr(sklearn.neighbors, "NearestNeighbors", NoTree)
    mol, defs = _para()
    xyz, _ = _para_ensemble(mol)
    fit = _fit(mol, defs, xyz, mass_floor=0.05)
    assert seen["metric"] == "precomputed" and seen["shape"] == (xyz.shape[0],) * 2

    probe = _para_frames(mol, [(0.65, 0.25), (0.6 + np.pi, 0.1), (2.2, -1.6)], seed=9)
    relabelled = _relabel(probe, _flip(fit.descriptor_.enumeration))
    l1, _ = fit.predict(_selected(probe, PARA_TORSIONS.values()), coordinates=probe)
    l2, _ = fit.predict(_selected(relabelled, PARA_TORSIONS.values()), coordinates=relabelled)
    assert np.array_equal(l1, l2)
    # the third probe sits in an empty region: the density check, on the same distance, says so
    from md_tools.analysis import NOISE_LABEL
    assert l1[2] == NOISE_LABEL and l1[0] >= 0
    th1 = fit.descriptor_.embed(_closed(fit.descriptor_, probe, mol))
    th2 = fit.descriptor_.embed(_closed(fit.descriptor_, relabelled, mol))
    none = np.full(3, -1)
    c1, c2 = fit._neighbour_pass(th1, none)["core"], fit._neighbour_pass(th2, none)["core"]
    assert np.allclose(c1, c2, atol=1e-7)


@requires_sklearn
def test_the_memory_guard_refuses_and_names_the_subset_route():
    from md_tools.analysis import PrecomputedMemoryGuard

    mol, defs = _para()
    xyz, _ = _para_ensemble(mol, per=60)
    with pytest.raises(PrecomputedMemoryGuard, match="resampling=N"):
        _fit(mol, defs, xyz, mass_floor=0.05, max_precomputed_frames=100)
    fit = _fit(mol, defs, xyz, mass_floor=0.05, max_precomputed_frames=200, resampling=200,
               seed=2)
    s = fit.summary()
    assert s["n_fit_frames"] == 200 and s["resampling"] == 200 and fit.labels_.size == 240


@requires_sklearn
def test_angles_alone_are_refused_with_the_migration():
    from md_tools.analysis import t_hdbscan

    with pytest.raises(ValueError, match="symmetry=False"):
        t_hdbscan(np.zeros((50, 2)))
    with pytest.raises(TypeError, match="legacy-margin"):
        t_hdbscan(np.zeros((50, 2)), symmetry=False, min_vote=0.9)
    mol, defs = _para()
    xyz, _ = _para_ensemble(mol, per=20)
    with pytest.raises(MissingCoordinatesError):
        t_hdbscan(_selected(xyz, PARA_TORSIONS.values()), molecule=mol, torsion_definitions=defs)
    with pytest.raises(ValueError, match="independently|INDEPENDENTLY"):
        _fit(mol, defs, xyz, multiplicities=[1, 2, 2])


@requires_sklearn
def test_the_summary_records_the_symmetry_the_vote_and_the_unassigned_split():
    mol, defs = _para()
    xyz, _ = _para_ensemble(mol)
    s = _fit(mol, defs, xyz, mass_floor=0.05, broken_by=["a restraint on C5"]).summary()
    assert s["route"] == "symmetry-first"
    assert s["vote"]["rule"] == "fraction" and s["vote"]["k"] == 20
    assert s["vote"]["min_vote_fraction"] == 0.9
    assert s["symmetry"]["added_columns"] == ["aryl@1-3-4-10", "phenol@9-7-8-17"]
    assert s["symmetry"]["enumeration"]["n_automorphisms"] == 12
    assert s["symmetry_broken_by"] == ["a restraint on C5"]
    assert {"noise_mass", "ambiguous_mass", "unassigned_mass", "n_fit_frames"} <= set(s)
    assert any("NOT evidence of a free-energy barrier" in c for c in s["caveats"])


@requires_sklearn
def test_representatives_are_frames_of_their_cluster_and_alignment_moves_no_atom():
    mol, defs = _para()
    xyz, _ = _para_ensemble(mol)
    fit = _fit(mol, defs, xyz, mass_floor=0.05)
    reps = fit.representatives()
    for c, i in reps.items():
        assert fit.labels_[i] == c
        assert fit.vote_fraction_[i] == fit.vote_fraction_[fit.labels_ == c].max()
        al = fit.aligned_angles(c)
        # alignment only re-chooses the labelling: every aligned row is the frame's own closed
        # angles under the reported operation
        for row, f, op in zip(al["closed_angles"][:20], al["frames"][:20], al["operations"][:20]):
            assert np.allclose(wrap(row - fit.descriptor_.transform(
                fit.closed_theta_[f:f + 1], int(op))[0]), 0.0)


@requires_sklearn
def test_the_legacy_fit_uses_the_same_multiplicities_as_its_classifier():
    """Before, the fit ignored `multiplicities` while the vote and density used them: two basins
    pi apart were fitted as two clusters and then classified on a metric that called them one."""
    from md_tools.analysis import t_hdbscan

    rng = np.random.default_rng(8)
    th = wrap(np.concatenate([rng.normal(0.4, 0.1, 600), rng.normal(0.4 + np.pi, 0.1, 600)]))
    fit = t_hdbscan(th[:, None], symmetry=False, multiplicities=[2],
                    multiplicity_justification="test: a two-fold rotor").fit()
    assert fit.n_clusters_ == 1


@requires_sklearn
def test_training_frames_are_judged_leave_one_out_on_the_same_distance():
    """A training frame is its own nearest neighbour at distance zero. Counting it would shrink
    every training core distance by one rank and give every frame a free vote for its own label;
    both are computed here from an explicit quotient matrix with the diagonal excluded."""
    mol, defs = _para()
    xyz, _ = _para_ensemble(mol, per=60)
    fit = _fit(mol, defs, xyz, mass_floor=0.05)
    D = fit.descriptor_.distance_matrix(fit.X_)
    np.fill_diagonal(D, np.inf)
    ms = fit._core_k_
    assert np.allclose(fit.core_distance_, np.partition(D, ms - 1, axis=1)[:, ms - 1], atol=1e-9)
    clustered = np.flatnonzero(fit.train_labels_ >= 0)
    nn = k_nearest_stable(D[:, clustered], 20)
    winners = fit.train_labels_[clustered][nn]
    expected = np.array([np.bincount(r, minlength=fit.n_clusters_).max() for r in winners])
    assert np.array_equal(fit.n_winner_, expected)


def test_representative_drawing_rotates_properly_and_never_mirrors(tmp_path):
    """The drawing orients the superposed structures along their principal axes so nothing is
    clipped. That rotation must be PROPER: a reflection would draw the enantiomer of a chiral
    molecule, and the figure is what a reader checks a conformation against."""
    md = pytest.importorskip("mdtraj")
    pytest.importorskip("matplotlib")
    from md_tools.analysis import draw_representative_structures

    mol = _mol("C[C@H](N)C(=O)O", seed=3)                      # L-alanine: chiral
    xyz = np.stack([mol.GetConformer().GetPositions() / 10.0] * 2)          # nm
    xyz[1] += np.random.default_rng(1).normal(0, 0.005, xyz[1].shape)
    top = md.Topology()
    residue = top.add_residue("ALA", top.add_chain())
    atoms = [top.add_atom(a.GetSymbol() + str(a.GetIdx()), md.element.get_by_symbol(a.GetSymbol()),
                          residue) for a in mol.GetAtoms()]
    for b in mol.GetBonds():
        top.add_bond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    info = draw_representative_structures(md.Trajectory(xyz, top), {"A": 0, "B": 1},
                                          output=tmp_path / "reps.png")
    before, after = info["aligned_xyz_nm"], info["drawn_xyz_nm"]
    for frame in range(2):
        d0 = np.linalg.norm(before[frame][:, None] - before[frame][None], axis=-1)
        d1 = np.linalg.norm(after[frame][:, None] - after[frame][None], axis=-1)
        assert np.allclose(d0, d1, atol=1e-5)
        # the signed volume at the stereocentre (atom 1) keeps its sign: no mirror
        c, n = 1, [0, 2, 3]
        v0 = np.linalg.det(np.stack([before[frame][k] - before[frame][c] for k in n]))
        v1 = np.linalg.det(np.stack([after[frame][k] - after[frame][c] for k in n]))
        assert np.sign(v0) == np.sign(v1) != 0
    assert (tmp_path / "reps.png").stat().st_size > 0
