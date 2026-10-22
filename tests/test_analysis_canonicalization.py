"""Canonicalization: resonance-equivalent atoms are one atom in the symmetry enumeration.

A graph with localised bond orders and charges draws a carboxylate as `C=O` and `C-O-`, and RDKit's
self-match then finds no permutation that exchanges the two oxygens. With `canonicalize=True` the
charges and bond orders compared are the canonical form's, the resonance hybrid's. Each test names the mistake it exists
to catch: a delocalised group left asymmetric, an aromatic ring broken by averaging its Kekule
forms, a group of different shape merged with a similar one, and a torsion to one oxygen that is not
folded onto the torsion to the other.
"""
from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402
from rdkit.Chem import AllChem, rdMolTransforms                               # noqa: E402

from md_tools.analysis._symmetry_metric import (                              # noqa: E402
    IncompleteSymmetryEnumeration, SymmetricTorsionDescriptor, enumerate_symmetry, resonance_hybrid)
from md_tools.analysis._t_symmetry import TorsionDefinitions, wrap            # noqa: E402


def _h(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _orders(smiles):
    m = _h(smiles)
    return enumerate_symmetry(m, canonicalize=False).order, enumerate_symmetry(m).order


@pytest.mark.parametrize("smiles, drawn, hybrid", [
    ("CC(=O)[O-]", 6, 12),                  # carboxylate: the two O
    ("CNC(N)=[NH2+]", 24, 48),              # guanidinium: the two terminal N
    ("CC(N)=[NH2+]", 24, 48),               # amidinium
    ("C[N+](=O)[O-]", 6, 12),               # nitro
    ("[O-][N+](=O)[O-]", 2, 6),             # nitrate: three O, not two
    ("[O-]C(=O)[O-]", 2, 6),                # carbonate
    ("CS(=O)(=O)[O-]", 12, 36),             # sulfonate: by rule, RDKit does not generate it
    ("COP(=O)([O-])[O-]", 12, 36),          # phosphate monoester
    ("c1ccccc1C(=O)[O-]", 2, 4),            # benzoate: the ring flip times the carboxylate swap
])
def test_delocalised_groups_gain_the_equivalence_the_drawing_hides(smiles, drawn, hybrid):
    assert _orders(smiles) == (drawn, hybrid)


@pytest.mark.parametrize("smiles", [
    "CC(=O)O",                              # a carboxylic acid: C=O and C-OH ARE different
    "CC(N)=O",                              # an amide has one resonance structure that counts
    "NC(N)=O",                              # a urea
    "Oc1ccccc1",                            # an aromatic ring is already symmetric; averaging its
    "CC(=O)Nc1ccc(O)cc1",                   # Kekule forms must not break it
    "[O-]c1ccccc1",
    "CN=[N+]=[N-]",                         # an azide with different ends
])
def test_nothing_is_merged_that_chemistry_keeps_apart(smiles):
    drawn, hybrid = _orders(smiles)
    assert drawn == hybrid


def test_a_call_without_canonicalization_is_the_old_enumeration_and_says_so():
    m = _h("CC(=O)[O-]")
    assert enumerate_symmetry(m, canonicalize=False).describe()["canonicalization"] is None
    desc = enumerate_symmetry(m).describe()["canonicalization"]
    assert desc["n_resonance_structures"] == 2
    assert any(g["rule"].startswith("resonance") for g in desc["delocalised_groups"])


def test_the_hypervalent_rule_is_reported_as_a_rule():
    info = resonance_hybrid(_h("CS(=O)(=O)[O-]"))
    assert [g["rule"] for g in info.groups if "hypervalent" in g["rule"]] == ["hypervalent terminal atoms"]


def test_the_structure_limit_refuses_instead_of_being_used():
    with pytest.raises(IncompleteSymmetryEnumeration, match="resonance structures"):
        enumerate_symmetry(_h("CNC(N)=[NH2+]"), max_structures=3)


def _propionate():
    mol = _h("CCC(=O)[O-]")
    AllChem.EmbedMolecule(mol, randomSeed=3)
    # heavy atoms: 0 CH3, 1 CH2, 2 C, 3 O (=O), 4 O (-O-)
    return mol


def _frames(mol, angles):
    out = []
    for a in angles:
        conf = Chem.Conformer(mol.GetConformer())
        rdMolTransforms.SetDihedralRad(conf, 0, 1, 2, 3, float(a))
        out.append(conf.GetPositions())
    return np.asarray(out)


def test_a_torsion_to_one_carboxylate_oxygen_is_folded_onto_the_other():
    """C0-C1-C2-O3 and C0-C1-C2-O4 differ by half a turn in a planar carboxylate. With resonance the
    second is an image column of the first, the descriptor has two operations, and a frame and the
    same frame with the oxygens renamed are at distance zero. Without, nothing relates them."""
    mol = _propionate()
    defs = TorsionDefinitions(quads=((0, 1, 2, 3),), names=("chi",), indexing="zero-based",
                              units="radian")
    plain = SymmetricTorsionDescriptor(mol, defs, canonicalize=False)
    assert plain.is_trivial and plain.n_columns == 1

    d = SymmetricTorsionDescriptor(mol, defs)
    assert not d.is_trivial and d.n_columns == 2 and len(d.operations) == 2
    assert d.names[1] == "chi@0-1-2-4" and list(d.column_weights) == [0.5, 0.5]

    xyz = _frames(mol, [0.4, -2.0])
    swap = [i for i in range(mol.GetNumAtoms())]
    swap[3], swap[4] = 4, 3
    kw = dict(atom_mapping=list(range(mol.GetNumAtoms())), periodic=False)
    import types
    atoms = [types.SimpleNamespace(index=a.GetIdx(), element=types.SimpleNamespace(symbol=a.GetSymbol()))
             for a in mol.GetAtoms()]
    topology = types.SimpleNamespace(atoms=atoms, n_atoms=len(atoms),
                                     bonds=[(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
                                            for b in mol.GetBonds()])
    th = d.closed_angles(None, coordinates=xyz, topology=topology, **kw)
    th_swapped = d.closed_angles(None, coordinates=xyz[:, swap, :], topology=topology, **kw)
    assert np.allclose(wrap(th[:, 1] - th[:, 0] - np.pi), 0.0, atol=0.5)       # about half a turn
    dist = [d.distance(th[i], th_swapped[i])[0] for i in range(2)]
    assert dist == pytest.approx([0.0, 0.0], abs=1e-9)
    assert np.linalg.norm(d.embed(th) - d.embed(th_swapped)) > 1.0              # not the same row


def test_the_former_name_still_works_and_says_it_is_the_former_name():
    m = _h("CC(=O)[O-]")
    with pytest.warns(DeprecationWarning, match="canonicalize"):
        old = enumerate_symmetry(m, resonance=False).order
    assert old == enumerate_symmetry(m, canonicalize=False).order == 6
    with pytest.warns(DeprecationWarning):
        assert enumerate_symmetry(m, resonance=True).order == 12
