"""An isopropyl end (Val CB with CG1/CG2, Leu CG with CD1/CD2): the two methyl labels are one atom.

A torsion measured to CD1 jumps by about 120 degrees, not 180, when the two methyls are named the
other way round, so a per-torsion fold of period 2 is wrong, and so is any fold of one angle. The
symmetry-first route exchanges the labels as one operation on the whole row and takes the minimum
over them. The torsion measured to the unique hydrogen (HG) is label-free, which is what the truth
below is built from.

Every test builds its own molecule and coordinates; nothing reads a trajectory or a project path.
"""
from __future__ import annotations

import importlib.util
import types

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402
from rdkit.Chem import AllChem, rdMolTransforms                               # noqa: E402

from md_tools.analysis._symmetry_metric import SymmetricTorsionDescriptor, enumerate_symmetry  # noqa: E402
from md_tools.analysis._t_symmetry import TorsionDefinitions, dihedral, wrap  # noqa: E402

requires_sklearn = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,
                                      reason="sklearn (HDBSCAN) is an analysis-only dependency")

# Ac-Leu-NMe: 0 CH3, 1 C, 2 O, 3 N, 4 CA, 5 CB, 6 CG, 7 CD1, 8 CD2, 9 C', 10 O, 11 N, 12 CH3
LEU = "CC(=O)N[C@@H](CC(C)C)C(=O)NC"
LEU_TORSIONS = {"phi": (1, 3, 4, 9), "psi": (3, 4, 9, 11), "chi1": (3, 4, 5, 6), "chi2": (4, 5, 6, 7)}


def _leu():
    mol = Chem.AddHs(Chem.MolFromSmiles(LEU))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    hg = [n.GetIdx() for n in mol.GetAtomWithIdx(6).GetNeighbors() if n.GetAtomicNum() == 1][0]
    defs = TorsionDefinitions(quads=tuple(LEU_TORSIONS.values()), names=tuple(LEU_TORSIONS),
                              indexing="zero-based", units="radian")
    return mol, defs, hg


def _topology(mol):
    atoms = [types.SimpleNamespace(index=a.GetIdx(), element=types.SimpleNamespace(symbol=a.GetSymbol()))
             for a in mol.GetAtoms()]
    return types.SimpleNamespace(atoms=atoms, n_atoms=len(atoms), bonds=[
        (atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()]) for b in mol.GetBonds()])


def _methyl_swap(mol):
    """The permutation that renames CD1 as CD2 (and their hydrogens) and moves nothing else."""
    enum = enumerate_symmetry(mol)
    movable = {7, 8} | {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1
                        and a.GetNeighbors()[0].GetIdx() in (7, 8)}
    for p in enum.permutations:
        if p[7] == 8 and p[8] == 7 and all(p[k] == k for k in range(mol.GetNumAtoms()) if k not in movable):
            return p
    raise AssertionError("no CD1/CD2 exchange found")


def test_the_two_methyls_are_one_operation_and_the_pair_is_not_half_a_turn_apart():
    mol, defs, _ = _leu()
    d = SymmetricTorsionDescriptor(mol, defs)
    assert len(d.operations) == 2 and d.names[-1] == "chi2@4-5-6-8"
    assert list(d.column_weights[-2:]) == [0.5, 0.5]
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralRad(conf, 4, 5, 6, 7, 1.0)
    spacing = abs(float(np.rad2deg(wrap(rdMolTransforms.GetDihedralRad(conf, 4, 5, 6, 8) - 1.0))))
    assert 110 < spacing < 130            # a two-fold fold (half a turn) would be wrong


def test_val_has_the_same_operation():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N[C@@H](C(C)C)C(=O)NC"))        # Ac-Val-NMe
    AllChem.EmbedMolecule(mol, randomSeed=7)
    cb = 5
    cg = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(cb).GetNeighbors() if n.GetAtomicNum() == 6
                and n.GetIdx() != 4)
    defs = TorsionDefinitions(quads=((3, 4, cb, cg[0]),), names=("chi1",), indexing="zero-based",
                              units="radian")
    d = SymmetricTorsionDescriptor(mol, defs)
    assert len(d.operations) == 2 and d.n_columns == 2


@requires_sklearn
def test_clusters_do_not_depend_on_which_methyl_is_called_cd1():
    """Frames of three chi2 rotamers (defined by the torsion to the unique HG), half of them with
    CD1 and CD2 named the other way round. The symmetry-first fit recovers the rotamers exactly; a
    fit on the stored torsion to CD1 does not: its states are defined by the labels."""
    from sklearn.metrics import adjusted_rand_score

    from md_tools.analysis import t_hdbscan

    mol, defs, hg = _leu()
    swap = _methyl_swap(mol)
    rng = np.random.default_rng(1)
    n = 600
    truth = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    centres = np.deg2rad([60.0, 180.0, -60.0])
    frames = []
    for i in range(n):
        conf = Chem.Conformer(mol.GetConformer())
        rdMolTransforms.SetDihedralRad(conf, 4, 5, 6, hg, float(wrap(centres[truth[i]] + rng.normal(0, 0.12))))
        rdMolTransforms.SetDihedralRad(conf, 3, 4, 5, 6, float(wrap(3.0 + rng.normal(0, 0.1))))
        xyz = conf.GetPositions()
        frames.append(xyz[list(swap)] if rng.random() < 0.5 else xyz)
    xyz = np.asarray(frames)
    tors = np.column_stack([dihedral(*(xyz[:, i] for i in q)) for q in LEU_TORSIONS.values()])

    sym = t_hdbscan(tors, molecule=mol, torsion_definitions=defs, coordinates=xyz,
                    atom_mapping=list(range(mol.GetNumAtoms())), periodic=False,
                    topology=_topology(mol), mass_floor=0.05).fit()
    legacy = t_hdbscan(tors, symmetry=False, mass_floor=0.05).fit()
    assert adjusted_rand_score(truth, sym.labels_) == pytest.approx(1.0)
    assert sorted(round(sym.cluster_mass_[c], 2) for c in range(sym.n_clusters_)) == \
        sorted(round(float(m), 2) for m in np.bincount(truth) / n)
    assert adjusted_rand_score(truth, legacy.labels_) < 0.5
