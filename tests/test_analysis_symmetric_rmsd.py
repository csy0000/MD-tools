"""Symmetry-corrected RMSD: the minimum over the symmetry operations of the superposed RMSD.

Each test names the mistake it exists to catch: a relabelled copy of one structure reported as
different, an operation that is not applied to the whole set, a minimum that is not the minimum
(checked against brute force), and an atom set that a permutation carries outside itself.
"""
from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402
from rdkit.Chem import AllChem                                                # noqa: E402

from md_tools.analysis import (enumerate_symmetry, heavy_atoms, intercluster_rmsd,   # noqa: E402
                               operations_on_atoms, symmetric_rmsd)


def _kabsch(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    S[-1] *= np.sign(np.linalg.det(U @ Vt))
    return float(np.sqrt(max((Xc ** 2).sum() + (Yc ** 2).sum() - 2 * S.sum(), 0.0) / len(X)))


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def _mol(smiles, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _setup(smiles):
    mol = _mol(smiles)
    atoms = heavy_atoms(mol)
    return mol, atoms, operations_on_atoms(enumerate_symmetry(mol), atoms)


def test_a_relabelled_copy_is_at_zero_and_an_index_rmsd_calls_it_different():
    """Paracetamol with the ring flip: the heavy atoms of one structure written two ways."""
    mol, atoms, ops = _setup("CC(=O)Nc1ccc(O)cc1")
    assert len(ops) == 2                                               # identity and the ring flip
    X = mol.GetConformer().GetPositions()[atoms]
    flip = ops[1]
    Y = X[flip] @ _rotation(np.random.default_rng(0)).T + 3.0          # same geometry, renamed, moved
    assert symmetric_rmsd(X[None], Y[None], ops)[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert _kabsch(X, Y) > 0.5                                         # by atom index: large
    _, which = symmetric_rmsd(X[None], Y[None], ops, return_operation=True)
    assert which[0, 0] == 1


def test_it_equals_brute_force_over_the_operations_on_random_data():
    mol, atoms, ops = _setup("CC(C)[C@H](N)C(=O)[O-]")                  # valine: methyls and O both swap
    assert len(ops) >= 4
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, len(atoms), 3)) * 2
    Y = rng.normal(size=(4, len(atoms), 3)) * 2
    got = symmetric_rmsd(X, Y, ops, workers=2)
    want = np.array([[min(_kabsch(x, y[o]) for o in ops) for y in Y] for x in X])
    assert np.allclose(got, want, atol=1e-9)
    assert (got <= np.array([[_kabsch(x, y) for y in Y] for x in X]) + 1e-12).all()


def test_the_carboxylate_swap_is_an_operation_only_with_canonicalization():
    mol = _mol("CCC(=O)[O-]")
    atoms = heavy_atoms(mol)
    canon = operations_on_atoms(enumerate_symmetry(mol), atoms)
    plain = operations_on_atoms(enumerate_symmetry(mol, canonicalize=False), atoms)
    assert len(canon) == 2 and len(plain) == 1
    X = mol.GetConformer().GetPositions()[atoms]
    Y = X[canon[1]]
    assert symmetric_rmsd(X[None], Y[None], canon)[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert symmetric_rmsd(X[None], Y[None], plain)[0, 0] > 0.05


def test_an_atom_set_that_a_permutation_carries_outside_itself_is_refused():
    mol = _mol("CC(=O)[O-]")
    enum = enumerate_symmetry(mol)
    one_oxygen = [a for a in heavy_atoms(mol) if a != 3]               # drops one carboxylate O
    with pytest.raises(ValueError, match="outside the set"):
        operations_on_atoms(enum, one_oxygen)


def test_intercluster_matrix_shape_symmetry_and_the_diagonal_is_the_spread():
    mol, atoms, ops = _setup("CC(=O)Nc1ccc(O)cc1")
    rng = np.random.default_rng(2)
    base = mol.GetConformer().GetPositions()[atoms]
    A = base + rng.normal(0, 0.02, size=(30, len(atoms), 3))
    B = base + np.array([0.0, 0.0, 0.0]) + rng.normal(0, 0.02, size=(20, len(atoms), 3))
    B[:, 0, :] += [1.5, 0.0, 0.0]                                       # a real, larger difference
    X = np.concatenate([A, B, base[None]])
    labels = np.array([0] * 30 + [1] * 20 + [-1])
    out = intercluster_rmsd(X, labels, ops)
    assert out["clusters"] == [0, 1] and out["mean"].shape == (2, 2)
    assert np.allclose(out["mean"], out["mean"].T, equal_nan=True)
    assert out["mean"][0, 1] > 3 * out["mean"][0, 0]                    # between > within
    assert out["mean"][0, 0] > 0 and out["n_frames"] == {0: 30, 1: 20}  # noise frame (-1) left out
    small = intercluster_rmsd(X, labels, ops, max_per_cluster=10, seed=1)
    assert small["n_frames"] == {0: 10, 1: 10} and small["n_frames_in_cluster"] == {0: 30, 1: 20}
