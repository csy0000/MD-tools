"""Symmetry-corrected RMSD: the minimum over the molecule's symmetry operations of the superposed RMSD.

A configuration written with two atoms named the other way round (the oxygens of a carboxylate, the
two sides of a phenyl ring, the two methyls of a valine) is the SAME configuration, and an RMSD by
atom index calls it different. The correction is a minimum over the operations of the molecule's
symmetry group, the same group and the same enumeration the torsional distance uses
(`enumerate_symmetry`, canonicalised by default), restricted to the atoms being compared.

    atoms, operations = operations_on_atoms(enumerate_symmetry(mol), heavy_atoms(mol))
    r = symmetric_rmsd(X[:, atoms], ref[None, atoms], operations)         # (n_frames, 1)

THE MINIMUM IS EXACT. For one fixed operation the optimal rotation is Kabsch's, so the minimum over
the operations of the Kabsch RMSD is the global minimum over rotation and permutation together;
nothing here is an alternating or greedy search. Cost is `len(operations)` superpositions per pair,
and that count is what the restriction to a chosen atom set keeps small: on c(RGDfV) the heavy
atoms have 16 distinct operations where the whole molecule has 147 456.

WHICH ATOMS. Heavy atoms are the default (`heavy_atoms`). Hydrogens that no selected quantity moves
only multiply the operations, and a per-group assignment that treats them independently can pair
atoms across different heavy atoms and report an RMSD BELOW the minimum over valid operations
(measured: cpptraj `symmrmsd` on heavy atoms plus polar hydrogens, 29 of 100 frames, by up to
0.07 A). An operation here is always a whole-molecule automorphism, so that cannot happen.

NUMPY ONLY, and it needs no RDKit itself: it takes the enumeration, not the molecule.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["heavy_atoms", "operations_on_atoms", "symmetric_rmsd", "intercluster_rmsd"]


def heavy_atoms(molecule) -> List[int]:
    """Indices of the non-hydrogen atoms of an RDKit molecule, in atom order."""
    return [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1]


def operations_on_atoms(enumeration, atoms: Sequence[int]) -> List[np.ndarray]:
    """The distinct symmetry operations as seen on `atoms`, identity first.

    `enumeration` is a `SymmetryEnumeration`. Each operation is an integer array `p` of positions
    WITHIN `atoms`: relabelling atom `atoms[k]` as `atoms[p[k]]`. Whole-molecule permutations that
    act identically on these atoms are one operation, which is the collapse that makes a
    heavy-atom RMSD cost 16 superpositions and not 147 456. Refuses an atom set that a permutation
    carries outside itself: the comparison would then describe a different set of atoms for each
    operation.
    """
    atoms = [int(a) for a in atoms]
    pos = {a: k for k, a in enumerate(atoms)}
    if len(pos) != len(atoms):
        raise ValueError("atoms must be distinct")
    seen = {tuple(range(len(atoms)))}
    ops = [np.arange(len(atoms))]
    for perm in enumeration.permutations:
        try:
            image = tuple(pos[int(perm[a])] for a in atoms)
        except KeyError:
            raise ValueError("a symmetry operation maps the chosen atoms onto atoms outside the "
                             "set; choose a union of symmetry orbits (all heavy atoms, or heavy "
                             "atoms plus every hydrogen on them)") from None
        if image not in seen:
            seen.add(image)
            ops.append(np.array(image))
    return ops


def _centre(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=1, keepdims=True)


def _one_operation(Xc: np.ndarray, Yc: np.ndarray, x2: np.ndarray, y2: np.ndarray, n_atoms: int,
                   rows: int) -> np.ndarray:
    """Superposed RMSD of every row of `Xc` against every row of `Yc` (already centred and
    permuted), as an `(n, m)` array, in blocks of `rows` rows."""
    n, m = Xc.shape[0], Yc.shape[0]
    out = np.empty((n, m))
    Ya = [np.ascontiguousarray(Yc[:, :, b].T) for b in range(3)]          # (N, m)
    for lo in range(0, n, rows):
        hi = min(n, lo + rows)
        H = np.empty((hi - lo, m, 3, 3))
        for a in range(3):
            xa = Xc[lo:hi, :, a]
            for b in range(3):
                H[:, :, a, b] = xa @ Ya[b]
        flat = H.reshape(-1, 3, 3)
        s = np.linalg.svd(flat, compute_uv=False)
        s[:, 2] *= np.sign(np.linalg.det(flat))                           # no reflections
        msd = (x2[lo:hi, None] + y2[None, :] - 2.0 * s.sum(axis=1).reshape(hi - lo, m)) / n_atoms
        out[lo:hi] = np.sqrt(np.maximum(msd, 0.0))
    return out


def symmetric_rmsd(X: np.ndarray, Y: np.ndarray, operations: Sequence[np.ndarray], *,
                   return_operation: bool = False, rows: int = 128, workers: int = 1):
    """Minimum over `operations` of the superposed RMSD, for every pair of one frame of `X` and one
    of `Y`.

    `X` is `(n, N, 3)` and `Y` is `(m, N, 3)`, in the same units, over the SAME atoms in the SAME
    order (`operations` are positions within those atoms; see `operations_on_atoms`). Returns an
    `(n, m)` array in the units of the input, and with `return_operation` also the index of the
    minimising operation per pair (the earlier operation on a tie). Each operation relabels `Y`;
    the group is closed under inverses, so relabelling one side is enough.

    `workers > 1` runs the operations in threads (the work is LAPACK, which releases the GIL).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 3 or Y.ndim != 3 or X.shape[1:] != Y.shape[1:] or X.shape[2] != 3:
        raise ValueError(f"expected (n, N, 3) and (m, N, 3) over the same atoms; got "
                         f"{X.shape} and {Y.shape}")
    n_atoms = X.shape[1]
    ops = [np.asarray(o, dtype=np.int64) for o in operations]
    if not ops:
        raise ValueError("at least one operation (the identity) is required")
    for o in ops:
        if o.shape != (n_atoms,) or sorted(o.tolist()) != list(range(n_atoms)):
            raise ValueError("an operation is not a permutation of the compared atoms")
    Xc, Yc = _centre(X), _centre(Y)
    x2, y2 = (Xc ** 2).sum(axis=(1, 2)), (Yc ** 2).sum(axis=(1, 2))

    def run(o):
        return _one_operation(Xc, Yc[:, o, :], x2, y2, n_atoms, rows)

    if workers > 1 and len(ops) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            stack = list(pool.map(run, ops))
    else:
        stack = [run(o) for o in ops]
    stack = np.stack(stack)                                                # (ops, n, m)
    best = stack.min(axis=0)
    if not return_operation:
        return best
    tie = 1e-12 * max(1.0, float(best.max()))
    return best, np.argmax(stack <= best[None] + tie, axis=0).astype(np.int64)


def intercluster_rmsd(X: np.ndarray, labels: Sequence[int], operations: Sequence[np.ndarray], *,
                      max_per_cluster: Optional[int] = 1000, seed: int = 0, workers: int = 1
                      ) -> Dict[str, object]:
    """Mean and standard deviation of the symmetry-corrected RMSD between and within clusters.

    `X` is `(n_frames, N, 3)` and `labels` the cluster of each frame; frames with a negative label
    (noise, ambiguous) are left out. Entry `(a, b)` summarises every pair of one frame from cluster
    `a` and one from cluster `b`; the diagonal summarises the distinct pairs inside a cluster
    (`i < j`), so it is the within-cluster spread, not a self-distance of zero. A cluster larger
    than `max_per_cluster` is represented by that many frames drawn without replacement (seeded;
    `None` uses every frame), and `n_frames` records how many were used.
    """
    X = np.asarray(X, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    if X.shape[0] != lab.size:
        raise ValueError(f"{lab.size} labels for {X.shape[0]} frames")
    clusters = [int(c) for c in np.unique(lab) if c >= 0]
    rng = np.random.default_rng(seed)
    pick = {}
    for c in clusters:
        idx = np.flatnonzero(lab == c)
        if max_per_cluster is not None and idx.size > int(max_per_cluster):
            idx = np.sort(rng.choice(idx, size=int(max_per_cluster), replace=False))
        pick[c] = idx
    k = len(clusters)
    mean = np.full((k, k), np.nan)
    std = np.full((k, k), np.nan)
    for ia, a in enumerate(clusters):
        for ib, b in enumerate(clusters):
            if ib < ia:
                mean[ia, ib], std[ia, ib] = mean[ib, ia], std[ib, ia]
                continue
            block = symmetric_rmsd(X[pick[a]], X[pick[b]], operations, workers=workers)
            values = block[np.triu_indices(block.shape[0], 1)] if a == b else block.ravel()
            if values.size:
                mean[ia, ib] = mean[ib, ia] = float(values.mean())
                std[ia, ib] = std[ib, ia] = float(values.std())
    return {"clusters": clusters, "mean": mean, "std": std,
            "n_frames": {c: int(pick[c].size) for c in clusters},
            "n_frames_in_cluster": {c: int((lab == c).sum()) for c in clusters},
            "n_operations": len(operations)}
