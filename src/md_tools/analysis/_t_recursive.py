"""Recursive splitting of a torsional clustering by the geometry of the atoms its torsions move.

A torsional cluster is a density basin in torsion space. Nothing in the fit looks at how far apart
its members are in Cartesian space, so a cluster can be internally diverse: two sub-populations that
HDBSCAN's condensed tree kept together (a parent more stable than its children), or whose torsion
difference is small but whose atoms are not. `refine_by_rmsd` closes that: a cluster whose
within-cluster symmetry-corrected RMSD exceeds a cutoff (default 2 A) is fitted AGAIN, alone, with
`allow_single_cluster=False`, and the split is kept only if the refit yields at least two clusters.
The children are tested in turn.

WHAT IS MEASURED. The heavy atoms the torsions move, which is every heavy atom of the molecule: a
torsion turns everything attached beyond its central bond, so the four atoms that DEFINE it are a
small part of what it displaces (on folic acid they are 14 of 32, and they make a cluster that is
2.2 A wide over all heavy atoms look 0.8 A wide). `atoms='quadruplets'` measures only the defining
atoms, closed under the molecular symmetry, and a list measures exactly those. Hydrogens are left
out, as in the other RMSD of this package; the consequence is stated in `refine_by_rmsd`.

WHAT STOPS IT, in the order they apply: the cluster is within the cutoff; it is smaller than
`min_frames`; the depth limit is reached; the refit finds no second cluster. The last is what makes
the loop finite: every accepted split strictly shrinks the frames per node.

This is a post-processing of a fitted estimator. It returns a record and changes neither the
estimator nor its `predict`: new frames are still classified by the first-level vote.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["DEFAULT_CUTOFF_ANGSTROM", "moved_atoms", "torsion_atoms", "refine_by_rmsd"]

DEFAULT_CUTOFF_ANGSTROM = 2.0
_NM_TO_ANGSTROM = {"nm": 10.0, "nanometer": 10.0, "angstrom": 1.0, "a": 1.0, "å": 1.0}


def moved_atoms(fit) -> List[int]:
    """Every heavy atom of the molecule: what the selected torsions displace, sorted."""
    mol = fit.descriptor_.molecule
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def torsion_atoms(fit) -> List[int]:
    """Heavy atoms of the torsion quadruplets, closed under the molecular symmetry, sorted.

    Indices are the reference molecule's. The columns an operation adds (image columns) are
    included, so the set is the one the symmetry-aware distance itself touches.
    """
    d = fit.descriptor_
    mol = d.molecule
    seed = {int(i) for q in d.quads for i in q if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1}
    closed = set(seed)
    for perm in d.enumeration.permutations:
        closed.update(int(perm[a]) for a in seed)
    return sorted(closed)


def _unit_factor(coordinates, units: Optional[str]) -> float:
    if units is None:
        if hasattr(coordinates, "xyz"):
            return _NM_TO_ANGSTROM["nm"]
        raise ValueError("units is required for an array of coordinates ('angstrom' or 'nm'); an "
                         "mdtraj.Trajectory is taken as nm")
    try:
        return _NM_TO_ANGSTROM[str(units).lower()]
    except KeyError:
        raise ValueError(f"units must be 'angstrom' or 'nm', not {units!r}") from None


def _spread(rmsd_block: np.ndarray, statistic) -> Dict[str, float]:
    """Mean, std and the deciding statistic of the distinct pairs of one cluster's sample."""
    values = rmsd_block[np.triu_indices(rmsd_block.shape[0], 1)]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "statistic": float("nan")}
    stat = float(values.mean()) if statistic == "mean" else float(np.quantile(values, statistic))
    return {"mean": float(values.mean()), "std": float(values.std()), "statistic": stat}


def _subset(fit, rows: np.ndarray):
    """A copy of the fitted estimator restricted to `rows`, ready to fit again.

    Shares the descriptor (the symmetry and the metric are the same) and carries the frames' own
    closed angles, so no coordinate is needed to refit.
    """
    sub = copy.copy(fit)
    sub.theta_ = fit.theta_[rows]
    sub.closed_theta_ = fit.closed_theta_[rows]
    sub.X_ = fit.X_[rows]
    sub.weights_ = fit.weights_[rows]
    sub.weights_uniform_ = bool(np.allclose(sub.weights_, sub.weights_[0]))
    sub.n_frames_ = int(rows.size)
    sub.fitted_ = False
    if fit.resampling is not False:
        sub.resampling = min(int(fit.resampling), int(rows.size))
    return sub


def refine_by_rmsd(fit, coordinates, *, cutoff: float = DEFAULT_CUTOFF_ANGSTROM,
                   units: Optional[str] = None, atoms=None,
                   statistic="mean", max_depth: int = 4, min_frames: Optional[int] = None,
                   max_per_cluster: int = 1000, seed: int = 0, workers: int = 1,
                   box_vectors=None) -> Dict[str, Any]:
    """Split every cluster of a fitted `THDBSCAN` whose internal RMSD exceeds `cutoff`.

    `coordinates` are the frames the fit was made from, in the same order, as an
    `mdtraj.Trajectory` (nm) or an `(n_frames, n_atoms, 3)` array with `units`. The atom mapping,
    topology and periodicity of the fit are reused, and a molecule broken across the box is made
    whole as the torsions were. `cutoff` is in angstrom. `atoms=None` measures every heavy atom (what
    the torsions move); `'quadruplets'` only the atoms that define the torsions (`torsion_atoms`); a
    list is reference-molecule indices and must be a union of symmetry orbits.

    A cluster is WIDE when `statistic` of its pairwise RMSD exceeds `cutoff`: `'mean'` (default) or
    a quantile in (0, 1). The mean of a cluster that is 90 % one structure and 10 % another 4 A away
    is 0.7 A, so a minority sub-population can hide in it; a quantile such as 0.95 does not hide it.
    Pairs are drawn from at most `max_per_cluster` frames per cluster (seeded).

    A wide cluster is refitted alone with the fit's own settings and `allow_single_cluster=False`.
    The split is kept only when that yields at least two clusters; frames the refit calls noise or
    ambiguous take those codes, so a split can only move mass out of a cluster, never into one that
    was not a part of it. Hydrogens are not measured: two clusters that differ only in a hydroxyl
    proton (paracetamol's A and B) have the same heavy-atom RMSD and are neither merged nor split
    by this.

    Returns `labels` (every frame; leaf clusters numbered by size, noise and ambiguous codes as in
    the fit), `n_clusters`, `cluster_mass`, `tree` (one record per node examined: id, parent,
    depth, frames, the RMSD summary, `action`), `atoms` and the settings.
    """
    if not fit.fitted_:
        raise RuntimeError("call fit() before refine_by_rmsd()")
    if not fit.symmetry:
        raise NotImplementedError("refine_by_rmsd() needs the symmetry-first route: the atoms and "
                                  "operations come from the molecule")
    if not float(cutoff) > 0.0:
        raise ValueError("cutoff must be positive (angstrom)")
    if not (statistic == "mean" or (isinstance(statistic, float) and 0.0 < statistic < 1.0)):
        raise ValueError("statistic must be 'mean' or a quantile in (0, 1)")
    if int(max_depth) < 1:
        raise ValueError("max_depth must be at least 1")
    from ._symmetric_rmsd import operations_on_atoms, symmetric_rmsd

    factor = _unit_factor(coordinates, units)
    kw = fit._coordinate_kwargs
    xyz = fit.descriptor_._mapped_whole(coordinates, kw["atom_mapping"], box_vectors,
                                        kw["periodic"], kw["topology"])
    if xyz.shape[0] != fit.n_frames_:
        raise ValueError(f"{xyz.shape[0]} coordinate frames for {fit.n_frames_} fitted frames")
    if atoms is None:
        chosen = moved_atoms(fit)
    elif isinstance(atoms, str):
        if atoms != "quadruplets":
            raise ValueError("atoms must be None (all heavy atoms), 'quadruplets', or a list of "
                             "atom indices")
        chosen = torsion_atoms(fit)
    else:
        chosen = sorted({int(a) for a in atoms})
    ops = operations_on_atoms(fit.descriptor_.enumeration, chosen)
    X = np.asarray(xyz[:, chosen, :], dtype=np.float64) * factor
    min_frames = 2 * int(fit.k) if min_frames is None else int(min_frames)
    rng = np.random.default_rng(seed)

    labels = np.asarray(fit.labels_, dtype=np.int64).copy()
    out = labels.copy()                                  # unassigned codes stay as the fit made them
    tree: List[Dict[str, Any]] = []
    leaves: List[np.ndarray] = []

    def examine(rows: np.ndarray, depth: int, parent: Optional[int]) -> None:
        node = len(tree)
        pick = rows if rows.size <= max_per_cluster else np.sort(
            rng.choice(rows, size=int(max_per_cluster), replace=False))
        spread = _spread(symmetric_rmsd(X[pick], X[pick], ops, workers=workers), statistic)
        rec = {"node": node, "parent": parent, "depth": depth, "n_frames": int(rows.size),
               "rmsd_mean": spread["mean"], "rmsd_std": spread["std"],
               "rmsd_statistic": spread["statistic"], "action": None, "children": []}
        tree.append(rec)
        if not spread["statistic"] > float(cutoff):
            rec["action"] = "within-cutoff"
        elif rows.size < min_frames:
            rec["action"] = "too-small"
        elif depth >= int(max_depth):
            rec["action"] = "depth-limit"
        else:
            child = _subset(fit, rows)
            child.allow_single_cluster = False
            try:
                child.fit()
                got = int(child.n_clusters_)
            except RuntimeError:                         # no draw produced a cluster
                got = 0
            groups = [rows[child.labels_ == c] for c in range(got)]
            groups = [g for g in groups if g.size]       # a fitted cluster no frame's vote reaches
            if len(groups) < 2:
                rec["action"] = "no-split"
            else:
                rec["action"] = "split"
                for members in groups:
                    rec["children"].append(len(tree))
                    examine(members, depth + 1, node)
                # frames the refit left unassigned keep ITS codes, not the parent's label
                lost = child.labels_ < 0
                out[rows[lost]] = child.labels_[lost]
                return
        rec["leaf"] = len(leaves)
        leaves.append(rows)

    for c in range(int(fit.n_clusters_)):
        rows = np.flatnonzero(labels == c)
        if rows.size:
            examine(rows, 0, None)

    order = sorted(range(len(leaves)), key=lambda i: (-leaves[i].size, i))
    final = out.copy()                                   # every frame of a cluster is in a leaf
    renumber = {}                                        # or was given a code by the refit that
    for new, old in enumerate(order):                    # dropped it
        final[leaves[old]] = new
        renumber[old] = new
    for rec in tree:
        if "leaf" in rec:
            rec["cluster"] = renumber[rec.pop("leaf")]
    total = float(fit.weights_.sum())
    return {
        "labels": final, "n_clusters": len(leaves),
        "cluster_mass": {i: float(fit.weights_[final == i].sum() / total)
                         for i in range(len(leaves))},
        "n_splits": sum(1 for r in tree if r["action"] == "split"),
        "tree": tree, "atoms": chosen, "n_operations": len(ops),
        "settings": {"cutoff_angstrom": float(cutoff), "statistic": statistic,
                     "max_depth": int(max_depth), "min_frames": int(min_frames),
                     "max_per_cluster": int(max_per_cluster), "allow_single_cluster": False},
    }
