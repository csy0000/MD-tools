"""Hierarchical torsional clustering: two-round HDBSCAN, a thermodynamic cut, torsion bits, mass order.

`hierarchical_clustering(fit, coordinates)` takes a fitted symmetry-first `THDBSCAN` and returns a
finer, labelled partition. It exists because one HDBSCAN fit keeps two basins in one cluster when
the parent is more stable than its children (the condensed tree's `eom` rule), and a fit with a
smaller `min_samples` trades that for clusters that are only noise. The pipeline, in the order it
runs on every node (the whole ensemble is the first node):

1. ROUND 1. HDBSCAN on every torsion.
2. ROUND 2. A torsion on which no pair of round-1 clusters differs is DROPPED and HDBSCAN is fitted
   again on the rest. "Differs" is a permutation test: the two clusters' frames are pooled and
   reshuffled to the same sizes `torsion_null_test` times, and the observed circular W2 distance
   must exceed the 95th percentile of the reshuffled ones. A torsion that a symmetry operation moves
   together with a kept torsion is never dropped, because without it the operation maps one state
   onto the other and the two cannot be told apart (the aryl ring flip carries both the aryl-CO and
   the aryl-N torsion). The round-1 record (per-pair observed and null W2, what was dropped, both
   cluster counts) is kept in `rounds`, because it says why a torsion went.
3. A THERMODYNAMIC CUT on a leaf (or on a node HDBSCAN could not split). On the circular free-energy
   profile of ONE torsion in the node, cut 1 is the centre of the highest plateau (an empty bin counts
   as `max(F) + 1`, so a barrier nobody crossed is a candidate); cut 2 is any other bin whose free
   energy exceeds the lowest point of the smaller arc by at least `kT_cut` (default 1.6 kT, about
   1 kcal/mol at 300 K). Two cuts are needed on a circle: every route between the arcs must be
   blocked. Both arcs must hold at least `mass_floor` of ALL frames. The cut is taken directly only
   when that torsion is the ONLY one with such a pair, because then no other torsion can connect the
   arcs (the marginal profile has integrated them all). Frames within one bin of a NON-EMPTY cut are
   marked ambiguous: the cut passes through density there.
4. TORSION BITS. For each torsion the clusters are grouped by pairwise W2 (complete linkage,
   `bits_threshold` degrees), the groups are numbered from 0 by the frames they own, and clusters
   with identical bit vectors are merged.
5. NUMBERING by cluster mass, largest first. Not by structure: neighbouring numbers are not similar
   clusters.

WHAT THIS IS NOT. A cluster is still a density basin, not a metastable state. The two-round rule and
the cut are thresholds on a histogram built from the sampled frames; with a few dozen frames in a
node they are noisy, and a single trajectory gives no resampling stability. `kT_cut` is in units of
kT so the rule needs no temperature; 1.6 kT is 0.95 kcal/mol at 300 K.

The per-torsion terms of the distance were tried as a cache for round 2 and are NOT used: the
product route of `SymmetricTorsionDescriptor.distance_matrix` took 0.07 s for 1000 frames, building
the cache 0.86 s.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["hierarchical_clustering", "DEFAULT_KT_CUT", "DEFAULT_TORSION_NULL_TEST",
           "DEFAULT_BITS_THRESHOLD_DEGREES"]

DEFAULT_KT_CUT = 1.6
DEFAULT_TORSION_NULL_TEST = 40
DEFAULT_BITS_THRESHOLD_DEGREES = 20.0
NULL_QUANTILE = 0.95
MIN_PAIR_FRAMES = 5                     # a cluster pair is skipped by the null test below this
_W2_QUANTILES = 120


def _circular_w2(x, y, m: int = _W2_QUANTILES) -> float:
    """Circular Wasserstein-2 distance between two samples of angles, in degrees.

    The quantile functions on `m` levels, minimised over cyclic shifts of one and over the three
    2*pi branches. Angles are in [-pi, pi).
    """
    q = (np.arange(m) + 0.5) / m
    X = np.quantile(np.asarray(x, dtype=float) + np.pi, q)
    Y = np.quantile(np.asarray(y, dtype=float) + np.pi, q)
    j = np.arange(m)
    s = j[:, None] + j[None, :]
    Yj = Y[s % m] + 2.0 * np.pi * (s >= m)
    d = X[None, :] - Yj
    best = min(float(np.mean((d - sh) ** 2, axis=1).min()) for sh in (-2 * np.pi, 0.0, 2 * np.pi))
    return float(np.degrees(np.sqrt(best)))


def _coupling_groups(descriptor, n_selected: int) -> List[set]:
    """Selected columns that ONE irreducible symmetry operation moves together.

    A product of two operations with disjoint supports is not a new coupling, so only the minimal
    supports are kept and overlapping ones are merged.
    """
    quads = [tuple(int(a) for a in q) for q in descriptor.quads[:n_selected]]
    supports = set()
    for perm in np.asarray(descriptor.enumeration.permutations):
        moved = frozenset(k for k, q in enumerate(quads)
                          if tuple(int(perm[a]) for a in q) not in (q, q[::-1]))
        if moved:
            supports.add(moved)
    minimal = [s for s in supports if not any(o < s for o in supports)]
    groups: List[set] = []
    for m in minimal:
        m = set(m)
        for g in [g for g in groups if g & m]:
            groups.remove(g)
            m |= g
        groups.append(m)
    return groups


def _profile_cuts(x: np.ndarray, kT_cut: float, floor_frames: float, n_bins: int):
    """Two cuts on one circular histogram profile, or None.

    F = -ln p in kT, zero at its minimum, empty bins set to `max(F) + 1`. Cut 1 is the centre of the
    contiguous plateau at the maximum. Cut 2 is a bin whose F exceeds the lowest point of the smaller
    arc by at least `kT_cut`; each arc must hold at least `floor_frames` frames.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    cnt = np.histogram(x, edges)[0].astype(float)
    p = cnt / cnt.sum()
    with np.errstate(divide="ignore"):
        F = -np.log(p)
    fin = np.isfinite(F)
    F = F - F[fin].min()
    F[~fin] = F[fin].max() + 1.0
    top = set(np.flatnonzero(F >= F.max() - 1e-9).tolist())
    i0 = min(top)
    run = [i0]
    j = i0
    while (j + 1) % n_bins in top and (j + 1) % n_bins not in run:
        j = (j + 1) % n_bins
        run.append(j)
    j = i0
    while (j - 1) % n_bins in top and (j - 1) % n_bins not in run:
        j = (j - 1) % n_bins
        run.insert(0, j)
    c1 = run[len(run) // 2]
    for c2 in sorted((i for i in range(n_bins) if i != c1), key=lambda i: -F[i]):
        if F[c2] < kT_cut:                 # sorted by F: no later bin clears it (also implied by the
            break                          # test below, since F[minor] >= 0; kept as the early exit)
        ia = np.array([(c1 + 1 + k) % n_bins for k in range((c2 - c1) % n_bins - 1)], int)
        ib = np.array([(c2 + 1 + k) % n_bins for k in range((c1 - c2) % n_bins - 1)], int)
        if ia.size == 0 or ib.size == 0:
            continue
        ma, mb = p[ia].sum(), p[ib].sum()
        if min(ma, mb) * len(x) < floor_frames:
            continue
        minor = ia if ma < mb else ib
        if F[c2] - F[minor].min() >= kT_cut:
            half = np.pi / n_bins
            return {"bins": (c1, c2), "F_cut1": float(F[c1]), "F_cut2": float(F[c2]),
                    "minor_fraction": float(min(ma, mb)),
                    "cut1_degrees": float(np.degrees(edges[c1] + half)),
                    "cut2_degrees": float(np.degrees(edges[c2] + half))}
    return None


class _Hierarchy:
    def __init__(self, fit, coordinates, *, aligned, torsion_null_test, kT_cut, mass_floor,
                 max_depth, min_node_frames, n_bins, seed, verbose):
        d = fit.descriptor_
        self.fit, self.coordinates = fit, coordinates
        self.molecule = d.molecule
        self.K = int(fit.n_torsions_)
        self.names = list(fit.names)
        self.quads = [tuple(int(a) for a in q) for q in d.quads[:self.K]]
        self.theta = fit.theta_
        self.N = int(fit.n_frames_)
        self.AL = aligned[:, :self.K]
        self.null = int(torsion_null_test)
        self.kT_cut = float(kT_cut)
        self.floor = float(mass_floor) * self.N
        self.max_depth, self.min_node = int(max_depth), int(min_node_frames)
        self.n_bins, self.seed, self.verbose = int(n_bins), int(seed), bool(verbose)
        self.groups = _coupling_groups(d, self.K)
        self.leaves: List[Dict[str, Any]] = []
        self.ambiguous: List[np.ndarray] = []
        self.rounds: List[Dict[str, Any]] = []
        self.cuts: List[Dict[str, Any]] = []
        self.log: List[str] = []

    def say(self, indent: str, text: str) -> None:
        line = f"{indent}{text}"
        self.log.append(line)
        if self.verbose:
            print(line, flush=True)

    def _fit_on(self, rows: np.ndarray, cols: Sequence[int]):
        from ._t_hdbscan import THDBSCAN
        from ._t_symmetry import TorsionDefinitions

        fit = self.fit
        defs = TorsionDefinitions(quads=tuple(self.quads[c] for c in cols),
                                  names=tuple(self.names[c] for c in cols),
                                  indexing="zero-based", units="radian")
        kw: Dict[str, Any] = dict(self.fit._coordinate_kwargs)
        mw = fit.metric_weights
        if mw is not None and np.ndim(mw) == 1 and len(mw) == self.K:
            mw = np.asarray(mw)[list(cols)]
        extra: Dict[str, Any] = {}
        if fit.vote_rule == "fraction":
            extra["min_vote_fraction"] = fit.min_vote_fraction
        else:
            extra["min_vote"] = fit.min_vote
        sub = THDBSCAN(self.theta[rows][:, list(cols)], molecule=self.molecule,
                       torsion_definitions=defs, coordinates=self.coordinates[rows],
                       mass_floor=min(0.5, self.floor / len(rows)), allow_single_cluster=False,
                       vote_rule=fit.vote_rule, k=fit.k, metric_weights=mw,
                       min_samples_fraction=fit.min_samples_fraction, min_samples=fit.min_samples,
                       **extra, **kw)
        try:
            return sub.fit()
        except RuntimeError:                       # no draw produced a cluster
            return None

    # ---------------------------------------------------------------- one node: rounds 1 and 2
    def node(self, rows: np.ndarray, level: int, tag: str) -> bool:
        ind = "  " * level
        K = self.K
        f1 = self._fit_on(rows, list(range(K)))
        if f1 is None or f1.n_clusters_ < 1:
            self.say(ind, f"{tag} round 1: no clusters")
            return False
        n1, l1 = int(f1.n_clusters_), f1.labels_
        al1 = self.AL[rows]
        rng = np.random.default_rng(self.seed)
        table: List[List[Dict[str, float]]] = []
        dropped: List[int] = []
        for k in range(K):
            info: List[Dict[str, float]] = []
            for a, b in (itertools.combinations(range(n1), 2) if self.null else ()):
                xa, xb = al1[l1 == a, k], al1[l1 == b, k]
                if len(xa) <= MIN_PAIR_FRAMES or len(xb) <= MIN_PAIR_FRAMES:
                    continue
                obs = _circular_w2(xa, xb)
                pool, na = np.concatenate([xa, xb]), len(xa)
                nulls = []
                for _ in range(self.null):
                    pm = rng.permutation(len(pool))
                    nulls.append(_circular_w2(pool[pm[:na]], pool[pm[na:]]))
                info.append({"pair": (a, b), "w2": obs,
                             "null95": float(np.quantile(nulls, NULL_QUANTILE))})
            table.append(info)
            if info and not any(r["w2"] > r["null95"] for r in info):
                dropped.append(k)
        D0 = list(dropped)
        dropped = [k for k in dropped
                   if all(j in D0 for g in self.groups if k in g for j in g)]
        keep = [k for k in range(K) if k not in dropped]
        rec: Dict[str, Any] = {"tag": tag, "level": level, "frames": int(len(rows)),
                               "round1_clusters": n1, "w2": table,
                               "dropped": [self.names[k] for k in dropped],
                               "kept_by_coupling": [self.names[k] for k in D0 if k not in dropped]}
        self.rounds.append(rec)
        self.say(ind, f"{tag} frames {len(rows)} round 1: {n1} clusters; dropped "
                      f"{[self.names[k].split()[0] for k in dropped]}")
        if not keep:
            return False
        if dropped:
            f2 = self._fit_on(rows, keep)
            if f2 is None:
                self.say(ind, "round 2: no clusters")
                return False
            l2, n2 = f2.labels_, int(f2.n_clusters_)
        else:
            l2, n2 = l1, n1
        rec["round2_clusters"] = n2
        self.say(ind, f"round 2: {n2} clusters {[int((l2 == c).sum()) for c in range(n2)]}")
        for c in range(n2):
            self.settle(rows[l2 == c], f"{tag}.{c}", level, len(rows), ind)
        return True

    # ---------------------------------------------------------------- one cluster: the cut
    def settle(self, rows, tag, level, parent_n, ind) -> None:
        ang = self.AL[rows]
        found = [_profile_cuts(ang[:, k], self.kT_cut, self.floor, self.n_bins)
                 for k in range(self.K)]
        ks = [k for k in range(self.K) if found[k] is not None]
        if not ks:
            self.leaves.append({"tag": tag, "rows": rows, "status": "leaf"})
            return
        if len(rows) >= self.min_node and level + 1 < self.max_depth and len(rows) < parent_n:
            if self.node(rows, level + 1, tag):
                return
        if len(ks) == 1 and self.cut(rows, ang, ks[0], found[ks[0]], tag, level, ind):
            return
        self.leaves.append({"tag": tag, "rows": rows, "status": "two-well left",
                            "torsions": [self.names[k] for k in ks]})
        self.say(ind, f"{tag}: two-well left in {[self.names[k].split()[0] for k in ks]}")

    def cut(self, rows, ang, k, found, tag, level, ind) -> bool:
        nb = self.n_bins
        c1, c2 = found["bins"]
        edges = np.linspace(-np.pi, np.pi, nb + 1)
        b = np.clip(np.digitize(ang[:, k], edges) - 1, 0, nb - 1)
        amb = np.zeros(len(rows), bool)
        for c in (c1, c2):
            if (b == c).any():                     # a non-empty cut: its bin and both neighbours
                amb |= np.abs(((b - c + nb // 2) % nb) - nb // 2) <= 1
        in_a = ((b - c1) % nb) <= ((c2 - c1) % nb)
        A, B = in_a & ~amb, ~in_a & ~amb
        if min(A.sum(), B.sum()) < self.floor:
            return False
        self.say(ind, f"{tag}: cut on {self.names[k].split()[0]} -> {int(A.sum())} + "
                      f"{int(B.sum())}, ambiguous {int(amb.sum())}")
        self.cuts.append({"tag": tag, "torsion": self.names[k], "frames": (int(A.sum()), int(B.sum())),
                          "ambiguous": int(amb.sum()), **{key: found[key] for key in
                                                          ("cut1_degrees", "cut2_degrees")}})
        self.ambiguous.append(rows[amb])
        for sfx, mm in (("a", A), ("b", B)):
            self.settle(rows[mm], tag + sfx, level + 1, len(rows), ind + "  ")
        return True


def _bits(aligned: np.ndarray, labels: np.ndarray, n_clusters: int, threshold: float):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    K = aligned.shape[1]
    own = np.array([(labels == c).sum() for c in range(n_clusters)])
    bits = np.zeros((n_clusters, K), int)
    for j in range(K):
        xs = [aligned[labels == c, j] for c in range(n_clusters)]
        W = np.zeros((n_clusters, n_clusters))
        for a, b in itertools.combinations(range(n_clusters), 2):
            W[a, b] = W[b, a] = _circular_w2(xs[a], xs[b])
        g = fcluster(linkage(squareform(W, checks=False), "complete"), t=threshold,
                     criterion="distance")
        mass = {q: own[g == q].sum() for q in set(g)}
        rank = {q: i for i, q in enumerate(sorted(mass, key=lambda q: -mass[q]))}
        bits[:, j] = [rank[q] for q in g]
    return bits


def hierarchical_clustering(fit, coordinates, *, gauge=None,
                            torsion_null_test: int = DEFAULT_TORSION_NULL_TEST,
                            kT_cut: float = DEFAULT_KT_CUT, mass_floor: float = 0.01,
                            bits_threshold: float = DEFAULT_BITS_THRESHOLD_DEGREES,
                            merge_by_bits: bool = True, max_depth: int = 6,
                            min_node_frames: int = 40, n_bins: int = 72, seed: int = 0,
                            rmsd: bool = True, units: Optional[str] = None,
                            max_per_cluster: int = 500, verbose: bool = False) -> Dict[str, Any]:
    """Two-round HDBSCAN, thermodynamic cut, torsion bits and mass numbering; see the module docstring.

    `fit` is a FITTED symmetry-first `THDBSCAN` on every torsion of interest; its aligned angles
    (`fit.align_all(per_cluster=True, gauge=gauge)`) are the angles every profile, test and bit is
    computed from, so the labelling the clusters are drawn in is one labelling. `coordinates` are the
    frames the fit was made from, in order, indexable by frame (an `mdtraj.Trajectory`, or an array
    for which `units` is needed when `rmsd`).

    `torsion_null_test` reshuffles per cluster pair in the round-2 drop test (0 disables the drop,
    nothing is discarded). `kT_cut` is the barrier in kT. `mass_floor` is a fraction of ALL frames:
    the smallest cluster HDBSCAN may form in a node and the smallest arc a cut may leave.
    `bits_threshold` is in degrees of W2; `merge_by_bits=False` keeps the bit table but merges
    nothing. Frames in no leaf get `NOISE_LABEL`, frames next to a non-empty cut `AMBIGUOUS_LABEL`.

    Returns `labels` (clusters 0.. by mass, largest first), `n_clusters`, `cluster_mass`,
    `n_unassigned`, `n_ambiguous`, `rounds` (round-1 and round-2 record per node), `cuts`, `leaves`,
    `bits` (clusters x torsions), `bit_groups`, `merged`, `rmsd` (`intercluster_rmsd` output in the
    final numbering, or None), `aligned_angles`, `torsion_names`, `quads`, `molecule`, `settings`
    and `log`.
    """
    if not fit.fitted_:
        raise RuntimeError("call fit() before hierarchical_clustering()")
    if not fit.symmetry:
        raise NotImplementedError("hierarchical_clustering() needs the symmetry-first route")
    if not fit.weights_uniform_:
        raise NotImplementedError("frame weights are not supported; resample the ensemble first")
    if int(torsion_null_test) < 0 or not float(kT_cut) > 0.0:
        raise ValueError("torsion_null_test must be >= 0 and kT_cut positive")
    if not 0.0 < float(mass_floor) < 0.5:
        raise ValueError("mass_floor must be a fraction of all frames in (0, 0.5)")
    if int(fit.n_clusters_) < 1:
        raise ValueError("the fit has no cluster to align to")
    from ._t_hdbscan import AMBIGUOUS_LABEL, NOISE_LABEL

    aligned = fit.align_all(per_cluster=True, gauge=gauge)["angles"]
    H = _Hierarchy(fit, coordinates, aligned=aligned, torsion_null_test=torsion_null_test,
                   kT_cut=kT_cut, mass_floor=mass_floor, max_depth=max_depth,
                   min_node_frames=min_node_frames, n_bins=n_bins, seed=seed, verbose=verbose)
    root = np.arange(H.N)
    H.node(root, 0, "R")
    lab = np.full(H.N, NOISE_LABEL, dtype=np.int64)
    for i, leaf in enumerate(sorted(H.leaves, key=lambda z: -len(z["rows"]))):
        lab[leaf["rows"]] = i
    amb_rows = np.concatenate(H.ambiguous) if H.ambiguous else np.array([], dtype=int)
    lab[amb_rows] = AMBIGUOUS_LABEL
    k0 = int(lab.max()) + 1 if (lab >= 0).any() else 0
    bits = _bits(H.AL, lab, k0, bits_threshold) if k0 > 1 else np.zeros((k0, H.K), int)
    signature: Dict[tuple, List[int]] = {}
    for c in range(k0):
        signature.setdefault(tuple(bits[c]), []).append(c)
    merged = [v for v in signature.values() if len(v) > 1]
    groups = (list(signature.values()) if merge_by_bits else [[c] for c in range(k0)])
    out = np.where(lab >= 0, NOISE_LABEL, lab)
    weights = fit.weights_
    order = sorted(groups, key=lambda g: (-sum(int((lab == c).sum()) for c in g), min(g)))
    for new, members in enumerate(order):                         # mass numbering, largest first
        for c in members:
            out[lab == c] = new
    n_out = len(order)
    total = float(weights.sum())
    mass = {i: float(weights[out == i].sum() / total) for i in range(n_out)}
    result: Dict[str, Any] = {
        "labels": out, "n_clusters": n_out, "cluster_mass": mass,
        "n_unassigned": int((out < 0).sum()), "n_ambiguous": int((out == AMBIGUOUS_LABEL).sum()),
        "rounds": H.rounds, "cuts": H.cuts,
        "leaves": [{"tag": l["tag"], "frames": int(len(l["rows"])), "status": l["status"]}
                   for l in H.leaves],
        "bits": bits, "bit_groups": [int(bits[:, j].max() + 1) if k0 else 0 for j in range(H.K)],
        "merged": merged, "rmsd": None, "aligned_angles": H.AL, "torsion_names": H.names,
        "quads": H.quads, "molecule": H.molecule, "log": H.log,
        "settings": {"torsion_null_test": H.null, "kT_cut": H.kT_cut, "mass_floor": float(mass_floor),
                     "floor_frames": H.floor, "bits_threshold": float(bits_threshold),
                     "merge_by_bits": bool(merge_by_bits), "max_depth": H.max_depth,
                     "min_node_frames": H.min_node, "n_bins": H.n_bins, "seed": H.seed,
                     "null_quantile": NULL_QUANTILE},
    }
    if rmsd and n_out:
        from ._symmetric_rmsd import heavy_atoms, intercluster_rmsd, operations_on_atoms
        from ._t_recursive import _unit_factor

        kw = fit._coordinate_kwargs
        xyz = fit.descriptor_._mapped_whole(coordinates, kw["atom_mapping"], None, kw["periodic"],
                                            kw["topology"])
        atoms = heavy_atoms(H.molecule)
        ops = operations_on_atoms(fit.descriptor_.enumeration, atoms)
        X = np.asarray(xyz[:, atoms, :], dtype=np.float64) * _unit_factor(coordinates, units)
        result["rmsd"] = intercluster_rmsd(X, np.where(out >= 0, out, -1), ops,
                                           max_per_cluster=max_per_cluster, seed=seed)
        result["rmsd"]["atoms"] = list(atoms)
    return result
