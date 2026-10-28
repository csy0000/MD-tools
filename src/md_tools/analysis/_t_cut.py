"""Cut-route torsional clustering: recursive free-energy cuts on symmetry-invariant coordinates.

`cut_clustering(fit)` partitions the frames of a `THDBSCAN` (fitted or not: only its angles and,
on the symmetry-first route, its symmetry operations are read) without HDBSCAN. Every frame gets a
label. The steps, in the order they run:

1. INVARIANT COORDINATES. A torsion no symmetry operation moves is its own coordinate. The
   torsions one operation moves together (a coupling group: a ring flip, a carboxylate) are
   replaced by averages over that group's own operations: z = <exp(i m theta_sigma(c))> for each
   torsion (m the number of images, which folds 2*theta for a two-fold flip) and
   <exp(i(theta_sigma(b) - theta_sigma(a)))> for each pair, cut on arg z. A pair sharing a central
   bond (two hydrogens on one nitrogen, or a hydrogen and the ring on one) is left out: their
   difference measures the geometry at the atom, not a rotamer. Where |z| < `z_min` the coordinate
   is ill-defined for that frame (the images cancel); such frames are left out of THAT
   coordinate's profile, and are still assigned by their value if a cut is made on it.
2. EMPTY-BIN CUTS (level 0). Cut only through bins with no count, recursively, until no coordinate
   of any cell has one that leaves both sides at least `mass_floor` of ALL frames.
3. ANNEALING. Each cell starts at E* = the largest finite free-energy range of any coordinate in
   it and lowers E* by `kT_step` down to `kT_cut`. At each E* the cell is cut on the coordinate
   with the highest barrier that clears E* (ties: coordinate order); both children continue at
   that E*.
4. NUMBERING by cluster mass, largest first.

THE CUT on one coordinate of one cell. F = -ln p in kT on `n_bins` bins over the circle. A cut is
placed at the MIDPOINT of a bin, so it is a dividing surface and no frame is left over.
* A coordinate NOT yet cut in the cell's ancestry is periodic and needs two cuts: cut 1 at the
  centre of the highest plateau (an empty bin ranks above every filled one), cut 2 the highest-F
  bin more than `min_p2p` from cut 1 with min(F(cut 1), F(cut 2)) minus the lowest F of the
  smaller arc at least E*.
* A coordinate ALREADY cut is bounded: the cell lives on one arc between two inherited cuts
  (inside or outside the parent's pair, whichever side it is). One cut suffices, it is searched
  only inside that arc, and both pieces, measured ALONG the arc up to the inherited boundaries,
  must be wider than `min_p2p`. The highest-F bin qualifies first; among equal F (an empty run)
  the one leaving the larger smaller side. Its barrier is F(cut) minus the lowest F of the
  smaller side.
* Both sides must hold at least `mass_floor` of all frames.

WHAT THIS IS NOT. A cluster is a region bounded by barriers in one-dimensional projections; two
basins that overlap in every projection are not separated, and a barrier is read from sampled
frames, so near the floor it is noisy. `kT_cut` is in kT and needs no temperature: 1 kT is 0.60
kcal/mol at 300 K.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import numpy as np

from ._t_hierarchy import _coupling_groups, _profile_cuts

__all__ = ["cut_clustering", "DEFAULT_CUT_KT", "DEFAULT_CUT_MASS_FLOOR", "DEFAULT_MIN_P2P"]

DEFAULT_CUT_KT = 1.0
DEFAULT_CUT_MASS_FLOOR = 0.001
DEFAULT_MIN_P2P = np.pi / 3.0           # radians: 60 degrees
DEFAULT_KT_STEP = 0.5
DEFAULT_Z_MIN = 0.5


def _free_energy(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """F = -ln p in kT, zero at its minimum, inf where a bin has no count."""
    cnt = np.histogram(x, edges)[0].astype(float)
    with np.errstate(divide="ignore"):
        F = -np.log(cnt / cnt.sum())
    fin = np.isfinite(F)
    F = F - F[fin].min()
    F[~fin] = np.inf
    return F


def _arc(c1: int, c2: int, n: int) -> List[int]:
    """Bins strictly between c1 and c2, counter-clockwise."""
    return [(c1 + 1 + k) % n for k in range((c2 - c1) % n - 1)]


def _periodic_barrier(x, c1, c2, edges) -> float:
    n = len(edges) - 1
    cnt = np.histogram(x, edges)[0]
    F = _free_energy(x, edges)
    ia, ib = _arc(c1, c2, n), _arc(c2, c1, n)
    minor = ia if cnt[ia].sum() < cnt[ib].sum() else ib
    return float(min(F[c1], F[c2]) - F[minor].min())


def _bounded_cut(x_all, x_valid, kT, floor_frames, lo, hi, edges, min_sep):
    """One cut inside the arc lo -> hi (bins whose centres are the inherited cuts), or None.

    `x_valid` builds the profile, `x_all` (every frame of the cell) the side masses. With `kT` nan a
    cut is accepted only through an empty bin.
    """
    n = len(edges) - 1
    width = 360.0 / n
    half = np.pi / n
    L = (hi - lo) % n
    F = _free_energy(x_valid, edges)
    u = np.mod(x_all - (edges[lo] + half), 2.0 * np.pi)
    best = None
    for k in range(1, L):
        if k * width <= min_sep or (L - k) * width <= min_sep:
            continue
        c = (lo + k) % n
        if np.isnan(kT) and np.isfinite(F[c]):
            continue
        n_a = int((u < np.radians(k * width)).sum())
        n_b = len(x_all) - n_a
        if min(n_a, n_b) < floor_frames:
            continue
        minor = ([(lo + i) % n for i in range(0, k)] if n_a < n_b
                 else [(lo + i) % n for i in range(k + 1, L + 1)])
        b = float(F[c] - F[minor].min())
        if np.isnan(kT) or b >= kT:
            key = (F[c], min(n_a, n_b))
            if best is None or key > best[0]:
                best = (key, c, b)
    if best is None:
        return None
    return {"bin": best[1], "barrier": best[2]}


class _Cutter:
    def __init__(self, Y, valid, names, *, mass_floor, kT_cut, kT_step, min_sep, n_bins, verbose):
        self.Y, self.valid, self.names = Y, valid, names
        self.N, self.M = Y.shape
        self.floor = float(mass_floor) * self.N
        self.kT_cut, self.kT_step, self.min_sep = float(kT_cut), float(kT_step), float(min_sep)
        self.n_bins = int(n_bins)
        self.edges = np.linspace(-np.pi, np.pi, self.n_bins + 1)
        self.half = np.pi / self.n_bins
        self.verbose = bool(verbose)
        self.nodes: List[Dict[str, Any]] = []
        self.log: List[str] = []

    def say(self, text: str) -> None:
        self.log.append(text)
        if self.verbose:
            print(text, flush=True)

    def new(self, rows, box, parent) -> int:
        self.nodes.append({"rows": rows, "box": box, "parent": parent, "children": [],
                           "cut": None})
        i = len(self.nodes) - 1
        if parent is not None:
            self.nodes[parent]["children"].append(i)
        return i

    def find(self, rows, kT, box):
        """The highest-barrier cut of the cell at E* = kT (nan: empty bins only), or None."""
        best = None
        for j in range(self.M):
            x = self.Y[rows[self.valid[rows, j]], j]
            if len(x) < 2 * self.floor:
                continue
            if box[j] is None:
                r = _profile_cuts(x, kT, self.floor, self.n_bins,
                                  min_separation_degrees=self.min_sep)
                if r is None:
                    continue
                b = float("inf") if np.isnan(kT) else _periodic_barrier(x, *r["bins"], self.edges)
                cut = ("periodic", r["bins"])
            else:
                r = _bounded_cut(self.Y[rows, j], x, kT, self.floor, *box[j], self.edges,
                                 self.min_sep)
                if r is None:
                    continue
                b = float("inf") if np.isnan(kT) else r["barrier"]
                cut = ("bounded", (r["bin"],))
            if best is None or b > best[0]:
                best = (b, j, cut)
        if best is None:
            return None
        b, j, (kind, bins) = best
        x = self.Y[rows, j]
        n, e, h = self.n_bins, self.edges, self.half
        if kind == "periodic":
            c1, c2 = bins
            in_a = np.mod(x - (e[c1] + h), 2 * np.pi) < np.mod(e[c2] - e[c1], 2 * np.pi)
            box_a, box_b = (c1, c2), (c2, c1)
        else:
            lo, hi = box[j]
            c = bins[0]
            in_a = np.mod(x - (e[lo] + h), 2 * np.pi) < np.mod(e[c] - e[lo], 2 * np.pi)
            box_a, box_b = (lo, c), (c, hi)
        if min(in_a.sum(), (~in_a).sum()) < self.floor:
            return None
        return {"j": j, "kind": kind, "barrier": b, "bins": bins, "A": rows[in_a],
                "B": rows[~in_a], "box_a": box_a, "box_b": box_b,
                "widths": tuple(float(((q[1] - q[0]) % n) * 360.0 / n) for q in (box_a, box_b))}

    def split(self, i, c, level) -> tuple:
        node = self.nodes[i]
        node["cut"] = {"coordinate": self.names[c["j"]], "kind": c["kind"], "level_kT": level,
                       "barrier_kT": None if np.isinf(c["barrier"]) else float(c["barrier"]),
                       "positions_degrees": [float(np.degrees(self.edges[b] + self.half))
                                             for b in c["bins"]],
                       "child_widths_degrees": list(c["widths"]),
                       "frames": [int(len(c["A"])), int(len(c["B"]))]}
        depth = 0
        p = i
        while self.nodes[p]["parent"] is not None:
            p, depth = self.nodes[p]["parent"], depth + 1
        what = "empty bins" if level is None else f"barrier {c['barrier']:.2f} kT at E*={level:.2f}"
        self.say(f"{'  ' * depth}{len(node['rows'])} frames: {c['kind']} cut on "
                 f"{self.names[c['j']]} ({what}) -> {len(c['A'])} + {len(c['B'])}")
        out = []
        for side in ("a", "b"):
            box = dict(node["box"])
            box[c["j"]] = c["box_" + side]
            out.append(self.new(c[side.upper()], box, i))
        return tuple(out)

    def fmax(self, rows) -> float:
        out = 0.0
        for j in range(self.M):
            x = self.Y[rows[self.valid[rows, j]], j]
            if len(x) == 0:
                continue
            cnt = np.histogram(x, self.edges)[0]
            nz = cnt[cnt > 0]
            out = max(out, float(np.log(nz.max() / nz.min())))
        return out

    def empty(self, i) -> None:
        c = self.find(self.nodes[i]["rows"], float("nan"), self.nodes[i]["box"])
        if c is None:
            self.anneal(i, float("inf"))
            return
        a, b = self.split(i, c, None)
        self.empty(a)
        self.empty(b)

    def anneal(self, i, E) -> None:
        rows = self.nodes[i]["rows"]
        E = min(E, self.fmax(rows))
        while E >= self.kT_cut - 1e-9:
            c = self.find(rows, E, self.nodes[i]["box"])
            if c is not None:
                a, b = self.split(i, c, float(E))
                self.anneal(a, E)
                self.anneal(b, E)
                return
            if E <= self.kT_cut + 1e-9:
                return
            E = max(E - self.kT_step, self.kT_cut)


def _invariant_coordinates(fit, z_min: float):
    """Cut coordinates, their names, and the per-frame validity mask (|z| >= z_min)."""
    K = int(fit.n_torsions_)
    names = [str(n) for n in (fit.names or [f"t{k + 1}" for k in range(K)])]
    short = [n.split()[0] for n in names]
    if not fit.symmetry:
        return (np.angle(np.exp(1j * np.asarray(fit.theta_, float))), short,
                np.ones(fit.theta_.shape, bool), [])
    d = fit.descriptor_
    C = np.asarray(fit.closed_theta_, float)
    sig = [list(op.sigma) for op in d.operations]
    support = [frozenset(c for c in range(K) if s[c] != c) for s in sig]
    groups = _coupling_groups(d, K)
    moved = set().union(*groups) if groups else set()
    quads = [tuple(int(a) for a in q) for q in d.quads[:K]]
    cols, cname, mags, record = [], [], [], []
    for k in range(K):
        if k not in moved:
            cols.append(np.angle(np.exp(1j * C[:, k])))
            cname.append(short[k])
            mags.append(np.ones(len(C)))
    for g in groups:
        sub = [s for s, sp in zip(sig, support) if sp <= g]
        g = sorted(g)
        record.append({"torsions": [names[k] for k in g], "operations": len(sub)})
        for c in g:
            m = len({s[c] for s in sub})
            z = np.mean([np.exp(1j * m * C[:, s[c]]) for s in sub], axis=0)
            cols.append(np.angle(z))
            cname.append(f"{m}*{short[c]}" if m > 1 else short[c])
            mags.append(np.abs(z))
        for a, b in itertools.combinations(g, 2):
            if {quads[a][1], quads[a][2]} == {quads[b][1], quads[b][2]}:
                continue                      # one central bond: the difference is geometry
            z = np.mean([np.exp(1j * (C[:, s[b]] - C[:, s[a]])) for s in sub], axis=0)
            cols.append(np.angle(z))
            cname.append(f"{short[b]}-{short[a]}")
            mags.append(np.abs(z))
    Y = np.column_stack(cols)
    valid = np.column_stack(mags) >= float(z_min)
    return Y, cname, valid, record


def cut_clustering(fit, *, kT_cut: float = DEFAULT_CUT_KT,
                   mass_floor: float = DEFAULT_CUT_MASS_FLOOR,
                   min_p2p: float = DEFAULT_MIN_P2P,
                   kT_step: float = DEFAULT_KT_STEP, n_bins: int = 72,
                   z_min: float = DEFAULT_Z_MIN, gauge=None, reference: int = 0,
                   verbose: bool = False) -> Dict[str, Any]:
    """Cut-route clustering of the frames of `fit`; see the module docstring for the rule.

    `fit` is a `THDBSCAN` (`t_hdbscan(...)`), fitted or not. On the symmetry-first route the cut
    coordinates are symmetry-invariant; with `symmetry=False` they are the torsions themselves.
    `kT_cut` (default 1 kT) is the lowest barrier accepted; `mass_floor` (default 0.1 %) is the
    smallest side a cut may leave, as a fraction of ALL frames; `min_p2p` (pi/3, radians) is the
    smallest cut-to-cut distance: between the two cuts on a periodic coordinate, and the narrowest
    piece a cut inside a bounded one may leave, measured along it. `gauge` and `reference` only set
    how `aligned_angles` (what `plot_hist` draws) are labelled: every frame in the labelling nearest
    frame `reference`, with `gauge` as in `THDBSCAN.align_all`.

    Returns `labels` (every frame, clusters 0.. by mass), `n_clusters`, `cluster_mass`,
    `coordinates` (cut coordinate names), `coupling_groups`, `tree` (one record per node: frames,
    parent, children, the cut made there, the box it lives in, and `cluster` for a leaf), `cuts`,
    `aligned_angles`, `torsion_names`, `quads`, `molecule`, `settings` and `log`. `plot_hist` takes
    the result as it takes a `hierarchical_clustering` one.
    """
    kT_cut = float(kT_cut)
    if not kT_cut > 0.0:
        raise ValueError("kT_cut must be positive (kT)")
    if not 0.0 < float(mass_floor) < 0.5:
        raise ValueError("mass_floor must be a fraction of all frames in (0, 0.5)")
    if not float(kT_step) > 0.0:
        raise ValueError("kT_step must be positive (kT)")
    if not 0.0 <= float(min_p2p) < np.pi:
        raise ValueError("min_p2p must be in [0, pi) radians")
    if int(n_bins) < 8:
        raise ValueError("n_bins must be at least 8")
    if not fit.weights_uniform_:
        raise NotImplementedError("frame weights are not supported; resample the ensemble first")
    Y, cname, valid, groups = _invariant_coordinates(fit, z_min)
    H = _Cutter(Y, valid, cname, mass_floor=mass_floor, kT_cut=kT_cut, kT_step=kT_step,
                min_sep=round(float(np.degrees(min_p2p)), 9),   # pi/3 is 60.0, not 59.99..
                n_bins=n_bins, verbose=verbose)
    root = H.new(np.arange(H.N), {j: None for j in range(H.M)}, None)
    H.empty(root)

    leaves = sorted((i for i, n in enumerate(H.nodes) if not n["children"]),
                    key=lambda i: (-len(H.nodes[i]["rows"]), i))
    labels = np.full(H.N, -1, dtype=np.int64)
    for c, i in enumerate(leaves):
        labels[H.nodes[i]["rows"]] = c
        H.nodes[i]["cluster"] = c
    assert (labels >= 0).all(), "a frame in no leaf"

    K = int(fit.n_torsions_)
    if fit.symmetry:
        d = fit.descriptor_
        C = fit.closed_theta_
        g = None if gauge is None else d._gauge_column(gauge, C[int(reference)])
        aligned = d.align_to_reference(C, C[int(reference)], gauge=g)["angles"][:, :K]
        aligned = np.angle(np.exp(1j * aligned))
        quads = [tuple(int(a) for a in q) for q in d.quads[:K]]
        molecule = d.molecule
    else:
        aligned, quads, molecule = np.angle(np.exp(1j * np.asarray(fit.theta_, float))), None, None
    half = np.pi / H.n_bins

    def box_deg(box):
        return {cname[j]: None if b is None else
                [float(np.degrees(H.edges[b[0]] + half)), float(np.degrees(H.edges[b[1]] + half))]
                for j, b in box.items()}

    tree = [{"node": i, "parent": n["parent"], "children": list(n["children"]),
             "frames": int(len(n["rows"])), "cut": n["cut"], "box_degrees": box_deg(n["box"]),
             **({"cluster": n["cluster"]} if "cluster" in n else {})}
            for i, n in enumerate(H.nodes)]
    names = [str(n) for n in (fit.names or [f"t{k + 1}" for k in range(K)])]
    return {
        "labels": labels, "n_clusters": len(leaves),
        "cluster_mass": {c: float((labels == c).mean()) for c in range(len(leaves))},
        "coordinates": cname, "coupling_groups": groups, "tree": tree,
        "cuts": [dict(n["cut"], node=i) for i, n in enumerate(H.nodes) if n["cut"]],
        "aligned_angles": aligned, "torsion_names": names, "quads": quads, "molecule": molecule,
        "log": H.log,
        "settings": {"kT_cut": kT_cut, "mass_floor": float(mass_floor),
                     "floor_frames": H.floor, "min_p2p": float(min_p2p),
                     "kT_step": H.kT_step, "n_bins": H.n_bins, "z_min": float(z_min),
                     "reference": int(reference), "gauge": gauge},
    }
