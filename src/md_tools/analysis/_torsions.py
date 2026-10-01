"""The periodic torsion metric, the HDBSCAN fit, and the vote classifier.

EXTRACTED, NOT WRITTEN HERE. These are seven functions and their transitive dependencies, lifted
from the hpREST2 project's `torsion_clustering.py` (~1600 lines) with that session's agreement.
What was LEFT BEHIND is the point of extracting rather than vendoring the module: the redundancy
matrix, the mutual-information surrogates and the sensitivity scan stay there, so nothing in this
file reaches a mutual-information implementation. `md_tools.analysis.torsional_mi` is a separate
module and neither imports the other -- verified by the extraction itself, which computed the
transitive closure of the seven and found the MI layer outside it.

PRIVATE ON PURPOSE. The public surface is `md_tools.analysis.t_hdbscan`; these are the parts it
is built from. They are documented because a reader following a result back has to be able to see
what the metric does, not because callers should reach them.

sklearn is imported INSIDE the functions that need it, never at module import, so
`import md_tools.analysis` costs nothing in an environment without the `analysis` extra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: One turn. Angles live on a circle and every distance below respects that.
TWO_PI = 2.0 * np.pi


def wrap(a):
    """Wrap angles to ``[-pi, pi)``."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % TWO_PI - np.pi


#: Angle-unit spellings this module accepts. There is no default and no sniffing: a caller that
#: does not say which it means is refused, because degrees read as radians is a silent 57x error
#: that still clusters and still looks plausible.
UNITS = ("radians", "degrees")


#: Resultant length below which a circular mean carries no direction and is reported as undefined.
#: 0.1 is the campaign default; at R = 0.1 the mean angle's own circular standard error over even a
#: thousand frames is tens of degrees, so quoting it as a location would be a fiction.
MIN_RESULTANT_FOR_MEAN = 0.1


#: Smallest cluster mass, as a FRACTION of the ensemble, that is allowed to be a cluster. 1% is the
#: campaign default (user decision, 2026-09-29). A fraction rather than a frame count is what keeps
#: the question the same at every dataset size -- the older absolute default capped at 500 frames,
#: which silently loosened the floor to 0.17% at 300000 frames and 0.05% at a million.
#:
#: IT IS APPLIED BY SETTING `min_cluster_size` AND PINNING `min_samples` SEPARATELY, AND THE SECOND
#: HALF IS NOT OPTIONAL. sklearn defaults `min_samples` to `min_cluster_size`, and `min_samples` sets
#: the core distances -- so raising the mass floor alone changes the DENSITY ESTIMATE rather than
#: filtering clusters out of a fixed one. Measured on 15000 hot RGDfV frames: at a 5% floor with
#: `min_samples` left to follow, all 15000 frames became noise and 0 clusters survived; with
#: `min_samples` pinned at 75 the same floor returned exactly the two clusters above 5% and left them
#: unchanged (ARI 0.993 against the unfiltered fit).
MIN_CLUSTER_FRACTION_DEFAULT = 0.01


#: `min_samples` used when a mass fraction sets `min_cluster_size`, AS A FRACTION OF n.
#: Chosen 2026-09-30: tying it to the mass floor makes the whole procedure scale-invariant, so the
#: question asked is identical at every dataset size, and it removes the `min_cluster_size` <
#: `min_samples` regime in which a 1% floor is not enforceable at all (at n=1024 a 1% floor is 10
#: frames against a k of 25, and a planted, well-separated 1% cluster was recovered 0/5 times).
#: Tied, the same cluster is recovered 5/5 at n=1024 and the seeds agree from n=4096 (RGDfV) or
#: n=2048 (ALA) instead of never agreeing at all.
#:
#: THE COST IS MEASURED AND REAL. `min_samples` is the k of a k-nearest-neighbour density estimate,
#: so tying it to n means the ESTIMATOR changes with n and a convergence ladder no longer compares
#: like with like. On ALA it also smooths away a marginal state: against a direct fit the tied
#: setting plateaus at ARI 0.978 while a pinned k=25 reaches 0.990, and it absorbs a 1.44% basin
#: at (phi,psi) = (-141, +6) into alpha_R. Use `min_samples=MIN_SAMPLES_COUNT_LEGACY` (25) to
#: recover the pinned behaviour; both are recorded in the result's notes.
MIN_SAMPLES_FRACTION_DEFAULT = 0.01


#: Default abstention threshold for :func:`classify_to_clusters`. Chosen 2026-09-30 to pair with the
#: tied `min_samples` above, which is what makes it necessary: the coarser density estimate orphans
#: a basin's sparse rim, the classifier then commits those frames by a bare majority, and on ALA
#: that put 459 cold-cell frames into a basin 130 deg away in phi. At 0.9 that drops to 1.
#:
#: IT IS A BOUNDARY DETECTOR, NOT A NOISE DETECTOR, and the difference is measured: against a direct
#: fit's own noise it recalls only 6.7%, because a frame can sit in a sparse region and still have
#: all k neighbours agree. Most of what it abstains on is cluster INTERFACES. On a pinned k=25 fit,
#: where nothing is orphaned, it therefore costs accuracy -- 2.904% -> 2.834% on the ALA cold cell
#: against a direct count of 2.905%. Set 0.0 to commit every frame.
MIN_VOTE_MARGIN_DEFAULT = 0.9


N_SEED_DEFAULT = 5


#: Clusters up to this size get the O(n^2) exact medoid when it is asked for. Above it the exact
#: variant is refused rather than silently approximated.
EXACT_MEDOID_MAX = 5000


#: HDBSCAN's `brute` path materialises the full n x n distance matrix. Measured in this
#: environment at n = 20000: 6400 MB for `brute` against 32 MB for `auto`, which is 1.0% of the
#: dense n^2 and flat in the torsion count from 2 to 60 torsions. `auto` is therefore the only
#: default, and `brute` is refused unless the caller lifts the guard knowingly.
SUBQUADRATIC_ALGORITHMS = ("auto", "kd_tree", "ball_tree")


@dataclass(frozen=True)
class TorsionSpec:
    """One torsion: a name and the four atom indices that define it.

    ATOM INDEX CONVENTION. Indices are ZERO-BASED and refer to the topology the trajectory is read
    with -- the same convention as `configs`/`src/<system>/config/cv.yaml` and
    `mdtraj.compute_dihedrals`, and the same indices `hprest2.md.dihedrals` takes. They are NOT
    Amber one-based masks. The quadruplet (i, j, k, l) defines the dihedral about the j-k bond,
    measured from the i-j-k plane to the j-k-l plane, sign by the right-hand rule about j->k.
    """

    name: str
    atom_indices: Optional[Tuple[int, int, int, int]] = None
    multiplicity: int = 1
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.atom_indices is not None and len(self.atom_indices) != 4:
            raise ValueError(f"torsion {self.name!r}: need 4 atom indices, got "
                             f"{len(self.atom_indices)}")
        if int(self.multiplicity) < 1:
            raise ValueError(f"torsion {self.name!r}: multiplicity must be >= 1")
        if not (self.weight > 0 and math.isfinite(self.weight)):
            raise ValueError(f"torsion {self.name!r}: weight must be finite and positive")


def validate_angles(theta, units: str, *, n_torsions: Optional[int] = None) -> np.ndarray:
    """Shape, finiteness and unit checks; returns a float64 array in RADIANS.

    NOTHING IS DISCARDED. A non-finite frame raises and names the offending rows, because dropping
    it would change the denominator of every population this module later reports while leaving no
    trace in the record. The caller decides what a bad frame means; this function refuses to guess.
    """
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS}, got {units!r}. There is no default: "
                         f"degrees interpreted as radians is a silent 57x error.")
    a = np.asarray(theta, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"angles must be 2-D (n_frames, n_torsions), got shape {a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"angles are empty: shape {a.shape}")
    if n_torsions is not None and a.shape[1] != n_torsions:
        raise ValueError(f"angles have {a.shape[1]} torsions but {n_torsions} names/specs given")
    bad = ~np.isfinite(a)
    if bad.any():
        rows = np.flatnonzero(bad.any(axis=1))
        shown = ", ".join(str(int(r)) for r in rows[:10])
        raise ValueError(f"{rows.size} frame(s) hold non-finite angles (rows {shown}"
                         f"{', ...' if rows.size > 10 else ''}). Frames are never silently "
                         f"dropped: remove them upstream and say so, or repair the input.")
    if units == "degrees":
        a = np.deg2rad(a)
    return a


def validate_metric_weights(weights, n_torsions: int) -> np.ndarray:
    """Per-torsion metric weights: positive, finite, one per torsion."""
    if weights is None:
        return np.ones(n_torsions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n_torsions:
        raise ValueError(f"{w.size} metric weights for {n_torsions} torsions")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("metric weights must all be finite and strictly positive")
    return w


def validate_frame_weights(weights, n_frames: int) -> Optional[np.ndarray]:
    """Statistical frame weights: non-negative, finite, not all zero. NOT metric weights."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n_frames:
        raise ValueError(f"{w.size} frame weights for {n_frames} frames")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("frame weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("frame weights sum to zero")
    return w


def validate_multiplicities(specs: Sequence[TorsionSpec], justification: Optional[str]) -> None:
    """A multiplicity other than 1 identifies angles differing by 2*pi/k and must be argued for.

    Multiplicity k makes the feature `[cos(k*theta), sin(k*theta)]`, so the metric becomes
    `2*sum_j w_j*[1 - cos(k_j*(theta_j - phi_j))]` and two conformers related by a 2*pi/k rotation
    become THE SAME POINT. That is right for a genuinely symmetric rotor (a phenyl ring flip,
    k = 2) and wrong for a backbone torsion, where it would merge distinct basins. So it is never
    a default and never inferred: the caller must both set it and say why.
    """
    ks = {s.name: int(s.multiplicity) for s in specs if int(s.multiplicity) != 1}
    if ks and not (justification and str(justification).strip()):
        raise ValueError(f"multiplicity != 1 on {sorted(ks)} identifies conformers related by a "
                         f"2*pi/k rotation. Pass multiplicity_justification naming the molecular "
                         f"symmetry that makes that true, or leave multiplicity at 1.")


def embed(theta_rad: np.ndarray, metric_weights: np.ndarray,
          multiplicities: Optional[Sequence[int]] = None) -> np.ndarray:
    """`x_j = sqrt(w_j) * [cos(k_j*theta_j), sin(k_j*theta_j)]`, columns interleaved per torsion.

    Every embedded point has the SAME norm, `|x|^2 = sum_j w_j`, because each torsion contributes a
    point on a circle of radius `sqrt(w_j)`. That fact is used by :func:`representative_indices`.
    """
    n, m = theta_rad.shape
    k = (np.ones(m, dtype=np.int64) if multiplicities is None
         else np.asarray(multiplicities, dtype=np.int64).ravel())
    if k.size != m:
        raise ValueError(f"{k.size} multiplicities for {m} torsions")
    s = np.sqrt(metric_weights)[None, :]
    a = theta_rad * k[None, :].astype(np.float64)
    X = np.empty((n, 2 * m), dtype=np.float64)
    X[:, 0::2] = np.cos(a) * s
    X[:, 1::2] = np.sin(a) * s
    return X


def circular_mean_resultant(theta_rad: np.ndarray, frame_weights: Optional[np.ndarray] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column circular mean (radians, wrapped) and resultant length R in [0, 1].

    R is the length of the mean unit vector: 1 for a delta, 0 for anything uniform or for two
    equal lobes a half-turn apart. A LOW R DOES NOT MEAN "BROAD": it also happens when the
    distribution is bimodal and symmetric, in which case the mean angle points nowhere real. The
    caller gets both numbers so it can refuse to quote the mean; :func:`cluster_torsions` does
    exactly that at :data:`MIN_RESULTANT_FOR_MEAN`.
    """
    a = np.asarray(theta_rad, dtype=np.float64)
    # THE TWO PATHS USED TO DISAGREE ON 1-D INPUT (fixed 2026-09-30). The unweighted branch took
    # `z.mean(axis=0)` and handled a 1-D series correctly; the weighted branch took
    # `w[:, None] * z`, which on a 1-D series of length n broadcasts to n x n -- silently
    # allocating 288 MB for n = 6000 and returning n values where one was meant. Normalising to
    # 2-D up front and squeezing back makes both branches agree for either input shape.
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    z = np.exp(1j * a)
    if frame_weights is None:
        mz = z.mean(axis=0)
    else:
        w = np.asarray(frame_weights, dtype=np.float64).ravel()
        if w.size != a.shape[0]:
            raise ValueError(f"{w.size} frame weights for {a.shape[0]} frames")
        mz = (w[:, None] * z).sum(axis=0) / w.sum()
    mu, R = wrap(np.angle(mz)), np.abs(mz)
    return (mu[0], R[0]) if squeeze else (mu, R)


def representative_indices(X: np.ndarray, labels: np.ndarray, *, method: str = "chord_medoid",
                           exact_max: int = EXACT_MEDOID_MAX) -> Dict[int, int]:
    """One ACTUAL FRAME per cluster. Never an average of coordinates or of angles.

    Averaging is not an option here and the reason is geometric, not stylistic: the mean of two
    Cartesian structures is not a physical structure, and the circular mean of a bimodal torsion
    points at the barrier between its two basins -- the one place the molecule is never found. So
    the representative is always an index into the input.

    `chord_medoid` (default, EXACT and O(n_C * 2m)). Minimises the sum of SQUARED chord distances
    to the cluster's other members. This is exact, not an approximation, and the reason is that
    every embedded point has the same norm `W = sum_j w_j`:

        sum_i d^2(x, x_i) = 2*n_C*W - 2 * x . S,    S = sum_i x_i

    so the minimiser over the cluster's own members is the member maximising `x . S`. Expanded,
    `x . S = n_C * sum_j w_j * R_j * cos(theta_j - mu_j)`: the frame best aligned with the
    per-torsion circular means, each torsion counting by its own resultant length, so a torsion
    that is diffuse inside the cluster automatically counts less. No pairwise distances are formed.

    `exact_medoid` minimises the sum of UNSQUARED distances -- the textbook medoid, more robust to
    a straggler and NOT the same point in general. It needs the cluster's pairwise distances, so it
    is O(n_C^2) and is REFUSED above `exact_max` rather than quietly degraded.
    """
    if method not in ("chord_medoid", "exact_medoid"):
        raise ValueError(f"unknown representative method {method!r}")
    out: Dict[int, int] = {}
    for lab in sorted(set(int(v) for v in labels)):
        if lab < 0:
            continue
        members = np.flatnonzero(labels == lab)
        Xc = X[members]
        if method == "chord_medoid":
            S = Xc.sum(axis=0)
            out[lab] = int(members[int(np.argmax(Xc @ S))])
        else:
            if members.size > exact_max:
                raise ValueError(
                    f"cluster {lab} holds {members.size} frames; exact_medoid is O(n^2) and is "
                    f"capped at {exact_max}. Use method='chord_medoid', which is exact for the "
                    f"squared-chord medoid and linear in the cluster size.")
            # |x - y|^2 = 2W - 2 x.y for equal-norm points, so one Gram matrix gives every distance
            G = Xc @ Xc.T
            W = float(Xc[0] @ Xc[0])
            D = np.sqrt(np.maximum(0.0, 2.0 * W - 2.0 * G))
            out[lab] = int(members[int(np.argmin(D.sum(axis=1)))])
    return out


@dataclass
class ClusterResult:
    """Per-frame assignment plus per-cluster description, with everything needed to reproduce it."""

    labels: np.ndarray
    membership_strength: np.ndarray
    names: List[str]
    clusters: List[dict]
    noise_fraction: float
    noise_fraction_weighted: Optional[float]
    n_clusters: int
    traj_ids: np.ndarray
    frame_ids: np.ndarray
    representatives: Dict[int, int]
    settings: dict
    provenance: dict
    warnings: List[str] = field(default_factory=list)

    @property
    def all_noise(self) -> bool:
        return self.n_clusters == 0

    def to_json(self) -> dict:
        """JSON-ready. Per-frame arrays are lists; they are the point of the record, not a detail."""
        return {
            "names": list(self.names),
            "n_frames": int(self.labels.size),
            "n_clusters": int(self.n_clusters),
            "noise_fraction": float(self.noise_fraction),
            "noise_fraction_weighted": (None if self.noise_fraction_weighted is None
                                        else float(self.noise_fraction_weighted)),
            "clusters": self.clusters,
            "representatives": {str(k): int(v) for k, v in self.representatives.items()},
            "labels": [int(v) for v in self.labels],
            "membership_strength": [float(v) for v in self.membership_strength],
            "traj_ids": [int(v) for v in self.traj_ids],
            "frame_ids": [int(v) for v in self.frame_ids],
            "settings": self.settings,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
        }


def _backend() -> dict:
    """Which HDBSCAN is doing the work, and its version. Recorded, never assumed."""
    try:
        import sklearn
        from sklearn.cluster import HDBSCAN  # noqa: F401
        return {"backend": "sklearn.cluster.HDBSCAN", "version": sklearn.__version__,
                "note": "the maintained implementation in this project's analysis environment; "
                        "the standalone `hdbscan` package is not installed here"}
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "no maintained HDBSCAN available. sklearn >= 1.3 ships sklearn.cluster.HDBSCAN; "
            f"importing it failed with {exc!r}") from exc


def cluster_torsions(theta, *, units: str, names: Optional[Sequence[str]] = None,
                     specs: Optional[Sequence[TorsionSpec]] = None,
                     metric_weights=None, frame_weights=None,
                     traj_ids=None, frame_ids=None,
                     min_cluster_size: Optional[int] = None,
                     min_cluster_fraction: Optional[float] = None,
                     mass_floor_post_hoc: float = 0.0,
                     min_samples: Optional[int] = None,
                     min_samples_fraction: Optional[float] = None,
                     cluster_selection_method: str = "eom",
                     cluster_selection_epsilon: float = 0.0,
                     allow_single_cluster: bool = False,
                     algorithm: str = "auto",
                     allow_quadratic_memory: bool = False,
                     representative_method: str = "chord_medoid",
                     multiplicity_justification: Optional[str] = None,
                     min_resultant_for_mean: float = MIN_RESULTANT_FOR_MEAN,
                     n_jobs: Optional[int] = None,
                     seed: Optional[int] = None) -> ClusterResult:
    """Cluster an ensemble jointly over all supplied torsions. See the module docstring first.

    HDBSCAN IS RUN ON THE EMBEDDED FEATURES WITH PLAIN EUCLIDEAN DISTANCE, never on a precomputed
    matrix: `metric='precomputed'` would force the dense `n x n` array this module exists to
    avoid, and the embedding makes it unnecessary because Euclidean distance on the features IS the
    chord metric.

    NOISE IS KEPT AS -1. HDBSCAN's noise label means "this frame is not in a region dense enough
    to call a cluster", which for an MD ensemble is a real and common answer -- transition-region
    frames, brief excursions, the tail of a basin. Reassigning them to the nearest cluster would
    invent membership and inflate every population. They are carried through and counted.
    """
    if specs is not None:
        if names is not None:
            raise ValueError("pass names or specs, not both")
        names = [s.name for s in specs]
        if metric_weights is None:
            mw = np.array([float(s.weight) for s in specs], dtype=np.float64)
            metric_weights = None if np.allclose(mw, 1.0) else mw
        mult = [int(s.multiplicity) for s in specs]
        validate_multiplicities(specs, multiplicity_justification)
    else:
        mult = None

    n_tors = None if names is None else len(list(names))
    th = validate_angles(theta, units, n_torsions=n_tors)
    n, m = th.shape
    names = [str(x) for x in (names if names is not None else [f"tor{j}" for j in range(m)])]
    w_metric = validate_metric_weights(metric_weights, m)
    w_frame = validate_frame_weights(frame_weights, n)
    if mult is None:
        mult = [1] * m

    tids = (np.zeros(n, dtype=np.int64) if traj_ids is None
            else np.asarray(traj_ids, dtype=np.int64).ravel())
    fids = (np.arange(n, dtype=np.int64) if frame_ids is None
            else np.asarray(frame_ids, dtype=np.int64).ravel())
    if tids.size != n or fids.size != n:
        raise ValueError(f"traj_ids/frame_ids must have {n} entries, got {tids.size}/{fids.size}")

    if algorithm not in SUBQUADRATIC_ALGORITHMS and not allow_quadratic_memory:
        raise ValueError(
            f"algorithm={algorithm!r} materialises the full n x n distance matrix "
            f"({n * n * 8 / 1e6:.0f} MB at n={n}). Use one of {SUBQUADRATIC_ALGORITHMS}, or pass "
            f"allow_quadratic_memory=True if you mean it.")

    if min_cluster_size is not None and min_cluster_fraction is not None:
        raise ValueError("pass min_cluster_size or min_cluster_fraction, not both")
    if not (0.0 <= float(mass_floor_post_hoc) < 1.0):
        raise ValueError("mass_floor_post_hoc must be in [0, 1)")

    frac_used = None
    if min_cluster_size is not None:
        mcs, mcs_src = int(min_cluster_size), "caller (absolute)"
    else:
        frac_used = (MIN_CLUSTER_FRACTION_DEFAULT if min_cluster_fraction is None
                     else float(min_cluster_fraction))
        if not (0.0 < frac_used < 1.0):
            raise ValueError("min_cluster_fraction must be in (0, 1)")
        mcs = int(max(2, math.ceil(frac_used * n)))
        mcs_src = (f"min_cluster_fraction={frac_used} of n={n}"
                   + ("" if min_cluster_fraction is not None else " (module default)"))

    # `min_samples` is resolved INDEPENDENTLY of how `min_cluster_size` was chosen. Tying it to a
    # fraction of n keeps the 1% floor enforceable at every n (mcs >= min_samples holds by
    # construction), which is the whole reason for the default; it is still its own knob, so
    # raising the mass floor alone cannot silently coarsen the density estimate -- measured, a 5%
    # floor with min_samples following returned 0 clusters and 100% noise on an ensemble that has
    # two clusters above 5%.
    if min_samples is not None:
        if min_samples_fraction is not None:
            raise ValueError("pass min_samples or min_samples_fraction, not both")
        ms_src = f"min_samples={int(min_samples)} given explicitly"
    else:
        ms_frac = (MIN_SAMPLES_FRACTION_DEFAULT if min_samples_fraction is None
                   else float(min_samples_fraction))
        if not (0.0 < ms_frac < 1.0):
            raise ValueError("min_samples_fraction must be in (0, 1)")
        min_samples = int(max(2, round(ms_frac * n)))
        ms_src = (f"min_samples_fraction={ms_frac} of n={n}"
                  + ("" if min_samples_fraction is not None else " (module default)"))
    if mcs < 2:
        raise ValueError("min_cluster_size must be >= 2")
    if mcs > n:
        raise ValueError(f"min_cluster_size={mcs} exceeds the {n} frames supplied")

    backend = _backend()
    from sklearn.cluster import HDBSCAN

    X = embed(th, w_metric, mult)
    est = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples,
                  cluster_selection_epsilon=float(cluster_selection_epsilon),
                  cluster_selection_method=cluster_selection_method,
                  allow_single_cluster=bool(allow_single_cluster),
                  metric="euclidean", algorithm=algorithm, n_jobs=n_jobs, copy=False)
    est.fit(X)
    labels = np.asarray(est.labels_, dtype=np.int64)
    strength = np.asarray(getattr(est, "probabilities_", np.full(n, np.nan)), dtype=np.float64)

    warn: List[str] = []
    # ---- post-hoc mass floor: a REPORTING choice on a FIXED fit, not a change to the fit -------
    # Distinct from min_cluster_fraction, which stops HDBSCAN promoting the small cluster at all and
    # lets excess-of-mass choose a different partition instead. This one leaves the partition alone
    # and moves sub-floor clusters into the noise label. The frames are NOT noise in any physical
    # sense -- they were assigned, and this relabels them -- so what was removed is recorded.
    removed_post_hoc: List[dict] = []
    if mass_floor_post_hoc > 0.0:
        for lab in sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0):
            sel = labels == lab
            f = float(sel.sum()) / n
            if f < mass_floor_post_hoc:
                removed_post_hoc.append({"label": lab, "n_frames": int(sel.sum()),
                                         "fraction_frames": f})
                labels[sel] = -1
        if removed_post_hoc:
            tot = sum(d["fraction_frames"] for d in removed_post_hoc)
            warn.append(
                f"POST-HOC MASS FLOOR {mass_floor_post_hoc}: {len(removed_post_hoc)} cluster(s) "
                f"holding {tot * 100:.2f}% of frames were RELABELLED to noise "
                f"({[d['label'] for d in removed_post_hoc]}). The fit was not changed and those "
                f"frames are not noise in any physical sense -- they were assigned and then "
                f"removed from the report. Use min_cluster_fraction instead to stop HDBSCAN "
                f"promoting them in the first place, which lets the selection choose a different "
                f"partition rather than discarding one.")
            # relabel survivors to a contiguous 0..k-1 so the record has no gaps
            surv = sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0)
            remap = {old: new for new, old in enumerate(surv)}
            labels = np.array([remap.get(int(v), -1) for v in labels], dtype=np.int64)
    ids = sorted(int(v) for v in set(labels.tolist()) if int(v) >= 0)
    n_clusters = len(ids)
    if n_clusters == 0:
        warn.append(
            f"ALL {n} FRAMES ARE NOISE: no region reached the density required by "
            f"min_cluster_size={mcs}. This is a real outcome, not a failure -- it says the "
            f"ensemble has no basin of that size at this density, commonly because "
            f"min_cluster_size is too large for the data or the ensemble is genuinely diffuse. "
            f"Nothing downstream should treat the empty cluster list as an error.")
    elif n_clusters == 1:
        tail = ("permitted and this is that answer" if allow_single_cluster else
                "reported as HDBSCAN found it. Note that with allow_single_cluster=False the "
                "excess-of-mass selection cannot return the root of the condensed tree, so a "
                "genuinely unimodal ensemble tends to come back as ALL-NOISE rather than as one "
                "cluster; set allow_single_cluster=True if one basin is a physically expected "
                "outcome")
        warn.append(f"ONE cluster and {float(np.mean(labels < 0)) * 100:.1f}% noise. With "
                    f"allow_single_cluster={allow_single_cluster}, a single-cluster answer is "
                    f"{tail}.")

    tot_w = None if w_frame is None else float(w_frame.sum())
    clusters = []
    for lab in ids:
        sel = labels == lab
        n_c = int(sel.sum())
        mu, R = circular_mean_resultant(th[sel])
        per_tors = []
        for j in range(m):
            ok = bool(R[j] >= float(min_resultant_for_mean))
            per_tors.append({
                "name": names[j],
                "circular_mean_rad": (float(mu[j]) if ok else None),
                "circular_mean_deg": (float(np.rad2deg(mu[j])) if ok else None),
                "resultant_length": float(R[j]),
                "circular_mean_defined": ok,
                "note": (None if ok else
                         f"resultant length {R[j]:.3f} < {float(min_resultant_for_mean)}: the "
                         f"circular mean has no direction here (diffuse, or bimodal with lobes "
                         f"about a half-turn apart) and is reported as null rather than quoted"),
            })
        rec = {
            "label": lab,
            "n_frames": n_c,
            "fraction_frames": float(n_c / n),
            "mean_membership_strength": float(np.nanmean(strength[sel])),
            "torsions": per_tors,
        }
        if w_frame is not None:
            rec["weighted_population"] = float(w_frame[sel].sum() / tot_w)
        clusters.append(rec)

    noise_frac = float(np.mean(labels < 0))
    noise_frac_w = (None if w_frame is None
                    else float(w_frame[labels < 0].sum() / tot_w))

    reps = representative_indices(X, labels, method=representative_method)
    for rec in clusters:
        i = reps.get(rec["label"])
        rec["representative"] = (None if i is None else {
            "index": int(i), "traj_id": int(tids[i]), "frame_id": int(fids[i]),
            "method": representative_method,
            "angles_rad": [float(v) for v in th[i]],
            "angles_deg": [float(v) for v in np.rad2deg(th[i])],
            "note": "an ACTUAL frame of the input, chosen by medoid; never an average of "
                    "coordinates or of angles",
        })

    settings = {
        "units_in": units,
        "angles_stored_as": "radians",
        "min_cluster_size": mcs,
        "min_cluster_size_source": mcs_src,
        "min_cluster_fraction": frac_used,
        "min_cluster_fraction_effective": (float(mcs) / n),
        "mass_floor_post_hoc": float(mass_floor_post_hoc),
        "removed_by_post_hoc_mass_floor": removed_post_hoc,
        "min_samples": (None if min_samples is None else int(min_samples)),
        "min_samples_source": ms_src,
        "min_samples_note": "None lets the backend use min_cluster_size, which COUPLES the mass "
                            "floor to the density estimate: raising the floor then changes the "
                            "estimate rather than filtering a fixed one. When a mass fraction sets "
                            "min_cluster_size this module pins min_samples separately for that "
                            "reason. Larger values make the density estimate more conservative and "
                            "grow the noise set.",
        "cluster_selection_method": cluster_selection_method,
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
        "allow_single_cluster": bool(allow_single_cluster),
        "algorithm": algorithm,
        "metric": "euclidean on the cos/sin embedding == chord distance on the product of circles",
        "metric_weights": [float(v) for v in w_metric],
        "multiplicities": [int(v) for v in mult],
        "multiplicity_justification": multiplicity_justification,
        "representative_method": representative_method,
        "min_resultant_for_mean": float(min_resultant_for_mean),
        "seed": (None if seed is None else int(seed)),
        "seed_note": "HDBSCAN as used here is deterministic; the seed is recorded for the "
                     "sensitivity scan and any caller-side subsampling, not for the fit",
        "torsion_definitions": ([{"name": s.name,
                                  "atom_indices": (None if s.atom_indices is None
                                                   else list(s.atom_indices)),
                                  "multiplicity": int(s.multiplicity),
                                  "weight": float(s.weight)} for s in specs]
                                if specs is not None else None),
        "atom_index_convention": "zero-based, into the topology the trajectory was read with; "
                                 "quadruplet (i,j,k,l) is the dihedral about the j-k bond",
    }
    provenance = {
        "hdbscan": backend,
        "numpy": np.__version__,
        "frame_weights_supplied": w_frame is not None,
        "frame_weights_role": (
            "population bookkeeping ONLY. The HDBSCAN fit is of the UNWEIGHTED empirical density "
            "of the frames given, so weighting the populations does NOT make this a weighted "
            "target-ensemble density estimate: the cluster BOUNDARIES remain those of the sampled "
            "density. Statistical frame weights are separate from the per-torsion metric weights."),
        "noise_policy": "HDBSCAN noise is retained as label -1 and included in the population "
                        "normalisation; it is never reassigned to a cluster",
        "membership_strength_meaning": (
            "the backend's per-frame cluster-membership score. It is an ALGORITHMIC score from the "
            "condensed tree, NOT a thermodynamic probability, NOT a Boltzmann weight, and not "
            "comparable between fits with different parameters."),
        "not_metastable_states": (
            "these are density clusters in torsion space. No time ordering entered the metric and "
            "no dynamical quantity was estimated, so they are NOT kinetically metastable states "
            "and must not be reported as such without a separate dynamical analysis."),
    }
    return ClusterResult(labels=labels, membership_strength=strength, names=names,
                         clusters=clusters, noise_fraction=noise_frac,
                         noise_fraction_weighted=noise_frac_w, n_clusters=n_clusters,
                         traj_ids=tids, frame_ids=fids, representatives=reps,
                         settings=settings, provenance=provenance, warnings=warn)


# ------------------------------------------------- resampling a target ensemble for an unweighted fit
RESAMPLE_METHODS = ("systematic", "rejection", "multinomial")


def weighted_resample(frame_weights, *, size: Optional[int] = None,
                      method: str = "systematic", seed: int = 0,
                      allow_duplicates: bool = False) -> Dict[str, Any]:
    """Indices of an UNWEIGHTED sample drawn from the weighted target ensemble.

    WHY THIS EXISTS. HDBSCAN cannot take frame weights: its density is the CORE DISTANCE, the
    distance to the k-th nearest neighbour, which is a rank statistic over neighbour counts, so a
    frame of weight 10 is still one neighbour and `sklearn.cluster.HDBSCAN` has no `sample_weight`.
    Weighting the POPULATIONS afterwards leaves the cluster BOUNDARIES those of the sampled
    density. On a biased ensemble that is not a detail -- `phi_VAL0`'s arc holds 14.36 % of the
    as-sampled mass against 0.79 % reweighted. Drawing an unweighted sample FROM the target and
    fitting that gives boundaries of the target ensemble.

    METHODS. `'systematic'` (default) is Madow pi-ps sampling WITHOUT replacement: inclusion
    probability `pi_i = m*w_i/sum(w)`, capped at 1 with iterative renormalisation so that a frame
    too heavy to sample probabilistically becomes a CERTAINTY UNIT rather than silently breaking
    the probabilities. It returns exactly `size` DISTINCT frames and has lower variance than
    drawing independently. `'rejection'` keeps frame `i` with probability `w_i/max(w)`: exactly
    unbiased, distinct frames, but the size is random and collapses when the weights are skewed.

    `'multinomial'` (with replacement) IS REFUSED unless `allow_duplicates=True`, and should stay
    refused for a density fit: duplicate frames sit at distance exactly 0, so the k-th nearest
    neighbour distance is 0 and the density estimate is destroyed rather than degraded. Duplicates
    also inflate Kish ESS without adding information -- see
    `hprest2.efficiency.duplicate_frames_inflate_kish`.

    DEFAULT SIZE IS THE KISH ESS, floored. Resampling cannot manufacture information: asking for
    more frames than the ESS supports returns a set whose apparent sample size overstates what the
    reweighting can carry. The chosen and the ESS-implied sizes are both reported so a caller who
    overrides it does so visibly.
    """
    w = np.asarray(frame_weights, dtype=np.float64).ravel()
    n = w.size
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("frame weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("frame weights sum to zero")
    if method not in RESAMPLE_METHODS:
        raise ValueError(f"method must be one of {RESAMPLE_METHODS}, not {method!r}")
    p = w / w.sum()
    ess = float(1.0 / np.sum(p * p))
    m = int(size) if size is not None else int(np.floor(ess))
    m = max(1, min(m, n))
    rng = np.random.default_rng(seed)
    warnings: List[str] = []
    if size is not None and size > ess:
        warnings.append(
            f"requested size {size} exceeds the Kish ESS {ess:.0f}: the sample will LOOK larger "
            f"than the information the reweighting supports")

    certainty: np.ndarray = np.array([], dtype=np.intp)
    if method == "systematic":
        pool = np.arange(n)
        pi = m * p
        while True:
            over = pi > 1.0
            if not over.any() or pool.size == 0:
                break
            certainty = np.concatenate([certainty, pool[over]])
            pool = pool[~over]
            m_left = m - certainty.size
            if m_left <= 0 or pool.size == 0:
                pi = np.array([])
                break
            pw = w[pool]
            pi = m_left * pw / pw.sum()
        if certainty.size:
            warnings.append(f"{certainty.size} frame(s) had inclusion probability >= 1 and were "
                            f"taken as certainty units")
        picked = list(certainty)
        if pool.size and pi.size:
            m_left = m - certainty.size
            # EXPLICIT SYSTEMATIC SCAN, not searchsorted. With inclusion probabilities <= 1 each
            # unit can be selected at most once, but searchsorted plus a clip at the last index
            # can still map two targets onto one unit at a floating-point boundary, which silently
            # returned 855 frames for a target of 856. Scanning emits exactly one pick per target.
            u = rng.uniform(0.0, 1.0)
            acc, t = 0.0, u
            got = 0
            for pos in range(pool.size):
                acc += pi[pos]
                while got < m_left and t < acc:
                    picked.append(pool[pos])
                    got += 1
                    t += 1.0
                if got >= m_left:
                    break
            if got < m_left:
                warnings.append(f"systematic scan emitted {got} of {m_left} probabilistic picks; "
                                f"the inclusion probabilities summed to {pi.sum():.6f}")
        sel = np.asarray(picked, dtype=np.intp)
        if np.unique(sel).size != sel.size:
            # can only happen if a unit spans more than one target, i.e. pi > 1 slipped through
            warnings.append("systematic sampling produced a repeated unit; deduplicated")
            sel = np.unique(sel)
    elif method == "rejection":
        keep = rng.uniform(size=n) < (w / w.max())
        sel = np.flatnonzero(keep)
        if size is not None:
            warnings.append("rejection sampling ignores `size`: its sample size is random")
    else:  # multinomial
        if not allow_duplicates:
            raise ValueError(
                "method='multinomial' draws WITH replacement, and duplicate frames sit at "
                "distance exactly 0, which drives HDBSCAN's core distance to 0 and destroys the "
                "density estimate. Pass allow_duplicates=True only if the consumer is not a "
                "density estimator.")
        sel = np.sort(rng.choice(n, size=m, replace=True, p=p))

    sel = np.asarray(sel, dtype=np.intp)
    uniq = int(np.unique(sel).size)
    return dict(
        indices=sel, method=method, seed=int(seed),
        size_requested=(None if size is None else int(size)),
        size_achieved=int(sel.size), n_distinct=uniq,
        has_duplicates=bool(uniq != sel.size),
        n_input=int(n), ess_input=ess, ess_implied_size=int(np.floor(ess)),
        certainty_units=int(certainty.size),
        weight_captured=float(p[np.unique(sel)].sum()),
        warnings=warnings,
        note=("the returned frames are an UNWEIGHTED sample of the weighted target ensemble; fit "
              "them without weights and the cluster boundaries are the TARGET's. Resampling does "
              "not create information: the ESS bounds what the reweighting can support, and mass "
              "in regions the biased run never visited is unrecoverable by any weighting."),
    )


# --------------------------------------------- classify held-out frames against a fitted partition
def classify_to_clusters(theta_train, labels_train, theta_query, *, units: str = "radians",
                         metric_weights=None, multiplicities: Optional[Sequence[int]] = None,
                         k: int = 15, exclude_noise: bool = True,
                         block: int = 4096, use_tree: bool = True,
                         min_vote_margin: float = MIN_VOTE_MARGIN_DEFAULT) -> Dict[str, Any]:
    """k-NN assignment of `theta_query` to the clusters of an already-fitted partition.

    WHY THIS EXISTS. The expensive step is the density clustering, which scales as ~N^1.83 in this
    20-D embedding. The cluster COUNT converges at N ~ 1024 (measured: two clusters, unanimous over
    five draws, at every size from 1024 to 65536), so the partition can be fitted on a small
    resampled subset and every remaining frame ASSIGNED to it -- which is O(N*k) and cheap. The
    cluster MASS then follows from a weighted sum over all frames, at full statistical precision,
    without ever clustering the full set.

    `sklearn.cluster.HDBSCAN` HAS NO `approximate_predict`; that belongs to the standalone `hdbscan`
    package, which is not installed here. So this is an explicit k-NN classifier rather than a call
    into the fitted model, and it is a DIFFERENT estimator from HDBSCAN's own assignment. It is
    validated by agreement against a direct fit, never assumed to reproduce one.

    `k` NEIGHBOURS, NOT ONE REPRESENTATIVE. Assigning to the nearest medoid imposes a Voronoi
    partition, which is convex; HDBSCAN's clusters need not be. A k-NN vote follows an arbitrary
    shape as long as the training subset covers it, which a stratified resample does.

    NOISE. With `exclude_noise=True` (default) training frames labelled -1 do not vote, so every
    query frame receives a real cluster. THAT IS NOT THE SAME AS HDBSCAN'S OWN OUTPUT: HDBSCAN
    withholds sparse frames as noise, and this classifier does not reproduce that set -- it commits
    them. Pass `exclude_noise=False` to let noise act as its own class and be assigned like any
    other.
    """
    a_tr = validate_angles(theta_train, units=units)
    a_q = validate_angles(theta_query, units=units)
    if a_tr.shape[1] != a_q.shape[1]:
        raise ValueError(f"train has {a_tr.shape[1]} torsions, query has {a_q.shape[1]}")
    lab = np.asarray(labels_train, dtype=np.int64).ravel()
    if lab.size != a_tr.shape[0]:
        raise ValueError(f"{lab.size} labels for {a_tr.shape[0]} training frames")
    w = validate_metric_weights(metric_weights, a_tr.shape[1])
    Xtr = embed(a_tr, w, multiplicities)
    Xq = embed(a_q, w, multiplicities)
    if exclude_noise:
        keep = lab >= 0
        if not keep.any():
            raise ValueError("every training frame is noise; nothing to classify against")
        Xtr, lab = Xtr[keep], lab[keep]
    kk = int(min(k, Xtr.shape[0]))
    W = float(np.sum(w))
    classes = np.unique(lab)
    out = np.empty(a_q.shape[0], dtype=np.int64)
    votes_margin = np.empty(a_q.shape[0], dtype=np.float64)
    # The k nearest neighbours can come from a spatial tree instead of a full distance matrix,
    # and the answer is IDENTICAL, not approximate: the embedding is built so that the chord
    # metric is exactly Euclidean in it, so a Euclidean KD/ball tree on the embedded training
    # points ranks neighbours the same way the chord metric does. The brute-force path stays as
    # the reference the tree is tested against -- it is O(n_query * n_train) and dominated the
    # cost of the whole fit-small/classify-all scheme (measured: 115 s of a 126 s iteration).
    tree = None
    if use_tree and Xtr.shape[0] > kk:
        try:
            from sklearn.neighbors import NearestNeighbors
            tree = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(Xtr)
        except Exception:
            tree = None

    for lo in range(0, a_q.shape[0], block if tree is None else max(block, 65536)):
        hi = min(lo + (block if tree is None else max(block, 65536)), a_q.shape[0])
        if tree is not None:
            nn = tree.kneighbors(Xq[lo:hi], n_neighbors=kk, return_distance=False)
        else:
            # same identity the chord metric allows: |x|^2 = W for every embedded point
            D2 = (2.0 * W) - 2.0 * (Xq[lo:hi] @ Xtr.T)
            np.maximum(D2, 0.0, out=D2)
            nn = np.argpartition(D2, kk - 1, axis=1)[:, :kk]
        nl = lab[nn]
        cnt = np.stack([(nl == c).sum(axis=1) for c in classes], axis=1)
        top = np.argmax(cnt, axis=1)
        out[lo:hi] = classes[top]
        srt = np.sort(cnt, axis=1)
        votes_margin[lo:hi] = (srt[:, -1] - (srt[:, -2] if srt.shape[1] > 1 else 0)) / kk
    # ABSTENTION. Without this the classifier commits every frame, which overrides the fit's own
    # decision to withhold: measured on ALA, 459 frames that HDBSCAN called noise (membership
    # probability exactly 0) were forced into the largest cluster by a bare 9-of-15 vote, moving
    # 0.15 pp of mass into the wrong basin. `min_vote_margin` returns those to -1 so the mass is
    # reported short rather than misattributed. It is NOT the same criterion as HDBSCAN's noise
    # (that is a density threshold, this is a vote agreement), so the threshold is calibrated
    # against a direct fit rather than assumed -- see scripts/hprest2_ala_margin_gate.py.
    n_abstained = 0
    if min_vote_margin > 0.0:
        weak = votes_margin < float(min_vote_margin)
        n_abstained = int(weak.sum())
        out = out.copy()
        out[weak] = -1
    return dict(labels=out, k=kk, classes=[int(c) for c in classes],
                vote_margin=votes_margin, min_vote_margin=float(min_vote_margin),
                n_abstained=n_abstained,
                n_train=int(Xtr.shape[0]), n_query=int(a_q.shape[0]),
                excluded_noise_from_training=bool(exclude_noise),
                note="k-NN assignment on the periodic chord metric. A DIFFERENT estimator from "
                     "HDBSCAN's own assignment; validate by agreement against a direct fit. With "
                     "noise excluded from voting every query frame is committed to a cluster, "
                     "which HDBSCAN itself would not do.")


def cluster_mass(labels, frame_weights=None) -> Dict[str, Any]:
    """Per-cluster mass: the WEIGHTED fraction if weights are given, the frame fraction if not.

    This is where reweighting belongs. The partition is fitted on an unweighted (or resampled)
    subset; the MASS is a weighted sum over every frame, so it carries the full statistical
    precision of the ensemble and the thermodynamic weights at the same time. The two are
    reported side by side because they are different quantities and conflating them is the error
    this separation exists to prevent.
    """
    lab = np.asarray(labels, dtype=np.int64).ravel()
    n = lab.size
    if frame_weights is None:
        w = np.ones(n)
        weighted = False
    else:
        w = np.asarray(frame_weights, dtype=np.float64).ravel()
        if w.size != n:
            raise ValueError(f"{w.size} frame weights for {n} labels")
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("frame weights must be finite and non-negative")
        weighted = True
    tot = float(w.sum())
    p = w / tot
    ess = float(1.0 / np.sum(p * p))
    def _entry(m):
        return dict(n_frames=int(m.sum()),
                    fraction_frames=float(m.mean()),
                    mass=float(w[m].sum() / tot),
                    ess_in_cluster=(float(w[m].sum() ** 2 / np.sum(w[m] ** 2))
                                    if m.any() and w[m].sum() > 0 else 0.0))

    # Noise is reported SEPARATELY, never as clusters[-1]. A consumer summing over `clusters`
    # to get the populations must not silently pick up the unassigned frames, and -1 is not one
    # physical state in any case: it is whatever the density estimate declined to commit.
    out = {int(c): _entry(lab == c) for c in np.unique(lab) if c >= 0}
    noise = _entry(lab < 0)
    return dict(clusters=out, noise=noise, weighted=weighted, ess_total=ess, n_frames=int(n),
                note=("mass is the WEIGHTED fraction; fraction_frames is the unweighted count "
                      "fraction. They differ whenever the ensemble is biased, and only `mass` is "
                      "a thermodynamic population." if weighted else
                      "no weights given, so mass == fraction_frames. On an unbiased ensemble "
                      "(e.g. the cold tau=0 rung) that IS the thermodynamic population."))


