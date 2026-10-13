"""`t_hdbscan` -- torsional HDBSCAN as a fit/predict estimator, SYMMETRY-FIRST by default.

THE DEFAULT WORKFLOW (since the symmetry-first change):

    enumerate molecular symmetry  ->  build the symmetry-aware torsional distance
                                  ->  fit HDBSCAN on that distance
                                  ->  classify frames with the SAME distance and an absolute vote

```python
fit = t_hdbscan(torsions, molecule=mol, torsion_definitions=defs,
                coordinates=traj, atom_mapping=mapping).fit()
```

A configuration is described by torsions measured through one LABELLING of its atoms, and a
molecular symmetry relabels atoms without moving any. So the distance between two frames is the
minimum over the symmetry-equivalent descriptions of them (`md_tools.analysis._symmetry_metric`),
and frames that differ only by a relabelling are at distance zero BEFORE HDBSCAN builds its
density. Symmetry-related basins can therefore share a cluster during fitting, before the minimum
cluster size is applied -- which is the point, and the difference from the older
cluster-then-merge workflow (still available, see below).

THE INPUT CONTRACT. Angles alone cannot establish a symmetry, so the default needs:

* `torsions` -- `(n_frames, n_selected)` stored angles, in the order of `torsion_definitions`;
* `molecule` -- an RDKit molecule with explicit hydrogens (the SDF the system was built from);
* `torsion_definitions` -- a `TorsionDefinitions` whose quadruplets index that molecule;
* `coordinates` and `atom_mapping` -- needed whenever a symmetry operation maps a selected torsion
  onto a quadruplet that is not in the table (paracetamol: always). An `mdtraj.Trajectory` carries
  its unit cell and topology; an array needs `box_vectors` (or `periodic=False`) and `topology`.

Missing information RAISES with the call that fixes it. Nothing guesses a symmetry, and nothing
falls back to ordinary clustering on its own.

THE LEGACY ROUTE is `symmetry=False`: the plain weighted cos/sin Euclidean distance with no
symmetry, which is what this estimator computed before. Add `vote_rule='legacy-margin'` to restore
the historical vote as well (k = 15, margin `(n_first - n_second)/k >= 0.9`, own label counted).
Symmetry merging after the fit is still `TorsionSymmetry` (see `_t_symmetry`).

ASSIGNMENT. After the fit every frame is classified by its k = 20 nearest CLUSTERED training
frames under the same distance, excluding itself: it is assigned when the winning cluster holds
at least 90 % of `k_effective` votes, equality accepted -- 18 of 20 assigns, 17 of 20 does not.
`k_effective` is 20, or the number of eligible training frames when there are fewer. Two
unassigned codes, and they are different claims:

* `NOISE_LABEL` (-2): DENSITY. The frame's `min_samples`-th neighbour is farther than that of any
  clustered training frame -- nowhere in particular. Takes precedence.
* `AMBIGUOUS_LABEL` (-1): VOTE. The neighbours disagree. This is a statement about the sample
  near the frame; it is NOT evidence of a physical free-energy barrier, which nothing here
  measures. (Formerly `BARRIER_LABEL`; the name is kept as an alias.)

A CLUSTER HERE IS NOT A METASTABLE STATE. Nothing looks at time.

WRITTEN BY THE hpREST2 SESSION and ported here with its agreement; the symmetry-first route and
the vote rule were added in md-tools. NEEDS THE `analysis` EXTRA (scikit-learn; rdkit for the
default route), imported lazily.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._torsions import (K_NEIGHBOURS_DEFAULT, K_NEIGHBOURS_LEGACY,
                        MIN_CLUSTER_FRACTION_DEFAULT, MIN_SAMPLES_FRACTION_DEFAULT,
                        MIN_VOTE_FRACTION_DEFAULT, MIN_VOTE_MARGIN_LEGACY, N_SEED_DEFAULT,
                        VOTE_RULES, _UNSET, classify_to_clusters,
                        cluster_mass, cluster_torsions, embed, k_nearest_stable,
                        legacy_vote_argument_error, validate_angles, validate_frame_weights,
                        validate_metric_weights, vote, weighted_resample, wrap)

#: DEFAULT TRUE since 2026-10-01, which is NOT sklearn's default and is the one place this
#: estimator deliberately departs from it.
#:
#: sklearn's EOM selection refuses the root of the condensed tree unless this is set, so with it
#: False a UNIMODAL ensemble cannot be reported as one state -- the blob is split instead, and the
#: failure is silent and confident: measured, a 4000-frame single Gaussian returns 2-3 clusters
#: with a plausible mass split and nothing in the output saying one state was never a candidate.
#:
#: THE COST OF TRUE IS REAL AND POINTS THE OTHER WAY: allowing the root lets a genuinely
#: multi-state ensemble collapse into one cluster when the between-basin density is not low enough
#: to beat the root's stability. Over-splitting is the harder error to notice, so it is the one the
#: default avoids. `summary()` records the flag either way.
ALLOW_SINGLE_CLUSTER_DEFAULT = True

#: A frame the DENSITY test excludes: its neighbourhood is thinner than that of any frame the fit
#: put in a cluster. "Nowhere in particular."
NOISE_LABEL = -2

#: A frame the VOTE declines: fewer than `min_vote_fraction` of its neighbours agree. AMBIGUOUS,
#: not a barrier: neighbour disagreement says nothing about free energy.
AMBIGUOUS_LABEL = -1

#: The former name of `AMBIGUOUS_LABEL`, same value. Kept so existing callers keep working; new
#: code should not read "barrier" into a vote.
BARRIER_LABEL = AMBIGUOUS_LABEL

#: THE RESOURCE GUARD for the default route. The symmetry-aware distance is a MINIMUM over
#: relabellings, which is not a Euclidean distance in any fixed embedding, so it cannot use a
#: KD-tree and is handed to HDBSCAN as an exact dense matrix. That costs `8 n^2` bytes per copy,
#: and sklearn's precomputed path holds about three (the matrix, its copy, the mutual
#: reachability). Measured on the 4000-frame paracetamol tutorial: see the clustering notebook.
#: At this default (10000 frames) one matrix is 800 MB. Above it the fit REFUSES and names the
#: subset route (`resampling=N`) rather than silently changing the metric.
MAX_PRECOMPUTED_FRAMES_DEFAULT = 10000

__all__ = ["ALLOW_SINGLE_CLUSTER_DEFAULT", "AMBIGUOUS_LABEL", "BARRIER_LABEL", "NOISE_LABEL",
           "MAX_PRECOMPUTED_FRAMES_DEFAULT", "PrecomputedMemoryGuard", "THDBSCAN", "t_hdbscan"]


class PrecomputedMemoryGuard(MemoryError):
    """The exact symmetry-aware distance matrix would exceed the configured frame limit."""


def _symmetry_contract_error(missing: Sequence[str]) -> ValueError:
    return ValueError(
        "t_hdbscan clusters SYMMETRY-FIRST by default, and angles alone cannot establish a "
        f"molecular symmetry. Missing: {', '.join(missing)}. Pass molecule=<RDKit Mol with "
        "explicit hydrogens>, torsion_definitions=<TorsionDefinitions indexing it> and, when the "
        "symmetry maps a selected torsion onto one that is not in the table, coordinates=<mdtraj "
        "trajectory> with atom_mapping=<trajectory index of each reference atom>. For the "
        "previous behaviour -- the plain cos/sin Euclidean distance with no symmetry -- pass "
        "symmetry=False; add vote_rule='legacy-margin' to restore the historical vote too.")


def _resolve_sizes(n: int, mass_floor: float, min_samples: Optional[int],
                   min_samples_fraction: Optional[float]) -> Dict[str, Any]:
    """`min_cluster_size` and `min_samples`, by exactly the rules `cluster_torsions` applies."""
    mcs = int(max(2, math.ceil(float(mass_floor) * n)))
    if min_samples is not None:
        if min_samples_fraction is not None:
            raise ValueError("pass min_samples or min_samples_fraction, not both")
        ms, src = int(min_samples), f"min_samples={int(min_samples)} given explicitly"
    else:
        frac = (MIN_SAMPLES_FRACTION_DEFAULT if min_samples_fraction is None
                else float(min_samples_fraction))
        if not (0.0 < frac < 1.0):
            raise ValueError("min_samples_fraction must be in (0, 1)")
        ms = int(max(2, round(frac * n)))
        src = f"min_samples_fraction={frac} of n={n}" + (
            "" if min_samples_fraction is not None else " (module default)")
    if mcs > n:
        raise ValueError(f"min_cluster_size={mcs} exceeds the {n} frames supplied")
    return {"min_cluster_size": mcs, "min_samples": ms, "min_samples_source": src,
            "min_cluster_size_source": f"min_cluster_fraction={mass_floor} of n={n}"}


class THDBSCAN:
    """Torsional HDBSCAN with a fit/predict interface. See the module docstring first.

    `mass_floor` is HDBSCAN's `min_cluster_size` as a fraction of the fitted frames. `resampling`
    is `False` or an int N: fit N frames drawn without replacement in proportion to the weights,
    then classify every frame. `seed` is the NUMBER of such draws (default 5), not an RNG seed.

    VOTE. `vote_rule='fraction'` (default): `k` defaults to 20 and `min_vote_fraction` to 0.90.
    `vote_rule='legacy-margin'`: `k` defaults to 15 and `min_vote` (the historical MARGIN) to
    0.90. Passing `min_vote` without naming the legacy rule raises: a margin is never
    reinterpreted as a fraction.
    """

    def __init__(self, torsions, *, molecule=None, torsion_definitions=None,
                 coordinates=None, atom_mapping=None, box_vectors=None,
                 periodic: Optional[bool] = None, topology=None,
                 symmetry: bool = True,
                 weights=None, resampling=False, seed: int = N_SEED_DEFAULT,
                 mass_floor: float = MIN_CLUSTER_FRACTION_DEFAULT,
                 vote_rule: str = "fraction", min_vote_fraction=_UNSET, min_vote=_UNSET,
                 k: Optional[int] = None,
                 units: str = "radians", names: Optional[Sequence[str]] = None,
                 metric_weights=None, multiplicities=None,
                 multiplicity_justification: Optional[str] = None,
                 min_samples_fraction: Optional[float] = None,
                 min_samples: Optional[int] = None,
                 allow_single_cluster: bool = ALLOW_SINGLE_CLUSTER_DEFAULT,
                 max_precomputed_frames: int = MAX_PRECOMPUTED_FRAMES_DEFAULT,
                 max_matches: Optional[int] = None, use_chirality: bool = True,
                 stored_tolerance: Optional[float] = None,
                 broken_by: Optional[Sequence[str]] = None):
        self.theta_ = validate_angles(torsions, units)
        self.n_frames_, self.n_torsions_ = self.theta_.shape
        w = validate_frame_weights(weights, self.n_frames_)
        self.weights_ = np.ones(self.n_frames_) if w is None else np.asarray(w, dtype=np.float64)
        self.weights_uniform_ = bool(np.allclose(self.weights_, self.weights_[0]))

        if resampling is False or resampling is None:
            self.resampling = False
        else:
            r = int(resampling)
            if r < 2:
                raise ValueError(f"resampling must be False or an int >= 2, got {resampling!r}")
            self.resampling = min(r, self.n_frames_)
        if self.resampling is not False and int(seed) < 1:
            raise ValueError("seed (the number of draws) must be at least 1")
        if not (0.0 < float(mass_floor) < 1.0):
            raise ValueError("mass_floor must be in (0, 1)")

        # ---- the vote rule: an explicit legacy margin is refused unless the legacy rule is named
        if vote_rule not in VOTE_RULES:
            raise ValueError(f"vote_rule must be one of {VOTE_RULES}, not {vote_rule!r}")
        self.vote_rule = vote_rule
        if vote_rule == "fraction":
            if min_vote is not _UNSET:
                raise legacy_vote_argument_error("min_vote", min_vote)
            threshold = (MIN_VOTE_FRACTION_DEFAULT if min_vote_fraction is _UNSET
                         else float(min_vote_fraction))
            self.k = K_NEIGHBOURS_DEFAULT if k is None else int(k)
        else:
            if min_vote_fraction is not _UNSET:
                raise TypeError("min_vote_fraction belongs to vote_rule='fraction'; the legacy "
                                "rule takes min_vote (its margin)")
            threshold = MIN_VOTE_MARGIN_LEGACY if min_vote is _UNSET else float(min_vote)
            self.k = K_NEIGHBOURS_LEGACY if k is None else int(k)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("min_vote_fraction / min_vote must be in [0, 1]")
        if self.k < 1:
            raise ValueError("k must be at least 1")
        self.vote_threshold = float(threshold)
        self.min_vote_fraction = threshold if vote_rule == "fraction" else None
        self.min_vote = threshold if vote_rule == "legacy-margin" else None

        self.seed = int(seed)
        self.mass_floor = float(mass_floor)
        self.metric_weights = metric_weights
        self.multiplicities = multiplicities
        self.multiplicity_justification = multiplicity_justification
        self.min_samples = min_samples
        self.min_samples_fraction = min_samples_fraction
        self.allow_single_cluster = bool(allow_single_cluster)
        self.max_precomputed_frames = int(max_precomputed_frames)
        self.broken_by = list(broken_by or [])
        self.symmetry = bool(symmetry)
        self.fitted_ = False

        if self.symmetry:
            missing = [nm for nm, v in (("molecule", molecule),
                                        ("torsion_definitions", torsion_definitions)) if v is None]
            if missing:
                raise _symmetry_contract_error(missing)
            if multiplicities is not None and any(int(v) != 1 for v in multiplicities):
                raise ValueError(
                    "multiplicities fold each torsion INDEPENDENTLY by 2*pi/k, which is the "
                    "per-torsion folding the symmetry-first distance exists to replace: it would "
                    "also identify configurations where only one substituent rotated. With "
                    "symmetry=True the molecular symmetry is enumerated instead; leave "
                    "multiplicities at 1, or use symmetry=False.")
            from ._symmetry_metric import (STORED_AGREEMENT_TOLERANCE, SymmetricTorsionDescriptor)
            from ._t_symmetry import MAX_AUTOMORPHISMS

            if int(self.n_torsions_) != torsion_definitions.n_torsions:
                raise ValueError(f"{self.n_torsions_} torsion columns for "
                                 f"{torsion_definitions.n_torsions} torsion definitions")
            if names is not None and [str(x) for x in names] != list(torsion_definitions.names):
                raise ValueError("names disagree with torsion_definitions.names; the definitions "
                                 "are the authority, so pass one or make them match")
            self.names = list(torsion_definitions.names)
            self.descriptor_ = SymmetricTorsionDescriptor(
                molecule, torsion_definitions, metric_weights=metric_weights,
                max_matches=(MAX_AUTOMORPHISMS if max_matches is None else int(max_matches)),
                use_chirality=use_chirality)
            self._coordinate_kwargs = dict(atom_mapping=atom_mapping, periodic=periodic,
                                           topology=topology)
            self.stored_tolerance = (STORED_AGREEMENT_TOLERANCE if stored_tolerance is None
                                     else float(stored_tolerance))
            self.closed_theta_ = self.descriptor_.closed_angles(
                self.theta_, coordinates=coordinates, box_vectors=box_vectors,
                tolerance=self.stored_tolerance, **self._coordinate_kwargs)
            self.X_ = self.descriptor_.embed(self.closed_theta_)
        else:
            if molecule is not None or torsion_definitions is not None or coordinates is not None:
                raise ValueError("symmetry=False uses the plain Euclidean torsion distance and "
                                 "takes no molecule, torsion_definitions or coordinates; drop "
                                 "them, or keep symmetry=True")
            self.names = None if names is None else [str(x) for x in names]
            self.descriptor_ = None

    # ---------------------------------------------------------------------- shared bookkeeping

    @property
    def route_(self) -> str:
        return "symmetry-first" if self.symmetry else "legacy-euclidean"

    def _draws(self):
        if self.resampling is False:
            return [(None, np.arange(self.n_frames_))]
        return [(s, np.sort(weighted_resample(self.weights_, size=self.resampling,
                                              seed=s)["indices"]))
                for s in range(self.seed)]

    def _canonical_labels(self, labels):
        order = [c for c in np.unique(labels) if c >= 0]
        if not order:
            raise RuntimeError("every training frame is noise; nothing to classify against")
        sizes = {c: int((labels == c).sum()) for c in order}
        remap = {c: i for i, c in enumerate(sorted(order, key=lambda c: (-sizes[c], c)))}
        canon = np.full(labels.shape, -1, dtype=np.int64)
        for c, i in remap.items():
            canon[labels == c] = i
        return canon, len(remap)

    # ---------------------------------------------------------------------------------- fitting

    def fit(self) -> "THDBSCAN":
        """Fit the partition, then classify every frame. Returns self."""
        if self.symmetry:
            self._fit_symmetric()
        else:
            self._fit_legacy()
        self._finish()
        return self

    # .................................................................... symmetry-first route

    def _guard(self, n_fit: int) -> None:
        one = 8 * n_fit * n_fit
        self.precomputed_bytes_ = int(one)
        if n_fit > self.max_precomputed_frames:
            raise PrecomputedMemoryGuard(
                f"the symmetry-aware distance is fitted as an EXACT dense matrix: {n_fit} frames "
                f"need {one / 2 ** 20:.0f} MiB per copy, and HDBSCAN's precomputed path holds "
                f"about three. That exceeds max_precomputed_frames={self.max_precomputed_frames}. "
                f"Fit a subset with resampling=N (N <= {self.max_precomputed_frames}; frames are "
                f"drawn in proportion to the weights, and EVERY frame is then classified with the "
                f"same distance), or raise max_precomputed_frames knowingly. The metric is never "
                f"swapped for a Euclidean one to make it fit.")

    def _fit_symmetric(self) -> None:
        from sklearn.cluster import HDBSCAN

        draws, counts = [], []
        n_fit = self.n_frames_ if self.resampling is False else int(self.resampling)
        self._guard(n_fit)
        for s, idx in self._draws():
            D = self.descriptor_.distance_matrix(self.X_[idx])
            sizes = _resolve_sizes(idx.size, self.mass_floor, self.min_samples,
                                   self.min_samples_fraction)
            est = HDBSCAN(min_cluster_size=sizes["min_cluster_size"],
                          min_samples=sizes["min_samples"], metric="precomputed",
                          cluster_selection_method="eom",
                          allow_single_cluster=self.allow_single_cluster, copy=True)
            est.fit(D)
            labels = np.asarray(est.labels_, dtype=np.int64)
            del D
            draws.append((s, idx, labels, sizes))
            counts.append(int(len([c for c in np.unique(labels) if c >= 0])))
        self._pick_draw(draws, counts)
        self.settings_.update({
            "metric": "symmetry-aware quotient distance: min over molecular relabellings of the "
                      "orbit-weighted cos/sin chord distance, as an exact precomputed matrix",
            "algorithm": "precomputed (dense)",
            "precomputed_bytes_per_copy": int(self.precomputed_bytes_),
            "max_precomputed_frames": self.max_precomputed_frames,
            "cluster_selection_method": "eom",
            "allow_single_cluster": self.allow_single_cluster,
        })
        self._classify_symmetric_all()

    def _pick_draw(self, draws, counts) -> None:
        vals, cnt = np.unique(counts, return_counts=True)
        mode = int(vals[int(np.argmax(cnt))])
        self.n_clusters_per_draw_ = counts
        self.draw_agreement_ = float(cnt.max() / len(counts))
        pick = next((d for d in draws if len([c for c in np.unique(d[2]) if c >= 0]) == mode), None)
        if pick is None or mode == 0:
            raise RuntimeError(f"no draw produced a cluster (counts {counts}); the mass floor "
                               f"{self.mass_floor} may be above every basin's population")
        s, idx, labels, sizes = pick
        canon, n_clusters = self._canonical_labels(labels)
        self.canonical_draw_ = s
        self.train_index_ = idx
        self.train_labels_ = canon
        self.n_clusters_ = n_clusters
        self.settings_ = {"min_cluster_size": sizes["min_cluster_size"],
                          "min_cluster_size_source": sizes["min_cluster_size_source"],
                          "min_samples": sizes["min_samples"],
                          "min_samples_source": sizes["min_samples_source"]}

    def _neighbour_pass(self, Xq: np.ndarray, query_train_pos: np.ndarray,
                        block_bytes: int = 256 * 2 ** 20) -> Dict[str, np.ndarray]:
        """Density core distance and the vote for each query, from ONE symmetry-aware distance.

        `query_train_pos[i]` is the training row query `i` IS, or -1. That row is excluded from
        both the density neighbours and the vote (leave-one-out), so a training frame is judged
        by the others exactly as a held-out frame would be.
        """
        Xt = self.X_[self.train_index_]
        lab = self.train_labels_
        clustered = np.flatnonzero(lab >= 0)
        n_t, n_q = Xt.shape[0], Xq.shape[0]
        ms = max(1, min(int(self.settings_["min_samples"]), n_t - 1))
        rows = max(1, int(block_bytes // max(1, 8 * n_t * (len(self.descriptor_.operations) + 2))))
        core = np.empty(n_q)
        is_self = query_train_pos >= 0
        eligible = clustered.size - np.isin(query_train_pos, clustered).astype(np.int64)
        if np.any(eligible < 1):
            raise RuntimeError("a frame has no eligible clustered neighbour to vote")
        k_eff = np.minimum(self.k, eligible)
        kmax = int(k_eff.max())
        nl = np.full((n_q, kmax), -1, dtype=np.int64)
        op_to_nearest = np.zeros(n_q, dtype=np.int64)
        for lo in range(0, n_q, rows):
            hi = min(n_q, lo + rows)
            D = self.descriptor_.distance_matrix(Xq[lo:hi], Xt)
            sel = np.flatnonzero(is_self[lo:hi])
            D[sel, query_train_pos[lo:hi][sel]] = np.inf
            core[lo:hi] = np.partition(D, ms - 1, axis=1)[:, ms - 1]
            Dc = D[:, clustered]
            got = lab[clustered[k_nearest_stable(Dc, kmax)]]
            got[np.arange(kmax)[None, :] >= k_eff[lo:hi][:, None]] = -1
            nl[lo:hi] = got
        v = vote(nl, k_eff, rule="fraction", threshold=self.vote_threshold)
        return {"core": core, **v}

    def _classify_symmetric_all(self) -> None:
        pos = np.full(self.n_frames_, -1, dtype=np.int64)
        pos[self.train_index_] = np.arange(self.train_index_.size)
        out = self._neighbour_pass(self.X_, pos)
        train_core = out["core"][self.train_index_]
        inc = self.train_labels_ >= 0
        self.density_threshold_ = float(train_core[inc].max()) if inc.any() else float("inf")
        self.density_agreement_ = float(np.mean((train_core > self.density_threshold_)
                                                == (self.train_labels_ < 0)))
        labels = out["labels"].copy()
        labels[out["core"] > self.density_threshold_] = NOISE_LABEL
        self.labels_ = labels
        self.vote_fraction_ = out["vote_fraction"]
        self.n_winner_ = out["n_winner"]
        self.k_effective_ = out["k_effective"]
        self.core_distance_ = out["core"]
        self._core_k_ = max(1, min(int(self.settings_["min_samples"]),
                                   self.train_index_.size - 1))
        self.density_proxy_note_ = (
            f"a frame is NOISE_LABEL ({NOISE_LABEL}) if its distance to its {self._core_k_}-th "
            f"nearest training frame (itself excluded) under the SAME symmetry-aware distance "
            f"exceeds {self.density_threshold_:.6g}, the largest such distance among clustered "
            f"training frames. A proxy for HDBSCAN's tree criterion, not that criterion: it agrees "
            f"with HDBSCAN's own verdict on {100 * self.density_agreement_:.2f} % of the training "
            f"frames.")

    # ............................................................................ legacy route

    def _fit_legacy(self) -> None:
        draws, counts = [], []
        for s, idx in self._draws():
            f = cluster_torsions(
                self.theta_[idx], units="radians", names=self.names,
                metric_weights=self.metric_weights, multiplicities=self.multiplicities,
                multiplicity_justification=self.multiplicity_justification,
                min_cluster_fraction=self.mass_floor, min_samples=self.min_samples,
                min_samples_fraction=self.min_samples_fraction,
                allow_single_cluster=self.allow_single_cluster)
            draws.append((s, idx, f))
            counts.append(int(f.n_clusters))
        vals, cnt = np.unique(counts, return_counts=True)
        mode = int(vals[int(np.argmax(cnt))])
        self.n_clusters_per_draw_ = counts
        self.draw_agreement_ = float(cnt.max() / len(counts))
        pick = next((d for d in draws if int(d[2].n_clusters) == mode), None)
        if pick is None or mode == 0:
            raise RuntimeError(f"no draw produced a cluster (counts {counts}); the mass floor "
                               f"{self.mass_floor} may be above every basin's population")
        s, idx, f = pick
        self.canonical_draw_ = s
        self.train_index_ = idx
        self.settings_ = dict(f.settings)
        canon, n_clusters = self._canonical_labels(f.labels)
        self.train_theta_, self.train_labels_ = self.theta_[idx], canon
        self.n_clusters_ = n_clusters
        self._set_names()
        self._fit_density_threshold()
        idx_all, _ = self._assign(self.theta_, training_rows=self._training_rows_of_all())
        self.labels_ = idx_all

    def _training_rows_of_all(self) -> np.ndarray:
        pos = np.full(self.n_frames_, -1, dtype=np.int64)
        pos[self.train_index_] = np.arange(self.train_index_.size)
        return pos

    def _legacy_embed(self, theta):
        return embed(theta, validate_metric_weights(self.metric_weights, self.n_torsions_),
                     self.multiplicities)

    def _fit_density_threshold(self) -> None:
        """LEGACY ROUTE. The density scale a query frame must meet to be eligible at all.

        A query frame's distance to its `min_samples`-th nearest TRAINING neighbour, against the
        largest such distance among training frames the fit put in a cluster. Unchanged from the
        historical estimator, so the legacy comparison isolates the vote rule.
        """
        X = self._legacy_embed(self.train_theta_)
        ms = int(self.settings_.get("min_samples") or 5)
        ms = max(1, min(ms, X.shape[0] - 1))
        W = float(np.sum(validate_metric_weights(self.metric_weights, self.n_torsions_)))
        d = np.empty(X.shape[0])
        for lo in range(0, X.shape[0], 4096):
            hi = min(lo + 4096, X.shape[0])
            D2 = np.maximum((2.0 * W) - 2.0 * (X[lo:hi] @ X.T), 0.0)
            part = np.partition(D2, ms, axis=1)[:, ms]
            d[lo:hi] = np.sqrt(part)
        self._train_core_ = d
        self._core_k_ = ms
        inc = self.train_labels_ >= 0
        self.density_threshold_ = (float(d[inc].max()) if inc.any() else float("inf"))
        proxy_noise = d > self.density_threshold_
        hdb_noise = self.train_labels_ < 0
        self.density_agreement_ = float(np.mean(proxy_noise == hdb_noise))
        self.density_proxy_note_ = (
            f"a query frame is NOISE_LABEL ({NOISE_LABEL}) if its distance to its {ms}-th nearest "
            f"training neighbour exceeds {self.density_threshold_:.6g}, the largest such distance "
            f"among clustered training frames. This is a PROXY for HDBSCAN's tree criterion, not "
            f"that criterion: it agrees with HDBSCAN's own verdict on "
            f"{100 * self.density_agreement_:.2f} % of the training frames.")

    def _query_core(self, theta) -> np.ndarray:
        """LEGACY ROUTE. Distance to the `min_samples`-th nearest TRAINING neighbour."""
        w = validate_metric_weights(self.metric_weights, self.n_torsions_)
        Xt = self._legacy_embed(self.train_theta_)
        Xq = embed(theta, w, self.multiplicities)
        W = float(np.sum(w))
        ms = max(1, min(self._core_k_, Xt.shape[0] - 1))
        out = np.empty(Xq.shape[0])
        for lo in range(0, Xq.shape[0], 4096):
            hi = min(lo + 4096, Xq.shape[0])
            D2 = np.maximum((2.0 * W) - 2.0 * (Xq[lo:hi] @ Xt.T), 0.0)
            out[lo:hi] = np.sqrt(np.partition(D2, ms - 1, axis=1)[:, ms - 1])
        return out

    def _assign(self, theta, training_rows=None) -> Tuple[np.ndarray, np.ndarray]:
        """LEGACY ROUTE. Vote, then density, on the Euclidean embedding."""
        kwargs = dict(units="radians", metric_weights=self.metric_weights,
                      multiplicities=self.multiplicities, k=self.k, vote_rule=self.vote_rule)
        if self.vote_rule == "fraction":
            kwargs["min_vote_fraction"] = self.vote_threshold
            kwargs["query_training_rows"] = training_rows
        else:
            kwargs["min_vote_margin"] = self.vote_threshold
        out = classify_to_clusters(self.train_theta_, self.train_labels_, theta, **kwargs)
        idx = np.asarray(out["labels"], dtype=np.int64)
        idx[idx < 0] = AMBIGUOUS_LABEL
        # DENSITY TAKES PRECEDENCE: "nowhere in particular" is the stronger statement.
        idx[self._query_core(theta) > self.density_threshold_] = NOISE_LABEL
        self._last_vote_ = out
        return idx, self._names_for(idx)

    # ------------------------------------------------------------------------------- outputs

    def _names_for(self, idx) -> np.ndarray:
        return np.array([self.noise_name_ if i == NOISE_LABEL
                         else self.ambiguous_name_ if i == AMBIGUOUS_LABEL
                         else self.cluster_names_[i] for i in idx], dtype=object)

    @property
    def barrier_name_(self) -> str:
        return self.ambiguous_name_

    def _set_names(self) -> None:
        self.cluster_names_ = [f"state_{i}" for i in range(self.n_clusters_)]
        self.noise_name_ = "noise"
        self.ambiguous_name_ = "ambiguous"

    def _finish(self) -> None:
        self._set_names()
        if not self.symmetry:
            vote_out = self._last_vote_
            if self.vote_rule == "fraction":
                self.vote_fraction_ = np.asarray(vote_out["vote_fraction"])
                self.n_winner_ = np.asarray(vote_out["n_winner"])
                self.k_effective_ = np.asarray(vote_out["k_effective"])
            else:
                self.vote_margin_ = np.asarray(vote_out["vote_margin"])
                self.k_effective_ = np.full(self.n_frames_, int(vote_out["k"]))
        cm = cluster_mass(np.where(self.labels_ >= 0, self.labels_, -1),
                          frame_weights=self.weights_)
        total = float(self.weights_.sum())
        self.cluster_mass_ = {i: float(self.weights_[self.labels_ == i].sum() / total)
                              for i in range(self.n_clusters_)}
        self.noise_mass_ = float(self.weights_[self.labels_ == NOISE_LABEL].sum() / total)
        self.ambiguous_mass_ = float(self.weights_[self.labels_ == AMBIGUOUS_LABEL].sum() / total)
        self.unassigned_mass_ = self.noise_mass_ + self.ambiguous_mass_
        self.ess_total_ = float(cm["ess_total"])
        self.fitted_ = True

    def predict(self, test_torsions, *, units: str = "radians", coordinates=None,
                box_vectors=None) -> Tuple[np.ndarray, np.ndarray]:
        """`(cluster_idx, cluster_names)` for new frames, by the fit's distance and vote.

        On the default route, `coordinates` (and `box_vectors` for an array) are needed whenever
        the descriptor has columns absent from the table; the atom mapping and topology of the
        fit are reused.
        """
        if not self.fitted_:
            raise RuntimeError("call fit() before predict()")
        th = validate_angles(test_torsions, units, n_torsions=self.n_torsions_)
        if not self.symmetry:
            return self._assign(th)
        closed = self.descriptor_.closed_angles(th, coordinates=coordinates,
                                                box_vectors=box_vectors,
                                                tolerance=self.stored_tolerance,
                                                **self._coordinate_kwargs)
        out = self._neighbour_pass(self.descriptor_.embed(closed),
                                   np.full(th.shape[0], -1, dtype=np.int64))
        labels = out["labels"].copy()
        labels[out["core"] > self.density_threshold_] = NOISE_LABEL
        return labels, self._names_for(labels)

    def fit_predict(self, *, units: str = "radians") -> Tuple[np.ndarray, np.ndarray]:
        """Fit, then the labels of the training ensemble itself."""
        self.fit()
        return self.labels_, self._names_for(self.labels_)

    # --------------------------------------------------------------- representatives and stats

    def _route_distances(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        if self.symmetry:
            return self.descriptor_.distance_matrix(self.X_[rows_a], self.X_[rows_b])
        X = self._legacy_embed(self.theta_)
        W = float(np.sum(validate_metric_weights(self.metric_weights, self.n_torsions_)))
        return np.sqrt(np.maximum(2.0 * W - 2.0 * (X[rows_a] @ X[rows_b].T), 0.0))

    def representatives(self, max_members: int = 5000) -> Dict[int, int]:
        """One ACTUAL FRAME per cluster: the strongest vote, ties by the route's own medoid.

        The candidates are the members with the highest vote score (fraction; margin on the
        legacy-margin route). Among them the frame with the smallest summed distance to the
        cluster's members wins, under the SAME distance the fit used -- the symmetry-aware one by
        default, so a member written in another labelling is not counted as far away. The legacy
        margin route keeps its historical tie-break (first frame), so an archived figure is
        reproduced. Clusters over `max_members` frames are scored against their training members.
        """
        if not self.fitted_:
            raise RuntimeError("call fit() before representatives()")
        score = (self.vote_margin_ if self.vote_rule == "legacy-margin" else self.vote_fraction_)
        out = {}
        for c in range(self.n_clusters_):
            members = np.flatnonzero(self.labels_ == c)
            if members.size == 0:
                continue
            best = score[members].max()
            cand = members[score[members] >= best - 1e-12]
            if self.vote_rule == "legacy-margin" or cand.size == 1:
                out[c] = int(cand[0])
                continue
            ref = members
            if ref.size > max_members:
                ref = np.intersect1d(members, self.train_index_)
            total = np.zeros(cand.size)
            for lo in range(0, cand.size, 512):
                total[lo:lo + 512] = self._route_distances(cand[lo:lo + 512], ref).sum(axis=1)
            out[c] = int(cand[int(np.argmin(total))])
        return out

    def aligned_angles(self, cluster: int, reference: Optional[int] = None) -> Dict[str, Any]:
        """The SELECTED torsions of a cluster's members, each frame written in the labelling
        closest to the reference frame.

        A cluster on the default route may hold one conformer written in two labellings; a
        histogram of the raw `aryl` column would then show two peaks that are one geometry. Each
        member is relabelled by the operation minimising its distance to the reference (the
        cluster representative unless given). The relabelling moves no atom, and the operation
        used is returned per frame.
        """
        if not self.symmetry:
            members = np.flatnonzero(self.labels_ == cluster)
            return {"frames": members, "angles": self.theta_[members],
                    "operations": np.zeros(members.size, dtype=np.int64)}
        members = np.flatnonzero(self.labels_ == cluster)
        ref = self.representatives()[cluster] if reference is None else int(reference)
        _, ops = self.descriptor_.distance_matrix(self.X_[[ref]], self.X_[members],
                                                  return_operation=True)
        ops = ops[0].astype(np.int64)
        aligned = np.empty((members.size, self.descriptor_.n_columns))
        for op in np.unique(ops):
            sel = ops == op
            aligned[sel] = self.descriptor_.transform(self.closed_theta_[members[sel]], int(op))
        return {"frames": members, "reference": ref, "operations": ops,
                "angles": wrap(aligned[:, :self.n_torsions_]), "closed_angles": wrap(aligned)}

    # ------------------------------------------------------------------------------ summary

    def summary(self) -> Dict[str, Any]:
        """Everything a result file should record, including what the fit does NOT establish."""
        if not self.fitted_:
            raise RuntimeError("call fit() before summary()")
        n_fit = int(self.train_index_.size)
        out = dict(
            route=self.route_,
            n_frames=int(self.n_frames_), n_torsions=int(self.n_torsions_),
            n_fit_frames=n_fit,
            n_clusters=int(self.n_clusters_), cluster_names=list(self.cluster_names_),
            cluster_mass={int(k): float(v) for k, v in self.cluster_mass_.items()},
            noise_mass=float(self.noise_mass_),
            ambiguous_mass=float(self.ambiguous_mass_),
            unassigned_mass=float(self.unassigned_mass_),
            resampling=(False if self.resampling is False else int(self.resampling)),
            resampling_draws=(None if self.resampling is False else int(self.seed)),
            seed=int(self.seed), mass_floor=self.mass_floor,
            vote=self._vote_record(),
            allow_single_cluster=self.allow_single_cluster,
            single_cluster_excluded_by_construction=(not self.allow_single_cluster),
            noise_label=NOISE_LABEL, ambiguous_label=AMBIGUOUS_LABEL,
            density_threshold=float(self.density_threshold_),
            density_proxy_agreement=float(self.density_agreement_),
            density_proxy_note=self.density_proxy_note_,
            weights_uniform=self.weights_uniform_,
            frame_weights_role=(
                "POPULATIONS are weighted sums over every frame. The DENSITY the fit sees is that "
                "of the fitted frames, unweighted -- unless resampling=N, which draws the fitted "
                "frames in proportion to the weights so the density is the target's."),
            ess_total=float(self.ess_total_),
            settings=self.settings_,
            n_clusters_per_draw=list(self.n_clusters_per_draw_),
            draw_agreement=float(self.draw_agreement_),
            canonical_draw=self.canonical_draw_,
            symmetry=(self.descriptor_.describe() if self.symmetry else
                      {"enabled": False, "note": "legacy route: plain cos/sin Euclidean distance; "
                       "any symmetry handling is a separate post-hoc merge"}),
            symmetry_broken_by=list(self.broken_by),
            caveats=self._caveats())
        # The legacy summary carried the threshold under `min_vote`; keep it on that route so an
        # existing result reader finds the same field.
        if self.vote_rule == "legacy-margin":
            out["min_vote"] = self.min_vote
        return out

    def _vote_record(self) -> Dict[str, Any]:
        keff = np.asarray(self.k_effective_)
        rec = {"rule": self.vote_rule, "k": int(self.k),
               "k_effective_min": int(keff.min()), "k_effective_max": int(keff.max())}
        if self.vote_rule == "fraction":
            rec.update(min_vote_fraction=float(self.min_vote_fraction),
                       statement=(f"assigned when n_winner / k_effective >= "
                                  f"{self.min_vote_fraction} (equality accepted); with k = 20 "
                                  f"and 0.90 that is 18 of 20. Each frame's own label is "
                                  f"excluded from its vote."))
        else:
            rec.update(min_vote_margin=float(self.min_vote),
                       statement="HISTORICAL: abstain when (n_first - n_second)/k < min_vote; the "
                                 "frame's own label is counted")
        return rec

    def _caveats(self) -> List[str]:
        out = [
            "cluster labels are ordered by descending fitted population; HDBSCAN's own numbering "
            "is arbitrary and unrelated between fits",
            "draw_agreement is PRECISION, not accuracy: the count is stable under resampling, "
            "which does not say a small state was seen",
            "a cluster is a density basin, NOT a metastable state; nothing here looks at time",
            f"two kinds of unassigned frame: {NOISE_LABEL} is DENSITY noise (thinner than any "
            f"clustered frame's neighbourhood) and {AMBIGUOUS_LABEL} is AMBIGUOUS (the vote did "
            f"not reach the threshold). An ambiguous frame is NOT evidence of a free-energy "
            f"barrier: neighbour disagreement describes the sample, not the energy. Density takes "
            f"precedence where both apply.",
            self.density_proxy_note_,
        ]
        if self.symmetry:
            out.append(
                "symmetry-related configurations are at distance zero, so clusters are of "
                "STRUCTURALLY distinct conformers. That is not a claim of equal populations for "
                "the relabelled basins inside one cluster: an atom-specific restraint, parameter "
                "or REST2 region can break the energetic symmetry without changing the graph"
                + (f" -- declared here: {self.broken_by}" if self.broken_by else "") + ".")
        if not self.allow_single_cluster:
            out.append("allow_single_cluster is False, so a UNIMODAL ensemble cannot be reported "
                       "as one state: EOM refuses the root of the condensed tree.")
        else:
            out.append("allow_single_cluster is True (this module's default, NOT sklearn's): a "
                       "unimodal ensemble CAN be one state, and a multi-state one can collapse "
                       "into one cluster; check n_clusters == 1 against the marginals.")
        if not self.weights_uniform_ and self.resampling is False:
            out.append("NON-UNIFORM weights with resampling=False: the partition was fitted on "
                       "the SAMPLED ensemble and only the populations are reweighted.")
        return out

    def __repr__(self) -> str:
        st = (f"fitted, {self.n_clusters_} clusters" if self.fitted_ else "unfitted")
        return (f"t_hdbscan(route={self.route_}, n_frames={self.n_frames_}, "
                f"n_torsions={self.n_torsions_}, resampling={self.resampling}, "
                f"vote_rule={self.vote_rule}, k={self.k}, threshold={self.vote_threshold}, {st})")


#: The name used at the call site. A class is the estimator; this alias keeps
#: `t_hdbscan(torsions, ...)` reading as specified.
t_hdbscan = THDBSCAN
