"""`t_hdbscan` -- torsional HDBSCAN as a fit/predict estimator.

WRITTEN BY THE hpREST2 SESSION and ported here with its agreement; the reasoning in this file is
theirs. It takes angles and optional per-frame weights and knows nothing about where they came
from -- no free-energy estimator, no replica ladder, no file layout.

NEEDS THE `analysis` EXTRA (`pip install '.[analysis]'`, or the `analysis-env` environment), for
scikit-learn. The import is lazy, inside the functions that use it, so `import md_tools` and
`import md_tools.analysis` cost nothing in an environment that has no sklearn and fail by name
rather than by traceback if you then ask for a fit.

USAGE

    state_classifier = t_hdbscan(torsions)                        # uniform weights, all frames
    state_classifier = t_hdbscan(torsions, resampling=4096)       # weighted draws, 5 seeds
    state_classifier = t_hdbscan(torsions, weights=w)             # MBAR weights, direct fit
    state_classifier.fit()
    cluster_idx, cluster_names = state_classifier.predict(test_torsions)

`torsions` is `(n_frames, n_torsions)` in radians by default; `weights` is `(n_frames,)` and
defaults to 1. `cluster_idx` is `(m_frames,)` of int: noise and abstentions are -1, clusters are
0, 1, 2, ... ordered by DESCENDING population, so cluster 0 is always the largest. That ordering is
chosen because HDBSCAN's own label numbers are arbitrary and unrelated between fits -- without a
canonical order, two fits of the same ensemble produce incomparable integers.

WHAT `resampling` IS FOR. HDBSCAN cannot weight its density estimate: the core distance is the
k-th nearest neighbour distance, a RANK statistic, so there is no `sample_weight` to pass. With
`resampling=False` the weights therefore enter only the POPULATIONS, and the partition is still
fitted on the sampled ensemble -- an over-sampled region still shapes it. With `resampling=N` the
fit sees the target distribution itself: N frames are drawn without replacement with probability
proportional to the weights, the fit runs on those, and every frame is classified back.

A CLUSTER HERE IS NOT A METASTABLE STATE. Nothing in this module looks at time. Two density peaks
separated by a sampled region are two clusters whatever the barrier between them, and a long-lived
state that is geometrically diffuse may be one cluster or none. Metastability needs dynamical
evidence this estimator does not have.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._torsions import (MIN_CLUSTER_FRACTION_DEFAULT, MIN_SAMPLES_FRACTION_DEFAULT,
                         MIN_VOTE_MARGIN_DEFAULT, N_SEED_DEFAULT, classify_to_clusters,
                         cluster_mass, cluster_torsions, validate_angles,
                         validate_frame_weights, weighted_resample)

__all__ = ["THDBSCAN", "t_hdbscan"]


class THDBSCAN:
    """Torsional HDBSCAN with a fit/predict interface. See the module docstring.

    Parameters mirror the names used when this was specified, not sklearn's internal ones:
    `mass_floor` is HDBSCAN's `min_cluster_size` expressed as a fraction of n, and `min_vote` is
    the classifier's abstention threshold.

    `resampling` is `False` or an int N. `seed` is the NUMBER OF INDEPENDENT DRAWS when resampling
    (default 5) and is ignored otherwise; it is not a random-number seed, and the draws themselves
    are deterministic given N so that a fit is reproducible.
    """

    def __init__(self, torsions, *, weights=None, resampling=False,
                 seed: int = N_SEED_DEFAULT,
                 mass_floor: float = MIN_CLUSTER_FRACTION_DEFAULT,
                 min_vote: float = MIN_VOTE_MARGIN_DEFAULT,
                 units: str = "radians", names: Optional[Sequence[str]] = None,
                 metric_weights=None, multiplicities=None,
                 min_samples_fraction: Optional[float] = None,
                 min_samples: Optional[int] = None, k: int = 15,
                 allow_single_cluster: bool = True):
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
        if not (0.0 <= float(min_vote) <= 1.0):
            raise ValueError("min_vote must be in [0, 1]")

        self.seed = int(seed)
        self.mass_floor = float(mass_floor)
        self.min_vote = float(min_vote)
        self.names = (None if names is None else [str(x) for x in names])
        self.metric_weights = metric_weights
        self.multiplicities = multiplicities
        self.min_samples = min_samples
        self.min_samples_fraction = min_samples_fraction
        self.k = int(k)
        # TRUE BY DEFAULT HERE, which DIVERGES from the hpREST2 original. Without it a unimodal
        # ensemble cannot come back as one state: sklearn's EOM selection refuses the root of the
        # condensed tree, so a single Gaussian blob is SPLIT rather than reported whole --
        # measured, a 4000-frame unimodal ensemble returns 3 clusters with the flag off.
        #
        # The upstream default was False on the ground that allowing the root lets a multi-state
        # ensemble collapse into it. That risk is real but it is LOUD: one cluster holding
        # everything is visible in the first line of any summary. The failure the other way is
        # silent -- three clusters with a plausible mass split and nothing saying one state was
        # never a candidate -- and a tutorial's reader is far more likely to meet a rigid molecule
        # with one basin than a multi-state ensemble they cannot recognise. Decided by this
        # package's owner on 2026-10-01; `single_cluster_excluded_by_construction` still reports
        # which way the flag was set, so neither case is inferred.
        self.allow_single_cluster = bool(allow_single_cluster)
        self.fitted_ = False

    # ---------------------------------------------------------------------------------- fitting

    def _fit_once(self, theta, traj_ids=None):
        return cluster_torsions(
            theta, units="radians", names=self.names, metric_weights=self.metric_weights,
            min_cluster_fraction=self.mass_floor, min_samples=self.min_samples,
            min_samples_fraction=self.min_samples_fraction,
            allow_single_cluster=self.allow_single_cluster, traj_ids=traj_ids)

    def fit(self) -> "THDBSCAN":
        """Fit the partition. Returns self, so `t_hdbscan(x).fit()` is one expression."""
        if self.resampling is False:
            fit = self._fit_once(self.theta_)
            train_theta, train_labels = self.theta_, fit.labels
            self.settings_ = dict(fit.settings)
            self.n_clusters_per_draw_ = [int(fit.n_clusters)]
            self.draw_agreement_ = 1.0
        else:
            draws = []
            for s in range(self.seed):
                idx = weighted_resample(self.weights_, size=self.resampling, seed=s)["indices"]
                f = self._fit_once(self.theta_[idx])
                draws.append((s, idx, f))
            counts = [int(f.n_clusters) for _, _, f in draws]
            vals, cnt = np.unique(counts, return_counts=True)
            mode = int(vals[int(np.argmax(cnt))])
            self.n_clusters_per_draw_ = counts
            self.draw_agreement_ = float(cnt.max() / len(counts))
            # THE CANONICAL PARTITION IS THE FIRST DRAW AT THE MODAL COUNT. Averaging partitions is
            # not defined -- cluster labels are not numbers to be averaged -- so one draw has to be
            # the estimator and the rest are evidence about its stability. The first at the mode is
            # chosen for determinism; `draw_agreement_` says how much trust it has earned.
            pick = next((d for d in draws if int(d[2].n_clusters) == mode), None)
            if pick is None or mode == 0:
                raise RuntimeError(f"no draw produced a cluster (counts {counts}); the mass floor "
                                   f"{self.mass_floor} may be above every basin's population")
            s, idx, f = pick
            train_theta, train_labels = self.theta_[idx], f.labels
            self.settings_ = dict(f.settings)
            self.canonical_draw_ = int(s)

        # ---- relabel 0.. by DESCENDING population, so cluster 0 is always the largest
        order = [c for c in np.unique(train_labels) if c >= 0]
        if not order:
            raise RuntimeError("every training frame is noise; nothing to classify against")
        sizes = {c: int((train_labels == c).sum()) for c in order}
        remap = {c: i for i, c in enumerate(sorted(order, key=lambda c: -sizes[c]))}
        canon = np.full(train_labels.shape, -1, dtype=np.int64)
        for c, i in remap.items():
            canon[train_labels == c] = i

        self.train_theta_, self.train_labels_ = train_theta, canon
        self.n_clusters_ = len(remap)
        self.cluster_names_ = [f"state_{i}" for i in range(self.n_clusters_)]
        self.noise_name_ = "noise"
        # populations of the FULL ensemble under this partition, with the weights
        idx_all, _ = self._assign(self.theta_)
        cm = cluster_mass(idx_all, frame_weights=self.weights_)
        self.labels_ = idx_all
        self.cluster_mass_ = {i: cm["clusters"][i]["mass"] for i in sorted(cm["clusters"])}
        self.noise_mass_ = float(cm["noise"]["mass"])
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------------------ prediction

    def _assign(self, theta) -> Tuple[np.ndarray, np.ndarray]:
        out = classify_to_clusters(
            self.train_theta_, self.train_labels_, theta, units="radians",
            metric_weights=self.metric_weights, multiplicities=self.multiplicities,
            k=self.k, min_vote_margin=self.min_vote)
        idx = np.asarray(out["labels"], dtype=np.int64)
        names = np.array([self.noise_name_ if i < 0 else self.cluster_names_[i] for i in idx],
                         dtype=object)
        return idx, names

    def predict(self, test_torsions, *, units: str = "radians") -> Tuple[np.ndarray, np.ndarray]:
        """`(cluster_idx, cluster_names)` for `(m_frames, n_torsions)` of angles.

        `cluster_idx` is `(m_frames,)` int64: -1 for noise and for frames the vote gate declined to
        place, then 0, 1, ... by descending population. `cluster_names` is the matching
        `(m_frames,)` of strings; the ordered mapping is on `.cluster_names_`.
        """
        if not self.fitted_:
            raise RuntimeError("call fit() before predict()")
        th = validate_angles(test_torsions, units, n_torsions=self.n_torsions_)
        return self._assign(th)

    def fit_predict(self, *, units: str = "radians") -> Tuple[np.ndarray, np.ndarray]:
        """Fit, then assign the training ensemble itself."""
        self.fit()
        return self.labels_, np.array(
            [self.noise_name_ if i < 0 else self.cluster_names_[i] for i in self.labels_],
            dtype=object)

    def summary(self) -> Dict[str, Any]:
        """Everything a result file should record, including what the fit does NOT establish."""
        if not self.fitted_:
            raise RuntimeError("call fit() before summary()")
        return dict(
            n_frames=int(self.n_frames_), n_torsions=int(self.n_torsions_),
            n_clusters=int(self.n_clusters_), cluster_names=list(self.cluster_names_),
            cluster_mass={int(k): float(v) for k, v in self.cluster_mass_.items()},
            noise_mass=float(self.noise_mass_),
            resampling=(False if self.resampling is False else int(self.resampling)),
            seed=int(self.seed), mass_floor=self.mass_floor, min_vote=self.min_vote,
            allow_single_cluster=self.allow_single_cluster,
            # REPORTED AS A FIELD, not only as prose. A reader who gets 3 clusters from a unimodal
            # ensemble must be able to see that 1 was impossible by construction -- EOM refuses the
            # root of the condensed tree unless allow_single_cluster is set. No docstring can do
            # this job, because the failure is silent and confident: a plausible mass split and
            # nothing saying the root was never a candidate.
            single_cluster_excluded_by_construction=(not self.allow_single_cluster),
            weights_uniform=self.weights_uniform_, settings=self.settings_,
            n_clusters_per_draw=list(self.n_clusters_per_draw_),
            draw_agreement=float(self.draw_agreement_),
            canonical_draw=getattr(self, "canonical_draw_", None),
            caveats=self._caveats())

    def _caveats(self) -> List[str]:
        """Everything a reader of a result file must know that the numbers do not say."""
        out = [
            "cluster labels are ordered by descending population; HDBSCAN's own numbering is "
            "arbitrary and unrelated between fits",
            "draw_agreement is PRECISION, not accuracy: it says the count is stable under "
            "resampling, not that a small state was seen. Check that "
            "settings.min_cluster_size >= settings.min_samples first",
            "min_vote is a BOUNDARY detector, not a density test: against a direct fit's own "
            "noise it recalls about 7 %",
            "a cluster is a density basin, NOT a metastable state; nothing here looks at time",
        ]
        if not self.allow_single_cluster:
            out.append(
                "allow_single_cluster is False, so a UNIMODAL ensemble cannot be reported as one "
                "state: EOM refuses the root of the condensed tree and the blob is split instead. "
                "Measured: a 4000-frame single Gaussian returns 3 clusters. Pass "
                "allow_single_cluster=True if one state is a possible answer.")
        if not self.weights_uniform_ and self.resampling is False:
            out.append(
                "NON-UNIFORM weights with resampling=False: the partition was fitted on the "
                "SAMPLED ensemble and only the populations are reweighted. Pass resampling=N to "
                "fit the target distribution itself.")
        return out

    def __repr__(self) -> str:
        st = (f"fitted, {self.n_clusters_} clusters" if self.fitted_ else "unfitted")
        return (f"t_hdbscan(n_frames={self.n_frames_}, n_torsions={self.n_torsions_}, "
                f"resampling={self.resampling}, seed={self.seed}, mass_floor={self.mass_floor}, "
                f"min_vote={self.min_vote}, {st})")


#: The name used at the call site. A class is the estimator; this alias keeps
#: `t_hdbscan(torsions, ...)` reading as specified.
t_hdbscan = THDBSCAN
