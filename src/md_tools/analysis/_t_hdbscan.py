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
                         cluster_mass, cluster_torsions, embed, validate_angles,
                         validate_frame_weights, validate_metric_weights,
                         weighted_resample)

#: DEFAULT TRUE since 2026-10-01, which is NOT sklearn's default and is the one place this
#: estimator deliberately departs from it.
#:
#: sklearn's EOM selection refuses the root of the condensed tree unless this is set, so with it
#: False a UNIMODAL ensemble cannot be reported as one state -- the blob is split instead, and the
#: failure is silent and confident: measured, a 4000-frame single Gaussian returns 2-3 clusters
#: with a plausible mass split and nothing in the output saying one state was never a candidate.
#: For a torsional ensemble at 300 K, and especially for a molecule with few rotatable bonds,
#: unimodal is an ordinary answer, so a default that cannot express it invents structure in a
#: common case.
#:
#: THE COST OF TRUE IS REAL AND POINTS THE OTHER WAY: allowing the root lets a genuinely
#: multi-state ensemble collapse into one cluster when the between-basin density is not low enough
#: to beat the root's stability. That is a false negative where False gives a false positive. The
#: trade was decided deliberately: a merged partition reports ONE population, which is visibly
#: wrong and prompts a check, whereas a split unimodal blob reports several plausible populations
#: that look like a result. Over-splitting is the harder error to notice, so it is the one the
#: default avoids. `summary()` records the flag either way.
ALLOW_SINGLE_CLUSTER_DEFAULT = True

#: Label for a frame the DENSITY test excludes: its neighbourhood is thinner than that of any
#: frame the fit put in a cluster. "Nowhere in particular."
NOISE_LABEL = -2

#: Label for a frame the VOTE gate declines: its neighbours disagree about which cluster it belongs
#: to. "Between two somewheres" -- a barrier frame.
BARRIER_LABEL = -1

__all__ = ["ALLOW_SINGLE_CLUSTER_DEFAULT", "BARRIER_LABEL", "NOISE_LABEL", "THDBSCAN",
           "t_hdbscan"]


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
                 allow_single_cluster: bool = ALLOW_SINGLE_CLUSTER_DEFAULT):
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
        self._fit_density_threshold()
        self.cluster_names_ = [f"state_{i}" for i in range(self.n_clusters_)]
        self.noise_name_ = "noise"
        self.barrier_name_ = "barrier"
        # populations of the FULL ensemble under this partition, with the weights
        idx_all, _ = self._assign(self.theta_)
        cm = cluster_mass(idx_all, frame_weights=self.weights_)
        self.labels_ = idx_all
        self.cluster_mass_ = {i: cm["clusters"][i]["mass"] for i in sorted(cm["clusters"])}
        self.noise_mass_ = float(cm["noise"]["mass"])
        self.fitted_ = True
        return self

    def _fit_density_threshold(self) -> None:
        """The density scale a query frame must meet to be eligible for a cluster at all.

        WHY A PROXY IS NEEDED. HDBSCAN's noise label is a statement about the condensed tree: a
        frame is noise if it fell out below the birth lambda of every selected cluster. That is
        defined only for frames that were IN the fit. `predict` must judge frames the fit never
        saw, so it needs a density criterion computable against the training set.

        THE CRITERION: a query frame's distance to its `min_samples`-th nearest TRAINING neighbour,
        compared against the largest such distance among training frames the fit put in a cluster.
        Denser than the sparsest clustered frame means eligible. It is the same quantity HDBSCAN's
        core distance uses, measured the same way, which is what makes the comparison meaningful.

        IT IS NOT HDBSCAN'S CRITERION AND DOES NOT REPRODUCE IT. A threshold on one core distance
        cannot express a tree. `density_agreement_` records how often it agrees with HDBSCAN's own
        verdict on the training frames, so the proxy's quality is a measured number rather than an
        assumption.
        """
        X = embed(self.train_theta_, validate_metric_weights(self.metric_weights,
                                                             self.n_torsions_),
                  self.multiplicities)
        ms = int(self.settings_.get("min_samples") or 5)
        ms = max(1, min(ms, X.shape[0] - 1))
        W = float(np.sum(validate_metric_weights(self.metric_weights, self.n_torsions_)))
        d = np.empty(X.shape[0])
        for lo in range(0, X.shape[0], 4096):
            hi = min(lo + 4096, X.shape[0])
            D2 = np.maximum((2.0 * W) - 2.0 * (X[lo:hi] @ X.T), 0.0)
            # ms-th nearest EXCLUDING self, so index ms in the sorted row
            part = np.partition(D2, ms, axis=1)[:, ms]
            d[lo:hi] = np.sqrt(part)
        self._train_core_ = d
        self._core_k_ = ms
        inc = self.train_labels_ >= 0
        self.density_threshold_ = (float(d[inc].max()) if inc.any() else float("inf"))
        # how well the proxy reproduces HDBSCAN's own verdict on the frames it DID judge
        proxy_noise = d > self.density_threshold_
        hdb_noise = self.train_labels_ < 0
        self.density_agreement_ = float(np.mean(proxy_noise == hdb_noise))
        self.density_proxy_note_ = (
            f"a query frame is NOISE_LABEL ({NOISE_LABEL}) if its distance to its {ms}-th nearest "
            f"training neighbour exceeds {self.density_threshold_:.6g}, the largest such distance "
            f"among clustered training frames. This is a PROXY for HDBSCAN's tree criterion, not "
            f"that criterion: it agrees with HDBSCAN's own verdict on "
            f"{100 * self.density_agreement_:.2f} % of the training frames.")

    # ------------------------------------------------------------------------------ prediction

    def _query_core(self, theta) -> np.ndarray:
        """Distance from each query frame to its `min_samples`-th nearest TRAINING neighbour."""
        w = validate_metric_weights(self.metric_weights, self.n_torsions_)
        Xt = embed(self.train_theta_, w, self.multiplicities)
        Xq = embed(theta, w, self.multiplicities)
        W = float(np.sum(w))
        ms = max(1, min(self._core_k_, Xt.shape[0] - 1))
        out = np.empty(Xq.shape[0])
        for lo in range(0, Xq.shape[0], 4096):
            hi = min(lo + 4096, Xq.shape[0])
            D2 = np.maximum((2.0 * W) - 2.0 * (Xq[lo:hi] @ Xt.T), 0.0)
            out[lo:hi] = np.sqrt(np.partition(D2, ms - 1, axis=1)[:, ms - 1])
        return out

    def _assign(self, theta) -> Tuple[np.ndarray, np.ndarray]:
        out = classify_to_clusters(
            self.train_theta_, self.train_labels_, theta, units="radians",
            metric_weights=self.metric_weights, multiplicities=self.multiplicities,
            k=self.k, min_vote_margin=self.min_vote)
        idx = np.asarray(out["labels"], dtype=np.int64)
        # The gate's refusals are BARRIER frames: neighbours disagree about which cluster.
        idx[idx < 0] = BARRIER_LABEL
        # DENSITY TAKES PRECEDENCE. "Nowhere in particular" is the stronger statement than
        # "between two somewheres": a frame too sparse to belong anywhere is not meaningfully on a
        # barrier between clusters it is not near. So NOISE_LABEL overwrites BARRIER_LABEL.
        idx[self._query_core(theta) > self.density_threshold_] = NOISE_LABEL
        names = np.array([self.noise_name_ if i == NOISE_LABEL
                          else self.barrier_name_ if i == BARRIER_LABEL
                          else self.cluster_names_[i] for i in idx], dtype=object)
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
            noise_label=NOISE_LABEL, barrier_label=BARRIER_LABEL,
            density_threshold=float(self.density_threshold_),
            density_proxy_agreement=float(self.density_agreement_),
            density_proxy_note=self.density_proxy_note_,
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
            f"two kinds of unassigned, and they are different claims: {NOISE_LABEL} is DENSITY "
            f"noise (the neighbourhood is thinner than any clustered frame's -- nowhere in "
            f"particular) and {BARRIER_LABEL} is a BARRIER frame (the neighbours disagree about "
            f"which cluster -- between two somewheres). Measured on a two-basin fixture they "
            f"overlap only partly: a frame can be sparse yet unanimous, or dense yet split. "
            f"Density takes precedence where both apply.",
            self.density_proxy_note_,
        ]
        if not self.allow_single_cluster:
            out.append(
                "allow_single_cluster is False, so a UNIMODAL ensemble cannot be reported as one "
                "state: EOM refuses the root of the condensed tree and the blob is split instead. "
                "Measured: a 4000-frame single Gaussian returns 2-3 clusters. Pass "
                "allow_single_cluster=True if one state is a possible answer.")
        else:
            out.append(
                "allow_single_cluster is True (this module's default, NOT sklearn's): a unimodal "
                "ensemble CAN be reported as one state. The cost is the opposite error -- a "
                "genuinely multi-state ensemble can collapse into one cluster if the "
                "between-basin density does not beat the root's stability. If n_clusters is 1, "
                "check it against the torsion marginals before believing it.")
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
