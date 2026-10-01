"""Contract tests for `t_hdbscan`, written against the INTERFACE and not against a file layout.

Deliberately portable: every test constructs its own angles, calls only the public surface
(`t_hdbscan(...)`, `.fit()`, `.predict()`, `.summary()`), and asserts the behaviour the contract
promises. Nothing here reads a path, a config or a project module, so the file can move to
`md_tools/analysis/tests/` unchanged -- which was md-tools-dev0.6.1's condition for accepting them.

Each test says the mistake it exists to catch. A guard that cannot be shown able to fail is not a
guard; where a test pins a number, the number was measured rather than expected.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn (which ships HDBSCAN) is an analysis-only dependency")

from md_tools.analysis._t_hdbscan import THDBSCAN, t_hdbscan


def three_blobs(rng, n=(6000, 3000, 1000), s=0.2):
    """Three well-separated blobs in 2 torsions. Separation is 2.5 rad against sigma 0.2, i.e.
    12 sigma, so no test here depends on HDBSCAN resolving a marginal case."""
    return np.vstack([rng.normal(0.0, s, size=(n[0], 2)),
                      rng.normal(2.5, s, size=(n[1], 2)),
                      rng.normal(-2.5, s, size=(n[2], 2))])


# ------------------------------------------------------------------------------- the interface


@requires_sklearn
def test_the_alias_is_the_class_so_the_call_site_reads_as_specified():
    assert t_hdbscan is THDBSCAN


@requires_sklearn
def test_fit_returns_self_so_construction_and_fitting_chain():
    rng = np.random.default_rng(0)
    c = t_hdbscan(three_blobs(rng))
    assert c.fit() is c


@requires_sklearn
def test_predict_before_fit_refuses_rather_than_returning_nonsense():
    rng = np.random.default_rng(1)
    c = t_hdbscan(three_blobs(rng))
    with pytest.raises(RuntimeError, match="fit"):
        c.predict(np.zeros((5, 2)))


@requires_sklearn
def test_predict_rejects_a_different_torsion_count():
    """An (m, n_torsions) contract is only a contract if the width is checked. Silently accepting
    a 3-torsion query against a 2-torsion fit would embed or broadcast something meaningless."""
    rng = np.random.default_rng(2)
    c = t_hdbscan(three_blobs(rng)).fit()
    with pytest.raises(ValueError):
        c.predict(np.zeros((10, 3)))


# ------------------------------------------------------------------------- the label semantics


@requires_sklearn
def test_cluster_idx_is_integer_one_dimensional_and_matches_the_query_length():
    rng = np.random.default_rng(3)
    c = t_hdbscan(three_blobs(rng)).fit()
    q = rng.uniform(-np.pi, np.pi, size=(137, 2))
    idx, names = c.predict(q)
    assert idx.ndim == 1 and idx.shape == (137,)
    assert np.issubdtype(idx.dtype, np.integer)
    assert names.shape == (137,)


@requires_sklearn
def test_labels_start_at_zero_and_are_contiguous_with_noise_at_minus_one():
    """The contract: noise is -1, clusters are 0, 1, 2, ... A gap in the numbering, or a cluster
    labelled -1, would break any caller indexing an array by cluster id."""
    rng = np.random.default_rng(4)
    c = t_hdbscan(three_blobs(rng)).fit()
    idx, _ = c.predict(np.vstack([three_blobs(rng, n=(40, 30, 20)),
                                  rng.uniform(-np.pi, np.pi, size=(60, 2))]))
    present = sorted(set(idx.tolist()))
    clusters = [v for v in present if v >= 0]
    assert clusters == list(range(len(clusters))), f"not contiguous from 0: {present}"
    assert all(v >= -1 for v in present), f"a label below -1 appeared: {present}"


@requires_sklearn
def test_labels_are_ordered_by_descending_population():
    """Forced, not cosmetic: HDBSCAN's own numbering is arbitrary and unrelated between fits, so
    without a canonical order two fits of one ensemble yield incomparable integers. Cluster 0 must
    be the largest. Planted populations here are 60/30/10 %."""
    rng = np.random.default_rng(5)
    c = t_hdbscan(three_blobs(rng)).fit()
    masses = [c.cluster_mass_[i] for i in range(c.n_clusters_)]
    assert masses == sorted(masses, reverse=True), masses
    assert c.cluster_mass_[0] == pytest.approx(0.60, abs=0.02)
    assert c.cluster_mass_[2] == pytest.approx(0.10, abs=0.02)


@requires_sklearn
def test_cluster_names_align_with_the_indices_frame_by_frame():
    rng = np.random.default_rng(6)
    c = t_hdbscan(three_blobs(rng)).fit()
    idx, names = c.predict(rng.uniform(-np.pi, np.pi, size=(200, 2)))
    for i, nm in zip(idx, names):
        assert nm == (c.noise_name_ if i < 0 else c.cluster_names_[i])


@requires_sklearn
def test_the_metric_is_periodic_so_the_wrap_does_not_split_a_basin():
    """The defect the cos/sin embedding exists to prevent: a naive clusterer on raw angles sees
    the -pi and +pi halves of ONE basin as maximally far apart.

    Two basins, one of them straddling the seam. The right answer is 2 clusters with the seam
    basin intact -- NOT 3 with it cut at +-180. Deliberately not a unimodal test: with
    allow_single_cluster False, EOM refuses the root and a single blob is split regardless of
    periodicity, so a unimodal fixture would fail for an unrelated reason (measured: 3 clusters).
    """
    rng = np.random.default_rng(7)
    seam = (rng.normal(np.pi, 0.15, size=(2000, 1)) + np.pi) % (2 * np.pi) - np.pi
    other = rng.normal(0.0, 0.15, size=(2000, 1))
    assert (seam < -3.0).sum() > 200 and (seam > 3.0).sum() > 200, "fixture must straddle the cut"
    c = t_hdbscan(np.vstack([seam, other]), mass_floor=0.05).fit()
    assert c.n_clusters_ == 2, f"the seam basin was cut: {c.n_clusters_} clusters"
    idx, _ = c.predict(np.array([[np.deg2rad(179.0)], [np.deg2rad(-179.0)],
                                 [np.deg2rad(0.0)]]))
    assert idx[0] == idx[1], "+179 and -179 deg must land in the same cluster"
    assert idx[2] != idx[0], "the other basin must be a different cluster"


@requires_sklearn
def test_a_unimodal_ensemble_needs_allow_single_cluster_to_come_back_as_one_state():
    """A limitation worth pinning rather than discovering. sklearn's EOM selection refuses the
    root of the condensed tree unless allow_single_cluster is set, so one Gaussian blob is SPLIT
    when it is off. Both halves of this are measured.

    MIGRATED, not rewritten: this test was written when the flag defaulted to False and relied on
    that default. md-tools defaults it to TRUE, so the off case now says so explicitly. The
    behaviour being pinned is unchanged -- which is the point of making the flag explicit here
    rather than deleting the half that used to come for free."""
    rng = np.random.default_rng(71)
    th = rng.normal(0.0, 0.15, size=(4000, 1))
    assert t_hdbscan(th, mass_floor=0.02, allow_single_cluster=False).fit().n_clusters_ > 1
    assert t_hdbscan(th, mass_floor=0.02, allow_single_cluster=True).fit().n_clusters_ == 1


@requires_sklearn
def test_the_default_still_separates_a_genuinely_multi_state_ensemble():
    """The cost of allow_single_cluster=True is the OPPOSITE error -- a real multi-state ensemble
    collapsing into one cluster when the between-basin density does not beat the root's stability.
    This pins that it does not happen on well-separated basins, which is what makes the default
    safe to take rather than merely convenient."""
    rng = np.random.default_rng(72)
    c = t_hdbscan(three_blobs(rng)).fit()
    assert c.allow_single_cluster is True
    assert c.n_clusters_ == 3


@requires_sklearn
def test_one_state_is_a_possible_answer_by_default():
    """THE DEFAULT ITSELF, pinned. md-tools sets allow_single_cluster=True so that a rigid
    molecule with one torsional basin is REPORTED as one state rather than split into a plausible
    three. The upstream package defaults it to False; a silent change back would make every
    unimodal result wrong in a way no caller could see, so the default is a test."""
    rng = np.random.default_rng(71)
    th = rng.normal(0.0, 0.15, size=(4000, 1))
    fit = t_hdbscan(th, mass_floor=0.02).fit()
    assert fit.allow_single_cluster is True, "md-tools defaults allow_single_cluster to True"
    assert fit.n_clusters_ == 1, "a unimodal ensemble must come back as one state by default"
    assert fit.summary()["single_cluster_excluded_by_construction"] is False


# ------------------------------------------------------------------------------ the two paths


@requires_sklearn
def test_resampling_is_off_by_default():
    rng = np.random.default_rng(8)
    c = t_hdbscan(three_blobs(rng))
    assert c.resampling is False
    assert c.mass_floor == 0.01 and c.min_vote == 0.9 and c.seed == 5


@requires_sklearn
def test_weights_default_to_one_and_uniform_weights_give_frame_fractions():
    rng = np.random.default_rng(9)
    th = three_blobs(rng)
    a = t_hdbscan(th).fit()
    b = t_hdbscan(th, weights=np.ones(len(th))).fit()
    assert a.weights_uniform_ and b.weights_uniform_
    assert np.array_equal(a.labels_, b.labels_)
    for i in range(a.n_clusters_):
        assert a.cluster_mass_[i] == pytest.approx(b.cluster_mass_[i])
        frac = float((a.labels_ == i).mean())
        assert a.cluster_mass_[i] == pytest.approx(frac)


@requires_sklearn
def test_resampling_an_int_fits_that_many_frames_over_that_many_draws():
    rng = np.random.default_rng(10)
    c = t_hdbscan(three_blobs(rng), resampling=4096, seed=5).fit()
    assert c.resampling == 4096
    assert len(c.n_clusters_per_draw_) == 5
    assert c.draw_agreement_ == 1.0 and c.n_clusters_ == 3
    assert c.canonical_draw_ in range(5)


@requires_sklearn
def test_resampling_clamps_to_the_ensemble_size_rather_than_sampling_with_replacement():
    """Replacement is refused in the resampler because duplicate frames give zero distances and a
    core distance of 0, which silently corrupts the density estimate."""
    rng = np.random.default_rng(11)
    th = three_blobs(rng, n=(500, 300, 200))
    c = t_hdbscan(th, resampling=4096, seed=2).fit()
    assert c.resampling == 1000


@requires_sklearn
def test_resampling_makes_the_FIT_follow_the_weights_not_the_frame_counts():
    """The difference between the two paths, stated as a measurement. The large blob holds 60 % of
    the FRAMES and is down-weighted 20x; with resampling the partition is fitted on the target, so
    the up-weighted blobs must carry most of the mass."""
    rng = np.random.default_rng(12)
    th = three_blobs(rng)
    w = np.ones(len(th)); w[:6000] = 0.05
    c = t_hdbscan(th, weights=w, resampling=4096, seed=3).fit()
    assert c.n_clusters_ == 3
    big_by_frames = float((c.labels_ == 0).mean())
    assert c.cluster_mass_[0] + c.cluster_mass_[1] > 0.7
    assert big_by_frames != pytest.approx(c.cluster_mass_[0], abs=0.05)


@requires_sklearn
def test_direct_fit_with_non_uniform_weights_declares_the_weaker_claim():
    """With resampling off the partition is fitted on the SAMPLED ensemble and only the
    populations are reweighted. That must appear in summary(), because a caller cannot otherwise
    tell which of the two things they got."""
    rng = np.random.default_rng(13)
    th = three_blobs(rng)
    w = np.ones(len(th)); w[:6000] = 0.05
    s = t_hdbscan(th, weights=w, resampling=False).fit().summary()
    assert s["weights_uniform"] is False and s["resampling"] is False
    assert any("only the populations are reweighted" in c for c in s["caveats"])


# ----------------------------------------------------------------------- floors and validation


@requires_sklearn
@pytest.mark.parametrize("n", [1024, 4096, 16384])
def test_the_mass_floor_stays_enforceable_at_every_n(n):
    """Why min_samples is a FRACTION. With a pinned count the 1 % floor sits below min_samples for
    small n and cannot bind -- measured, a planted well-separated 1 % cluster is then recovered
    0/5 times while all five seeds agree on the wrong count."""
    rng = np.random.default_rng(14)
    th = rng.uniform(-np.pi, np.pi, size=(n, 2))
    s = t_hdbscan(th, mass_floor=0.01).fit().summary() if n > 2000 else None
    c = t_hdbscan(th, mass_floor=0.01)
    c.fit() if s is None else None
    st = (s or c.summary())["settings"]
    assert st["min_cluster_size"] >= st["min_samples"], (
        f"floor {st['min_cluster_size']} below the density estimate {st['min_samples']} at n={n}")


@requires_sklearn
def test_a_planted_one_percent_cluster_is_actually_found():
    """The floor must be able to BIND, not merely be configured. Three blobs with the third at
    exactly 1 % of n, well separated."""
    rng = np.random.default_rng(15)
    n = 8000
    k = n // 100
    th = np.vstack([rng.normal(0.0, 0.08, size=((n - k) // 2, 3)),
                    rng.normal(2.1, 0.08, size=(n - k - (n - k) // 2, 3)),
                    rng.normal(-2.1, 0.08, size=(k, 3))])
    c = t_hdbscan(th, mass_floor=0.01).fit()
    assert c.n_clusters_ == 3
    assert min(c.cluster_mass_.values()) == pytest.approx(0.01, abs=0.004)


@requires_sklearn
def test_min_vote_zero_commits_every_frame_and_a_high_gate_abstains():
    rng = np.random.default_rng(16)
    th = three_blobs(rng)
    q = np.vstack([three_blobs(rng, n=(50, 50, 50)), rng.uniform(-np.pi, np.pi, size=(150, 2))])
    open_idx, _ = t_hdbscan(th, min_vote=0.0).fit().predict(q)
    gated_idx, _ = t_hdbscan(th, min_vote=0.9).fit().predict(q)
    assert (open_idx < 0).sum() == 0
    assert (gated_idx < 0).sum() > (open_idx < 0).sum()


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
def test_mass_floor_is_range_checked(bad):
    with pytest.raises(ValueError, match="mass_floor"):
        t_hdbscan(np.zeros((100, 2)), mass_floor=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_min_vote_is_range_checked(bad):
    with pytest.raises(ValueError, match="min_vote"):
        t_hdbscan(np.zeros((100, 2)), min_vote=bad)


@pytest.mark.parametrize("bad", [1, 0, -5])
def test_resampling_must_be_false_or_at_least_two(bad):
    with pytest.raises(ValueError, match="resampling"):
        t_hdbscan(np.zeros((100, 2)), resampling=bad)


def test_seed_count_must_be_positive_when_resampling():
    with pytest.raises(ValueError, match="seed"):
        t_hdbscan(np.zeros((100, 2)), resampling=50, seed=0)


def test_weight_length_must_match_the_frame_count():
    with pytest.raises(ValueError):
        t_hdbscan(np.zeros((100, 2)), weights=np.ones(99))


# ------------------------------------------------------------------------------------ summary


@requires_sklearn
def test_summary_records_the_defaults_and_the_three_standing_caveats():
    """A result file has to be readable without the code that produced it."""
    rng = np.random.default_rng(17)
    s = t_hdbscan(three_blobs(rng)).fit().summary()
    assert s["mass_floor"] == 0.01 and s["min_vote"] == 0.9 and s["resampling"] is False
    assert s["n_clusters"] == len(s["cluster_names"]) == len(s["cluster_mass"])
    joined = " ".join(s["caveats"])
    assert "NOT a metastable state" in joined
    assert "PRECISION, not accuracy" in joined
    assert "BOUNDARY detector" in joined


@requires_sklearn
def test_masses_and_noise_sum_to_one():
    rng = np.random.default_rng(18)
    c = t_hdbscan(three_blobs(rng), min_vote=0.9).fit()
    assert sum(c.cluster_mass_.values()) + c.noise_mass_ == pytest.approx(1.0)


@requires_sklearn
def test_summary_says_when_a_single_cluster_was_impossible_by_construction():
    """The defect that matters is silent: a unimodal ensemble returns several clusters with a
    plausible mass split and nothing saying the root was never a candidate. summary() must carry it
    as a FIELD, so a reader who gets 3 can see why 1 was excluded."""
    rng = np.random.default_rng(19)
    th = rng.normal(0.0, 0.15, size=(4000, 1))
    off = t_hdbscan(th, mass_floor=0.02, allow_single_cluster=False).fit().summary()
    on = t_hdbscan(th, mass_floor=0.02, allow_single_cluster=True).fit().summary()
    assert off["single_cluster_excluded_by_construction"] is True
    assert off["allow_single_cluster"] is False and off["n_clusters"] > 1
    assert on["single_cluster_excluded_by_construction"] is False
    assert on["n_clusters"] == 1
    assert any("UNIMODAL" in c for c in off["caveats"])
    assert not any("UNIMODAL" in c for c in on["caveats"])
