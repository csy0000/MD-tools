"""Contract tests for `t_hdbscan`, written against the INTERFACE and not against a file layout.

Deliberately portable: every test constructs its own angles, calls only the public surface
(`t_hdbscan(...)`, `.fit()`, `.predict()`, `.summary()`), and asserts the behaviour the contract
promises. Nothing here reads a path, a config or a project module, so the file can move to
`md_tools/analysis/tests/` unchanged -- which was md-tools-dev0.6.1's condition for accepting them.

Each test says the mistake it exists to catch. A guard that cannot be shown able to fail is not a
guard; where a test pins a number, the number was measured rather than expected.

MIGRATED FOR THE SYMMETRY-FIRST DEFAULT. `t_hdbscan(angles)` now refuses, because angles alone
cannot establish a molecular symmetry. These tests are about the density estimator, its labels,
its mass floor and its resampling, which the legacy-distance route (`symmetry=False`) shares with
the default; so the name `t_hdbscan` below is that route. The vote is the new default (k = 20,
absolute fraction 0.90) except where a test is about the historical margin and says so with
`vote_rule='legacy-margin'`. The default route's own tests are in
`test_analysis_symmetry_first.py`.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn (which ships HDBSCAN) is an analysis-only dependency")

from md_tools.analysis._t_hdbscan import (ALLOW_SINGLE_CLUSTER_DEFAULT, AMBIGUOUS_LABEL,
                                          BARRIER_LABEL, NOISE_LABEL, THDBSCAN)
from md_tools.analysis._t_hdbscan import t_hdbscan as _public_t_hdbscan


def t_hdbscan(*args, **kwargs):
    """The legacy-distance route, which is what an angles-only call means now. See the header."""
    kwargs.setdefault("symmetry", False)
    return THDBSCAN(*args, **kwargs)


def three_blobs(rng, n=(6000, 3000, 1000), s=0.2):
    """Three well-separated blobs in 2 torsions. Separation is 2.5 rad against sigma 0.2, i.e.
    12 sigma, so no test here depends on HDBSCAN resolving a marginal case."""
    return np.vstack([rng.normal(0.0, s, size=(n[0], 2)),
                      rng.normal(2.5, s, size=(n[1], 2)),
                      rng.normal(-2.5, s, size=(n[2], 2))])


# ------------------------------------------------------------------------------- the interface


@requires_sklearn
def test_the_alias_is_the_class_so_the_call_site_reads_as_specified():
    assert _public_t_hdbscan is THDBSCAN


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
    """The contract: clusters are 0, 1, 2, ... contiguous. A gap in the numbering, or a cluster
    with a negative label, would break any caller indexing an array by cluster id."""
    rng = np.random.default_rng(4)
    c = t_hdbscan(three_blobs(rng)).fit()
    idx, _ = c.predict(np.vstack([three_blobs(rng, n=(40, 30, 20)),
                                  rng.uniform(-np.pi, np.pi, size=(60, 2))]))
    present = sorted(set(idx.tolist()))
    clusters = [v for v in present if v >= 0]
    assert clusters == list(range(len(clusters))), f"not contiguous from 0: {present}"
    assert all(v in (NOISE_LABEL, BARRIER_LABEL) or v >= 0 for v in present), \
        f"an unexpected negative label appeared: {present}"


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
        want = (c.noise_name_ if i == NOISE_LABEL else
                c.barrier_name_ if i == BARRIER_LABEL else c.cluster_names_[i])
        assert nm == want


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
def test_a_unimodal_ensemble_comes_back_as_ONE_state_by_default():
    """The reason this module's default departs from sklearn's. With allow_single_cluster False,
    EOM refuses the root of the condensed tree and one Gaussian blob is SPLIT -- a silent,
    confident error that reports several plausible populations where the truth is one state. Both
    directions are measured, because the default is only defensible if the False behaviour is
    actually wrong on this fixture."""
    rng = np.random.default_rng(71)
    th = rng.normal(0.0, 0.15, size=(4000, 1))
    assert t_hdbscan(th, mass_floor=0.02).fit().n_clusters_ == 1, "the default must allow one state"
    assert t_hdbscan(th, mass_floor=0.02, allow_single_cluster=False).fit().n_clusters_ > 1


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


# ------------------------------------------------------------------------------ the two paths


@requires_sklearn
def test_resampling_is_off_by_default():
    rng = np.random.default_rng(8)
    c = t_hdbscan(three_blobs(rng))
    assert c.resampling is False
    assert c.mass_floor == 0.01 and c.seed == 5
    # the vote default: 20 neighbours, absolute winning fraction 0.90 (18 of 20)
    assert c.vote_rule == "fraction" and c.k == 20 and c.min_vote_fraction == 0.9
    assert c.min_vote is None
    # NOT sklearn's default, deliberately: see ALLOW_SINGLE_CLUSTER_DEFAULT
    assert c.allow_single_cluster is ALLOW_SINGLE_CLUSTER_DEFAULT is True


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
def test_min_vote_controls_BARRIER_frames_only_not_density_noise():
    """min_vote=0 abstains from nothing, so no BARRIER frames -- but density noise is a separate
    verdict from separate machinery and must survive. Before the two codes were split this test
    asserted "nothing negative", which silently also asserted that HDBSCAN's noise had been
    overwritten."""
    # THE FIXTURE MATTERS AND THE OBVIOUS ONE IS WRONG. Well-separated blobs with empty space
    # between them produce NO barrier frames at all: every frame the gate would refuse is also
    # sparse, so density precedence claims it as -2. A barrier frame must be DENSE and SPLIT, which
    # needs a populated region between basins -- here a uniform background that bridges them.
    rng = np.random.default_rng(16)
    th = np.concatenate([rng.normal(np.deg2rad(-90), np.deg2rad(8), 2000),
                         rng.normal(np.deg2rad(90), np.deg2rad(8), 2000),
                         rng.uniform(-np.pi, np.pi, 200)])[:, None]
    open_idx, _ = t_hdbscan(th, mass_floor=0.02, vote_rule="legacy-margin",
                            min_vote=0.0).fit().predict(th)
    gated_idx, _ = t_hdbscan(th, mass_floor=0.02, vote_rule="legacy-margin",
                             min_vote=0.9).fit().predict(th)
    assert (open_idx == BARRIER_LABEL).sum() == 0
    assert (gated_idx == BARRIER_LABEL).sum() > 0
    # density noise is a separate verdict and appears under BOTH gates, unchanged by min_vote
    assert (open_idx == NOISE_LABEL).sum() > 0
    assert (gated_idx == NOISE_LABEL).sum() > 0
    assert (open_idx == NOISE_LABEL).sum() == (gated_idx == NOISE_LABEL).sum()


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
def test_mass_floor_is_range_checked(bad):
    with pytest.raises(ValueError, match="mass_floor"):
        t_hdbscan(np.zeros((100, 2)), mass_floor=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_min_vote_is_range_checked(bad):
    with pytest.raises(ValueError, match="min_vote"):
        t_hdbscan(np.zeros((100, 2)), min_vote_fraction=bad)
    with pytest.raises(ValueError, match="min_vote"):
        t_hdbscan(np.zeros((100, 2)), vote_rule="legacy-margin", min_vote=bad)


def test_a_legacy_margin_is_never_read_as_a_fraction():
    """`min_vote` was a MARGIN. Accepting it under the new rule would silently change what 0.9
    means; it is refused with the two ways forward."""
    with pytest.raises(TypeError, match="min_vote_fraction=.*vote_rule='legacy-margin'"):
        t_hdbscan(np.zeros((100, 2)), min_vote=0.9)


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
    assert s["mass_floor"] == 0.01 and s["resampling"] is False
    assert s["vote"]["rule"] == "fraction" and s["vote"]["k"] == 20
    assert s["vote"]["min_vote_fraction"] == 0.9 and "18 of 20" in s["vote"]["statement"]
    assert s["n_clusters"] == len(s["cluster_names"]) == len(s["cluster_mass"])
    joined = " ".join(s["caveats"])
    assert "NOT a metastable state" in joined
    assert "PRECISION, not accuracy" in joined
    assert "NOT evidence of a free-energy barrier" in joined


@requires_sklearn
def test_masses_and_noise_sum_to_one():
    rng = np.random.default_rng(18)
    c = t_hdbscan(three_blobs(rng)).fit()
    # `noise_mass_` is DENSITY noise only; the vote's refusals are `ambiguous_mass_`
    assert sum(c.cluster_mass_.values()) + c.noise_mass_ + c.ambiguous_mass_ == \
        pytest.approx(1.0)
    assert c.unassigned_mass_ == pytest.approx(c.noise_mass_ + c.ambiguous_mass_)


@requires_sklearn
def test_summary_says_when_a_single_cluster_was_impossible_by_construction():
    """The defect that matters is silent: a unimodal ensemble returns several clusters with a
    plausible mass split and nothing saying the root was never a candidate. summary() must carry it
    as a FIELD, so a reader who gets 3 can see why 1 was excluded."""
    rng = np.random.default_rng(19)
    th = rng.normal(0.0, 0.15, size=(4000, 1))
    off = t_hdbscan(th, mass_floor=0.02, allow_single_cluster=False).fit().summary()
    on = t_hdbscan(th, mass_floor=0.02).fit().summary()
    assert off["single_cluster_excluded_by_construction"] is True
    assert off["allow_single_cluster"] is False and off["n_clusters"] > 1
    assert on["single_cluster_excluded_by_construction"] is False
    assert on["n_clusters"] == 1
    assert any("UNIMODAL" in c for c in off["caveats"])
    assert not any("UNIMODAL" in c for c in on["caveats"])


# ---------------------------------------------------- the two unassigned codes (user, 2026-10-01)


@requires_sklearn
def test_noise_is_minus_two_and_barrier_is_minus_one():
    """The two kinds of unassigned are DIFFERENT CLAIMS and must not share a code: -2 density
    noise ("nowhere in particular"), -1 barrier ("between two somewheres"). Collapsing them
    discarded one verdict -- the single-code version silently committed most of HDBSCAN's noise
    frames to clusters."""
    assert NOISE_LABEL == -2 and AMBIGUOUS_LABEL == -1 and BARRIER_LABEL == AMBIGUOUS_LABEL
    rng = np.random.default_rng(20)
    th = np.concatenate([rng.normal(np.deg2rad(-90), np.deg2rad(8), 2000),
                         rng.normal(np.deg2rad(90), np.deg2rad(8), 2000),
                         rng.uniform(-np.pi, np.pi, 200)])[:, None]
    c = t_hdbscan(th, mass_floor=0.02).fit()
    idx, names = c.predict(th)
    present = set(idx.tolist())
    assert NOISE_LABEL in present, "a uniform background must produce density noise"
    assert BARRIER_LABEL in present, "two basins must produce some barrier frames"
    assert set(names[idx == NOISE_LABEL]) == {"noise"}
    assert set(names[idx == BARRIER_LABEL]) == {"ambiguous"}


@requires_sklearn
def test_density_takes_precedence_over_the_barrier_code():
    """Where both apply, -2 wins: a frame too sparse to belong anywhere is not meaningfully on a
    barrier between clusters it is not near. Pinned because the opposite precedence would report
    isolated frames as barriers, which reads as a transition state."""
    rng = np.random.default_rng(21)
    th = np.concatenate([rng.normal(np.deg2rad(-90), np.deg2rad(8), 1500),
                         rng.normal(np.deg2rad(90), np.deg2rad(8), 1500),
                         rng.uniform(-np.pi, np.pi, 300)])[:, None]
    c = t_hdbscan(th, mass_floor=0.02).fit()
    idx, _ = c.predict(th)
    sparse = c._query_core(th) > c.density_threshold_
    assert np.all(idx[sparse] == NOISE_LABEL), "a sparse frame must be -2 whatever the vote said"


@requires_sklearn
def test_min_vote_zero_still_reports_density_noise():
    """Turning the gate off must remove BARRIER frames and leave NOISE frames: the two codes come
    from independent machinery, so one switch must not silence the other."""
    rng = np.random.default_rng(22)
    th = np.concatenate([rng.normal(np.deg2rad(-90), np.deg2rad(8), 1500),
                         rng.normal(np.deg2rad(90), np.deg2rad(8), 1500),
                         rng.uniform(-np.pi, np.pi, 300)])[:, None]
    idx, _ = t_hdbscan(th, mass_floor=0.02, vote_rule="legacy-margin",
                       min_vote=0.0).fit().predict(th)
    assert (idx == BARRIER_LABEL).sum() == 0
    assert (idx == NOISE_LABEL).sum() > 0


@requires_sklearn
def test_the_density_proxy_reports_its_own_agreement_with_hdbscan():
    """`predict` must judge frames the fit never saw, but HDBSCAN's noise label is a statement
    about the condensed tree and exists only for fitted frames. So -2 uses a core-distance PROXY,
    and the result must report how well that proxy reproduces the real verdict rather than imply
    it is the same criterion."""
    rng = np.random.default_rng(23)
    th = np.concatenate([rng.normal(np.deg2rad(-90), np.deg2rad(8), 2000),
                         rng.normal(np.deg2rad(90), np.deg2rad(8), 2000),
                         rng.uniform(-np.pi, np.pi, 200)])[:, None]
    s = t_hdbscan(th, mass_floor=0.02).fit().summary()
    assert s["noise_label"] == -2 and s["ambiguous_label"] == -1
    assert 0.0 < s["density_threshold"] < np.inf
    assert 0.0 <= s["density_proxy_agreement"] <= 1.0
    assert s["density_proxy_agreement"] > 0.95
    assert "PROXY" in s["density_proxy_note"]
    assert any("two kinds of unassigned" in c and "AMBIGUOUS" in c for c in s["caveats"])
