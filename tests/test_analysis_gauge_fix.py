"""A gauge on one torsion column, for drawing: the window is a PRESENTATION cut, not a distance.

`align_to_reference(gauge=(column, lo, hi))` keeps only the operations that write `column` in the
half-open window and takes the nearest to the reference among them. Paracetamol's ring flip shifts
the `aryl` and the `phenol` torsion by half a turn AT ONCE, so it is the joint symmetry this exists
for: fix `aryl` to a half-circle and `phenol` follows it, never folded on its own.

Each test names the mistake it catches: the window folded on one column while the partner is left
behind, a frame on the cut that is not reported, a window nothing can satisfy that is silently
filled, and a gauge that leaks into the clustering.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")

from md_tools.analysis._symmetry_metric import SymmetricTorsionDescriptor        # noqa: E402
from md_tools.analysis._t_symmetry import wrap                                   # noqa: E402

from test_analysis_dirichlet_alignment import (_closed, _ensemble, _fit, _frames, _para,   # noqa: E402
                                               requires_sklearn)

HALF = (-np.pi / 2, np.pi / 2)


def _setup(settings):
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    return mol, d, _closed(d, _frames(mol, settings), mol)


def test_the_gauge_column_lands_in_the_window_and_its_partner_moves_with_it():
    rng = np.random.default_rng(1)
    settings = np.column_stack([rng.uniform(-np.pi, np.pi, 300), rng.uniform(-np.pi, np.pi, 300)])
    mol, d, th = _setup(settings)
    out = d.align_to_reference(th, th[0], gauge=("aryl", *HALF))
    aryl, phenol = out["angles"][:, 1], out["angles"][:, 2]
    ok = out["gauge_satisfied"]
    # Every frame lands in the window except those ON the cut, where the two images of the ring
    # torsion are half a turn apart only up to the coordinate noise: both are then just outside.
    assert ok.mean() > 0.95
    assert ((aryl[ok] >= HALF[0] - 1e-9) & (aryl[ok] < HALF[1] + 1e-9)).all()
    assert (out["gauge_edge_distance"][~ok] < 0.05).all()
    # The pair is either left alone or shifted TOGETHER by half a turn; (pi, 0) would mean one
    # column was folded without the other.
    d_aryl = np.abs(wrap(aryl - th[:, 1]))[ok]
    d_phenol = np.abs(wrap(phenol - th[:, 2]))[ok]
    assert np.allclose(np.where(d_aryl > 1.0, np.pi, 0.0), np.where(d_phenol > 1.0, np.pi, 0.0))
    assert (d_aryl > 1.0).any() and (d_aryl < 1.0).any()          # both cases occur here
    # whole-row operation, no frame lost
    for row, aligned, op in zip(th, out["angles"], out["operations"]):
        assert np.allclose(wrap(aligned - d.transform(row[None], int(op))[0]), 0.0, atol=1e-12)


def test_a_frame_and_its_relabelled_copy_are_written_identically_under_a_gauge():
    from test_analysis_dirichlet_alignment import _flip

    mol, d, th = _setup([(0.6, 0.2), (-1.0, 3.0), (2.5, -2.0)])
    xyz = _frames(mol, [(0.6, 0.2), (-1.0, 3.0), (2.5, -2.0)])
    th_rel = _closed(d, xyz[:, list(_flip(d.enumeration)), :], mol)
    a = d.align_to_reference(th, th[0], gauge=("aryl", *HALF))
    b = d.align_to_reference(th_rel, th[0], gauge=("aryl", *HALF))
    assert np.allclose(wrap(a["angles"] - b["angles"]), 0.0, atol=1e-9)


def test_a_frame_on_the_cut_is_reported_as_near_the_edge():
    mol, d, th = _setup([(np.pi / 2 - 0.004, 0.2), (0.0, 0.2)])
    out = d.align_to_reference(th, th[1], gauge=("aryl", *HALF))
    assert out["gauge_edge_distance"][0] < 0.02 < out["gauge_edge_distance"][1]
    assert out["gauge_edge_distance"][1] == pytest.approx(np.pi / 2, abs=0.05)


def test_a_window_nothing_can_satisfy_takes_the_closest_miss_and_says_so():
    mol, d, th = _setup([(np.pi / 2, 0.2), (0.0, 0.2)])
    out = d.align_to_reference(th, th[1], gauge=("aryl", -0.1, 0.1))     # images are a half turn apart
    assert out["gauge_satisfied"].tolist() == [False, True]
    assert out["gauge_edge_distance"][0] == pytest.approx(np.pi / 2 - 0.1, abs=0.05)
    assert out["gauge_edge_distance"][1] == pytest.approx(0.1, abs=0.02)


def test_no_gauge_and_a_full_circle_window_are_the_plain_alignment():
    mol, d, th = _setup([(0.6, 0.2), (-1.0, 3.0), (2.5, -2.0)])
    plain = d.align_to_reference(th, th[0])
    assert "gauge_satisfied" not in plain
    full = d.align_to_reference(th, th[0], gauge=("aryl", -np.pi, np.pi))
    assert np.allclose(full["angles"], plain["angles"]) and (full["operations"] == plain["operations"]).all()


@pytest.mark.parametrize("gauge, message", [
    (("nope", -1.0, 1.0), "not a column"),
    (("aryl", 1.0, 1.0), "wider than 0"),
    (("aryl", -1.0, 9.0), "at most one turn"),
    (("aryl",), "gauge must be"),
    (("aryl", -1.0, 0.0, 1.0), "gauge must be"),
    (("aryl", -1.0), "wider than 0"),
    ((99, -1.0, 1.0), "outside"),
])
def test_a_malformed_gauge_is_refused_by_name(gauge, message):
    mol, d, th = _setup([(0.6, 0.2)])
    with pytest.raises(ValueError, match=message):
        d.align_to_reference(th, th[0], gauge=gauge)


@requires_sklearn
def test_the_gauge_reaches_align_all_and_never_the_clustering():
    mol, defs = _para()
    xyz = _ensemble(mol)
    fit = _fit(mol, defs, xyz)
    plain = fit.align_all(per_cluster=True)
    gauged = fit.align_all(per_cluster=True, gauge=("aryl", *HALF))
    aryl = gauged["angles"][:, 1]
    aryl = gauged["angles"][:, 1]
    assert gauged["gauge_satisfied"].all()
    assert ((aryl >= HALF[0] - 1e-9) & (aryl < HALF[1] + 1e-9)).all()
    assert gauged["selected"].shape == plain["selected"].shape
    refit = _fit(mol, defs, xyz)
    assert (refit.labels_ == fit.labels_).all()              # the fit is the same object of study


def test_a_width_centres_the_window_on_the_reference_value():
    """The reference is written in the MIDDLE of the window, so the cut is as far from it as it can be."""
    mol, d, th = _setup([(1.1, 0.2), (1.1 + np.pi, 0.2 + np.pi), (1.1 + 0.3, 0.2)])
    out = d.align_to_reference(th, th[0], gauge=("aryl", np.pi))
    column, lo, hi = out["gauge_window"]
    assert (hi - lo) == pytest.approx(np.pi) and 0.5 * (lo + hi) == pytest.approx(th[0, column])
    # a frame that IS the reference's other labelling is written exactly where the reference is
    assert wrap(out["angles"][1, 1] - out["angles"][0, 1]) == pytest.approx(0.0, abs=0.05)
    assert out["gauge_edge_distance"][0] == pytest.approx(np.pi / 2, abs=1e-6)
    # the same frames under a window at zero are NOT centred on the data
    fixed = d.align_to_reference(th, th[0], gauge=("aryl", *HALF))
    assert fixed["gauge_edge_distance"][0] < out["gauge_edge_distance"][0]


@requires_sklearn
def test_align_all_resolves_one_window_from_the_global_reference():
    mol, defs = _para()
    xyz = _ensemble(mol)
    fit = _fit(mol, defs, xyz)
    out = fit.align_all(per_cluster=True, gauge=("aryl", np.pi))
    ref_value = fit.closed_theta_[fit.density_peak(0), 1]
    column, lo, hi = out["gauge_window"]
    assert 0.5 * (lo + hi) == pytest.approx(ref_value) and hi - lo == pytest.approx(np.pi)
    ok = out["gauge_satisfied"]
    assert ok.mean() > 0.95
    # centred on the peak, no frame is anywhere near the cut for this ensemble
    assert (out["gauge_edge_distance"][ok] > 0.3).all()


def test_a_two_value_gauge_needs_a_reference_to_centre_on():
    mol, d, th = _setup([(0.6, 0.2)])
    with pytest.raises(ValueError):
        d._gauge_column(("aryl", np.pi))                      # no reference to centre on
