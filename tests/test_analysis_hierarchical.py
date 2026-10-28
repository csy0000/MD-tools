"""Hierarchical torsional clustering: two-round HDBSCAN, thermodynamic cut, torsion bits, mass order.

`hierarchical_clustering` is built from small rules, and each rule has a way to be wrong that still
produces plausible clusters. The tests name that mistake: a W2 that does not wrap at pi, a cut that
fires through a low route (one cut on a circle), a cut taken when two torsions are two-well, a drop
that ignores coupling, bit groups numbered in the wrong order, clusters numbered by structure
instead of by mass, a plot that shows fewer or more clusters than asked.

CPU only, and the clustering fixtures are the paracetamol ones of the recursive-split tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit.Chem import rdMolTransforms                                        # noqa: E402,F401

from md_tools.analysis import _t_hierarchy as H                               # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rs = _load("test_analysis_recursive_split")
requires_sklearn = _rs.requires_sklearn
N_BINS = 72


def _from_counts(counts):
    """Angles at bin centres, `counts[i]` of them in bin i of the 72 five-degree bins."""
    centres = -np.pi + (np.arange(N_BINS) + 0.5) * 2 * np.pi / N_BINS
    return np.repeat(centres, np.asarray(counts, dtype=int))


def _three_well_counts(route):
    """Wells at 0, 90 and 180 degrees; the arc through -90 empty; `route` counts join the wells."""
    c = np.zeros(N_BINS)
    c[34:38] = (200, 500, 500, 200)                      # 0
    c[52:56] = (150, 300, 300, 150)                      # 90
    c[[70, 71, 0, 1]] = (200, 400, 400, 200)             # 180
    c[38:52] = route
    c[56:70] = route
    return c


# ------------------------------------------------------------------------------------------ W2

def test_circular_w2_is_zero_for_the_same_sample_and_recovers_a_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 0.1, 400)
    assert H._circular_w2(x, x) == pytest.approx(0.0, abs=1e-9)
    assert H._circular_w2(x, x + np.radians(30)) == pytest.approx(30.0, abs=0.5)


def test_circular_w2_wraps_across_pi():
    """Two samples either side of the cut are 10 degrees apart, not 350: a W2 on the line says 350
    and would call a pair of identical profiles different."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.05, 400)
    a = np.angle(np.exp(1j * (np.pi - np.radians(5) + base)))
    b = np.angle(np.exp(1j * (np.pi + np.radians(5) + base)))
    assert H._circular_w2(a, b) == pytest.approx(10.0, abs=0.7)


# ------------------------------------------------------------------------------------------ cuts

def test_two_blocked_wells_give_two_cuts():
    x = _from_counts(_three_well_counts(route=6))
    got = H._profile_cuts(x, kT_cut=1.6, floor_frames=10, n_bins=N_BINS)
    assert got is not None and got["F_cut2"] >= 1.6 and got["minor_fraction"] > 0.0
    assert got["bins"][0] in range(2, 34)                # cut 1 sits in the empty arc at -90


def test_a_single_well_has_no_cut():
    c = np.zeros(N_BINS)
    c[34:38] = (200, 500, 500, 200)
    assert H._profile_cuts(_from_counts(c), 1.6, 10, N_BINS) is None


def test_a_low_route_through_a_shallow_third_well_means_no_split():
    """One cut on a circle is not a barrier: the arcs stay connected by the route through 90 degrees.
    With that route at about 1.4 kT no second cut exists, so the profile is not two-well."""
    x = _from_counts(_three_well_counts(route=120))
    assert H._profile_cuts(x, kT_cut=1.6, floor_frames=10, n_bins=N_BINS) is None


def test_the_barrier_is_measured_from_the_minor_wells_floor_not_the_global_minimum():
    """Well B sits 1.0 kT above well A, and the route between them is 2.2 kT: the barrier B has to
    climb is 1.2 kT, below the 1.6 threshold, although the route is above 1.6 from A's floor."""
    c = np.zeros(N_BINS)
    c[34:38] = (200, 500, 500, 200)
    c[38:42] = 55                                        # the route, a narrow ridge
    c[42:46] = (80, 184, 184, 80)                        # well B, 1.0 kT above A's floor
    assert H._profile_cuts(_from_counts(c), 1.6, 10, N_BINS) is None
    assert H._profile_cuts(_from_counts(c), 1.0, 10, N_BINS) is not None


def test_the_threshold_is_the_one_asked_for():
    x = _from_counts(_three_well_counts(route=120))        # a 1.4 kT route
    assert H._profile_cuts(x, 1.6, 10, N_BINS) is None
    assert H._profile_cuts(x, 1.0, 10, N_BINS) is not None


def test_the_floor_is_on_the_minor_arc_in_frames():
    """A cut whose smaller arc holds fewer frames than the floor is never returned; no arc can hold
    more than half the frames, so a floor above half returns nothing."""
    x = _from_counts(_three_well_counts(route=6))
    for floor in (10, 800, 1400):
        got = H._profile_cuts(x, 1.6, floor, N_BINS)
        assert got is not None and got["minor_fraction"] * len(x) >= floor
    assert H._profile_cuts(x, 1.6, 0.51 * len(x), N_BINS) is None


def test_an_empty_bin_counts_as_the_highest_free_energy():
    """Empty bins are the best candidate cuts: a barrier nobody crossed, not a hole in the profile."""
    x = _from_counts(_three_well_counts(route=0))
    got = H._profile_cuts(x, 1.6, 10, N_BINS)
    assert got is not None and got["F_cut1"] == pytest.approx(got["F_cut2"])


# ------------------------------------------------------------------------------------------ bits

def test_clusters_with_the_same_profiles_share_every_bit_and_the_larger_group_is_bit_zero():
    rng = np.random.default_rng(2)
    n = [150, 120, 40]
    ang = np.concatenate([rng.normal(0.0, 0.1, n[0]), rng.normal(0.0, 0.1, n[1]),
                          rng.normal(1.5, 0.1, n[2])])[:, None]
    lab = np.repeat([0, 1, 2], n)
    bits = H._bits(ang, lab, 3, threshold=20.0)
    assert bits[0, 0] == bits[1, 0] == 0 and bits[2, 0] == 1


# ------------------------------------------------------------------------------------------ coupling

def test_the_ring_flip_couples_the_aryl_and_the_phenol_torsion():
    mol, defs = _rs._fx._para()
    from md_tools.analysis import SymmetricTorsionDescriptor

    d = SymmetricTorsionDescriptor(mol, defs)
    assert H._coupling_groups(d, 3) == [{1, 2}]


# ------------------------------------------------------------------------------------------ pipeline

@pytest.fixture(scope="module")
def basins():
    mol, defs, xyz, truth = _rs._two_basins(400, 200)
    fit = _rs._fx._fit(mol, defs, xyz, mass_floor=0.05)
    return mol, defs, xyz, truth, fit


@requires_sklearn
def test_two_basins_become_two_clusters_numbered_by_mass(basins):
    mol, defs, xyz, truth, fit = basins
    r = H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05)
    assert r["n_clusters"] == 2
    sizes = [int((r["labels"] == c).sum()) for c in range(2)]
    assert sizes[0] > sizes[1], "cluster 1 must be the larger one"
    assert max(np.mean((r["labels"] == 0) == (truth == 0)), np.mean((r["labels"] == 0) == (truth == 1))) > 0.99
    assert r["cluster_mass"][0] > r["cluster_mass"][1]
    assert r["rmsd"] is not None and np.asarray(r["rmsd"]["mean"]).shape == (2, 2)
    assert r["n_unassigned"] == int((r["labels"] < 0).sum())


@requires_sklearn
def test_the_round_one_record_is_kept_with_its_null_values(basins):
    mol, defs, xyz, truth, fit = basins
    r = H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05, rmsd=False)
    rec = r["rounds"][0]
    omega = rec["w2"][0][0]
    assert rec["round1_clusters"] == 2 and omega["w2"] > 100 and omega["null95"] < 60
    assert "omega" not in rec["dropped"], "the torsion the basins differ on is never dropped"


@requires_sklearn
def test_torsions_the_basins_share_are_dropped_in_round_two(basins):
    mol, defs, xyz, truth, fit = basins
    r = H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05, seed=3, rmsd=False)
    rec = r["rounds"][0]
    assert rec["dropped"] == ["aryl", "phenol"] and rec["round2_clusters"] == 2


@requires_sklearn
def test_a_torsion_is_not_dropped_while_its_coupled_partner_is_kept(basins):
    """With this seed the aryl torsion passes the null test (not different) and the phenol does not;
    the ring flip moves them together, so dropping aryl alone would let the flip map one state onto
    the other."""
    mol, defs, xyz, truth, fit = basins
    r = H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05, seed=0, rmsd=False)
    rec = r["rounds"][0]
    assert "aryl" not in rec["dropped"] and rec["kept_by_coupling"] == ["aryl"]


@requires_sklearn
def test_without_the_null_test_nothing_is_dropped(basins):
    mol, defs, xyz, truth, fit = basins
    r = H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05,
                                  torsion_null_test=0, rmsd=False)
    assert r["rounds"][0]["dropped"] == []


def _hierarchy(fit, xyz, **kw):
    args = dict(torsion_null_test=40, kT_cut=1.6, mass_floor=0.05, max_depth=0, min_node_frames=40,
                n_bins=N_BINS, seed=0, verbose=False)
    args.update(kw)
    return H._Hierarchy(fit, xyz, aligned=fit.align_all(per_cluster=True)["angles"], **args)


@requires_sklearn
def test_a_node_hdbscan_cannot_split_is_cut_on_the_one_torsion_that_is_two_well(basins):
    mol, defs, xyz, truth, fit = basins
    h = _hierarchy(fit, xyz)
    h.settle(np.arange(len(xyz)), "R", 0, len(xyz), "")
    assert [c["torsion"] for c in h.cuts] == ["omega"]
    sizes = sorted(len(l["rows"]) for l in h.leaves)
    assert sizes == [200, 400] and sum(len(a) for a in h.ambiguous) == 0


@requires_sklearn
def test_no_direct_cut_when_two_torsions_are_two_well():
    """Omega and the aryl torsion both have two states, perfectly correlated. A cut on one marginal is
    exact only when no other dimension can connect the arcs; with two candidates the node is left
    whole and said to be."""
    rng = np.random.default_rng(5)
    mol, defs = _rs._fx._para()
    settings = []
    for om, ar, n in ((np.pi, 0.6, 250), (0.0, 2.4, 250)):
        settings += [(om + rng.normal(0, 0.1), ar + rng.normal(0, 0.1), 0.2 + rng.normal(0, 0.1))
                     for _ in range(n)]
    xyz = _rs._positions(mol, settings, seed=5)
    fit = _rs._fx._fit(mol, defs, xyz, mass_floor=0.05)
    h = _hierarchy(fit, xyz)
    h.settle(np.arange(len(xyz)), "R", 0, len(xyz), "")
    assert h.cuts == []
    assert h.leaves[0]["status"] == "two-well left" and len(h.leaves[0]["torsions"]) >= 2


@requires_sklearn
def test_frames_beside_a_non_empty_cut_are_ambiguous_not_assigned(basins):
    mol, defs, xyz, truth, fit = basins
    h = _hierarchy(fit, xyz)
    ang = h.AL[:, 0].copy()
    cut_bin = 20
    rng = np.random.default_rng(0)
    centre = -np.pi + (cut_bin + 0.5) * 2 * np.pi / N_BINS
    ang[rng.choice(len(ang), 6, replace=False)] = centre          # frames sitting ON a cut bin
    h.AL = h.AL.copy()
    h.AL[:, 0] = ang
    found = {"bins": (cut_bin, 60), "F_cut1": 5.0, "F_cut2": 5.0, "minor_fraction": 0.3,
             "cut1_degrees": 0.0, "cut2_degrees": 0.0}
    h.cut(np.arange(len(xyz)), h.AL, 0, found, "R", 0, "")
    assert sum(len(a) for a in h.ambiguous) >= 6


# ------------------------------------------------------------------------------------------ guards

@requires_sklearn
def test_inputs_that_cannot_be_clustered_are_refused_by_name(basins):
    mol, defs, xyz, truth, fit = basins
    with pytest.raises(ValueError, match="kT_cut"):
        H.hierarchical_clustering(fit, xyz, kT_cut=0.0)
    with pytest.raises(ValueError, match="mass_floor"):
        H.hierarchical_clustering(fit, xyz, mass_floor=0.7)
    unfitted = _rs._fx._fit(mol, defs, xyz, mass_floor=0.05)
    unfitted.fitted_ = False
    with pytest.raises(RuntimeError, match="fit"):
        H.hierarchical_clustering(unfitted, xyz)
    weighted = _rs._fx._fit(mol, defs, xyz, mass_floor=0.05)
    weighted.weights_uniform_ = False
    with pytest.raises(NotImplementedError, match="weights"):
        H.hierarchical_clustering(weighted, xyz)


# ------------------------------------------------------------------------------------------ plots

@pytest.fixture(scope="module")
def result(basins):
    mol, defs, xyz, truth, fit = basins
    return H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05, rmsd=False)


def test_show_compound_draws_the_molecule_with_each_torsion_labelled(result):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    from md_tools.analysis import show_compound

    ax = show_compound(result["molecule"], result["quads"], torsion_enumerate=True, size=400)
    assert len(ax.images) == 1 and ax.images[0].get_array().shape[0] == 400
    with pytest.raises(ValueError, match="names"):
        show_compound(result["molecule"], result["quads"], names=["only one"])
    plt.close("all")


def test_plot_hist_shows_every_cluster_up_to_n_cluster_and_the_selected_torsions(result):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    from md_tools.analysis import plot_hist

    fig = plot_hist(result, torsion_show="all", n_cluster=10, show_molecule=False)
    assert len(fig.axes) == (1 + 2) * 3, "2 clusters exist, so 10 asked shows 2, plus the all-frames row"
    fig = plot_hist(result, torsion_show=np.array([0, 2]), n_cluster=1, show_molecule=False)
    assert len(fig.axes) == (1 + 1) * 2
    fig = plot_hist(result, torsion_show=["omega"], n_cluster=2, kind="density", show_molecule=False)
    assert len(fig.axes) == 3
    with pytest.raises(ValueError, match="kind"):
        plot_hist(result, kind="log")
    with pytest.raises(ValueError, match="torsion"):
        plot_hist(result, torsion_show=["nope"])
    with pytest.raises(ValueError, match="n_cluster"):
        plot_hist(result, n_cluster=0)
    plt.close("all")


def test_plot_hist_draws_the_barrier_line_at_kT_cut(result):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    from md_tools.analysis import plot_hist

    kT = 0.6
    fig = plot_hist(result, torsion_show=[0], n_cluster=1, kT=kT, show_molecule=False)
    ys = [l.get_ydata()[0] for l in fig.axes[0].lines if len(set(l.get_ydata())) == 1]
    assert any(abs(y - result["settings"]["kT_cut"] * kT) < 1e-9 for y in ys)
    plt.close("all")


def test_marked_clusters_get_a_tinted_row_and_a_star(result):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    from md_tools.analysis import plot_hist

    fig = plot_hist(result, torsion_show=[0], n_cluster=2, show_molecule=False, mark=[1])
    rows = fig.axes                                        # all-frames, cluster 1, cluster 2
    assert [a.get_facecolor()[:3] != (1.0, 1.0, 1.0) for a in rows] == [False, False, True]
    assert "★" in rows[2].get_ylabel() and "★" not in rows[1].get_ylabel()
    plt.close("all")


def test_nan_accepts_a_cut_only_through_empty_bins():
    """`kT_cut=nan`: a route with any count, however thin, is not a barrier; two empty arcs are."""
    thin = _from_counts(_three_well_counts(route=6))           # one empty arc only
    assert H._profile_cuts(thin, float("nan"), 10, N_BINS) is None
    empty = _from_counts(_three_well_counts(route=0))          # wells separated by empty arcs
    got = H._profile_cuts(empty, float("nan"), 10, N_BINS)
    assert got is not None and got["F_cut1"] == float("inf")
    cnt = np.histogram(empty, np.linspace(-np.pi, np.pi, N_BINS + 1))[0]
    assert all(cnt[b] == 0 for b in got["bins"]), "both cuts must be bins with no count"


def test_nan_is_accepted_by_the_pipeline_and_nothing_is_ambiguous(basins):
    mol, defs, xyz, truth, fit = basins
    r = H.hierarchical_clustering(fit, xyz, units="angstrom", mass_floor=0.05, kT_cut=float("nan"),
                                  rmsd=False)
    assert r["n_clusters"] == 2 and r["n_ambiguous"] == 0 and np.isnan(r["settings"]["kT_cut"])
    with pytest.raises(ValueError, match="kT_cut"):
        H.hierarchical_clustering(fit, xyz, kT_cut=-1.0)
