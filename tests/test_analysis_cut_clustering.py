"""Cut-route torsional clustering: recursive free-energy cuts on symmetry-invariant coordinates.

`cut_clustering` partitions frames by cutting one-dimensional free-energy profiles, recursively.
Each test names the mistake it exists to catch: a frame left without a label, clusters numbered by
something other than mass, a cut inside an already-cut coordinate that measures its distance across
the excluded part of the circle, pi/3 read as 59.999 degrees, two symmetry images of one
conformation cut apart, a lower barrier cut before a higher one, and a tree that loses frames.

CPU only. The symmetry tests use the paracetamol fixtures of the Dirichlet-alignment tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")

from md_tools.analysis import _t_cut as T                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_dirichlet_alignment_fixtures", Path(__file__).with_name("test_analysis_dirichlet_alignment.py"))
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)
requires_sklearn = _fx.requires_sklearn

N_BINS = 72
EDGES = np.linspace(-np.pi, np.pi, N_BINS + 1)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def _from_counts(counts):
    """Angles at bin centres, `counts[i]` of them in bin i of the 72 five-degree bins."""
    return np.repeat(CENTRES, np.asarray(counts, dtype=int))


def _plain(angles, names=None):
    from md_tools.analysis import t_hdbscan

    angles = np.asarray(angles, float)
    names = names or [f"x{k}" for k in range(angles.shape[1])]
    return t_hdbscan(angles, symmetry=False, names=names)


def _sample(rng, counts, n):
    """`n` angles drawn from the histogram `counts`, uniformly inside each bin."""
    p = np.asarray(counts, float) / np.sum(counts)
    b = rng.choice(N_BINS, size=n, p=p)
    return EDGES[b] + rng.uniform(0.0, 2 * np.pi / N_BINS, n)


# ------------------------------------------------------------------------------------ labelling

def test_every_frame_gets_a_label_and_clusters_are_numbered_by_mass():
    rng = np.random.default_rng(0)
    centres = [(-2.0, -2.0, 4000), (1.0, -2.0, 1000), (-2.0, 1.0, 2500), (1.0, 1.0, 300)]
    x, truth = [], []
    for k, (a, b, n) in enumerate(centres):
        x.append(np.column_stack([rng.normal(a, 0.15, n), rng.normal(b, 0.15, n)]))
        truth += [k] * n
    x, truth = np.angle(np.exp(1j * np.vstack(x))), np.array(truth)
    r = T.cut_clustering(_plain(x), mass_floor=0.01)
    assert r["n_clusters"] == 4
    assert (r["labels"] >= 0).all()
    sizes = np.bincount(r["labels"])
    assert list(sizes) == sorted(sizes, reverse=True)
    rank = np.argsort([-n for _, _, n in centres])               # cluster i is the i-th largest
    for i, k in enumerate(rank):
        assert (r["labels"][truth == k] == i).all()


def test_the_tree_accounts_for_every_frame():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.concatenate([rng.normal(-1.5, 0.2, 3000), rng.normal(1.5, 0.2, 1000)]),
                         rng.normal(0.0, 0.2, 4000)])
    r = T.cut_clustering(_plain(np.angle(np.exp(1j * x))), mass_floor=0.01)
    tree = r["tree"]
    assert tree[0]["parent"] is None and tree[0]["frames"] == 4000
    for node in tree:
        if node["children"]:
            assert sum(tree[c]["frames"] for c in node["children"]) == node["frames"]
            assert node["cut"] is not None and "cluster" not in node
        else:
            assert node["cut"] is None
            assert node["frames"] == int((r["labels"] == node["cluster"]).sum())
    assert sorted(n["cluster"] for n in tree if not n["children"]) == list(range(r["n_clusters"]))


# ------------------------------------------------------------------------------------ the cut rule

def _bounded_counts():
    """A box from bin 10 to bin 50; a well over bins 12-41, an empty gap at bins 42-45, a well at
    46-49. Every bin outside the box is empty (a cell of a cut coordinate has nothing there)."""
    c = np.zeros(N_BINS)
    c[12:42] = 100
    c[46:50] = 100
    return c


def test_a_bounded_cut_is_refused_when_a_piece_is_narrower_than_the_separation():
    """The only gap is 20-40 degrees from the inherited boundary at bin 50. Measured around the
    circle the gap is far from bin 10, but the cell has nothing beyond its box: the piece between
    the cut and the boundary is what must exceed the separation."""
    x = _from_counts(_bounded_counts())
    assert T._bounded_cut(x, x, float("nan"), 10, 10, 50, EDGES, 60.0) is None
    got = T._bounded_cut(x, x, float("nan"), 10, 10, 50, EDGES, 15.0)
    assert got is not None and got["bin"] in range(42, 46) and np.isinf(got["barrier"])


def test_a_bounded_cut_is_searched_only_inside_the_box():
    """Bins 55-71 and 0-5 are empty too, and far from both boundaries around the circle; a search
    over the whole circle would take one of them. Inside the box there is nothing to cut."""
    c = np.zeros(N_BINS)
    c[12:49] = 100
    x = _from_counts(c)
    assert T._bounded_cut(x, x, float("nan"), 10, 10, 50, EDGES, 30.0) is None


def test_a_bounded_cut_needs_the_barrier_and_the_floor():
    c = np.zeros(N_BINS)
    c[12:30] = 1000
    c[30] = 300 * np.exp(-1.5)              # 1.5 kT above the floor of the SMALLER side (31-48)
    c[31:49] = 300
    x = _from_counts(c)
    got = T._bounded_cut(x, x, 1.0, 10, 10, 50, EDGES, 60.0)
    assert got is not None and got["bin"] == 30
    assert got["barrier"] == pytest.approx(1.5, abs=0.02)
    assert T._bounded_cut(x, x, 2.0, 10, 10, 50, EDGES, 60.0) is None
    smaller = int(c[30]) + 18 * 300            # the cut bin's frames sit on the cut: count them
    assert T._bounded_cut(x, x, 1.0, smaller + 1, 10, 50, EDGES, 60.0) is None


def test_the_default_separation_is_pi_over_3_and_exactly_60_degrees_is_refused():
    """The only two empty bins are 60 degrees apart. np.degrees(pi / 3) is 59.999..., which would
    call them more than 60 apart and cut."""
    assert T.DEFAULT_MIN_P2P == pytest.approx(np.pi / 3)
    c = np.full(N_BINS, 20)
    c[0] = c[12] = 0
    x = np.column_stack([_from_counts(c), np.zeros(int(c.sum()))])
    assert T.cut_clustering(_plain(x))["n_clusters"] == 1
    assert T.cut_clustering(_plain(x), min_p2p=np.radians(55))["n_clusters"] == 2


def test_annealing_cuts_the_higher_barrier_first():
    """Coordinate x1 has a 4 kT barrier and x0 a 2 kT one, neither empty. The root must be cut on
    x1, although x0 comes first in coordinate order; each half is then cut on x0."""
    rng = np.random.default_rng(2)

    def two_wells(barrier):
        c = np.ones(N_BINS) * 1000.0
        c[30:36] = c[66:72] = 1000.0 * np.exp(-barrier)
        return c

    n = 100_000
    x = np.column_stack([_sample(rng, two_wells(2.0), n), _sample(rng, two_wells(4.0), n)])
    r = T.cut_clustering(_plain(np.angle(np.exp(1j * x))))
    root = r["tree"][0]["cut"]
    assert root["coordinate"] == "x1" and root["barrier_kT"] == pytest.approx(4.0, abs=0.3)
    second = [r["tree"][c]["cut"] for c in r["tree"][0]["children"]]
    assert all(s is not None and s["coordinate"] == "x0" for s in second)
    assert all(s["barrier_kT"] == pytest.approx(2.0, abs=0.3) for s in second)
    assert r["n_clusters"] == 4


def test_when_two_coordinates_clear_the_same_level_the_higher_barrier_is_cut():
    """A step so large that the first level below the start is kT_cut itself, where both the
    2.5 kT and the 4 kT barrier clear it: the choice is then the barrier ranking, not the order
    in which annealing reaches them."""
    rng = np.random.default_rng(5)

    def two_wells(barrier):
        c = np.ones(N_BINS) * 1000.0
        c[30:36] = c[66:72] = 1000.0 * np.exp(-barrier)
        return c

    n = 100_000
    x = np.column_stack([_sample(rng, two_wells(2.5), n), _sample(rng, two_wells(4.0), n)])
    r = T.cut_clustering(_plain(np.angle(np.exp(1j * x))), kT_step=100.0)
    root = r["tree"][0]["cut"]
    assert root["level_kT"] == pytest.approx(1.0)
    assert root["coordinate"] == "x1"


def test_a_cell_of_a_cut_coordinate_uses_the_bounded_rule():
    """Bins 21 and 47 are empty and the root is cut there. The cell between them has one more
    empty bin, 44, which leaves a 10-degree sliver against the inherited cut at 47. Profiled as a
    full circle, the cell's empty outside (bins 47 round to 21) is far from 44 and the sliver
    would be cut off as a third cluster."""
    c = np.full(N_BINS, 50)
    c[[21, 44, 47]] = 0
    x = np.column_stack([_from_counts(c), np.zeros(int(c.sum()))])
    r = T.cut_clustering(_plain(x), mass_floor=0.01)
    assert r["tree"][0]["cut"]["positions_degrees"] == pytest.approx([-72.5, 57.5])
    assert r["n_clusters"] == 2


# ------------------------------------------------------------------------------------ symmetry

def _para_unfitted(settings, seed=0):
    from md_tools.analysis import t_hdbscan

    mol, defs = _fx._para()
    xyz = _fx._frames(mol, settings, seed=seed)
    stored = np.column_stack([_fx.dihedral(*(xyz[:, i] for i in q))
                              for q in _fx.PARA_TORSIONS.values()])
    return mol, defs, xyz, stored, t_hdbscan(
        stored, molecule=mol, torsion_definitions=defs, coordinates=xyz,
        atom_mapping=list(range(mol.GetNumAtoms())), periodic=False, topology=_fx._topology(mol))


def test_a_relabelled_frame_has_the_same_cut_coordinates():
    """The ring flip renames the ring atoms. The stored torsions change; the cut coordinates must
    not, or one conformation would be cut away from itself."""
    from md_tools.analysis import t_hdbscan

    settings = [(0.6, 0.2), (-1.0, 3.0), (2.5, -2.0)]
    mol, defs, xyz, stored, fit = _para_unfitted(settings)
    flip = list(_fx._flip(fit.descriptor_.enumeration))
    xyz_r = xyz[:, flip, :]
    stored_r = np.column_stack([_fx.dihedral(*(xyz_r[:, i] for i in q))
                                for q in _fx.PARA_TORSIONS.values()])
    fit_r = t_hdbscan(stored_r, molecule=mol, torsion_definitions=defs, coordinates=xyz_r,
                      atom_mapping=list(range(mol.GetNumAtoms())), periodic=False,
                      topology=_fx._topology(mol))
    assert not np.allclose(np.angle(np.exp(1j * (stored - stored_r))), 0.0, atol=1e-3)
    Y, names, valid, groups = T._invariant_coordinates(fit, T.DEFAULT_Z_MIN)
    Y_r, names_r, _, _ = T._invariant_coordinates(fit_r, T.DEFAULT_Z_MIN)
    assert names == names_r and groups
    assert np.allclose(np.angle(np.exp(1j * (Y - Y_r))), 0.0, atol=1e-6)


@requires_sklearn
def test_two_symmetry_images_of_one_conformation_are_one_cluster():
    """Half the frames are (aryl, phenol) and half the ring-flipped naming of the same conformation,
    (aryl + pi, phenol + pi). The plain torsions put an empty gap between them; the cut route on the
    invariant coordinates must not."""
    rng = np.random.default_rng(3)
    n = 300
    base = [(0.6 + rng.normal(0, 0.1), 0.2 + rng.normal(0, 0.1)) for _ in range(2 * n)]
    settings = base[:n] + [(a + np.pi, p + np.pi) for a, p in base[n:]]
    _, _, _, stored, fit = _para_unfitted(settings)
    sym = T.cut_clustering(fit, mass_floor=0.05)
    plain = T.cut_clustering(_plain(stored), mass_floor=0.05)
    assert plain["n_clusters"] >= 2
    assert sym["n_clusters"] == 1


# ------------------------------------------------------------------------------------ surface

def test_refusals_name_what_to_change():
    fit = _plain(np.zeros((50, 2)))
    for kw, word in (({"kT_cut": 0.0}, "kT_cut"), ({"mass_floor": 0.6}, "mass_floor"),
                     ({"kT_step": 0.0}, "kT_step"), ({"min_p2p": np.pi}, "min_p2p"),
                     ({"n_bins": 4}, "n_bins")):
        with pytest.raises(ValueError, match=word):
            T.cut_clustering(fit, **kw)


def test_plot_hist_draws_a_cut_result():
    import matplotlib

    matplotlib.use("Agg")
    from md_tools.analysis import cut_clustering, plot_hist

    rng = np.random.default_rng(4)
    x = np.column_stack([np.concatenate([rng.normal(-1.5, 0.2, 600), rng.normal(1.5, 0.2, 400)]),
                         rng.normal(0.0, 0.2, 1000)])
    r = cut_clustering(_plain(np.angle(np.exp(1j * x))))
    fig = plot_hist(r, n_cluster=10, show_molecule=False)
    assert len(fig.axes) == 2 * (1 + r["n_clusters"])
    assert r["settings"]["kT_cut"] == T.DEFAULT_CUT_KT == 1.0
    assert r["settings"]["mass_floor"] == T.DEFAULT_CUT_MASS_FLOOR == 0.001
    assert r["settings"]["min_p2p"] == pytest.approx(np.pi / 3)


def test_the_result_carries_the_cut_coordinates_per_frame():
    rng = np.random.default_rng(5)
    x = np.angle(np.exp(1j * rng.normal(0.0, 1.0, (500, 2))))
    r = T.cut_clustering(_plain(x, ["a", "b"]))
    assert r["coordinates"] == ["a", "b"]
    assert r["coordinate_values"].shape == (500, 2)
    np.testing.assert_allclose(r["coordinate_values"], x)
    assert r["coordinate_valid"].all()


def test_plot_split_tree_draws_one_panel_per_cell_and_the_cuts_in_it():
    import matplotlib

    matplotlib.use("Agg")
    from md_tools.analysis import cut_clustering, plot_split_tree

    rng = np.random.default_rng(6)
    x = np.column_stack([np.concatenate([rng.normal(-1.5, 0.2, 600), rng.normal(1.5, 0.2, 400)]),
                         rng.normal(0.0, 0.2, 1000)])
    r = cut_clustering(_plain(np.angle(np.exp(1j * x)), ["a", "b"]))
    nodes = len(r["tree"])
    assert nodes >= 3
    fig = plot_split_tree(r, leaf_torsions=["a"])
    assert len(fig.axes) == nodes
    root = r["tree"][0]["cut"]
    drawn = sorted(float(line.get_xdata()[0]) for line in fig.axes[0].lines[1:])
    assert drawn == pytest.approx(sorted(root["positions_degrees"]))
    fig = plot_split_tree(r, pair=("a", "b"))
    assert len(fig.axes) == 3 * nodes                    # the 2D panel and its two marginals
    for kw, word in (({"pair": ("a", "c")}, "pair"), ({"leaf_torsions": ["c"]}, "leaf torsion")):
        with pytest.raises(ValueError, match=word):
            plot_split_tree(r, **kw)
    with pytest.raises(ValueError, match="cut_clustering"):
        plot_split_tree({"labels": r["labels"]})
