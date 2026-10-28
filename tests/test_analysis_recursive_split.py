"""Recursive splitting of a torsional clustering by the RMSD of the atoms its torsions move.

`THDBSCAN.refine_by_rmsd` refits a cluster whose internal symmetry-corrected RMSD exceeds a cutoff,
alone and with `allow_single_cluster=False`, and keeps the split only if it yields two clusters.
Paracetamol is small, so the cutoffs here are in tenths of an angstrom; what is tested is the
mechanism, not the 2 A default.

Each test names the mistake it exists to catch: a wide cluster left whole, a tight one split, a
frame lost or relabelled by a split that did not happen, a stop rule that does not stop, a mean that
hides a minority, and an atom set a symmetry operation carries outside itself.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402
from rdkit.Chem import rdMolTransforms                                        # noqa: E402

# The fixtures of the Dirichlet-alignment tests, loaded BY PATH (the tests directory is not a package).
_spec = importlib.util.spec_from_file_location(
    "_dirichlet_alignment_fixtures", Path(__file__).with_name("test_analysis_dirichlet_alignment.py"))
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)
requires_sklearn = _fx.requires_sklearn


def _positions(mol, settings, noise=0.01, seed=0):
    """Frames of paracetamol with (omega, aryl, phenol) set; omega 0 vs pi is cis vs trans."""
    rng = np.random.default_rng(seed)
    out = []
    for om, ar, ph in settings:
        conf = Chem.Conformer(mol.GetConformer())
        rdMolTransforms.SetDihedralRad(conf, 0, 1, 3, 4, float(om))
        rdMolTransforms.SetDihedralRad(conf, 1, 3, 4, 5, float(ar))
        rdMolTransforms.SetDihedralRad(conf, 6, 7, 8, 17, float(ph))
        out.append(conf.GetPositions())
    xyz = np.asarray(out)
    return xyz + rng.normal(0.0, noise, xyz.shape)


def _two_basins(n_trans=300, n_cis=300, seed=0):
    rng = np.random.default_rng(seed)
    mol, defs = _fx._para()
    truth, settings = [], []
    for k, (om, n) in enumerate(((np.pi, n_trans), (0.0, n_cis))):
        truth += [k] * n
        settings += [(om + rng.normal(0, 0.12), 0.6 + rng.normal(0, 0.12), 0.2 + rng.normal(0, 0.12))
                     for _ in range(n)]
    return mol, defs, _positions(mol, settings, seed=seed), np.array(truth)


def _merged(mol, defs, xyz, **kw):
    """A fit whose first-level partition lumps both basins in one cluster, as HDBSCAN's condensed
    tree can when the parent is more stable than its children."""
    fit = _fx._fit(mol, defs, xyz, mass_floor=0.05, **kw)
    fit.labels_ = np.where(fit.labels_ >= 0, 0, fit.labels_)
    fit.n_clusters_ = 1
    return fit


def _purity(labels, truth):
    return max(np.mean((labels == 0) == (truth == 0)), np.mean((labels == 0) == (truth == 1)))


@requires_sklearn
def test_a_wide_cluster_is_split_into_the_two_basins_it_lumped():
    mol, defs, xyz, truth = _two_basins()
    fit = _merged(mol, defs, xyz)
    r = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3)
    assert r["n_clusters"] == 2 and r["n_splits"] == 1
    assert _purity(r["labels"], truth) > 0.99
    root = r["tree"][0]
    assert root["action"] == "split" and root["rmsd_mean"] > 0.3 and len(root["children"]) == 2
    assert all(r["tree"][i]["action"] == "within-cutoff" and r["tree"][i]["rmsd_mean"] < 0.3
               for i in root["children"])
    assert sum(r["cluster_mass"].values()) == pytest.approx(1.0, abs=0.01)
    assert r["settings"]["allow_single_cluster"] is False


@requires_sklearn
def test_a_cluster_inside_the_cutoff_is_left_exactly_as_the_fit_made_it():
    mol, defs, xyz, _ = _two_basins()
    fit = _merged(mol, defs, xyz)
    before = fit.labels_.copy()
    r = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=2.0)           # the default
    assert r["n_clusters"] == 1 and r["n_splits"] == 0
    assert r["tree"][0]["action"] == "within-cutoff"
    assert np.array_equal(r["labels"], before)
    assert np.array_equal(fit.labels_, before)                          # the fit itself is untouched


@requires_sklearn
def test_the_default_cutoff_is_two_angstrom():
    import inspect
    from md_tools.analysis._t_recursive import DEFAULT_CUTOFF_ANGSTROM, refine_by_rmsd
    assert DEFAULT_CUTOFF_ANGSTROM == 2.0
    assert inspect.signature(refine_by_rmsd).parameters["cutoff"].default == 2.0


@requires_sklearn
def test_noise_and_ambiguous_codes_survive_and_no_frame_is_lost():
    mol, defs, xyz, _ = _two_basins()
    fit = _merged(mol, defs, xyz)
    fit.labels_[:20] = -2
    fit.labels_[20:30] = -1
    r = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3)
    assert r["labels"].shape == fit.labels_.shape
    assert (r["labels"][:20] == -2).all() and (r["labels"][20:30] == -1).all()
    assert set(np.unique(r["labels"])) <= {-2, -1, 0, 1}
    assert r["labels"][30:].min() >= -2


@requires_sklearn
def test_the_recursion_stops_at_the_depth_limit_and_at_the_minimum_size():
    mol, defs, xyz, _ = _two_basins()
    fit = _merged(mol, defs, xyz)
    r = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.03, max_depth=1)     # children are "wide" too
    assert r["tree"][0]["action"] == "split"
    assert {r["tree"][i]["action"] for i in r["tree"][0]["children"]} == {"depth-limit"}
    assert r["n_clusters"] == 2
    r = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3, min_frames=10_000)
    assert [t["action"] for t in r["tree"]] == ["too-small"] and r["n_clusters"] == 1


@requires_sklearn
def test_a_mean_hides_a_minority_and_a_quantile_does_not():
    mol, defs, xyz, truth = _two_basins(n_trans=540, n_cis=60)
    fit = _merged(mol, defs, xyz)
    by_mean = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3)
    assert by_mean["n_clusters"] == 1 and by_mean["tree"][0]["rmsd_mean"] < 0.3
    by_tail = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3, statistic=0.9)
    assert by_tail["n_clusters"] == 2 and _purity(by_tail["labels"], truth) > 0.98


@requires_sklearn
def test_the_measured_atoms_are_the_heavy_torsion_atoms_closed_under_the_symmetry():
    from md_tools.analysis import operations_on_atoms, torsion_atoms
    mol, defs, xyz, _ = _two_basins(n_trans=60, n_cis=60)
    fit = _fx._fit(mol, defs, xyz, mass_floor=0.05)
    atoms = torsion_atoms(fit)
    assert all(mol.GetAtomWithIdx(a).GetAtomicNum() > 1 for a in atoms)
    quad_heavy = {i for q in defs.quads for i in q if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
    assert quad_heavy < set(atoms)                        # the ring flip adds the partner carbons
    operations_on_atoms(fit.descriptor_.enumeration, atoms)          # a union of orbits: no refusal
    with pytest.raises(ValueError, match="outside the set"):
        operations_on_atoms(fit.descriptor_.enumeration, sorted(quad_heavy))


@requires_sklearn
def test_refusals_name_what_to_change():
    mol, defs, xyz, _ = _two_basins(n_trans=60, n_cis=60)
    fit = _fx._fit(mol, defs, xyz, mass_floor=0.05)
    with pytest.raises(ValueError, match="units is required"):
        fit.refine_by_rmsd(xyz)
    with pytest.raises(ValueError, match="statistic"):
        fit.refine_by_rmsd(xyz, units="angstrom", statistic="max")
    with pytest.raises(ValueError, match="cutoff"):
        fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.0)
    with pytest.raises(ValueError, match="fitted frames"):
        fit.refine_by_rmsd(xyz[:-1], units="angstrom")
    from md_tools.analysis import t_hdbscan
    legacy = t_hdbscan(np.zeros((50, 2)) + np.random.default_rng(0).normal(0, 0.1, (50, 2)),
                       symmetry=False).fit()
    with pytest.raises(NotImplementedError, match="symmetry-first"):
        legacy.refine_by_rmsd(xyz, units="angstrom")
    unfitted = t_hdbscan(fit.theta_, molecule=mol, torsion_definitions=defs, coordinates=xyz,
                         atom_mapping=list(range(mol.GetNumAtoms())), periodic=False,
                         topology=_fx._topology(mol))
    with pytest.raises(RuntimeError, match="fit"):
        unfitted.refine_by_rmsd(xyz, units="angstrom")


@requires_sklearn
def test_the_refit_never_allows_a_single_cluster_whatever_the_first_fit_did(monkeypatch):
    """The first fit may have allowed one cluster; the refit of a wide cluster must not, because
    the question it answers is "is there more than one", and allowing the root answers no."""
    from md_tools.analysis import THDBSCAN

    mol, defs, xyz, _ = _two_basins()
    fit = _merged(mol, defs, xyz, allow_single_cluster=True)
    assert fit.allow_single_cluster is True
    seen = []
    original = THDBSCAN.fit

    def spy(self):
        seen.append((self.n_frames_, self.allow_single_cluster))
        return original(self)

    monkeypatch.setattr(THDBSCAN, "fit", spy)
    fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3)
    assert seen and all(flag is False for _, flag in seen)      # every call here is a refit
    assert fit.allow_single_cluster is True


@requires_sklearn
def test_frames_the_refit_drops_leave_the_cluster_instead_of_keeping_its_label():
    """Scattered frames inside the lumped cluster are noise to the refit. They must not keep the
    parent's label: every labelled frame is in a leaf, and the leaves count them."""
    mol, defs, xyz, _ = _two_basins()
    rng = np.random.default_rng(5)
    scatter = _positions(mol, [tuple(rng.uniform(-np.pi, np.pi, 3)) for _ in range(15)], seed=9)
    xyz = np.concatenate([xyz, scatter])
    fit = _merged(mol, defs, xyz)
    fit.labels_[-15:] = 0                                              # lumped in with the rest
    r = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3)
    leaf_frames = sum(t["n_frames"] for t in r["tree"] if "cluster" in t)
    assert int((r["labels"] >= 0).sum()) == leaf_frames
    assert int((r["labels"][-15:] >= 0).sum()) < 15


@requires_sklearn
def test_a_basin_wider_than_the_cutoff_is_split_and_no_child_is_empty():
    """The cutoff is the only guard against splitting ONE broad basin: set below its natural width
    it is cut into pieces. What must hold even then is that no leaf is empty and every frame is
    accounted for."""
    mol, defs = _fx._para()
    rng = np.random.default_rng(1)
    xyz = _positions(mol, [(np.pi + rng.normal(0, .25), 0.6 + rng.normal(0, .25),
                            0.2 + rng.normal(0, .25)) for _ in range(1500)])
    fit = _fx._fit(mol, defs, xyz, mass_floor=0.05, allow_single_cluster=True)
    assert fit.n_clusters_ == 1
    wide = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.05)
    assert wide["n_clusters"] >= 2 and all(t["n_frames"] > 0 for t in wide["tree"])
    assert sum(t["n_frames"] for t in wide["tree"] if "cluster" in t) == int((wide["labels"] >= 0).sum())
    tight = fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.2)
    assert tight["n_clusters"] == 1 and tight["n_splits"] == 0


@requires_sklearn
def test_the_default_atoms_are_every_heavy_atom_and_quadruplets_is_the_smaller_set():
    """The default measures what the torsions MOVE, which is every heavy atom; the four atoms that
    define a torsion are a part of it. A default of the defining atoms understates a cluster's width."""
    from md_tools.analysis import moved_atoms, torsion_atoms

    mol, defs, xyz, truth = _two_basins(60, 60)
    fit = _merged(mol, defs, xyz)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    assert moved_atoms(fit) == heavy
    assert set(torsion_atoms(fit)) < set(heavy)
    assert fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3)["atoms"] == heavy
    assert fit.refine_by_rmsd(xyz, units="angstrom", cutoff=0.3, atoms="quadruplets")["atoms"] \
        == torsion_atoms(fit)
