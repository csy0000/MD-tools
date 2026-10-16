"""Writing every frame in the labelling nearest ONE reference (the Dirichlet-domain rule).

`SymmetricTorsionDescriptor.align_to_reference`, `THDBSCAN.density_peak` and
`THDBSCAN.align_all` choose, for each frame, the symmetry operation that puts its whole row of
angles nearest a reference, so a histogram of one torsion is not the sum of one conformer written
twice. Every test builds its own molecule; nothing reads a trajectory or a project path.

Each test names the mistake it exists to catch: an operation applied to one column and not the
others, a frame lost or duplicated, a margin that cannot tell a boundary frame from a safe one, and
a one-reference alignment that cuts through a cluster the reference is not part of.
"""
from __future__ import annotations

import importlib.util
import types

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="symmetry needs the `analysis` extra")
from rdkit import Chem                                                        # noqa: E402
from rdkit.Chem import AllChem, rdMolTransforms                               # noqa: E402

from md_tools.analysis._symmetry_metric import SymmetricTorsionDescriptor, enumerate_symmetry  # noqa: E402
from md_tools.analysis._t_symmetry import TorsionDefinitions, dihedral, wrap  # noqa: E402

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="sklearn (HDBSCAN) is an analysis-only dependency")

PARA_TORSIONS = {"omega": (0, 1, 3, 4), "aryl": (1, 3, 4, 5), "phenol": (6, 7, 8, 17)}


def _para():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    defs = TorsionDefinitions(quads=tuple(PARA_TORSIONS.values()), names=tuple(PARA_TORSIONS),
                              indexing="zero-based", units="radian")
    return mol, defs


def _topology(mol):
    atoms = [types.SimpleNamespace(index=a.GetIdx(),
                                   element=types.SimpleNamespace(symbol=a.GetSymbol()))
             for a in mol.GetAtoms()]
    bonds = [(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    return types.SimpleNamespace(atoms=atoms, bonds=bonds, n_atoms=len(atoms))


def _frames(mol, settings, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for aryl, phenol in settings:
        conf = Chem.Conformer(mol.GetConformer())
        rdMolTransforms.SetDihedralRad(conf, 1, 3, 4, 5, float(aryl))
        rdMolTransforms.SetDihedralRad(conf, 6, 7, 8, 17, float(phenol))
        out.append(conf.GetPositions())
    xyz = np.asarray(out)
    return xyz + rng.normal(0.0, noise, xyz.shape)


def _closed(desc, xyz, mol):
    return desc.closed_angles(None, coordinates=xyz, atom_mapping=list(range(mol.GetNumAtoms())),
                              periodic=False, topology=_topology(mol))


def _flip(enumeration):
    for p in enumeration.permutations:
        if p[5] == 10 and p[6] == 9 and p[11:14] == (11, 12, 13):
            return p
    raise AssertionError("no ring flip found")


def _ensemble(mol, per=150, seed=0):
    """Basins 0 and 1 are one conformer in two labellings (A); 2 and 3 are another (B). B differs
    from A by half a turn in the phenol, which the ring flip turns into half a turn in the aryl:
    B is exactly as far from A's peak under one labelling as under the other."""
    rng = np.random.default_rng(seed)
    settings = []
    for a, p in [(0.6, 0.2), (0.6 + np.pi, 0.2 + np.pi), (0.6, 0.2 + np.pi), (0.6 + np.pi, 0.2)]:
        settings.append(np.column_stack([a + rng.normal(0, 0.12, per), p + rng.normal(0, 0.12, per)]))
    return _frames(mol, np.vstack(settings), noise=0.01, seed=seed)


def _fit(mol, defs, xyz, **kw):
    from md_tools.analysis import t_hdbscan

    stored = np.column_stack([dihedral(*(xyz[:, i] for i in q)) for q in PARA_TORSIONS.values()])
    return t_hdbscan(stored, molecule=mol, torsion_definitions=defs, coordinates=xyz,
                     atom_mapping=list(range(mol.GetNumAtoms())), periodic=False,
                     topology=_topology(mol), **kw).fit()


def test_a_frame_and_its_relabelled_copy_are_written_identically():
    """The point of the rule. Frame y is frame x under another naming, so the two have different
    stored angles and different operations, and the SAME aligned row."""
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _frames(mol, [(0.6, 0.2), (-1.0, 3.0), (2.5, -2.0)])
    th = _closed(d, xyz, mol)
    th_rel = _closed(d, xyz[:, list(_flip(d.enumeration)), :], mol)
    reference = th[0]
    a = d.align_to_reference(th, reference)
    b = d.align_to_reference(th_rel, reference)
    assert not np.allclose(wrap(th - th_rel), 0.0, atol=1e-3)          # stored angles differ
    assert np.allclose(wrap(a["angles"] - b["angles"]), 0.0, atol=1e-9)
    assert (a["operations"] != b["operations"]).any()


def test_the_operation_acts_on_the_whole_row_and_no_frame_is_lost():
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    th = _closed(d, _frames(mol, [(0.6, 0.2), (-1.0, 3.0), (2.5, -2.0), (0.1, 0.1)]), mol)
    out = d.align_to_reference(th, th[2])
    assert out["angles"].shape == th.shape and out["operations"].shape == (th.shape[0],)
    for row, aligned, op, dist in zip(th, out["angles"], out["operations"], out["distance"]):
        assert np.allclose(wrap(aligned - d.transform(row[None], int(op))[0]), 0.0, atol=1e-12)
        assert dist == pytest.approx(np.linalg.norm(d.embed(aligned[None])[0] - d.embed(th[2:3])[0]))
    # the minimum is never worse than leaving the labelling alone
    assert (out["distance"] <= out["distance_identity"] + 1e-12).all()


def test_the_margin_is_zero_on_the_wall_and_large_inside():
    """aryl and phenol each a quarter turn from the reference: the identity labelling and the ring
    flip are equally far. A frame on top of the reference has a wide margin."""
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    ref = np.array([[3.0, 0.0, 0.0, np.pi, np.pi]])          # omega, aryl, phenol, aryl@10, phenol@9
    wall = np.array([[3.0, np.pi / 2, np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2]])
    inside = ref.copy()
    out = d.align_to_reference(np.vstack([wall, inside]), ref[0])
    assert out["margin"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["margin"][1] > 1.0
    assert out["operations"][1] == 0 and out["operations"][0] == 0     # a tie goes to the earlier one


@requires_sklearn
def test_the_density_peak_is_the_densest_member_of_its_cluster_and_needs_the_symmetry_route():
    from md_tools.analysis import t_hdbscan

    mol, defs = _para()
    xyz = _ensemble(mol)
    fit = _fit(mol, defs, xyz, mass_floor=0.05)
    assert fit.n_clusters_ == 2
    for c in range(2):
        peak = fit.density_peak(c)
        members = np.flatnonzero(fit.labels_ == c)
        assert fit.labels_[peak] == c
        assert fit.core_distance_[peak] == fit.core_distance_[members].min()
    stored = np.column_stack([dihedral(*(xyz[:, i] for i in q)) for q in PARA_TORSIONS.values()])
    legacy = t_hdbscan(stored, symmetry=False, mass_floor=0.05).fit()
    with pytest.raises(NotImplementedError):
        legacy.density_peak(0)


@requires_sklearn
def test_one_reference_cuts_a_cluster_that_is_equidistant_and_two_levels_does_not():
    """B sits exactly between A's labelling and its flip, so against A's peak the choice of
    labelling is arbitrary for every B frame. Aligning B to ITS OWN peak removes that."""
    mol, defs = _para()
    fit = _fit(mol, defs, _ensemble(mol), mass_floor=0.05)
    one = fit.align_all()
    two = fit.align_all(per_cluster=True)
    b = fit.labels_ == 1
    assert np.median(one["margin"][b]) < 0.35
    assert np.median(two["margin"][b]) > 1.0
    # the reference cluster is untouched by the second level
    a = fit.labels_ == 0
    assert np.array_equal(one["operations"][a], two["operations"][a])
    # B's own peak is a member of B, and is the only frame placed against the global reference
    assert two["cluster_references"][1] == fit.density_peak(1)
    assert fit.labels_[two["cluster_references"][1]] == 1
    # one frame per frame: nothing dropped, every row is the frame's own angles under its operation
    assert two["angles"].shape == fit.closed_theta_.shape
    for i in (0, 10, 200, 450):
        assert np.allclose(wrap(two["angles"][i] - fit.descriptor_.transform(
            fit.closed_theta_[i:i + 1], int(two["operations"][i]))[0]), 0.0, atol=1e-12)
    # a cluster is more compact in its labelling than the one-reference cut left it
    aryl = fit.descriptor_.names.index("aryl")
    spread = lambda x: 1.0 - np.hypot(np.cos(x).mean(), np.sin(x).mean())
    assert spread(two["angles"][b, aryl]) < spread(one["angles"][b, aryl])


def test_a_crystal_with_two_assignments_is_written_in_the_labelling_nearest_the_reference():
    """A crystal ligand matched onto the molecule has one assignment per graph match. Written through
    the ring flip, the same structure is a different row. The nearest labelling to the reference is
    the one that is not the flip of it, whichever order the candidates arrive in, and a column that
    cannot be measured (through a hydrogen) takes no part."""
    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    reference = d.closed_angles(None, coordinates=_frames(mol, [(0.6, 0.2)], noise=0.0), 
                                atom_mapping=list(range(mol.GetNumAtoms())), periodic=False,
                                topology=_topology(mol))[0]
    crystal = reference.copy()
    crystal[[1, 3]] += 0.1                        # aryl and its image, a little off the reference
    flipped = crystal[list(d.operations[1].sigma)]
    for rows in (np.vstack([crystal, flipped]), np.vstack([flipped, crystal])):
        rows = rows.copy()
        rows[:, [2, 4]] = np.nan                  # the phenol columns go through H17 / H18
        out = d.nearest_labelling(rows, reference)
        assert np.allclose(wrap(out["angles"][[0, 1, 3]] - crystal[[0, 1, 3]]), 0.0, atol=1e-12)
        assert out["columns_used"] == [0, 1, 3] and out["margin"] > 0.5
    with pytest.raises(ValueError, match="reference row has an unmeasurable"):
        d.nearest_labelling(crystal[None], np.where(np.arange(5) == 2, np.nan, reference))


def test_a_crystal_ligand_has_one_assignment_per_graph_match_and_the_rows_are_the_flip_of_each_other():
    """A crystal ligand without hydrogens, atoms in another order. Paracetamol's heavy-atom graph has
    ONE nontrivial match (the ring flip), so there are two assignments, their rows are the two
    labellings of one structure, and the torsion through the phenol H is NaN."""
    from md_tools.analysis._symmetry_metric import reference_assignments

    mol, defs = _para()
    d = SymmetricTorsionDescriptor(mol, defs)
    xyz = _frames(mol, [(0.6, 0.2)], noise=0.0)
    conf = mol.GetConformer()
    for i, p in enumerate(xyz[0]):
        conf.SetAtomPosition(i, p.tolist())
    heavy = Chem.RemoveHs(mol)
    order = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]                        # reversed: another atom order
    crystal = Chem.RenumberAtoms(heavy, order)
    rows = reference_assignments(crystal, mol, d.quads)
    assert rows.shape == (2, d.n_columns)
    assert np.isnan(rows[:, [2, 4]]).all() and np.isfinite(rows[:, [0, 1, 3]]).all()   # phenol columns
    flip = rows[0][list(d.operations[1].sigma)]
    assert any(np.allclose(wrap(r[[0, 1, 3]] - flip[[0, 1, 3]]), 0.0, atol=1e-6) for r in rows)
    assert not np.allclose(wrap(rows[0][[1, 3]] - rows[1][[1, 3]]), 0.0, atol=0.1)
    with pytest.raises(ValueError, match="heavy atoms"):
        reference_assignments(Chem.RenumberAtoms(heavy, order), Chem.AddHs(Chem.MolFromSmiles("CCO")), d.quads)
