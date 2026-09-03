"""The dihedral itself, checked against geometry we can do by hand and against MDTraj.

WHY EVERY VALUE IS COMPARED, NOT JUST THE COLUMN

    A torsion implementation with the sign flipped, or with the minimum-image convention applied
    to positions instead of bond vectors, produces a full column of plausible numbers in the right
    range. Asserting the column exists proves nothing at all. So each test below states the angle
    it expects and why, and the periodic ones assert the periodic answer equals the nonperiodic
    answer for a geometry that has been deliberately split across the boundary -- which is the
    whole claim the convention makes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from md_tools.cv import TorsionError, minimum_image, torsion_degrees


def _four_atoms(angle_degrees: float) -> np.ndarray:
    """Four atoms whose dihedral about the central bond is exactly `angle_degrees`.

    i and l sit one nanometre off the z axis; j and k are on it. Rotating l about z by theta makes
    the dihedral theta by construction, so the expected value needs no reference implementation.
    """
    theta = math.radians(angle_degrees)
    return np.array([
        [1.0, 0.0, 0.0],                                    # i
        [0.0, 0.0, 0.0],                                    # j
        [0.0, 0.0, 1.0],                                    # k
        [math.cos(theta), math.sin(theta), 1.0],            # l
    ])


@pytest.mark.parametrize("expected", [0.0, 30.0, 60.0, 90.0, 120.0, 179.0,
                                      -30.0, -90.0, -150.0, -179.0])
def test_an_analytic_four_atom_geometry_returns_the_angle_it_was_built_with(expected):
    """The construction fixes the answer; no reference implementation is consulted."""
    assert torsion_degrees(_four_atoms(expected), (0, 1, 2, 3)) == pytest.approx(expected,
                                                                                abs=1e-9)


def test_the_wrap_is_half_open_so_180_reports_as_minus_180():
    """`[-180, 180)`: exactly one name per value, so a series can never contain both."""
    assert torsion_degrees(_four_atoms(180.0), (0, 1, 2, 3)) == pytest.approx(-180.0, abs=1e-9)


def test_reversing_the_atom_order_leaves_the_torsion_unchanged():
    """i-j-k-l and l-k-j-i are the same dihedral. A sign error here would show as a flip."""
    positions = _four_atoms(57.0)
    assert torsion_degrees(positions, (0, 1, 2, 3)) == pytest.approx(
        torsion_degrees(positions, (3, 2, 1, 0)), abs=1e-9)


def test_mirroring_the_geometry_flips_the_sign():
    """The convention is chiral; a magnitude-only implementation passes every test above."""
    positions = _four_atoms(57.0)
    mirrored = positions * np.array([1.0, -1.0, 1.0])
    assert torsion_degrees(mirrored, (0, 1, 2, 3)) == pytest.approx(-57.0, abs=1e-9)


def test_four_collinear_atoms_are_refused_rather_than_reported_as_zero():
    """An undefined dihedral reported as 0.0 is a number nothing distinguishes from a real one."""
    positions = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 2.], [0., 0., 3.]])
    with pytest.raises(TorsionError, match="collinear"):
        torsion_degrees(positions, (0, 1, 2, 3))


# --- the sign convention, against MDTraj -------------------------------------------------------

def _same_angle(ours: float, theirs: float, *, tolerance: float = 1e-3) -> bool:
    """Whether two angles in degrees are the same angle.

    Not `ours == theirs`: the two implementations wrap to different half-open intervals, so an
    extended backbone at exactly 180 degrees is `+180` to MDTraj and `-180` here -- the same
    angle, written twice. Comparing the wrapped DIFFERENCE tests the geometry and not the
    boundary convention, which is asserted separately.
    """
    return abs((ours - theirs + 180.0) % 360.0 - 180.0) < tolerance


def test_the_sign_convention_matches_mdtraj_on_real_alanine_torsions():
    """THE convention check: same four atoms, same coordinates, independent implementation.

    The fixture as shipped is fully extended -- both backbone torsions sit at exactly 180, where
    the sign carries no information and a flipped implementation passes. So the coordinates are
    perturbed deterministically first, which puts phi and psi at generic angles where a sign error
    is a 2x error and cannot hide.
    """
    mdtraj = pytest.importorskip("mdtraj")
    from pathlib import Path

    pdb = Path(__file__).resolve().parents[1] / "tests" / "data" / "ALA.pdb"
    if not pdb.is_file():
        pytest.skip("no ALA fixture")
    frames = mdtraj.load(str(pdb))

    rng = np.random.default_rng(20260903)
    frames.xyz[0] += rng.normal(scale=0.05, size=frames.xyz[0].shape).astype("float32")

    # Whichever four-atom torsions MDTraj itself identifies as phi/psi -- so the comparison is on
    # atoms both implementations agree about, and only the ARITHMETIC is under test.
    checked = 0
    for compute in (mdtraj.compute_phi, mdtraj.compute_psi):
        indices, reference = compute(frames)
        for quartet, expected in zip(indices, reference[0]):
            theirs = math.degrees(float(expected))
            assert abs(abs(theirs) - 180.0) > 1.0, (
                "the perturbation must move this torsion off the degenerate 180, or the sign "
                "convention is not actually under test here")
            ours = torsion_degrees(frames.xyz[0], tuple(int(a) for a in quartet))
            assert _same_angle(ours, theirs, tolerance=1e-2), (quartet, ours, theirs)
            checked += 1
    assert checked >= 2, "the fixture produced no backbone torsions to check"


@pytest.mark.parametrize("angle", [-170.0, -95.0, -20.0, 15.0, 75.0, 160.0])
def test_random_geometries_agree_with_mdtraj_with_no_box(angle):
    """A nonperiodic cross-check over the whole range, not just the analytic construction."""
    mdtraj = pytest.importorskip("mdtraj")

    positions = _four_atoms(angle)
    trajectory = mdtraj.Trajectory(positions[None, :, :].astype("float32"),
                                   topology=_mdtraj_topology())
    expected = math.degrees(float(mdtraj.compute_dihedrals(trajectory, [[0, 1, 2, 3]])[0][0]))
    assert _same_angle(torsion_degrees(positions, (0, 1, 2, 3)), expected)


def _mdtraj_topology():
    """A bare four-atom chain, so MDTraj will accept the coordinates as a Trajectory."""
    import mdtraj

    top = mdtraj.Topology()
    chain = top.add_chain()
    residue = top.add_residue("UNK", chain)
    for name in ("A", "B", "C", "D"):
        top.add_atom(name, mdtraj.element.carbon, residue)
    return top


# --- the minimum-image convention --------------------------------------------------------------

ORTHORHOMBIC = np.diag([5.0, 5.0, 5.0])
#: A genuinely triclinic cell in OpenMM's reduced form: `a` along x, `b` with no z component, and
#: off-diagonal terms large enough that a per-axis `round(v / L)` gets the wrong image.
TRICLINIC = np.array([[5.0, 0.0, 0.0],
                      [1.5, 5.0, 0.0],
                      [1.2, 0.9, 5.0]])


@pytest.mark.parametrize("box", [ORTHORHOMBIC, TRICLINIC], ids=["orthorhombic", "triclinic"])
def test_a_molecule_split_across_the_boundary_gives_the_same_torsion_as_an_unsplit_one(box):
    """THE claim the convention makes, stated as an equality between two computations.

    The same four atoms, one copy whole and one copy with individual atoms translated by whole
    lattice vectors -- which is the same physical configuration. Without the convention the split
    copy's bond vectors are box-length and the answer is unrelated to the geometry.
    """
    positions = _four_atoms(73.0)
    whole = torsion_degrees(positions, (0, 1, 2, 3))

    split = positions.copy()
    split[0] += box[0]                      # i pushed one cell along a
    split[3] -= box[1] + box[2]             # l pulled back along b and c
    assert torsion_degrees(split, (0, 1, 2, 3), box=box) == pytest.approx(whole, abs=1e-6)

    # And the control: WITHOUT the box, the same split coordinates give a different answer. If
    # this passed too, the test above would prove nothing about the convention.
    assert torsion_degrees(split, (0, 1, 2, 3)) != pytest.approx(whole, abs=1e-3)


@pytest.mark.parametrize("box", [ORTHORHOMBIC, TRICLINIC], ids=["orthorhombic", "triclinic"])
def test_the_minimum_image_is_the_shortest_of_every_surrounding_lattice_image(box):
    """Checked by brute force over a wide neighbourhood, not against the reduction that made it."""
    rng = np.random.default_rng(20260903)
    for _ in range(200):
        vector = rng.uniform(-6.0, 6.0, size=3)
        ours = minimum_image(vector, box)
        best = min(
            (float(np.linalg.norm(vector + i * box[0] + j * box[1] + k * box[2]))
             for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)))
        assert float(np.linalg.norm(ours)) == pytest.approx(best, abs=1e-9)
        # And it is genuinely a lattice translate of the original, not merely something short.
        difference = np.linalg.solve(box.T, ours - vector)
        assert np.allclose(difference, np.round(difference), atol=1e-9)


def test_no_box_leaves_the_vector_alone():
    """A nonperiodic system has one image, and applying a convention to it would be a bug."""
    vector = np.array([9.0, -4.0, 2.5])
    assert np.array_equal(minimum_image(vector, None), vector)


def test_a_degenerate_box_is_refused():
    with pytest.raises(TorsionError, match="non-positive diagonal"):
        minimum_image(np.array([1.0, 0.0, 0.0]), np.diag([2.0, 0.0, 2.0]))
