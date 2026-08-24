"""The dodecahedron, measured rather than asserted.

An audit on 2026-08-24 was asked to prove the implementation wrong before changing it. It could not:
every geometric invariant holds on the generated OpenMM `System`. What was wrong was the
explanation. Comments in `solvation.py`, `config.py` and two documents called `width/sqrt(2)` the
"minimum image distance", and derived from that a claim that `padding = 1.2` yields "roughly 0.4 nm
of real clearance". The measured clearance there is 1.489 nm.

The distinction these tests pin down, for the three shapes OpenMM builds:

===================  ==============  =================  ====================================
quantity             cube            dodecahedron       what it is for
===================  ==============  =================  ====================================
shortest translation `width`         `width`            how far the solute is from its image
min perp. height     `width`         `width/sqrt(2)`    what OpenMM's cutoff check uses
max legal cutoff     `width/2`       `width/(2 sqrt2)`  half the height
===================  ==============  =================  ====================================

The shortest lattice translation is `width` for ALL THREE shapes. That is the fact the old comments
denied, and the reason they understated real clearance.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm.app import Modeller  # noqa: E402

SHAPES = ("cube", "dodecahedron", "octahedron")


def _vectors(width, shape):
    modeller = Modeller.__new__(Modeller)
    return np.array([[x for x in v] for v in modeller._computeBoxVectors(width, shape)])


def _shortest_translation(v, span=3):
    """By enumeration, not by formula -- the formula is the thing under test."""
    a, b, c = v
    return min(np.linalg.norm(n[0] * a + n[1] * b + n[2] * c)
               for n in itertools.product(range(-span, span + 1), repeat=3) if n != (0, 0, 0))


def _perpendicular_heights(v):
    a, b, c = v
    volume = abs(np.linalg.det(v))
    return (volume / np.linalg.norm(np.cross(b, c)),
            volume / np.linalg.norm(np.cross(c, a)),
            volume / np.linalg.norm(np.cross(a, b)))


# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_the_shortest_lattice_translation_is_the_width_for_every_shape(shape):
    """The claim the old comments denied. Enumerated over -3..3, as the audit specified."""
    width = 4.0
    assert _shortest_translation(_vectors(width, shape)) == pytest.approx(width, rel=1e-12)


def test_a_wider_enumeration_finds_nothing_shorter():
    """Guards the enumeration itself: a too-small span could miss the true minimum."""
    v = _vectors(4.0, "dodecahedron")
    assert _shortest_translation(v, span=5) == pytest.approx(_shortest_translation(v, span=3))


def test_twelve_translations_tie_at_the_minimum_for_a_dodecahedron():
    """A rhombic dodecahedron has 12 nearest neighbours. Fewer would mean a different lattice."""
    a, b, c = _vectors(4.0, "dodecahedron")
    shortest = _shortest_translation(np.array([a, b, c]))
    tied = [n for n in itertools.product(range(-3, 4), repeat=3)
            if n != (0, 0, 0)
            and abs(np.linalg.norm(n[0]*a + n[1]*b + n[2]*c) - shortest) < 1e-9]
    assert len(tied) == 12


@pytest.mark.parametrize("shape,factor", [
    ("cube", 1.0), ("dodecahedron", 1.0 / np.sqrt(2.0)), ("octahedron", np.sqrt(6.0) / 3.0),
])
def test_the_minimum_perpendicular_height_has_the_expected_factor(shape, factor):
    """`V/|b x c|` and cyclic permutations -- the audit's definition, not `min(diag)`."""
    width = 4.0
    assert min(_perpendicular_heights(_vectors(width, shape))) == pytest.approx(
        width * factor, rel=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_diagonal_shortcut_equals_the_true_perpendicular_height(shape):
    """`minimum_reduced_box_height` uses `min(a_x, b_y, c_z)`; that must be the real height.

    It is a shortcut, and shortcuts are worth checking against the definition they stand in for.
    """
    from md_templates.openmm.solvation import minimum_reduced_box_height

    v = _vectors(4.0, shape)
    assert minimum_reduced_box_height(v) == pytest.approx(
        min(_perpendicular_heights(v)), rel=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_height_and_translation_are_not_interchangeable(shape):
    """The whole point. For a cube they coincide; for the other two they must not."""
    v = _vectors(4.0, shape)
    height = min(_perpendicular_heights(v))
    translation = _shortest_translation(v)
    if shape == "cube":
        assert height == pytest.approx(translation)
    else:
        assert height < translation
        assert height / translation < 0.85


def test_the_dodecahedral_height_understates_clearance_by_29_percent():
    """The size of the error the old comments made, pinned so it cannot be reintroduced."""
    v = _vectors(4.0, "dodecahedron")
    ratio = min(_perpendicular_heights(v)) / _shortest_translation(v)
    assert ratio == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-12)
    assert (1.0 - ratio) == pytest.approx(0.2929, abs=1e-4)


# -------------------------------------------------------------------------------------------
# Padding semantics
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("radius,padding", [(0.4555, 2.0), (0.4555, 0.5), (1.5, 2.0), (3.0, 2.0)])
def test_padding_is_a_lower_bound_on_solute_image_separation(radius, padding):
    """`width = max(2R + padding, 2*padding)`, and separation >= width - 2R.

    Never an upper bound: when `padding > 2R` the `2*padding` term wins and the separation exceeds
    the padding that was asked for.
    """
    width = max(2 * radius + padding, 2 * padding)
    separation = _shortest_translation(_vectors(width, "dodecahedron")) - 2 * radius
    assert separation >= padding - 1e-12


def test_the_repository_default_delivers_more_than_the_two_nanometres_requested():
    """Alanine at the 2.0 nm default: the `2*padding` branch, so clearance exceeds the request."""
    radius, padding = 0.4555, 2.0
    width = max(2 * radius + padding, 2 * padding)
    assert width == pytest.approx(4.0)
    assert _shortest_translation(_vectors(width, "dodecahedron")) - 2 * radius == pytest.approx(
        3.089, abs=1e-3)


def test_the_old_comment_understated_clearance_at_padding_1_2():
    """It claimed ~0.4 nm; the measured value is 1.489 nm. Pinned so the claim cannot return."""
    radius, padding = 0.4555, 1.2
    width = max(2 * radius + padding, 2 * padding)
    height_based = width / np.sqrt(2.0) - 2 * radius        # the wrong calculation
    true_clearance = width - 2 * radius                     # the right one
    assert true_clearance == pytest.approx(1.489, abs=1e-3)
    assert height_based < true_clearance


# -------------------------------------------------------------------------------------------
# Cutoff legality is a SEPARATE invariant from image clearance
# -------------------------------------------------------------------------------------------
def test_cutoff_legality_and_image_clearance_are_independent():
    """A box can clear the solute generously and still be illegal for the cutoff.

    They are separate checks and neither implies the other; conflating them is what produced the
    wrong explanation in the first place.
    """
    radius, padding, cutoff = 0.4555, 1.2, 1.0
    width = max(2 * radius + padding, 2 * padding)
    clearance = _shortest_translation(_vectors(width, "dodecahedron")) - 2 * radius
    height = min(_perpendicular_heights(_vectors(width, "dodecahedron")))
    assert clearance > 1.4                       # generous clearance
    assert 2 * cutoff > height                   # and yet illegal for a 1.0 nm cutoff


def test_openmm_itself_accepts_the_repository_default_box():
    """The authoritative check: OpenMM raises on an illegal cutoff, so a built Context is proof."""
    from openmm import unit

    width, cutoff = 4.0, 1.0
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*(_vectors(width, "dodecahedron") * unit.nanometer))
    system.addParticle(12.0 * unit.amu)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.PME)
    force.setCutoffDistance(cutoff * unit.nanometer)
    force.addParticle(0.0, 0.3, 0.5)
    system.addForce(force)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    assert context is not None

    # and it refuses one that exceeds half the height, which is what makes the above meaningful
    force.setCutoffDistance(1.5 * unit.nanometer)          # > width/(2 sqrt 2) = 1.414
    with pytest.raises(Exception):
        openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
