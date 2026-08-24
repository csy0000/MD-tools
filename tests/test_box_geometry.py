"""Three geometric quantities that are not the same number, and were once treated as one.

For OpenMM's reduced triclinic boxes:

* the **shortest nonzero lattice translation** is how far a point sits from its own periodic image.
  For the cube, rhombic dodecahedron and truncated octahedron OpenMM builds, it equals the box
  WIDTH. This is what a solute-to-periodic-copy clearance is measured against.
* the **minimum reduced-box height**, `min(a_x, b_y, c_z)`, is what OpenMM's periodic-box check
  compares the cutoff against. It is `width`, `width/sqrt(2)` and `sqrt(6)/3 * width` respectively.
* the **solute bounding radius** is half the solute's diameter under OpenMM's own definition.

Using the second where the first belongs understates the real separation by 29% in a dodecahedron.
On a real macrocycle box that was the difference between an apparent 0.7 nm clearance and an actual
2.4 nm one -- and it produced a false verdict that a valid trajectory was unusable.

Every test here derives its expectation independently: by enumeration, by an explicit formula, or by
constructing a real OpenMM `Context`, never by calling the function under test.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SHAPES = ["cube", "dodecahedron", "octahedron"]

#: min(a_x, b_y, c_z) / width, per shape. Independent of the implementation.
HEIGHT_FRACTION = {
    "cube": 1.0,
    "dodecahedron": 1.0 / np.sqrt(2.0),
    "octahedron": np.sqrt(6.0) / 3.0,
}

#: volume / width^3, per shape.
VOLUME_FRACTION = {
    "cube": 1.0,
    "dodecahedron": 1.0 / np.sqrt(2.0),
    "octahedron": 4.0 * np.sqrt(2.0) / (3.0 * np.sqrt(3.0)) * np.sqrt(3.0) / 2.0,
}


def _brute_force_shortest_translation(vectors, span=3):
    """Independent enumeration, wider than the implementation's own search."""
    best = np.inf
    for i, j, k in itertools.product(range(-span, span + 1), repeat=3):
        if (i, j, k) == (0, 0, 0):
            continue
        best = min(best, float(np.linalg.norm(i * vectors[0] + j * vectors[1] + k * vectors[2])))
    return best


# ---------------------------------------------------------------------------------------------
# 1. local vectors equal the installed OpenMM's
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("width", [1.0, 3.0, 5.137])
def test_fallback_vectors_equal_installed_openmm(shape, width):
    from openmm.app import Modeller

    from md_templates.openmm.solvation import _box_vectors, _box_vectors_fallback

    theirs = np.array([[v.x, v.y, v.z]
                       for v in Modeller._computeBoxVectors(None, width, shape)])
    assert np.allclose(_box_vectors_fallback(width, shape), theirs, atol=1e-12)
    assert np.allclose(_box_vectors(width, shape), theirs, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# 2. the shortest lattice translation is the WIDTH for all three shapes
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("width", [1.0, 2.5, 4.7315])
def test_shortest_lattice_translation_is_the_box_width(shape, width):
    from md_templates.openmm.solvation import _box_vectors, shortest_lattice_translation

    vectors = _box_vectors(width, shape)
    brute = _brute_force_shortest_translation(vectors)
    assert brute == pytest.approx(width, rel=1e-12), f"{shape}: enumeration gave {brute}"
    assert shortest_lattice_translation(vectors) == pytest.approx(width, rel=1e-12)


def test_the_translation_and_the_height_genuinely_differ():
    """If these ever coincided for a dodecahedron the distinction would be untestable."""
    from md_templates.openmm.solvation import (_box_vectors, minimum_reduced_box_height,
                                               shortest_lattice_translation)

    vectors = _box_vectors(4.0, "dodecahedron")
    translation = shortest_lattice_translation(vectors)
    height = minimum_reduced_box_height(vectors)
    assert translation > height
    assert height / translation == pytest.approx(1 / np.sqrt(2), rel=1e-12)


# ---------------------------------------------------------------------------------------------
# 3. the reduced-box height has the three stated factors
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_reduced_box_height_factors(shape):
    from md_templates.openmm.solvation import _box_vectors, minimum_reduced_box_height

    width = 3.0
    height = minimum_reduced_box_height(_box_vectors(width, shape))
    assert height == pytest.approx(width * HEIGHT_FRACTION[shape], rel=1e-12)


# ---------------------------------------------------------------------------------------------
# 4. openmm semantics reproduces the width formula and delivers the requested clearance
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("radius,padding", [(0.4, 2.0), (0.9, 2.0), (1.5, 2.0), (0.2, 1.2)])
def test_openmm_semantics_reproduces_the_width_formula(radius, padding):
    """`width = max(2R + padding, 2*padding)`, and clearance is at least the padding asked for."""
    from md_templates.openmm.solvation import shortest_lattice_translation, _box_vectors

    expected_width = max(2 * radius + padding, 2 * padding)
    vectors = _box_vectors(expected_width, "dodecahedron")
    clearance = shortest_lattice_translation(vectors) - 2 * radius
    assert clearance >= padding - 1e-12, (
        f"radius {radius}, padding {padding}: clearance {clearance:.4f} is below the request")


@pytest.mark.slow
def test_resolved_geometry_matches_the_formula_end_to_end():
    """Through `_resolve_box`, on a real solute, with every recorded number reconciled."""
    from openmm import app, unit

    from md_templates.openmm.solvation import _resolve_box

    pdb = app.PDBFile(str(REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
                          / "systems" / "ace_ala_nme.pdb"))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    cfg = {
        "solvation": {"box_shape": "dodecahedron", "padding_nm": 2.0,
                      "padding_semantics": "openmm", "cutoff_fit_policy": "grow"},
        "system_build": {"nonbonded_cutoff_nm": 1.0, "minimum_image_margin_nm": 0.10},
    }
    g = _resolve_box(modeller, cfg)

    radius = g["solute_bounding_radius_nm"]
    width = g["box_width_nm"]
    assert g["box_width_requested_nm"] == pytest.approx(max(2 * radius + 2.0, 4.0), abs=1e-4)
    assert g["shortest_lattice_translation_nm"] == pytest.approx(width, abs=1e-4)
    assert g["solute_image_clearance_nm"] == pytest.approx(width - 2 * radius, abs=1e-4)
    assert g["min_reduced_box_height_nm"] == pytest.approx(width / np.sqrt(2), abs=1e-4)
    assert g["solute_image_clearance_nm"] >= 2.0 - 1e-4, (
        "openmm semantics must deliver at least the requested bounding-sphere clearance")
    # the legacy fields still hold the height, and are labelled as such
    assert g["min_image_distance_nm"] == pytest.approx(g["min_reduced_box_height_nm"], abs=1e-9)
    assert "not the solute-to-periodic-copy distance" in g["legacy_field_note"]


# ---------------------------------------------------------------------------------------------
# 5 & 6. cutoff growth uses the HEIGHT, and the result survives a real Context
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_cutoff_growth_uses_the_height_and_builds_a_real_context():
    """A tiny solute forces the grow branch; the grown box must satisfy OpenMM itself."""
    import openmm
    from openmm import app, unit

    from md_templates.openmm.solvation import _resolve_box

    pdb = app.PDBFile(str(REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
                          / "systems" / "ace_ala_nme.pdb"))
    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    cfg = {
        "solvation": {"box_shape": "dodecahedron", "padding_nm": 0.5,   # deliberately too small
                      "padding_semantics": "openmm", "cutoff_fit_policy": "grow"},
        "system_build": {"nonbonded_cutoff_nm": 1.0, "minimum_image_margin_nm": 0.10},
    }
    g = _resolve_box(modeller, cfg)
    assert g["grown_for_cutoff"] is True
    assert g["min_reduced_box_height_nm"] == pytest.approx(2.0 + 0.10, abs=1e-4)

    modeller.addSolvent(forcefield, model="tip4pew",
                        boxVectors=unit.Quantity(
                            tuple(openmm.Vec3(*v) for v in g["box_vectors_nm"]), unit.nanometer),
                        ionicStrength=0.15 * unit.molar, neutralize=True)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=app.HBonds)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                 0.002 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(modeller.positions)
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert np.isfinite(energy)
    del context, integrator


@pytest.mark.parametrize("contraction", [0.0, 0.01, 0.02, 0.03])
def test_a_1nm_cutoff_survives_the_expected_npt_contraction(contraction):
    """The margin exists so an NPT contraction does not cross OpenMM's hard limit mid-run."""
    from md_templates.openmm.solvation import _box_vectors, minimum_reduced_box_height

    cutoff, margin = 1.0, 0.10
    # the smallest box the grow policy would produce
    width = (2 * cutoff + margin) / HEIGHT_FRACTION["dodecahedron"]
    contracted = minimum_reduced_box_height(_box_vectors(width * (1 - contraction), "dodecahedron"))
    assert contracted >= 2 * cutoff, (
        f"a {contraction:.0%} contraction leaves height {contracted:.4f} nm, below the "
        f"{2 * cutoff} nm OpenMM requires")


# ---------------------------------------------------------------------------------------------
# 7. representative ALA and RGDfV boxes are internally consistent
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,radius", [("alanine", 0.473), ("cyclo-RGDfV", 0.880)])
def test_representative_boxes_are_physically_consistent(name, radius):
    from md_templates.openmm.solvation import (_box_vectors, minimum_reduced_box_height,
                                               shortest_lattice_translation)

    padding, cutoff, margin = 2.0, 1.0, 0.10
    width = max(2 * radius + padding, 2 * padding)
    vectors = _box_vectors(width, "dodecahedron")

    translation = shortest_lattice_translation(vectors)
    height = minimum_reduced_box_height(vectors)
    clearance = translation - 2 * radius
    volume = abs(np.linalg.det(vectors))

    assert clearance >= padding - 1e-9, f"{name}: clearance {clearance:.3f} < padding {padding}"
    assert height >= 2 * cutoff + margin, f"{name}: height {height:.3f} too small for the cutoff"
    assert volume == pytest.approx(width ** 3 / np.sqrt(2), rel=1e-9)
    assert clearance > 2 * cutoff, (
        f"{name}: with {padding} nm padding the solute cannot see its own copy through the cutoff")


# ---------------------------------------------------------------------------------------------
# 8. volume ratios at equal periodic-copy spacing
# ---------------------------------------------------------------------------------------------

def test_volume_ratios_at_equal_copy_spacing_match_openmm():
    """The reason a dodecahedron is used at all: same copy spacing, less water.

    At equal shortest-lattice-translation -- equal spacing between periodic copies -- the
    dodecahedron holds 1/sqrt(2) of a cube's volume, which is the 70.7% OpenMM's documentation
    quotes.
    """
    from md_templates.openmm.solvation import _box_vectors, shortest_lattice_translation

    spacing = 5.0
    volumes = {}
    for shape in SHAPES:
        vectors = _box_vectors(spacing, shape)
        assert shortest_lattice_translation(vectors) == pytest.approx(spacing, rel=1e-12)
        volumes[shape] = abs(np.linalg.det(vectors))

    assert volumes["dodecahedron"] / volumes["cube"] == pytest.approx(1 / np.sqrt(2), rel=1e-9)
    assert volumes["dodecahedron"] / volumes["cube"] == pytest.approx(0.7071, abs=1e-4)
    assert volumes["octahedron"] / volumes["cube"] == pytest.approx(0.7698, abs=1e-4)
    assert volumes["dodecahedron"] < volumes["octahedron"] < volumes["cube"]
