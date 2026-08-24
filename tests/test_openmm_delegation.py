"""Where this package delegates to OpenMM, the delegation must be exact.

Two pieces of geometry and mass handling used to be reimplemented here. They now call OpenMM, which
is better -- one definition instead of two that can drift -- but only while the delegation really is
equivalent. These tests are what makes that checkable rather than assumed.

One of them, `Modeller._computeBoxVectors`, is a **private** method. Depending on a private API is a
deliberate trade: it avoids duplicating a definition, at the cost of a dependency that can move
without notice. The fallback below keeps working if it does, and the test says so loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SHAPES = ["cube", "dodecahedron", "octahedron"]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("width", [1.0, 3.0, 4.7315])
def test_box_vectors_match_openmm(shape, width):
    """The local fallback must reproduce OpenMM's private helper exactly.

    If this fails, OpenMM changed or moved `_computeBoxVectors`. The fallback still runs, so nothing
    breaks silently -- but the two definitions have diverged and the difference must be understood
    before it reaches a box.
    """
    from openmm.app import Modeller

    from md_templates.openmm.solvation import _box_vectors_fallback

    theirs = np.array([[v.x, v.y, v.z]
                       for v in Modeller._computeBoxVectors(None, width, shape)])
    mine = _box_vectors_fallback(width, shape)
    assert np.allclose(mine, theirs, atol=1e-12), f"{shape}: {mine} vs {theirs}"


@pytest.mark.parametrize("shape", SHAPES)
def test_the_delegating_path_returns_the_same_vectors(shape):
    from md_templates.openmm.solvation import _box_vectors, _box_vectors_fallback

    assert np.allclose(_box_vectors(3.3, shape), _box_vectors_fallback(3.3, shape), atol=1e-12)


def test_an_unsupported_shape_is_refused_by_both_paths():
    from md_templates.openmm.solvation import _box_vectors, _box_vectors_fallback

    with pytest.raises(ValueError, match="unsupported"):
        _box_vectors(3.0, "hexagonal-prism")
    with pytest.raises(ValueError, match="unsupported"):
        _box_vectors_fallback(3.0, "hexagonal-prism")


def test_the_minimum_image_factor_is_what_the_padding_maths_assumes():
    """min-image is the smallest diagonal element of the reduced vectors, not the box width."""
    from md_templates.openmm.solvation import _box_vectors

    width = 4.0
    # truncated octahedron: the reduced vectors are (w,0,0), (w/3, 2sqrt2 w/3, 0),
    # (-w/3, sqrt2 w/3, sqrt6 w/3), so the smallest diagonal element is sqrt(6)/3 -- NOT sqrt(3)/2,
    # which is the factor for a different convention and understates the box by 6%.
    expected = {"cube": 1.0, "dodecahedron": 1 / np.sqrt(2), "octahedron": np.sqrt(6) / 3}
    for shape, fraction in expected.items():
        vectors = _box_vectors(width, shape)
        assert np.min(np.diag(vectors)) == pytest.approx(width * fraction, rel=1e-9), shape


# ---------------------------------------------------------------------------------------------
# hydrogen mass repartitioning
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_openmm_hydrogen_mass_matches_the_previous_implementation():
    """The delegation that replaced hand-rolled HMR, asserted atom by atom.

    OpenMM skips residues it made rigid, so with `rigidWater=True` its behaviour is exactly the
    "solute" scope this package wants. If that ever stops being true this test fails rather than a
    run quietly integrating repartitioned water.
    """
    from openmm import app, unit

    from md_templates.openmm.system import repartition_hydrogen_mass

    pdb = app.PDBFile(str(REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
                          / "systems" / "ace_ala_nme.pdb"))
    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(forcefield, model="tip4pew",
                        boxSize=(3.0, 3.0, 3.0) * unit.nanometer,
                        ionicStrength=0.15 * unit.molar, neutralize=True)
    n_solute = pdb.topology.getNumAtoms()

    def masses(system):
        return np.array([system.getParticleMass(i).value_in_unit(unit.amu)
                         for i in range(system.getNumParticles())])

    common = dict(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                  constraints=app.HBonds, rigidWater=True)
    delegated = forcefield.createSystem(modeller.topology,
                                        hydrogenMass=3.024 * unit.amu, **common)
    manual = forcefield.createSystem(modeller.topology, **common)
    repartition_hydrogen_mass(manual, modeller.topology, 3.024, range(n_solute))

    assert np.allclose(masses(delegated), masses(manual), atol=1e-9), (
        "OpenMM's hydrogenMass no longer matches this package's solute-scoped repartitioning")
    assert masses(delegated).sum() == pytest.approx(masses(manual).sum(), abs=1e-6)


@pytest.mark.slow
def test_the_verifier_refuses_water_that_was_repartitioned():
    """The check that OpenMM itself does not perform."""
    from openmm import app, unit

    from md_templates.openmm.system import verify_hydrogen_mass_repartitioning

    pdb = app.PDBFile(str(REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
                          / "systems" / "ace_ala_nme.pdb"))
    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(forcefield, model="tip4pew",
                        boxSize=(3.0, 3.0, 3.0) * unit.nanometer,
                        ionicStrength=0.15 * unit.molar, neutralize=True)

    # rigidWater=False lets OpenMM repartition water too, which this package forbids
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=app.HBonds, rigidWater=False,
                                     hydrogenMass=3.024 * unit.amu)
    with pytest.raises(ValueError, match="rigid water must not be repartitioned"):
        verify_hydrogen_mass_repartitioning(system, modeller.topology, 3.024, None)


@pytest.mark.slow
def test_the_verifier_refuses_a_heavy_atom_stripped_below_one_amu():
    """A target too high for a methyl leaves the carbon nearly massless. OpenMM does this silently."""
    from openmm import app, unit

    from md_templates.openmm.system import verify_hydrogen_mass_repartitioning

    pdb = app.PDBFile(str(REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
                          / "systems" / "ace_ala_nme.pdb"))
    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology, constraints=app.HBonds,
                                     hydrogenMass=5.0 * unit.amu)
    with pytest.raises(ValueError, match="was left with"):
        verify_hydrogen_mass_repartitioning(system, modeller.topology, 5.0, None)
