"""Building the implicit-solvent System, through ParmEd, exactly.

The Hamiltonian-defining path is::

    st = parmed.load_file(prmtop, xyz=inpcrd)
    parmed.tools.changeRadii(st, "mbondi3").execute()
    system = st.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        implicitSolvent=app.GBn2,
        removeCMMotion=True,
    )

## Why not `AmberPrmtopFile.createSystem()`

Because it is a different Hamiltonian. The two routes look interchangeable and are not. Measured on
this machine for ACE-ALA-NME, ff19SB, mbondi3 radii written by tleap:

    force                  ParmEd (reference)   AmberPrmtopFile        diff
    CustomGBForce                   -63.9813          -47.9292     16.0521
    NonbondedForce                  -97.7455          -97.7455     -0.0000
    PeriodicTorsionForce             10.0010           10.0010     -0.0000
    CMAPTorsionForce                 -1.6941           -1.6941      0.0000
    TOTAL                          -151.8192         -135.7670     16.0521

Every force agrees to 0.0000 kJ/mol except `CustomGBForce`, which differs by **16.05 kJ/mol** with
identical per-particle GB parameters and identical radii. The discrepancy is not the radii -- it is
the construction branch. Under REST2 that offset is several kT of spurious work, so every replica in
a ladder must sit on the same branch. This reproduces the figure documented by the pinned reference
(`csy0000/partitioned-REST2` at 537d5b6c), independently, here.

## Why `changeRadii` runs unconditionally

For a tleap-built protein topology it is a no-op: tleap has already written mbondi3 and the measured
maximum radius change is 0.0000 A. For an OpenFF/Sage topology it is load-bearing -- those prmtops
carry no GB radii at all, and without it the radii are zero. Applying it always is therefore safe
for the first case and required for the second, which is cheaper than deciding per route and
getting the decision wrong.

## What this module does not do

It does not run dynamics, and the Amber files it writes are construction intermediates and
provenance for OpenMM. There is no Amber execution engine here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = [
    "AMBER_TOPOLOGY_NAME",
    "AMBER_COORDINATE_NAME",
    "build_implicit_system",
    "implicit_provenance",
    "radii_of",
    "write_amber_files_from_openmm",
]

#: One canonical name for each Amber artefact. `rst7` rather than `inpcrd` so there is exactly one
#: coordinate authority in the bundle; ParmEd writes either from the same Structure.
AMBER_TOPOLOGY_NAME = "system.prmtop"
AMBER_COORDINATE_NAME = "system.rst7"

#: The OpenMM implicit-solvent objects, by canonical name.
_GB_MODELS = {
    "HCT": "HCT",
    "OBC1": "OBC1",
    "OBC2": "OBC2",
    "GBn": "GBn",
    "GBn2": "GBn2",
}


def _gb_object(model: str):
    from openmm import app

    if model not in _GB_MODELS:
        raise ValueError(f"unsupported implicit model {model!r}; known: {sorted(_GB_MODELS)}")
    return getattr(app, _GB_MODELS[model])


def radii_of(structure) -> list:
    """Per-atom GB radii in angstrom, for checking that `changeRadii` did what it claims."""
    return [float(atom.solvent_radius) for atom in structure.atoms]


def build_implicit_system(prmtop_path: Path, coordinate_path: Optional[Path] = None, *,
                          implicit_model: str = "GBn2", radii: str = "mbondi3",
                          remove_cm_motion: bool = True):
    """Build the implicit-solvent System, and report what the radius change actually did.

    Returns `(system, info)`. `info` records the radii before and after `changeRadii`, so a bundle
    can state whether the call mattered for this topology rather than asserting that it did.
    """
    import parmed as pmd
    from openmm import app
    from parmed.tools import changeRadii

    gb_object = _gb_object(implicit_model)

    structure = (pmd.load_file(str(prmtop_path), xyz=str(coordinate_path))
                 if coordinate_path is not None else pmd.load_file(str(prmtop_path)))
    before = radii_of(structure)
    changeRadii(structure, str(radii)).execute()
    after = radii_of(structure)

    system = structure.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        implicitSolvent=gb_object,
        removeCMMotion=remove_cm_motion,
    )

    max_change = max((abs(a - b) for a, b in zip(after, before)), default=0.0)
    info = {
        "construction": "parmed.Structure.createSystem",
        "construction_note": (
            "NOT AmberPrmtopFile.createSystem: the two differ by ~16 kJ/mol in CustomGBForce on "
            "ACE-ALA-NME with identical radii, so the branch is part of the Hamiltonian"),
        "implicit_model": implicit_model,
        "radii": radii,
        "nonbonded_method": "NoCutoff",
        "constraints": "HBonds",
        "remove_cm_motion": bool(remove_cm_motion),
        "radii_max_change_angstrom": max_change,
        "radii_change_was_a_no_op": max_change == 0.0,
        "n_particles": system.getNumParticles(),
        "n_constraints": system.getNumConstraints(),
        "uses_periodic_boundary_conditions": system.usesPeriodicBoundaryConditions(),
    }
    if system.usesPeriodicBoundaryConditions():
        raise RuntimeError(
            "the implicit-solvent System reports periodic boundary conditions, which it must not "
            "have: implicit solvent has no box. Refusing to publish it.")
    return system, info


def write_amber_files_from_openmm(topology, system, positions, out_dir: Path) -> dict:
    """Serialise an OpenMM topology/System to Amber files through ParmEd.

    Used by the OpenFF/Sage route, whose parameters exist only as an OpenMM System. The System
    passed here must be built WITHOUT constraints: constraints belong to `createSystem` on the way
    back out, and baking them into the topology would apply them twice.
    """
    import parmed as pmd
    from parmed import openmm as pmd_openmm

    out_dir = Path(out_dir)
    structure = pmd_openmm.load_topology(topology, system=system, xyz=positions)
    prmtop = out_dir / AMBER_TOPOLOGY_NAME
    coordinates = out_dir / AMBER_COORDINATE_NAME
    structure.save(str(prmtop), overwrite=True)
    structure.save(str(coordinates), format="rst7", overwrite=True)
    return {"prmtop": prmtop, "coordinates": coordinates, "parmed_version": pmd.__version__}


def implicit_provenance(info: dict) -> dict:
    """The record a bundle keeps about how its implicit System was built."""
    import openmm
    import parmed as pmd

    return {
        **info,
        "parmed_version": pmd.__version__,
        "openmm_version": openmm.__version__,
        "amber_files_are_provenance_only": (
            "system.prmtop and system.rst7 are construction intermediates and provenance for the "
            "OpenMM System. This repository has no Amber execution engine."),
        "no_water": True,
        "no_ions": True,
        "no_periodic_box": True,
        "no_barostat": "implicit solvent has no volume, so pressure is undefined",
    }
