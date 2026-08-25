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

import json
from pathlib import Path
from typing import Optional

__all__ = [
    "AMBER_TOPOLOGY_NAME",
    "build_amber_topology_via_tleap",
    "build_implicit_bundle_inputs",
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


def _implicit_hmr_record(system, structure, scope: str, target: Optional[float],
                         mass_before: float) -> dict:
    """What the repartitioning actually did, measured on the built System.

    Reported rather than asserted: the count comes from comparing each particle's mass against the
    topology's element, so a record claiming 12 repartitioned hydrogens means twelve were found to
    have moved, not that twelve were asked for.
    """
    from openmm import unit as _u

    masses = [system.getParticleMass(i).value_in_unit(_u.dalton)
              for i in range(system.getNumParticles())]
    total_after = sum(masses)
    n_hydrogens = sum(1 for a in structure.atoms if a.atomic_number == 1)
    moved = sum(1 for a, m in zip(structure.atoms, masses)
                if a.atomic_number == 1 and abs(m - a.mass) > 1e-9)
    heaviest_donor_drop = min(
        (m for a, m in zip(structure.atoms, masses) if a.atomic_number != 1), default=None)
    record = {
        "scope": scope,
        "target_hydrogen_mass_amu": (float(target) if target is not None else None),
        "n_hydrogens": n_hydrogens,
        "n_hydrogens_repartitioned": moved,
        "total_mass_amu_before": round(float(mass_before), 6),
        "total_mass_amu_after": round(float(total_after), 6),
        "lightest_heavy_atom_amu": (round(float(heaviest_donor_drop), 6)
                                    if heaviest_donor_drop is not None else None),
    }
    if scope in ("solute", "all"):
        # Under implicit solvent there is no solvent to exclude, so the two scopes coincide.
        record["scope_note"] = ("implicit solvent has no solvent atoms, so 'solute' and 'all' "
                                "select the same particles")
        if abs(total_after - mass_before) > 1e-6:
            raise RuntimeError(
                f"hydrogen mass repartitioning changed the total mass from {mass_before:.6f} to "
                f"{total_after:.6f} amu. Repartitioning moves mass between bonded atoms and must "
                f"conserve it; a change means mass was created or destroyed.")
        if heaviest_donor_drop is not None and heaviest_donor_drop < 1.0:
            raise RuntimeError(
                f"repartitioning drove a heavy atom to {heaviest_donor_drop:.4f} amu, which is "
                f"lighter than hydrogen. Lower the target hydrogen mass ({target}).")
    return record


def build_implicit_system(prmtop_path: Path, coordinate_path: Optional[Path] = None, *,
                          implicit_model: str = "GBn2", radii: str = "mbondi3",
                          remove_cm_motion: bool = True,
                          hydrogen_mass_amu: Optional[float] = None,
                          hmr_scope: str = "none"):
    """Build the implicit-solvent System, and report what the radius change actually did.

    Returns `(system, info)`. `info` records the radii before and after `changeRadii`, so a bundle
    can state whether the call mattered for this topology rather than asserting that it did.

    `hydrogen_mass_amu` repartitions at BUILD time, through ParmEd's own `createSystem`, so the
    serialized System is what will actually be integrated. It used to be impossible to ask for:
    the implicit route dropped the request and returned 1.008 amu hydrogens, which a `-hmr-v1`
    profile then integrated at 4 fs.

    Under implicit solvent there is no solvent, so the solute IS the whole system and `hmr_scope`
    "solute" and "all" name the same set of atoms. Both are accepted and recorded, rather than
    letting "solute" look like a setting that was ignored.
    """
    import parmed as pmd
    from openmm import app
    from openmm import unit as u
    from parmed.tools import changeRadii

    gb_object = _gb_object(implicit_model)

    structure = (pmd.load_file(str(prmtop_path), xyz=str(coordinate_path))
                 if coordinate_path is not None else pmd.load_file(str(prmtop_path)))
    before = radii_of(structure)
    changeRadii(structure, str(radii)).execute()
    after = radii_of(structure)

    scope = str(hmr_scope or "none")
    if scope not in ("none", "solute", "all"):
        raise ValueError(f"hmr_scope must be 'none', 'solute' or 'all'; got {scope!r}")
    if scope != "none" and hydrogen_mass_amu is None:
        raise ValueError(
            f"hmr_scope={scope!r} asks for hydrogen mass repartitioning but no "
            "hydrogen_mass_amu was given, so there is no target mass to repartition to.")
    if scope == "none" and hydrogen_mass_amu is not None:
        raise ValueError(
            f"hydrogen_mass_amu={hydrogen_mass_amu!r} was given with hmr_scope='none', so it "
            "would be silently ignored while the manifest recorded a repartitioned System.")

    mass_before = sum(a.mass for a in structure.atoms)
    system = structure.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        implicitSolvent=gb_object,
        removeCMMotion=remove_cm_motion,
        # ParmEd applies the repartitioning itself. Delegating avoids a THIRD implementation of
        # arithmetic that already exists twice in system.py, and keeps the masses inside the
        # System that gets serialized.
        **({"hydrogenMass": float(hydrogen_mass_amu) * u.dalton} if scope != "none" else {}),
    )

    hmr_record = _implicit_hmr_record(system, structure, scope, hydrogen_mass_amu, mass_before)

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
        "hmr": hmr_record,
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


def build_amber_topology_via_tleap(pdb_path: Path, out_dir: Path, *, radii: str = "mbondi3",
                                   protein_forcefield: str = "leaprc.protein.ff19SB") -> dict:
    """Build prmtop/rst7 for a peptide or protein with tleap.

    `set default PBRadii mbondi3` is issued BEFORE `saveAmberParm`, which is what writes the radii
    into the topology. `changeRadii` still runs later and is a no-op here -- belt and braces for the
    case where a topology arrives without them.

    tleap's stdout is captured and kept: a run that "succeeded" while dropping an atom is a real
    failure mode, and the log is the only place it shows.
    """
    import shutil
    import subprocess

    if shutil.which("tleap") is None:
        raise RuntimeError(
            "tleap was not found on PATH. Implicit preparation of a peptide route builds its Amber "
            "topology with tleap (AmberTools); activate an environment that provides it.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prmtop = out_dir / AMBER_TOPOLOGY_NAME
    coordinates = out_dir / AMBER_COORDINATE_NAME
    leap_pdb = out_dir / "tleap_out.pdb"
    script = out_dir / "tleap.in"
    log = out_dir / "tleap.log"

    script.write_text("\n".join([
        f"source {protein_forcefield}",
        f"set default PBRadii {radii}",
        f"mol = loadPdb {pdb_path.resolve()}",
        f"saveAmberParm mol {prmtop.resolve()} {coordinates.resolve()}",
        f"savePdb mol {leap_pdb.resolve()}",
        "quit",
    ]) + "\n", encoding="utf-8")

    result = subprocess.run(["tleap", "-f", str(script)], capture_output=True, text=True,
                            check=False, cwd=str(out_dir))
    log.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0 or not prmtop.is_file():
        raise RuntimeError(
            f"tleap failed building the implicit topology from {pdb_path.name}.\n"
            f"  Its log is at {log}.\n{(result.stderr or result.stdout)[-1500:]}")
    return {
        "prmtop": prmtop,
        "coordinates": coordinates,
        "topology_pdb": leap_pdb,
        "tleap_input": script,
        "tleap_log": log,
        "tleap_commands": script.read_text(encoding="utf-8").splitlines(),
        "protein_forcefield": protein_forcefield,
        "radii_requested": radii,
    }


def build_implicit_bundle_inputs(*, route: str, cfg: dict, staging: Path,
                                 pdb: Optional[Path] = None, smiles: Optional[str] = None,
                                 implicit_model: str = "GBn2", radii: str = "mbondi3",
                                 hydrogen_mass_amu: Optional[float] = None,
                                 hmr_scope: str = "none") -> dict:
    """Produce every Amber and OpenMM artefact an implicit bundle needs.

    Two routes reach the same ParmEd construction from different directions:

    * `peptide` -- tleap writes the topology with mbondi3 radii already in it;
    * `ligand` -- the vetted OpenFF/Sage parameterisation is serialised to Amber files through
      ParmEd. That System is built with `constraints=None` on purpose: constraints are applied by
      `createSystem` on the way back out, and baking them in here would apply them twice.
    """
    from openmm import XmlSerializer, app

    staging = Path(staging)
    if route == "peptide":
        if pdb is None:
            raise ValueError("the peptide route needs a PDB input")
        amber = build_amber_topology_via_tleap(pdb, staging, radii=radii)
        topology_source = amber["topology_pdb"]
    elif route == "ligand":
        amber = _amber_files_for_ligand(cfg, staging, smiles=smiles)
        topology_source = amber["topology_pdb"]
    else:
        raise ValueError(f"implicit preparation has no route {route!r}; expected peptide or ligand")

    system, info = build_implicit_system(
        amber["prmtop"], amber["coordinates"],
        implicit_model=implicit_model, radii=radii,
        hydrogen_mass_amu=hydrogen_mass_amu, hmr_scope=hmr_scope)

    (staging / "system.xml").write_text(XmlSerializer.serialize(system), encoding="utf-8")
    pdb_file = app.PDBFile(str(topology_source))
    with (staging / "topology.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(pdb_file.topology, pdb_file.positions, handle, keepIds=True)

    # The same build record the explicit path writes, so everything downstream -- the bundle
    # manifest, the REST2 bridge, omega exclusion -- reads one shape. `geometry` is null rather
    # than absent: "this System has no periodic box" is a fact worth recording, and a missing key
    # would be indistinguishable from a record that forgot to write it.
    from .system import classify_omega_bonds, omega_central_bonds

    topology = app.PDBFile(str(staging / "topology.pdb")).topology
    solute = list(range(system.getNumParticles()))
    # The ligand route needs the SDF: bond orders are not recoverable from a topology, and amide
    # detection depends on them. It is written by the same step that built the conformer.
    ligand_sdf = amber.get("ligand_sdf")
    omega_info = classify_omega_bonds(
        topology, solute, route=("peptide" if route == "peptide" else "ligand"),
        ligand_sdf=(Path(ligand_sdf) if ligand_sdf else None))
    build_record = {
        "suffix": "system",
        "route": route,
        "input_route": "pdb" if route == "peptide" else "smiles",
        "n_particles": system.getNumParticles(),
        "n_constraints": system.getNumConstraints(),
        "n_solute_atoms": system.getNumParticles(),
        "n_waters": 0,
        "ions": None,
        "salt": None,
        "water": None,
        "geometry": None,
        "nonbonded": {"method": "NoCutoff", "cutoff_nm": None},
        # measured on the built System, not restated from the request
        "hmr": info["hmr"],
        "constraints": "HBonds",
        "rigid_water": False,
        "omega_central_bonds": omega_central_bonds(topology, solute),
        **{k: omega_info[k] for k in
           ("omega_unscaled_bonds", "omega_proline_like_scaled_bonds",
            "omega_unclassified_candidates", "omega_detection_method", "omega_detail")
           if k in omega_info},
        "degrees_of_freedom": (3 * system.getNumParticles() - system.getNumConstraints() - 3),
        "implicit": info,
    }
    (staging / "system_simbox.json").write_text(
        json.dumps(build_record, indent=2) + "\n", encoding="utf-8")

    return {
        "system": system,
        "system_xml": staging / "system.xml",
        "topology_pdb": staging / "topology.pdb",
        "build_record": build_record,
        # Surfaced at the top level so the caller writing forcefield.json states what the System
        # actually carries. Without it that file said "none" while system_manifest.json said
        # "solute", and the bundle's own three-way agreement check refused to publish -- correctly.
        "hmr": info["hmr"],
        "prmtop": amber["prmtop"],
        "coordinates": amber["coordinates"],
        "n_particles": system.getNumParticles(),
        "n_solute_atoms": system.getNumParticles(),   # implicit: the solute IS the system
        "route": route,
        "build": {**info, **{k: v for k, v in amber.items()
                             if k in ("tleap_commands", "protein_forcefield", "radii_requested",
                                      "small_molecule_forcefield", "charge_method")}},
    }


def _amber_files_for_ligand(cfg: dict, staging: Path, *, smiles: Optional[str]) -> dict:
    """OpenFF/Sage parameters, serialised to Amber files through ParmEd."""
    from openmm import app

    from .system import build_forcefield, initial_structure

    if not smiles:
        raise ValueError("the implicit ligand route needs a SMILES input")

    structure = initial_structure(smiles, staging / "structure", cfg)
    ligand_sdf = Path(structure["solute_sdf"])
    forcefield, ff_info = build_forcefield(cfg, ligand_sdf=ligand_sdf, route="ligand")

    solute = app.PDBFile(str(structure["solute_pdb"]))
    # No constraints here: they are applied by createSystem on the way back out.
    bare = forcefield.createSystem(solute.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    written = write_amber_files_from_openmm(solute.topology, bare, solute.positions, staging)

    topology_pdb = staging / "ligand_topology.pdb"
    with topology_pdb.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(solute.topology, solute.positions, handle, keepIds=True)
    return {
        "prmtop": written["prmtop"],
        "coordinates": written["coordinates"],
        "topology_pdb": topology_pdb,
        "ligand_sdf": str(ligand_sdf),
        "small_molecule_forcefield": (cfg.get("forcefield") or {}).get("ligand"),
        "charge_method": (cfg.get("forcefield") or {}).get("ligand_charge_method"),
        "forcefield_info": ff_info,
    }
