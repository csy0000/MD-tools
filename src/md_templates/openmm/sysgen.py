"""Build an OpenMM System from a structure and `sys.config.yaml`.

The chemistry is the validated implementation that already exists -- protonation, the dodecahedral
box, PME construction, the ParmEd GBn2/mbondi3 path, omega classification. What is new here is that
it is driven directly by the user's YAML and writes seven small files, instead of a bundle with a
schema version, a checksum manifest and a vendored copy of this package.

`solute.yaml` is where the classification work is recorded ONCE, at build time: solute indices, the
default REST2 enhanced region, and the omega-excluded bonds. The generated run scripts read it and
never import this package.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import ConfigError, resolve_sys_config, sha256_of_document, write_yaml

OUTPUT_FILES = ("system.xml", "topology.pdb", "initial_state.xml", "solute.yaml",
                "resolved_sys.config.yaml", "provenance.yaml", "sys-gen.log")


class Log:
    """A readable preparation log, echoed to stdout as it is written."""

    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.lines: list[str] = []
        self.path = Path(path)
        self.echo = echo

    def __call__(self, message: str) -> None:
        self.lines.append(message)
        if self.echo:
            print(f"  {message}")

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return self.path


def _legacy_cfg(resolved: dict[str, Any]) -> dict[str, Any]:
    """Map the user's YAML onto the dict shape the existing builders take.

    A translation layer, deliberately in one function and one direction. The builders were written
    against a large nested configuration; rewriting their signatures would risk the chemistry for a
    cosmetic gain, so the mapping is explicit and the science is untouched.
    """
    from .config_legacy import DEFAULTS

    cfg = yaml.safe_load(yaml.safe_dump(DEFAULTS))
    solute = resolved.get("solute") or {}
    forcefield = resolved.get("forcefield") or {}
    constraints = resolved.get("constraints") or {}
    implicit = resolved["solvation"] == "implicit"

    cfg["forcefield"]["protein"] = forcefield.get("protein")
    cfg["forcefield"]["water"] = _water_xml(forcefield.get("water"))
    cfg["forcefield"]["ligand"] = _openff_name(solute.get("ligand_forcefield"))
    cfg["forcefield"]["ligand_charge_method"] = solute.get("ligand_charge_method")

    cfg["system_build"]["constraints"] = constraints.get("type", "HBonds")
    cfg["system_build"]["rigid_water"] = bool(constraints.get("rigid_water", not implicit))
    cfg["system_build"]["hydrogen_mass_amu"] = constraints.get("hydrogen_mass_amu")
    cfg["system_build"]["hmr_scope"] = (
        "solute" if constraints.get("hydrogen_mass_amu") is not None else "none")

    if implicit:
        cfg["system_build"]["nonbonded_method"] = "NoCutoff"
        cfg["system_build"]["nonbonded_cutoff_nm"] = None
        cfg["system_build"]["ewald_error_tolerance"] = None
        cfg["system_build"]["use_dispersion_correction"] = False
        cfg["system_build"]["switch_distance_nm"] = None
        cfg["system_build"]["minimum_image_margin_nm"] = None
    else:
        solvent = resolved.get("solvent") or {}
        cfg["system_build"]["nonbonded_cutoff_nm"] = solvent.get("cutoff_nm", 1.0)
        cfg["solvation"]["water_model"] = str(solvent.get("model", "OPC")).lower()
        cfg["solvation"]["box_shape"] = solvent.get("box_shape", "dodecahedron")
        cfg["solvation"]["padding_nm"] = solvent.get("padding_nm", 2.0)
        cfg["solvation"]["ionic_strength_molar"] = solvent.get("ionic_strength_molar", 0.15)
        cfg["solvation"]["positive_ion"] = solvent.get("positive_ion", "Na+")
        cfg["solvation"]["negative_ion"] = solvent.get("negative_ion", "Cl-")
    return cfg


def _water_xml(name: Optional[str]) -> Optional[str]:
    """`opc.xml` is what a user writes; `amber19/opc.xml` is the file that also defines the ions.

    OpenMM ships both. The top-level `opc.xml` has water only, so a box that needed Na+ and Cl-
    failed with "No template found for residue (NA)" -- after solvation had already placed them.
    The amber19 copy carries the ion templates alongside the same water parameters.
    """
    if not name:
        return None
    text = str(name).strip()
    if "/" in text:
        return text                                # already qualified; take it as written
    return f"amber19/{text}"


def _openff_name(name: Optional[str]) -> Optional[str]:
    """`sage-2.2.0` is what a user writes; `openff-2.2.0` is what the toolkit loads."""
    if not name:
        return None
    text = str(name).strip().lower()
    if text.startswith("sage-"):
        return "openff-" + text[len("sage-"):]
    return text


def _solute_document(topology, solute_indices, omega, *, route: str) -> dict[str, Any]:
    residues = []
    solute_set = set(int(i) for i in solute_indices)
    for residue in topology.residues():
        indices = [a.index for a in residue.atoms()]
        if any(i in solute_set for i in indices):
            residues.append({"index": residue.index, "name": residue.name,
                             "atoms": [int(min(indices)), int(max(indices))]})
    return {
        "schema_version": 1,
        "route": route,
        "n_solute_atoms": len(solute_set),
        # A contiguous range in every path this repository builds; recorded as first/last so the
        # generated scripts need no list of thousands of integers.
        "solute_atom_range": [int(min(solute_set)), int(max(solute_set))] if solute_set else None,
        "solute_atom_indices_are_contiguous": (
            bool(solute_set) and max(solute_set) - min(solute_set) + 1 == len(solute_set)),
        "residues": residues,
        "rest2": {
            "enhanced_region": "solute",
            # Bonds whose torsion must NOT be scaled: scaling an omega torsion lets a peptide bond
            # rotate at the hot rungs and the ladder samples cis/trans interconversion that the
            # cold rung never sees.
            "omega_excluded_bonds": [[int(a), int(b)] for a, b in omega.get(
                "omega_unscaled_bonds", [])],
            "omega_proline_like_scaled_bonds": [[int(a), int(b)] for a, b in omega.get(
                "omega_proline_like_scaled_bonds", [])],
            "omega_detection_method": omega.get("omega_detection_method"),
        },
    }


def _provenance(sys_config: dict, inputs: dict[str, str]) -> dict[str, Any]:
    from .provenance_min import package_provenance

    return {
        **package_provenance(),
        "generated": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_hashes": inputs,
            "sys_config_hash": sha256_of_document(sys_config),
        },
    }


def generate_system(*, input_path: Path, config_path: Path, output_folder: Path,
                    echo: bool = True) -> dict[str, Any]:
    """Prepare a System and write the seven output files."""
    from openmm import XmlSerializer, app

    from .provenance_min import sha256_file

    input_path = Path(input_path).resolve()
    config_path = Path(config_path).resolve()
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "sys-gen.log", echo=echo)

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved = resolve_sys_config(document)
    implicit = resolved["solvation"] == "implicit"
    peptide = bool((resolved.get("solute") or {}).get("peptide", True))
    route = "peptide" if peptide else "ligand"

    log(f"input        : {input_path.name}")
    log(f"solvation    : {resolved['solvation']}"
        + (f" ({(resolved.get('implicit_solvent') or {}).get('model')}"
           f"/{(resolved.get('implicit_solvent') or {}).get('radii')})" if implicit
           else f" ({(resolved.get('solvent') or {}).get('model')})"))
    log(f"route        : {route}")

    cfg = _legacy_cfg(resolved)
    staging = out / "_work"
    staging.mkdir(parents=True, exist_ok=True)

    if implicit:
        record = _build_implicit(input_path, cfg, staging, route=route, log=log)
    else:
        record = _build_explicit(input_path, cfg, staging, route=route, log=log)

    topology = app.PDBFile(str(record["topology_pdb"])).topology
    system = XmlSerializer.deserialize(Path(record["system_xml"]).read_text())

    solute_indices = list(range(record["n_solute_atoms"]))
    # Both builders already classify the omega bonds while they have the topology and the ligand
    # bond orders in hand. Recomputing here would be a second implementation of the same decision.
    omega = record.get("omega") or {}
    if "omega_unscaled_bonds" not in omega:
        from .system import classify_omega_bonds
        omega = classify_omega_bonds(topology, solute_indices, route=route,
                                     ligand_sdf=record.get("ligand_sdf"))
    log(f"solute atoms : {record['n_solute_atoms']} of {system.getNumParticles()} particles")
    log(f"omega bonds  : {len(omega.get('omega_unscaled_bonds', []))} excluded from scaling "
        f"({omega.get('omega_detection_method')})")

    shutil.copy2(record["system_xml"], out / "system.xml")
    shutil.copy2(record["topology_pdb"], out / "topology.pdb")
    shutil.copy2(record["initial_state"], out / "initial_state.xml")

    write_yaml(out / "solute.yaml",
               _solute_document(topology, solute_indices, omega, route=route))
    write_yaml(out / "resolved_sys.config.yaml", resolved)
    write_yaml(out / "provenance.yaml",
               _provenance(document, {input_path.name: sha256_file(input_path)}))

    if system.usesPeriodicBoundaryConditions():
        vectors = system.getDefaultPeriodicBoxVectors()
        log("box vectors  : " + "; ".join(
            f"({v.x:.3f}, {v.y:.3f}, {v.z:.3f})" for v in vectors))
    else:
        log("box          : none (implicit solvent)")
    log(f"constraints  : {system.getNumConstraints()}")
    log("written      : " + ", ".join(OUTPUT_FILES))
    log.save()
    shutil.rmtree(staging, ignore_errors=True)
    return {"output_folder": str(out), "n_particles": system.getNumParticles(),
            "n_solute_atoms": record["n_solute_atoms"], "implicit": implicit}


def _build_explicit(input_path: Path, cfg: dict, staging: Path, *, route: str, log: Log) -> dict:
    """Protonate, solvate in the dodecahedral box, build the System."""
    from openmm import XmlSerializer, app

    from .solvation import solvate
    from .system import build_system, initial_structure

    ligand_sdf = None
    if route == "ligand":
        prepared = initial_structure(_smiles_from(input_path), staging, cfg)
        source = Path(prepared["pdb"])
        ligand_sdf = Path(prepared["sdf"])
        log(f"ligand       : {prepared.get('canonical_smiles', '')[:60]}")
    else:
        source = staging / "input.pdb"
        shutil.copy2(input_path, source)

    from .system import protonate

    protonated = protonate(source, staging, cfg, ligand_sdf=ligand_sdf,
                           input_route=("smiles" if route == "ligand" else "pdb"))
    log(f"protonation  : pH {protonated.get('ph')}, "
        f"{protonated.get('n_hydrogens_before')} -> {protonated.get('n_hydrogens_after')} hydrogens")

    solvated = solvate(Path(protonated["output_pdb"]), staging, cfg,
                       ligand_sdf=ligand_sdf, route=route)
    ions = solvated.get("ions") or {}
    log(f"solvation    : {solvated['n_waters']} waters, "
        f"ions {ions.get('counts', ions)}, box {solvated.get('box_shape')} "
        f"({solvated.get('box_volume_nm3', 0):.1f} nm^3)")

    built = build_system(Path(solvated["output_pdb"]), staging, cfg, solvated["n_solute_atoms"],
                         ligand_sdf=ligand_sdf, route=route)
    log(f"system       : PME, cutoff {cfg['system_build']['nonbonded_cutoff_nm']} nm, "
        f"{cfg['system_build']['constraints']}")

    pdb = app.PDBFile(str(solvated["output_pdb"]))
    system = XmlSerializer.deserialize(Path(built["system_xml"]).read_text())
    state_path = _write_initial_state(system, pdb, staging)
    return {"system_xml": built["system_xml"], "topology_pdb": solvated["output_pdb"],
            "initial_state": state_path, "n_solute_atoms": solvated["n_solute_atoms"],
            "ligand_sdf": ligand_sdf, "omega": built}


def _build_implicit(input_path: Path, cfg: dict, staging: Path, *, route: str, log: Log) -> dict:
    """The validated ParmEd GBn2 + mbondi3 path."""
    from openmm import XmlSerializer, app

    from .implicit import build_implicit_bundle_inputs

    smiles = _smiles_from(input_path) if route == "ligand" else None
    pdb_input = input_path if route == "peptide" else None
    built = build_implicit_bundle_inputs(
        route=("peptide" if route == "peptide" else "ligand"),
        cfg=cfg, staging=staging, pdb=pdb_input, smiles=smiles,
        implicit_model="GBn2", radii="mbondi3",
        hydrogen_mass_amu=cfg["system_build"].get("hydrogen_mass_amu"),
        hmr_scope=str(cfg["system_build"].get("hmr_scope") or "none"))
    log("system       : GBn2 / mbondi3 via ParmEd.Structure.createSystem (NOT AmberPrmtopFile: "
        "the two differ by ~16 kJ/mol in CustomGBForce)")
    hmr = built.get("hmr") or {}
    if hmr.get("scope") != "none":
        log(f"hmr          : {hmr['n_hydrogens_repartitioned']} hydrogens to "
            f"{hmr['target_hydrogen_mass_amu']} amu, total mass conserved")

    pdb = app.PDBFile(str(built["topology_pdb"]))
    system = XmlSerializer.deserialize(Path(built["system_xml"]).read_text())
    state_path = _write_initial_state(system, pdb, staging)
    return {"system_xml": built["system_xml"], "topology_pdb": built["topology_pdb"],
            "initial_state": state_path, "n_solute_atoms": built["n_solute_atoms"],
            "ligand_sdf": None, "omega": built.get("build_record") or {}}


def _write_initial_state(system, pdb, staging: Path) -> Path:
    """Positions and box vectors, with no velocities: nothing has been integrated yet."""
    from openmm import Context, VerletIntegrator, XmlSerializer, unit

    context = Context(system, VerletIntegrator(0.001 * unit.picoseconds))
    context.setPositions(pdb.positions)
    if system.usesPeriodicBoundaryConditions():
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    state = context.getState(getPositions=True)
    path = staging / "initial_state.xml"
    path.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    del context
    return path


def _smiles_from(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"{path} is empty; a non-peptide input must contain a SMILES string")
    return text.splitlines()[0].split()[0]
