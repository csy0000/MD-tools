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

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import ConfigError, resolve_sys_config, sha256_of_document, write_yaml
from .forcefield_record import build_forcefield_record

OUTPUT_FILES = ("system.xml", "topology.pdb", "solute.pdb", "initial_state.xml", "solute.yaml",
                "forcefield.json", "resolved_sys.config.yaml", "provenance.yaml", "SHA256SUMS",
                "sys-gen.log")

#: Copied verbatim so the bundle can be rebuilt from what a user actually supplied.
ORIGINAL_INPUTS_DIR = "original_inputs"
#: Construction artifacts worth keeping: they carry bond orders, charges or radii that the
#: parameterisation depended on. Caches, environments and the whole `_work/` tree are not kept.
PREPARATION_DIR = "preparation"
PREPARATION_KEEP = ("tleap.in", "tleap.log", "solute.sdf", "ligand.sdf", "solute.mol2",
                    "ligand.mol2", "solute.frcmod", "ligand.frcmod")
PREPARATION_KEEP_SUFFIXES = (".prmtop", ".rst7", ".inpcrd")


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
    from .builder_defaults import DEFAULTS

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


SYSTEM_PROVENANCE_FORMAT = "md-templates-system-provenance/v1"
CHECKSUM_MANIFEST = "SHA256SUMS"


def _provenance(*, sys_config: dict, resolved: dict, out: Path, original_relative: str,
                original_sha256: str, system, payload: dict[str, str],
                command: list[str]) -> dict[str, Any]:
    """Everything needed to know what produced this bundle, and from what.

    Paths are relative to `inputs/`. An absolute path would say where the bundle happened to be
    written, which stops being true the moment it is moved -- and moving it is the point.
    """
    from .provenance_min import (environment_versions, implementation_identity, sha256_file)

    box = None
    if system.usesPeriodicBoundaryConditions():
        vectors = system.getDefaultPeriodicBoxVectors()
        box = [[float(v.x), float(v.y), float(v.z)] for v in vectors]

    return {
        "format": SYSTEM_PROVENANCE_FORMAT,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # An argument list, not a shell string: a string has to be re-parsed to be used, and
        # re-parsing is where quoting mistakes turn into a different command.
        "command": list(command),
        "implementation": implementation_identity(),
        "environment": environment_versions(),
        "original_input": {"path": original_relative, "sha256": original_sha256},
        "sys_config_hash": sha256_of_document(sys_config),
        "resolved_sys_config_hash": sha256_of_document(resolved),
        "forcefield_json_sha256": (sha256_file(out / "forcefield.json")
                                   if (out / "forcefield.json").is_file() else None),
        "system": {
            "topology_atoms": payload.get("topology_atoms"),
            "openmm_particles": payload.get("openmm_particles"),
            "solute_atoms": payload.get("solute_atoms"),
            "periodic": bool(system.usesPeriodicBoundaryConditions()),
            "box_vectors_nm": box,
        },
        "payload_paths": payload.get("paths"),
        "checksum_manifest": CHECKSUM_MANIFEST,
    }


def _copy_original_input(input_path: Path, out: Path) -> str:
    """The user's own file, byte-for-byte, under its own name.

    A collision is refused rather than resolved: silently renaming or overwriting would make the
    recorded checksum describe a file that is not the one the reader is looking at.
    """
    directory = out / ORIGINAL_INPUTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / input_path.name
    if destination.exists():
        from .provenance_min import sha256_file

        if sha256_file(destination) != sha256_file(input_path):
            raise ConfigError(
                f"{destination} already exists with different content. Refusing to overwrite the "
                f"recorded original input; write this system into a new output folder.")
    else:
        shutil.copy2(input_path, destination)
    return f"{ORIGINAL_INPUTS_DIR}/{destination.name}"


def _keep_preparation_artifacts(staging: Path, out: Path) -> list[str]:
    """Only the construction artifacts that carry science, at their staging-relative subpath.

    A prmtop carries the radii and charges the System was built from; `tleap.log` records what
    tleap did. A solvent scratch file or a cache carries neither, and copying the whole `_work/`
    tree would bury the ones that matter.

    The subpath is preserved rather than flattened to a basename. Two routes can both produce a
    `solute.sdf` in different staging directories, and collapsing them onto one name would either
    silently drop one or record a checksum for the wrong file. A genuine content collision at the
    same subpath is refused; an identical rerun is not a collision and is included again, so the
    recorded artifact list is complete whether or not this is the first run.
    """
    kept: list[str] = []
    if not staging.is_dir():
        return kept
    from .provenance_min import sha256_file

    destination = out / PREPARATION_DIR
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        if path.name not in PREPARATION_KEEP and path.suffix not in PREPARATION_KEEP_SUFFIXES:
            continue
        relative = path.relative_to(staging).as_posix()
        target = destination / relative
        if target.exists():
            if sha256_file(target) != sha256_file(path):
                raise ConfigError(
                    f"{PREPARATION_DIR}/{relative} already exists with different content. "
                    f"Refusing to overwrite a retained construction artifact; write this system "
                    f"into a new output folder.")
            # Identical: a rerun of the same build. Still recorded, so the list does not depend
            # on whether the directory happened to be fresh.
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        kept.append(f"{PREPARATION_DIR}/{relative}")
    return kept


def write_checksum_manifest(directory: Path, *, exclude=(CHECKSUM_MANIFEST,)) -> Path:
    """`SHA256SUMS` over every regular file here, written last.

    Deterministic: sorted relative POSIX paths, streamed contents, `sha256sum` format. The manifest
    cannot hash itself, and that exclusion is the only one.
    """
    from .provenance_min import sha256_file

    directory = Path(directory)
    skip = set(exclude)
    lines = []
    for path in sorted(directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in skip or any(part == "_work" for part in path.relative_to(directory).parts):
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    manifest = directory / CHECKSUM_MANIFEST
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def verify_checksum_manifest(directory: Path) -> dict[str, Any]:
    """Recompute every digest. Returns what matched, what changed and what went missing."""
    from .provenance_min import sha256_file

    directory = Path(directory)
    manifest = directory / CHECKSUM_MANIFEST
    if not manifest.is_file():
        return {"ok": False, "reason": f"{CHECKSUM_MANIFEST} is missing"}
    ok, mismatched, missing = [], [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) == digest:
            ok.append(relative)
        else:
            mismatched.append(relative)
    return {"ok": not mismatched and not missing, "verified": len(ok),
            "mismatched": mismatched, "missing": missing}


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
    _write_solute_pdb(record["topology_pdb"], solute_indices, out / "solute.pdb")
    log(f"solute.pdb   : {len(solute_indices)} atoms, the topology a solute-only DCD needs")

    write_yaml(out / "solute.yaml",
               _solute_document(topology, solute_indices, omega, route=route))
    write_yaml(out / "resolved_sys.config.yaml", resolved)

    # The user's own file, and the construction artifacts that carry science. Both before the
    # force-field record, which checksums whatever ligand representation survived.
    original_relative = _copy_original_input(input_path, out)
    kept = _keep_preparation_artifacts(staging, out)
    log(f"original     : {original_relative} (kept byte-for-byte)")
    if kept:
        log("preparation  : " + ", ".join(kept))

    artifacts = {}
    for relative in kept:
        if relative.endswith((".sdf", ".mol2")):
            artifacts.setdefault("ligand_sdf", relative)
        elif relative.endswith(".prmtop"):
            artifacts.setdefault("prmtop", relative)
        elif relative.endswith((".rst7", ".inpcrd")):
            artifacts.setdefault("coordinates", relative)
        elif relative.endswith("tleap.log"):
            artifacts.setdefault("tleap_log", relative)
    forcefield = build_forcefield_record(resolved=resolved, route=route, record=record,
                                         inputs_dir=out, artifacts=artifacts)
    (out / "forcefield.json").write_text(
        json.dumps(forcefield, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    log(f"forcefield   : {forcefield['protein']['openmm_resource']}"
        + (f" + {forcefield['water']['openmm_resource']}" if forcefield["water"]["openmm_resource"]
           else f" + {(forcefield['implicit_solvent'] or {}).get('model')}"
                f"/{(forcefield['implicit_solvent'] or {}).get('radii')}"))

    write_yaml(out / "provenance.yaml", _provenance(
        sys_config=document, resolved=resolved, out=out,
        original_relative=original_relative, original_sha256=sha256_file(input_path),
        system=system,
        payload={"topology_atoms": topology.getNumAtoms(),
                 "openmm_particles": system.getNumParticles(),
                 "solute_atoms": record["n_solute_atoms"],
                 "paths": {"system": "system.xml", "topology": "topology.pdb",
                           "solute_topology": "solute.pdb", "initial_state": "initial_state.xml",
                           "solute": "solute.yaml", "forcefield": "forcefield.json",
                           "resolved_config": "resolved_sys.config.yaml",
                           "original_input": original_relative,
                           "preparation": kept or None}},
        command=list(sys.argv)))

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
    # Last, so it covers the finished bundle including the log that describes writing it.
    manifest = write_checksum_manifest(out)
    n = len(manifest.read_text(encoding="utf-8").strip().splitlines())
    if echo:
        print(f"  checksums    : {n} files in {CHECKSUM_MANIFEST}")
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
        # `solute_pdb` / `solute_sdf` are the keys initial_structure actually returns; `pdb`/`sdf`
        # never existed, so the SMILES route raised KeyError before reaching parameterisation.
        source = Path(prepared["solute_pdb"])
        ligand_sdf = Path(prepared["solute_sdf"])
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
            "ligand_sdf": ligand_sdf, "omega": built,
            # Carried through for forcefield.json: what the box actually ended up containing is
            # part of how the system was parameterised, not a log line.
            "n_waters": solvated.get("n_waters"), "ions": solvated.get("ions"),
            "box_shape": solvated.get("box_shape"),
            "box_volume_nm3": solvated.get("box_volume_nm3")}


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
        # An explicit, recorded choice rather than a library default: including the ACE
        # surface-area term changes the energy by ~16 kJ/mol (~6 kT) on ACE-ALA-NME.
        nonpolar_sasa=bool((cfg.get("implicit_solvent") or {}).get("nonpolar_sasa", False)),
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
            "ligand_sdf": built.get("ligand_sdf"),
            "omega": built.get("build_record") or {},
            # The builder's own report of what it loaded: the tleap protein resource, the radii,
            # and on the ligand route the OpenFF report from build_forcefield. Surfaced here so
            # forcefield.json states the resources actually used rather than re-deriving them.
            "implicit_report": built.get("build") or {},
            "hmr": built.get("hmr") or {}}


def _write_solute_pdb(topology_pdb: Path, solute_indices, path: Path) -> Path:
    """The topology a solute-only trajectory needs.

    `DCDReporter(atomSubset=...)` writes only those atoms, so the frames cannot be read against the
    whole-system topology. Built from the SAME indices recorded in solute.yaml, so the subset
    trajectory and this file cannot disagree.
    """
    from openmm import app

    source = app.PDBFile(str(topology_pdb))
    keep = set(int(i) for i in solute_indices)
    modeller = app.Modeller(source.topology, source.positions)
    modeller.delete([a for a in source.topology.atoms() if a.index not in keep])
    with Path(path).open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    return Path(path)


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
