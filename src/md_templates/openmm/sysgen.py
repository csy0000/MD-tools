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

from . import md_data_contract as MD
from .config import (ConfigError, openff_resource, resolve_sys_config, sha256_of_document,
                     write_yaml)
from .defaults import DEFAULT_PADDING_NM, DEFAULT_SOLVENT
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
    cfg["forcefield"]["ligand"] = openff_resource(solute.get("ligand_forcefield"))
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
        cfg["solvation"]["water_model"] = str(solvent.get("model", DEFAULT_SOLVENT)).lower()
        cfg["solvation"]["box_shape"] = solvent.get("box_shape", "dodecahedron")
        cfg["solvation"]["padding_nm"] = solvent.get("padding_nm", DEFAULT_PADDING_NM)
        cfg["solvation"]["ionic_strength_molar"] = solvent.get("ionic_strength_molar", 0.15)
        cfg["solvation"]["positive_ion"] = solvent.get("positive_ion", "Na+")
        cfg["solvation"]["negative_ion"] = solvent.get("negative_ion", "Cl-")
    return cfg


#: Short water labels an older configuration may still carry, and the QUALIFIED OpenMM resource
#: each one has to become.
#:
#: OpenMM ships both spellings. The top-level `opc.xml` / `tip3p.xml` define water only, so a box
#: that needed Na+ and Cl- failed with "No template found for residue (NA)" -- after solvation had
#: already placed them. The `amber14/` and `amber19/` copies carry the ion templates alongside the
#: same water parameters.
#:
#: An explicit table rather than a blanket `amber19/` prefix: that prefix silently turned
#: `tip3p.xml` into `amber19/tip3p.xml`, which OpenMM does not ship. Written defaults are already
#: qualified, so this only rescues a hand-written or pre-0.4 file.
_QUALIFIED_WATER = {
    "opc.xml": "amber19/opc.xml",
    "opc3.xml": "amber19/opc3.xml",
    "tip3p.xml": "amber14/tip3p.xml",
    "tip3pfb.xml": "amber14/tip3pfb.xml",
}


def _water_xml(name: Optional[str]) -> Optional[str]:
    """The OpenMM water resource that will actually be loaded, or a refusal that names the file."""
    if not name:
        return None
    text = str(name).strip()
    if "/" in text:
        return text                                # already qualified; take it as written
    qualified = _QUALIFIED_WATER.get(text.lower())
    if qualified is None:
        raise ConfigError(
            f"forcefield.water = {text!r} is not a resource this package can qualify. OpenMM ships "
            f"the ion-carrying water files under a directory, so write the full resource name "
            f"(for example {', '.join(sorted(set(_QUALIFIED_WATER.values())))}). Known short "
            f"labels: {', '.join(sorted(_QUALIFIED_WATER))}.")
    return qualified


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
                command: list[str], forcefield: Optional[dict] = None,
                dataset: Optional[dict] = None) -> dict[str, Any]:
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
        # Whether this tree is a contract-managed MD-data dataset, and which validator said so.
        # An unregistered local tree says so explicitly rather than leaving it to be assumed.
        "dataset": dataset,
        "system": {
            "topology_atoms": payload.get("topology_atoms"),
            "openmm_particles": payload.get("openmm_particles"),
            "solute_atoms": payload.get("solute_atoms"),
            "periodic": bool(system.usesPeriodicBoundaryConditions()),
            "box_vectors_nm": box,
            "n_constraints": system.getNumConstraints(),
            "n_forces": system.getNumForces(),
            "force_classes": sorted({type(system.getForce(i)).__name__
                                     for i in range(system.getNumForces())}),
        },
        # A summary of `forcefield.json`, so a reader of this file alone can say which Hamiltonian
        # was built. The full record, including checksums and package versions, is that file --
        # `forcefield_json_sha256` above ties the two together.
        "forcefield_summary": _forcefield_summary(forcefield or {}),
        "payload_paths": payload.get("paths"),
        "checksum_manifest": CHECKSUM_MANIFEST,
    }


def _forcefield_summary(forcefield: dict[str, Any]) -> dict[str, Any]:
    """The parameterisation decisions, from `forcefield.json`, in one flat block.

    Every key is present and null where it does not apply, so "implicit solvent has no water model"
    reads as information rather than as a gap. Values are copied, never re-derived: re-deriving
    them here would be a second implementation that can disagree with the first.
    """
    protein = forcefield.get("protein") or {}
    ligand = forcefield.get("ligand") or {}
    water = forcefield.get("water") or {}
    explicit = forcefield.get("explicit_solvent") or {}
    implicit = forcefield.get("implicit_solvent") or {}
    nonbonded = forcefield.get("nonbonded") or {}
    constraints = forcefield.get("constraints") or {}
    geometry = explicit.get("box_geometry") or {}
    return {
        "solvation": forcefield.get("solvation"),
        "route": forcefield.get("route"),
        "protein_forcefield": protein.get("openmm_resource") or protein.get("tleap_resource"),
        "protein_forcefield_includes": protein.get("openmm_resource_includes"),
        "ligand_forcefield": ligand.get("openff_resource"),
        "ligand_charge_method": ligand.get("charge_method"),
        "ligand_charge_scheme": ligand.get("charge_model"),
        "water_model": water.get("model"),
        "water_forcefield": water.get("openmm_resource"),
        "implicit_model": implicit.get("model") if forcefield.get("implicit_solvent") else None,
        "implicit_radii": implicit.get("radii") if forcefield.get("implicit_solvent") else None,
        "implicit_nonpolar_sasa": (implicit.get("nonpolar_sasa")
                                   if forcefield.get("implicit_solvent") else None),
        "implicit_parameter_coverage": (implicit.get("parameter_coverage")
                                        if forcefield.get("implicit_solvent") else None),
        "nonbonded": dict(nonbonded),
        "box": {
            "shape": explicit.get("box_shape"),
            "volume_nm3": explicit.get("box_volume_nm3"),
            "vectors_nm": explicit.get("box_vectors_nm"),
            "padding_nm_requested": geometry.get("padding_nm_requested"),
            "solute_image_clearance_nm": geometry.get("solute_image_clearance_nm"),
            "min_reduced_box_height_nm": geometry.get("min_reduced_box_height_nm"),
            "required_cutoff_height_nm": geometry.get("required_cutoff_height_nm"),
            "grown_for_cutoff": geometry.get("grown_for_cutoff"),
        } if forcefield.get("explicit_solvent") else None,
        "salt": explicit.get("salt") if forcefield.get("explicit_solvent") else None,
        "constraints": dict(constraints),
    }


#: The header on a generated `dataset.yaml`, so a reader knows which repository owns the contract
#: and which one merely wrote the file.
DATASET_HEADER = """\
# MD-data dataset manifest, schema v1.
#
# The CONTRACT is owned by csy0000/MD-data (docs/contracts/dataset-v1.md). This file was written
# by MD-templates from the `dataset:` block of sys.config.yaml and validated with MD-data's own
# validator; MD-templates carries no copy of that schema.
#
# Paths are relative: `path` to $MD_DATA, each component to this dataset root. Marking this
# dataset complete or archived is a deliberate MD-data operation by its owner, not something any
# generation does.
"""


def _plan_dataset(resolved: dict[str, Any], out: Path) -> dict[str, Any]:
    """Everything the contract needs, resolved and validated BEFORE anything is built.

    Every refusal a misconfigured `dataset:` block can produce happens here: the two roots, the
    required fields, the identity/path agreement, and MD-data's own metadata validation. What is
    deliberately not done here is the on-disk layout check -- the component directories do not
    exist yet, because this generation is what creates them.
    """
    from .provenance_min import implementation_identity

    block = dict(resolved.get("dataset") or {})
    if not block.get("enabled"):
        # An ordinary local inputs/ + MD/ tree. Labelled, so nothing downstream can mistake it for
        # a registered dataset.
        return {"contract_managed": False,
                "note": ("unregistered local generation: no dataset.yaml, and NOT MD-data "
                         "compliant. Set dataset.enabled true in sys.config.yaml for a "
                         "contract-managed dataset."),
                "validator": MD.md_data_identity()}

    MD.require_md_data()
    # The claimed templates.commit must be the commit that is actually generating this. Checked
    # here, before the System is built, because a provenance that cannot be verified is a reason
    # not to start rather than something to discover after an hour of solvation.
    established = MD.check_templates_commit((block.get("templates") or {}).get("commit"))
    location = MD.resolve_roots(out, component=MD.COMMON_COMPONENT)
    manifest = MD.build_manifest(
        dataset_block=block, location=location,
        components=[MD.component_entry(
            MD.COMMON_COMPONENT, kind="shared-input",
            description="Prepared system, force-field record and initial state, shared by every "
                        "method component of this dataset.")],
        templates_version=implementation_identity()["version"])
    MD.validate(manifest)                       # metadata only: the tree does not exist yet
    return {"contract_managed": True, "manifest": manifest, "location": location,
            "generator": established}


def _write_dataset_manifest(plan: dict[str, Any], *, echo: bool = True) -> dict[str, Any]:
    """Write `dataset.yaml`, then validate it again WITH the root now that the tree exists."""
    if not plan["contract_managed"]:
        return {"contract_managed": False, "note": plan["note"], "validator": plan["validator"]}

    manifest = dict(plan["manifest"])
    location = plan["location"]
    root = Path(location["dataset_root"])
    existing = root / MD.MANIFEST_NAME
    if existing.is_file():
        previous = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        manifest["components"] = MD.merge_components(previous.get("components") or [],
                                                     manifest["components"])
    path = MD.write_manifest(root, manifest, header=DATASET_HEADER)
    report = MD.validate(manifest, root=location["md_data"],
                         dataset_root=location["dataset_root"])
    if echo:
        print(f"  dataset      : {report['dataset_id']} ({report['role']}, {report['status']}) "
              f"at {report['path']}")
        print(f"  manifest     : {path.name} validated by md-data "
              f"{report['validator']['version']} (contract v"
              f"{report['validator']['contract_version']})")
    established = plan.get("generator") or {}
    if echo and established.get("dirty"):
        print(f"  WARNING      : templates.commit {established['commit'][:12]} is this "
              f"checkout's HEAD, but the working tree has uncommitted changes, so the recorded "
              f"pin does not fully describe what ran. provenance.yaml records git_dirty: true.")
    # `md_data` / `md_data_local` are deliberately absent: they are this machine's storage
    # location, and the record must survive the tree being moved.
    return {"contract_managed": True, "manifest": MD.MANIFEST_NAME, **report}


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

    # The contract is checked BEFORE anything is built. A misconfigured dataset should cost a
    # second, not a solvated system and a full parameterisation.
    dataset_plan = _plan_dataset(resolved, out)
    if dataset_plan["contract_managed"]:
        log(f"dataset      : {dataset_plan['manifest']['dataset_id']} "
            f"({dataset_plan['manifest']['role']}) at {dataset_plan['location']['path']}")

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
                                         inputs_dir=out, artifacts=artifacts, builder=cfg)
    (out / "forcefield.json").write_text(
        json.dumps(forcefield, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    log(f"forcefield   : {forcefield['protein']['openmm_resource']}"
        + (f" + {forcefield['water']['openmm_resource']}" if forcefield["water"]["openmm_resource"]
           else f" + {(forcefield['implicit_solvent'] or {}).get('model')}"
                f"/{(forcefield['implicit_solvent'] or {}).get('radii')}"))

    dataset = _write_dataset_manifest(dataset_plan, echo=echo)
    write_yaml(out / "provenance.yaml", _provenance(
        sys_config=document, resolved=resolved, out=out, dataset=dataset,
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
        command=list(sys.argv), forcefield=forcefield))

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
            "n_solute_atoms": record["n_solute_atoms"], "implicit": implicit,
            "dataset": dataset}


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

    geometry = solvated.get("geometry") or {}
    log(f"box geometry : requested {geometry.get('padding_nm_requested')} nm padding -> "
        f"solute-to-copy clearance {geometry.get('solute_image_clearance_nm')} nm, "
        f"reduced-box height {geometry.get('min_reduced_box_height_nm')} nm "
        f"(needs {geometry.get('required_cutoff_height_nm')} nm for a "
        f"{geometry.get('nonbonded_cutoff_nm')} nm cutoff"
        + (", box grown to fit" if geometry.get("grown_for_cutoff") else "") + ")")

    built = build_system(Path(solvated["output_pdb"]), staging, cfg, solvated["n_solute_atoms"],
                         ligand_sdf=ligand_sdf, route=route)
    nonbonded = built.get("nonbonded") or {}
    log(f"system       : {nonbonded.get('method')}, cutoff {nonbonded.get('cutoff_nm')} nm, "
        f"switching {nonbonded.get('switching')}, dispersion correction "
        f"{nonbonded.get('dispersion_correction')}, Ewald tolerance "
        f"{nonbonded.get('ewald_error_tolerance')}, {cfg['system_build']['constraints']}")

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
            "box_volume_nm3": solvated.get("box_volume_nm3"),
            "box_vectors_nm": solvated.get("box_vectors_nm"),
            # The full geometry resolution: requested padding, the three distances that are not
            # interchangeable, and whether the box had to be grown for the cutoff. Recorded rather
            # than reduced to one number, because "1.5 nm padding" alone does not say what
            # clearance the built box actually has.
            "box_geometry": solvated.get("geometry"),
            "salt": solvated.get("salt"),
            "water_model": solvated.get("water_model"),
            "water_packing_model": solvated.get("water_packing_model"),
            "water_packing_substituted": solvated.get("water_packing_substituted"),
            "water_model_reconciled": solvated.get("water_model_reconciled")}


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
