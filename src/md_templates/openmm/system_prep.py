"""Molecular preparation, ending before any dynamics.

This is the implementation behind ``MD_system_gen.py``. It reuses the package's existing
construction path (``equilibration.build_simbox``, which despite its module name performs no
dynamics) and stops there, then writes the portable system bundle that ``MD_input_gen.py`` consumes.

What "stops there" means precisely: the bundle contains an **initial** State -- positions and
periodic box vectors as built, with no velocities, because nothing has been integrated. A bundle
from this module has never seen a minimiser.

The bundle is deliberately more than ``System.xml``. A serialized OpenMM System carries parameters
but no atom or residue names, so it is not a topology and cannot be read back into something a human
or a downstream tool can index. The bundle therefore always carries a topology-bearing structure
alongside it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["prepare_system", "SYSTEM_MANIFEST_SCHEMA_VERSION"]

#: Bumped when the manifest's shape changes. `MD_input_gen.py` validates it on read rather than
#: assuming, so an old bundle fails loudly instead of being silently misread.
SYSTEM_MANIFEST_SCHEMA_VERSION = 1

#: Files every system bundle must contain. The list is checked after writing, so a bundle that is
#: missing a piece is caught here rather than by a run three stages later.
REQUIRED_BUNDLE_FILES = (
    "system.xml",
    "topology.pdb",
    "topology.cif",
    "initial_state.xml",
    "forcefield.json",
    "system_manifest.json",
    "system.yaml",
    "checksums.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Atomic write: temp file in the destination filesystem, fsync, replace."""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _runtime_cfg_from_system_config(config: dict, input_path: Path, input_format: str,
                                    system_type: str) -> tuple[dict, Optional[str], Optional[Path]]:
    """Project system_config.json onto the package's runtime configuration tree.

    system_config.json owns chemistry and system-building choices ONLY. Anything protocol-shaped
    (minimisation, equilibration, production, reporting) is rejected here rather than silently
    ignored, because a user who wrote it there believes it took effect.
    """
    from .config import DEFAULTS
    import copy

    protocol_keys = {"minimization", "equilibration", "production", "protocol", "reporting",
                     "integrator", "execution"}
    intruders = sorted(protocol_keys & set(config))
    if intruders:
        raise ValueError(
            f"system_config.json contains protocol settings: {', '.join(intruders)}. "
            "Chemistry and system building live here; minimisation, equilibration, production, "
            "reporting and execution belong in md_config.json and are consumed by "
            "MD_input_gen.py. Leaving them here would mean they are never applied."
        )

    cfg = copy.deepcopy(DEFAULTS)

    # Conformer generation needs an explicit seed. The package DEFAULTS leave it None because the
    # canonical pipeline fills it from the randomness block; this front end has no such block, so it
    # derives one. Without a seed the SMILES route dies on int(None) inside ETKDG -- a failure the
    # dry-run path cannot reach, since it never builds anything.
    #
    # Derived rather than hard-coded: the conformer seed decides which starting structure the whole
    # bundle is built around, so it belongs to the run's seed map like every other stream.
    from .seeds import DEFAULT_MASTER_SEED, derive_seed

    randomness = config.get("randomness") or {}
    master = int(randomness.get("master_seed", DEFAULT_MASTER_SEED))
    explicit = randomness.get("structure_seed")
    seed = int(explicit) if explicit is not None else derive_seed(master, "structure/conformer")
    cfg["structure"]["etkdg"]["seed"] = seed
    cfg["run"]["seed"] = master
    for stage in ("equilibration",):
        if stage in cfg and isinstance(cfg[stage], dict):
            cfg[stage].setdefault("seed", seed)

    system_block = config.get("system") or {}
    cfg["system"]["slug"] = system_block.get("id") or input_path.stem.lower().replace("-", "_")
    cfg["system"]["solute_kind"] = "ligand" if system_type == "ligand" else "peptide"

    for section in ("forcefield", "solvation", "system_build"):
        if section in config:
            cfg[section].update(config[section])

    if input_format == "smi":
        check_ligand_build_matches_the_route(config, cfg)

    smiles = None
    pdb: Optional[Path] = None
    if input_format == "smi":
        text = input_path.read_text().strip().splitlines()
        if not text:
            raise ValueError(f"{input_path} is empty")
        smiles = text[0].split()[0]
    else:
        pdb = input_path

    return cfg, smiles, pdb


def check_ligand_build_matches_the_route(config: dict, cfg: dict) -> None:
    """The declared chemistry must be the chemistry that will actually be used.

    `ligand_build` is copied into the bundle manifest as a record of how the molecule was built. If
    it names a charge model or parameterisation route that the resolved force field does not use,
    that record is false -- and it is exactly the kind of falsehood nobody notices, because both
    values look plausible in isolation. Checked here rather than in the entry point so that callers
    which never touch the command line are covered too.
    """
    declared = config.get("ligand_build") or {}
    ff = cfg.get("forcefield") or {}
    pairs = (
        ("parameterization_route", ff.get("ligand"), "forcefield.ligand"),
        ("charge_model", ff.get("ligand_charge_method"), "forcefield.ligand_charge_method"),
    )
    problems = [
        f"    ligand_build.{field}={declared[field]!r} but the resolved {source}={resolved!r}"
        for field, resolved, source in pairs
        if field in declared and resolved is not None and declared[field] != resolved
    ]
    if problems:
        raise ValueError(
            "the declared ligand_build does not describe the parameterisation that would run:\n"
            + "\n".join(problems)
            + "\n  Change the declaration, or state the force field you meant under `forcefield`. "
              "This block is\n  persisted as provenance, so a mismatch would record chemistry that "
              "never happened."
        )


def prepare_system(*, input_path: Path, input_format: str, system_type: str, config: dict,
                   outdir: Path, overwrite: bool = False) -> dict:
    """Build a portable system bundle. Runs no dynamics.

    Transactional: everything is written into a temporary directory in the same filesystem and
    moved into place only once the required-file check passes, so an interrupted run leaves either
    the previous bundle or nothing -- never a half-written one whose checksums describe a mixture.
    """
    from .destination import SYSTEM_BUNDLE_TARGETS, check_destination, publish
    from .equilibration import build_simbox

    outdir = Path(outdir)
    # Checked BEFORE any work: parameterising a ligand can cost half an hour, and discovering the
    # destination was occupied only at the publish step would throw all of it away.
    check_destination(outdir, SYSTEM_BUNDLE_TARGETS, overwrite=overwrite, what="system bundle")
    staging = outdir.parent / f".{outdir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        cfg, smiles, pdb = _runtime_cfg_from_system_config(
            config, input_path, input_format, system_type)

        # The construction path, reused unchanged. It solvates, ionises and parameterises; it
        # integrates nothing.
        info = build_simbox(cfg, staging, "system", smiles=smiles, pdb=pdb)

        system_xml = staging / "system.xml"
        topology_pdb = staging / "topology.pdb"
        Path(info["system_xml"]).replace(system_xml)
        Path(info["topology_pdb"]).replace(topology_pdb)

        # An initial State: positions and box vectors as built, no velocities, nothing integrated.
        initial_state = staging / "initial_state.xml"
        _write_initial_state(system_xml, topology_pdb, initial_state)
        _write_topology_cif(topology_pdb, staging / "topology.cif")

        forcefield = {
            "route": info.get("route"),
            "input_route": info.get("input_route"),
            "system_type": system_type,
            **(info.get("forcefield") or {}),
            "nonbonded": info.get("nonbonded"),
            "hmr": info.get("hmr"),
            "constraints_note": (
                "constraints and hydrogen mass are properties of the built System and are recorded "
                "here; changing either requires a new system bundle, not a new protocol."
            ),
        }
        _write_json(staging / "forcefield.json", forcefield)

        # copy the exact molecular input so the bundle is self-contained and relocatable
        original = staging / "original_inputs"
        original.mkdir(exist_ok=True)
        shutil.copy2(input_path, original / input_path.name)

        manifest = {
            "schema_version": SYSTEM_MANIFEST_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "MD_system_gen.py",
            "prepared_through": "minimisation NOT run; this bundle has never been integrated",
            "system": {
                "id": cfg["system"]["slug"],
                "type": system_type,
                "input_file": input_path.name,
                "input_format": input_format,
                "input_sha256": _sha256(input_path),
                "smiles": smiles,
            },
            "composition": {
                "n_solute_atoms": info["n_solute_atoms"],
                "n_particles": info["n_particles"],
                "n_waters": info.get("n_waters"),
                "ions": info.get("ions"),
                "degrees_of_freedom": info.get("degrees_of_freedom"),
                "n_constraints": info.get("n_constraints"),
            },
            # Neutralising counterions and added salt pairs are DIFFERENT quantities and are
            # reported separately; collapsing them misstates the ionic strength.
            "salt": info.get("salt"),
            "geometry": info.get("geometry"),
            "water": info.get("water"),
            "omega": {
                "central_bonds": info.get("omega_central_bonds"),
                "detection_method": info.get("omega_detection_method"),
            },
            "files": {name: name for name in REQUIRED_BUNDLE_FILES},
            "resolved_system_config": config,
            "outputs_are_amber_or_gromacs": False,
            "adapter_status": {
                "openmm": "implemented",
                "amber": "not implemented -- would emit prmtop/rst7",
                "gromacs": "not implemented -- would emit top/gro; a tpr is stage-specific and "
                           "belongs to MD input generation, not molecular preparation",
            },
        }
        _write_json(staging / "system_manifest.json", manifest)

        # The package's legacy system manifest, written from the SAME chemistry. The runner reads
        # it, so emitting it here is what lets a generated project be handed to the existing REST2
        # runner instead of teaching that runner a second bundle format. It carries chemistry only
        # -- no protocol -- so it stays inside this generator's remit.
        _write_system_yaml(staging / "system.yaml", config, manifest, info)

        checksums = {p.name: _sha256(p) for p in sorted(staging.iterdir())
                     if p.is_file() and p.name != "checksums.json"}
        _write_json(staging / "checksums.json", {
            "algorithm": "sha256",
            "note": "covers every bundle file except this one, which cannot contain its own hash",
            "files": checksums,
        })

        missing = [f for f in REQUIRED_BUNDLE_FILES if not (staging / f).is_file()]
        if missing:
            raise RuntimeError(
                f"system bundle is incomplete, refusing to publish it: missing {missing}"
            )

        publish(staging, outdir, overwrite=overwrite, what="system bundle")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "bundle_dir": str(outdir),
        "system_manifest": str(outdir / "system_manifest.json"),
        "n_solute_atoms": info["n_solute_atoms"],
        "n_particles": info["n_particles"],
    }


def _write_initial_state(system_xml: Path, topology_pdb: Path, out_state: Path) -> None:
    """Serialize positions and box vectors as an OpenMM State. No velocities: nothing has moved."""
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile
    import openmm

    system = XmlSerializer.deserialize(system_xml.read_text())
    pdb = PDBFile(str(topology_pdb))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)   # required to make a Context
    platform = openmm.Platform.getPlatformByName("Reference")      # never touches a GPU
    context = openmm.Context(system, integrator, platform)
    context.setPositions(pdb.positions)
    if pdb.topology.getPeriodicBoxVectors() is not None:
        context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    state = context.getState(getPositions=True)
    out_state.write_text(XmlSerializer.serialize(state))
    del context, integrator


def _write_topology_cif(topology_pdb: Path, out_cif: Path) -> None:
    from openmm.app import PDBFile, PDBxFile

    pdb = PDBFile(str(topology_pdb))
    with out_cif.open("w") as handle:
        PDBxFile.writeFile(pdb.topology, pdb.positions, handle, keepIds=True)


def _write_system_yaml(path: Path, config: dict, manifest: dict, info: dict) -> None:
    """Emit the package's legacy system manifest from the prepared chemistry.

    Chemistry only. Protocol belongs to md_config.json and is never written here.
    """
    import yaml

    system = manifest["system"]
    ff = config.get("forcefield") or {}
    solv = config.get("solvation") or {}
    doc = {
        "schema_version": 1,
        "system_id": system["id"],
        "display_name": (config.get("system") or {}).get("display_name", system["id"]),
        "input": {
            "route": "smiles" if system["input_format"] == "smi" else "pdb",
            "expected_formal_charge": (config.get("ligand_build") or {}).get("formal_charge", 0),
        },
        "parameterization": {
            "small_molecule_forcefield": ff.get("ligand"),
            "charge_method": ff.get("ligand_charge_method"),
            "protein_forcefield": ff.get("protein"),
            "water_forcefield": ff.get("water"),
        },
        "solvation": {
            "water_model": solv.get("water_model"),
            "box_shape": solv.get("box_shape"),
            "padding_nm": solv.get("padding_nm"),
            "ionic_strength_molar": solv.get("ionic_strength_molar"),
            "positive_ion": solv.get("positive_ion", "Na+"),
            "negative_ion": solv.get("negative_ion", "Cl-"),
        },
    }
    if doc["input"]["route"] == "smiles":
        doc["input"]["smiles"] = system.get("smiles")
        doc["input"]["canonical_isomeric_smiles"] = system.get("smiles")
    else:
        doc["input"]["pdb"] = system["input_file"]
        doc["input"]["pdb_sha256"] = system["input_sha256"]
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
