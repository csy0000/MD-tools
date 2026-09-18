"""`md-openmm build-top --parameterize` -- one small molecule to one reusable parameter package.

The other mode of `build-top` builds a SYSTEM: a solute in a box, ready to integrate. This one
builds PARAMETERS: the molecule alone, charged once, written as a package that any later build can
reuse and that `data-register --ligand-package` can register. It is a flag on `build-top` rather
than a new command, for the same reason `--rest2-scaler` is: the package has four work commands.

What it writes into one directory:

```text
<dir>/  molecule.sdf       the package: the exact chemical state, in package atom order
        parameters.ffxml   the package: self-contained OpenMM parameters
        metadata.json      the package: identities, charges, provenance, digests
        parameter.config   the package: what a build must match to reuse these parameters
        TYL.sdf            a readable copy of the molecule, named for the residue
        TYL.pdb            the molecule as a topology (`-op`)
        TYL.xml            the molecule alone as a serialised System (`-os`)
```

The first four ARE a package, so the directory can be registered and found by a catalog search.
The last three are what a person, a tutorial and a later `build-top` command line point at; they
are declared in the metadata with their digests, and a package whose copies were modified is
refused, so "a package holds exactly these files" stays a check rather than a comment.

`TYL.xml` is the molecule ALONE: no solvent, no box, no cutoff and no constraints. It is a
parameters artefact for reading and comparing, not a system to integrate -- a run's Hamiltonian
comes from a `build-top` build that loads this package.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..ligands.package import (CRITERIA_NAME, FFXML_NAME, METADATA_NAME, MOLECULE_NAME,
                               PackageError, READABLE_COPIES_KEY, load_package)
from .record import LogWriter, file_facts, sha256_file
from .strict import ConfigError

__all__ = ["PARAMETERIZE_SUFFIXES", "parameterize_ligand", "read_molecule"]

#: What `--parameterize -i` reads. `.mol2` is accepted here and nowhere else: it carries a
#: molecular graph with bond orders, which is what parameterisation needs, and it is what several
#: docking and preparation tools write. `.smi` states the chemistry and no coordinates, so a
#: conformer is EMBEDDED exactly as `build-top` embeds one (ETKDGv3 from the run seed, then MMFF):
#: the package needs a conformer for its molecule.sdf and for the readable topology.
#:
#: The conformer is NOT part of a package's identity -- that is the chemical-state digest plus the
#: parameter digest -- so the same molecule parameterised from a .smi and from a .sdf gets the same
#: parameter id with DIFFERENT molecule.sdf coordinates. Measured by md-tools-propka: two builds
#: from the SMILES gave identical charges and the same id as the package built from the SDF.
PARAMETERIZE_SUFFIXES = (".sdf", ".mol2", ".smi")


def read_molecule(path: Path, *, cfg: Optional[dict] = None, work: Optional[Path] = None):
    """The one molecule in *path*, hydrogens kept, or a refusal naming what is wrong with it.

    Returns `(molecule, how)`, where `how` says where the coordinates came from -- read, or
    embedded from a SMILES -- so the record can tell the routes apart. `molecule.sdf` legitimately
    differs between them while the parameter id does not.
    """
    from rdkit import Chem

    from ..openmm.system import initial_structure, read_single_sdf_molecule

    path = Path(path)
    if path.suffix.lower() == ".smi":
        from .top import read_single_smiles

        smiles, _name = read_single_smiles(path)
        prepared = initial_structure(smiles, Path(work) / "structure", cfg or {})
        mol = Chem.MolFromMolFile(prepared["solute_sdf"], removeHs=False)
        seed = ((cfg or {}).get("structure", {}).get("etkdg", {}).get("seed")
                or (cfg or {}).get("run", {}).get("seed"))
        return mol, (f"embedded from the SMILES {smiles} (ETKDGv3 seed {seed}, "
                     f"{prepared['n_conformers_embedded']} conformers, "
                     f"{prepared['mmff_variant']}-minimised, lowest kept)")
    if path.suffix.lower() == ".sdf":
        return read_single_sdf_molecule(path), "read from the SDF, used as given"
    mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    if mol is None:
        raise ConfigError(
            f"-i {path}: RDKit could not read this file as a mol2. A file whose atom types or "
            f"bond block it cannot parse arrives here; check that it is the file you meant, and "
            f"that its hydrogens are explicit.")
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        raise ConfigError(f"-i {path}: carries no 3D conformer, so it supplies no coordinates.")
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ConfigError(
            f"-i {path}: carries no explicit hydrogens. The molecule is parameterised exactly as "
            f"given and none are added, so an implicit-hydrogen file would be parameterised as "
            f"its heavy-atom skeleton.")
    return mol, "read from the mol2, used as given"


def _ligand_system(package) -> str:
    """The molecule alone, serialised: no solvent, no cutoff, no constraints."""
    from openmm import XmlSerializer

    from ..ligands.parameters import ligand_system

    return XmlSerializer.serialize(ligand_system(package.ffxml_text, package.mol,
                                                 package.atom_names, package.template_name))


def _topology_pdb(package, path: Path) -> None:
    from rdkit import Chem

    mol = Chem.Mol(package.mol)
    for atom, name in zip(mol.GetAtoms(), package.atom_names):
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(f"{name:<4}"[:4],
                                                    residueName=package.residue_name,
                                                    residueNumber=1, isHeteroAtom=True))
    mol.SetProp("_Name", package.residue_name)
    Chem.MolToPDBFile(mol, str(path))


def _declare_copies(directory: Path, names: list[str]) -> None:
    """Record the readable copies in the metadata, with their digests, and re-verify the package."""
    metadata_path = directory / METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[READABLE_COPIES_KEY] = {
        name: {"bytes": (directory / name).stat().st_size,
               "sha256": sha256_file(directory / name)}
        for name in sorted(names)}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


def parameterize_ligand(*, input_path: Path, config_path: Optional[Path], out_pdb: Path,
                        out_system: Path, out_log: Path, residue_name: str,
                        register: bool = False, overwrite: bool = False,
                        user_config: Optional[str] = None, md_data: Optional[str] = None,
                        echo: bool = True) -> dict[str, Any]:
    """Parameterise one molecule into a package directory. Returns the record written to `out_log`."""
    import re

    from ..ligands.build import LIGANDS_DIRNAME  # noqa: F401  (documents the sibling layout)
    from ..openmm.builders import _legacy_cfg
    from ..openmm.system_config import resolve_sys_config
    from .top import _sys_document, catalog_roots, reserved_residue_names, resolve_build_config

    input_path, out_pdb, out_system, out_log = (Path(input_path), Path(out_pdb),
                                                Path(out_system), Path(out_log))
    if not input_path.is_file():
        raise ConfigError(f"-i {input_path}: no such file")
    if input_path.suffix.lower() not in PARAMETERIZE_SUFFIXES:
        raise ConfigError(
            f"-i {input_path}: --parameterize needs a molecular graph with BOND ORDERS, so the "
            f"input must be {', '.join(PARAMETERIZE_SUFFIXES)}, not {input_path.suffix}. A .sdf "
            f"or .mol2 supplies the coordinates too; a .smi states the chemistry and the "
            f"conformer is embedded. A structure alone -- a .pdb, a .cif -- carries no bond "
            f"orders, and they cannot be recovered from coordinates.")
    if not re.fullmatch(r"[A-Za-z0-9]{3}", str(residue_name)):
        raise ConfigError(f"--resname {residue_name!r} must be three letters or digits: it is the "
                          f"PDB residue-name field, and it names the files beside the package.")
    residue_name = str(residue_name).upper()
    if residue_name in reserved_residue_names():
        raise ConfigError(f"--resname {residue_name} already names water, an ion or a protein "
                          f"residue; solvent selection and the torsion classifier read residue "
                          f"names.")

    # THE OUTPUT DIRECTORY is where the package goes, and `-op`/`-os` are copies inside it. Two
    # directories would mean a package whose readable copies live somewhere else, which is exactly
    # the loose folder that cannot be registered.
    directory = out_pdb.parent.resolve()
    if out_system.parent.resolve() != directory:
        raise ConfigError(
            f"-op {out_pdb} and -os {out_system} are in different directories. --parameterize "
            f"writes ONE directory: the package, with the readable copies beside it.")

    if out_log.parent.resolve() == directory:
        raise ConfigError(
            f"-log {out_log} is inside the package directory. A package holds its own files and "
            f"the readable copies it declares, and nothing else -- a log beside them would make "
            f"the directory unloadable. Write the log one level up, or anywhere else.")

    resolved = resolve_build_config(config_path)
    stated = resolved.pop("_stated", {})
    kind = str(resolved["solute"]["kind"])
    if kind != "ligand":
        raise ConfigError(
            f"solute.kind is {kind!r}; --parameterize writes a package of PARAMETERS, so the "
            f"configuration must say `kind: ligand`.\n"
            f"  A package records a chemical state and its parameters. It records no `kind`, and "
            f"nothing a build matches against depends on one: `peptide-like` is a BUILD-time "
            f"property -- the same force field and the same charges as `ligand`, plus a validated "
            f"peptide-chemistry map over the result, which is what drives the mbondi3 corrections "
            f"under implicit solvent.\n"
            f"  So a peptide-like solute is parameterised here with `kind: ligand`, and the "
            f"package it produces is reused by a `kind: peptide-like` build unchanged.")
    if resolved["ligands"]:
        raise ConfigError("`ligands` maps instances in a structure; --parameterize has one "
                          "molecule and no structure.")
    sys_resolved = resolve_sys_config(_sys_document(resolved))
    cfg = _legacy_cfg(sys_resolved)

    existing = [p for p in (out_pdb, out_system) if p.exists()]
    package_files = [directory / name for name in (MOLECULE_NAME, FFXML_NAME, METADATA_NAME,
                                                   CRITERIA_NAME)]
    existing += [p for p in package_files if p.exists()]
    if existing and not overwrite:
        raise ConfigError(
            f"refusing to replace {', '.join(str(p) for p in existing)}. Pass --overwrite to "
            f"replace them deliberately; a package that is silently rewritten leaves every build "
            f"that recorded its digest describing parameters that no longer exist.")

    for target in (out_pdb, out_system, out_log):
        target.parent.mkdir(parents=True, exist_ok=True)

    log = LogWriter(out_log, record_type="build-top", echo=echo)
    log("md-openmm build-top --parameterize")
    log("=" * 68)
    log.heading("Command")
    log.field("input", input_path)
    log.field("config", config_path if config_path else "(none -- built-in defaults)")
    log.field("package directory", directory)
    log.field("residue name", residue_name)

    log.heading("Resolved configuration")
    for key in ("ligand_forcefield", "ligand_charge_method", "compound_id", "parameters"):
        origin = "set" if key in stated.get("solute", ()) else "default"
        log.field(f"solute.{key}", f"{resolved['solute'][key]}   ({origin})")

    try:
        staging = Path(tempfile.mkdtemp(prefix=".parameterize-", dir=directory))
        try:
            # AFTER the log exists: a SMILES input embeds a conformer here, which is work rather
            # than a check on the inputs, and the seed it used belongs in the record.
            mol, coordinates = read_molecule(input_path, cfg=cfg, work=staging)
            log.field("coordinates", coordinates)
            from ..ligands.build import attach_ligand_package

            log.heading("Parameters")
            prepared_sdf = staging / "prepared.sdf"
            prepared_pdb = staging / "prepared.pdb"
            from rdkit import Chem

            Chem.MolToMolFile(Chem.Mol(mol), str(prepared_sdf))
            Chem.MolToPDBFile(Chem.Mol(mol), str(prepared_pdb))
            attached = attach_ligand_package(
                prepared_sdf=prepared_sdf, prepared_pdb=prepared_pdb, staging=staging,
                settings={
                    "forcefield": sys_resolved["forcefield"]["ligand"],
                    "charge_method": sys_resolved["forcefield"]["ligand_charge_method"],
                    "compound_id": resolved["solute"]["compound_id"],
                    "aliases": list(resolved["solute"]["aliases"] or []),
                    "parameters": resolved["solute"]["parameters"],
                    "catalog_roots": catalog_roots(resolved, config_path),
                    "residue_name": residue_name,
                    "input_name": input_path.name,
                    "input_sha256": file_facts(input_path)["sha256"],
                })
            if attached is None:
                raise ConfigError(
                    f"solute.ligand_forcefield = {resolved['solute']['ligand_forcefield']!r} is a "
                    f"GAFF force field, which the package format does not support yet, so "
                    f"--parameterize has nothing to write. Use a SMIRNOFF force field.")
            record = attached["record"]
            package = attached["package"]
            log.field("package", f"{record['reference']}   ({record['how']})")
            log.field("charges", f"{package.metadata['charges']['method']} / "
                                 f"{package.metadata['charges'].get('scheme')} by "
                                 f"{package.metadata['charges'].get('backend_id')}")
            log.field("force field", package.metadata["forcefield"]["resource"])

            # ASSEMBLED IN ITS OWN DIRECTORY, not in the staging root: the working files beside
            # it -- the prepared molecule, the package the attach step wrote -- are not part of
            # the package, and a package directory holds nothing it does not declare.
            assembled = staging / "package"
            assembled.mkdir()
            for name in (MOLECULE_NAME, FFXML_NAME, METADATA_NAME, CRITERIA_NAME):
                shutil.copy2(package.path / name, assembled / name)
            copies = [f"{residue_name}.sdf", out_pdb.name, out_system.name]
            shutil.copy2(assembled / MOLECULE_NAME, assembled / f"{residue_name}.sdf")
            _topology_pdb(package, assembled / out_pdb.name)
            (assembled / out_system.name).write_text(_ligand_system(package), encoding="utf-8")
            _declare_copies(assembled, copies)
            load_package(assembled, expected_directory_name=False)

            log.heading("Outputs")
            for name in sorted({MOLECULE_NAME, FFXML_NAME, METADATA_NAME, CRITERIA_NAME, *copies}):
                target = directory / name
                tmp = target.with_name(target.name + ".partial")
                shutil.copy2(assembled / name, tmp)
                os.replace(tmp, target)
                log.field(name, target)
        finally:
            # BEFORE the package is read back: the staging directory lives inside the package
            # directory so the final move is a rename within one filesystem, and a package holds
            # nothing it does not declare -- including a half-finished directory of its own.
            shutil.rmtree(staging, ignore_errors=True)
        placed = load_package(directory, expected_directory_name=False)
        log.field("re-read check", f"{placed.reference} verifies in place  OK")

        registration = None
        if register:
            from ..ligands.catalog import catalog_root_for, register_package
            from ..registry.errors import RegistrationError
            from ..registry.userconfig import load_user_config, resolve_md_data

            log.heading("Registration")
            try:
                try:
                    document, _, _ = load_user_config(user_config)
                except RegistrationError:
                    if user_config:
                        raise
                    document = {}
                root, origin = resolve_md_data(document, override=md_data)
                catalog = catalog_root_for(root)
                # data-register's own implementation, called rather than repeated: registration is
                # write-once and verified there, and a second copy of that logic would be a second
                # policy about what may enter the catalog.
                registered, destination, newly = register_package(directory, catalog)
                registration = {"reference": registered.reference,
                                "catalog": str(catalog), "md_data_from": origin,
                                "newly_written": newly}
                log.field("catalog", f"{catalog}   (${'MD_DATA'} from {origin})")
                log.field("registered", f"{registered.reference}   "
                                        f"({'written' if newly else 'already present'})")
            except (RegistrationError, PackageError) as exc:
                raise ConfigError(f"--register: {exc}") from exc

        log.update(
            mode="parameterize",
            input=file_facts(input_path),
            resolved_config=sys_resolved,
            stated_keys={k: list(v) for k, v in stated.items()},
            interpretation={"route": "parameterize", "residue_name": residue_name,
                            "input_format": input_path.suffix.lstrip("."),
                            "coordinates": coordinates},
            ligand_packages={"packages": [{**placed.summary(), "path": str(directory.name)}],
                             "attached": record, "placed_in": str(directory.name)},
            registration=registration,
            outputs={"package": {"directory": str(directory.name),
                                 "reference": placed.reference,
                                 "package_sha256": placed.package_sha256},
                     "molecule_pdb": file_facts(out_pdb),
                     "system_xml": file_facts(out_system),
                     "molecule_sdf": file_facts(directory / f"{residue_name}.sdf")},
        )
        log.complete()
        log.heading("Summary")
        log(f"  {placed.reference} written to {directory}")
        log(f"  status: completed")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        raise

    log.save()
    return log.record
