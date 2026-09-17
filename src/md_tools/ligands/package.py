"""A ligand parameter package: one chemical state, one saved parameter set, three files.

    <compound-id>/<parameter-id>/
        molecule.sdf       the exact chemical state, atom order = package atom order
        parameters.ffxml   self-contained OpenMM parameters, names derived from the parameter id
        metadata.json      identities, charges, provenance and artifact digests

Created in exactly one of two ways, and loaded in exactly one:

  create_package               charges are generated HERE, once, and never again for this package
  import_package_from_system   charges are read off an existing UNSCALED System and every other
                               term that System carries is checked against the regenerated ones
  load_package                 verifies everything below and runs no parameterisation at all

IDENTITY. `parameter_id = "param_" + sha256(canonical_json({chemical_state, parameters}))[:12]`,
where `chemical_state` is the state digest (`identity.chemical_state`) and `parameters` the digest
of the canonical parameter table (`parameters.parameter_table`). The directory name is checked
against that derivation on every load; it is never the source of the identity.

ATOM IDENTITY. Atom i of the package is the i-th atom block of `molecule.sdf`. Atom names are the
`MDT_ATOM_NAMES` property of that record -- element plus per-element count unless the creator
supplied names -- and they are the names the residue template and every built topology carry, so
one compound has the same atom names in every environment it is built in.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .identity import (CHEMICAL_STATE_SCHEME, chemical_state, check_compound_id,
                       unassigned_stereo)
from .parameters import (PARAMETER_TABLE_SCHEMA, canonical_json, ligand_system,
                         nonbonded_scales, parameter_digest, parameter_table, rename_ffxml)

__all__ = [
    "ATOM_NAMES_PROPERTY",
    "LigandPackage",
    "PACKAGE_FILES",
    "PACKAGE_SCHEMA",
    "PackageError",
    "SUPPORTED_CHARGE_METHODS",
    "create_package",
    "import_package_from_system",
    "load_package",
    "parameter_id_for",
]

PACKAGE_SCHEMA = "md-tools-ligand-package/1"
MOLECULE_NAME = "molecule.sdf"
FFXML_NAME = "parameters.ffxml"
METADATA_NAME = "metadata.json"
#: What a build SEARCHES: the three things that decide whether these parameters may be reused --
#: topology, protonation state and the charge implementation -- plus the force field, in a
#: browsable YAML file. Derived from `metadata.json` and re-derived on every load, so the two
#: cannot drift; `md_tools.ligands.match` is the one definition of what they mean.
CRITERIA_NAME = "parameter.config"
PACKAGE_FILES = (MOLECULE_NAME, FFXML_NAME, METADATA_NAME, CRITERIA_NAME)
ATOM_NAMES_PROPERTY = "MDT_ATOM_NAMES"
PARAMETER_ID_PREFIX = "param_"
IDENTITY_SCHEME = ("parameter_id = 'param_' + sha256(canonical_json({'chemical_state': "
                   "<state digest>, 'parameters': <parameter digest>}))[:12]")

#: Charge methods a package can be CREATED with. What each one runs is recorded in the metadata.
SUPPORTED_CHARGE_METHODS = ("am1bcc", "am1bcc_nagl", "nagl")

#: Force-field families a package can be created from. GAFF is not here yet: its ffxml shares
#: atom classes with the installed gaff.xml rather than carrying per-atom parameters, and a
#: converter that keeps its improper ordering exact has not been validated. Refused by name.
SUPPORTED_FAMILIES = ("smirnoff",)


class PackageError(ValueError):
    """A package that cannot be created, or one that does not verify."""


@dataclass(frozen=True)
class LigandPackage:
    """A verified package. Constructed only by `load_package`."""

    path: Path
    metadata: dict[str, Any]
    ffxml_text: str
    atom_names: tuple[str, ...]
    table: dict[str, Any] = field(repr=False)
    mol: Any = field(repr=False, compare=False)
    #: The match criteria this package declares, as `parameter.config` holds them.
    criteria: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def compound_id(self) -> str:
        return self.metadata["compound"]["id"]

    @property
    def parameter_id(self) -> str:
        return self.metadata["parameter_id"]

    @property
    def reference(self) -> str:
        """`CHEMBL112/param_0123456789ab` -- how a configuration names this package."""
        return f"{self.compound_id}/{self.parameter_id}"

    @property
    def template_name(self) -> str:
        return self.metadata["template_name"]

    @property
    def residue_name(self) -> str:
        return self.metadata["compound"]["residue_name"]

    @property
    def package_sha256(self) -> str:
        return package_digest(self.path)

    @property
    def conventions(self) -> dict[str, Any]:
        return self.table["conventions"]

    def summary(self) -> dict[str, Any]:
        """What a build record says about the package it used."""
        return {
            "reference": self.reference,
            "compound_id": self.compound_id,
            "parameter_id": self.parameter_id,
            "template_name": self.template_name,
            "residue_name": self.residue_name,
            "package_sha256": self.package_sha256,
            "parameter_digest": self.metadata["parameter_digest"],
            "chemical_state_digest": self.metadata["chemical_state"]["digest"],
            "forcefield": self.metadata["forcefield"]["resource"],
            "charge_method": self.metadata["charges"]["method"],
            "net_formal_charge": self.metadata["chemical_state"]["net_formal_charge"],
            "n_atoms": len(self.atom_names),
        }

    def copy_into(self, root: Path) -> Path:
        """Copy the three files to `<root>/<compound>/<parameter>/`, verified, and return it.

        An existing copy is accepted only if it is byte-identical; it is never overwritten.
        """
        destination = Path(root) / self.compound_id / self.parameter_id
        if destination.exists():
            if package_digest(destination) != self.package_sha256:
                raise PackageError(
                    f"{destination} already holds a package with this identity whose files "
                    f"differ from {self.path}. Refusing to overwrite it.")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.parameter_id}-", dir=destination.parent))
        try:
            for name in PACKAGE_FILES:
                shutil.copy2(self.path / name, staging / name)
            if package_digest(staging) != self.package_sha256:
                raise PackageError(f"the copy of {self.reference} in {staging} does not verify")
            os.rename(staging, destination)
        except BaseException:
            for name in PACKAGE_FILES:
                (staging / name).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
            raise
        return destination


# ------------------------------------------------------------------------------------------------
# digests and identity
# ------------------------------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_digest(directory: Path) -> str:
    """One digest for the three files together: sha256 over `name sha256` lines, sorted."""
    lines = []
    for name in PACKAGE_FILES:
        path = Path(directory) / name
        if not path.is_file():
            raise PackageError(f"{directory}: missing {name}")
        lines.append(f"{name} {_sha256_bytes(path.read_bytes())}")
    return _sha256_bytes("\n".join(sorted(lines)).encode())


def parameter_id_for(state_digest: str, table_digest: str) -> str:
    payload = canonical_json({"chemical_state": state_digest, "parameters": table_digest})
    return PARAMETER_ID_PREFIX + _sha256_bytes(payload.encode())[:12]


# ------------------------------------------------------------------------------------------------
# the molecule
# ------------------------------------------------------------------------------------------------
def default_atom_names(mol) -> list[str]:
    """Element symbol plus per-element count: C1, C2, O1, H1 ... -- at most four characters."""
    counts: dict[str, int] = {}
    names = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        names.append(f"{symbol}{counts[symbol]}")
    return names


def _check_atom_names(names: Sequence[str], n_atoms: int) -> list[str]:
    names = [str(n) for n in names]
    if len(names) != n_atoms:
        raise PackageError(f"{len(names)} atom names for {n_atoms} atoms")
    if len(set(names)) != len(names):
        raise PackageError("atom names must be unique within the molecule")
    bad = [n for n in names if not (1 <= len(n) <= 4) or not n.isascii() or " " in n]
    if bad:
        raise PackageError(f"atom names must be 1-4 printable characters without spaces (PDB "
                           f"columns 13-16); refused: {bad[:5]}")
    return names


def _prepared_molecule(mol):
    """A sanitised copy with explicit hydrogens and stereochemistry assigned, or a refusal."""
    from rdkit import Chem

    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    if any(a.GetAtomicNum() > 1 and a.GetTotalNumHs(includeNeighbors=False) for a in mol.GetAtoms()):
        raise PackageError(
            "the molecule has implicit hydrogens. A package defines an exact chemical state, and "
            "hydrogens are what distinguish its tautomers and protomers, so every hydrogen must "
            "be an explicit atom.")
    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        Chem.AssignStereochemistryFrom3D(mol)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    missing = unassigned_stereo(mol)
    if missing:
        raise PackageError(
            f"the molecule has {len(missing)} unassigned stereo element(s) {missing}. A package "
            f"describes one stereoisomer; supply 3D coordinates or a SMILES with the "
            f"stereochemistry stated.")
    return mol


def read_package_molecule(path: Path):
    """The single record of molecule.sdf, hydrogens kept, with its atom names."""
    from rdkit import Chem

    records = [m for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)]
    if len(records) != 1 or records[0] is None:
        raise PackageError(f"{path}: expected exactly one readable molecule record")
    mol = records[0]
    if not mol.HasProp(ATOM_NAMES_PROPERTY):
        raise PackageError(f"{path}: has no {ATOM_NAMES_PROPERTY} property")
    names = _check_atom_names(mol.GetProp(ATOM_NAMES_PROPERTY).split(), mol.GetNumAtoms())
    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        Chem.AssignStereochemistryFrom3D(mol)
    return mol, names


def _write_molecule(mol, names: Sequence[str], residue_name: str, path: Path) -> None:
    from rdkit import Chem

    out = Chem.Mol(mol)
    out.SetProp("_Name", residue_name)
    out.SetProp(ATOM_NAMES_PROPERTY, " ".join(names))
    with Chem.SDWriter(str(path)) as writer:
        writer.SetKekulize(True)
        writer.write(out)


def _offmol(mol, charges: Optional[Sequence[float]] = None):
    import numpy as np
    from openff.toolkit import Molecule
    from openff.units import unit as off_unit

    offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=False, hydrogens_are_explicit=True)
    if charges is not None:
        offmol.partial_charges = np.asarray(charges, dtype=float) * off_unit.elementary_charge
    return offmol


# ------------------------------------------------------------------------------------------------
# charges -- the one place in this package that generates them
# ------------------------------------------------------------------------------------------------
def _generate_charges(offmol, method: str) -> dict[str, Any]:
    """Assign partial charges to *offmol* with *method*, and say what actually ran."""
    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

    wrappers = [t.__class__.__name__ for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits]
    method = str(method).lower()
    if method == "am1bcc":
        if "AmberToolsToolkitWrapper" not in wrappers:
            raise PackageError(
                "charge method 'am1bcc' needs AmberTools' sqm, and the OpenFF toolkit registry "
                f"only has {wrappers}. Activate the environment that provides AmberTools. No "
                "other charge method is substituted.")
        openeye = any("OpenEye" in w for w in wrappers)
        scheme = "am1bccelf10" if openeye else "am1bcc"
        offmol.assign_partial_charges(scheme)
        return {"method": "am1bcc", "scheme": scheme,
                "backend_id": "openeye" if openeye else "ambertools-sqm",
                "backend": "OpenEye" if openeye else "AmberTools sqm",
                "conformer": "generated by the OpenFF toolkit for the charge calculation"}
    if method in ("am1bcc_nagl", "nagl"):
        if "NAGLToolkitWrapper" not in wrappers:
            raise PackageError(f"charge method {method!r} needs OpenFF NAGL, which is not "
                               f"registered ({wrappers})")
        from ..openmm.system import resolve_nagl_am1bcc_model

        model = resolve_nagl_am1bcc_model()
        offmol.assign_partial_charges(model["path"])
        return {"method": method, "scheme": model["name"], "backend_id": "openff-nagl",
                "backend": "OpenFF NAGL",
                "model": {"name": model["name"], "sha256": model["sha256"]}}
    raise PackageError(f"charge method {method!r} cannot create a package; supported: "
                       f"{', '.join(SUPPORTED_CHARGE_METHODS)}")


def _software_versions() -> dict[str, Any]:
    from ..build.record import package_versions

    versions = package_versions()
    try:
        from importlib.metadata import version

        versions["openff-nagl-models"] = version("openff-nagl-models")
    except Exception:
        versions["openff-nagl-models"] = None
    return versions


# ------------------------------------------------------------------------------------------------
# creation
# ------------------------------------------------------------------------------------------------
def _smirnoff_ffxml(offmol, resource: str) -> tuple[str, Any]:
    """Parameters for a molecule that ALREADY carries its charges; no charge call is possible.

    The generator uses the molecule's own partial charges when they are nonzero, and the check
    below makes that a fact rather than a hope: if charges were absent the generator would run
    AM1-BCC itself, silently, which is the regeneration a package exists to prevent.
    """
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    if offmol.partial_charges is None or not any(abs(q) > 0 for q in offmol.partial_charges.m):
        raise PackageError("internal: the molecule has no charges before parameter assignment")
    generator = SMIRNOFFTemplateGenerator(molecules=[offmol], forcefield=resource)
    if not generator._molecule_has_user_charges(offmol):
        raise PackageError("internal: openmmforcefields does not see the assigned charges")
    ffxml = generator.generate_residue_template(offmol)
    return ffxml, generator.get_openmm_system(offmol)


def _compare_tables(package: dict[str, Any], reference: dict[str, Any], *, where: str,
                    skip_masses: bool = False, missing_bonds_ok: Iterable | dict = (),
                    rel: float = 1e-9) -> list[str]:
    """Differences between two parameter tables, as sentences. Empty means they agree."""
    import math

    problems: list[str] = []

    def close(a, b):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)

    for i, (pa, pr) in enumerate(zip(package["atoms"], reference["atoms"])):
        if pa[0] != pr[0]:
            problems.append(f"{where}: atom {i} element {pa[0]} vs {pr[0]}")
        for label, k in (("mass", 1), ("charge", 2), ("sigma", 3), ("epsilon", 4)):
            if skip_masses and k == 1:
                continue
            if not close(pa[k], pr[k]):
                problems.append(f"{where}: atom {i} {label} {pa[k]!r} vs {pr[k]!r}")
    if len(package["atoms"]) != len(reference["atoms"]):
        problems.append(f"{where}: {len(package['atoms'])} vs {len(reference['atoms'])} atoms")
    missing_ok = (dict(missing_bonds_ok) if isinstance(missing_bonds_ok, dict)
                  else {tuple(sorted(p)): None for p in missing_bonds_ok})
    for key, width in (("bonds", 2), ("angles", 3), ("proper_torsions", 4),
                       ("improper_torsions", 4), ("exceptions", 2)):
        def keyed(rows, torsion=key.endswith("torsions")):
            out: dict[tuple, list] = {}
            for row in rows:
                if torsion:
                    out.setdefault(tuple(row[:width]), []).append(tuple(row[width:]))
                else:
                    out[tuple(row[:width])] = [tuple(row[width:])]
            return {k: sorted(v) for k, v in out.items()}

        a, r = keyed(package[key]), keyed(reference[key])
        for term in sorted(set(a) | set(r)):
            if term not in r and key == "bonds" and term in missing_ok:
                length = missing_ok[term]
                if length is not None and not close(a[term][0][0], length):
                    problems.append(f"{where}: bond {term} length {a[term][0][0]!r} is not the "
                                    f"constraint distance {length!r}")
                continue
            if term not in a or term not in r:
                problems.append(f"{where}: {key} {term} only in "
                                f"{'package' if term in a else 'reference'}")
                continue
            if len(a[term]) != len(r[term]) or any(
                    len(x) != len(y) or not all(close(float(p), float(q)) for p, q in zip(x, y))
                    for x, y in zip(a[term], r[term])):
                problems.append(f"{where}: {key} {term} {a[term]} vs {r[term]}")
    for name in ("lj14scale",):
        if package["conventions"][name] != reference["conventions"][name]:
            problems.append(f"{where}: {name} differs")
    if not close(package["conventions"]["coulomb14scale"], reference["conventions"]["coulomb14scale"]) \
            and abs(package["conventions"]["coulomb14scale"]
                    - reference["conventions"]["coulomb14scale"]) > 1e-9:
        problems.append(f"{where}: coulomb14scale differs")
    return problems


CRITERIA_HEADER = """\
# What a build must match to REUSE these parameters, and nothing else.
#
# This file is YAML despite the .config suffix, as every configuration in this package is. It is
# DERIVED from metadata.json and re-derived whenever the package is loaded, so it cannot drift
# from the parameters it describes; `md_tools.ligands.match` is the one definition of what a
# match means, and both the catalog search and the decision to parameterise instead consult it.
#
# All four are compared, and a near match is a difference:
#   topology     the heavy-atom skeleton (tautomers and protomers share it)
#   protonation  the exact chemical state: every hydrogen, formal charges, bond orders, stereo
#   charges      the method, the scheme it resolved to, and the implementation that ran
#   forcefield   the exact small-molecule force field resource
"""


def criteria_document(metadata: dict[str, Any], mol) -> dict[str, Any]:
    """The contents of `parameter.config`: the match criteria plus what they identify."""
    from .match import criteria_of_metadata

    document = criteria_of_metadata(metadata, mol)
    document.update({
        "compound": {"id": metadata["compound"]["id"],
                     "aliases": list(metadata["compound"]["aliases"]),
                     "residue_name": metadata["compound"]["residue_name"]},
        "parameter_id": metadata["parameter_id"],
        "parameter_digest": metadata["parameter_digest"],
        # Recorded, NOT compared: a newer AmberTools or toolkit does not by itself make the saved
        # numbers wrong, and requiring equality would force a regeneration on every upgrade. A
        # build that cares can read this; the search reports it as a note.
        "charge_software": {k: (metadata["software"] or {}).get(k)
                            for k in ("ambertools", "openff-toolkit", "openmmforcefields",
                                      "openff-nagl-models", "rdkit")},
    })
    return document


def write_criteria(metadata: dict[str, Any], mol, path: Path) -> None:
    import yaml

    text = CRITERIA_HEADER + yaml.safe_dump(criteria_document(metadata, mol), sort_keys=True,
                                            default_flow_style=False)
    Path(path).write_text(text, encoding="utf-8")


def read_criteria(path: Path) -> dict[str, Any]:
    from ..build.strict import load_yaml_strictly

    document = load_yaml_strictly(Path(path).read_text(encoding="utf-8"), source=str(path))
    if not isinstance(document, dict):
        raise PackageError(f"{path}: expected a mapping")
    return document


def _finish_package(*, mol, names: list[str], raw_ffxml: str, reference_table: dict[str, Any],
                    compound_id: str, aliases: Sequence[str], residue_name: str, resource: str,
                    charges: dict[str, Any], provenance: dict[str, Any], out_root: Path,
                    reference_label: str, missing_bonds_ok: Iterable = (),
                    skip_masses: bool = False) -> "LigandPackage":
    """Rename, hash, verify against the reference, write, and load back through `load_package`."""
    state = chemical_state(mol)
    placeholder_ffxml, _ = rename_ffxml(raw_ffxml, template_name="MDT_PENDING",
                                        type_prefix="MDT_PENDING", atom_names=names)
    c14, lj14 = nonbonded_scales(placeholder_ffxml)
    system = ligand_system(placeholder_ffxml, mol, names, "MDT_PENDING")
    table = parameter_table(system, mol, coulomb14scale=c14, lj14scale=lj14)
    problems = _compare_tables(table, reference_table, where=reference_label,
                               missing_bonds_ok=missing_bonds_ok, skip_masses=skip_masses)
    if problems:
        raise PackageError(
            f"the package parameters do not reproduce {reference_label} ({len(problems)} "
            f"difference(s)); nothing was written:\n  " + "\n  ".join(problems[:20]))
    table_digest = parameter_digest(table)
    parameter_id = parameter_id_for(state["digest"], table_digest)
    template_name = f"MDT_{parameter_id}"
    ffxml, normalisation = rename_ffxml(raw_ffxml, template_name=template_name,
                                        type_prefix=template_name, atom_names=names)

    metadata = {
        "schema_version": PACKAGE_SCHEMA,
        "identity_scheme": IDENTITY_SCHEME,
        "parameter_id": parameter_id,
        "template_name": template_name,
        "compound": {"id": compound_id, "aliases": sorted(set(str(a) for a in aliases)),
                     "residue_name": residue_name},
        "chemical_state": state,
        "atom_identity": (f"atom i is the i-th atom block of {MOLECULE_NAME}; names are its "
                          f"{ATOM_NAMES_PROPERTY} property and the residue template's atom names"),
        "atoms": [{"index": i, "name": names[i], "element": atom.GetSymbol(),
                   "formal_charge": int(atom.GetFormalCharge()),
                   "partial_charge_e": table["atoms"][i][2]}
                  for i, atom in enumerate(mol.GetAtoms())],
        "net_partial_charge_e": table["net_charge_e"],
        "forcefield": {"family": "smirnoff", "resource": resource},
        "charges": charges,
        "nonbonded_conventions": {**table["conventions"], "ffxml_normalisation": normalisation},
        "parameter_table_schema": PARAMETER_TABLE_SCHEMA,
        "parameter_digest": table_digest,
        "supported_force_terms": ["NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce",
                                  "PeriodicTorsionForce"],
        "software": _software_versions(),
        "provenance": {**provenance,
                       "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat()},
    }
    out_root = Path(out_root)
    destination = out_root / compound_id / parameter_id
    if destination.exists():
        existing = load_package(destination)
        if existing.metadata["parameter_digest"] != table_digest:
            raise PackageError(f"{destination} exists with different parameters")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{parameter_id}-", dir=destination.parent))
    try:
        _write_molecule(mol, names, residue_name, staging / MOLECULE_NAME)
        (staging / FFXML_NAME).write_text(ffxml, encoding="utf-8")
        metadata["artifacts"] = {
            name: {"bytes": (staging / name).stat().st_size,
                   "sha256": _sha256_bytes((staging / name).read_bytes())}
            for name in (MOLECULE_NAME, FFXML_NAME)}
        (staging / METADATA_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True)
                                             + "\n", encoding="utf-8")
        write_criteria(metadata, mol, staging / CRITERIA_NAME)
        load_package(staging, expected_directory_name=False)
        os.rename(staging, destination)
    except BaseException:
        for name in PACKAGE_FILES:
            (staging / name).unlink(missing_ok=True)
        if staging.exists():
            staging.rmdir()
        raise
    return load_package(destination)


def _common_checks(compound_id: str, residue_name: str, resource: str) -> tuple[str, str, str]:
    import re

    compound_id = check_compound_id(compound_id)
    if not re.fullmatch(r"[A-Z0-9]{3}", str(residue_name)):
        raise PackageError(f"residue name {residue_name!r} must be three upper-case letters or "
                           f"digits (the PDB residue-name field)")
    from ..openmm.ligand_forcefield import is_gaff, resolve_ligand_forcefield

    resolved = resolve_ligand_forcefield(resource)
    if is_gaff(resolved):
        raise PackageError(
            f"force field {resolved!r}: GAFF packages are not supported yet. A GAFF ffxml shares "
            f"atom classes with gaff.xml instead of carrying per-atom parameters, and its "
            f"improper ordering through a per-atom conversion has not been validated. Use a "
            f"SMIRNOFF force field (sage-2.2.1) for a reusable package.")
    return compound_id, str(residue_name), resolved


def create_package(mol, *, compound_id: str, residue_name: str, out_root: Path,
                   forcefield: str = "sage-2.2.1", charge_method: str = "am1bcc",
                   aliases: Sequence[str] = (), atom_names: Optional[Sequence[str]] = None,
                   source: Optional[dict[str, Any]] = None) -> LigandPackage:
    """Parameterise *mol* once and write the package. The ONLY charge-generating entry point."""
    compound_id, residue_name, resource = _common_checks(compound_id, residue_name, forcefield)
    mol = _prepared_molecule(mol)
    names = _check_atom_names(atom_names if atom_names is not None else default_atom_names(mol),
                              mol.GetNumAtoms())
    offmol = _offmol(mol)
    charges = _generate_charges(offmol, charge_method)
    charge_values = [float(q) for q in offmol.partial_charges.m]
    offmol = _offmol(mol, charge_values)
    raw_ffxml, reference_system = _smirnoff_ffxml(offmol, resource)
    c14, lj14 = nonbonded_scales(raw_ffxml)
    reference_table = parameter_table(reference_system, mol, coulomb14scale=c14, lj14scale=lj14)
    return _finish_package(
        mol=mol, names=names, raw_ffxml=raw_ffxml, reference_table=reference_table,
        compound_id=compound_id, aliases=aliases, residue_name=residue_name, resource=resource,
        charges={**charges, "source": "generated"},
        provenance={"route": "generated", "source": source or {}},
        out_root=out_root, reference_label=f"the {resource} System openmmforcefields built")


def import_package_from_system(mol, *, system, atom_indices: Sequence[int], compound_id: str,
                               residue_name: str, out_root: Path, forcefield: str,
                               charge_method: str, aliases: Sequence[str] = (),
                               atom_names: Optional[Sequence[str]] = None,
                               source: Optional[dict[str, Any]] = None,
                               charge_provenance: Optional[dict[str, Any]] = None) -> LigandPackage:
    """Recover a package from an already-built, UNSCALED System, without generating charges.

    `charge_provenance` states what produced the charges in the SOURCE build -- the scheme and the
    implementation its own record names -- and is stored beside `source: imported`; this function
    cannot establish it and does not guess it. `backend_id` is required, and must be one of
    `md_tools.ligands.match.BACKEND_IDS`: a package whose record cannot say WHICH implementation
    of a charge method produced its numbers cannot be searched for or reused safely.

    `atom_indices[i]` is the System particle of package atom i. The charges are read off the
    System; the force field then assigns every other parameter with those charges fixed, and the
    result must reproduce EVERY term the System carries for those atoms -- Lennard-Jones, 1-4
    exceptions, angles, torsions and each unconstrained bond. A bond the System constrained has no
    force constant left in it; its length must equal the constraint distance, and its force
    constant is the force field's. Masses are not compared, because hydrogen mass repartitioning
    legitimately changes them in the System; the package carries the force field's own masses.

    A REST2- or AIS-scaled state fails this comparison by construction -- its torsions, charges
    and Lennard-Jones depths are scaled -- and is refused, as are the callers that pass one knowingly.
    """
    from openmm import NonbondedForce, unit

    from .match import BACKEND_IDS

    compound_id, residue_name, resource = _common_checks(compound_id, residue_name, forcefield)
    if (charge_provenance or {}).get("backend_id") not in BACKEND_IDS:
        raise PackageError(
            f"import_package_from_system needs charge_provenance['backend_id'], one of "
            f"{BACKEND_IDS}: the source build's record says which implementation of "
            f"{charge_method!r} produced the charges being recovered, and a package that cannot "
            f"name it cannot be matched against a build's requirements.")
    mol = _prepared_molecule(mol)
    names = _check_atom_names(atom_names if atom_names is not None else default_atom_names(mol),
                              mol.GetNumAtoms())
    atom_indices = [int(i) for i in atom_indices]
    if len(atom_indices) != mol.GetNumAtoms() or len(set(atom_indices)) != len(atom_indices):
        raise PackageError("atom_indices must name one distinct System particle per package atom")
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(nonbonded) != 1:
        raise PackageError(f"the source System has {len(nonbonded)} NonbondedForce; expected 1")
    charges = [nonbonded[0].getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in atom_indices]
    formal = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if abs(sum(charges) - formal) > 1e-4:
        raise PackageError(f"the source System's charges on these atoms sum to {sum(charges):.6f}, "
                           f"not the molecule's formal charge {formal}; wrong atoms or a scaled "
                           f"System")
    offmol = _offmol(mol, charges)
    raw_ffxml, _ = _smirnoff_ffxml(offmol, resource)
    c14, lj14 = nonbonded_scales(raw_ffxml)
    reference_table, constrained = _subsystem_table(system, mol, atom_indices, c14, lj14)
    return _finish_package(
        mol=mol, names=names, raw_ffxml=raw_ffxml, reference_table=reference_table,
        compound_id=compound_id, aliases=aliases, residue_name=residue_name, resource=resource,
        charges={**(charge_provenance or {}), "method": charge_method, "source": "imported",
                 "scheme": (charge_provenance or {}).get("scheme"),
                 "note": ("read from the source System's NonbondedForce; not regenerated")},
        provenance={"route": "imported-from-system", "source": source or {},
                    "constrained_bonds_in_source": [list(p) for p in sorted(constrained)],
                    "masses": "force field masses; the source System's masses were not compared "
                              "(hydrogen mass repartitioning changes them)"},
        out_root=out_root, reference_label="the source System",
        missing_bonds_ok={pair: reference_table["_constraint_lengths"][pair]
                          for pair in constrained},
        skip_masses=True)


def _subsystem_table(system, mol, atom_indices: Sequence[int], c14: float, lj14: float):
    """The parameter table of *system* restricted to *atom_indices*, renumbered to package order.

    Constrained bonds are returned separately, with their lengths checked into the table's
    `bonds` only via the caller's comparison (the length is compared below).
    """
    from openmm import (HarmonicAngleForce, HarmonicBondForce, NonbondedForce,
                        PeriodicTorsionForce, unit)

    from .parameters import classify_torsion, _q

    local = {p: i for i, p in enumerate(atom_indices)}
    bonded = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(),
               b.GetEndAtomIdx())) for b in mol.GetBonds()}
    table: dict[str, Any] = {"atoms": [], "bonds": [], "angles": [], "proper_torsions": [],
                             "improper_torsions": [], "exceptions": [],
                             "conventions": {"coulomb14scale": c14, "lj14scale": lj14}}
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            for i, particle in enumerate(atom_indices):
                q, s, e = (_q(x) for x in force.getParticleParameters(particle))
                table["atoms"].append([mol.GetAtomWithIdx(i).GetSymbol(), 0.0, q, s, e])
            for k in range(force.getNumExceptions()):
                i, j, qq, s, e = force.getExceptionParameters(k)
                if i in local and j in local:
                    a, b = sorted((local[i], local[j]))
                    table["exceptions"].append([a, b, _q(qq), _q(s), _q(e)])
                elif (i in local) != (j in local) and (_q(qq) or _q(e)):
                    raise PackageError("the source System has a nonzero exception between the "
                                       "ligand and another atom; a covalently attached or "
                                       "modified ligand cannot be imported as a package")
        elif isinstance(force, HarmonicBondForce):
            for k in range(force.getNumBonds()):
                i, j, length, kk = force.getBondParameters(k)
                if i in local and j in local:
                    a, b = sorted((local[i], local[j]))
                    table["bonds"].append([a, b, _q(length), _q(kk)])
        elif isinstance(force, HarmonicAngleForce):
            for k in range(force.getNumAngles()):
                i, j, l, theta, kk = force.getAngleParameters(k)
                if i in local:
                    a, c = sorted((local[i], local[l]))
                    table["angles"].append([a, local[j], c, _q(theta), _q(kk)])
        elif isinstance(force, PeriodicTorsionForce):
            for k in range(force.getNumTorsions()):
                i, j, kk, l, n, phase, fk = force.getTorsionParameters(k)
                if i in local:
                    quartet = [local[i], local[j], local[kk], local[l]]
                    row = [int(n), _q(phase), _q(fk)]
                    if classify_torsion(quartet, bonded) == "proper":
                        if tuple(reversed(quartet)) < tuple(quartet):
                            quartet = quartet[::-1]
                        table["proper_torsions"].append([*quartet, *row])
                    else:
                        table["improper_torsions"].append([*quartet, *row])
    constrained: dict[tuple[int, int], float] = {}
    for k in range(system.getNumConstraints()):
        i, j, distance = system.getConstraintParameters(k)
        if i in local and j in local:
            constrained[tuple(sorted((local[i], local[j])))] = _q(distance)
    present = {(row[0], row[1]) for row in table["bonds"]}
    missing = bonded - present
    unexplained = missing - set(constrained)
    if unexplained:
        raise PackageError(f"the source System has no bond term and no constraint for bonds "
                           f"{sorted(unexplained)}")
    table["_constraint_lengths"] = constrained
    return table, set(missing)


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------
_REQUIRED_METADATA = ("schema_version", "identity_scheme", "parameter_id", "template_name",
                      "compound", "chemical_state", "atoms", "forcefield", "charges",
                      "parameter_digest", "parameter_table_schema", "artifacts", "software",
                      "provenance", "nonbonded_conventions", "atom_identity",
                      "net_partial_charge_e", "supported_force_terms")


def load_package(directory: Path, *, expected_directory_name: bool = True) -> LigandPackage:
    """Read a package and verify every claim it makes. Runs no parameterisation."""
    directory = Path(directory)
    for name in PACKAGE_FILES:
        if not (directory / name).is_file():
            raise PackageError(f"{directory}: not a ligand package (missing {name})")
    extra = sorted(p.name for p in directory.iterdir() if p.name not in PACKAGE_FILES)
    if extra:
        raise PackageError(f"{directory}: unexpected files {extra}; a package holds exactly "
                           f"{', '.join(PACKAGE_FILES)}")
    try:
        metadata = json.loads((directory / METADATA_NAME).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackageError(f"{directory / METADATA_NAME}: not valid JSON ({exc})") from exc
    missing = [k for k in _REQUIRED_METADATA if k not in metadata]
    unknown = sorted(set(metadata) - set(_REQUIRED_METADATA))
    if missing or unknown:
        raise PackageError(f"{directory / METADATA_NAME}: missing {missing}, unknown {unknown}")
    if metadata["schema_version"] != PACKAGE_SCHEMA:
        raise PackageError(f"{directory}: package schema {metadata['schema_version']!r} is not "
                           f"{PACKAGE_SCHEMA!r}")
    if metadata["parameter_table_schema"] != PARAMETER_TABLE_SCHEMA:
        raise PackageError(f"{directory}: parameter table schema "
                           f"{metadata['parameter_table_schema']!r} is not understood")
    for name in (MOLECULE_NAME, FFXML_NAME):
        recorded = metadata["artifacts"].get(name) or {}
        actual = _sha256_bytes((directory / name).read_bytes())
        if recorded.get("sha256") != actual:
            raise PackageError(f"{directory / name}: sha256 {actual} does not match the metadata "
                               f"({recorded.get('sha256')}); the package was modified")

    compound_id = check_compound_id(metadata["compound"]["id"])
    mol, names = read_package_molecule(directory / MOLECULE_NAME)
    if [a["name"] for a in metadata["atoms"]] != names:
        raise PackageError(f"{directory}: metadata atom names disagree with {MOLECULE_NAME}")
    for atom, record in zip(mol.GetAtoms(), metadata["atoms"]):
        if atom.GetSymbol() != record["element"] or atom.GetFormalCharge() != record["formal_charge"]:
            raise PackageError(f"{directory}: metadata atom {record['index']} disagrees with "
                               f"{MOLECULE_NAME}")

    state = chemical_state(mol)
    recorded_state = metadata["chemical_state"]
    if recorded_state.get("scheme") != CHEMICAL_STATE_SCHEME:
        raise PackageError(f"{directory}: chemical state scheme {recorded_state.get('scheme')!r}")
    same_rdkit = (metadata["software"] or {}).get("rdkit") == _rdkit_version()
    if same_rdkit and state["digest"] != recorded_state["digest"]:
        raise PackageError(f"{directory}: the chemical state of {MOLECULE_NAME} is "
                           f"{state['canonical_smiles']}, not the recorded "
                           f"{recorded_state['canonical_smiles']}")
    if state["fixed_h_inchi"] != recorded_state["fixed_h_inchi"]:
        raise PackageError(f"{directory}: the fixed-H InChI of {MOLECULE_NAME} does not match "
                           f"the recorded chemical state")

    ffxml_text = (directory / FFXML_NAME).read_text(encoding="utf-8")
    c14, lj14 = nonbonded_scales(ffxml_text)
    system = ligand_system(ffxml_text, mol, names, metadata["template_name"])
    table = parameter_table(system, mol, coulomb14scale=c14, lj14scale=lj14)
    digest = parameter_digest(table)
    if digest != metadata["parameter_digest"]:
        raise PackageError(f"{directory}: the parameters in {FFXML_NAME} hash to {digest}, not "
                           f"the recorded {metadata['parameter_digest']}")
    expected_id = parameter_id_for(recorded_state["digest"], digest)
    if metadata["parameter_id"] != expected_id:
        raise PackageError(f"{directory}: parameter_id {metadata['parameter_id']!r} is not the "
                           f"identity derived from its contents ({expected_id})")
    if metadata["template_name"] != f"MDT_{expected_id}":
        raise PackageError(f"{directory}: template name {metadata['template_name']!r} is not "
                           f"derived from the parameter id")
    for atom, record in zip(table["atoms"], metadata["atoms"]):
        if atom[2] != record["partial_charge_e"]:
            raise PackageError(f"{directory}: metadata partial charge of atom {record['index']} "
                               f"is not the ffxml's")
    criteria = read_criteria(directory / CRITERIA_NAME)
    derived = criteria_document(metadata, mol)
    if criteria != derived:
        differing = sorted(k for k in set(criteria) | set(derived)
                           if criteria.get(k) != derived.get(k))
        raise PackageError(
            f"{directory / CRITERIA_NAME} does not describe this package: {differing} differ from "
            f"what metadata.json and {MOLECULE_NAME} say. It is derived, so it is never edited by "
            f"hand; a build searching the catalog would match against something the parameters do "
            f"not support.")

    if expected_directory_name:
        if directory.name != expected_id:
            raise PackageError(f"{directory}: the directory is named {directory.name!r} but the "
                               f"package is {expected_id}; a directory label is not an identity")
        if directory.parent.name != compound_id:
            raise PackageError(f"{directory}: the parent directory is {directory.parent.name!r} "
                               f"but the package's compound is {compound_id}")
    return LigandPackage(path=directory, metadata=metadata, ffxml_text=ffxml_text,
                         atom_names=tuple(names), table=table, mol=mol, criteria=criteria)


def _rdkit_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("rdkit")
    except Exception:
        return None
