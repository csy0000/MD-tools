"""The shared ligand parameter catalog: `$MD_DATA/parameters/ligands/<compound-id>/<parameter-id>/`.

A catalog is not a simulation dataset and is not laid out like one: the registered-dataset
contract (`md_tools.data_contract`) is year/project/data, while a parameter package belongs to a
compound and is shared by every project that simulates it. Registration into the catalog is
write-once. A package that is already there is accepted only if it is byte-identical; a package is
never replaced, because every build that recorded its digest would then describe parameters that
no longer exist.

A build READS the catalog and copies the package it uses into its own build directory, so the build
stays self-contained if the catalog moves or is lost. A build never writes to the catalog; that is
`md-openmm data-register --ligand-package`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

from .match import CRITERIA_SCHEMA
from .package import (CRITERIA_NAME, LigandPackage, PackageError, load_package,
                      read_criteria)

__all__ = [
    "CATALOG_SUBPATH",
    "search_for_match",
    "REFERENCE_PATTERN",
    "catalog_root_for",
    "default_catalog_root",
    "find_package",
    "parse_reference",
    "register_package",
    "search_catalog",
]

CATALOG_SUBPATH = Path("parameters") / "ligands"
REFERENCE_PATTERN = re.compile(r"(?P<compound>[^/\s]+)/(?P<parameter>param_[0-9a-f]{12})")


def catalog_root_for(md_data: Path) -> Path:
    return Path(md_data) / CATALOG_SUBPATH


def default_catalog_root() -> Optional[Path]:
    """`$MD_DATA/parameters/ligands` when the managed storage root is known; else None.

    Resolved the way registration resolves `$MD_DATA` -- the environment variable, then the user
    configuration -- but a missing root is not an error here: a build that names its catalog
    explicitly, or reuses nothing, does not need one.
    """
    if os.environ.get("MD_DATA"):
        return catalog_root_for(Path(os.environ["MD_DATA"]).expanduser())
    try:
        from ..registry.userconfig import load_user_config, resolve_md_data

        document, _, _ = load_user_config()
        root, _ = resolve_md_data(document)
        return catalog_root_for(root)
    except Exception:
        return None


def parse_reference(reference: str) -> tuple[str, str]:
    from .identity import check_compound_id

    match = REFERENCE_PATTERN.fullmatch(str(reference))
    if not match:
        raise PackageError(
            f"parameter reference {reference!r} is not `<compound-id>/param_<12 hex>`, for "
            f"example `CHEMBL112/param_e932f4c4f371`. A package is named by its identity, never "
            f"by a compound name alone: one compound can have several parameter packages.")
    return check_compound_id(match["compound"]), match["parameter"]


def find_package(reference: str, roots: Iterable[Path]) -> LigandPackage:
    """Load `<compound>/<parameter>` from the first root that holds it, verified."""
    compound, parameter = parse_reference(reference)
    searched = []
    for root in roots:
        if root is None:
            continue
        candidate = Path(root) / compound / parameter
        searched.append(str(Path(root)))
        if candidate.is_dir():
            return load_package(candidate)
    raise PackageError(
        f"parameter package {reference} was not found in "
        f"{', '.join(searched) if searched else 'any catalog (none is configured: set $MD_DATA or ligand_catalog.path)'}. "
        f"A build reuses only packages that exist; it does not regenerate one under this name.")


def register_package(package_dir: Path, catalog_root: Path) -> tuple[LigandPackage, Path, bool]:
    """Copy a verified package into the catalog. Returns (package, destination, newly_written).

    The SOURCE directory may be named anything: a local package folder is `build/parameter/`, not
    `param_<id>/`, and registration is what gives it the catalog's shape. The DESTINATION is built
    from the package's own identity by `copy_into`, so `<compound>/param_<id>/` is still what the
    catalog holds -- and the copy placed there is loaded again, with its name checked.
    """
    package = load_package(Path(package_dir), expected_directory_name=False)
    destination = Path(catalog_root) / package.compound_id / package.parameter_id
    if destination.exists():
        # The same identity was registered before, typically by another build of the same
        # molecule. Its files may differ in what is not identity -- a creation time, the reference
        # conformer's coordinates -- and the catalog copy is kept as it is. Anything that IS
        # identity must agree, or one of the two packages is lying about its contents.
        existing = load_package(destination)
        for key in ("parameter_digest",):
            if existing.metadata[key] != package.metadata[key]:
                raise PackageError(f"{destination} holds a different {key} under the same "
                                   f"parameter id; refusing to register {package_dir}")
        if (existing.metadata["chemical_state"]["digest"]
                != package.metadata["chemical_state"]["digest"]):
            raise PackageError(f"{destination} holds a different chemical state under the same "
                               f"parameter id; refusing to register {package_dir}")
        return existing, destination, False
    placed = package.copy_into(catalog_root)
    load_package(placed)
    return package, placed, True


def search_catalog(text: str, root: Path) -> list[dict]:
    """Packages whose compound id, aliases, residue name or canonical SMILES contain *text*."""
    import json

    needle = str(text).lower()
    found = []
    root = Path(root)
    if not root.is_dir():
        return found
    for metadata_path in sorted(root.glob("*/param_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        haystack = [metadata["compound"]["id"], metadata["compound"]["residue_name"],
                    metadata["chemical_state"]["canonical_smiles"],
                    *metadata["compound"].get("aliases", [])]
        if any(needle in str(item).lower() for item in haystack):
            found.append({"reference": f"{metadata['compound']['id']}/{metadata['parameter_id']}",
                          "aliases": metadata["compound"].get("aliases", []),
                          "canonical_smiles": metadata["chemical_state"]["canonical_smiles"],
                          "forcefield": metadata["forcefield"]["resource"],
                          "charge_method": metadata["charges"]["method"]})
    return found


def search_for_match(request: dict, roots: Iterable[Path]) -> tuple[Optional[LigandPackage],
                                                                    dict]:
    """The first package in *roots* whose declared criteria match *request*, and why.

    Candidates are compared on their `parameter.config`, which is cheap to read and is what the
    catalog publishes; only the one that matches is then LOADED, which verifies its parameters,
    digests and identity in full before anything reuses it. A candidate whose file cannot be read
    is reported as a candidate that was skipped, not silently ignored: a catalog entry nobody can
    parse is a fact the build record should carry.

    The report lists every candidate considered with the reason it did or did not match, so a
    build can record WHY it reused a package -- or why it went on to parameterise the molecule.
    """
    from .match import matches

    considered: list[dict] = []
    for root in roots:
        if root is None or not Path(root).is_dir():
            continue
        for metadata_path in sorted(Path(root).glob("*/param_*/metadata.json")):
            directory = metadata_path.parent
            reference = f"{directory.parent.name}/{directory.name}"
            criteria_path = directory / CRITERIA_NAME
            try:
                # A package written before the criteria file existed declares the same things in
                # its metadata, so it is loaded and its criteria derived rather than skipped: it
                # is an older package, not an incomplete one.
                candidate = read_criteria(criteria_path) if criteria_path.is_file() else None
                if candidate is None or candidate.get("schema_version") != CRITERIA_SCHEMA:
                    # Absent, or written in another vocabulary: derive it from the package itself
                    # rather than compare two shapes. That costs a full load for such an entry and
                    # keeps older catalogs usable instead of quietly unmatchable.
                    candidate = load_package(directory).criteria
            except Exception as exc:
                considered.append({"reference": reference, "root": str(root), "matched": False,
                                   "skipped": f"its criteria could not be read ({exc})"})
                continue
            verdict = matches(request, candidate)
            considered.append({"reference": reference, "root": str(root), **verdict.as_dict()})
            if verdict.matched:
                package = load_package(directory)
                return package, {"decision": "reuse", "matched": reference,
                                 "catalog_root": str(root), "considered": considered}
    return None, {"decision": "parameterise", "matched": None, "considered": considered}
