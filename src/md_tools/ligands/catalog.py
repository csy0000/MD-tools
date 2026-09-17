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

from .package import LigandPackage, PackageError, load_package

__all__ = [
    "CATALOG_SUBPATH",
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
    """Copy a verified package into the catalog. Returns (package, destination, newly_written)."""
    package = load_package(Path(package_dir))
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
