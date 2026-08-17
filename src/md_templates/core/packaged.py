"""Reading the catalog that travels inside an installed distribution.

PR 1 made the catalog loadable from a repository checkout. This module makes the *same* catalog
loadable from an installed wheel, with no checkout, no Git, no working-directory assumption and no
network — and lets an installed copy resolve a real immutable identity, which previously it could
not do at all (it raised `NoProvenanceError`, correctly, because nothing told it which commit its
bytes came from).

Three things must hold before an identity comes out of here, and they are checked in this order:

1. **integrity** — every packaged resource hashes to what `resource_manifest.json` says, nothing
   claimed is missing, and nothing present is unclaimed;
2. **provenance** — `build_provenance.json` says the source was clean and names the commit;
3. **binding** — the provenance record's `resource_manifest_sha256` matches the manifest actually
   read, so a commit cannot be paired with a catalog it did not describe.

Failing any of them raises before an identity is returned. A wheel is a mutable directory in
site-packages: none of this authenticates GitHub or defends against an attacker who can rewrite the
wheel *and* its metadata consistently. What it does defend against is the realistic failure — bytes
that drifted, a resource that went missing, a metadata record paired with the wrong catalog, and a
build from a tree nobody can name.

The identity always speaks in **logical repository paths**: `templates/rest2/openmm/explicit-water/
template.yaml`, never a path inside the wheel. Where the bytes physically sit in a distribution is a
packaging detail, and letting it into an identity would make the identity un-lookupable in the
repository it names.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .identity import IdentityError, NoProvenanceError, TemplateIdentity, build_identity
from .paths import PathError, normalise_repo_relative
from .registry import (
    RegistryError,
    TemplateCatalog,
    check_descriptor_agreement,
    check_entry_path_consistency,
    validate_registry_document,
)
from .resources import (
    BUILD_PROVENANCE_FILENAME,
    PACKAGED_SUBDIR,
    RESOURCE_MANIFEST_FILENAME,
    BuildProvenance,
    ResourceManifest,
    explain_source_state,
    sha256_bytes,
)
from .template import TemplateError, parse_descriptor

__all__ = [
    "PackagedCatalogError", "ResourceIntegrityError", "BuildProvenanceError",
    "UnresolvedBuildProvenanceError", "PackagedCatalog",
    "packaged_catalog_root", "load_packaged_catalog", "load_packaged_provenance",
    "verify_packaged_catalog", "packaged_catalog_available", "describe_packaged_catalog",
    "resolve_packaged_identity", "PACKAGED_ORIGIN", "PACKAGED_REFERENCE_POLICY",
]

PACKAGED_ORIGIN = "packaged-distribution"
PACKAGED_REFERENCE_POLICY = "validated-at-build"


class PackagedCatalogError(RegistryError):
    """The catalog packaged in this distribution is missing, malformed or inconsistent."""


class ResourceIntegrityError(PackagedCatalogError):
    """A packaged resource does not match the manifest that claims it."""


class BuildProvenanceError(PackagedCatalogError):
    """The packaged build-provenance record is missing, malformed or unsupported."""


class UnresolvedBuildProvenanceError(NoProvenanceError):
    """The distribution was built from a source with no provable commit.

    A `NoProvenanceError` subclass on purpose: to every caller this is the same refusal PR 1 already
    made for a non-Git directory, just discovered from packaged metadata rather than from a missing
    `.git`.
    """


@dataclass(frozen=True)
class PackagedCatalog:
    """A validated packaged catalog together with the provenance of the source that built it."""

    catalog: TemplateCatalog
    provenance: BuildProvenance
    manifest: ResourceManifest

    @property
    def resolved(self) -> bool:
        return self.provenance.resolved


def packaged_catalog_root() -> Path:
    """The directory holding the packaged catalog inside the installed package.

    Uses `importlib.resources`, never the working directory, so it is correct from any cwd and for
    any install location.
    """
    from importlib import resources

    try:
        anchor = resources.files("md_templates.core")
    except (ImportError, TypeError) as exc:  # pragma: no cover - defensive
        raise PackagedCatalogError(f"cannot locate the md_templates.core package: {exc}") from exc

    root = Path(str(anchor)) / PACKAGED_SUBDIR
    if not root.is_dir():
        raise PackagedCatalogError(
            f"this installation carries no packaged catalog ({root} is absent). It was built "
            f"without the catalog staging step, or from a distribution that predates it. Use "
            f"load_catalog(<repository root>) for a source checkout."
        )
    return root


def _read(root: Path, relative: str, *, field: str) -> bytes:
    """Read one packaged file, refusing traversal and symlinks exactly as the source loader does."""
    try:
        normalise_repo_relative(relative, field=field)
    except PathError as exc:
        raise ResourceIntegrityError(str(exc)) from exc

    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise ResourceIntegrityError(
                f"{field}: {relative!r} passes through a symlink at {part!r} inside the "
                f"distribution. Packaging must not become a way to validate mutable bytes outside "
                f"the source tree."
            )
    if not current.is_file():
        raise ResourceIntegrityError(
            f"{field}: the manifest claims {relative!r} but this distribution does not contain it"
        )
    return current.read_bytes()


def _load_manifest(root: Path) -> tuple[ResourceManifest, str]:
    path = root / RESOURCE_MANIFEST_FILENAME
    if not path.is_file():
        raise PackagedCatalogError(
            f"packaged catalog has no {RESOURCE_MANIFEST_FILENAME}; its bytes cannot be trusted"
        )
    raw = path.read_bytes()
    try:
        manifest = ResourceManifest.model_validate(json.loads(raw.decode("utf-8")))
    except Exception as exc:
        raise PackagedCatalogError(f"{RESOURCE_MANIFEST_FILENAME}: {exc}") from exc
    return manifest, sha256_bytes(raw)


def _load_provenance(root: Path, manifest_sha256: str) -> BuildProvenance:
    path = root / BUILD_PROVENANCE_FILENAME
    if not path.is_file():
        raise BuildProvenanceError(
            f"packaged catalog has no {BUILD_PROVENANCE_FILENAME}, so nothing states which commit "
            f"its bytes came from"
        )
    try:
        provenance = BuildProvenance.model_validate(json.loads(path.read_bytes().decode("utf-8")))
    except Exception as exc:
        raise BuildProvenanceError(f"{BUILD_PROVENANCE_FILENAME}: {exc}") from exc

    if provenance.resource_manifest_sha256 != manifest_sha256:
        raise BuildProvenanceError(
            f"{BUILD_PROVENANCE_FILENAME} describes a catalog whose manifest hashes to "
            f"{provenance.resource_manifest_sha256}, but this distribution's manifest hashes to "
            f"{manifest_sha256}. Provenance and catalog have been mixed, so the commit it names "
            f"does not describe these bytes."
        )
    return provenance


def verify_packaged_catalog(root: Path) -> PackagedCatalog:
    """Integrity-check and load a packaged catalog rooted at `root`.

    Public because it is the whole contract, and because a tamper test that could only exercise it
    through the no-argument API would have to reinstall the package for every case. `root` selects
    *which* packaged catalog to check; it never relaxes what is checked.
    """
    manifest, manifest_sha256 = _load_manifest(root)
    provenance = _load_provenance(root, manifest_sha256)

    # Integrity first: every claimed resource present and unchanged.
    verified: dict[str, bytes] = {}
    for relative, expected in sorted(manifest.resources.items()):
        payload = _read(root, relative, field=f"resources[{relative!r}]")
        actual = sha256_bytes(payload)
        if actual != expected:
            raise ResourceIntegrityError(
                f"resources[{relative!r}]: packaged bytes hash to {actual}, but the manifest "
                f"records {expected}. The file has been modified since this distribution was built."
            )
        verified[relative] = payload

    # And nothing unclaimed: a descriptor smuggled in beside the manifest would otherwise be
    # invisible to integrity checking while still looking like part of the catalog.
    metadata = {RESOURCE_MANIFEST_FILENAME, BUILD_PROVENANCE_FILENAME}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in metadata or relative in manifest.resources:
            continue
        raise ResourceIntegrityError(
            f"{relative!r} is present in the packaged catalog but absent from "
            f"{RESOURCE_MANIFEST_FILENAME}, so nothing vouches for its bytes"
        )

    registry_bytes = verified.get("registry.yaml")
    if registry_bytes is None:
        raise PackagedCatalogError(
            f"{RESOURCE_MANIFEST_FILENAME} does not claim registry.yaml, so this distribution has "
            f"no catalog index"
        )

    import yaml

    document = yaml.safe_load(registry_bytes.decode("utf-8"))
    if not isinstance(document, dict):
        raise PackagedCatalogError("packaged registry.yaml is not a mapping")
    registry = validate_registry_document(document, origin="packaged registry.yaml")

    descriptors = {}
    for entry in registry.templates:
        check_entry_path_consistency(entry)
        payload = verified.get(entry.template_path)
        if payload is None:
            raise PackagedCatalogError(
                f"templates[{entry.template_id!r}].template_path: {entry.template_path} is "
                f"registered but not packaged. A registry entry with no descriptor is a broken "
                f"index, not a plan."
            )
        try:
            descriptor = parse_descriptor(yaml.safe_load(payload.decode("utf-8")),
                                          origin=f"packaged {entry.template_path}")
        except TemplateError as exc:
            raise PackagedCatalogError(str(exc)) from exc
        check_descriptor_agreement(entry, descriptor, origin=f"packaged {entry.template_path}")
        descriptors[entry.template_id] = descriptor

    catalog = TemplateCatalog(root=root, registry=registry, descriptors=descriptors,
                              origin=PACKAGED_ORIGIN,
                              reference_policy=PACKAGED_REFERENCE_POLICY)
    return PackagedCatalog(catalog=catalog, provenance=provenance, manifest=manifest)


def load_packaged_catalog() -> TemplateCatalog:
    """Load and integrity-check the catalog built into this installation.

    Needs no repository root, no Git and no network. Returns the same registry entries and
    descriptors the source loader returns for the commit the distribution was built from — the
    equivalence is asserted by test, not assumed.

    Succeeds even when build provenance is unresolved: listing what a distribution contains is
    useful regardless of whether its bytes can be named. `resolve_packaged_identity` is where
    provenance becomes mandatory.
    """
    return verify_packaged_catalog(packaged_catalog_root()).catalog


def load_packaged_provenance() -> BuildProvenance:
    """The validated build-provenance record, resolved or not. For inspection and diagnostics."""
    root = packaged_catalog_root()
    manifest, manifest_sha256 = _load_manifest(root)
    return _load_provenance(root, manifest_sha256)


def resolve_packaged_identity(template_ref: str, *,
                              root: Optional[Path] = None) -> TemplateIdentity:
    """Resolve one packaged template to its immutable identity.

    **No commit may be supplied by the caller**, exactly as in a checkout: the commit comes from the
    packaged provenance record and from nowhere else. A convenience API that accepted a
    `TrustedProvenance` here would reintroduce the PR 1 defect through a new door — the whole point
    is that an installed copy can only claim what its build proved.
    """
    packaged = verify_packaged_catalog(root if root is not None else packaged_catalog_root())
    entry = packaged.catalog.require(template_ref)
    provenance = packaged.provenance

    if not provenance.resolved:
        raise UnresolvedBuildProvenanceError(
            f"this distribution carries unresolved build provenance "
            f"({provenance.source_state}), so {entry.template_id!r} has no immutable identity: "
            f"{explain_source_state(provenance.source_state)} "
            f"The packaged catalog still lists and validates; only identity is refused, because an "
            f"identity that names no reproducible commit is worse than none."
        )

    declared_url = packaged.catalog.canonical_url
    if provenance.canonical_url != declared_url:
        raise BuildProvenanceError(
            f"build provenance names repository {provenance.canonical_url!r} but the packaged "
            f"registry declares {declared_url!r}"
        )

    try:
        return build_identity(declared_url, provenance.commit_sha, entry.template_path)
    except IdentityError as exc:
        raise BuildProvenanceError(f"{BUILD_PROVENANCE_FILENAME}: {exc}") from exc


def packaged_catalog_available() -> bool:
    """Whether this installation carries a packaged catalog at all. Never raises."""
    try:
        packaged_catalog_root()
    except PackagedCatalogError:
        return False
    return True


def describe_packaged_catalog() -> Optional[dict]:
    """A small diagnostic summary, or None when nothing is packaged. Never raises on absence."""
    if not packaged_catalog_available():
        return None
    packaged = verify_packaged_catalog(packaged_catalog_root())
    return {
        "origin": packaged.catalog.origin,
        "reference_policy": packaged.catalog.reference_policy,
        "resource_manifest_schema_version": packaged.manifest.schema_version,
        "build_provenance_schema_version": packaged.provenance.schema_version,
        "resolved": packaged.provenance.resolved,
        "source_state": packaged.provenance.source_state,
        "commit_sha": packaged.provenance.commit_sha,
        "templates": sorted(packaged.catalog.descriptors),
    }
