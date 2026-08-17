"""Engine-neutral catalog and identity contracts.

This package answers "which templates exist, what does each claim, and what immutably identifies
one?" -- and nothing else. It builds no System, opens no run directory and dispatches no execution.
In PR 1 of the multi-method migration it is metadata plus validation: `md-openmm` behaviour, the
canonical configuration models, the profiles, the bundle schemas and the restart contract are all
untouched, and every descriptor records `implementation.dispatch: legacy-direct` to say so.

**Nothing here imports OpenMM, OpenFF or RDKit, and nothing here may start to.** Listing and
validating the catalog has to work in a minimal environment -- a laptop, a CI lint job, a machine
that could never run the simulations -- so the dependency floor is YAML plus pydantic. A test
asserts the closure, because this is the kind of boundary that erodes through one convenient import.

    registry.py   the root discovery index and its agreement with the tree
    template.py   the strict typed descriptor, including status enums that weak evidence cannot raise
    identity.py   canonical URL + full commit SHA + exact template path, and the dirty-tree refusal
    paths.py      the one repository-relative path rule both of the above use
    resources.py  the resource-integrity and build-provenance schemas, shared with the build
    packaged.py   the same catalog read from an installed distribution, with no checkout

`load_catalog(root)` reads a repository checkout; `load_packaged_catalog()` reads the copy built into
an installed distribution. They return the same entries and descriptors for the commit a wheel was
built from -- asserted by test, not assumed -- and differ only in what they can honestly verify:
a checkout confirms that every `repository_references` entry exists, while a wheel reports them as
`validated-at-build`, because those paths name documentation that does not travel in a wheel.
"""
from __future__ import annotations

from .identity import (  # noqa: F401
    DirtyWorkingTreeError,
    GitProvenance,
    IdentityError,
    NoProvenanceError,
    TemplateIdentity,
    TrustedProvenance,
    UnknownTemplateError,
    build_identity,
    inspect_provenance,
    normalise_commit_sha,
    resolve_identity,
)
from .packaged import (  # noqa: F401
    BuildProvenanceError,
    PackagedCatalog,
    PackagedCatalogError,
    ResourceIntegrityError,
    UnresolvedBuildProvenanceError,
    describe_packaged_catalog,
    load_packaged_catalog,
    load_packaged_provenance,
    packaged_catalog_available,
    packaged_catalog_root,
    resolve_packaged_identity,
    verify_packaged_catalog,
)
from .paths import PathError, normalise_repo_relative  # noqa: F401
from .resources import (  # noqa: F401
    BUILD_PROVENANCE_SCHEMA_VERSION,
    RESOURCE_MANIFEST_SCHEMA_VERSION,
    BuildProvenance,
    ResourceManifest,
)
from .registry import (  # noqa: F401
    CANONICAL_FORM,
    REGISTRY_FILENAME,
    REGISTRY_SCHEMA_VERSION,
    Registry,
    RegistryEntry,
    RegistryError,
    TemplateCatalog,
    load_catalog,
)
from .template import (  # noqa: F401
    TEMPLATE_SCHEMA_VERSION,
    TemplateDescriptor,
    TemplateError,
    parse_descriptor,
)

__all__ = [
    "CANONICAL_FORM", "REGISTRY_FILENAME", "REGISTRY_SCHEMA_VERSION", "TEMPLATE_SCHEMA_VERSION",
    "Registry", "RegistryEntry", "RegistryError", "TemplateCatalog", "load_catalog",
    "TemplateDescriptor", "TemplateError", "parse_descriptor",
    "TemplateIdentity", "GitProvenance", "TrustedProvenance", "IdentityError",
    "DirtyWorkingTreeError", "NoProvenanceError", "UnknownTemplateError",
    "build_identity", "inspect_provenance", "normalise_commit_sha", "resolve_identity",
    "PathError", "normalise_repo_relative",
    # the packaged catalog: the same catalog, read from an installed distribution
    "load_packaged_catalog", "resolve_packaged_identity", "load_packaged_provenance",
    "verify_packaged_catalog", "packaged_catalog_root", "packaged_catalog_available",
    "describe_packaged_catalog", "PackagedCatalog", "PackagedCatalogError",
    "ResourceIntegrityError", "BuildProvenanceError", "UnresolvedBuildProvenanceError",
    "BuildProvenance", "ResourceManifest",
    "BUILD_PROVENANCE_SCHEMA_VERSION", "RESOURCE_MANIFEST_SCHEMA_VERSION",
]
