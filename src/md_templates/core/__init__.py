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
from .paths import PathError, normalise_repo_relative  # noqa: F401
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
]
