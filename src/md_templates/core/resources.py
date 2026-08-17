"""Schemas shared by the build that packages the catalog and the runtime that reads it.

Two small records travel inside the built distribution beside the catalog bytes:

    resource_manifest.json   what was packaged, and the SHA-256 of each file's exact bytes
    build_provenance.json    whether the source those bytes came from had a provable commit

They are separate because they answer separate questions. The manifest says *these are the bytes*;
the provenance says *and they came from that commit*. Integrity without provenance is a wheel whose
contents are internally consistent but anonymous; provenance without integrity is a commit SHA
stapled to whatever happens to be on disk. An identity needs both, and the provenance record binds
itself to the manifest by hash so the pair cannot be mixed and matched.

Both carry their own integer schema version, independent of the registry, template, bundle,
canonical-configuration and run-state versions. They describe packaging, and packaging changes on a
different clock from anything scientific.

**Everything here is deterministic.** No build time, hostname, user, absolute path or branch name
appears in either record — two builds of the same commit produce byte-identical metadata, which is
what lets a rebuild be compared rather than merely repeated. That is also why neither record carries
a free-text explanation: `source_state` is a closed set, and the human sentence is produced at read
time by the code that raises.

This module imports only the standard library and pydantic, because the build backend imports it
before the package is installed.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "RESOURCE_MANIFEST_SCHEMA_VERSION", "BUILD_PROVENANCE_SCHEMA_VERSION",
    "RESOURCE_MANIFEST_FILENAME", "BUILD_PROVENANCE_FILENAME", "SOURCE_PROVENANCE_FILENAME",
    "PACKAGED_SUBDIR", "SOURCE_STATES", "RESOLVED_STATES",
    "ResourceManifest", "BuildProvenance",
    "sha256_bytes", "canonical_json_bytes", "explain_source_state",
]

#: Independent of every other schema version in the repository.
RESOURCE_MANIFEST_SCHEMA_VERSION = 1
BUILD_PROVENANCE_SCHEMA_VERSION = 1

#: Where the packaged catalog lives inside the installed package, and what the two records are
#: called. `_packaged` is underscored because it is generated: it is written into the build tree at
#: build time and is never a tracked source directory.
PACKAGED_SUBDIR = "_packaged"
RESOURCE_MANIFEST_FILENAME = "resource_manifest.json"
BUILD_PROVENANCE_FILENAME = "build_provenance.json"

#: Written into an sdist so a wheel built from an unmodified archive can inherit the commit its
#: source came from. Never written into the working tree of a checkout.
SOURCE_PROVENANCE_FILENAME = ".catalog_source_provenance.json"

#: The closed set of source states a build may record. Finite and machine-readable by design: a
#: consumer branches on these, and a free-text reason would invite parsing prose.
SOURCE_STATES = (
    "clean-git-checkout",        # HEAD queried, tree proven clean
    "verified-source-archive",   # sdist carrying provenance whose catalog bytes still match
    "dirty-source-tree",         # HEAD known, uncommitted or untracked changes present
    "unverifiable-git-status",   # HEAD known, `git status` failed or timed out
    "no-verifiable-git-provenance",  # no Git metadata and no verified archive record
)

#: Only these two may carry a commit SHA. The rest are refusals.
RESOLVED_STATES = ("clean-git-checkout", "verified-source-archive")

_FULL_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: dict) -> bytes:
    """Sorted keys, fixed separators, one trailing newline.

    The records are hashed and compared byte-for-byte, so their serialisation has to be a function
    of their content alone.
    """
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResourceManifest(_Strict):
    """What was packaged, and the hash of each file's exact bytes.

    `resources` is keyed by **logical repository path** — `registry.yaml`,
    `templates/rest2/openmm/explicit-water/template.yaml` — not by a location inside the wheel. The
    logical path is what an identity is made of, so it is what the manifest speaks in; where the
    bytes physically sit in a distribution is an implementation detail that must never leak into an
    identity.
    """

    schema_version: int
    resources: dict[str, str]
    references_validated_at_build: list[str]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _check_version(cls, v):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"schema_version: must be an integer, got {v!r}")
        if v != RESOURCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version: {v!r} is not supported; this build understands "
                f"{RESOURCE_MANIFEST_SCHEMA_VERSION}"
            )
        return v

    @field_validator("resources")
    @classmethod
    def _check_resources(cls, v):
        from .paths import PathError, normalise_repo_relative

        if not v:
            raise ValueError("resources: must not be empty")
        for path, digest in v.items():
            # Path traversal in a manifest is a read primitive: `../../etc/...` would make the
            # loader hash and trust a file outside the package.
            try:
                normalise_repo_relative(path, field=f"resources[{path!r}]")
            except PathError as exc:
                raise ValueError(str(exc)) from exc
            if not isinstance(digest, str) or not _SHA256.match(digest):
                raise ValueError(
                    f"resources[{path!r}]: {digest!r} is not a lowercase 64-character SHA-256"
                )
        return v

    @field_validator("references_validated_at_build")
    @classmethod
    def _check_references(cls, v):
        from .paths import PathError, normalise_repo_relative

        for i, ref in enumerate(v):
            try:
                normalise_repo_relative(ref, field=f"references_validated_at_build[{i}]")
            except PathError as exc:
                raise ValueError(str(exc)) from exc
        if list(v) != sorted(v):
            raise ValueError("references_validated_at_build: must be sorted, for determinism")
        return v

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class BuildProvenance(_Strict):
    """Whether the source that produced these bytes had a provable commit.

    `commit_sha` is present **exactly** when `resolved` is true, and only two source states may
    resolve. The model refuses every other combination, so a record cannot describe a dirty tree
    while carrying a SHA that a careless reader would use anyway.
    """

    schema_version: int
    resolved: bool
    source_state: Literal[SOURCE_STATES]  # type: ignore[valid-type]
    commit_sha: Optional[str] = None
    canonical_url: str
    resource_manifest_sha256: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def _check_version(cls, v):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"schema_version: must be an integer, got {v!r}")
        if v != BUILD_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version: {v!r} is not supported; this build understands "
                f"{BUILD_PROVENANCE_SCHEMA_VERSION}"
            )
        return v

    @field_validator("resource_manifest_sha256")
    @classmethod
    def _check_manifest_hash(cls, v):
        if not isinstance(v, str) or not _SHA256.match(v):
            raise ValueError(
                f"resource_manifest_sha256: {v!r} is not a lowercase 64-character SHA-256"
            )
        return v

    @model_validator(mode="after")
    def _resolved_iff_sha(self):
        if self.resolved:
            if self.source_state not in RESOLVED_STATES:
                raise ValueError(
                    f"resolved is true but source_state is {self.source_state!r}; only "
                    f"{list(RESOLVED_STATES)} may carry a commit"
                )
            if self.commit_sha is None:
                raise ValueError("resolved is true but no commit_sha is recorded")
            if not _FULL_SHA.match(self.commit_sha):
                raise ValueError(
                    f"commit_sha: {self.commit_sha!r} is not a full lowercase 40-character "
                    f"hexadecimal commit SHA. Abbreviated SHAs, tags and branch names are refused."
                )
        else:
            if self.commit_sha is not None:
                raise ValueError(
                    f"resolved is false but commit_sha {self.commit_sha!r} is recorded. An "
                    f"unresolved build must not carry a SHA that something downstream might use."
                )
            if self.source_state in RESOLVED_STATES:
                raise ValueError(
                    f"resolved is false but source_state is {self.source_state!r}"
                )
        return self

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


#: The human sentence lives here, not in the record, so the record stays deterministic and the
#: explanation stays maintainable.
_EXPLANATIONS = {
    "dirty-source-tree": (
        "the source tree had uncommitted or untracked changes when this distribution was built, so "
        "no commit describes the catalog bytes inside it. Rebuild from a committed tree."
    ),
    "unverifiable-git-status": (
        "`git status` failed or timed out while this distribution was built, so the tree could not "
        "be shown to be clean. A tree that cannot be proven clean is treated as dirty. Rebuild "
        "where Git works."
    ),
    "no-verifiable-git-provenance": (
        "this distribution was built from a source tree with no Git metadata and no verified source "
        "archive record, so there is nothing to identify the catalog bytes with. Build from a "
        "checkout, or from an unmodified sdist produced by one."
    ),
    "clean-git-checkout": "built from a clean Git checkout",
    "verified-source-archive": (
        "built from a source archive whose recorded provenance still matches its catalog bytes"
    ),
}


def explain_source_state(source_state: str) -> str:
    return _EXPLANATIONS.get(source_state, f"unrecognised source state {source_state!r}")
