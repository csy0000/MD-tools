"""`dataset.yaml`, contract version 2.

One dataset directory holds exactly one authoritative `dataset.yaml`, and that manifest is the
dataset's identity. Metadata only: nothing in this module opens, hashes or walks simulation data.

Every rule here is a rule that was worth a failure at some point. They are grouped below by what
they protect against.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, ValidationError,
                      model_validator)

CONTRACT_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("2.0",)

MANIFEST_NAME = "dataset.yaml"
RESOLVED_NAME = "dataset.resolved.yaml"
INVENTORY_NAME = "SHA256SUMS"

#: The segment that marks a dataset as shared rather than project-owned.
COMMON_SEGMENT = "common"

#: Exactly four decimal digits. Not a range: a plausibility window would have to be revised, and
#: a wrong-but-plausible year is exactly what this cannot catch anyway.
YEAR_PATTERN = r"\d{4}"

#: One safe path segment: no separator, no traversal, no leading dot, no control character.
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"

_SLUG = r"[a-z0-9][a-z0-9-]*"
_SHA1 = r"[0-9a-f]{40}"
_HTTPS = r"https://[^\s]+"

DatasetRole = Literal["project", "common"]
DatasetStatus = Literal["active", "complete", "archived"]
ComponentType = Literal["simulation", "shared-input", "analysis", "reference"]
ComponentStatus = Literal["active", "complete", "archived", "failed"]

#: Statuses at which a dataset stops being writable. Completed data are shared research
#: infrastructure: changing them retroactively changes every comparison already made against them.
READ_ONLY_STATUSES: frozenset[str] = frozenset({"complete", "archived"})


class DatasetError(ValueError):
    """A manifest violates the contract. The message names the field path."""


class StrictModel(BaseModel):
    """Unknown keys are an error, and values are frozen.

    Forbidding unknown keys is what keeps a typo from silently becoming metadata: a manifest with
    `dataset_nam:` would otherwise validate, and the misspelling would travel with the data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Person(StrictModel):
    """A scientific identity, not a machine login.

    Per-run operator and host details belong in the run records, where they are useful for
    debugging. A dataset manifest travels; it carries who is answerable for the science.
    """

    person_id: str = Field(pattern=_SLUG, min_length=2, max_length=128)
    name: str = Field(min_length=1, max_length=4096)
    orcid: str | None = Field(default=None, pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
    affiliation: str | None = Field(default=None, max_length=512)


class SourceRepository(StrictModel):
    """A repository pinned at an exact commit.

    Both fields are required together. Naming a repository without a commit records where to look
    but not what ran, which is the failure this contract exists to prevent.
    """

    repository: str = Field(pattern=_HTTPS, max_length=2048)
    commit: str = Field(pattern=_SHA1)
    version: str | None = Field(default=None, max_length=64)


class Component(StrictModel):
    """One method or asset directory inside a dataset.

    `cMD`, `REST2` and the prepared inputs they share are components of ONE dataset when they were
    generated as a single comparison suite -- not three datasets that happen to be adjacent.
    """

    name: str = Field(pattern=NAME_PATTERN, min_length=1, max_length=64)
    type: ComponentType
    method: str | None = Field(default=None, max_length=128)
    path: str = Field(min_length=1, max_length=4096)
    status: ComponentStatus
    linked: bool = Field(default=False)
    description: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _check(self) -> "Component":
        _check_relative("path", self.path)
        if self.type == "simulation" and not self.method:
            raise ValueError(
                "method: a simulation component must name the method that produced it (for "
                "example 'cMD' or 'REST2'); the directory name is not evidence")
        if self.linked and self.status == "active":
            raise ValueError(
                "status: a linked component points at content owned by another dataset and must "
                "not be 'active'; new output belongs in a component this dataset owns")
        return self


class Dataset(StrictModel):
    """The authoritative manifest at the root of one dataset directory.

    Canonical location, contract v2:

        $MD_DATA/{year}/{project_name}/{data_name}/dataset.yaml
        $MD_DATA/{year}/common/{project_name}/{data_name}/dataset.yaml
    """

    schema_version: str = Field(min_length=1, max_length=4096)
    dataset_id: str = Field(pattern=_SLUG, min_length=2, max_length=128)
    path: str = Field(min_length=1, max_length=4096)
    year: str = Field(pattern=YEAR_PATTERN)
    project_name: str = Field(pattern=NAME_PATTERN, min_length=1, max_length=64)
    data_name: str = Field(pattern=NAME_PATTERN, min_length=1, max_length=64)
    role: DatasetRole
    system: str = Field(min_length=1, max_length=4096)
    created_at: AwareDatetime
    created_by: Person
    status: DatasetStatus
    origin: SourceRepository
    software: SourceRepository | None = None
    components: list[Component] = Field(min_length=1)
    derived_from: list[str] = Field(default_factory=list)
    completed_at: AwareDatetime | None = None
    archived_at: AwareDatetime | None = None
    notes: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="before")
    @classmethod
    def _version(cls, data: Any) -> Any:
        """Refuse an unsupported schema_version before reporting anything else.

        A v1 manifest read by a v2 reader must fail on the version, not on the shape of a path.
        The first message a person sees should say which contract they have, not send them
        looking for a missing month segment.
        """
        if isinstance(data, dict):
            version = data.get("schema_version")
            if version is not None and version not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError(
                    f"schema_version: unsupported value {version!r} for the dataset contract; "
                    f"this build supports {', '.join(SUPPORTED_SCHEMA_VERSIONS)}. Contract v1 "
                    f"used a '{{namespace}}/{{yyyy-mm}}/{{dataset_name}}' path with a month "
                    f"segment; v2 is year-first with no month and is not a relaxation of v1.")
        return data

    @property
    def read_only(self) -> bool:
        return self.status in READ_ONLY_STATUSES

    @model_validator(mode="after")
    def _check(self) -> "Dataset":
        _check_relative("path", self.path)
        parts = self.path.split("/")

        # -- the path IS the identity, so every segment is checked against its own field ----
        expected = canonical_path(year=self.year, project_name=self.project_name,
                                  data_name=self.data_name, common=self.role == "common")
        if self.path != expected:
            raise ValueError(
                f"path: {self.path!r} does not match the canonical path for these fields, "
                f"{expected!r}. The path is derived from year, role, project_name and data_name; "
                f"it is never written independently of them.")
        if self.role == "common" and parts[1] != COMMON_SEGMENT:
            raise ValueError(f"path: a 'common' dataset lives under "
                             f"{{year}}/{COMMON_SEGMENT}/...; got {self.path!r}")
        if self.role == "project" and len(parts) > 1 and parts[1] == COMMON_SEGMENT:
            raise ValueError(
                f"role: the {COMMON_SEGMENT!r} segment is reserved for shared datasets; a "
                f"project-owned dataset must not be placed under it")

        # -- the year means ONE thing, and it is checked ------------------------------------
        #
        # v2 fixes an ambiguity in v1, where the dated segment was the CREATION month but the
        # data were often finished later. `year` here is the year the dataset was COMPLETED --
        # the year the data it contains became final -- because registration acts on finished
        # data and that is the date a reader is looking for. For a dataset still active there is
        # nothing completed yet, so the creation year is used and the rule says so.
        reference = self.completed_at or self.created_at
        if self.year != f"{reference.year:04d}":
            which = "completed_at" if self.completed_at else "created_at"
            raise ValueError(
                f"year: the path segment {self.year!r} must equal the year of {which} "
                f"({reference.isoformat()}, i.e. {reference.year:04d}). The year is the year the "
                f"data were completed; for a dataset that is still active it is the year it was "
                f"created. It is a fact about the data, not a label chosen at registration.")

        # -- status and its timestamps agree -------------------------------------------------
        if self.status == "complete" and self.completed_at is None:
            raise ValueError("completed_at: required when status is 'complete'")
        if self.status == "archived":
            if self.archived_at is None:
                raise ValueError("archived_at: required when status is 'archived'")
            if self.completed_at is None:
                raise ValueError("completed_at: an archived dataset was complete first; "
                                 "archiving is a storage move, not a substitute for completion")
        if self.status == "active":
            if self.completed_at is not None:
                raise ValueError("completed_at: must be absent while status is 'active'")
            if self.archived_at is not None:
                raise ValueError("archived_at: must be absent while status is 'active'")
        if self.status == "complete" and self.archived_at is not None:
            raise ValueError("archived_at: set status to 'archived' as well, or remove it")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at: must not precede created_at")
        if (self.archived_at is not None and self.completed_at is not None
                and self.archived_at < self.completed_at):
            raise ValueError("archived_at: must not precede completed_at")

        # -- components ----------------------------------------------------------------------
        names = [c.name for c in self.components]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"components: duplicate component name(s) {duplicates}")
        paths = [c.path.rstrip("/") for c in self.components]
        for outer in paths:
            for inner in paths:
                if outer != inner and (inner == outer or inner.startswith(outer + "/")):
                    raise ValueError(
                        f"components: {inner!r} is nested inside {outer!r}. Components partition "
                        f"the dataset; overlapping ones make the inventory ambiguous about which "
                        f"component owns a file.")
        if self.read_only:
            active = [c.name for c in self.components if c.status == "active"]
            if active:
                raise ValueError(
                    f"components: a {self.status!r} dataset cannot contain active components "
                    f"{active}; finish them or create a new dataset")
        if self.dataset_id in self.derived_from:
            raise ValueError("derived_from: a dataset must not be derived from itself")
        return self


def canonical_path(*, year: str, project_name: str, data_name: str, common: bool = False) -> str:
    """The canonical logical path, relative to $MD_DATA. No month segment.

    One function, used both to build a destination and to check a manifest, so the two cannot
    disagree about what canonical means.
    """
    for label, value in (("year", year), ("project_name", project_name),
                         ("data_name", data_name)):
        if not isinstance(value, str) or not value:
            raise DatasetError(f"{label}: must be a non-empty string")
    if not re.fullmatch(YEAR_PATTERN, year):
        raise DatasetError(f"year: {year!r} is not exactly four decimal digits")
    check_segment("project_name", project_name)
    check_data_path("data_name", data_name)
    if common:
        return f"{year}/{COMMON_SEGMENT}/{project_name}/{data_name}"
    return f"{year}/{project_name}/{data_name}"


def check_data_path(label: str, value: str) -> str:
    """A dataset name that MAY be several segments deep, each one validated on its own.

    `data_name` used to be a single segment, which made the whole hierarchy of a reference set --
    `2026-09/ALA/cMD-hot/run1` -- expressible only by flattening it into one name. A directory
    tree that can be browsed by system and by method is worth more than a name that can be
    globbed, and nothing about the guarantees depends on the depth.

    What does NOT change is what a segment may be. Every component goes through `check_segment`,
    so `..`, absolute forms, leading dots, control characters and the Windows separator are
    refused exactly as before -- the traversal this validation exists to stop is stopped at every
    level rather than at the first. Empty components (`a//b`, a leading or trailing slash) are
    refused too: they would collapse silently and two different names would resolve to one path.
    """
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{label}: must be a non-empty string")
    if value.startswith("/") or value.endswith("/"):
        raise DatasetError(
            f"{label}: {value!r} begins or ends with a separator. A dataset path is relative and "
            f"its components are named; an empty component would collapse and two different "
            f"names would land on one path.")
    parts = value.split("/")
    if any(not part for part in parts):
        raise DatasetError(f"{label}: {value!r} has an empty component")
    for index, part in enumerate(parts):
        check_segment(f"{label}[{index}]" if len(parts) > 1 else label, part)
    return value


def check_segment(label: str, value: str) -> str:
    """One safe path segment, or a refusal naming what is wrong with it.

    Rejects separators, `.` and `..`, absolute forms, leading dots, control characters and the
    Windows separator. A name that reaches the filesystem decides where data are written, so it is
    validated before it is joined to anything.
    """
    if value in (".", ".."):
        raise DatasetError(f"{label}: {value!r} is a traversal, not a name")
    if "/" in value or "\\" in value:
        raise DatasetError(f"{label}: {value!r} contains a path separator; it must be exactly one "
                           f"path segment")
    if value.startswith("."):
        raise DatasetError(f"{label}: {value!r} must not begin with a dot")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise DatasetError(f"{label}: {value!r} contains a control character")
    if not re.fullmatch(NAME_PATTERN, value):
        raise DatasetError(
            f"{label}: {value!r} is not a valid name. Use letters, digits, and any of . _ - "
            f"after a leading letter or digit.")
    if len(value) > 64:
        raise DatasetError(f"{label}: {value!r} is longer than 64 characters")
    return value


def _check_relative(label: str, value: str) -> None:
    """No absolute path, no traversal, no machine path.

    A manifest travels with the data. An absolute path in it is wrong on every machine except the
    one that wrote it, and leaks that machine's layout besides.
    """
    if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError(f"{label}: {value!r} is absolute; manifests carry paths relative to "
                         f"their declared root")
    if any(part == ".." for part in re.split(r"[\\/]", value)):
        raise ValueError(f"{label}: {value!r} traverses outside its root")


def validate_dataset(document: Any) -> Dataset:
    """Validate a manifest document, raising `DatasetError` with every violation listed."""
    try:
        return Dataset.model_validate(document)
    except ValidationError as exc:
        raise DatasetError(_format(exc)) from None


def validate_dataset_file(path: Path) -> Dataset:
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        raise DatasetError(f"{path}: no such manifest")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path}: not valid YAML -- {exc}") from None
    try:
        return validate_dataset(document)
    except DatasetError as exc:
        raise DatasetError(f"{path}: {exc}") from None


def _format(exc: ValidationError) -> str:
    lines = [f"{len(exc.errors())} contract violation(s)"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(document)"
        lines.append(f"  $.{location}: {error['msg']}")
    return "\n".join(lines)


def json_schema() -> dict[str, Any]:
    """The JSON schema, generated FROM the model above.

    Generated rather than maintained, so the published schema and the validator that actually runs
    cannot drift apart. A test regenerates it and compares.
    """
    schema = Dataset.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "MD-data dataset manifest"
    schema["$id"] = f"https://github.com/csy0000/MD-tools/schemas/dataset-v{CONTRACT_VERSION}.json"
    return schema
