"""`extension.yaml`, contract version 2: continuing a simulation.

Ported from `csy0000/MD-data`, branch `protect-md-project-dev-test`, commit
`20d982eb463ed439095f1b95e00ff1b1d75906b4`, and adapted to the v2 year-first layout. Nothing here
imports `md_data`.

An extension record is **not** a second identity manifest. `dataset.yaml` remains authoritative for
what a dataset IS; `extension.yaml` records what was asked for and what happened.

Two modes, and the difference between them is whether the thing being extended is still writable:

`in-place`
    The dataset is still `active` and is being lengthened within itself. The record is written
    while it is active and STAYS there afterwards: once it reaches a terminal status it is the
    provenance of how the dataset reached its final length. It is never rewritten as a
    `new-dataset` record, because it did happen in place.

`new-dataset`
    The parent is `complete` or `archived` and therefore immutable, so a NEW dataset is created
    under the v2 layout and this record points back at the parent. The parent is never written to,
    and its manifest is never loaded by this validator -- it is named, not opened.

WHAT V2 ADDS

The instruction this port was written for requires one thing v1 did not record: if a runtime may
overwrite the checkpoint an extension restarts from, that checkpoint is COPIED and its hash is
recorded. A restart point that the run itself can rewrite is not a restart point -- the join
becomes unreproducible the moment the continuation advances past it, and nothing in the record
would show that it had.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AwareDatetime, Field, ValidationError, model_validator

from .model import (Dataset, DatasetError, Person, SourceRepository, StrictModel, _SHA1, _SLUG,
                    NAME_PATTERN, _check_relative, _format)

CONTRACT_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("2.0",)

EXTENSION_NAME = "extension.yaml"

ExtensionMode = Literal["in-place", "new-dataset"]
ExtensionStatus = Literal["requested", "running", "complete", "failed", "abandoned"]

#: Statuses that describe something finished rather than something in flight. A terminal record is
#: history, and history survives the dataset completing.
TERMINAL_STATUSES: frozenset[str] = frozenset({"complete", "failed", "abandoned"})

_SHA256 = r"[0-9a-f]{64}"


class ExtensionTarget(StrictModel):
    """The dataset and component whose simulation is being continued."""

    dataset_id: str = Field(pattern=_SLUG, min_length=2, max_length=128)
    component: str = Field(pattern=NAME_PATTERN, min_length=1, max_length=64)
    source_checkpoint: str = Field(min_length=1, max_length=4096)
    #: The digest of the checkpoint the continuation actually restarted from.
    #:
    #: Required, and required to be of the COPY when one was made. A hash of a file a running
    #: simulation may overwrite describes the join point only until the run passes it.
    source_checkpoint_sha256: str = Field(pattern=_SHA256)
    #: True when the checkpoint was copied out of the parent before the run started. It must be
    #: true whenever the parent component is writable, which for an in-place extension it always
    #: is -- the run is writing into that very component.
    source_checkpoint_copied: bool = False
    original_length_ns: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> "ExtensionTarget":
        _check_relative("source_checkpoint", self.source_checkpoint)
        return self


class Extension(StrictModel):
    """One continuation request and its outcome, beside the `dataset.yaml` it belongs to.

    Timestamp semantics by status:

        requested   no outcome fields at all
        running     started_at set, completed_at absent
        complete    started_at, completed_at and final_length_ns all set
        failed      optional; completed_at means WHEN THE ATTEMPT STOPPED and requires
        abandoned   started_at. final_length_ns records partial output if any exists.
    """

    schema_version: str = Field(min_length=1, max_length=4096)
    mode: ExtensionMode
    dataset_id: str = Field(pattern=_SLUG, min_length=2, max_length=128)
    target: ExtensionTarget
    output_component: str = Field(pattern=NAME_PATTERN, min_length=1, max_length=64)
    additional_length_ns: float = Field(gt=0)
    reason: str = Field(min_length=1, max_length=4096)
    requested_by: Person
    requested_at: AwareDatetime
    status: ExtensionStatus
    generation: SourceRepository
    combined_length_ns: float | None = Field(default=None, gt=0)
    restart_step: int | None = Field(default=None, ge=0)
    restart_time_ps: float | None = Field(default=None, ge=0)
    #: Components of THIS dataset that are symlinks into the parent. Listed so the output component
    #: can be proven not to be one of them.
    parent_links: list[str] = Field(default_factory=list)
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    final_length_ns: float | None = Field(default=None, gt=0)
    notes: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="before")
    @classmethod
    def _version(cls, data: Any) -> Any:
        if isinstance(data, dict):
            version = data.get("schema_version")
            if version is not None and version not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError(
                    f"schema_version: unsupported value {version!r} for the extension contract; "
                    f"this build supports {', '.join(SUPPORTED_SCHEMA_VERSIONS)}.")
        return data

    @model_validator(mode="after")
    def _check(self) -> "Extension":
        if self.mode == "in-place":
            if self.target.dataset_id != self.dataset_id:
                raise ValueError(
                    f"target.dataset_id: an in-place extension continues the dataset it lives in; "
                    f"got {self.target.dataset_id!r} inside {self.dataset_id!r}. Use mode "
                    f"'new-dataset' to extend a different dataset.")
            for field in ("combined_length_ns", "restart_step", "restart_time_ps"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"{field}: only a 'new-dataset' extension records it; an in-place "
                        f"continuation has no separate parent length to combine")
            # The run is writing into the very component it restarted from, so the checkpoint it
            # restarted from is one the run can overwrite. It must have been copied.
            if not self.target.source_checkpoint_copied:
                raise ValueError(
                    "target.source_checkpoint_copied: an in-place extension writes into the "
                    "component it restarted from, so the checkpoint is one the run itself may "
                    "overwrite. Copy it before starting and record the copy's hash, or the join "
                    "point stops being reproducible the moment the run passes it.")
        else:
            if self.target.dataset_id == self.dataset_id:
                raise ValueError(
                    "target.dataset_id: a 'new-dataset' extension points at its parent, which "
                    "must be a different dataset than the one this record lives in")
            if self.combined_length_ns is None:
                raise ValueError(
                    "combined_length_ns: required so a reader can tell how long the extended "
                    "trajectory is without opening either dataset")
            if self.restart_step is None:
                raise ValueError(
                    "restart_step: required; it is what makes the join point reproducible")
            expected = self.target.original_length_ns + self.additional_length_ns
            if abs(self.combined_length_ns - expected) > 1e-6 * max(1.0, expected):
                raise ValueError(
                    f"combined_length_ns: {self.combined_length_ns} does not equal "
                    f"original_length_ns + additional_length_ns ({expected})")

        if self.output_component in self.parent_links:
            raise ValueError(
                f"output_component: {self.output_component!r} is listed in parent_links, so it "
                f"points into the parent dataset. New output must go to a component this dataset "
                f"owns.")

        if self.status == "requested":
            for field in ("started_at", "completed_at", "final_length_ns"):
                if getattr(self, field) is not None:
                    raise ValueError(f"{field}: must be absent while status is 'requested'")
        if self.status == "running":
            if self.started_at is None:
                raise ValueError("started_at: required when status is 'running'")
            if self.completed_at is not None:
                raise ValueError("completed_at: must be absent while status is 'running'")
        if self.status == "complete":
            for field in ("started_at", "completed_at", "final_length_ns"):
                if getattr(self, field) is None:
                    raise ValueError(f"{field}: required when status is 'complete'")
        if self.status in ("failed", "abandoned"):
            # Everything is optional: an attempt may be abandoned before it ever ran. The one
            # thing that cannot happen is stopping something that never started.
            if self.completed_at is not None and self.started_at is None:
                raise ValueError(
                    f"started_at: required when completed_at records when a {self.status!r} "
                    f"attempt stopped")

        if self.started_at is not None and self.started_at < self.requested_at:
            raise ValueError("started_at: must not precede requested_at")
        if (self.completed_at is not None and self.started_at is not None
                and self.completed_at < self.started_at):
            raise ValueError("completed_at: must not precede started_at")
        if self.final_length_ns is not None:
            expected = self.target.original_length_ns + self.additional_length_ns
            if self.mode == "new-dataset" and self.final_length_ns > expected + 1e-6 * max(1.0, expected):
                raise ValueError(
                    f"final_length_ns: {self.final_length_ns} exceeds the requested combined "
                    f"length ({expected}); an extension cannot produce more than it asked for")
        return self


def check_dataset_extension(dataset: Dataset, extension: Extension) -> list[str]:
    """Checks that only make sense with a dataset and its extension record together.

    The parent of a `new-dataset` extension is deliberately NOT loaded: it is named by
    `target.dataset_id`, and opening it would make validation depend on another dataset being
    present and unchanged.
    """
    problems: list[str] = []
    names = {c.name: c for c in dataset.components}

    if extension.dataset_id != dataset.dataset_id:
        problems.append(
            f"$.dataset_id: {extension.dataset_id!r} does not match the dataset.yaml beside it "
            f"({dataset.dataset_id!r}); extension.yaml belongs to exactly one dataset")

    output = names.get(extension.output_component)
    if output is None:
        problems.append(
            f"$.output_component: {extension.output_component!r} is not a component of "
            f"{dataset.dataset_id!r}; declare it in dataset.yaml first")
    elif output.linked:
        problems.append(
            f"$.output_component: component {output.name!r} is linked, so writing to it would "
            f"write into the dataset that owns it. New output needs a component this dataset owns.")

    for index, link in enumerate(extension.parent_links):
        component = names.get(link)
        if component is None:
            problems.append(
                f"$.parent_links[{index}]: {link!r} is not a component of {dataset.dataset_id!r}")
        elif not component.linked:
            problems.append(
                f"$.parent_links[{index}]: component {link!r} is not marked 'linked: true' in "
                f"dataset.yaml, so it is not an alias into the parent")

    # Output cannot have been finished after the thing holding it was declared finished.
    if (extension.status == "complete" and extension.completed_at is not None
            and dataset.completed_at is not None
            and extension.completed_at > dataset.completed_at):
        problems.append(
            f"$.completed_at: the extension finished at {extension.completed_at.isoformat()}, "
            f"after the dataset was declared complete at {dataset.completed_at.isoformat()}. "
            f"Output cannot be completed after the dataset containing it was declared complete.")

    if extension.mode == "in-place":
        # A terminal in-place record is history and survives the dataset completing. What a
        # completed dataset must NOT carry is an extension still in flight, because that asserts
        # writing into finished data.
        if dataset.read_only and extension.status not in TERMINAL_STATUSES:
            problems.append(
                f"$.status: an in-place extension is {extension.status!r} in a "
                f"{dataset.status!r} dataset, which asserts that output is still being written "
                f"into data declared immutable. Finish or abandon it, or extend into a new "
                f"dataset.")
    else:
        if extension.target.dataset_id not in dataset.derived_from:
            problems.append(
                f"$.target.dataset_id: {extension.target.dataset_id!r} is not listed in the "
                f"dataset's derived_from, so the manifest does not record where this data came "
                f"from")
    return problems


def validate_extension(document: Any) -> Extension:
    try:
        return Extension.model_validate(document)
    except ValidationError as exc:
        raise DatasetError(_format(exc)) from None


def validate_extension_file(path: Path) -> Extension:
    path = Path(path)
    if path.is_dir():
        path = path / EXTENSION_NAME
    if not path.is_file():
        raise DatasetError(f"{path}: no such extension record")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path}: not valid YAML -- {exc}") from None
    try:
        return validate_extension(document)
    except DatasetError as exc:
        raise DatasetError(f"{path}: {exc}") from None


def json_schema() -> dict[str, Any]:
    """Generated FROM the model above, never maintained by hand."""
    schema = Extension.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "MD-data extension record"
    schema["$id"] = (f"https://github.com/csy0000/MD-tools/schemas/"
                     f"extension-v{CONTRACT_VERSION}.json")
    return schema
