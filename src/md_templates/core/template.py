"""The strict typed template descriptor.

A `template.yaml` describes what a template *is* -- which method, which engine, which implementation
currently provides it, what it can do, which persistent schemas it reads and writes, and how far it
has actually been validated. It is metadata. Loading or validating a descriptor changes no runtime
behaviour in PR 1: nothing dispatches through these files yet, and `implementation.dispatch` records
that in the descriptor itself rather than leaving a reader to assume otherwise.

Two design decisions are worth stating because they are what make the file useful rather than
decorative.

**Implementation status and scientific status are separate fields with separate enums.** They are
different claims and they move independently. "The code runs, is covered by tests, and produces
bundles that relocate and resume" says nothing about whether the physics has been validated for
production use. Collapsing them into one "status" is how a template that merely executes comes to be
described as validated.

**Evidence carries its own scope, and the model refuses to let weak evidence raise a status.** A CPU
smoke profile is picoseconds of unvalidated settings and can never be scientific evidence; a
system-specific pilot result is evidence about that system, not about the general method. Both rules
are enforced by validators rather than left to the honesty of whoever edits the YAML next.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .paths import PathError, normalise_repo_relative

__all__ = [
    "TEMPLATE_SCHEMA_VERSION", "TemplateError", "TemplateDescriptor",
    "IMPLEMENTATION_STATUSES", "SCIENTIFIC_STATUSES", "EVIDENCE_KINDS", "EVIDENCE_SCOPES",
]

#: Independent of the bundle, canonical-configuration and run-state schema versions. A descriptor
#: gaining a field is not a reason to invalidate a bundle, and a bundle format change is not a
#: reason to rewrite every descriptor.
TEMPLATE_SCHEMA_VERSION = 1

IMPLEMENTATION_STATUSES = ("planned", "implemented-untested", "implemented-and-tested")
SCIENTIFIC_STATUSES = ("unvalidated", "pilot-supported", "validated")
EVIDENCE_KINDS = ("cpu-smoke", "regression", "pilot", "scientific-validation")
EVIDENCE_SCOPES = ("general", "system-specific", "example")


class TemplateError(ValueError):
    """A descriptor is malformed, internally inconsistent, or claims unearned validation."""


class Strict(BaseModel):
    """Unknown keys are an error at every level.

    Same convention as the canonical configuration models, and for the same reason: a descriptor
    that silently ignores a misspelled key describes something other than what its author wrote.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


_IDENT = "identifiers are lowercase alphanumerics separated by single '-' or '.' characters"


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: must be a non-empty string")
    if value != value.strip() or value != value.lower():
        raise ValueError(f"{field}: {value!r} must be lowercase and unpadded; {_IDENT}")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-.")
    if not set(value) <= allowed or value[0] in "-." or value[-1] in "-.":
        raise ValueError(f"{field}: {value!r} is not a normalised identifier; {_IDENT}")
    return value


# ------------------------------------------------------------------------------------------------
# blocks
# ------------------------------------------------------------------------------------------------

class MethodBlock(Strict):
    """Which method this template implements, and the method contract's version."""

    id: str
    api_version: int = Field(ge=1)
    aliases: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v):
        return _identifier(v, "method.id")

    @field_validator("aliases")
    @classmethod
    def _check_aliases(cls, v):
        return [_identifier(a, "method.aliases[]") for a in v]


class EngineBlock(Strict):
    """Which engine executes it, and the engine contract's version."""

    id: str
    api_version: int = Field(ge=1)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v):
        return _identifier(v, "engine.id")


class IdentityBlock(Strict):
    """How this template is identified. One scheme is supported and it is named explicitly."""

    mode: Literal["git-commit-plus-template-path"]


class ImplementationBlock(Strict):
    """What actually provides this template today.

    `dispatch` is the honest field. In PR 1 it is `legacy-direct` everywhere: the descriptors exist,
    but execution still goes through the current `md-openmm` command and the current Python
    namespace, and nothing routes through the catalog. A reader must be able to learn that from the
    descriptor rather than from a journal entry.
    """

    binding: Literal["current-openmm-implementation"]
    dispatch: Literal["legacy-direct", "catalog"]
    python_namespace: str
    cli_command: str
    notes: list[str] = Field(default_factory=list)


class InputRoute(Strict):
    """A declared input route. The route is never inferred from a file's contents."""

    id: str
    description: str

    @field_validator("id")
    @classmethod
    def _check_id(cls, v):
        return _identifier(v, "input_routes[].id")


class Capability(Strict):
    """One operational capability, with its support state stated rather than implied."""

    id: str
    supported: bool
    description: str

    @field_validator("id")
    @classmethod
    def _check_id(cls, v):
        return _identifier(v, "capabilities[].id")


class MethodFeature(Strict):
    """A method-specific switch: whether it is supported, its default, and whether it can be turned off.

    This is metadata about runtime behaviour, never a control over it. Recording that REST2 omega
    exclusion defaults to enabled does not enable it; the runtime default lives in the profiles and
    in `md_templates.openmm.config.DEFAULTS`, and PR 1 changes neither.
    """

    id: str
    supported: bool
    default_enabled: bool
    disableable: bool
    description: str

    @field_validator("id")
    @classmethod
    def _check_id(cls, v):
        return _identifier(v, "method_features[].id")


class BundleSchemaBlock(Strict):
    """Which persistent bundle schema versions this template reads and writes."""

    readable: list[int]
    writable: list[int]

    @model_validator(mode="after")
    def _check(self):
        if not self.readable or not self.writable:
            raise ValueError("bundle_schema: readable and writable must both be non-empty")
        missing = sorted(set(self.writable) - set(self.readable))
        if missing:
            raise ValueError(
                f"bundle_schema: version(s) {missing} are writable but not readable, so this "
                f"template would produce bundles it cannot itself validate"
            )
        return self


class RestartBlock(Strict):
    """The continuation guarantees, and the authority a resume trusts."""

    resumable: bool
    authority: Literal["committed-generation-record"]
    guarantees: list[str]
    limitations: list[str] = Field(default_factory=list)


class ProfileProviderBlock(Strict):
    """Where the scientific default profiles currently come from.

    Deliberately named as a *provider* reference rather than a template-local resource list.
    The profile files live inside the current OpenMM implementation package and PR 1 does not move
    or copy them; pretending they were template-local would make the descriptor describe a layout
    that does not exist. `template_local: false` is the field that keeps that honest, and template-
    local profile migration is deferred to the method-activation PRs.
    """

    provider: Literal["current-openmm-implementation"]
    template_local: bool
    python_resource: str
    profile_ids: list[str]
    note: str

    @model_validator(mode="after")
    def _check(self):
        if self.template_local:
            raise ValueError(
                "profiles.template_local must be false until profiles actually move into the "
                "template directory; PR 1 does not move packaged resources"
            )
        return self


class Evidence(Strict):
    """One piece of evidence, with an explicit scope and kind.

    `establishes_general_scientific_validation` is the only field that can raise a template's
    scientific status, and two validators bound what may set it.
    """

    id: str
    description: str
    kind: Literal[EVIDENCE_KINDS]  # type: ignore[valid-type]
    scope: Literal[EVIDENCE_SCOPES]  # type: ignore[valid-type]
    establishes_general_scientific_validation: bool = False

    @field_validator("id")
    @classmethod
    def _check_id(cls, v):
        return _identifier(v, "validation.evidence[].id")

    @model_validator(mode="after")
    def _weak_evidence_cannot_validate(self):
        if not self.establishes_general_scientific_validation:
            return self
        if self.kind == "cpu-smoke":
            raise ValueError(
                f"validation.evidence[{self.id!r}]: a CPU smoke profile is picoseconds of "
                f"unvalidated settings and is prohibited as scientific evidence"
            )
        if self.kind == "regression":
            raise ValueError(
                f"validation.evidence[{self.id!r}]: regression tests show the implementation did "
                f"not change; they are not scientific validation"
            )
        if self.scope != "general":
            raise ValueError(
                f"validation.evidence[{self.id!r}]: scope {self.scope!r} evidence is about that "
                f"system or example only and cannot establish general validation of the method"
            )
        return self


class ValidationBlock(Strict):
    """Implementation status and scientific status, kept apart, each with its own enum."""

    implementation_status: Literal[IMPLEMENTATION_STATUSES]  # type: ignore[valid-type]
    scientific_status: Literal[SCIENTIFIC_STATUSES]  # type: ignore[valid-type]
    open_gates: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_must_be_earned(self):
        ids = [e.id for e in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("validation.evidence: duplicate evidence id")
        general = [e for e in self.evidence if e.establishes_general_scientific_validation]
        if self.scientific_status == "validated" and not general:
            raise ValueError(
                "validation.scientific_status is 'validated' but no evidence entry establishes "
                "general scientific validation. Running is not validation."
            )
        if self.scientific_status == "pilot-supported" and not any(
                e.kind == "pilot" for e in self.evidence):
            raise ValueError(
                "validation.scientific_status is 'pilot-supported' but no pilot evidence is listed"
            )
        return self


# ------------------------------------------------------------------------------------------------
# the descriptor
# ------------------------------------------------------------------------------------------------

class TemplateDescriptor(Strict):
    """A complete `template.yaml`."""

    schema_version: int
    template_id: str
    display_name: str
    summary: str
    method: MethodBlock
    engine: EngineBlock
    identity: IdentityBlock
    implementation: ImplementationBlock
    input_routes: list[InputRoute]
    capabilities: list[Capability]
    method_features: list[MethodFeature] = Field(default_factory=list)
    bundle_schema: BundleSchemaBlock
    restart: RestartBlock
    profiles: ProfileProviderBlock
    validation: ValidationBlock
    repository_references: list[str] = Field(default_factory=list)

    # `mode="before"`: see the identical note in registry.py. A schema version is a contract
    # identifier, and "1" is not the integer 1.
    @field_validator("schema_version", mode="before")
    @classmethod
    def _check_schema_version(cls, v):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"schema_version: must be an integer, got {v!r}")
        if v != TEMPLATE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version: {v!r} is not supported; this build understands "
                f"{TEMPLATE_SCHEMA_VERSION}. The template schema version is independent of the "
                f"bundle, canonical-configuration and run-state schema versions."
            )
        return v

    @field_validator("repository_references")
    @classmethod
    def _check_references(cls, v):
        out = []
        for i, ref in enumerate(v):
            try:
                out.append(normalise_repo_relative(ref, field=f"repository_references[{i}]"))
            except PathError as exc:
                raise ValueError(str(exc)) from exc
        return out

    @model_validator(mode="after")
    def _internally_consistent(self):
        expected = f"{self.method.id}/{self.engine.id}"
        if not self.template_id.startswith(expected + "/"):
            raise ValueError(
                f"template_id: {self.template_id!r} must begin with "
                f"{expected!r}, matching the declared method and engine"
            )
        _identifier(self.template_id.split("/")[-1], "template_id (variant component)")
        if not self.input_routes:
            raise ValueError("input_routes: at least one declared route is required")
        for name, items in (("input_routes", self.input_routes),
                            ("capabilities", self.capabilities),
                            ("method_features", self.method_features)):
            ids = [x.id for x in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name}: duplicate id")
        return self

    def feature(self, feature_id: str) -> Optional[MethodFeature]:
        for f in self.method_features:
            if f.id == feature_id:
                return f
        return None


def parse_descriptor(document: object, *, origin: str = "<descriptor>") -> TemplateDescriptor:
    """Validate one already-loaded descriptor document."""
    if not isinstance(document, dict):
        raise TemplateError(f"{origin}: expected a mapping, got {type(document).__name__}")
    try:
        return TemplateDescriptor.model_validate(document)
    except Exception as exc:  # pydantic ValidationError, or a validator's ValueError
        raise TemplateError(f"{origin}: {exc}") from exc
