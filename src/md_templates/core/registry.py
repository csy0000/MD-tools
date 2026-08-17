"""The root registry: a minimal discovery index over the templates in this repository.

`registry.yaml` answers one question -- which templates exist, and where is each one's descriptor --
and deliberately answers nothing else. Settings, profiles, scientific defaults and validation
evidence live in the descriptors. A registry that duplicated them would be a second copy of the
truth, and the two copies would disagree the first time someone edited one.

**The registry never records a commit SHA.** Its content has to stay valid at every later commit,
with identity resolved from whichever immutable commit contains the file -- see `identity.py`. A SHA
written into `registry.yaml` would be stale the moment the file was committed, since committing it
changes the commit.

Loading is engine-neutral by construction: this module imports YAML and pydantic and nothing else,
so listing and validating templates works in a minimal environment with no OpenMM, no OpenFF and no
RDKit installed. That is the property that lets a future catalog CLI enumerate templates on a
machine that cannot run any of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .paths import PathError, assert_under_templates, normalise_repo_relative
from .template import TemplateDescriptor, TemplateError, parse_descriptor

__all__ = [
    "REGISTRY_SCHEMA_VERSION", "REGISTRY_FILENAME", "CANONICAL_FORM",
    "RegistryError", "RegistryEntry", "Registry", "TemplateCatalog", "load_catalog",
    "load_registry_document",
]

#: Independent of every other schema version in the repository.
REGISTRY_SCHEMA_VERSION = 1
REGISTRY_FILENAME = "registry.yaml"

#: The one identity form this registry declares. Stored as a literal so a descriptor consumer can
#: check that the registry it loaded means what it thinks it means.
CANONICAL_FORM = "{canonical_url}@{full_commit_sha}#{template_path}"


class RegistryError(ValueError):
    """The registry is malformed, inconsistent with the tree, or contains duplicates."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: must be a non-empty string")
    if value != value.strip() or value != value.lower():
        raise ValueError(f"{field}: {value!r} must be lowercase and unpadded")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-.")
    if not set(value) <= allowed or value[0] in "-." or value[-1] in "-.":
        raise ValueError(f"{field}: {value!r} is not a normalised identifier")
    return value


class RepositoryBlock(_Strict):
    canonical_url: str

    @field_validator("canonical_url")
    @classmethod
    def _check(cls, v):
        text = v.strip() if isinstance(v, str) else v
        if not isinstance(text, str) or not text.startswith("https://"):
            raise ValueError(f"repository.canonical_url: {v!r} must be an https URL")
        if text.endswith("/") or text.endswith(".git"):
            raise ValueError(
                f"repository.canonical_url: {v!r} must be the bare canonical form, with no trailing "
                f"'/' or '.git', so one repository has exactly one spelling inside an identity"
            )
        return text


class IdentityBlock(_Strict):
    scheme: str
    canonical_form: str

    @field_validator("scheme")
    @classmethod
    def _check_scheme(cls, v):
        if v != "git-commit-plus-template-path":
            raise ValueError(
                f"identity.scheme: {v!r} is not supported. The only immutable identity is a full "
                f"Git commit SHA plus the exact template path."
            )
        return v

    @field_validator("canonical_form")
    @classmethod
    def _check_form(cls, v):
        if v != CANONICAL_FORM:
            raise ValueError(f"identity.canonical_form: expected {CANONICAL_FORM!r}, got {v!r}")
        return v


class RegistryEntry(_Strict):
    """One template's discovery record."""

    template_id: str
    template_path: str
    method: str
    engine: str
    method_aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("method", "engine")
    @classmethod
    def _check_ids(cls, v, info):
        return _identifier(v, f"templates[].{info.field_name}")

    @field_validator("method_aliases", "tags")
    @classmethod
    def _check_lists(cls, v, info):
        out = [_identifier(x, f"templates[].{info.field_name}[]") for x in v]
        if len(set(out)) != len(out):
            raise ValueError(f"templates[].{info.field_name}: duplicate entry")
        return out

    @field_validator("template_path")
    @classmethod
    def _check_path(cls, v):
        try:
            return assert_under_templates(v, field="templates[].template_path")
        except PathError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("template_id")
    @classmethod
    def _check_template_id(cls, v):
        if not isinstance(v, str) or not v:
            raise ValueError("templates[].template_id: must be a non-empty string")
        parts = v.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"templates[].template_id: {v!r} must be '<method>/<engine>/<variant>', matching "
                f"the descriptor's directory below templates/"
            )
        for part in parts:
            _identifier(part, "templates[].template_id component")
        return v


class Registry(_Strict):
    """The parsed `registry.yaml`, before it has been checked against the tree."""

    schema_version: int
    repository: RepositoryBlock
    identity: IdentityBlock
    templates: list[RegistryEntry]

    # `mode="before"` on purpose: pydantic's lax coercion would turn "1", 1.0 and True into the
    # integer 1, so a registry declaring a string or a boolean version would validate. A schema
    # version is a discrete contract identifier, not a number to be converted into.
    @field_validator("schema_version", mode="before")
    @classmethod
    def _check_version(cls, v):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"schema_version: must be an integer, got {v!r}")
        if v != REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version: {v!r} is not supported; this build understands "
                f"{REGISTRY_SCHEMA_VERSION}"
            )
        return v

    @field_validator("templates")
    @classmethod
    def _check_templates(cls, v):
        if not v:
            raise ValueError("templates: at least one template is required")
        return v


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------

def load_registry_document(path: Path) -> dict:
    """Read one registry document with the safe YAML loader. No `eval`, no arbitrary tags."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise RegistryError(f"{path}: registry is empty")
    if not isinstance(data, dict):
        raise RegistryError(f"{path}: expected a mapping at the top level, got "
                            f"{type(data).__name__}")
    return data


@dataclass(frozen=True)
class TemplateCatalog:
    """A validated registry bound to the tree it describes."""

    root: Path
    registry: Registry
    descriptors: dict

    @property
    def canonical_url(self) -> str:
        return self.registry.repository.canonical_url

    @property
    def entries(self) -> list[RegistryEntry]:
        return list(self.registry.templates)

    def get(self, ref: str) -> Optional[RegistryEntry]:
        """Look one template up by `template_id` or by `template_path`."""
        for entry in self.registry.templates:
            if ref == entry.template_id or ref == entry.template_path:
                return entry
        return None

    def require(self, ref: str) -> RegistryEntry:
        entry = self.get(ref)
        if entry is None:
            from .identity import UnknownTemplateError

            known = sorted(e.template_id for e in self.registry.templates)
            raise UnknownTemplateError(
                f"{ref!r} is not a registered template. An identity is only ever minted for a "
                f"template the registry lists. Known: {known}"
            )
        return entry

    def descriptor(self, ref: str) -> TemplateDescriptor:
        return self.descriptors[self.require(ref).template_id]


def _check_entry_against_tree(root: Path, entry: RegistryEntry) -> TemplateDescriptor:
    parts = entry.template_path.split("/")
    _, method_dir, engine_dir, variant_dir, _ = parts

    if method_dir != entry.method:
        raise RegistryError(
            f"templates[{entry.template_id!r}].template_path: first component below templates/ is "
            f"{method_dir!r} but the declared method is {entry.method!r}"
        )
    if engine_dir != entry.engine:
        raise RegistryError(
            f"templates[{entry.template_id!r}].template_path: second component below templates/ is "
            f"{engine_dir!r} but the declared engine is {entry.engine!r}"
        )
    directory = f"{method_dir}/{engine_dir}/{variant_dir}"
    if entry.template_id != directory:
        raise RegistryError(
            f"templates[].template_id: {entry.template_id!r} must equal the descriptor directory "
            f"relative to templates/, which is {directory!r}"
        )

    path = root / entry.template_path
    if not path.is_file():
        raise RegistryError(
            f"templates[{entry.template_id!r}].template_path: {entry.template_path} does not exist "
            f"under {root}. A registry entry with no descriptor is a broken index, not a plan."
        )

    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        descriptor = parse_descriptor(document, origin=entry.template_path)
    except TemplateError as exc:
        raise RegistryError(str(exc)) from exc

    if descriptor.template_id != entry.template_id:
        raise RegistryError(
            f"{entry.template_path}: descriptor template_id {descriptor.template_id!r} disagrees "
            f"with the registry entry {entry.template_id!r}"
        )
    if descriptor.method.id != entry.method:
        raise RegistryError(
            f"{entry.template_path}: descriptor method {descriptor.method.id!r} disagrees with the "
            f"registry entry {entry.method!r}"
        )
    if descriptor.engine.id != entry.engine:
        raise RegistryError(
            f"{entry.template_path}: descriptor engine {descriptor.engine.id!r} disagrees with the "
            f"registry entry {entry.engine!r}"
        )
    for ref in descriptor.repository_references:
        if not (root / ref).exists():
            raise RegistryError(
                f"{entry.template_path}: repository_references entry {ref!r} does not exist"
            )
    return descriptor


def load_catalog(root: Path, *, registry_filename: str = REGISTRY_FILENAME) -> TemplateCatalog:
    """Load and fully validate the catalog rooted at `root`.

    Validation covers the registry's own schema, duplicate IDs and paths, path normalisation, the
    descriptor's existence and schema, and the agreement between the two -- because a registry that
    validates while disagreeing with the descriptor it points at is worse than one that fails.
    """
    root = Path(root)
    document = load_registry_document(root / registry_filename)
    try:
        registry = Registry.model_validate(document)
    except Exception as exc:
        raise RegistryError(f"{root / registry_filename}: {exc}") from exc

    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    for i, entry in enumerate(registry.templates):
        if entry.template_id in seen_ids:
            raise RegistryError(
                f"templates[{i}].template_id: {entry.template_id!r} duplicates templates"
                f"[{seen_ids[entry.template_id]}]"
            )
        if entry.template_path in seen_paths:
            raise RegistryError(
                f"templates[{i}].template_path: {entry.template_path!r} duplicates templates"
                f"[{seen_paths[entry.template_path]}]"
            )
        seen_ids[entry.template_id] = i
        seen_paths[entry.template_path] = i

    descriptors = {e.template_id: _check_entry_against_tree(root, e) for e in registry.templates}
    return TemplateCatalog(root=root, registry=registry, descriptors=descriptors)


def normalise_reference(value: str, *, field: str = "reference") -> str:
    """Exported for callers that need the same path rule without loading a catalog."""
    try:
        return normalise_repo_relative(value, field=field)
    except PathError as exc:
        raise RegistryError(str(exc)) from exc
