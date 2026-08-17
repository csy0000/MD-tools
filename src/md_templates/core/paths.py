"""Repository-relative path normalisation, shared by the registry and by identity resolution.

Both need the same answer to "is this a legal reference to a file inside this repository?", and
identity is built from the *normalised* path, so the rule has to live in one place. If the registry
accepted `templates/rest2/../rest2/openmm/explicit-water/template.yaml` and identity resolution
accepted the collapsed form, two different strings would name one template and the identity would
stop being a function of the template.

The rule is deliberately strict rather than forgiving: nothing here collapses, resolves or repairs a
path. A path that is not already normalised is rejected, because repairing it would mean two spellings
map to one identity while a reviewer reading the registry sees only one of them.
"""
from __future__ import annotations

__all__ = ["PathError", "normalise_repo_relative", "TEMPLATES_ROOT", "DESCRIPTOR_FILENAME"]

TEMPLATES_ROOT = "templates"
DESCRIPTOR_FILENAME = "template.yaml"


class PathError(ValueError):
    """A repository-relative reference is absolute, escaping, or not normalised."""


def normalise_repo_relative(value: object, *, field: str) -> str:
    """Return `value` as a validated repository-relative POSIX path.

    Accepts only a string that is already normalised: relative, `/`-separated, with no empty, `.`
    or `..` components and no trailing separator. Everything else raises `PathError` naming the
    dotted field and the offending value.

    Platform independence is the point of rejecting rather than converting. `os.path.normpath`
    turns `a\\b` into one component on POSIX and two on Windows, so a registry validated on one
    platform would describe a different tree on the other.
    """
    if not isinstance(value, str):
        raise PathError(f"{field}: must be a string, got {type(value).__name__}")
    if not value:
        raise PathError(f"{field}: must not be empty")
    if "\\" in value:
        raise PathError(
            f"{field}: {value!r} contains a backslash. Repository paths are POSIX-separated so they "
            f"mean the same thing on every platform."
        )
    if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise PathError(f"{field}: {value!r} is absolute; repository-relative paths are required")
    if value.endswith("/"):
        raise PathError(f"{field}: {value!r} has a trailing separator")

    parts = value.split("/")
    for part in parts:
        if part == "":
            raise PathError(f"{field}: {value!r} has an empty path component")
        if part == ".":
            raise PathError(f"{field}: {value!r} has a '.' component; give the normalised path")
        if part == "..":
            raise PathError(
                f"{field}: {value!r} escapes the repository with '..'. Parent traversal is refused "
                f"even when it would resolve inside, because the identity is built from the literal "
                f"normalised path."
            )
    return value


def assert_under_templates(value: str, *, field: str) -> str:
    """A descriptor path must be `templates/<method>/<engine>/<variant>/template.yaml`."""
    normalise_repo_relative(value, field=field)
    parts = value.split("/")
    if parts[0] != TEMPLATES_ROOT:
        raise PathError(f"{field}: {value!r} must live under {TEMPLATES_ROOT}/")
    if parts[-1] != DESCRIPTOR_FILENAME:
        raise PathError(
            f"{field}: {value!r} must end in {DESCRIPTOR_FILENAME}. One filename means a consumer "
            f"never has to guess which file in a template directory is the descriptor."
        )
    if len(parts) != 5:
        raise PathError(
            f"{field}: {value!r} must have exactly the components "
            f"{TEMPLATES_ROOT}/<method>/<engine>/<variant>/{DESCRIPTOR_FILENAME}"
        )
    return value
