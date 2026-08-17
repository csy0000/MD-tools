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

import os
from pathlib import Path

__all__ = ["PathError", "normalise_repo_relative", "resolve_within_repository",
           "TEMPLATES_ROOT", "DESCRIPTOR_FILENAME"]

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


def resolve_within_repository(root: Path, relative: str, *, field: str) -> Path:
    """Return `root/relative`, refusing any symlink along the way.

    String normalisation is not enough. A committed `template.yaml` can be a *symlink* to bytes
    outside the checkout: `Path.is_file()`, `read_text()` and `exists()` all follow it silently,
    Git reports the tree clean because the link itself is unchanged, and identity resolution then
    hands back `SHA + path` for content the commit does not contain. The identity would name bytes
    nobody can reproduce, which is the one thing it exists to prevent.

    Every component below `root` is checked, not just the last: a symlinked *directory* redirects
    everything beneath it just as effectively as a symlinked file.

    Symlinks are refused outright rather than resolved-and-permitted, including links that stay
    inside the repository. Allowing them would mean deciding, per link, whether the target is both
    inside the tree and represented by the same commit -- a judgement that is easy to get subtly
    wrong and that buys nothing the catalog needs. A template directory holding real files is not a
    hardship.

    `root` itself is not component-checked, since a repository legitimately sits under a symlinked
    parent (`/tmp` on macOS, a symlinked home). Containment is verified against its resolved form
    instead, which covers that case correctly.
    """
    root = Path(root)
    normalise_repo_relative(relative, field=field)

    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise PathError(
                f"{field}: {relative!r} passes through a symlink at {part!r} ({current} -> "
                f"{os.readlink(current)}). Symlinked components are refused: they let a clean "
                f"checkout parse bytes the commit does not contain, so the commit SHA would no "
                f"longer identify what was read."
            )

    # Belt and braces. With no symlinked component this holds by construction; it also catches a
    # link swapped in between the walk above and the read that follows.
    resolved_root = os.path.realpath(root)
    resolved = os.path.realpath(current)
    if resolved != resolved_root and not resolved.startswith(resolved_root + os.sep):
        raise PathError(
            f"{field}: {relative!r} resolves to {resolved}, which is outside the repository at "
            f"{resolved_root}"
        )
    return current


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
