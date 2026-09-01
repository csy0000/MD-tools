"""Where the user's identity and their managed storage root come from.

Never the installed package, never the source repository, never `site-packages`. Those locations
may not exist, may be read-only, are shared between projects, and would mix one machine's paths
with files meant to be distributable. A wheel is not a place to keep somebody's storage root.

Precedence, highest first:

  1. an explicit `--user-config PATH`;
  2. the `MD_TOOLS_CONFIG` environment variable;
  3. `${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config`.

The managed storage root is resolved separately and on its own precedence, because it changes far
more often than an identity does:

  1. an explicit CLI override;
  2. the `MD_DATA` environment variable;
  3. `machine.md_data` from the user configuration.

Which source supplied the root is always printed. A registration that writes to the wrong root is
discovered much later, and by then the data are somewhere nobody looks.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import yaml

from .errors import RegistrationError

CONFIG_SCHEMA_VERSION = "1.0"
CONFIG_BASENAME = "user.config"

EXAMPLE = """\
# MD-tools user configuration.
#
# This file is YAML, despite the .config suffix. It holds who you are scientifically and where
# your managed storage root is. It is NOT part of the installed package and must never be checked
# into a repository: it is specific to one person on one machine.
#
# Location, highest precedence first:
#   1. md-openmm data-register --user-config PATH
#   2. $MD_TOOLS_CONFIG
#   3. ${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config
#
# Create it with:
#   md-openmm data-register --init
#
schema_version: "1.0"

user:
  # A stable identifier for you, lowercase, used as the person_id in every dataset manifest you
  # register. Changing it later makes your old and new datasets look like different people.
  person_id: "your-name-here"
  # Your name as it should appear in provenance.
  name: "Your Name"
  # An ORCID, or null. Recorded as-is; it is not looked up or verified.
  orcid: null
  # Optional affiliation.
  affiliation: null

machine:
  # The managed storage root: the directory registered datasets are written beneath. An absolute
  # path. $MD_DATA overrides this, and an explicit --md-data overrides both.
  md_data: "/absolute/path/to/MD_DATA"
"""


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "md-tools" / CONFIG_BASENAME


def config_path(explicit: str | Path | None = None) -> tuple[Path, str]:
    """The configuration path and which rule supplied it."""
    if explicit:
        return Path(explicit).expanduser(), "--user-config"
    from_env = os.environ.get("MD_TOOLS_CONFIG")
    if from_env:
        return Path(from_env).expanduser(), "MD_TOOLS_CONFIG"
    return default_config_path(), "XDG default"


def load_user_config(explicit: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    path, origin = config_path(explicit)
    if not path.is_file():
        raise RegistrationError(
            f"no user configuration at {path} (from {origin}).\n"
            f"Create one with:\n"
            f"    md-openmm data-register --init")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RegistrationError(f"{path}: not valid YAML -- {exc}") from None
    if not isinstance(document, dict):
        raise RegistrationError(f"{path}: the configuration must be a mapping")
    version = document.get("schema_version")
    if str(version) != CONFIG_SCHEMA_VERSION:
        raise RegistrationError(
            f"{path}: schema_version is {version!r}, this build understands "
            f"{CONFIG_SCHEMA_VERSION!r}")
    user = document.get("user") or {}
    for field in ("person_id", "name"):
        if not user.get(field):
            raise RegistrationError(f"{path}: user.{field} is required")
    return document, path, origin


def resolve_md_data(document: dict[str, Any], *, override: str | None = None) -> tuple[Path, str]:
    """The managed storage root, and which source supplied it."""
    if override:
        root, origin = Path(override).expanduser(), "--md-data"
    elif os.environ.get("MD_DATA"):
        root, origin = Path(os.environ["MD_DATA"]).expanduser(), "$MD_DATA"
    else:
        stated = (document.get("machine") or {}).get("md_data")
        if not stated:
            raise RegistrationError(
                "no managed storage root: machine.md_data is unset in the user configuration, "
                "$MD_DATA is unset, and no --md-data was given")
        root, origin = Path(stated).expanduser(), "machine.md_data"
    if not root.is_absolute():
        raise RegistrationError(f"the managed storage root {root} (from {origin}) is not an "
                                f"absolute path")
    if not root.is_dir():
        raise RegistrationError(f"the managed storage root {root} (from {origin}) is not an "
                                f"existing directory")
    return root.resolve(), origin


def init_user_config(*, explicit: str | Path | None = None, force: bool = False,
                     noninteractive: bool = False, name: str | None = None,
                     person_id: str | None = None, orcid: str | None = None,
                     md_data: str | None = None) -> Path:
    """Create the user configuration. Interactive on a TTY, fully flag-driven otherwise."""
    import sys

    path, origin = config_path(explicit)
    if path.exists() and not force:
        raise RegistrationError(
            f"{path} already exists (from {origin}). Pass --force to overwrite it deliberately; "
            f"overwriting silently would discard an identity that existing datasets already "
            f"reference.")

    interactive = (not noninteractive) and sys.stdin.isatty()
    if interactive:
        person_id = person_id or input("person_id (stable, lowercase): ").strip()
        name = name or input("name: ").strip()
        orcid = orcid or (input("ORCID (blank for none): ").strip() or None)
        md_data = md_data or input("managed storage root ($MD_DATA): ").strip()
    missing = [flag for flag, value in (("--person-id", person_id), ("--name", name),
                                        ("--md-data", md_data)) if not value]
    if missing:
        raise RegistrationError(
            f"missing {', '.join(missing)}. With --noninteractive every value must be supplied "
            f"by flag, because there is no terminal to ask.")

    root = Path(md_data).expanduser()
    if not root.is_absolute():
        raise RegistrationError(f"--md-data {md_data} is not an absolute path")
    if not root.is_dir():
        raise RegistrationError(f"--md-data {root} is not an existing directory. Create it first; "
                                f"initialising a configuration that points nowhere just moves the "
                                f"failure to the first registration.")

    document = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "user": {"person_id": person_id, "name": name, "orcid": orcid or None,
                 "affiliation": None},
        "machine": {"md_data": str(root.resolve())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, stat.S_IRWXU)
    except OSError:
        pass                                   # a filesystem without POSIX modes; not fatal
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text("# Written by `md-openmm data-register --init`.\n"
                   + yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    os.replace(tmp, path)
    return path
