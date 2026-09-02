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

  # How OpenMM runs ON THIS MACHINE. Hardware, not science: a protocol is the same experiment
  # wherever it runs, so the platform does not belong in a protocol configuration that is shared
  # between machines and committed to a repository.
  openmm:
    # CUDA or CPU. CUDA is the built-in default and there is NO automatic fallback -- a run that
    # quietly moved to the CPU finishes, writes a trajectory and reports success two orders of
    # magnitude later. CPU here is a deliberate machine-wide choice, recorded as such, and
    # `--cpu` is the per-run override.
    platform: CUDA
    # mixed is the OpenMM default and what the validated runs used. single is faster and less
    # accurate; double is neither, on the hardware this targets.
    precision: mixed
    # How a rank picks its CUDA device under MPI. local_rank gives one rank per visible device,
    # which is the only policy that keeps a ladder off a single GPU.
    device_policy: local_rank
"""

#: The machine OpenMM defaults, used when the user configuration says nothing. CUDA, because the
#: alternative -- guessing from what happens to be available -- is how a run ends up on hardware
#: nobody chose.
MACHINE_OPENMM_DEFAULTS = {"platform": "CUDA", "precision": "mixed",
                           "device_policy": "local_rank"}

MACHINE_PLATFORMS = ("CUDA", "CPU")
MACHINE_PRECISIONS = ("mixed", "single", "double")
MACHINE_DEVICE_POLICIES = ("local_rank", "openmm")


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


def _load_document(path: Path, origin: str) -> dict[str, Any]:
    """Read one user configuration STRICTLY. Duplicate keys are fatal and are named.

    `yaml.safe_load` keeps the last of a repeated key and says nothing, so a file that sets
    `platform` twice runs as one of them and the other is invisible. This is a file a person edits
    by hand, which is exactly where that happens.
    """
    from ..build.strict import ConfigError, load_yaml_strictly

    try:
        document = load_yaml_strictly(path.read_text(encoding="utf-8"), source=str(path))
    except ConfigError as duplicate:
        raise RegistrationError(f"{path} (from {origin}): {duplicate}") from None
    except yaml.YAMLError as exc:
        raise RegistrationError(f"{path} (from {origin}): not valid YAML -- {exc}") from None
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise RegistrationError(
            f"{path} (from {origin}): the configuration must be a mapping of sections, not a "
            f"{type(document).__name__}.")
    return document


def load_user_config(explicit: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    path, origin = config_path(explicit)
    if not path.is_file():
        raise RegistrationError(
            f"no user configuration at {path} (from {origin}).\n"
            f"Create one with:\n"
            f"    md-openmm data-register --init")
    document = _load_document(path, origin)
    version = document.get("schema_version")
    if str(version) != CONFIG_SCHEMA_VERSION:
        raise RegistrationError(
            f"{path}: schema_version is {version!r}, this build understands "
            f"{CONFIG_SCHEMA_VERSION!r}")
    user = document.get("user") or {}
    if not isinstance(user, dict):
        raise RegistrationError(
            f"{path}: `user` must be a mapping of fields, not a {type(user).__name__}.")
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
        # The OpenMM defaults are written out rather than left implicit, so the file a person
        # opens shows what their machine will actually do.
        "machine": {"md_data": str(root.resolve()),
                    "openmm": dict(MACHINE_OPENMM_DEFAULTS)},
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


def resolve_machine_openmm(document: dict[str, Any] | None = None) -> dict[str, Any]:
    """The machine's OpenMM settings, and where each came from.

    Never raises for an ABSENT block: a configuration written before `machine.openmm` existed is
    still a valid configuration, and it resolves to the built-in defaults. A block that IS present
    and states something unsupported is refused, because that is a choice someone made and got
    wrong, not an omission.
    """
    document = document or {}
    stated = ((document.get("machine") or {}).get("openmm")) or {}
    if not isinstance(stated, dict):
        raise RegistrationError("machine.openmm must be a mapping of settings")

    unknown = sorted(set(stated) - set(MACHINE_OPENMM_DEFAULTS))
    if unknown:
        raise RegistrationError(
            f"machine.openmm: unknown setting(s) {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(MACHINE_OPENMM_DEFAULTS))}.")

    resolved = dict(MACHINE_OPENMM_DEFAULTS)
    resolved.update({key: value for key, value in stated.items() if value is not None})

    platform = str(resolved["platform"]).upper()
    if platform not in MACHINE_PLATFORMS:
        raise RegistrationError(
            f"machine.openmm.platform = {resolved['platform']!r}. Supported: "
            f"{', '.join(MACHINE_PLATFORMS)}.\n"
            f"  CUDA is the normal default; CPU is an intentional machine-wide CPU default. "
            f"There is no automatic fallback between them, and no other platform is offered here "
            f"-- OpenCL and Reference would each need their own validation.")
    resolved["platform"] = platform
    if str(resolved["precision"]) not in MACHINE_PRECISIONS:
        raise RegistrationError(
            f"machine.openmm.precision = {resolved['precision']!r}. Supported: "
            f"{', '.join(MACHINE_PRECISIONS)}.")
    if str(resolved["device_policy"]) not in MACHINE_DEVICE_POLICIES:
        raise RegistrationError(
            f"machine.openmm.device_policy = {resolved['device_policy']!r}. Supported: "
            f"{', '.join(MACHINE_DEVICE_POLICIES)}.")

    resolved["origin"] = "machine.openmm" if stated else "built-in default"
    return resolved


def machine_openmm_settings(explicit: str | Path | None = None) -> dict[str, Any]:
    """The machine's OpenMM settings, through the ordinary configuration discovery order.

    ABSENT IS NOT INVALID, and the difference is the whole point of this function.

    A missing configuration file is legitimate here in a way it is not for registration: running a
    simulation needs neither an identity nor a storage root, and refusing to run because nobody has
    registered data yet would be absurd. So an absent file resolves to the built-in defaults.

    An EXISTING file that cannot be read, or that says something wrong, is fatal. It used to be
    caught by the same `except RegistrationError` and answered with those same defaults, which
    meant a malformed configuration -- bad YAML, a duplicate key, an unknown field, an invalid
    platform -- silently became CUDA / mixed / local_rank. The machine then ran on settings nobody
    chose, and the file that was supposed to say otherwise was never mentioned again.

    `MD_TOOLS_CONFIG` naming a file that does not exist is a BROKEN REFERENCE, not an absence:
    somebody meant that path, and answering with defaults hides a typo in an environment variable.
    """
    path, origin = config_path(explicit)

    if not path.is_file():
        if origin in ("--user-config", "MD_TOOLS_CONFIG"):
            raise RegistrationError(
                f"{origin} points at {path}, which does not exist.\n"
                f"  This is a broken reference rather than an absent configuration: something "
                f"named that path deliberately. Correct it, or unset it to use the built-in "
                f"defaults (CUDA, mixed precision, local-rank device placement).")
        resolved = resolve_machine_openmm({})
        resolved["config_path"] = None
        resolved["config_origin"] = f"{origin} (no file; built-in defaults)"
        return resolved

    document = _load_document(path, origin)
    version = document.get("schema_version")
    if version is not None and str(version) != CONFIG_SCHEMA_VERSION:
        raise RegistrationError(
            f"{path} (from {origin}): schema_version is {version!r}, this build understands "
            f"{CONFIG_SCHEMA_VERSION!r}.")
    # The `user` block is not this function's business, but a malformed one still means the file
    # is wrong, and a wrong file must not resolve to defaults just because the part this caller
    # needed happened to be absent.
    user = document.get("user")
    if user is not None and not isinstance(user, dict):
        raise RegistrationError(
            f"{path} (from {origin}): `user` must be a mapping of fields, not a "
            f"{type(user).__name__}.")
    machine = document.get("machine")
    if machine is not None and not isinstance(machine, dict):
        raise RegistrationError(
            f"{path} (from {origin}): `machine` must be a mapping of settings, not a "
            f"{type(machine).__name__}.")

    try:
        resolved = resolve_machine_openmm(document)
    except RegistrationError as invalid:
        # `resolve_machine_openmm` works on a document and cannot know where it came from. Every
        # refusal a person sees must name the file they have to edit, so the path is added here.
        raise RegistrationError(f"{path} (from {origin}): {invalid}") from None
    resolved["config_path"] = str(path)
    resolved["config_origin"] = origin
    return resolved
