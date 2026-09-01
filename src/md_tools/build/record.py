"""Human-readable logs that carry a machine-readable record.

Every log this package writes is meant to be read by a person, in the style of `leap.log`: the
command, what the configuration resolved to, what was actually done, the counts, the warnings, and
a short completion summary.

Free text is not a database. Deciding whether a run finished by looking for a hopeful sentence in
a log is how a truncated, killed, or still-running job gets registered as complete -- the sentence
is there because the code that printed it ran, not because the data are whole. So each log also
carries a **delimited, versioned machine record**: one YAML block between two fixed markers,
holding the facts a downstream tool is allowed to rely on.

The contract between writer and reader is exactly this:

  * a reader parses the block between the markers, and nothing else;
  * a log with no block, two blocks, an unparseable block, or a block whose `schema_version` is
    not understood is **not evidence of anything** and is refused;
  * `status: completed` inside the block is set only after outputs are flushed, reopened and
    verified. It is the single completion signal. Prose in the same file is display only.

Registration enforces the reader half of this in `md_tools.registry`.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

#: Bumped when the *meaning* of a field changes. A reader refuses a version it does not know
#: rather than guessing; that is why the rejection is strict rather than a best-effort parse.
RECORD_SCHEMA_VERSION = "md-tools-record/2.0"

BEGIN = "# ===== BEGIN md-tools machine record ====="
END = "# ===== END md-tools machine record ====="


class LogWriter:
    """A readable log that ends with one delimited machine record."""

    def __init__(self, path: Path, *, record_type: str, command: list[str] | None = None,
                 echo: bool = True) -> None:
        self.path = Path(path)
        self.echo = echo
        self.lines: list[str] = []
        self.record: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "record_type": record_type,
            "status": "started",
            "started_utc": _now(),
            "finished_utc": None,
            "command": list(command if command is not None else sys.argv),
            "environment": environment_facts(),
        }

    # -- the human half ---------------------------------------------------------------------

    def __call__(self, message: str = "") -> None:
        self.lines.append(message)
        if self.echo:
            print(message)

    def heading(self, title: str) -> None:
        self("")
        self(title)
        self("-" * len(title))

    def field(self, key: str, value: Any) -> None:
        self(f"  {key:<28}{value}")

    # -- the machine half -------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        self.record[key] = value

    def update(self, **values: Any) -> None:
        self.record.update(values)

    def fail(self, reason: str) -> None:
        self.record["status"] = "failed"
        self.record["failure_reason"] = reason
        self.record["finished_utc"] = _now()

    def complete(self) -> None:
        """Mark completion. Call only after outputs exist, are flushed and have been re-read."""
        self.record["status"] = "completed"
        self.record["finished_utc"] = _now()

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.lines).rstrip("\n")
        block = yaml.safe_dump(self.record, sort_keys=False, default_flow_style=False,
                               allow_unicode=True).rstrip("\n")
        text = (f"{body}\n\n{BEGIN}\n"
                + "\n".join(f"# {line}" if line else "#" for line in block.splitlines())
                + f"\n{END}\n")
        tmp = self.path.with_suffix(self.path.suffix + ".partial")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path


class RecordError(ValueError):
    """A log did not carry a usable machine record."""


def read_record(path: Path) -> dict[str, Any]:
    """Parse the one machine record in `path`, or refuse.

    Deliberately unforgiving. Every rejection below is a real way a half-written or forged log has
    been mistaken for a finished run.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RecordError(f"{path}: cannot be read -- {exc}") from None
    starts = [i for i, line in enumerate(text.splitlines()) if line.strip() == BEGIN]
    ends = [i for i, line in enumerate(text.splitlines()) if line.strip() == END]
    if not starts:
        raise RecordError(
            f"{path}: carries no machine record. A log without one is display text, not evidence "
            f"that anything completed.")
    if len(starts) != 1 or len(ends) != 1:
        raise RecordError(f"{path}: expected exactly one machine record, found "
                          f"{len(starts)} start and {len(ends)} end markers")
    if ends[0] < starts[0]:
        raise RecordError(f"{path}: the machine record's end marker precedes its start marker")
    lines = text.splitlines()[starts[0] + 1:ends[0]]
    stripped = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                         for line in lines)
    try:
        document = yaml.safe_load(stripped)
    except yaml.YAMLError as exc:
        raise RecordError(f"{path}: the machine record is not valid YAML -- {exc}") from None
    if not isinstance(document, dict):
        raise RecordError(f"{path}: the machine record is not a mapping")
    version = document.get("schema_version")
    if version != RECORD_SCHEMA_VERSION:
        raise RecordError(
            f"{path}: machine record schema_version is {version!r}, this build of MD-tools "
            f"understands {RECORD_SCHEMA_VERSION!r}. Refusing rather than guessing what a "
            f"different version meant.")
    for required in ("record_type", "status", "started_utc"):
        if required not in document:
            raise RecordError(f"{path}: the machine record has no {required!r}")
    return document


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_facts(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Identity of one file: a path that stays portable, its size and its digest.

    The path is recorded relative to a declared root wherever one is given. An absolute machine
    path in a manifest is not portable and leaks the layout of whoever happened to run it.
    """
    path = Path(path)
    name = str(path.relative_to(relative_to)) if relative_to else path.name
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def package_versions() -> dict[str, Any]:
    """Versions of what was actually imported, not what was requested.

    Absent packages are recorded as null rather than omitted: "RDKit was not installed" and "we
    forgot to look" are different facts, and only one of them is reproducible.
    """
    from importlib.metadata import PackageNotFoundError, version as _version

    names = ["md-tools", "openmm", "openff-toolkit", "openmmforcefields", "parmed", "rdkit",
             "ambertools", "numpy", "pyyaml"]
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            out[name] = _version(name)
        except PackageNotFoundError:
            out[name] = None
    try:  # OpenMM reports its own version independently of distribution metadata.
        import openmm
        out["openmm"] = openmm.__version__
    except Exception:
        pass
    return out


def source_commit() -> str | None:
    """The immutable commit of the MD-tools tree, when it can be established.

    Returns None rather than a guess. An installed wheel has no git tree, and inventing a commit
    would put an unverifiable claim into a provenance record.
    """
    import subprocess
    here = Path(__file__).resolve()
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(here.parent),
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def environment_facts() -> dict[str, Any]:
    """Where this ran. Enough to audit, nothing that is a secret.

    The username is recorded because provenance needs an actor, but no environment variables are
    copied wholesale: they routinely carry tokens, and a dataset manifest travels.
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "user": user,
        "packages": package_versions(),
        "md_tools_commit": source_commit(),
    }


def openmm_platform_facts(context_or_platform=None) -> dict[str, Any]:
    """Resolved OpenMM platform and its non-secret properties."""
    facts: dict[str, Any] = {"name": None, "properties": {}}
    if context_or_platform is None:
        return facts
    try:
        plat = getattr(context_or_platform, "getPlatform", None)
        plat = plat() if callable(plat) else context_or_platform
        facts["name"] = plat.getName()
        for key in plat.getPropertyNames():
            try:
                if hasattr(context_or_platform, "getPlatform"):
                    facts["properties"][key] = plat.getPropertyValue(context_or_platform, key)
                else:
                    facts["properties"][key] = plat.getPropertyDefaultValue(key)
            except Exception:
                continue
    except Exception:
        pass
    return facts
