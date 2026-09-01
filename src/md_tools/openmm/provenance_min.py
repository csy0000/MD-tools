"""What produced a directory, and from what.

Small on purpose. There is no build backend stamping a commit, no registry and no schema engine --
just the facts a reader needs months later to know which implementation wrote these files and which
inputs went in.

The awkward case this module exists to handle: an installed wheel has no `.git`, so `git_commit` is
null, and a version string alone does not distinguish two builds of `0.4.0.dev0`. The installed
fingerprint closes that gap by hashing the installed package's own source and template resources.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

#: Everything the fingerprint covers. Bytecode, caches and mutable metadata are excluded: they
#: differ between two installs of identical code and would make the fingerprint useless.
FINGERPRINT_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".json")
FINGERPRINT_EXCLUDED_DIRS = ("__pycache__", ".pytest_cache", ".git")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> Optional[str]:
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "-C", str(root), *args],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def installed_fingerprint() -> dict[str, Any]:
    """A deterministic digest of the installed `md_tools` implementation.

    Two installs of the same code fingerprint identically; a single edited line changes the digest.
    That is what identifies the implementation when `git_commit` is null, which is the normal case
    for anyone running from a wheel.

    Sorted relative POSIX paths and streamed contents, so the value does not depend on filesystem
    ordering or on where the package happens to live.
    """
    root = Path(__file__).resolve().parent.parent          # .../md_tools
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in FINGERPRINT_SUFFIXES:
            continue
        if any(part in FINGERPRINT_EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {"algorithm": "sha256-of-sorted-path-and-content-digests",
            "file_count": len(files),
            "value": digest.hexdigest()}


def direct_url_record() -> Optional[dict[str, Any]]:
    """PEP 610 `direct_url.json`: where this install came from, when pip recorded it."""
    from importlib import metadata

    try:
        distribution = metadata.distribution("md-tools")
        text = distribution.read_text("direct_url.json")
    except Exception:                              # noqa: BLE001
        return None
    if not text:
        return None
    try:
        import json

        return json.loads(text)
    except ValueError:
        return None


def implementation_identity() -> dict[str, Any]:
    """Which MD-tools wrote this. Never null in every field at once.

    `git_commit` is the best answer and is often unavailable. `installed_fingerprint` is always
    available, so "which implementation" always has an answer even from a wheel.
    """
    from importlib import metadata

    try:
        version = metadata.version("md-tools")
    except metadata.PackageNotFoundError:
        version = None

    status = _git("status", "--porcelain")
    return {
        "version": version,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": (None if status is None else bool(status)),
        "installed_fingerprint": installed_fingerprint(),
        "direct_url": direct_url_record(),
    }


#: The one repository URL. Everything that records the generator reads it from here.
REPOSITORY = "https://github.com/csy0000/MD-tools"

#: An exact 40-hex commit. A branch, a tag or an abbreviation is not a pin.
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def template_identity() -> dict[str, Any]:
    """THE canonical answer to "which MD-tools is this", resolved once.

    Everything that records the generator -- `dataset.yaml`, both `provenance.yaml` files,
    `resolved_sys.config.yaml`, `md.config.yaml`, every `stage.yaml`, every method record -- reads
    this. Before, each writer reached for `implementation_identity()["git_commit"]` on its own,
    and in a VCS-installed package that is `None` even though `direct_url.json` records the exact
    commit: the dataset manifest would carry a verified commit while the stage files beside it
    carried nulls, and nothing compared them.

    Two evidence routes, in order of directness:

    * `git checkout` -- a real checkout, `git rev-parse HEAD`. `dirty` says whether the working
      tree matches it.
    * `direct_url.json` -- PEP 610, written by pip when a package is installed from a VCS URL. A
      wheel built from a tarball has neither, and then `commit` is None.

    `commit` is None rather than guessed. A version string, a branch name, a date or an installed
    fingerprint identifies an implementation but is not a commit, and writing one as though it
    were would produce a pin that can be checked out and will not reproduce the run.
    """
    identity = implementation_identity()
    resolved = {
        "repository": REPOSITORY,
        "commit": None,
        "evidence": "unavailable",
        "version": identity["version"],
        "installed_fingerprint": identity["installed_fingerprint"]["value"],
        "dirty": None,
        # Spelled out rather than left to be inferred from `dirty`. A commit beside a modified
        # working tree looks exactly like a commit that reproduces the run, and this is the field
        # that says it does not. Contract-managed generation refuses this state outright; an
        # unregistered project may proceed and carries this statement in every record.
        "reproducible_from_commit": False,
        "reproducibility": ("no exact commit could be established for this installation, so these "
                            "files record WHAT ran only by installed fingerprint and version"),
        "detail": ("no Git checkout and no PEP 610 direct_url.json records an exact commit for "
                   "this installation"),
    }

    commit = (identity.get("git_commit") or "").strip().lower()
    if commit and COMMIT_PATTERN.match(commit):
        dirty = bool(identity.get("git_dirty"))
        resolved.update({
            "commit": commit, "evidence": "git checkout", "dirty": dirty,
            "reproducible_from_commit": not dirty,
            "reproducibility": (
                f"the working tree has uncommitted changes, so checking out {commit[:12]} does "
                f"NOT reproduce this generation; the installed fingerprint identifies what ran"
                if dirty else
                f"a clean checkout: {commit[:12]} reproduces this generation"),
            "detail": ("git rev-parse HEAD, with uncommitted changes in the working tree"
                       if dirty else "git rev-parse HEAD"),
        })
        return resolved

    direct = identity.get("direct_url") or {}
    vcs = direct.get("vcs_info") or {}
    installed = (vcs.get("commit_id") or "").strip().lower()
    if installed and COMMIT_PATTERN.match(installed):
        resolved.update({
            "commit": installed, "evidence": "direct_url.json",
            # A VCS install is a fixed set of bytes: there is no working tree to be dirty.
            "dirty": False, "reproducible_from_commit": True,
            "reproducibility": f"installed from a VCS pin: {installed[:12]} reproduces this "
                               f"generation",
            "detail": f"installed from {direct.get('url')} at {installed[:12]}",
        })
        if direct.get("url"):
            resolved["repository"] = str(direct["url"])
    return resolved


def environment_versions() -> dict[str, Any]:
    """The packages whose versions change what a build or a run produces."""
    import platform
    from importlib import metadata

    versions: dict[str, Any] = {"python": platform.python_version()}
    try:
        import openmm

        versions["openmm"] = openmm.version.short_version
    except Exception:                              # noqa: BLE001
        versions["openmm"] = None
    # Preparation-side packages. Absent ones are recorded as null rather than omitted, so a reader
    # can tell "not installed" from "we forgot to look".
    for name, distribution in (("openff_toolkit", "openff-toolkit"),
                               ("openmmforcefields", "openmmforcefields"),
                               ("parmed", "parmed"),
                               ("rdkit", "rdkit"),
                               ("numpy", "numpy"),
                               ("pyyaml", "pyyaml")):
        try:
            versions[name] = metadata.version(distribution)
        except Exception:                          # noqa: BLE001
            versions[name] = None
    return versions


def package_provenance() -> dict[str, Any]:
    """The short form kept for existing callers."""
    identity = implementation_identity()
    return {
        "md_tools": {
            "version": identity["version"],
            "git_commit": identity["git_commit"],
            "git_dirty": identity["git_dirty"],
            "installed_fingerprint": identity["installed_fingerprint"]["value"],
        },
        "openmm": {"version": environment_versions()["openmm"]},
    }
