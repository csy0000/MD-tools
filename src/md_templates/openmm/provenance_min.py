"""Concise reproducibility information: what produced a directory, and from what.

Deliberately small. If git metadata is unavailable the installed package version is recorded and
`git_commit` is null; generation is never blocked on it, and there is no build-time stamping, no
wheel identity and no source-tree digest.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def package_provenance() -> dict[str, Any]:
    from importlib import metadata

    try:
        version = metadata.version("md-templates")
    except metadata.PackageNotFoundError:
        version = None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    try:
        import openmm
        openmm_version = openmm.version.short_version
    except Exception:                              # noqa: BLE001
        openmm_version = None

    return {
        "md_templates": {
            "version": version,
            "git_commit": commit,
            "git_dirty": (None if status is None else bool(status)),
        },
        "openmm": {"version": openmm_version},
    }
