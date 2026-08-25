"""In-tree PEP 517 backend that stamps source identity into the distribution.

A wheel installed into a clean environment has no `.git` directory, so asking git at
project-generation time cannot answer "which commit of MD-templates produced this run?". That
question has to be answered while the build still has the checkout, and the answer has to travel
inside the artifact.

This backend wraps setuptools and writes `md_templates/_build_info.py` immediately before the
sdist or wheel is built. Everything else is delegated unchanged.

The generated module is deliberately data-only -- literals, no imports, no logic -- so that reading
it can never fail in a way that costs a run its provenance.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from setuptools import build_meta as _origin

# re-export the rest of the PEP 517 interface unchanged
get_requires_for_build_wheel = _origin.get_requires_for_build_wheel
get_requires_for_build_sdist = _origin.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _origin.prepare_metadata_for_build_wheel
get_requires_for_build_editable = getattr(_origin, "get_requires_for_build_editable", None)
prepare_metadata_for_build_editable = getattr(
    _origin, "prepare_metadata_for_build_editable", None)

_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _ROOT / "src" / "md_templates" / "_build_info.py"


def _git_raw(*args: str) -> tuple[bool, str]:
    """`(succeeded, stdout)` for one git query.

    Kept separate from `_git` because empty output and failure mean different things for
    `status --porcelain`: "" is a CLEAN tree, failure is UNKNOWN dirtiness. Collapsing both to
    None would record a clean tree for a build whose git query failed.
    """
    try:
        res = subprocess.run(["git", "-C", str(_ROOT), *args],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return res.returncode == 0, res.stdout.strip()


def _git(*args: str) -> str | None:
    """One git query, or None when it cannot be answered or answered emptily."""
    ok, out = _git_raw(*args)
    return out or None if ok else None


def _stamp() -> None:
    """Write the identity of the tree being built, or an explicit 'unknown' record.

    A build from an exported tarball with no `.git` writes commit=None. That is not silently
    tolerated later: project generation refuses to write a lock file without a full commit, so an
    unstamped build fails loudly at the point where the provenance would have been wrong, rather
    than producing a run nobody can trace.
    """
    if not (_ROOT / ".git").exists():
        commit = remote = tag = None
        dirty = None
    else:
        commit = _git("rev-parse", "HEAD")
        remote = _git("remote", "get-url", "origin")
        tag = _git("describe", "--tags", "--exact-match")
        status_ok, status = _git_raw("status", "--porcelain")
        # "" means clean; a FAILED query means unknown. Recording unknown as clean would let a
        # dirty build claim a clean provenance.
        dirty = bool(status) if status_ok else None

    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    _TARGET.write_text(
        '"""Generated at build time by build_backend/md_templates_build.py. Do not edit.\n'
        '\n'
        'This file is what lets a wheel installed outside the checkout still name the commit it\n'
        'was built from. It is regenerated on every build and is not tracked in git.\n'
        '"""\n'
        f"BUILD_COMMIT = {commit!r}\n"
        f"BUILD_REMOTE_URL = {remote!r}\n"
        f"BUILD_TAG = {tag!r}\n"
        f"BUILD_DIRTY = {dirty!r}\n",
        encoding="utf-8",
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _stamp()
    return _origin.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _stamp()
    return _origin.build_sdist(sdist_directory, config_settings)


if hasattr(_origin, "build_editable"):
    def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
        _stamp()
        return _origin.build_editable(wheel_directory, config_settings, metadata_directory)
