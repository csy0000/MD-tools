"""A setuptools build backend that bakes the source commit into the wheel.

WHY THIS EXISTS

A wheel is built from a DIRECTORY. Nothing in the build consults git, and the only origin note
pip keeps is the path it was told to build from:

    {"dir_info": {}, "url": "file:///.../MD-tools"}

So an installed package has no way to know which commit it came from, `source_commit()` returned
None, and every run record, every dataset manifest and every exported reference bundle carried
`md_tools_commit: null`. Recovering the commit afterwards took hashing all 99 installed files
against the git trees -- which worked, but is not a provenance system.

The commit has to be captured at BUILD time, because after the build the information is gone.
This backend delegates everything to setuptools and does one extra thing first: it writes
`src/md_tools/_commit.py` with the commit being built.

A DIRTY TREE BAKES NOTHING

If the working tree has uncommitted changes, the commit does not describe what is in the wheel,
so `None` is baked. That is the same rule `source_commit` already followed and the same rule
`registry.register._origin` enforces: a commit that does not describe what ran is a false
provenance claim, and false is worse than absent.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from setuptools.build_meta import *  # noqa: F401,F403  -- the rest of the PEP 517 interface
from setuptools.build_meta import build_sdist as _build_sdist
from setuptools.build_meta import build_wheel as _build_wheel

HERE = Path(__file__).resolve().parent


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def bake(root: Path = HERE) -> str | None:
    """Write `<root>/src/md_tools/_commit.py`. Returns what it recorded.

    `root` is a parameter so the policy -- clean tree bakes the commit, dirty tree bakes None --
    can be tested against a throwaway repository instead of against the checkout doing the build.
    """
    root = Path(root)
    commit = _git(root, "rev-parse", "HEAD")
    if commit and _git(root, "status", "--porcelain"):
        commit = None          # see the module docstring: a dirty tree describes no commit
    baked = root / "src" / "md_tools" / "_commit.py"
    baked.parent.mkdir(parents=True, exist_ok=True)
    baked.write_text(
        '"""The commit this distribution was built from. WRITTEN BY THE BUILD -- do not edit.\n'
        '\n'
        'None means the build could not establish one: not a git checkout, git unavailable, or a\n'
        'working tree with uncommitted changes. `md_tools.build.record.source_commit` prefers live\n'
        'git when running from a checkout and falls back to this only for an installed wheel.\n'
        '"""\n'
        f"COMMIT = {commit!r}\n",
        encoding="utf-8")
    return commit


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    bake()
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    bake()
    return _build_sdist(sdist_directory, config_settings)
