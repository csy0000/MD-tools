"""`md-template.lock.json` -- what generated a project, stated so another repository can cite it.

A generated project outlives the checkout that made it, and usually leaves the machine that made it.
Without this file the only honest answer to "which method is this?" is the directory name. With it,
the answer is a repository URL, a forty-character commit and a release tag -- enough for someone
else to fetch exactly the template that ran and check it against the hashes recorded here.

Never an abbreviated SHA and never a branch name. `dev` and `main` move; `a3f9c2` collides once a
repository is large enough. A citation has to survive both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import provenance
from .hashing import sha256_file

__all__ = ["LOCKFILE_NAME", "LOCKFILE_SCHEMA_VERSION", "build_lockfile", "human_identity",
           "write_lockfile"]

LOCKFILE_NAME = "md-template.lock.json"
LOCKFILE_SCHEMA_VERSION = 1

#: Canonical form of the repository, independent of how the checkout happens to be configured --
#: `git@github.com:` and `https://github.com/` are the same repository and must cite identically.
_CANONICAL_HOST = "https://github.com/"


def canonical_repository_url(raw: Optional[str]) -> Optional[str]:
    """Normalise an origin URL to its `https://github.com/owner/repo` form."""
    if not raw:
        return None
    url = raw.strip()
    if url.startswith("git@github.com:"):
        url = _CANONICAL_HOST + url[len("git@github.com:"):]
    elif url.startswith("ssh://git@github.com/"):
        url = _CANONICAL_HOST + url[len("ssh://git@github.com/"):]
    if url.endswith(".git"):
        url = url[:-len(".git")]
    return url


def build_lockfile(*, method: str, template_path: str, resolved_config: dict,
                   config_schema_version, exported_files: Optional[dict] = None,
                   generated_utc: Optional[str] = None) -> dict:
    """Assemble the identity record. `method` is `cMD` or `REST2`."""
    if method not in ("cMD", "REST2"):
        raise ValueError(f"method must be 'cMD' or 'REST2'; got {method!r}")

    git = provenance.git_state() or {}
    commit = git.get("commit")
    if commit is not None and len(commit) != 40:
        raise ValueError(
            f"the recorded commit must be the full 40-character SHA; got {commit!r}. An abbreviated "
            "SHA collides once a repository is large enough, and a citation has to survive that.")

    openmm_identity = provenance.openmm_identity()
    canonical = json.dumps(resolved_config, sort_keys=True, separators=(",", ":")).encode()

    return {
        "schema_version": LOCKFILE_SCHEMA_VERSION,
        "generated_utc": generated_utc or provenance.utc_timestamp(),
        "method": method,
        "engine": "openmm",
        "template": {
            "repository": canonical_repository_url(git.get("remote_url")),
            "commit": commit,
            "release_tag": git.get("tag"),
            "nearest_release_tag": git.get("nearest_tag"),
            "template_path": template_path,
            "source_tree_clean": (None if git.get("dirty") is None else not git["dirty"]),
            "note": ("A branch name is not an identity: `dev` and `main` move. Cite the release tag "
                     "together with the full commit."),
        },
        "openmm": {
            "short_version": openmm_identity.get("short_version"),
            "full_version": openmm_identity.get("full_version"),
            "git_revision": openmm_identity.get("git_revision"),
            "release_flag": openmm_identity.get("release_flag"),
            "note": ("full_version carries a `.dev-<sha>` suffix on the official builds because "
                     "they ship release = False; the sha is the release's own tag commit, so it is "
                     "a build stamp rather than a development snapshot."),
        },
        "package_version": provenance.package_version(),
        "configuration": {
            "schema_version": config_schema_version,
            "resolved_sha256": provenance.sha256_bytes(canonical)
            if hasattr(provenance, "sha256_bytes") else __import__("hashlib").sha256(
                canonical).hexdigest(),
            "canonicalisation": "json.dumps(sort_keys=True, separators=(',',':')), UTF-8",
        },
        "exported_files": exported_files or {},
    }


def human_identity(lock: dict) -> str:
    """Three lines a person can paste into a methods section."""
    template = lock["template"]
    tag = template.get("release_tag") or template.get("nearest_release_tag") or "(untagged)"
    return (f"method   = openmm/{lock['method']}\n"
            f"template = {_short_repo(template.get('repository'))}@{template.get('commit')}\n"
            f"release  = {tag}")


def _short_repo(url: Optional[str]) -> str:
    if not url:
        return "(unknown repository)"
    return url[len(_CANONICAL_HOST):] if url.startswith(_CANONICAL_HOST) else url


def write_lockfile(out_dir: Path, lock: dict) -> Path:
    """Write the lock file, plus the human-readable identity beside it."""
    out_dir = Path(out_dir)
    path = out_dir / LOCKFILE_NAME
    provenance.write_json(path, lock)
    (out_dir / "METHOD_IDENTITY.txt").write_text(human_identity(lock) + "\n", encoding="utf-8")
    return path


def hash_exported(root: Path, names) -> dict:
    """SHA-256 of each exported runtime file, so a relocated project can prove it is intact."""
    root = Path(root)
    out = {}
    for name in sorted(names):
        candidate = root / name
        if candidate.is_file():
            out[str(name)] = sha256_file(candidate)
    return out
