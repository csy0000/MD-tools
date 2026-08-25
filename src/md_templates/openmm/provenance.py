"""Provenance for portable explicit-solvent runs.

Everything here answers one question: given only an output directory, on a machine that has never
seen the source checkout, can a reader reconstruct what was computed and check that the inputs are
what the manifest says they are?

That means recording the toolchain (which decides the numbers), the hardware and platform (which
decides reproducibility class), the exact command and time, and a SHA-256 for every file that
matters. It also means being honest when something is unavailable: a missing version is recorded
as ``null`` with a reason rather than omitted, because an absent key is indistinguishable from a
key that was never collected.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from .schemas import sha256_file

#: Packages whose versions change the numbers, and are therefore part of the provenance rather
#: than of the environment description.
TOOLCHAIN = (
    "openmm",
    "openff-toolkit",
    "openmmforcefields",
    "rdkit",
    "ambertools",
    "mdtraj",
    "numpy",
    "scipy",
)


def _version(dist: str) -> Optional[str]:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        pass
    # ambertools and rdkit are frequently conda packages without dist metadata; fall back to the
    # module's own __version__ before giving up.
    modname = {"openff-toolkit": "openff.toolkit", "ambertools": "parmed"}.get(dist, dist)
    try:
        import importlib

        mod = importlib.import_module(modname.replace("-", "_"))
        v = getattr(mod, "__version__", None)
        return str(v) if v else None
    except Exception:                              # noqa: BLE001
        return None


def toolchain_versions() -> dict[str, Optional[str]]:
    """Version of every package that can change a number, ``None`` where unavailable."""
    return {name: _version(name) for name in TOOLCHAIN}


def package_version() -> str:
    try:
        return metadata.version("md-templates")
    except metadata.PackageNotFoundError:
        return "unknown"


def python_version() -> str:
    """The interpreter that generated the project; part of the identity the release review asked for."""
    return sys.version.split()[0]


def package_location() -> str:
    """Where the installed package actually lives — the fastest way to catch a shadowed import."""
    import md_templates

    return str(Path(md_templates.__file__).resolve().parent)


def installed_from_wheel() -> dict[str, Optional[str]]:
    """The wheel this package came from, and its SHA-256, when that is recoverable.

    pip records the installed artifact in ``direct_url.json`` (PEP 610). If the wheel file is
    still on disk we hash it, so a consuming repository can prove which build it ran. A
    source/editable install reports ``null``, which is the honest answer rather than a guess.
    """
    out: dict[str, Optional[str]] = {"wheel": None, "wheel_sha256": None, "install_kind": None}
    try:
        dist = metadata.distribution("md-templates")
    except metadata.PackageNotFoundError:
        out["install_kind"] = "not-installed"
        return out
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:                              # noqa: BLE001
        raw = None
    if not raw:
        out["install_kind"] = "index-or-unknown"
        return out
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        out["install_kind"] = "unparseable-direct-url"
        return out
    url = str(info.get("url", ""))
    if info.get("dir_info", {}).get("editable"):
        out["install_kind"] = "editable"
        return out
    if url.startswith("file://") and url.endswith(".whl"):
        path = Path(url[len("file://"):])
        out["wheel"] = path.name
        out["install_kind"] = "wheel"
        if path.is_file():
            out["wheel_sha256"] = sha256_file(path)
        return out
    out["install_kind"] = "source" if url else "unknown"
    return out


def git_state() -> dict[str, Optional[str]]:
    """Git SHA and dirty state of the *source* checkout, when this is running from one.

    A wheel installed elsewhere has no checkout, and that is not an error — it reports
    ``available: false``. Provenance must never fail a run.
    """
    try:
        root = Path(__file__).resolve().parents[3]
        if not (root / ".git").exists():
            return {"available": False, "commit": None, "dirty": None}
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        # The remote URL and the tag are what let ANOTHER repository cite this method. A commit
        # alone identifies the tree but not where to get it; a tag alone moves.
        remote = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"],
                                capture_output=True, text=True, timeout=10)
        tag = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--exact-match"],
                             capture_output=True, text=True, timeout=10)
        nearest = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--abbrev=0"],
                                 capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=30)
        if commit.returncode != 0:
            return {"available": False, "commit": None, "dirty": None}
        return {
            "available": True,
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
            # The remote URL and tag are what let ANOTHER repository cite this method: a commit
            # alone identifies the tree but not where to get it, and a tag alone moves.
            "remote_url": (remote.stdout.strip() or None) if remote.returncode == 0 else None,
            "tag": (tag.stdout.strip() or None) if tag.returncode == 0 else None,
            "nearest_tag": (nearest.stdout.strip() or None) if nearest.returncode == 0 else None,
        }
    except Exception:                              # noqa: BLE001 - never break a run
        return {"available": False, "commit": None, "dirty": None}


class UnknownSourceIdentityError(RuntimeError):
    """Raised when the MD-templates commit that produced a project cannot be established.

    Generation fails on this rather than writing `commit: null`. A run whose method cannot be
    named is not reproducible, and a lock file that records the absence of a commit looks, to
    every later reader, exactly like one that was never asked.
    """


#: Where a source identity came from, most trustworthy first. Recorded in the lock file so a
#: reader can weigh it: a live checkout is authoritative, a build stamp is a claim made by the
#: build, PEP 610 is a claim made by the installer.
IDENTITY_SOURCES = ("git-worktree", "build-stamp", "pep610-vcs")


def _build_stamp() -> dict[str, Any]:
    """Identity embedded at build time, for a wheel installed away from any checkout."""
    try:
        from .. import _build_info as info          # generated; absent in a bare source tree
    except Exception:                              # noqa: BLE001
        return {}
    commit = getattr(info, "BUILD_COMMIT", None)
    if not commit:
        return {}
    return {"commit": commit,
            "remote_url": getattr(info, "BUILD_REMOTE_URL", None),
            "tag": getattr(info, "BUILD_TAG", None),
            "dirty": getattr(info, "BUILD_DIRTY", None),
            "identity_source": "build-stamp"}


def _pep610_vcs() -> dict[str, Any]:
    """Identity recorded by pip for a `pip install git+https://...` install (PEP 610).

    pip writes the RESOLVED commit into `vcs_info.commit_id`, so this is a full SHA even when the
    user installed a branch or tag. It says nothing about dirtiness -- a VCS install is by
    construction a clean checkout of that commit.
    """
    try:
        dist = metadata.distribution("md-templates")
        raw = dist.read_text("direct_url.json")
    except Exception:                              # noqa: BLE001
        return {}
    if not raw:
        return {}
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    vcs = info.get("vcs_info") or {}
    commit = vcs.get("commit_id")
    if not commit:
        return {}
    return {"commit": commit,
            "remote_url": info.get("url"),
            "tag": vcs.get("requested_revision"),
            "dirty": False,
            "identity_source": "pep610-vcs"}


def source_identity() -> dict[str, Any]:
    """The MD-templates commit behind this process, from whichever source can establish it.

    Order is by trustworthiness, not convenience:

    1. a live git worktree -- authoritative, and the only source that can observe dirtiness now;
    2. the build stamp -- what the wheel was built from, the case that has no `.git` at all;
    3. PEP 610 `vcs_info` -- what pip resolved for a `git+https` install.

    Returns `{}` when none of them can name a full commit. Callers that write provenance must
    treat that as fatal; `require_source_identity` does.
    """
    live = git_state()
    if live.get("available") and live.get("commit"):
        return {"commit": live["commit"],
                "remote_url": live.get("remote_url"),
                "tag": live.get("tag"),
                "nearest_tag": live.get("nearest_tag"),
                "dirty": live.get("dirty"),
                "identity_source": "git-worktree"}
    for candidate in (_build_stamp(), _pep610_vcs()):
        if candidate:
            return candidate
    return {}


def require_source_identity() -> dict[str, Any]:
    """`source_identity()`, but refusing to return anything a lock file could not stand behind."""
    identity = source_identity()
    commit = identity.get("commit")
    if not commit or len(str(commit)) != 40:
        raise UnknownSourceIdentityError(
            "the MD-templates commit that would produce this project cannot be determined, so the "
            "run could not be traced back to a method.\n"
            f"  package     : {package_version()} at {package_location()}\n"
            f"  install kind: {installed_from_wheel().get('install_kind')}\n"
            "Tried, in order: a git worktree around the package, the build stamp embedded by\n"
            "build_backend/md_templates_build.py, and PEP 610 direct_url.json vcs_info.\n"
            "Fix by installing from a built wheel or from git rather than from a plain copy of\n"
            "the source tree:\n"
            "  pip install md-templates@git+https://github.com/csy0000/MD-templates@<40-char sha>\n"
            "  # or, from a checkout:  python -m build && pip install dist/*.whl"
        )
    return identity


def sqm_path() -> Optional[str]:
    """AmberTools' semi-empirical engine — AM1-BCC charges are unobtainable without it."""
    found = shutil.which("sqm")
    return str(Path(found).resolve()) if found else None


#: The OpenMM release this repository is tested against, as the exact stable version string.
SUPPORTED_OPENMM_VERSION = "8.6.0"

#: The commit the upstream ``8.6.0`` tag points at, verified against
#: ``api.github.com/repos/openmm/openmm/git/ref/tags/8.6.0``.
#:
#: This is the field that actually establishes identity. A string comparison against "8.6.0" would
#: accept any build claiming that number; the tag commit is what distinguishes the release from a
#: development snapshot taken near it.
SUPPORTED_OPENMM_GIT_REVISION = "c6173db6e8edd705eb59172bd21e9ce69c572405"


def openmm_identity() -> dict[str, Any]:
    """Every version string OpenMM reports about itself, kept apart on purpose.

    OpenMM's packaged ``version.py`` carries ``release = False`` on the official conda-forge and
    PyPI builds, so ``version`` and ``full_version`` gain a ``.dev-<short sha>`` suffix *even for
    the tagged release*: 8.5.2 reports ``8.5.2.dev-36a30cb`` and 8.6.0 reports
    ``8.6.0.dev-c6173db``, where each short sha is that release's own tag commit. It is a build
    stamp, not evidence of a development snapshot.

    Reading `version` and concluding "this is a dev build" is therefore wrong, and reading it and
    recording it as "8.6.0" would be a false record. Both strings are kept:

    * ``short_version`` -- the exact stable release string, and the one to compare against;
    * ``full_version`` -- what the build actually reports, recorded verbatim so a manifest never
      claims a version the installation did not state;
    * ``git_revision`` -- the identity that matters, checked against the tag.
    """
    try:
        import openmm
        import openmm.version as v
    except Exception as exc:                       # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "short_version": getattr(v, "short_version", None),
        "version": getattr(v, "version", None),
        "full_version": getattr(v, "full_version", None),
        "git_revision": getattr(v, "git_revision", None),
        "release_flag": getattr(v, "release", None),
        "library_path": getattr(v, "openmm_library_path", None),
        "dist_metadata_version": _version("openmm"),
        "platforms": openmm_platforms(),
        "plugin_load_failures": list(openmm.Platform.getPluginLoadFailures()),
    }


def check_openmm_version(expected_version: str = SUPPORTED_OPENMM_VERSION,
                         expected_revision: Optional[str] = SUPPORTED_OPENMM_GIT_REVISION,
                         ) -> tuple[bool, str]:
    """``(ok, message)`` for "is this the OpenMM release this repository is validated against?".

    Both halves are required. ``short_version`` alone would pass a rebuild of any commit that calls
    itself 8.6.0; ``git_revision`` alone would not notice a package whose Python layer disagrees
    with its compiled library.
    """
    identity = openmm_identity()
    if not identity.get("available"):
        return False, f"OpenMM is not importable: {identity.get('error')}"

    short = identity.get("short_version")
    revision = identity.get("git_revision")
    problems = []
    if short != expected_version:
        problems.append(
            f"short_version is {short!r}, expected {expected_version!r}")
    if expected_revision and revision != expected_revision:
        problems.append(
            f"git_revision is {revision!r}, expected {expected_revision!r} "
            f"(the commit the {expected_version} tag points at)")
    if problems:
        return False, (
            "this is not the validated OpenMM build: " + "; ".join(problems)
            + f".  Reported full_version: {identity.get('full_version')!r}.  A '.dev-<sha>' suffix "
              "on full_version is normal for an official release and is not itself a failure -- "
              "compare short_version and git_revision instead."
        )
    return True, (
        f"OpenMM {short} (reported {identity.get('full_version')!r}, "
        f"tag commit {revision[:7]}), platforms {identity.get('platforms')}"
    )


def openmm_platforms() -> list[str]:
    try:
        import openmm

        return [openmm.Platform.getPlatform(i).getName()
                for i in range(openmm.Platform.getNumPlatforms())]
    except Exception:                              # noqa: BLE001
        return []


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cuda_devices": None,
    }
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            res = subprocess.run(
                [nvsmi, "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0:
                info["cuda_devices"] = [line.strip() for line in res.stdout.splitlines()
                                        if line.strip()]
        except Exception:                          # noqa: BLE001
            pass
    return info


def utc_timestamp() -> str:
    """``YYYY-MM-DDTHH:MM:SSZ``. UTC because run directories from several machines get compared."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_stamp() -> str:
    """``YYYYMMDDTHHMMSSZ`` — the sortable prefix of an immutable run directory."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def invocation() -> dict[str, Any]:
    """The exact command, recorded without inventing paths that were not typed."""
    return {
        "argv": list(sys.argv),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "timestamp_utc": utc_timestamp(),
    }


def environment_block() -> dict[str, Any]:
    """The toolchain + platform half of any manifest this package writes."""
    return {
        "package_version": package_version(),
        "package_location": package_location(),
        "install": installed_from_wheel(),
        "git": git_state(),
        "python": sys.version.split()[0],
        "toolchain": toolchain_versions(),
        "sqm_path": sqm_path(),
        "openmm": openmm_identity(),
        "openmm_platforms": openmm_platforms(),
        "hardware": hardware(),
    }


def hash_tree(directory: Path, names: list[str]) -> dict[str, str]:
    """SHA-256 of each named file in `directory`; raises if one is missing."""
    directory = Path(directory)
    out: dict[str, str] = {}
    missing = [n for n in names if not (directory / n).is_file()]
    if missing:
        raise FileNotFoundError(f"missing from {directory}: {', '.join(sorted(missing))}")
    for name in names:
        out[name] = sha256_file(directory / name)
    return out


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
                    encoding="utf-8")
    return path
