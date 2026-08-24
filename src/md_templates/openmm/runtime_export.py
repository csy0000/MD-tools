"""Copy the package into a generated project, so the project outlives the checkout that made it.

A generated project is normally moved: to a cluster, to a collaborator, to an archive. Until now its
launchers carried the absolute path of the MD-templates checkout in `PYTHONPATH`, so the project ran
only on the machine that made it, only while that checkout existed, and -- worse -- silently picked
up whatever that checkout had become. Editing the template could change the behaviour of a project
generated months earlier.

Exporting a snapshot fixes both. `runtime/md_templates/` is the package as it was at generation
time, and the launchers point at it. A later `git pull` in the template cannot reach it.

What is deliberately NOT bundled: OpenMM, CUDA, Python itself. Those are prerequisites, documented
in the generated README. Portability here means "does not need the template checkout", not "does not
need a scientific stack" -- vendoring CUDA would be a different and much worse idea.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .hashing import sha256_file

__all__ = ["RUNTIME_DIR_NAME", "export_runtime", "runtime_manifest"]

RUNTIME_DIR_NAME = "runtime"

#: Directories inside the package that never belong in a snapshot.
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}

#: Suffixes that are build artefacts rather than source or data.
_SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so.orig"}


def _should_copy(path: Path) -> bool:
    if any(part in _SKIP_DIRS for part in path.parts):
        return False
    return path.suffix not in _SKIP_SUFFIXES


def export_runtime(out_dir: Path) -> dict:
    """Copy the `md_templates` package into `<out_dir>/runtime/` and describe what was written.

    Package DATA is copied as well as code -- profiles, schemas, manifests, worked examples. A
    snapshot that carried only `.py` files would import and then fail at the first profile lookup,
    which is a worse failure than not copying at all because it happens later.
    """
    import md_templates

    source = Path(md_templates.__file__).resolve().parent
    runtime = Path(out_dir) / RUNTIME_DIR_NAME
    target = runtime / "md_templates"
    if runtime.exists():
        shutil.rmtree(runtime)
    target.parent.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for item in sorted(source.rglob("*")):
        if not item.is_file() or not _should_copy(item.relative_to(source)):
            continue
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        copied.append(str(destination.relative_to(runtime)))

    (runtime / "README.txt").write_text(
        "This directory is a snapshot of the md_templates package as it existed when this project\n"
        "was generated. The launchers put it on PYTHONPATH, so the project does not need the\n"
        "MD-templates checkout it came from, and a later change to that checkout cannot alter this\n"
        "project's behaviour.\n\n"
        "OpenMM, CUDA and the Python interpreter are NOT bundled. They are prerequisites; see the\n"
        "project README.\n", encoding="utf-8")
    return {"runtime_dir": str(runtime), "n_files": len(copied), "files": copied}


def runtime_manifest(out_dir: Path) -> dict:
    """SHA-256 of every exported runtime file, so a relocated project can prove it is intact."""
    runtime = Path(out_dir) / RUNTIME_DIR_NAME
    if not runtime.is_dir():
        return {}
    return {str(p.relative_to(runtime)): sha256_file(p)
            for p in sorted(runtime.rglob("*")) if p.is_file()}
