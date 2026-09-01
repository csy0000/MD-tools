"""What is in a dataset, by relative path, size and digest.

The inventory is what makes "the copy is correct" a checkable statement instead of a hope.
`shutil.move` returning without raising says the call did not error; it says nothing about whether
every byte arrived, which is a different claim and the only one that matters across filesystems.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..build.record import sha256_file
from .errors import RegistrationError

INVENTORY_NAME = "SHA256SUMS"

#: Never part of the inventory: the inventory cannot contain its own digest, and the manifests are
#: written after it.
EXCLUDED = frozenset({INVENTORY_NAME, "dataset.yaml", "dataset.resolved.yaml",
                      "dataset.draft.yaml"})


def walk(root: Path) -> list[Path]:
    """Every regular file beneath `root`, sorted, with symlinks refused.

    A symlink inside a dataset would either dangle after the move or point outside it, and either
    way the inventory would describe something other than what a reader gets.
    """
    root = Path(root)
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RegistrationError(
                f"{path.relative_to(root)} is a symlink. A dataset holds its own bytes: a link "
                f"either dangles once the dataset moves or points at content this dataset does "
                f"not own. Replace it with the file, or exclude it from the dataset.")
        if path.is_dir():
            continue
        if path.name in EXCLUDED and path.parent == root:
            continue
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def build(root: Path) -> list[dict[str, Any]]:
    """The inventory: relative path, size and SHA-256 for every file."""
    root = Path(root).resolve()
    return [{"path": str(path.relative_to(root)),
             "bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in walk(root)]


def write(root: Path, entries: list[dict[str, Any]]) -> Path:
    """Write SHA256SUMS in the format `sha256sum -c` reads."""
    root = Path(root)
    body = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)
    path = root / INVENTORY_NAME
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


def verify(root: Path, entries: list[dict[str, Any]]) -> None:
    """Re-read the destination and prove it matches, or raise naming every difference."""
    root = Path(root).resolve()
    expected = {entry["path"]: entry for entry in entries}
    problems: list[str] = []
    for relative, entry in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            problems.append(f"  missing: {relative}")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            problems.append(f"  size differs: {relative} ({size} != {entry['bytes']})")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            problems.append(f"  digest differs: {relative} "
                            f"({digest[:12]}... != {entry['sha256'][:12]}...)")
    present = {str(p.relative_to(root)) for p in walk(root)}
    for extra in sorted(present - set(expected)):
        problems.append(f"  unexpected file at the destination: {extra}")
    if problems:
        raise RegistrationError(
            f"the destination {root} does not match the inventory:\n" + "\n".join(problems))
