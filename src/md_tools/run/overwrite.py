"""Replacing a run's outputs, as one transaction over the complete owned inventory.

WHY THIS IS NOT `rmtree` PLUS HOPE

    `--overwrite` used to remove the checkpoint tree and leave the rest to whichever reporter
    happened to open its file in `"w"`. That is three separate promises made by three different
    pieces of code, and the ones nobody made were the ones that mattered:

      a stream the new run does NOT write is not truncated by anybody. Turn off phase-space
      reporting and `--overwrite`, and the old `.phase_space.nc` is still sitting there, still
      named by the inventory, still looking like an output of the run that just finished;

      a CV definition sidecar from a previous cv.yaml survives a run with different columns;

      a crash between "removed some" and "opened the new ones" leaves a directory that is part
      one experiment and part another, with every file equally current-looking.

    So: everything owned moves out of the way FIRST, in one pass, before any new output is
    opened, and a marker on disk says a replacement is in progress. A crash leaves the marker,
    and the next invocation is told what it found rather than quietly running on top of it.

    Nothing outside the inventory is touched. The inventory is the definition of "owned", which
    is why it has to be complete -- and a file this module does not know about is a file it must
    not delete.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

#: Left behind while outputs are being moved aside, and removed when the move has finished.
#: Its presence means a previous `--overwrite` did not get to the end.
MARKER_NAME = ".md-tools-replacing.json"


class ReplacementError(SystemExit):
    """A replacement that cannot be completed safely. Refuses rather than half-applies."""


def _sync_directory(path: Path) -> None:
    try:
        handle = os.open(str(path), os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def owned_paths(inventory) -> list[Path]:
    """Every path the inventory claims, deduplicated, longest first.

    Longest first so a file inside a directory the inventory also names is moved before the
    directory is -- otherwise the directory move takes the file with it and the per-path record
    of what happened is wrong.
    """
    roles = getattr(inventory, "roles", None) or {}
    unique = {str(Path(path)): Path(path) for path in roles.values()}
    return sorted(unique.values(), key=lambda p: len(str(p)), reverse=True)


def find_incomplete_replacement(directory: Path) -> dict[str, Any] | None:
    """The marker a crashed replacement leaves, or None."""
    marker = Path(directory) / MARKER_NAME
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except ValueError:
        return {"unreadable": True}


def replace_owned_inventory(inventory, *, where: str, directory: Path | None = None,
                            keep: frozenset[str] = frozenset()) -> dict[str, str]:
    """Move every owned output aside, then delete it. Returns `{role: what happened}`.

    Ordering is the whole contract:

      1. write the marker, so a crash from here on is identifiable;
      2. move every existing owned path into a staging directory -- one `os.replace` each, so
         each individual path is either old-and-present or gone, never truncated;
      3. fsync the directory, so the moves are durable before anything new is created;
      4. delete the staging directory;
      5. remove the marker.

    A crash before (3) leaves some outputs in place and the marker present. A crash after (3)
    leaves nothing at any output path. In neither case does a NEW output exist yet, because this
    runs before the first one is opened -- which is the property that matters and the reason this
    is a separate function called at a specific point rather than something each reporter does
    for itself.
    """
    paths = [path for path in owned_paths(inventory)
             if path.exists() or path.is_symlink()]
    roles = {str(Path(path)): role for role, path in (getattr(inventory, "roles", None) or {}).items()}
    if directory is None:
        candidates = {p.parent for p in owned_paths(inventory)}
        directory = sorted(candidates, key=lambda p: len(str(p)))[0] if candidates else Path(".")
    directory = Path(directory)
    if not paths:
        return {}

    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".md-tools-replaced-{os.getpid()}-{int(time.time())}"
    marker = directory / MARKER_NAME
    marker.write_text(json.dumps({
        "what": where,
        "staging": staging.name,
        "paths": [str(path) for path in paths],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")

    staging.mkdir(parents=True, exist_ok=True)
    happened: dict[str, str] = {}
    try:
        for index, path in enumerate(paths):
            role = roles.get(str(path), path.name)
            if role in keep:
                happened[role] = "kept"
                continue
            # A unique name per path: two roles can share a basename (`cMD.csv` beside
            # `cMD.cv.csv` do not, but a caller's inventory may), and a collision inside the
            # staging directory would silently drop one of them.
            target = staging / f"{index:03d}-{path.name}"
            os.replace(str(path), str(target))
            happened[role] = "replaced"
        _sync_directory(directory)
    except OSError as failure:
        raise ReplacementError(
            f"{where}: could not move {path} aside to replace it ({failure}). Nothing new has "
            f"been written; the outputs still present are the previous run's. Resolve the "
            f"permission or filesystem problem, or choose a different -odir.") from None

    shutil.rmtree(staging, ignore_errors=True)
    marker.unlink(missing_ok=True)
    _sync_directory(directory)
    return happened
