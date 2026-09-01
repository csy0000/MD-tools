"""Durable transaction state, so an interrupted registration can be finished safely.

Registration moves data it cannot regenerate. Every step is therefore ordered so that an
interruption between any two steps leaves a state that is either recoverable or already correct,
and the state is written to disk before the step it authorises -- not after.

The order, and what each boundary guarantees:

    planned          nothing has been written. Interrupting here loses nothing.
    staged           the bytes are at a TEMPORARY destination beside the final one. The final
                     path does not exist yet, so an interruption cannot be mistaken for success.
    verified         the staged bytes have been re-read and every digest matches. Only now is
                     the copy known to be correct.
    committed        the staged directory has been renamed to the final path, atomically.
    source-removed   only after commit. The source is the only other copy, so it is the last
                     thing to go.
    linked           the source path is a symlink to the destination.
    complete         a final validation passed.

`shutil.move` is never used as evidence. Across filesystems it degrades to copy-then-delete, and
a copy that returns without raising has still not been shown to have arrived intact.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import RegistrationError

STATE_FORMAT = "md-tools-registration/2.0"

ORDER = ("planned", "staged", "verified", "committed", "source-removed", "linked", "complete")


class Transaction:
    """The durable record of one registration."""

    def __init__(self, path: Path, *, plan: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        if plan is not None:
            self.state = {"format": STATE_FORMAT, "stage": "planned",
                          "started_utc": _now(), "history": [], **plan}
        else:
            self.state = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RegistrationError(f"{self.path}: unreadable transaction state -- {exc}") from None
        if document.get("format") != STATE_FORMAT:
            raise RegistrationError(
                f"{self.path}: transaction state format is {document.get('format')!r}, this "
                f"build writes {STATE_FORMAT!r}. Refusing to resume a transaction written by a "
                f"different version rather than guessing what its stages meant.")
        if document.get("stage") not in ORDER:
            raise RegistrationError(f"{self.path}: unknown stage {document.get('stage')!r}")
        return document

    @property
    def stage(self) -> str:
        return str(self.state["stage"])

    def advance(self, stage: str, **extra: Any) -> None:
        """Record reaching `stage`, durably, BEFORE doing what that stage authorises."""
        if stage not in ORDER:
            raise RegistrationError(f"unknown stage {stage!r}")
        if ORDER.index(stage) < ORDER.index(self.stage):
            raise RegistrationError(f"cannot go back from {self.stage!r} to {stage!r}")
        self.state["history"].append({"stage": stage, "at": _now()})
        self.state["stage"] = stage
        self.state.update(extra)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".partial")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        # fsync before the rename: the rename is what makes the new state visible, and a state
        # that is visible but not durable is worse than one that is neither.
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    @classmethod
    def find(cls, path: Path) -> "Transaction | None":
        path = Path(path)
        return cls(path) if path.is_file() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def copy_tree(source: Path, destination: Path, *, entries: list[dict[str, Any]]) -> None:
    """Copy every inventoried file, creating parents, preserving mtimes.

    Explicitly file-by-file from the inventory rather than `shutil.copytree`, so that what is
    copied is exactly what was inventoried and verified -- not whatever the tree happened to
    contain at copy time, which may differ if anything was still writing.
    """
    import shutil

    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / entry["path"], target)
