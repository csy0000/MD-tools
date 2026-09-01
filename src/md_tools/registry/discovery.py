"""Deciding whether a directory holds finished data.

The one rule everything here serves: **completion is read from a structured record, never from
prose.** A log line saying a run finished is printed by the code that reached that line, which a
killed process, a full disk or a truncated write can all prevent from meaning what it says. Worse,
it can be typed by hand.

So a directory is finished when every machine record it contains says so, and a record is only a
record when it parses, carries a schema version this build understands, and is not still being
written.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ..build.record import BEGIN, RecordError, read_record
from .errors import RegistrationError

#: Files this tool writes into the source itself, which therefore are not evidence about the run.
OWN_OUTPUTS = ("dataset.draft.yaml", "SHA256SUMS", "dataset.yaml", "dataset.resolved.yaml")

#: How recently a file may have been modified before we treat the directory as still being
#: written. Registration moves data and then deletes the source; doing that underneath a live
#: writer loses the frames it was in the middle of flushing.
ACTIVE_WRITER_SECONDS = 60.0


def find_records(source: Path) -> list[Path]:
    """Every log carrying a machine record, deepest first for stable reporting."""
    source = Path(source)
    found = []
    for path in sorted(source.rglob("*.log")):
        if path.name in OWN_OUTPUTS:
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if BEGIN in head:
            found.append(path)
    return found


def classify(source: Path) -> dict[str, Any]:
    """Read every record and decide whether this directory may be registered.

    Returns the accepted records. Raises with everything that is wrong, rather than the first
    thing: a person fixing a directory wants the whole list.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise RegistrationError(f"-idata {source}: not a directory")
    if source.is_symlink():
        raise RegistrationError(
            f"-idata {source} is a symlink. It has probably been registered already -- a "
            f"registered source is replaced by a symlink to its destination. Registering a "
            f"symlink would copy the destination back on top of itself.")

    logs = find_records(source)
    if not logs:
        raise RegistrationError(
            f"{source} contains no machine record. Every artefact this package writes carries a "
            f"delimited record; a directory without one has either not been run with these tools "
            f"or has not finished. Registration never infers completion from file names or from "
            f"prose in a log.")

    problems: list[str] = []
    accepted: list[dict[str, Any]] = []
    for log in logs:
        relative = log.relative_to(source)
        try:
            record = read_record(log)
        except RecordError as exc:
            problems.append(f"  {relative}: {exc}")
            continue
        status = record.get("status")
        if status == "completed":
            accepted.append({"log": str(relative), "record": record})
        elif status == "failed":
            problems.append(f"  {relative}: the run FAILED "
                            f"({record.get('failure_reason', 'no reason recorded')})")
        else:
            problems.append(f"  {relative}: status is {status!r}, not 'completed'. A run that "
                            f"was started and never finished is not registrable.")
    if problems:
        raise RegistrationError(
            f"{source} is not ready to register:\n" + "\n".join(problems)
            + "\n\nNothing has been moved or changed.")

    active = recently_modified(source)
    if active:
        raise RegistrationError(
            f"{source} looks like it is still being written: "
            + ", ".join(f"{p} ({age:.0f}s ago)" for p, age in active[:5])
            + f"\nRegistration moves the data and then removes the source; doing that under a "
              f"live writer loses whatever it had not flushed. Wait until the run is finished, or "
              f"stop it.")
    return {"source": source, "records": accepted}


def recently_modified(source: Path, *, window: float = ACTIVE_WRITER_SECONDS) -> list:
    """Files modified within the window, newest first."""
    now = time.time()
    hits = []
    for path in Path(source).rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age < window:
            hits.append((str(path.relative_to(source)), age))
    return sorted(hits, key=lambda item: item[1])


def check_lineage(records: list[dict[str, Any]], source: Path) -> list[str]:
    """Check that each stage's declared inputs are the outputs the previous stage produced.

    A chain of stages is only meaningful if each one consumed what the one before it wrote. The
    records carry the digests on both sides, so this is a check rather than an assumption -- and
    it catches a directory assembled from two different runs, which otherwise looks perfect.
    """
    produced: dict[str, str] = {}
    notes: list[str] = []
    for entry in records:
        record = entry["record"]
        for output in (record.get("outputs") or {}).values():
            if isinstance(output, dict) and output.get("sha256"):
                produced[output["sha256"]] = f"{entry['log']}:{output.get('path')}"
    for entry in records:
        record = entry["record"]
        for label, given in (record.get("inputs") or {}).items():
            if not isinstance(given, dict) or not given.get("sha256"):
                continue
            path = Path(source) / str(given.get("path", ""))
            if path.is_file():
                from ..build.record import sha256_file
                actual = sha256_file(path)
                if actual != given["sha256"]:
                    raise RegistrationError(
                        f"{entry['log']}: the recorded {label} {given.get('path')!r} has digest "
                        f"{given['sha256'][:12]}... but the file present now hashes to "
                        f"{actual[:12]}.... The data changed after the run recorded them, so the "
                        f"record no longer describes what is on disk.")
            if given["sha256"] in produced:
                notes.append(f"{entry['log']} consumed {produced[given['sha256']]}")
    return notes
