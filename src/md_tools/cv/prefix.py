"""The committed prefix of a collective-variable series, and what makes it trustworthy.

WHY A ROW COUNT IS NOT ENOUGH

    A checkpoint that records only "12 rows were committed" detects a file that is SHORTER than
    12. It says nothing about whether those 12 rows are still the ones that were committed. A
    value edited in place, a step renumbered, a walker index changed -- the count still agrees,
    the continuation appends onto them, and the finished series is part measurement and part
    edit with nothing marking the boundary.

    So the generation records a DIGEST of exactly the header plus the committed rows. Not of the
    whole file: after a crash the file is legitimately longer than the checkpoint, because rows
    are flushed as they are written and the commit happens afterwards. Hashing the uncommitted
    tail as part of the selected generation would make every ordinary crash look like corruption.

WHY THE STRUCTURE IS PARSED TOO

    The digest proves the bytes are unchanged. It does not say what they mean, and a continuation
    is about to extend them -- so the prefix is also read as a table: exact columns, finite
    values, a monotonic grid with no duplicates or gaps, the identifiers this run expects. A
    mismatch is refused BEFORE a byte is written, because a continuation that has already
    truncated cannot decide afterwards that it should have refused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class CVPrefixError(ValueError):
    """A committed CV prefix that cannot be trusted. Refused rather than appended to."""


def read_lines(path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise CVPrefixError(f"{path} does not exist")
    return path.read_text(encoding="utf-8").splitlines()


def prefix_digest(path, rows: int) -> str:
    """sha256 of the header plus exactly the first `rows` data lines, newline-joined.

    The uncommitted tail is deliberately excluded -- see the module note.
    """
    lines = read_lines(path)
    if not lines:
        raise CVPrefixError(f"{path} is empty, so it has no committed prefix")
    rows = int(rows)
    if len(lines) - 1 < rows:
        raise CVPrefixError(
            f"{Path(path).name} holds {len(lines) - 1} row(s) and {rows} were committed. Records "
            f"the checkpoint believes exist have been lost; that is not a resumable state")
    body = "\n".join(lines[:1 + rows]) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def file_digest(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, *, rows: int, sidecar=None, definition=None, cost=None) -> dict:
    """What a checkpoint generation stores about one CV series, for a later continuation."""
    entry = {
        "rows": int(rows),
        "prefix_sha256": prefix_digest(path, rows),
        "sidecar_sha256": file_digest(sidecar) if sidecar else None,
    }
    if definition is not None:
        entry["schema_version"] = definition.schema_version
        entry["definition_sha256"] = definition.digest
        entry["columns"] = list(definition.names)
    if cost:
        entry["cost"] = dict(cost)
    return entry


def validate(path, entry, *, sidecar=None, definition=None, expect_columns=None,
             identifiers=None, interval=None, value_columns=None):
    """Every reason this committed prefix may not be appended to. Raises, or returns the count.

    `identifiers` maps a column name to the value every row must carry -- the state index and tau
    for a ladder, the path and source-frame index for AIS. Those are what distinguish one
    series from another that is otherwise identically shaped, which is exactly the confusion a
    swapped or misfiled series creates.
    """
    if entry is None or entry.get("rows") is None or not entry.get("prefix_sha256"):
        raise CVPrefixError(
            f"{Path(path).name}: this checkpoint does not record a committed collective-variable "
            f"prefix (row count and digest). It was written by a build that reported CVs without "
            f"binding them into the checkpoint transaction, so which rows are durable cannot be "
            f"established and a continuation would either duplicate or silently keep edited rows. "
            f"Start a fresh run with --overwrite.")

    rows = int(entry["rows"])
    actual = prefix_digest(path, rows)          # raises if the file is shorter than committed
    if actual != entry["prefix_sha256"]:
        raise CVPrefixError(
            f"{Path(path).name}: the {rows} committed row(s) are not the rows that were "
            f"committed -- their digest does not match the checkpoint. A value, a step or an "
            f"identifier has been edited in place. Refusing before anything is truncated or "
            f"appended.")

    if sidecar is not None and entry.get("sidecar_sha256") is not None:
        if file_digest(sidecar) != entry["sidecar_sha256"]:
            raise CVPrefixError(
                f"{Path(sidecar).name} has changed since the checkpoint was written, so the "
                f"committed rows would be read under a different interpretation")

    if definition is not None and entry.get("definition_sha256") not in (None,
                                                                        definition.digest):
        raise CVPrefixError(
            f"{Path(path).name}: the collective-variable definition has changed since the "
            f"checkpoint. Appending would put a different measurement under the same headings")

    lines = read_lines(path)
    header = lines[0].split(",")
    if expect_columns is not None and header != list(expect_columns):
        raise CVPrefixError(
            f"{Path(path).name}: columns are {header} and this run writes "
            f"{list(expect_columns)}")

    steps: list[int] = []
    for position, line in enumerate(lines[1:1 + rows]):
        fields = line.split(",")
        if len(fields) != len(header):
            raise CVPrefixError(
                f"{Path(path).name}: committed row {position} has {len(fields)} field(s), "
                f"not {len(header)}")
        cells = dict(zip(header, fields))
        try:
            steps.append(int(cells[header[0]]))
        except ValueError:
            raise CVPrefixError(
                f"{Path(path).name}: committed row {position} has a non-integer step "
                f"{cells[header[0]]!r}") from None
        for column, expected in (identifiers or {}).items():
            if column not in cells:
                raise CVPrefixError(f"{Path(path).name}: no {column!r} column to check")
            if not _same(cells[column], expected):
                raise CVPrefixError(
                    f"{Path(path).name}: committed row {position} reports {column}="
                    f"{cells[column]!r} and this run is {expected!r}. The series belongs to a "
                    f"different state or path")
        # The CV VALUE columns specifically. The caller names them because only it knows where
        # its protocol columns end -- an earlier draft computed the boundary here and got it
        # wrong in a way that made this loop iterate over nothing, so the finiteness check
        # silently tested no values at all.
        for column in (value_columns
                       if value_columns is not None
                       else (list(definition.names) if definition is not None else [])):
            value = cells.get(column)
            if value is None:
                continue
            try:
                number = float(value)
            except ValueError:
                raise CVPrefixError(
                    f"{Path(path).name}: committed row {position} has a non-numeric value "
                    f"{value!r} in {column}") from None
            if number != number or number in (float("inf"), float("-inf")):
                raise CVPrefixError(
                    f"{Path(path).name}: committed row {position} has a non-finite value in "
                    f"{column}")

    if steps != sorted(set(steps)):
        raise CVPrefixError(
            f"{Path(path).name}: the committed steps are not strictly increasing "
            f"({steps[:8]}...), so the series has a duplicate or is out of order")
    if interval:
        expected_grid = [steps[0] + n * int(interval) for n in range(len(steps))] if steps else []
        if steps != expected_grid:
            raise CVPrefixError(
                f"{Path(path).name}: the committed steps are not on the declared "
                f"{interval}-step grid ({steps[:8]}...), so a row is missing or misnumbered")
    return rows


def _same(text, expected) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(text) - expected) <= 1e-9
        except ValueError:
            return False
    return str(text) == str(expected)


def truncate(path, rows: int) -> int:
    """Cut the file back to the committed prefix. Returns how many rows it now holds.

    Called only AFTER `validate` has accepted the prefix: a continuation that has already
    truncated cannot decide afterwards that it should have refused.
    """
    lines = read_lines(path)
    rows = int(rows)
    Path(path).write_text("\n".join(lines[:1 + rows]) + "\n", encoding="utf-8")
    return rows
