"""What a saved scaled Hamiltonian IS: `build/<method>/system_state<n>.xml` and its `scaler.yaml`.

`md-openmm build-top --rest2-scaler` writes the states and the record (`md_tools.build.scaler`).
This module is the READING half, and it deliberately imports nothing from `md_tools.build`: every
runtime preflight -- a stage, a ladder, AIS -- asks it the same question, and they must not need the
build machinery to do so.

A SCALED FILE IS IDENTIFIED BY DIGEST, NEVER BY A GUESS AT ITS PARAMETERS. The record beside it
lists every state it wrote with its sha256, so "is this System already scaled, and to what" has an
answer that does not depend on reading force constants and deciding they look small. Scaling one
again takes solute-solute to (1-tau)^4 with entirely plausible numbers; a preflight refuses that by
asking here first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

#: The record `build-top --rest2-scaler` writes beside the states it scaled.
RECORD_NAME = "scaler.yaml"
#: Bumped when the record's shape changes; a reader refuses a format it was not written for.
RECORD_FORMAT = "md-tools-scaled-states/v1"


class ScaledStateError(ValueError):
    """A saved scaled state, or the record describing it, cannot be trusted."""


def state_system_name(index: int) -> str:
    """`system_state<n>.xml`. The one spelling."""
    return f"system_state{int(index)}.xml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scaler_record(path: str | Path) -> dict[str, Any]:
    """Read a `scaler.yaml`, refusing one that is not a complete record of this format."""
    path = Path(path)
    try:
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as broken:
        raise ScaledStateError(f"{path} could not be read as a scaler record: {broken}") from None
    if not isinstance(record, dict) or record.get("format") != RECORD_FORMAT:
        found = record.get("format") if isinstance(record, dict) else type(record).__name__
        raise ScaledStateError(
            f"{path} is not a {RECORD_FORMAT} record (format: {found!r}). Rebuild the states with "
            f"`md-openmm build-top --rest2-scaler`.")
    missing = [key for key in ("method", "schedule", "states", "source", "unscaled_torsions")
               if key not in record]
    if missing or not isinstance(record["states"], list) or not record["states"]:
        absent = missing or ["states"]
        raise ScaledStateError(f"{path} is an incomplete scaler record (missing {absent}); "
                               f"rebuild the states rather than trusting it.")
    return record


def scaled_state_identity(system_path: str | Path) -> dict[str, Any] | None:
    """Which scaled state this System file is, or None if no scaler record beside it names it.

    Returns ``{"record", "method", "state", "tau", "system_sha256"}``. None means there is no
    `scaler.yaml` in the file's directory -- an ordinary System, e.g. `build/built.xml`, or a
    parameter-only end state built some other way.

    RAISES rather than returning None when a record IS there and does not vouch for the file: a file
    it does not list, or one whose sha256 has changed since the build. Both are a directory that
    claims to describe its Systems and no longer does, and treating the file as unscaled is exactly
    the mistake that would scale it twice.
    """
    path = Path(system_path).resolve()
    record_path = path.parent / RECORD_NAME
    if not record_path.is_file():
        return None
    record = load_scaler_record(record_path)
    entry = next((state for state in record["states"] if state.get("file") == path.name), None)
    if entry is None:
        listed = [state.get("file") for state in record["states"]]
        raise ScaledStateError(
            f"{path.name} sits beside {record_path}, which lists only {listed}. A System in a "
            f"scaled-state directory that its record does not describe cannot be told apart from "
            f"a scaled one; move it out, or rebuild with `md-openmm build-top --rest2-scaler`.")
    digest = _sha256(path)
    if digest != entry.get("sha256"):
        raise ScaledStateError(
            f"{path.name} has sha256 {digest}, but {record_path} recorded {entry.get('sha256')} "
            f"when it was built. The file changed after the build, so what it scales is no longer "
            f"what the record says; rebuild with `md-openmm build-top --rest2-scaler`.")
    return {"record": str(record_path), "method": record["method"],
            "state": int(entry["state"]), "tau": float(entry["tau"]), "system_sha256": digest}
