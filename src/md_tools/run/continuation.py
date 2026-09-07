"""Validate what an invocation intends to CONTINUE, before it is allowed to write anything.

WHY THIS IS A MODULE AND NOT A STEP INSIDE EACH RUNTIME

    Every protocol already validated its committed collective-variable records -- and every one
    of them did it after opening its `.out`, its `.log` and its run-state record. So a refused
    continuation had already replaced the prior run's machine-readable provenance and completion
    status with a description of an invocation that never started. "Completion is read from a
    machine record" is precisely why those files may not move: the rejected attempt must not
    become the authoritative account of the run it declined to continue.

    The refusal reports itself on stderr, which lives outside the protected tree. Once preflight
    succeeds and execution begins, ordinary runtime logging resumes -- that is the other phase,
    and it is unchanged.

WHOLE-OPERATION, NOT FIRST-FAILURE

    For AIS every selected path is validated before work begins on any of them, and for a ladder
    every state is validated before either parent or destination is touched. Validating lazily
    means an invalid later path is discovered once an earlier one has already been truncated,
    which is the same defect in a different order.

READ-ONLY IS THE POINT

    Nothing here opens a file for writing, creates a directory, or replaces a pointer. It reads
    records, checks them against the protocol this invocation resolved, and raises. Reading
    updates access times; that is not mutation and callers do not treat it as such.
"""

from __future__ import annotations

from pathlib import Path


class ContinuationError(RuntimeError):
    """Existing data that may not be continued or verified. Raised before any output moves."""


def _cv_interval(document) -> int:
    """The configured CV cadence, or 0 when this run reports no collective variables."""
    block = (document or {}).get("collective_variables") or {}
    try:
        return int(block.get("interval_steps") or 0)
    except (TypeError, ValueError):
        return 0


def validate_ladder_continuation(out_dir, *, definition, taus, interval_steps,
                                 checkpoint_path) -> None:
    """A ladder's committed CV block, validated before the run opens a single file.

    Skipped entirely when there is nothing to continue: a fresh directory has no checkpoint, and
    a CV-disabled run has no definition. Neither is a refusal.
    """
    if definition is None or not interval_steps:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return

    from ..remd import storage
    from ..remd.cv_states import CVContinuationError, validate_for_continuation, validate_prefixes

    try:
        checkpoint = storage.ReplicaCheckpoint(checkpoint_path).read()
    except storage.StorageError:
        # Not a readable checkpoint. The ordinary continuation path reports that far better than
        # this one can, and refusing here would pre-empt its message with a worse one.
        return
    extra = checkpoint.get("extra") or {}
    if extra.get("cv_prefix") is None and extra.get("cv_rows") is None:
        return

    directory = Path(out_dir)
    try:
        validate_for_continuation(directory, definition, taus=list(taus),
                                  interval_steps=int(interval_steps),
                                  committed_rows=extra.get("cv_rows"))
        validate_prefixes(directory, definition, taus=list(taus),
                          interval_steps=int(interval_steps), block=extra.get("cv_prefix"))
    except CVContinuationError as refusal:
        raise ContinuationError(str(refusal)) from None


def validate_ais_continuation(out_dir, *, definition, schedule, chosen, selected) -> None:
    """EVERY selected AIS path, before work begins on any of them.

    A completed path is verified by the same rules that would refuse it on a later skip; a
    partial path's committed prefix is validated by the same rules that would refuse it at the
    truncation. Doing this for the whole selection first is what stops an invalid path 3 from
    being discovered after path 0 has already been rewritten.
    """
    if definition is None or not int(schedule.get("cv_interval_steps") or 0):
        return
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return

    from ..ais.run import CV_CSV, COMPLETION_NAME, _validate_cv_prefix, _verify_cv_series
    from ..openmm.checkpoint import CheckpointError, read_committed

    problems: list[str] = []
    for index in selected:
        directory = out_dir / f"path_{int(index):04d}"
        if not directory.is_dir():
            continue
        marker = directory / COMPLETION_NAME
        if marker.is_file():
            import json

            try:
                record = json.loads(marker.read_text(encoding="utf-8"))
            except ValueError as broken:
                problems.append(f"path {index}: {marker.name} is not readable JSON ({broken})")
                continue
            try:
                _verify_cv_series(directory / CV_CSV, record, schedule=schedule,
                                  index=int(index), frame=chosen[int(index)], marker=marker)
            except SystemExit as refusal:
                problems.append(str(refusal))
            continue

        try:
            committed = read_committed(directory)
        except CheckpointError:
            continue
        state = (committed or {}).get("state") or {}
        entry = state.get("cv_prefix")
        if entry is None:
            continue
        try:
            from ..ais.run import CV_COLUMNS

            _validate_cv_prefix(directory / CV_CSV, entry, definition=definition,
                                columns=list(CV_COLUMNS) + list(definition.names),
                                index=int(index), frame=chosen[int(index)],
                                interval=int(schedule["cv_interval_steps"]))
        except SystemExit as refusal:
            problems.append(str(refusal))

    if problems:
        raise ContinuationError(
            "the existing collective-variable output cannot be continued or verified:\n  - "
            + "\n  - ".join(problems))


def validate_stage_continuation(destination, *, stage, definition) -> None:
    """A cMD stage's committed prefix, before the stage opens its own records.

    The stage continues automatically from its committed generation, so this is the ordinary
    resume path and the point at which it would truncate its appendable streams.
    """
    if definition is None:
        return
    destination = Path(destination)
    if not destination.is_dir():
        return

    from ..cv.prefix import CVPrefixError
    from ..openmm.checkpoint import POINTER_NAME, CheckpointError, read_committed

    problems: list[str] = []
    for checkpoints in sorted(destination.rglob("*.checkpoints")):
        if not (checkpoints / POINTER_NAME).is_file():
            continue
        try:
            committed = read_committed(checkpoints)
        except CheckpointError:
            continue
        state = (committed or {}).get("state") or {}
        entry = state.get("cv_prefix")
        if entry is None:
            continue
        series = sorted(destination.rglob("*.cv.csv"))
        if not series:
            continue
        from ..md.stage import _validate_cv_prefix

        try:
            _validate_cv_prefix(entry, stage=stage, trajectory=series[0], fingerprint=None)
        except (CVPrefixError, SystemExit) as refusal:
            problems.append(str(refusal))

    if problems:
        raise ContinuationError(
            "the existing collective-variable output cannot be continued:\n  - "
            + "\n  - ".join(problems))
