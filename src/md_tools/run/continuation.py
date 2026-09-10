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
                          interval_steps=int(interval_steps), block=extra.get("cv_prefix"),
                          committed_step=checkpoint.get("step"),
                          committed_rows=extra.get("cv_rows"))
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


def validate_stage_continuation(destination, *, stage, definition, overwrite=False) -> None:
    """A cMD stage's committed prefix, before the stage opens its own records.

    The stage continues automatically from its committed generation, so this is the ordinary
    resume path and the point at which it would truncate its appendable streams.

    `--overwrite` is not a continuation. It starts CLEAN and deliberately does not load the
    generations it was asked to replace, so validating them would refuse a run precisely because
    the data it is about to discard is unusable -- which is the reason the flag exists.
    """
    if definition is None or overwrite:
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
        # THIS checkpoint's own series. A chain has one checkpoint tree per stage --
        # `<stem>.checkpoints` beside `<stem>.cv.csv` -- and taking the first `*.cv.csv` in the
        # directory paired stage `min`'s checkpoint with the production stage's series, then
        # refused the run over a disagreement between two different stages' files.
        stem = checkpoints.name[: -len(".checkpoints")]
        candidate = checkpoints.parent / f"{stem}.cv.csv"
        if not candidate.is_file():
            continue
        series = [candidate]
        # The series file directly. `_validate_cv_prefix` in the stage runtime derives it from
        # the TRAJECTORY path via `cv_csv_path`, so handing it the series produced a doubled
        # `.cv.cv.csv` and refused for a file that had never existed -- a real refusal for the
        # wrong reason, which is its own kind of wrong.
        from ..cv import prefix as cv_prefix

        try:
            cv_prefix.validate(series[0], entry, sidecar=series[0].with_suffix(".json"),
                               definition=definition)
        except CVPrefixError as refusal:
            problems.append(str(refusal))

    if problems:
        raise ContinuationError(
            "the existing collective-variable output cannot be continued:\n  - "
            + "\n  - ".join(problems))


def _definition_from(resolved, *, config_directory=None):
    """The CV definition a resolved document names, or None when reporting is off."""
    block = (resolved or {}).get("collective_variables") or {}
    path = block.get("file")
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute() and config_directory is not None:
        candidate = Path(config_directory) / candidate
    if not candidate.is_file():
        return None
    try:
        from ..cv import load_cv_definition

        return load_cv_definition(candidate)
    except Exception:      # noqa: BLE001 - the ordinary path reports a bad definition better
        return None


def validate_public_entry(resolved, out_dir, *, protocol, stage=None,
                          config_directory=None, overwrite=False) -> None:
    """The same read-only boundary, for `md-openmm md-run`.

    `md-run` creates `-odir` and writes `resolved.config` and the content-addressed definition
    copy ITSELF, before dispatching to the runtime whose preflight does the validating. So a
    refused continuation through the public command added two files to a tree it was declining
    to touch, while the identical operation through a generated wrapper added none. Two surfaces
    onto one runtime must not disagree about that, or the boundary is one nobody can rely on.

    Called before the directory is created and before anything is written.
    """
    if overwrite:
        return                                  # --overwrite starts clean; nothing to continue
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return                                  # nothing exists yet; nothing to protect
    definition = _definition_from(resolved, config_directory=config_directory)
    if definition is None:
        return

    interval = _cv_interval(resolved)
    if protocol in ("REST2", "rREST2") and stage is None:
        rest2 = (resolved or {}).get("rest2") or {}
        states = int(rest2.get("number_of_replicas") or 0)
        if not states:
            return
        tau_max = float(rest2.get("tau_max", 0.5))
        # THE SAME LADDER THE RUN USED, from the one function that defines it. This recomputed it
        # inline as `tau_max * i / (states - 1)`, unrounded, and compared the result against the
        # tau each `cv_stateN.json` recorded -- which came from `tau_ladder`, rounded to 6 places.
        # For four states they differ at the seventh decimal, the comparison allows 1e-12, and
        # every CV-enabled four-rung ladder was therefore unresumable:
        #
        #   cv_state1.json records tau 0.166667 for state 1 and this ladder resolves
        #   0.16666666666666666
        #
        # Nothing was wrong with the data. Two spellings of one ladder disagreed about it, and the
        # spelling that never ran was the one asked to judge.
        from ..remd.generated import tau_ladder

        taus = tau_ladder(states, tau_max) if states > 1 else [0.0]
        validate_ladder_continuation(
            out_dir, definition=definition, taus=taus, interval_steps=interval,
            checkpoint_path=out_dir / f"{protocol}_checkpoint.nc")
        return

    if protocol == "AIS":
        import json

        from ..ais.run import COMPLETION_NAME

        from ..openmm.checkpoint import CheckpointError, read_committed

        # COMPLETED AND PARTIAL PATHS BOTH.
        #
        # This used to read `completed.json` and nothing else, so an interrupted path -- one with
        # a committed generation and no marker, which is precisely the state a resume exists for
        # -- was not in the selection and was never looked at. The authoritative runtime does
        # refuse such a path before it truncates any of its tables, so no scientific output was at
        # risk; what was at risk is the other half of the boundary. `md-run` creates `-odir` and
        # rewrites `resolved.config` and the definition copy BEFORE it dispatches, so a resume
        # that the runtime was always going to refuse still replaced the prior run's authoritative
        # configuration record on its way to refusing. A path's source frame is recorded in its
        # committed checkpoint exactly as it is in its completion manifest, so including partial
        # paths costs one extra read and needs no new validation rule.
        chosen: dict[int, int] = {}
        for directory in sorted(out_dir.glob("path_*")):
            marker = directory / COMPLETION_NAME
            if marker.is_file():
                try:
                    record = json.loads(marker.read_text(encoding="utf-8"))
                    chosen[int(record["path_index"])] = int(record["source_frame_index"])
                except (ValueError, KeyError, TypeError):
                    continue
                continue
            # A partial path. An unreadable or absent committed generation is left to the
            # runtime, which reports it far better than this boundary can and reaches it before
            # it truncates anything.
            try:
                committed = read_committed(directory)
            except CheckpointError:
                continue
            state = (committed or {}).get("state") or {}
            try:
                chosen[int(state["path_index"])] = int(state["source_frame_index"])
            except (KeyError, TypeError, ValueError):
                continue
        if not chosen:
            return
        schedule = (resolved or {}).get("schedule") or {}
        ais = (resolved or {}).get("ais") or {}
        derived = {
            "cv_interval_steps": interval,
            "switching_steps": int(schedule.get("switching_steps")
                                   or ais.get("switching_steps") or 0),
        }
        if not derived["switching_steps"]:
            return
        validate_ais_continuation(
            out_dir, definition=definition, schedule=derived,
            chosen=[chosen.get(i, 0) for i in range(max(chosen) + 1)],
            selected=sorted(chosen))
        return

    # `stage` here is the stage NAME the input selected, not a mapping; the resolved document is
    # what carries the collective-variable block a stage check needs.
    validate_stage_continuation(out_dir, stage=resolved, definition=definition,
                                overwrite=overwrite)
