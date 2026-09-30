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


def validate_stage_continuation(destination, *, stage, definition, overwrite=False,
                                own_checkpoints=None, stage_name=None) -> None:
    """A cMD stage's committed records, before the stage opens its own.

    The stage continues automatically from its committed generation, so this is the ordinary
    resume path and the point at which it would truncate its appendable streams.

    WHICH TREE IS THIS STAGE'S. The equilibration stages of a chain share one `-odir`, so
    judging a stage on every `*.checkpoints` beneath it judges it on its NEIGHBOURS' records --
    measured: `eq_2` refused over `eq_1`'s completed checkpoint, which it never reads. So the
    step-convention check is asked only of the tree this invocation would actually resume,
    identified either by `own_checkpoints` (the runtime knows its own path) or by the stage name
    the generation itself records (`state["stage"]`, which `verify_completed_stage` matches on
    too). Given neither, nothing is assumed and the check is skipped rather than applied to
    somebody else's record.

    The CV prefixes stay directory-wide: each is matched to its own stage's series by stem, and a
    damaged one anywhere in the tree is worth refusing before a run appends to it.

    `--overwrite` is not a continuation. It starts CLEAN and deliberately does not load the
    generations it was asked to replace, so validating them would refuse a run precisely because
    the data it is about to discard is unusable -- which is the reason the flag exists.
    """
    if overwrite:
        return
    destination = Path(destination)
    if not destination.is_dir():
        return

    from ..cv.prefix import CVPrefixError
    from ..openmm.checkpoint import POINTER_NAME, CheckpointError, read_committed

    problems: list[str] = []
    #: Checkpoint trees written under the old `steps_done` convention. Collected rather than
    #: reported one by one: the explanation is the same for every one of them, and a chain has a
    #: tree per stage, so repeating it four times buries the list of what is affected.
    legacy: list[Path] = []
    own = Path(own_checkpoints).resolve() if own_checkpoints else None
    for tree in sorted(destination.rglob("*.checkpoints")):
        if not (tree / POINTER_NAME).is_file():
            continue
        try:
            committed = read_committed(tree)
        except CheckpointError:
            continue
        state = (committed or {}).get("state") or {}

        # WHICH STEP CONVENTION THIS RECORD USES, checked whether or not collective variables
        # are enabled: it is a property of the checkpoint, not of the reporting.
        #
        # `steps_done` is the stage's OWN progress, and a generation says so by carrying
        # `absolute_step` beside it. A record without that key was written by a build that stored
        # the ABSOLUTE step under `steps_done`, and once the stage's origin is unknown the two
        # are indistinguishable -- both are plain, plausible step counts. Reading the absolute
        # one as relative makes `remaining = steps - done` too small, so the stage integrates
        # less than it was asked for and reports completion anyway.
        #
        # It is refused HERE, in the read-only phase, rather than in the stage's resume branch.
        # A refusal there replaces the prior run's `.out` and machine record with a `status:
        # failed` account of a run that never started -- which is the same mistake the committed
        # CV prefix check was moved out of, and it was measured happening again.
        #
        # ONLY THE TREE THIS INVOCATION WOULD RESUME. The equilibration stages of a chain share
        # one `-odir`, so scanning every `*.checkpoints` beneath it judges a stage on its
        # NEIGHBOURS' records: `eq_2` was refused over `eq_1`'s completed checkpoint, which it
        # never reads. When the caller names its own tree, that is the only one considered.
        mine = (tree.resolve() == own if own is not None
                else (str(state.get("stage")) == str(stage_name) if stage_name else False))
        if mine and "absolute_step" not in state:
            legacy.append(tree)
            continue

        if definition is None:
            # Nothing else to check here: the remainder of this loop is about the CV series, and
            # this run has no definition to check one against.
            continue
        entry = state.get("cv_prefix")
        if entry is None:
            continue
        # THIS checkpoint's own series. A chain has one checkpoint tree per stage --
        # `<stem>.checkpoints` beside `<stem>.cv.csv` -- and taking the first `*.cv.csv` in the
        # directory paired stage `min`'s checkpoint with the production stage's series, then
        # refused the run over a disagreement between two different stages' files.
        stem = tree.name[: -len(".checkpoints")]
        candidate = tree.parent / f"{stem}.cv.csv"
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

    if legacy:
        raise ContinuationError(
            "the committed checkpoint(s) here were written by md-tools 0.6.3 or earlier and "
            "cannot be resumed by this version:\n  - "
            + "\n  - ".join(str(path) for path in legacy)
            + "\n  Each records `steps_done` without the `absolute_step` that says what that "
              "count is measured from. The pairing arrived in 0.6.4, so a record without it "
              "predates the fix -- which is all the record can prove, and is why that version "
              "is a bound rather than an exact one.\n"
              "  For a stage that began at step 0 the old and new meanings of `steps_done` "
              "agree. For a stage continued from a restart the old value is larger by however "
              "many steps came before, so resuming from it would integrate that many steps too "
              "few and report completion anyway. The stage's origin cannot be recovered from "
              "the record, so this is refused rather than guessed.\n"
              "  Pass --overwrite to run the stage again from the beginning, or continue it "
              "with the version of md-tools that wrote it.")
    if problems:
        raise ContinuationError(
            "the existing collective-variable output cannot be continued:\n  - "
            + "\n  - ".join(problems))


def _definition_from(resolved, *, config_directory=None, fallback_directory=None):
    """The CV definition a resolved document names, or None when reporting is off.

    TWO DIRECTORIES, TRIED IN ORDER, because the definition copy is PER RUN and the input is
    SHARED. `build-md` copies the content-addressed definition into the run directory and
    `md-run` writes its own beside the `resolved.config` it creates, so `-odir` is where a copy
    actually is; `input/` at the dataset root holds none. Resolving only against the input's
    directory returned None here, and a None definition makes `validate_public_entry` return
    early -- so the read-only boundary silently did nothing and a refused continuation wrote into
    the tree it was declining to touch.

    The fallback is not redundant: before the layout split the input and the copy shared one
    directory, and every run generated then still resolves that way.
    """
    block = (resolved or {}).get("collective_variables") or {}
    path = block.get("file")
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        for directory in (config_directory, fallback_directory):
            if directory is None:
                continue
            attempt = Path(directory) / candidate
            if attempt.is_file():
                candidate = attempt
                break
    if not candidate.is_file():
        return None
    try:
        from ..cv import load_cv_definition

        return load_cv_definition(candidate)
    except Exception:      # noqa: BLE001 - the ordinary path reports a bad definition better
        return None


def validate_public_entry(resolved, out_dir, *, protocol, stage=None,
                          config_directory=None, fallback_directory=None,
                          overwrite=False) -> None:
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
    definition = _definition_from(resolved, config_directory=config_directory,
                                  fallback_directory=fallback_directory)
    if definition is None:
        return

    interval = _cv_interval(resolved)
    if protocol == "REST2" and stage is None:
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
    # `stage_name` so the step-convention check can tell THIS stage's committed generation from
    # its neighbours' in a shared `-odir`: `md-run` creates the directory and rewrites
    # `resolved.config` before it dispatches, so a refusal the runtime was always going to make
    # must also be reachable here, before that happens.
    validate_stage_continuation(out_dir, stage=resolved, definition=definition,
                                overwrite=overwrite, stage_name=stage)
