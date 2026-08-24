"""Conventional MD as committed segments, in one run directory.

`cMD_1` used to be single-shot: reporters opened without append, re-running overwrote the outputs,
and nothing recorded where the last invocation stopped. Re-running it therefore looked like a
continuation and was a restart -- the worst kind of failure, because the outputs are plausible and
the physical time silently goes backwards.

This gives conventional MD the contract REST2 already has, using the *same* primitives in
`runstate`: an atomic committed-generation record, a binary checkpoint preferred for continuation
with a serialized State as announced fallback, continuity comparison before anything is opened for
append, and quarantine of any tail beyond the committed watermark.

## What is committed, and when

A segment is `steps_per_segment` steps. At its boundary, and only after the dynamics finished:

1. reporters are closed, so the trajectory files on disk end exactly at the boundary;
2. the checkpoint and State are written into the generation directory through temporary files;
3. the generation is committed atomically, naming its members and the watermarks.

If the process dies anywhere before step 3, the generation is not committed and the next invocation
quarantines whatever it left behind. That is why frame counts are recorded in the commit record
rather than measured from the files: a file length is a fact about the last crash, not about the
last committed boundary.

## Why the watermarks are per-stream

An all-atom trajectory at 100 ps and a solute trajectory at 10 ps reach a 500 ps boundary with 5 and
50 frames. Truncating both to "the frame count" would corrupt one of them, so each stream carries
its own watermark and each is truncated to its own committed length.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from .hashing import sha256_text
from typing import Optional

__all__ = [
    "CMD_RUN_STATE_VERSION",
    "UnsupportedCmdSchema",
    "assert_supported_cmd_schema",
    "CommittedOutputCorrupt",
    "RunDirectoryBusy",
    "hold_run_lock",
    "assert_committed_outputs_intact",
    "close_reporters",
    "continuity_hash",
    "inspect_committed_outputs",
    "verify_restart_matches_commit",
    "read_restart_position",
    "restore_restart_step",
    "plan_segment",
    "prepare_continuation",
    "commit_segment",
    "truncate_to_watermark",
]

#: Bumped when the persisted cMD run-state layout changes in a way an older reader would misread.
#: On-disk meaning of the cMD committed record. Version 3 completed the continuity contract: the
#: predecessor State, the manifest and the force-field file are now bound by their exact bytes
#: rather than by a pathname and a partial projection. A version-2 record cannot be reinterpreted
#: as a version-3 one -- it never carried those identities, so "unchanged" could not be checked --
#: and it is refused with regeneration guidance instead.
CMD_RUN_STATE_VERSION = 3

#: The reporting streams a cMD segment appends to. Each carries its own watermark.
_STREAMS = ("all_atom", "selected_atoms", "state_log")


class RunDirectoryBusy(RuntimeError):
    """Another process is already writing this run directory."""


def hold_run_lock(run_dir: Path):
    """Take an exclusive lock on a cMD run directory for the life of the returned handle.

    Nothing previously stopped two invocations from writing one run directory at once. They would
    both restore the same committed generation, both append to the same trajectories and both
    commit -- producing duplicated invocation indices and interleaved frames, while every file
    still looked individually well-formed.

    Discovered by a test that killed a launcher: killing the shell left the Python child running,
    so a "crashed" segment and its retry ran concurrently and the committed history came back as
    [1000, 2000, 2000]. A stale lock from a killed process is released by the operating system
    when the file descriptor closes, so this cannot wedge a run directory.
    """
    import fcntl

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = (run_dir / ".lock").open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RunDirectoryBusy(
            f"another process is writing {run_dir}. Two invocations sharing one run directory "
            "would both restore the same committed generation and both append to the same "
            "trajectories, so this one is refusing rather than interleaving with it."
        ) from error
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def plan_segment(payload: dict) -> dict:
    """Steps per segment and the reporting intervals, from the stage configuration."""
    reporting = payload.get("reporting") or {}
    steps = int(payload["steps"])
    if steps <= 0:
        raise ValueError("a cMD segment must run at least one step")
    return {
        "steps_per_segment": steps,
        "all_atom_interval": reporting.get("full_system_interval_steps"),
        "selected_atoms_interval": reporting.get("selected_atoms_interval_steps"),
        "state_log_interval": (reporting.get("state_interval_steps")
                               or reporting.get("full_system_interval_steps")),
    }


def _frames_at(interval: Optional[int], total_steps: int) -> int:
    """Frames a reporter at `interval` has written after `total_steps` steps."""
    if not interval:
        return 0
    return total_steps // int(interval)


def prepare_continuation(run_dir: Path, *, continuity: dict, plan: dict) -> dict:
    """Decide, before anything is opened for append, whether and how this segment continues.

    Returns a record describing the decision: whether this is the first segment, which generation to
    restore, what the absolute step and time are, and what each stream's committed frame count is.

    Nothing is mutated here. That ordering is the point -- an incompatible continuation must be
    refused while the previous segment's outputs are still exactly as it left them.
    """
    from . import runstate

    run_dir = Path(run_dir)
    state_file = run_dir / "run_state.json"

    if not state_file.is_file():
        return {
            "first_segment": True,
            "segments_completed": 0,
            "absolute_step": 0,
            "restore_from": None,
            "watermarks": {stream: 0 for stream in _STREAMS},
        }

    # Refuse before append, not after: `assert_continuable` compares the continuity-defining fields
    # and raises if the calculation changed.
    runstate.assert_continuable(run_dir, continuity)

    stored = runstate.read_run_state(run_dir)
    generation = runstate.committed_generation(run_dir)
    if generation is None:
        # A run directory exists but nothing was ever committed: the first segment died before its
        # boundary. Start over from equilibration rather than from a tail nobody committed.
        return {
            "first_segment": True,
            "segments_completed": 0,
            "absolute_step": 0,
            "restore_from": None,
            "watermarks": {stream: 0 for stream in _STREAMS},
            "note": "a previous invocation left an uncommitted tail; it is discarded",
        }

    record = runstate.committed_record(run_dir)
    assert_supported_cmd_schema(run_dir, record)
    return {
        "first_segment": False,
        "segments_completed": int(record.get("generation", generation)),
        "absolute_step": int(record.get("absolute_step", 0)),
        "absolute_time_ps": record.get("absolute_time_ps"),
        "restore_from": runstate.generation_dir(run_dir, generation),
        "watermarks": {stream: int((record.get("watermarks") or {}).get(stream, 0))
                       for stream in _STREAMS},
        "stored_invocations": len(stored.get("invocations", [])),
    }


class UnsupportedCmdSchema(RuntimeError):
    """A committed record written under an on-disk layout this build cannot verify."""


def assert_supported_cmd_schema(run_dir: Path, record: dict) -> None:
    """Refuse a committed record whose schema this build cannot check, before anything is opened.

    Reinterpreting an older record as a current one is the failure this guards: version 2 bound the
    predecessor State by pathname and the bundle by a partial projection, so a version-2 record has
    nothing to compare the new identities against. Treating its absence as "unchanged" would mean a
    resume silently skipped exactly the checks the version bump added.
    """
    if not record:
        return
    found = record.get("cmd_schema_version")
    if found == CMD_RUN_STATE_VERSION:
        return
    raise UnsupportedCmdSchema(
        f"{run_dir}: the committed record has cmd_schema_version {found!r}, and this build writes "
        f"{CMD_RUN_STATE_VERSION}.\n"
        "  Version 3 binds the predecessor State, system_manifest.json and forcefield.json by their "
        "exact bytes. A version-2 record never recorded those identities, so this build cannot "
        "verify that the calculation is unchanged, and treating them as unchanged would skip the "
        "checks the bump exists to add.\n"
        "  The committed physics is not lost. To continue this run, keep using the build that wrote "
        "it. To continue under this build, regenerate the project and start a fresh run; the "
        "existing trajectories remain valid as the record of the segments already committed."
    )


class CommittedOutputCorrupt(RuntimeError):
    """A committed stream is absent, short, or malformed. Continuation must stop."""


def inspect_committed_outputs(streams: list) -> list:
    """Phase one: look at every committed stream and change nothing.

    Finding 4. Recovery previously treated an absent or short file as "absent"/"kept", even when
    the atomic commit said more frames were durably written. Continuing from that produces a
    trajectory with a hole in its history while the run still reports completeness -- the worst
    combination, because every downstream index is silently wrong.

    The asymmetry matters: past the watermark is a TAIL, which is recoverable. Short of it is
    MISSING HISTORY, which is not, because the bytes are simply gone.

    A watermark of zero is the one case where absence is valid: nothing was committed yet.
    """
    problems = []
    for stream in streams:
        name, path, watermark, kind, n_atoms = (
            stream["name"], Path(stream["path"]), int(stream["watermark"]),
            stream["kind"], stream.get("n_atoms"))
        if watermark == 0:
            continue
        if not path.is_file():
            problems.append(f"    {name}: {path.name} is absent, but {watermark} "
                            f"{'frames' if kind == 'dcd' else 'rows'} were committed")
            continue
        try:
            present = (_dcd_state(path, n_atoms) if kind == "dcd" else _log_state(path))
        except Exception as error:                                    # noqa: BLE001
            problems.append(f"    {name}: {path.name} is malformed -- {error}")
            continue
        if present["count"] < watermark:
            problems.append(
                f"    {name}: {path.name} holds {present['count']} "
                f"{'frames' if kind == 'dcd' else 'rows'} but {watermark} were committed. "
                "That is missing history, not an uncommitted tail.")
        if kind == "dcd" and n_atoms is not None and present["n_atoms"] != n_atoms:
            problems.append(
                f"    {name}: {path.name} holds {present['n_atoms']} atoms per frame, but this "
                f"stream writes {n_atoms}. The trajectory does not describe this selection.")
    return problems


def _dcd_state(path: Path, n_atoms) -> dict:
    from .dcdtail import frame_offsets

    scan = frame_offsets(path)
    return {"count": scan["complete_frames"], "n_atoms": scan["n_atoms"],
            "trailing_bytes": scan["trailing_bytes"]}


def _log_state(path: Path) -> dict:
    """A state-data log: exactly one header, then monotonic rows."""
    lines = Path(path).read_text().splitlines()
    headers = [line for line in lines if line.startswith("#")]
    if len(headers) != 1:
        raise ValueError(f"expected exactly one header, found {len(headers)}")
    rows = [line for line in lines if not line.startswith("#")]
    steps = []
    for row in rows:
        try:
            steps.append(int(row.split(",")[0]))
        except (ValueError, IndexError) as error:
            raise ValueError(f"unparsable row {row[:40]!r}") from error
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise ValueError("step column is not strictly increasing")
    return {"count": len(rows), "n_atoms": None}


def assert_committed_outputs_intact(streams: list) -> None:
    """Refuse the continuation if any committed stream is not intact. Mutates nothing."""
    problems = inspect_committed_outputs(streams)
    if problems:
        raise CommittedOutputCorrupt(
            "refusing to continue: the committed outputs of this run are not intact.\n"
            + "\n".join(problems)
            + "\n\n  The commit record is the authority on what was durably written. A stream "
              "shorter than its\n  watermark cannot be repaired by appending -- the missing frames "
              "are gone, and appending after\n  them would leave every later index wrong while the "
              "run still reported success."
        )


def truncate_to_watermark(path: Path, stream: str, frames: int, *, n_atoms: int) -> dict:
    """Cut a trajectory or log back to its committed length before appending to it.

    A file longer than its watermark is an uncommitted tail: the previous process wrote frames and
    then died before the commit. Appending after them would leave the trajectory containing frames
    no committed generation accounts for, and every later index wrong by that amount.

    DCD truncation is delegated to `dcdtail`, which walks real Fortran records. The arithmetic that
    used to live here treated a frame as three coordinate blocks and missed the 56-byte unit-cell
    record that every periodic frame carries -- including a periodic ATOM-SUBSET frame, which a
    solvent-mode guess would have got wrong.
    """
    path = Path(path)
    if not path.is_file():
        return {"stream": stream, "action": "absent"}

    if path.suffix == ".dcd":
        from .dcdtail import truncate_to_frames

        result = truncate_to_frames(path, frames)
        return {"stream": stream, **result}

    lines = path.read_text().splitlines()
    if not lines:
        return {"stream": stream, "action": "empty"}
    keep = lines[: 1 + frames]                              # one header, then committed rows
    if len(keep) == len(lines):
        return {"stream": stream, "action": "kept", "rows": len(lines) - 1}
    _atomic_write_text(path, "\n".join(keep) + "\n")
    return {"stream": stream, "action": "truncated", "from": len(lines) - 1, "to": frames}


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file through a same-directory temporary, flushed and fsynced."""
    import os
    import tempfile

    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def close_reporters(sim) -> None:
    """Flush and close every reporter, refusing to continue if one cannot be closed.

    Finding 5. A close failure used to be swallowed, and the commit went ahead anyway -- so the
    commit record could point at a boundary whose trajectory bytes were still in a buffer that
    never reached the disk. If a stream cannot be closed, the boundary is not real and must not be
    committed.
    """
    failures = []
    for reporter in list(sim.reporters):
        for attribute in ("_out", "_traj_file", "_dcd"):
            handle = getattr(reporter, attribute, None)
            close = getattr(handle, "close", None)
            if close is None:
                continue
            try:
                flush = getattr(handle, "flush", None)
                if flush is not None:
                    flush()
                close()
            except Exception as error:                                # noqa: BLE001
                failures.append(f"{type(reporter).__name__}.{attribute}: "
                                f"{type(error).__name__}: {error}")
    sim.reporters.clear()
    if failures:
        raise RuntimeError(
            "refusing to commit this segment: its output streams did not close cleanly, so the "
            "bytes on disk may not reach the boundary the commit would claim.\n  "
            + "\n  ".join(failures))


def read_restart_position(sim) -> dict:
    """What the restart ACTUALLY restored: the step and time now in the Context.

    `Simulation.currentStep` is a property over `Context.getStepCount()` in OpenMM 8.5.2 and 8.6.0
    (re-verified on 8.6.0) -- reading
    it is reading the Context, and *assigning* it calls `Context.setStepCount`. That is why nothing
    may be assigned before this is read: an assignment overwrites the very value to be checked, and
    the comparison that follows then compares the commit against itself.
    """
    from openmm import unit

    state = sim.context.getState()
    return {
        "loaded_step": int(sim.context.getStepCount()),
        "loaded_time_ps": float(state.getTime().value_in_unit(unit.picosecond)),
    }


def verify_restart_matches_commit(sim, record: dict, *, position: Optional[dict] = None) -> dict:
    """Check what was restored against what the commit says was committed. Mutates nothing.

    A checkpoint or State that loads without error can still be the wrong one -- an older generation
    left behind, or a file copied from another run. Comparing step and time against the committed
    record turns "it loaded" into "it loaded the right thing", before any output opens.

    Pass `position` when it was captured before anything touched the Context; otherwise it is read
    here. Time is always authoritative. A restart reporting step 0 against a non-zero commit is
    treated as *carrying no step* rather than as a mismatch -- see `restore_restart_step`.
    """
    position = position if position is not None else read_restart_position(sim)
    loaded_step = int(position["loaded_step"])
    loaded_time = float(position["loaded_time_ps"])
    expected_step = int(record.get("absolute_step", 0))
    expected_time = record.get("absolute_time_ps")

    problems = []
    if expected_time is not None and abs(loaded_time - float(expected_time)) > 1e-6:
        problems.append(f"    time: restart holds {loaded_time} ps, commit says {expected_time} ps")
    step_carried = loaded_step != 0 or expected_step == 0
    if step_carried and loaded_step != expected_step:
        problems.append(f"    step: restart holds {loaded_step:,}, commit says {expected_step:,}")
    if problems:
        raise RuntimeError(
            "the restored restart does not match the committed generation:\n"
            + "\n".join(problems)
            + "\n  Refusing to append output to a run whose restart and commit record disagree."
        )
    return {**position, "step_carried_by_restart": step_carried,
            "expected_step": expected_step, "expected_time_ps": expected_time}


def restore_restart_step(sim, verification: dict, *, restart_source: str) -> dict:
    """Set the step count AFTER verification, and record where the value came from.

    Only two outcomes are possible, and both are recorded rather than inferred:

    * the restart carried the step and it already matched, so this is a no-op and the provenance
      says the step came from the restart itself;
    * the restart carried no step -- a State written by a build that did not preserve one -- so the
      step is taken from the atomic commit, which is the only other authority for it. The Context
      time was already verified against that same commit, so the two agree by construction.

    A State fallback is additionally recorded as a non-bitwise continuation: the State restores
    positions, velocities, box and time, but not the stochastic integrator's internal stream, so the
    trajectory from here diverges from the one an uninterrupted run would have produced.
    """
    expected_step = int(verification["expected_step"])
    if verification["step_carried_by_restart"]:
        origin = f"restart ({restart_source})"
    else:
        sim.context.setStepCount(expected_step)
        origin = "committed.json (the restart carried no step count)"
    provenance = {
        "restart_source": restart_source,
        "step_origin": origin,
        "started_absolute_step": expected_step,
        "started_absolute_time_ps": verification["loaded_time_ps"],
        "bitwise_continuation": restart_source == "checkpoint",
    }
    if restart_source != "checkpoint":
        provenance["note"] = (
            "serialized State fallback: positions, velocities, box and time are restored, but the "
            "stochastic integrator's internal stream is not, so this continuation is physically "
            "valid and NOT bitwise identical to an uninterrupted run."
        )
    return provenance


def commit_segment(run_dir: Path, sim, *, generation: int, plan: dict, absolute_step: int,
                   absolute_time_ps: float, continuity: dict, restart_source: str,
                   invocation: dict, started_step: int, started_time_ps: float,
                   extra: Optional[dict] = None) -> dict:
    """Close the boundary: flush, save both restart forms, then commit atomically.

    The ordering is the contract. Reporters are closed first so the files on disk end exactly at the
    boundary; the restart pair is written next; the commit record is written last and atomically.
    A crash before the final step leaves an uncommitted tail, which the next invocation removes.

    Finding 5: the completed invocation is part of THIS record. It used to be appended to
    `run_state.json` before the commit, so a crash in between left a phantom invocation that a
    retry would duplicate while `committed.json` still described the older physics. One atomic
    write, one authority.
    """
    from . import runstate

    run_dir = Path(run_dir)
    gdir = runstate.generation_dir(run_dir, generation)
    gdir.mkdir(parents=True, exist_ok=True)

    checkpoint, state = runstate.save_restart(sim, gdir)

    # Boundary 4: a complete restart pair that no commit record points at yet.
    from .faults import crash_point

    crash_point("after_restart_members")

    watermarks = {
        "all_atom": _frames_at(plan.get("all_atom_interval"), absolute_step),
        "selected_atoms": _frames_at(plan.get("selected_atoms_interval"), absolute_step),
        "state_log": _frames_at(plan.get("state_log_interval"), absolute_step),
    }
    completed = {
        "invocation": int(generation),
        "segment": int(generation),
        "started_absolute_step": int(started_step),
        "started_absolute_time_ps": float(started_time_ps),
        "ended_absolute_step": int(absolute_step),
        "ended_absolute_time_ps": float(absolute_time_ps),
        "restart_source": restart_source,
        **invocation,
    }
    previous = runstate.committed_record(run_dir)
    history = list(previous.get("invocation_history") or [])
    history.append(completed)

    record = {
        "cmd_schema_version": CMD_RUN_STATE_VERSION,
        "absolute_step": int(absolute_step),
        "absolute_time_ps": float(absolute_time_ps),
        "steps_per_segment": int(plan["steps_per_segment"]),
        "watermarks": watermarks,
        "restart_source_for_this_segment": restart_source,
        "continuity": continuity,
        "continuity_hash": continuity_hash(continuity),
        # lifetime history lives in the atomic record so monotonic accounting survives a crash
        "invocation_history": history,
        "invocations_completed": len(history),
        **(extra or {}),
    }
    # `commit_generation` takes the record as keyword extras, so the whole thing is one atomic
    # write rather than a commit followed by a second file nobody would notice missing.
    runstate.commit_generation(
        run_dir, generation, members=[checkpoint.name, state.name], **record)
    return record


def continuity_hash(contract: dict) -> str:
    """A deterministic hash over the canonical continuity contract."""
    import hashlib
    import json as _json

    canonical = _json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(canonical)
