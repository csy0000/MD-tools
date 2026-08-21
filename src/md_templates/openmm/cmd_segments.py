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
from pathlib import Path
from typing import Optional

__all__ = [
    "CMD_RUN_STATE_VERSION",
    "plan_segment",
    "prepare_continuation",
    "commit_segment",
    "truncate_to_watermark",
]

#: Bumped when the persisted cMD run-state layout changes in a way an older reader would misread.
CMD_RUN_STATE_VERSION = 1

#: The reporting streams a cMD segment appends to. Each carries its own watermark.
_STREAMS = ("all_atom", "selected_atoms", "state_log")


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


def truncate_to_watermark(path: Path, stream: str, frames: int, *, n_atoms: int) -> dict:
    """Cut a trajectory or log back to its committed length before appending to it.

    A file longer than its watermark is an uncommitted tail: the previous process wrote frames and
    then died before the commit. Appending after them would leave the trajectory containing frames
    that no committed generation accounts for, and every later index would be wrong by that amount.

    DCD carries its frame count in the header, so both the count and the file length are corrected.
    """
    path = Path(path)
    if not path.is_file():
        return {"stream": stream, "action": "absent"}

    if path.suffix == ".dcd":
        import struct

        with path.open("rb+") as handle:
            handle.seek(8)
            present = struct.unpack("<i", handle.read(4))[0]
            if present <= frames:
                return {"stream": stream, "action": "kept", "frames": present}
            header = 84 + 4 + 84 + 4 + 4 + 4 + 4 + 4          # DCD header + title + natom blocks
            frame_bytes = 3 * (4 + 4 * n_atoms + 4)
            handle.seek(8)
            handle.write(struct.pack("<i", frames))
            handle.truncate(_dcd_header_bytes(path) + frames * frame_bytes)
        return {"stream": stream, "action": "truncated", "from": present, "to": frames}

    # a text log: one header line plus one row per report
    lines = path.read_text().splitlines()
    if not lines:
        return {"stream": stream, "action": "empty"}
    keep = lines[: 1 + frames]
    if len(keep) == len(lines):
        return {"stream": stream, "action": "kept", "rows": len(lines) - 1}
    path.write_text("\n".join(keep) + "\n")
    return {"stream": stream, "action": "truncated", "from": len(lines) - 1, "to": frames}


def _dcd_header_bytes(path: Path) -> int:
    """Byte offset of the first frame, read from the file rather than assumed.

    OpenMM writes a fixed header, but the title block length is stored in the file and a hard-coded
    offset would silently corrupt a trajectory written by any other producer.
    """
    import struct

    with path.open("rb") as handle:
        first = struct.unpack("<i", handle.read(4))[0]           # 84
        handle.seek(4 + first + 4)
        title_size = struct.unpack("<i", handle.read(4))[0]
        handle.seek(4 + first + 4 + 4 + title_size + 4)
        natom_size = struct.unpack("<i", handle.read(4))[0]
        return 4 + first + 4 + 4 + title_size + 4 + 4 + natom_size + 4


def commit_segment(run_dir: Path, sim, *, generation: int, plan: dict, absolute_step: int,
                   absolute_time_ps: float, continuity: dict, restart_source: str,
                   extra: Optional[dict] = None) -> dict:
    """Close the boundary: flush, save both restart forms, then commit atomically.

    The ordering is the contract. Reporters are closed first so the files on disk end exactly at the
    boundary; the restart pair is written next; the commit record is written last and atomically.
    A crash before the final step leaves an uncommitted tail, which the next invocation removes.
    """
    from . import runstate

    run_dir = Path(run_dir)
    gdir = runstate.generation_dir(run_dir, generation)
    gdir.mkdir(parents=True, exist_ok=True)

    checkpoint, state = runstate.save_restart(sim, gdir)

    watermarks = {
        "all_atom": _frames_at(plan.get("all_atom_interval"), absolute_step),
        "selected_atoms": _frames_at(plan.get("selected_atoms_interval"), absolute_step),
        "state_log": _frames_at(plan.get("state_log_interval"), absolute_step),
    }
    record = {
        "cmd_schema_version": CMD_RUN_STATE_VERSION,
        "absolute_step": int(absolute_step),
        "absolute_time_ps": float(absolute_time_ps),
        "steps_per_segment": int(plan["steps_per_segment"]),
        "watermarks": watermarks,
        "restart_source_for_this_segment": restart_source,
        "continuity": continuity,
        **(extra or {}),
    }
    # `commit_generation` takes the record as keyword extras, so the whole thing is one atomic
    # write rather than a commit followed by a second file nobody would notice missing.
    runstate.commit_generation(
        run_dir, generation, members=[checkpoint.name, state.name], **record)
    return record
