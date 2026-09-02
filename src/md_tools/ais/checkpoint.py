"""A crash-atomic checkpoint for one AIS switching path.

THE FAILURE THIS EXISTS FOR

An AIS path's restartable state is two things that must agree: an OpenMM binary checkpoint holding
positions, velocities and parameters, and a sidecar holding everything outside the Context --
accumulated work, work since the last observation, and how many work rows, frames and state rows
are already on disk.

The previous design overwrote the checkpoint, then atomically replaced the sidecar. Atomic
replacement of the sidecar alone does not make the PAIR atomic. A crash in between leaves a NEW
Context checkpoint beside an OLD bookkeeping record, and a resume from that pair continues a
simulation from step 3000 while believing it is at step 2000: it re-emits rows that exist, and the
work integral it reports is the sum of a path that was never run. Nothing fails. The output looks
complete.

THE TRANSACTION

    path_0000/
      checkpoints/
        generation_000012.chk      an OpenMM checkpoint
        generation_000012.json     its sidecar, carrying the checkpoint's sha256
        generation_000013.chk      possibly a crash landed here
        generation_000013.json
      current_checkpoint.json      the POINTER. One small file, replaced atomically, last.

1. a new generation is written to new names -- the committed pair is never overwritten;
2. the checkpoint is flushed, fsynced, and digested;
3. the sidecar is written with that digest and every fingerprint, then flushed and fsynced;
4. the directory is fsynced, so both names are durable before anything points at them;
5. the pointer is replaced with `os.replace`, which is atomic on POSIX;
6. only then are older generations removed, and the previous one is kept.

A crash before step 5 leaves the old pointer, so the old pair is what resumes. A crash after it
leaves the new pointer, and both files it names are already durable. There is no window in which
the pointer names something incomplete.

READING

`read_committed` follows only the pointer. It verifies the sidecar parses, that both files exist,
and that the checkpoint's sha256 is the one the sidecar recorded, before anything is loaded. An
uncommitted newer generation is ignored -- never "the latest wins", because the newest file on
disk is exactly what a crash leaves behind.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

__all__ = ["CheckpointError", "POINTER_NAME", "GENERATIONS_DIR", "commit_generation",
           "read_committed", "clear_committed", "fault", "BOUNDARIES",
           "STREAM_BOUNDARIES"]

POINTER_NAME = "current_checkpoint.json"
GENERATIONS_DIR = "checkpoints"

#: How many committed generations to keep. Two, not one: the previous generation is the thing a
#: reader falls back to if the newest turns out to be unreadable, and deleting it the moment a new
#: pointer lands would remove the only alternative at the exact moment it might be needed.
KEEP_GENERATIONS = 2

#: Fault injection for the tests. Set to the name of a boundary and execution raises there, so a
#: crash at each step of the transaction is exercised deterministically rather than by racing a
#: real kill against a real write.
FAULT_ENVIRONMENT = "MD_TOOLS_CHECKPOINT_FAULT"

#: How many times a boundary is allowed to pass before it fires. Without it, a fault at the FIRST
#: commit leaves nothing committed and the interesting case -- resuming from a previous generation
#: while a newer one lies half-written -- cannot be reached at all.
FAULT_AFTER_ENVIRONMENT = "MD_TOOLS_CHECKPOINT_FAULT_AFTER"

#: The commit transaction's own boundaries.
BOUNDARIES = ("after-checkpoint-write", "after-checkpoint-sync", "after-sidecar-write",
              "before-pointer-replace", "after-pointer-replace")

#: The stream boundaries, raised by the path runner. A crash between a frame reaching the disk and
#: the checkpoint that vouches for it is the case the committed counters exist for.
STREAM_BOUNDARIES = ("before-frame", "after-frame", "before-work-row", "after-work-row",
                     "before-state-row", "after-state-row")

#: How many times each boundary has been passed in this process, so `FAULT_AFTER` can count.
_passed: dict[str, int] = {}


class CheckpointError(SystemExit):
    """A committed checkpoint that cannot be trusted. Refused rather than loaded."""


class _InjectedFault(RuntimeError):
    """A deliberate crash at a transaction boundary. Only the tests raise it."""


def fault(boundary: str) -> None:
    """Raise here if the tests armed this boundary. A no-op in every normal run.

    `FAULT_AFTER` lets a boundary pass a stated number of times first, so a crash can be placed
    at the second or third commit rather than only the first.
    """
    if os.environ.get(FAULT_ENVIRONMENT) != boundary:
        return
    allowed = int(os.environ.get(FAULT_AFTER_ENVIRONMENT) or 0)
    _passed[boundary] = _passed.get(boundary, 0) + 1
    if _passed[boundary] > allowed:
        raise _InjectedFault(f"injected crash at {boundary}")


#: The private spelling the transaction below uses.
_fault = fault


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_durably(path: Path, payload: bytes) -> None:
    """Write, flush and fsync. A file that is not fsynced is not on disk after a power loss."""
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    """fsync a directory, so the NAMES in it are durable, not only the file contents."""
    handle = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(handle)
    except OSError:
        pass                                              # some filesystems refuse; not fatal
    finally:
        os.close(handle)


def _generations(directory: Path) -> Path:
    return Path(directory) / GENERATIONS_DIR


def commit_generation(directory: str | Path, *, write_checkpoint: Callable[[Path], Any],
                      state: dict[str, Any]) -> dict[str, Any]:
    """Write and commit one generation. Returns the committed pointer document.

    `write_checkpoint` is given a path and must write the OpenMM checkpoint to it --
    `simulation.saveCheckpoint` in the real caller, a byte string in the tests. It is called with a
    NEW name every time, so the committed pair is never the file being written.
    """
    directory = Path(directory)
    generations = _generations(directory)
    generations.mkdir(parents=True, exist_ok=True)

    previous = _read_pointer(directory)
    number = int(previous["generation"]) + 1 if previous else 1

    binary = generations / f"generation_{number:06d}.chk"
    sidecar = generations / f"generation_{number:06d}.json"

    write_checkpoint(binary)
    _fault("after-checkpoint-write")

    with binary.open("rb") as handle:
        os.fsync(handle.fileno())
    digest = _sha256(binary)
    _fault("after-checkpoint-sync")

    document = {
        "generation": number,
        "checkpoint": binary.name,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": binary.stat().st_size,
        "state": dict(state),
    }
    _write_durably(sidecar, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode())
    _fault("after-sidecar-write")

    # Both files are durable, and nothing points at them yet. This is the moment the transaction
    # becomes committable: a crash here loses a generation and costs nothing.
    _sync_directory(generations)
    _fault("before-pointer-replace")

    pointer = directory / POINTER_NAME
    staging = directory / (POINTER_NAME + ".partial")
    _write_durably(staging, (json.dumps({"generation": number,
                                         "sidecar": sidecar.name}, indent=2) + "\n").encode())
    os.replace(staging, pointer)                          # atomic on POSIX
    _sync_directory(directory)
    _fault("after-pointer-replace")

    _prune(generations, keep_through=number)
    return read_committed(directory)


def _read_pointer(directory: Path) -> dict[str, Any] | None:
    pointer = Path(directory) / POINTER_NAME
    if not pointer.is_file():
        return None
    try:
        document = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A pointer that cannot be parsed is not a pointer. It is never repaired or guessed at:
        # the alternative is choosing a generation nobody committed.
        raise CheckpointError(
            f"{pointer} is unreadable. The committed checkpoint pointer is the only thing that "
            f"says which generation is complete; without it, resuming would mean choosing one, "
            f"and the newest file on disk is exactly what a crash leaves behind. Delete the path "
            f"directory to rerun it from its source frame.") from None
    if not isinstance(document, dict) or "generation" not in document:
        raise CheckpointError(f"{pointer} does not name a generation")
    return document


def read_committed(directory: str | Path) -> dict[str, Any] | None:
    """The last fully committed generation, verified. None when nothing has been committed.

    Everything is checked BEFORE the caller loads anything: the pointer parses, the sidecar it
    names exists and parses, the checkpoint it names exists, and its sha256 is the one recorded.
    A pair that fails any of these is refused rather than half-loaded.
    """
    directory = Path(directory)
    pointer = _read_pointer(directory)
    if pointer is None:
        return None

    generations = _generations(directory)
    sidecar = generations / pointer["sidecar"]
    if not sidecar.is_file():
        raise CheckpointError(
            f"the committed pointer names {sidecar.name}, which is missing. The checkpoint "
            f"transaction did not complete and cannot be repaired; delete the path directory to "
            f"rerun this path from its source frame.")
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except ValueError:
        raise CheckpointError(f"{sidecar} is not readable JSON") from None

    binary = generations / document["checkpoint"]
    if not binary.is_file():
        raise CheckpointError(
            f"the committed sidecar names {binary.name}, which is missing.")
    actual = _sha256(binary)
    if actual != document["checkpoint_sha256"]:
        raise CheckpointError(
            f"{binary.name} does not match the sha256 its sidecar recorded "
            f"(committed {document['checkpoint_sha256'][:16]}..., found {actual[:16]}...).\n"
            f"  The checkpoint has changed since it was committed, so the Context it restores is "
            f"not the one the bookkeeping describes. Refusing to resume it: the numbers would "
            f"look right and describe a different simulation.\n"
            f"  Delete the path directory ({binary.parent.parent}) to rerun this path from its "
            f"source frame. Its work is lost; the other paths are untouched.")

    return {"generation": document["generation"],
            "checkpoint": str(binary),
            "sidecar": str(sidecar),
            "state": document["state"]}


def clear_committed(directory: str | Path) -> None:
    """Remove the pointer and every generation. Used when a path completes."""
    directory = Path(directory)
    (directory / POINTER_NAME).unlink(missing_ok=True)
    generations = _generations(directory)
    if generations.is_dir():
        for path in generations.iterdir():
            path.unlink(missing_ok=True)
        generations.rmdir()


def _prune(generations: Path, *, keep_through: int) -> None:
    """Drop generations older than the last `KEEP_GENERATIONS`, and any uncommitted newer ones.

    Newer-than-committed files are removed because they are debris from a crash: keeping them
    invites a future reader to treat "newest" as "current", which is the exact mistake the pointer
    exists to prevent.
    """
    keep = set(range(max(keep_through - KEEP_GENERATIONS + 1, 1), keep_through + 1))
    for path in generations.glob("generation_*"):
        try:
            number = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if number not in keep:
            path.unlink(missing_ok=True)
