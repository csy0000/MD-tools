"""The AIS checkpoint transaction, with a deliberate crash at every boundary.

A crash-atomic commit cannot be tested by racing a real kill against a real write: the window is
microseconds wide and the interesting boundaries are the ones you would never hit. So the
transaction raises at each named boundary on request, and every one of them is exercised.

WHAT MUST BE TRUE AFTER ANY CRASH

    read_committed(...) returns the last FULLY committed generation, or refuses.

Never a partial one, and never the newest thing on disk -- the newest file is precisely what a
crash leaves behind. The design this replaces overwrote the binary checkpoint and then replaced
the sidecar, so a crash in between paired a Context from step 3000 with bookkeeping from step
2000. Resuming that re-emits rows that already exist and reports the work of a path nobody ran,
and nothing fails.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from md_tools.ais.checkpoint import (BOUNDARIES, CheckpointError, FAULT_ENVIRONMENT,
                                     POINTER_NAME, clear_committed, commit_generation,
                                     read_committed)


def _commit(directory: Path, *, step: int, payload: bytes, fault: str | None = None):
    """One generation, optionally crashing at a named boundary."""
    previous = os.environ.get(FAULT_ENVIRONMENT)
    if fault:
        os.environ[FAULT_ENVIRONMENT] = fault
    else:
        os.environ.pop(FAULT_ENVIRONMENT, None)
    try:
        return commit_generation(directory,
                                 write_checkpoint=lambda p: p.write_bytes(payload),
                                 state={"protocol_step": step, "work_rows": step // 100,
                                        "frames": step // 200, "cumulative_work_kj_mol": -step})
    finally:
        if previous is None:
            os.environ.pop(FAULT_ENVIRONMENT, None)
        else:
            os.environ[FAULT_ENVIRONMENT] = previous


@pytest.fixture
def path_directory(tmp_path):
    directory = tmp_path / "path_0000"
    directory.mkdir()
    return directory


def test_nothing_committed_reads_as_nothing(path_directory):
    assert read_committed(path_directory) is None


def test_a_commit_is_readable_and_carries_its_state(path_directory):
    _commit(path_directory, step=100, payload=b"state at 100")
    committed = read_committed(path_directory)
    assert committed["generation"] == 1
    assert committed["state"]["protocol_step"] == 100
    assert Path(committed["checkpoint"]).read_bytes() == b"state at 100"


def test_generations_advance_and_the_pointer_follows(path_directory):
    _commit(path_directory, step=100, payload=b"a")
    _commit(path_directory, step=200, payload=b"b")
    committed = read_committed(path_directory)
    assert committed["generation"] == 2
    assert committed["state"]["protocol_step"] == 200
    assert Path(committed["checkpoint"]).read_bytes() == b"b"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_crash_at_any_boundary_leaves_the_previous_generation_committed(boundary,
                                                                         path_directory):
    """The heart of it. One good generation, then a crash at each step of the next.

    Every boundary before the pointer replacement must leave generation 1 committed and intact.
    The one boundary AFTER it must leave generation 2 committed -- and complete, because both of
    its files were fsynced before the pointer named them.
    """
    _commit(path_directory, step=100, payload=b"first")
    before = read_committed(path_directory)
    assert before["state"]["protocol_step"] == 100

    with pytest.raises(RuntimeError):
        _commit(path_directory, step=200, payload=b"second", fault=boundary)

    after = read_committed(path_directory)
    if boundary == "after-pointer-replace":
        # The pointer landed, and everything it names was durable before it did.
        assert after["state"]["protocol_step"] == 200
        assert Path(after["checkpoint"]).read_bytes() == b"second"
    else:
        # The pointer never moved. The half-written generation is on disk and is IGNORED.
        assert after["state"]["protocol_step"] == 100, boundary
        assert Path(after["checkpoint"]).read_bytes() == b"first", boundary
    # Whatever happened, what comes back is complete: both files exist and the digest matches.
    assert Path(after["checkpoint"]).is_file() and Path(after["sidecar"]).is_file()


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_crash_before_the_first_commit_leaves_nothing_claimed(boundary, path_directory):
    """With no previous generation, a crash must leave `read_committed` returning None.

    Not an exception, and certainly not the half-written generation: a path that has never
    committed anything is a path that starts from its source frame.
    """
    with pytest.raises(RuntimeError):
        _commit(path_directory, step=100, payload=b"only", fault=boundary)

    if boundary == "after-pointer-replace":
        assert read_committed(path_directory)["state"]["protocol_step"] == 100
    else:
        assert read_committed(path_directory) is None, boundary


def test_an_uncommitted_generation_left_by_a_crash_is_never_selected(path_directory):
    """Explicitly: "the newest generation" is the wrong rule, and this is why."""
    _commit(path_directory, step=100, payload=b"committed")
    generations = path_directory / "checkpoints"
    (generations / "generation_000099.chk").write_bytes(b"never committed")
    (generations / "generation_000099.json").write_text(
        json.dumps({"generation": 99, "checkpoint": "generation_000099.chk",
                    "checkpoint_sha256": "0" * 64, "state": {"protocol_step": 9900}}),
        encoding="utf-8")

    committed = read_committed(path_directory)
    assert committed["generation"] == 1
    assert committed["state"]["protocol_step"] == 100


def test_a_checkpoint_that_changed_after_commit_is_refused(path_directory):
    """The digest is what makes "this is the file I committed" checkable rather than assumed."""
    _commit(path_directory, step=100, payload=b"good")
    Path(read_committed(path_directory)["checkpoint"]).write_bytes(b"tampered")

    with pytest.raises(CheckpointError, match="sha256"):
        read_committed(path_directory)


def test_a_missing_checkpoint_behind_a_valid_pointer_is_refused(path_directory):
    _commit(path_directory, step=100, payload=b"good")
    Path(read_committed(path_directory)["checkpoint"]).unlink()

    with pytest.raises(CheckpointError, match="missing"):
        read_committed(path_directory)


def test_a_missing_sidecar_behind_a_valid_pointer_is_refused(path_directory):
    _commit(path_directory, step=100, payload=b"good")
    Path(read_committed(path_directory)["sidecar"]).unlink()

    with pytest.raises(CheckpointError, match="missing"):
        read_committed(path_directory)


def test_an_unreadable_pointer_is_refused_rather_than_guessed_around(path_directory):
    """A pointer that cannot be parsed is never repaired: repairing it means choosing a
    generation nobody committed, which is the failure the pointer exists to prevent."""
    _commit(path_directory, step=100, payload=b"good")
    (path_directory / POINTER_NAME).write_text("{ not json", encoding="utf-8")

    with pytest.raises(CheckpointError):
        read_committed(path_directory)


def test_the_previous_generation_survives_a_new_commit(path_directory):
    """At least one older generation is kept, so the newest is not the only thing that exists."""
    for step in (100, 200, 300):
        _commit(path_directory, step=step, payload=f"state {step}".encode())
    generations = sorted((path_directory / "checkpoints").glob("generation_*.chk"))
    assert len(generations) >= 2, [p.name for p in generations]
    assert read_committed(path_directory)["state"]["protocol_step"] == 300


def test_clearing_removes_the_whole_transaction(path_directory):
    """A completed path keeps no checkpoint: leaving one invites a resume of finished work."""
    _commit(path_directory, step=100, payload=b"done")
    clear_committed(path_directory)
    assert not (path_directory / POINTER_NAME).exists()
    assert not (path_directory / "checkpoints").exists()
    assert read_committed(path_directory) is None


def test_the_committed_pair_is_never_the_file_being_written(path_directory):
    """A new generation gets NEW names, so the committed pair is never in flight.

    This is what makes the transaction atomic at all: overwriting the committed checkpoint means
    there is an instant when the pointer names a file that is half-written, and no amount of care
    with the sidecar afterwards can recover that.
    """
    _commit(path_directory, step=100, payload=b"first")
    first = Path(read_committed(path_directory)["checkpoint"])
    _commit(path_directory, step=200, payload=b"second")
    second = Path(read_committed(path_directory)["checkpoint"])
    assert first != second
    assert first.read_bytes() == b"first", "the committed generation was overwritten"
