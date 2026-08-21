"""Deterministic interruption at named commit boundaries, for tests only.

A time-based `kill -9` proves that *some* interruption is survivable. It does not prove *which*
boundary was interrupted, and the five boundaries around a segment commit fail in different ways:
one leaves a trajectory tail, one leaves half a restart pair, one leaves a complete restart pair
that no commit record points at, and one leaves a commit whose cache has not caught up. A test that
cannot say which one it hit cannot claim any of them are covered.

These hooks make the boundary selectable. They are armed only by an environment variable naming one
boundary, so nothing here runs in production: with the variable unset, `crash_point` reads one
`os.environ.get` and returns.

`os._exit` is used deliberately rather than `sys.exit` or an exception. It skips `finally` blocks,
`atexit` handlers and buffer flushing, which is what makes it a model of a process that was killed
rather than one that unwound cleanly -- and unwinding cleanly is precisely the behaviour these
tests must not accidentally rely on.
"""

from __future__ import annotations

import os

__all__ = ["CRASH_POINTS", "FAULT_ENV", "crash_point"]

#: Set to one of `CRASH_POINTS` to make the next arrival at that boundary kill the process.
FAULT_ENV = "MD_TEMPLATES_CRASH_AT"

#: The five boundaries, in the order a segment commit passes through them.
CRASH_POINTS = (
    # reporters still open: the trajectory carries frames past the last commit
    "before_reporters_close",
    # streams end exactly at the boundary, but no restart exists for it yet
    "after_reporters_close",
    # half a restart pair: the checkpoint is on disk, the portable State is not
    "after_checkpoint_member",
    # a complete restart pair that no commit record points at
    "after_restart_members",
    # committed, but run_state.json still describes the previous generation
    "after_atomic_commit",
)

#: Exit status used for an injected crash. 137 is what a SIGKILL victim reports through a shell,
#: so a test asserting on it is asserting on the same thing a real kill would produce.
CRASH_EXIT_STATUS = 137


def crash_point(name: str) -> None:
    """Kill this process if it was armed for boundary `name`. Otherwise do nothing at all."""
    if os.environ.get(FAULT_ENV) != name:
        return
    message = f"[fault-injection] exiting at boundary {name!r}\n"
    try:
        os.write(2, message.encode())
    except OSError:                                   # pragma: no cover - stderr already gone
        pass
    os._exit(CRASH_EXIT_STATUS)
