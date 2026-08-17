"""Restart I/O that touches an OpenMM `Simulation`: the engine half of the persistence contract.

`md_templates.core.persistence` owns the layout and the bookkeeping -- generation directories,
atomic writes, the committed-generation record, quarantine of an uncommitted tail. None of that
needs an engine. Writing and restoring a checkpoint or a serialized `State` does, so it lives here,
with the provider that knows what a `Simulation` is.

The split is what lets core stay importable in an environment with no OpenMM at all, which is the
property a catalog CLI and a listing command depend on. It is also the honest boundary: everything
above this line is a file-format question, everything below it is a question about a running system.

Portability, unchanged from before the move: the binary checkpoint is an exact same-platform
continuation; the serialized State is a portable fallback that preserves positions, velocities, box,
time and parameters but NOT a stochastic integrator's internal stream, so its use is announced and
recorded rather than silent.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from ...core.persistence import RunStateError, atomic_write_bytes, restart_members

__all__ = ["save_restart", "load_restart", "restart_members"]


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_restart(sim, gdir: Path, *, replica: Optional[int] = None) -> tuple[Path, Path]:
    """Write BOTH restart forms for one simulation into a generation directory.

    The checkpoint is the exact same-platform continuation; the serialized State is the portable
    fallback and carries positions, velocities, box vectors, time and parameters. Both are written
    through a temporary file and renamed, so a crash cannot leave a truncated file under a name the
    commit record will later point at.
    """
    from openmm import XmlSerializer

    gdir = Path(gdir)
    gdir.mkdir(parents=True, exist_ok=True)
    chk_name, state_name = restart_members(replica)

    fd, tmp_chk = tempfile.mkstemp(dir=str(gdir), prefix=f".{chk_name}.", suffix=".tmp")
    os.close(fd)
    sim.saveCheckpoint(tmp_chk)
    # OpenMM closes the file it wrote, but the bytes may still be in the page cache. fsync before
    # the rename, or a commit record could point at a checkpoint that a power loss truncates.
    with open(tmp_chk, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_chk, gdir / chk_name)
    _fsync_dir(gdir)

    state = sim.context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                 enforcePeriodicBox=False)
    atomic_write_bytes(gdir / state_name, XmlSerializer.serialize(state).encode("utf-8"))
    return gdir / chk_name, gdir / state_name


def load_restart(sim, gdir: Path, *, replica: Optional[int] = None,
                 allow_state_fallback: bool = True) -> str:
    """Restore one simulation from a committed generation. Returns "checkpoint" or "state".

    The binary checkpoint is preferred because it continues bit-for-bit on the same platform. If it
    is missing, corrupt, or was written by another platform, the portable State is used instead --
    that preserves positions, velocities, box, time and parameters, so the continuation is
    physically valid, but the stochastic integrator's internal stream is NOT restored and the
    trajectory diverges from the one an uninterrupted run would have produced. Reported, never
    silent.
    """
    from openmm import XmlSerializer

    gdir = Path(gdir)
    chk_name, state_name = restart_members(replica)
    chk, state_path = gdir / chk_name, gdir / state_name

    if chk.is_file():
        try:
            sim.loadCheckpoint(str(chk))
            return "checkpoint"
        except Exception as exc:                                     # noqa: BLE001
            if not allow_state_fallback:
                raise
            print(f"[restart] checkpoint {chk} unusable ({type(exc).__name__}: {exc}); "
                  "falling back to the serialized State", flush=True)

    if not allow_state_fallback or not state_path.is_file():
        raise RunStateError(
            f"no usable restart in {gdir}: checkpoint "
            f"{'unusable' if chk.is_file() else 'missing'} and "
            f"{'State missing' if not state_path.is_file() else 'State fallback disabled'}. "
            "Refusing to append output to a run that cannot be continued."
        )
    sim.context.setState(XmlSerializer.deserialize(state_path.read_text(encoding="utf-8")))
    return "state"
