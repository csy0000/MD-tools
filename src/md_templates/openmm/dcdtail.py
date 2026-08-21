"""Reading and truncating OpenMM DCD files by their actual records.

Finding 3. The previous truncation computed a frame as three coordinate blocks and nothing else::

    frame_bytes = 3 * (4 + 4 * n_atoms + 4)

That is right only for a *nonperiodic* DCD. OpenMM writes a periodic frame as a 48-byte unit-cell
payload wrapped in Fortran record markers -- **56 bytes** -- followed by the three coordinate
blocks. Truncating an explicit trajectory with the coordinate-only arithmetic lands 56 bytes per
frame short of the real boundary and corrupts the file.

Worse, the flag cannot be inferred from the solvent mode. `DCDReporter(..., atomSubset=...)` takes
its box vectors from the *topology*, so a selected-atom trajectory written with
``enforcePeriodicBox=False`` still carries a unit cell when the topology has one. The only reliable
source is the flag in the file, which is what this module reads.

## The format, as OpenMM 8.5.2 writes it

::

    offset  0   int32   84            opening Fortran marker for the 84-byte control record
    offset  4   char[4] "CORD"
    offset  8   int32   NSET          number of frames            <- updated on truncation
    offset 12   int32   first step
    offset 16   int32   interval
    offset 20   int32   last step     = first + (NSET-1)*interval <- updated on truncation
    offset 44   float32 dt
    offset 48   int32   boxFlag       1 if frames carry a unit cell
    offset 88   int32   84            closing marker
    offset 92   int32   164           title record
    offset 96   int32   NTITLE
    ...         char[80] * NTITLE
    ...         int32   164           closing marker
    ...         int32   4, natom, 4   atom-count record

then, per frame::

    [ int32 48 | 6 * float64 | int32 48 ]      only when boxFlag
    [ int32 4*natom | float32 * natom | int32 4*natom ]   x, then y, then z

Every length is read from the file and every Fortran marker is checked against its partner, so a
file that does not match this shape is refused rather than silently mis-parsed.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

__all__ = [
    "DCDFormatError",
    "read_header",
    "frame_offsets",
    "truncate_to_frames",
]

#: Byte offsets of the two header fields truncation has to keep consistent with the frame count.
_NSET_OFFSET = 8
_LAST_STEP_OFFSET = 20
_FIRST_STEP_OFFSET = 12
_INTERVAL_OFFSET = 16
_BOX_FLAG_OFFSET = 48


class DCDFormatError(ValueError):
    """A DCD that does not match the record structure this module can safely edit."""


def _read_marker(handle, where: str) -> int:
    raw = handle.read(4)
    if len(raw) != 4:
        raise DCDFormatError(f"{where}: file ended inside a Fortran record marker")
    return struct.unpack("<i", raw)[0]


def read_header(path: Path) -> dict:
    """Parse the DCD header. Returns frame count, atom count, periodicity and the data offset.

    Endianness is validated rather than assumed: OpenMM writes little-endian, and the opening marker
    of the control record is always 84. A file whose first four bytes are not 84 in little-endian is
    either big-endian or not a DCD, and either way this module must not edit it.
    """
    path = Path(path)
    with path.open("rb") as handle:
        opening = _read_marker(handle, "control record")
        if opening != 84:
            big_endian = struct.unpack(">i", struct.pack("<i", opening))[0]
            raise DCDFormatError(
                f"{path}: control record marker is {opening}, expected 84. "
                + (f"It is 84 when read big-endian ({big_endian}), so this file was written on a "
                   "big-endian machine and is not safe to edit here."
                   if big_endian == 84 else "This is not a DCD file this module can parse.")
            )
        if handle.read(4) != b"CORD":
            raise DCDFormatError(f"{path}: missing CORD signature")

        handle.seek(_NSET_OFFSET)
        n_frames = struct.unpack("<i", handle.read(4))[0]
        first_step = struct.unpack("<i", handle.read(4))[0]
        interval = struct.unpack("<i", handle.read(4))[0]
        handle.seek(_BOX_FLAG_OFFSET)
        box_flag = struct.unpack("<i", handle.read(4))[0]

        handle.seek(88)
        closing = _read_marker(handle, "control record close")
        if closing != 84:
            raise DCDFormatError(f"{path}: control record closes with {closing}, expected 84")

        title_open = _read_marker(handle, "title record")
        n_title = struct.unpack("<i", handle.read(4))[0]
        if title_open != 4 + 80 * n_title:
            raise DCDFormatError(
                f"{path}: title record declares {title_open} bytes but carries {n_title} titles")
        handle.seek(80 * n_title, os.SEEK_CUR)
        if _read_marker(handle, "title record close") != title_open:
            raise DCDFormatError(f"{path}: title record markers disagree")

        if _read_marker(handle, "atom count") != 4:
            raise DCDFormatError(f"{path}: atom-count record is not 4 bytes")
        n_atoms = struct.unpack("<i", handle.read(4))[0]
        if _read_marker(handle, "atom count close") != 4:
            raise DCDFormatError(f"{path}: atom-count record markers disagree")

        data_start = handle.tell()

    if n_atoms <= 0:
        raise DCDFormatError(f"{path}: atom count {n_atoms} is not positive")
    return {
        "path": path,
        "declared_frames": n_frames,
        "n_atoms": n_atoms,
        "periodic": bool(box_flag),
        "first_step": first_step,
        "interval": interval,
        "data_start": data_start,
        "file_size": path.stat().st_size,
    }


def frame_offsets(path: Path) -> dict:
    """Walk the real records and return the byte offset just past each COMPLETE frame.

    Walking rather than multiplying is the point: it validates every marker on the way, so a
    truncated or malformed tail is discovered instead of being absorbed into a wrong offset.

    Returns the header plus `boundaries` (one offset per complete frame) and `trailing_bytes`
    (whatever follows the last complete frame -- a partially written frame from a crash).
    """
    header = read_header(path)
    n_atoms = header["n_atoms"]
    coordinate_payload = 4 * n_atoms
    boundaries: list[int] = []

    with Path(path).open("rb") as handle:
        handle.seek(header["data_start"])
        while True:
            frame_start = handle.tell()
            try:
                if header["periodic"]:
                    if _read_marker(handle, "unit cell") != 48:
                        raise DCDFormatError(
                            f"{path}: frame {len(boundaries)} unit-cell record is not 48 bytes")
                    handle.seek(48, os.SEEK_CUR)
                    if _read_marker(handle, "unit cell close") != 48:
                        raise DCDFormatError(
                            f"{path}: frame {len(boundaries)} unit-cell markers disagree")
                for axis in "xyz":
                    opening = _read_marker(handle, f"{axis} block")
                    if opening != coordinate_payload:
                        raise DCDFormatError(
                            f"{path}: frame {len(boundaries)} {axis} block declares {opening} "
                            f"bytes, expected {coordinate_payload} for {n_atoms} atoms")
                    handle.seek(coordinate_payload, os.SEEK_CUR)
                    if _read_marker(handle, f"{axis} block close") != opening:
                        raise DCDFormatError(
                            f"{path}: frame {len(boundaries)} {axis} block markers disagree")
            except DCDFormatError:
                # An incomplete record is a partially written frame, not a corrupt file: the writer
                # died mid-frame. Stop at the last COMPLETE boundary and report the remainder.
                handle.seek(frame_start)
                break
            if handle.tell() > header["file_size"]:
                handle.seek(frame_start)
                break
            boundaries.append(handle.tell())
            if handle.tell() == header["file_size"]:
                break

        complete_end = boundaries[-1] if boundaries else header["data_start"]

    return {
        **header,
        "boundaries": boundaries,
        "complete_frames": len(boundaries),
        "trailing_bytes": header["file_size"] - complete_end,
    }


def truncate_to_frames(path: Path, keep: int) -> dict:
    """Cut a DCD back to `keep` complete frames, atomically.

    Both header fields are updated together. Leaving the last-step field pointing past the new
    final frame would produce a file whose frame count and time axis disagree, which any reader
    would interpret as a different trajectory than the one on disk.

    Written to a temporary file in the SAME directory, flushed, fsynced and renamed, so a crash
    during recovery cannot leave a half-truncated trajectory under the real name.
    """
    path = Path(path)
    scan = frame_offsets(path)
    if keep < 0:
        raise ValueError("cannot keep a negative number of frames")
    if keep > scan["complete_frames"]:
        raise DCDFormatError(
            f"{path}: asked to keep {keep} frames but only {scan['complete_frames']} complete "
            "frames are present. A file shorter than its committed watermark is missing history, "
            "not carrying a tail."
        )

    end = scan["boundaries"][keep - 1] if keep else scan["data_start"]
    if end == scan["file_size"] and keep == scan["declared_frames"]:
        return {"action": "kept", "frames": keep, "trailing_bytes": scan["trailing_bytes"]}

    directory = path.parent
    handle_fd, temporary = tempfile.mkstemp(dir=str(directory), prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle_fd)
    try:
        with path.open("rb") as source, open(temporary, "wb") as target:
            remaining = end
            while remaining > 0:
                block = source.read(min(1 << 20, remaining))
                if not block:
                    raise DCDFormatError(f"{path}: file ended before the computed frame boundary")
                target.write(block)
                remaining -= len(block)

            target.seek(_NSET_OFFSET)
            target.write(struct.pack("<i", keep))
            last_step = scan["first_step"] + (keep - 1) * scan["interval"] if keep else 0
            target.seek(_LAST_STEP_OFFSET)
            target.write(struct.pack("<i", last_step))

            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    return {
        "action": "truncated",
        "from_frames": scan["declared_frames"],
        "from_complete_frames": scan["complete_frames"],
        "frames": keep,
        "discarded_trailing_bytes": scan["file_size"] - end,
        "periodic": scan["periodic"],
        "n_atoms": scan["n_atoms"],
    }
