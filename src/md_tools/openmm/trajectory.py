"""What a trajectory file actually IS, decided from its bytes rather than from its name.

A suffix is a claim; the first few bytes are the fact. The two disagree more often than one would
like: a DCD renamed `.nc` because a pipeline expected NetCDF, a NetCDF written by a tool that
defaults to `.dcd`, a file copied under the wrong name. Every one of those reads perfectly well to
somebody and then fails, obscurely, several layers down inside a reader.

Only two formats are recognised, because only two are produced or consumed here:

    DCD     OpenMM's native writer, and the ordinary-MD output format. CHARMM-style: a 4-byte
            record length of 84 followed by the tag `CORD`.
    NetCDF  the AMBER convention, for AIS paths and REST2 state trajectories. Classic NetCDF
            begins `CDF\\x01`/`CDF\\x02`; the HDF5-backed NetCDF-4 form begins with the HDF5
            signature.

Anything else is `None` -- reported as unknown rather than guessed at. A wrong guess here would be
worse than no guess, because it would send the file to a reader that cannot read it.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["TRAJECTORY_SUFFIXES", "detect_trajectory_format", "check_trajectory_declaration",
           "describe_trajectory"]

#: The suffix each format is written with here. A file may of course be named anything; this is
#: what the two names are taken to CLAIM, and what a mismatch is reported against.
TRAJECTORY_SUFFIXES = {".dcd": "dcd", ".nc": "netcdf", ".netcdf": "netcdf", ".mdcrd": "netcdf"}

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def detect_trajectory_format(path: str | Path) -> str | None:
    """`"dcd"`, `"netcdf"`, or None when the leading bytes match neither."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return None
    if len(head) >= 8 and head[4:8] == b"CORD":
        # CHARMM DCD: a Fortran record marker of 84 then the tag. The marker's byte order varies,
        # so the tag is what is checked -- it is the same either way.
        return "dcd"
    if head[:3] == b"CDF" and len(head) >= 4 and head[3] in (1, 2, 5):
        return "netcdf"
    if head[:8] == _HDF5_SIGNATURE:
        return "netcdf"                                   # NetCDF-4, which is HDF5 underneath
    return None


def describe_trajectory(path: str | Path) -> str:
    """A short human phrase for a log line: the format, or that it could not be identified."""
    detected = detect_trajectory_format(path)
    return detected.upper() if detected else "an unrecognised format"


def check_trajectory_declaration(path: str | Path, *, what: str = "the trajectory") -> str:
    """Refuse a file whose suffix and contents disagree. Returns the detected format.

    A file that exists but is neither DCD nor NetCDF is refused too: it is going to fail in a
    reader either way, and failing here names the file and both formats instead of surfacing as
    an exception from inside mdtraj.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"{what} {path} "
            + ("is a directory, not a file" if path.is_dir() else "does not exist"))

    detected = detect_trajectory_format(path)
    claimed = TRAJECTORY_SUFFIXES.get(path.suffix.lower())

    if detected is None:
        raise SystemExit(
            f"{what} {path} is neither a DCD nor a NetCDF trajectory: its leading bytes match "
            f"no format this reads. Supported: DCD (OpenMM's ordinary-MD output) and AMBER "
            f"NetCDF (AIS paths, REST2 state trajectories).")
    if claimed is not None and claimed != detected:
        raise SystemExit(
            f"{what} {path} is named {path.suffix} -- which declares {claimed.upper()} -- but its "
            f"contents are {detected.upper()}.\n"
            f"  A suffix is a claim and the bytes are the fact, so this is refused rather than "
            f"read as one format and recorded as the other. Renaming a file does not convert it; "
            f"convert it, or name it for what it is.")
    return detected


def count_frames(path: str | Path) -> int:
    """How many frames the file holds, decided from its CONTENT like everything else here.

    Used by the checkpoint transaction: a commit records what each appendable stream held at the
    moment it was committed, and a resume cuts back to that. Both numbers have to come from the
    file itself, because the whole point is that the file and the bookkeeping can disagree.
    """
    import mdtraj

    path = Path(path)
    kind = detect_trajectory_format(path)
    if kind == "netcdf":
        with mdtraj.formats.NetCDFTrajectoryFile(str(path)) as handle:
            return len(handle)
    if kind == "dcd":
        with mdtraj.formats.DCDTrajectoryFile(str(path)) as handle:
            return len(handle)
    raise ValueError(f"{path} is not a trajectory this build can count "
                     f"({describe_trajectory(path)})")


def truncate_frames(path: str | Path, keep: int) -> int:
    """Rewrite `path` holding only its first `keep` frames. Returns how many it now holds.

    Rewritten through a temporary and moved into place with `os.replace`, so an interruption
    HERE leaves the original -- a truncation that is itself interrupted must not be the thing
    that destroys the trajectory it was repairing.

    A file already at or below `keep` is left untouched. Fewer frames than committed is a
    different failure -- records the checkpoint believes exist have been lost -- and silently
    padding or accepting it would hide it.
    """
    import os
    import tempfile

    import mdtraj

    path = Path(path)
    keep = int(keep)
    have = count_frames(path)
    if have <= keep:
        return have

    kind = detect_trajectory_format(path)
    opener = (mdtraj.formats.NetCDFTrajectoryFile if kind == "netcdf"
              else mdtraj.formats.DCDTrajectoryFile)
    handle, staging = tempfile.mkstemp(dir=str(path.parent), suffix=path.suffix)
    os.close(handle)
    staging = Path(staging)
    staging.unlink(missing_ok=True)                    # the writers want to create it themselves
    try:
        with opener(str(path)) as source, opener(str(staging), "w") as destination:
            payload = source.read(keep)
            if kind == "netcdf":
                coordinates, time, lengths, angles = payload
                destination.write(coordinates, time=time, cell_lengths=lengths,
                                  cell_angles=angles)
            else:
                coordinates, lengths, angles = payload
                destination.write(coordinates, cell_lengths=lengths, cell_angles=angles)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return count_frames(path)
