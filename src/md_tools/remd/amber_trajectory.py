"""Amber NetCDF trajectories: one file per fixed thermodynamic state.

A REST2 ladder has N fixed thermodynamic states. A walker moves between them when an exchange is
accepted, so the trajectory of a STATE is not the trajectory of a walker: after an accepted swap,
the next frame written to `whole_state2_prod1.nc` comes from whichever configuration now occupies
state 2.

The filename carries the state INDEX and never tau. `whole_state2_prod1.nc` means state 2; what
tau that state
holds belongs in validated metadata, where it can be checked, and not in a filename that anyone
can rename. Nothing here sorts trajectories lexicographically or parses a number back out of a
path -- the index is passed in.

WHAT MAKES THIS AN AMBER FILE, NOT MERELY A NETCDF ONE
    The Amber convention is a specific set of dimensions, variable names, units and global
    attributes. cpptraj identifies a file by `Conventions = "AMBER"` and then requires the rest to
    be there. This writes that convention exactly, and the tests read the result back with the
    installed cpptraj rather than asserting on our own bytes.

    Coordinates are angstrom; OpenMM works in nanometre, so the conversion happens here, once, at
    the boundary. Times are picoseconds in both.

NVT ONLY
    This milestone is NVT. The cell is written once per frame from a fixed box, which is what the
    Amber NVT convention expects. There is no variable-cell or NPT handling here, deliberately.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: What cpptraj looks for to identify the file at all.
AMBER_CONVENTIONS = "AMBER"
AMBER_CONVENTION_VERSION = "1.0"

#: OpenMM works in nanometre; Amber trajectories are angstrom.
ANGSTROM_PER_NM = 10.0


def state_trajectory_name(state_index, *, content="whole", segment=1):
    """`whole_state0_prod1.nc`, `solute_state0_prod1.nc`, ...

    THE INDEX IS THE STATE'S, not the walker's, and the name says so deliberately.

    Amber and GROMACS both write WALKER-following trajectories and sort them afterwards --
    cpptraj's own help says its default is to sort "by replica", and `remdtrajtemp` exists to
    extract frames at a given temperature, which is only necessary because the files are not
    temperature-sorted. GROMACS ships `demux.pl` for the same reason.

    MD-tools writes the sorted thing directly: after an accepted exchange the configuration now
    occupying state 2 is a different walker's, and it is that one which goes to state 2's file.
    So calling these `rep<i>` would match Amber's FILENAME while holding the opposite CONTENT,
    which is worse than not matching at all. `state<i>` describes what is in the file.

    A genuine walker-following trajectory is recoverable from `state_to_walker` in the exchange
    record, and would deserve its own name -- `walker<i>` -- rather than reusing this one.

    `tau` never appears in the name: two ladders with different tau at the same index would then
    write different files for the same state, and every script that globs them would silently
    read one run's ladder as another's.
    """
    index = int(state_index)
    if index < 0:
        raise ValueError(f"a state index cannot be negative; got {state_index}")
    if content not in ("whole", "solute"):
        raise ValueError(f"content must be 'whole' or 'solute'; got {content!r}")
    return f"{content}_state{index}_prod{int(segment)}.nc"


def state_cv_name(state_index, *, suffix="csv"):
    """`cv_state<i>.csv` / `.json` -- the CV series and its sidecar, for one STATE.

    `<content>_state<i>`, the same shape as `solute_state<i>_prod<N>.nc` and
    `whole_state<i>_prod<N>.nc`, because it describes the same thing: whatever occupied state i.
    These were `remd<i>.cv.csv` -- the last place the pre-rename name survived, which left one
    run directory using two spellings of "state i" and no way to tell from a filename whether a
    given file followed the state or the walker.
    """
    index = int(state_index)
    if index < 0:
        raise ValueError(f"a state index cannot be negative; got {state_index}")
    return f"cv_state{index}.{suffix}"


def state_index_from_name(name):
    """Only for validating a groupfile that names its outputs. Never used to discover order."""
    import re

    stem = Path(name).name
    matched = re.fullmatch(r"(whole|solute)_state(\d+)_prod(\d+)\.nc", stem)
    if matched:
        return int(matched.group(2))
    raise ValueError(
        f"{stem!r} is not a state trajectory name; expected "
        f"<whole|solute>_state<index>_prod<segment>.nc")
    digits = stem[len("remd"):-len(".nc")]
    if not digits.isdigit():
        raise ValueError(f"{stem!r} does not carry a numeric state index")
    return int(digits)


def validate_state_outputs(names):
    """A groupfile must name exactly one output per state, contiguous from zero.

    Checked before any file is created: a duplicate or a gap means two states would share a
    trajectory or one would have none, and discovering that after writing has begun leaves a
    dataset nobody can interpret.
    """
    indices = []
    for name in names:
        indices.append(state_index_from_name(name))
    duplicates = sorted({i for i in indices if indices.count(i) > 1})
    if duplicates:
        raise ValueError(
            f"state trajectory index/indices {duplicates} appear more than once in "
            f"{[Path(n).name for n in names]}; each state owns exactly one trajectory")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(
            f"state trajectory indices {sorted(indices)} are not contiguous from 0; "
            f"expected {list(range(len(indices)))}")
    if indices != sorted(indices):
        raise ValueError(
            f"state trajectories are given out of order: {[Path(n).name for n in names]}. "
            f"The position in the group file is the state index, so order is meaning.")
    return indices


class AmberTrajectoryWriter:
    """One Amber NetCDF trajectory, for one fixed thermodynamic state."""

    def __init__(self, path, *, n_atoms, tau, temperature_k, periodic, state_index=None,
                 application="REST2", program="md-tools", program_version="0"):
        import netCDF4

        self.path = Path(path)
        self.n_atoms = int(n_atoms)
        self.periodic = bool(periodic)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        dataset = netCDF4.Dataset(str(self.path), "w", format="NETCDF3_64BIT_OFFSET")
        self.dataset = dataset

        dataset.Conventions = AMBER_CONVENTIONS
        dataset.ConventionVersion = AMBER_CONVENTION_VERSION
        dataset.program = program
        dataset.programVersion = str(program_version)
        dataset.application = str(application)
        # Ours, not Amber's: the Hamiltonian this file was sampled at, so it can be CHECKED
        # rather than inferred from a filename or a directory name.
        #
        # `state_index` belongs to a ladder and is omitted for anything else. A conventional
        # stage wrote `application: REST2`, `title: REST2 state 0`, `tau: 0.0` and
        # `temperature_k: 0.0` into every file, because these four were hard-coded here and the
        # stage reporter had nothing to pass. A hot cMD run at tau = 0.5 -- which is exactly what
        # an AIS source ensemble is -- therefore recorded tau = 0.0, and the one place that could
        # have contradicted the directory name agreed with the wrong answer instead.
        if state_index is not None:
            dataset.title = f"{application} state {int(state_index)}"
            dataset.state_index = int(state_index)
        else:
            dataset.title = f"{application} tau={float(tau):g}"
        dataset.tau = float(tau)
        dataset.temperature_k = float(temperature_k)

        dataset.createDimension("frame", None)
        dataset.createDimension("spatial", 3)
        dataset.createDimension("atom", self.n_atoms)

        spatial = dataset.createVariable("spatial", "S1", ("spatial",))
        spatial[:] = np.array(list("xyz"), dtype="S1")

        time = dataset.createVariable("time", "f4", ("frame",))
        time.units = "picosecond"

        coordinates = dataset.createVariable("coordinates", "f4", ("frame", "atom", "spatial"))
        coordinates.units = "angstrom"

        if self.periodic:
            dataset.createDimension("cell_spatial", 3)
            dataset.createDimension("cell_angular", 3)
            dataset.createDimension("label", 5)
            cell_spatial = dataset.createVariable("cell_spatial", "S1", ("cell_spatial",))
            cell_spatial[:] = np.array(list("abc"), dtype="S1")
            cell_angular = dataset.createVariable("cell_angular", "S1",
                                                  ("cell_angular", "label"))
            cell_angular[:] = np.array([list("alpha"), list("beta "), list("gamma")], dtype="S1")
            lengths = dataset.createVariable("cell_lengths", "f8", ("frame", "cell_spatial"))
            lengths.units = "angstrom"
            angles = dataset.createVariable("cell_angles", "f8", ("frame", "cell_angular"))
            angles.units = "degree"
        self._frames = 0

    @classmethod
    def open_existing(cls, path, *, n_atoms, tau, from_frame, state_index=None):
        """Reopen a state trajectory for append, positioned at the committed marker.

        `from_frame` is the committed-frame count from the authoritative record, and it is where
        writing resumes. Rows at or past it are UNCOMMITTED -- a crash left them behind after the
        set was written but before the marker moved -- and they are overwritten in place. An Amber
        NetCDF cannot shrink, so overwriting is how they are retired; the marker, not the file
        length, is what says how many frames exist.

        Every identifying attribute is checked before the file is opened for writing. Continuing
        into a file that belongs to another state, another ladder or another system would corrupt
        it silently, and the filename alone is not evidence of any of those.
        """
        import netCDF4

        path = Path(path)
        seen = read_frames(path)
        if seen["conventions"] != AMBER_CONVENTIONS:
            raise ValueError(
                f"{path.name} is not an Amber trajectory (Conventions={seen['conventions']!r})")
        if state_index is not None and seen["state_index"] != int(state_index):
            raise ValueError(
                f"{path.name} records state {seen['state_index']}, not {int(state_index)}")
        if abs(seen["tau"] - float(tau)) > 1e-12:
            raise ValueError(
                f"{path.name} records tau {seen['tau']} but this ladder puts {float(tau)} at "
                f"state {int(state_index)}. This is a different ladder, not a continuation.")
        if seen["n_atoms"] != int(n_atoms):
            raise ValueError(
                f"{path.name} holds {seen['n_atoms']} atoms, not {int(n_atoms)}")
        if int(from_frame) > seen["n_frames"]:
            raise ValueError(
                f"{path.name} holds {seen['n_frames']} frames but the committed marker says "
                f"{int(from_frame)} exist. The marker only advances once every file has the row, "
                f"so this is corruption; it is refused rather than padded.")

        writer = object.__new__(cls)
        writer.path = path
        writer.n_atoms = int(n_atoms)
        writer.periodic = bool(seen["periodic"])
        writer.dataset = netCDF4.Dataset(str(path), "a", format="NETCDF3_64BIT_OFFSET")
        writer._frames = int(from_frame)
        return writer

    @property
    def n_frames(self):
        return self._frames

    def append(self, positions_nm, *, time_ps, box_nm=None):
        """One frame. Positions arrive in nanometre and are written in angstrom."""
        positions = np.asarray(positions_nm, dtype=float)
        if positions.shape != (self.n_atoms, 3):
            raise ValueError(
                f"{self.path.name}: expected positions of shape ({self.n_atoms}, 3), got "
                f"{positions.shape}")
        index = self._frames
        self.dataset.variables["coordinates"][index, :, :] = positions * ANGSTROM_PER_NM
        self.dataset.variables["time"][index] = float(time_ps)
        if self.periodic:
            if box_nm is None:
                raise ValueError(
                    f"{self.path.name} is periodic but no box was given; a configuration without "
                    f"its box is at an undefined density.")
            box = np.asarray(box_nm, dtype=float)
            lengths = np.linalg.norm(box, axis=1) * ANGSTROM_PER_NM
            self.dataset.variables["cell_lengths"][index, :] = lengths
            # NVT with a fixed rectangular cell; the angles are constant and written per frame
            # because the Amber convention stores them that way.
            self.dataset.variables["cell_angles"][index, :] = _cell_angles(box)
        self._frames += 1
        return index

    def sync(self):
        self.dataset.sync()

    def close(self):
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _cell_angles(box_nm):
    """alpha, beta, gamma in degrees, from the three box vectors."""
    a, b, c = (np.asarray(v, dtype=float) for v in box_nm)

    def angle(u, v):
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm == 0:
            return 90.0
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1.0, 1.0))))

    return [angle(b, c), angle(a, c), angle(a, b)]


def read_frames(path):
    """Everything a validator needs, read-only: counts, times, and the recorded state."""
    import netCDF4

    with netCDF4.Dataset(str(path), "r") as dataset:
        return {
            "conventions": getattr(dataset, "Conventions", None),
            "convention_version": getattr(dataset, "ConventionVersion", None),
            "state_index": int(getattr(dataset, "state_index", -1)),
            "tau": float(getattr(dataset, "tau", float("nan"))),
            "n_frames": len(dataset.dimensions["frame"]),
            "n_atoms": len(dataset.dimensions["atom"]),
            "times_ps": np.array(dataset.variables["time"][:], dtype=float).tolist(),
            "periodic": "cell_lengths" in dataset.variables,
        }
