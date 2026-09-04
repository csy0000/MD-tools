#!/usr/bin/env python
"""A phase-space ensemble: positions AND velocities AND boxes, in one validated NetCDF.

Copied verbatim into every generated project that produces or consumes one.

WHY THIS FORMAT EXISTS
    A DCD cannot store velocities. That is not a limitation to work around -- it means a DCD is a
    CONFIGURATION trajectory and not a phase-space ensemble, and a reservoir built from one cannot
    install a stored momentum because there is none to install.

    The previous rREST2 implementation drew Maxwell velocities on every refresh and described the
    result as a phase-space reservoir. Redrawing momenta is a legitimate, separately justified
    policy; it is not the same operation, and calling it the same made the difference invisible.

    So a phase-space source is its own format, distinct from AIS's coordinate-only `sources.dcd`,
    and it is written by the run that generated the ensemble -- the only place the velocities
    exist.

WHAT IS STORED
    positions[frame, atom, 3]     nanometres
    velocities[frame, atom, 3]    nanometres / picosecond
    box[frame, 3, 3]              nanometres, absent when the system is not periodic
    step[frame]                   ABSOLUTE integration step in the source run
    time_ps[frame]                physical time of that step

    plus, as attributes: the format version, the source run's identity, the Hamiltonian
    fingerprint, tau, temperature, ensemble, atom count and ordering digest, and the template
    identity that wrote it.

`last_frame` is written only AFTER a frame's arrays are complete, so a file truncated by a crash
reads back as the frames that were fully committed and no more.
"""
import datetime
import json
import os
from pathlib import Path

import numpy as np

#: Bumped when the meaning of the schema changes.
PHASE_SPACE_FORMAT = "md-tools-phase-space/v1"

#: Units, stated in the file rather than assumed by every reader.
POSITION_UNIT = "nanometer"
VELOCITY_UNIT = "nanometer/picosecond"
BOX_UNIT = "nanometer"


class PhaseSpaceError(RuntimeError):
    """The phase-space file cannot be written, opened, or trusted."""


def _atom_order_digest(topology):
    """A digest of the per-index atom identity, so a reordered topology cannot pass unnoticed."""
    import hashlib

    rows = []
    for atom in topology.atoms():
        residue = atom.residue
        rows.append((str(getattr(residue.chain, "id", "") or ""), int(residue.chain.index),
                     int(residue.index), str(residue.name), str(atom.name),
                     atom.element.symbol if atom.element is not None else ""))
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


class PhaseSpaceWriter:
    """Streams phase-space frames out of a running simulation.

    Used by a fixed-tau cMD run: it is the only place the velocities exist, so it is the only
    place they can be recorded.
    """

    def __init__(self, path, *, n_atoms, periodic, identity):
        import netCDF4

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.periodic = bool(periodic)
        dataset = netCDF4.Dataset(str(self.path), "w")
        dataset.format_version = PHASE_SPACE_FORMAT
        dataset.created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset.identity_json = json.dumps(identity, sort_keys=True, default=str)
        dataset.position_unit = POSITION_UNIT
        dataset.velocity_unit = VELOCITY_UNIT
        dataset.box_unit = BOX_UNIT
        dataset.createDimension("frame", None)
        dataset.createDimension("atom", int(n_atoms))
        dataset.createDimension("spatial", 3)
        dataset.createDimension("cell", 3)
        positions = dataset.createVariable("positions", "f8", ("frame", "atom", "spatial"))
        positions.units = POSITION_UNIT
        velocities = dataset.createVariable("velocities", "f8", ("frame", "atom", "spatial"))
        velocities.units = VELOCITY_UNIT
        velocities.long_name = ("velocities as recorded by the source run; a phase-space sample "
                                "is not a configuration")
        if self.periodic:
            box = dataset.createVariable("box", "f8", ("frame", "cell", "spatial"))
            box.units = BOX_UNIT
        step = dataset.createVariable("step", "i8", ("frame",))
        step.long_name = "absolute integration step in the source run"
        dataset.createVariable("time_ps", "f8", ("frame",))
        last = dataset.createVariable("last_frame", "i8")
        last.long_name = ("the last FULLY written frame; written after its arrays, so a truncated "
                          "file reads back as what was committed")
        last[0] = -1
        dataset.sync()
        self.dataset = dataset
        self._frames = 0

    def append(self, *, positions, velocities, box, step, time_ps):
        index = self._frames
        variables = self.dataset.variables
        variables["positions"][index, :, :] = np.asarray(positions, dtype=float)
        variables["velocities"][index, :, :] = np.asarray(velocities, dtype=float)
        if self.periodic:
            if box is None:
                raise PhaseSpaceError(
                    "this source is periodic but a frame carried no box; a configuration without "
                    "its box is at an undefined density")
            variables["box"][index, :, :] = np.asarray(box, dtype=float)
        variables["step"][index] = int(step)
        variables["time_ps"][index] = float(time_ps)
        self.dataset.sync()
        variables["last_frame"][0] = index
        self.dataset.sync()
        self._frames += 1
        return index

    def close(self):
        try:
            self.dataset.close()
        except Exception:
            pass


class PhaseSpaceReader:
    """Reads and VALIDATES a phase-space file. Every check is a read, never an existence test."""

    def __init__(self, path):
        import netCDF4

        self.path = Path(path)
        if not self.path.is_file():
            raise PhaseSpaceError(f"{self.path} does not exist")
        try:
            self.dataset = netCDF4.Dataset(str(self.path), "r")
        except Exception as failure:
            raise PhaseSpaceError(
                f"{self.path} could not be opened as NetCDF ({type(failure).__name__}: "
                f"{failure}). A truncated or corrupt phase-space file fails here.") from None
        version = getattr(self.dataset, "format_version", None)
        if version != PHASE_SPACE_FORMAT:
            self.close()
            raise PhaseSpaceError(
                f"{self.path} is {version!r}, not {PHASE_SPACE_FORMAT!r}. A coordinate-only "
                f"trajectory such as a DCD is NOT a phase-space reservoir: it cannot store "
                f"velocities, so there is nothing for `velocity_policy: stored` to install.")
        for required in ("positions", "velocities", "step", "time_ps", "last_frame"):
            if required not in self.dataset.variables:
                self.close()
                raise PhaseSpaceError(
                    f"{self.path} has no `{required}` variable; it is not a complete phase-space "
                    f"file even though it opened.")

    # -- properties ------------------------------------------------------------------------------

    @property
    def identity(self):
        raw = getattr(self.dataset, "identity_json", None)
        return None if raw is None else json.loads(raw)

    @property
    def periodic(self):
        return "box" in self.dataset.variables

    @property
    def n_atoms(self):
        return int(self.dataset.dimensions["atom"].size)

    @property
    def n_frames(self):
        """Only the FULLY committed frames. A partial trailing write is not a frame."""
        return int(self.dataset.variables["last_frame"][0]) + 1

    @property
    def units(self):
        return {"positions": getattr(self.dataset, "position_unit", None),
                "velocities": getattr(self.dataset, "velocity_unit", None),
                "box": getattr(self.dataset, "box_unit", None)}

    # -- reading ---------------------------------------------------------------------------------

    def frame(self, index):
        """`(positions, velocities, box_or_None, step, time_ps)` for one committed frame."""
        count = self.n_frames
        if not 0 <= int(index) < count:
            raise PhaseSpaceError(f"frame {index} is outside 0..{count - 1}")
        variables = self.dataset.variables
        positions = np.array(variables["positions"][int(index)], dtype=float)
        velocities = np.array(variables["velocities"][int(index)], dtype=float)
        box = (np.array(variables["box"][int(index)], dtype=float) if self.periodic else None)
        return (positions, velocities, box,
                int(variables["step"][int(index)]), float(variables["time_ps"][int(index)]))

    def steps(self):
        return np.array(self.dataset.variables["step"][:self.n_frames], dtype=np.int64)

    def times(self):
        return np.array(self.dataset.variables["time_ps"][:self.n_frames], dtype=float)

    def validate(self, *, expect_atoms=None, expect_periodic=None, require_velocities=False):
        """Read every committed frame and refuse anything a run must not propagate.

        This is deliberately eager: a reservoir with one non-finite velocity would otherwise be
        discovered at the refresh that installs it, thousands of steps into a run.
        """
        problems = []
        count = self.n_frames
        if count < 1:
            problems.append("no frame was fully committed to this phase-space file")
        if expect_atoms is not None and self.n_atoms != int(expect_atoms):
            problems.append(
                f"the file holds {self.n_atoms} atoms but the system has {int(expect_atoms)}; a "
                f"phase-space sample must be the same particles in the same order")
        if expect_periodic is not None and self.periodic != bool(expect_periodic):
            problems.append(
                f"the file {'has' if self.periodic else 'has no'} box vectors but the system "
                f"{'is' if expect_periodic else 'is not'} periodic")
        for index in range(count):
            positions, velocities, box, step, _time = self.frame(index)
            if positions.shape != (self.n_atoms, 3):
                problems.append(f"frame {index}: positions have shape {positions.shape}")
            if velocities.shape != (self.n_atoms, 3):
                problems.append(
                    f"frame {index}: velocities have shape {velocities.shape}, expected "
                    f"({self.n_atoms}, 3)")
            if not np.all(np.isfinite(positions)):
                problems.append(f"frame {index}: non-finite positions")
            if not np.all(np.isfinite(velocities)):
                problems.append(f"frame {index}: non-finite velocities")
            if require_velocities and not np.any(velocities):
                # Identically zero passes every shape and finiteness check, so a file converted
                # from a coordinate-only source -- where the velocity variable was simply never
                # written -- would look valid and then be installed as zero momentum, which is
                # not a sample from any Boltzmann distribution.
                problems.append(
                    f"frame {index}: velocities are identically zero, so this is a "
                    f"configuration-only frame rather than a phase-space sample")
            if self.periodic and (box is None or not np.all(np.isfinite(box))):
                problems.append(f"frame {index}: missing or non-finite box")
            if step < 0:
                problems.append(f"frame {index}: negative absolute step {step}")
        steps = self.steps()
        if steps.size and not np.all(np.diff(steps) > 0):
            problems.append("absolute steps are not strictly increasing; frames are out of order "
                            "or duplicated")
        return problems

    def close(self):
        try:
            self.dataset.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False


def write_selection(path, *, source, indices, identity):
    """Copy selected frames out of one phase-space file into a smaller one, preserving everything.

    Used to materialise a reservoir once from a longer source: the selection carries the same
    positions, velocities, boxes, absolute steps and times, so a reservoir frame remains traceable
    to the exact step of the run that produced it.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    writer = PhaseSpaceWriter(temporary, n_atoms=source.n_atoms, periodic=source.periodic,
                              identity=identity)
    try:
        for index in indices:
            positions, velocities, box, step, time_ps = source.frame(int(index))
            writer.append(positions=positions, velocities=velocities, box=box, step=step,
                          time_ps=time_ps)
    finally:
        writer.close()
    os.replace(temporary, path)
    return path


class PhaseSpaceReporter:
    """An OpenMM reporter that streams phase space out of a running Simulation.

    Plugged into `simulation.reporters` exactly like a DCDReporter, so a generated protocol gains a
    phase-space stream in one line and nothing else changes:

        simulation.reporters.append(PhaseSpaceReporter(path, interval, identity=..., ...))

    It records the ABSOLUTE step OpenMM reports, not a frame counter, so a reservoir frame remains
    traceable to the exact step of the run that produced it even across a restart.
    """

    def __init__(self, path, interval_steps, *, identity, periodic, timestep_fs):
        self.path = Path(path)
        self.interval = int(interval_steps)
        if self.interval < 1:
            raise PhaseSpaceError(f"the phase-space interval must be >= 1 step; got {interval_steps}")
        self.identity = identity
        self.periodic = bool(periodic)
        self.timestep_fs = float(timestep_fs)
        self._writer = None

    def describeNextReport(self, simulation):          # noqa: N802 - OpenMM's interface
        steps = self.interval - simulation.currentStep % self.interval
        # positions, velocities, forces, energies, wrapped -- velocities are the point.
        return (steps, True, True, False, False, self.periodic or None)

    def report(self, simulation, state):
        from openmm import unit

        if self._writer is None:
            self._writer = PhaseSpaceWriter(
                self.path, n_atoms=simulation.system.getNumParticles(),
                periodic=self.periodic, identity=self.identity)
        box = None
        if self.periodic:
            box = np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                unit.nanometer), dtype=float)
        # THE ABSOLUTE STEP, from the Context and nowhere else. `loadCheckpoint` restores the
        # step count, so the `step_offset=done` that used to be added here double-counted every
        # completed step after a resume -- a reservoir frame at step 25 came back labelled 50, and
        # the reservoir's own time axis then disagreed with the run that produced it.
        step = int(simulation.currentStep)
        self._writer.append(
            positions=state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            velocities=state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond),
            box=box, step=step, time_ps=step * self.timestep_fs / 1000.0)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
