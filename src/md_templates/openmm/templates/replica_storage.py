#!/usr/bin/env python
"""The replica-exchange storage this repository OWNS. One documented NetCDF schema.

Copied verbatim into every generated replica project. Depends on `netCDF4` and nothing else.

THREE RECORDS, THREE JOBS
    analysis NetCDF (-x)   the authoritative history: mapping, stored frames, decision energies,
                           proposals and acceptances, reservoir events, physical time, and the
                           scientific identity written BEFORE propagation begins.
    checkpoint NetCDF      the last committed configurations plus everything needed to continue:
                           positions, velocities, boxes, mapping, every RNG state, the rule's own
                           persistent state, the iteration and the original budget.
    restart.json (-r)      the atomic COMPLETED-run manifest. Evidence of completion; never a
                           precondition for resuming, because an interrupted run never writes one.

    A fourth, `<stem>.runstate.json`, is the atomic sidecar recording initialized / running /
    interrupted / failed / completed. It never claims completion.

THE SCHEMA
    Dimensions
        iteration   unlimited; one row per completed segment
        frame       unlimited; one row per STORED coordinate frame
        state       the ladder, ascending in tau
        walker      the continuous configurations; always as many as states
        atom, spatial, cell

    Variables (analysis)
        tau[state]                            the ladder, written once
        segment[iteration]                    which propagation segment this row is
        time_ps[iteration]                    physical time per replica at the end of it
        is_exchange[iteration]                whether an exchange was attempted here
        state_to_walker[iteration, state]     the mapping; every row a permutation
        proposed[iteration, state, state]     symmetric counts, non-cumulative
        accepted[iteration, state, state]     symmetric counts, non-cumulative
        u[iteration, state, state]            reduced potentials used for decisions
        u_evaluated[iteration, state, state]  1 where u was actually computed
        reservoir_state[iteration]            state refreshed, or -1
        reservoir_frame[iteration]            prepared frame used, or -1
        reservoir_accepted[iteration]         1, 0, or -1 where no attempt was made
        last_iteration                        the last FULLY committed row
        frame_iteration[frame]                which iteration a stored frame belongs to
        positions[frame, walker, atom, spatial]
        box[frame, walker, cell, spatial]     absent under implicit solvent

    `u_evaluated` exists so "not computed" is distinguishable from "computed as zero". A rule that
    evaluates four numbers per pair leaves the rest unevaluated, and a validator must not read that
    as a corrupt matrix.

    `last_iteration` is written LAST, after every array for that row. A run killed mid-write leaves
    it pointing at the previous row, so the storage a resume reads is never the half-written one.
"""
import datetime
import json
import os
from pathlib import Path

import numpy as np

#: Bumped when the meaning of the schema changes. Written into the file and checked on read.
SCHEMA_VERSION = "md-templates-replica-exchange/v1"

#: The completed-run manifest format.
MANIFEST_FORMAT = "md-templates-replica-restart/v1"

#: The atomic run-state sidecar format.
RUN_STATE_FORMAT = "md-templates-replica-runstate/v1"

#: The exact line a completed run prints, and nothing else on that line.
COMPLETION_MARKER = "run_status: completed"


class StorageError(RuntimeError):
    """The storage cannot be opened, written or continued."""


def run_state_path(storage):
    """The sidecar beside the analysis NetCDF: `rest2.nc` -> `rest2.runstate.json`.

    Derived from the STORAGE rather than from `-r`, because the storage is what a resume is given
    and what it is authoritative about. A run interrupted before any manifest exists is still
    findable from `-x` alone.
    """
    storage = Path(storage)
    return storage.with_name(storage.stem + ".runstate.json")


def write_atomic(path, text):
    """Write through a temporary in the same directory, then rename.

    No record this module writes is ever observable in a partial state: a half-written manifest
    after a crash would be a claim about a run that cannot be checked.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_run_state(storage, status, **extra):
    """Atomically record what this run is doing. Never claims completion early."""
    record = {"format": RUN_STATE_FORMAT, "status": status,
              "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "pid": os.getpid(), "storage": Path(storage).name}
    record.update(extra)
    write_atomic(run_state_path(storage), json.dumps(record, indent=2) + "\n")
    return record


def read_run_state(storage):
    path = run_state_path(storage)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class ReplicaReporter:
    """Writes and reads the analysis file. The only thing that knows the schema."""

    def __init__(self, path, *, mode="r"):
        import netCDF4

        self.path = Path(path)
        self._netCDF4 = netCDF4
        try:
            self.dataset = netCDF4.Dataset(str(self.path), mode)
        except Exception as failure:
            raise StorageError(
                f"{self.path} could not be opened as NetCDF ({type(failure).__name__}: "
                f"{failure}). A truncated or corrupt file fails here.") from None

    # -- creation ---------------------------------------------------------------------------------

    @classmethod
    def create(cls, path, *, n_states, n_atoms, has_box, identity, metadata):
        """Create the file and write everything that is fixed for the whole run.

        The scientific identity goes in HERE, before a single step is propagated, so an interrupted
        run carries in its own storage everything a continuation must be checked against.
        """
        import netCDF4

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = netCDF4.Dataset(str(path), "w")
        dataset.schema = SCHEMA_VERSION
        dataset.created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset.identity_json = json.dumps(identity, sort_keys=True)
        dataset.metadata_json = json.dumps(metadata, sort_keys=True)

        dataset.createDimension("iteration", None)
        dataset.createDimension("frame", None)
        dataset.createDimension("state", n_states)
        dataset.createDimension("walker", n_states)
        dataset.createDimension("atom", n_atoms)
        dataset.createDimension("spatial", 3)
        dataset.createDimension("cell", 3)

        tau = dataset.createVariable("tau", "f8", ("state",))
        tau.long_name = "tau[state] is the REST2 source parameter of that thermodynamic state"
        dataset.createVariable("segment", "i8", ("iteration",))
        dataset.createVariable("time_ps", "f8", ("iteration",))
        dataset.createVariable("is_exchange", "i1", ("iteration",))
        mapping = dataset.createVariable("state_to_walker", "i4", ("iteration", "state"))
        mapping.long_name = ("state_to_walker[iteration][state] is the walker whose configuration "
                             "occupied that state; every row is a permutation")
        dataset.createVariable("proposed", "i8", ("iteration", "state", "state"))
        dataset.createVariable("accepted", "i8", ("iteration", "state", "state"))
        u = dataset.createVariable("u", "f8", ("iteration", "state", "state"))
        u.long_name = ("u[iteration][i][j] is the reduced potential of the configuration in state "
                       "j evaluated in the Hamiltonian of state i, where it was evaluated")
        evaluated = dataset.createVariable("u_evaluated", "i1", ("iteration", "state", "state"))
        evaluated.long_name = ("1 where u was actually computed; a rule evaluating four numbers "
                               "per pair leaves the rest unevaluated, which is not corruption")
        dataset.createVariable("reservoir_state", "i4", ("iteration",))
        dataset.createVariable("reservoir_frame", "i4", ("iteration",))
        dataset.createVariable("reservoir_accepted", "i1", ("iteration",))
        last = dataset.createVariable("last_iteration", "i8")
        last.long_name = ("the last FULLY committed iteration; written after every array for that "
                          "row, so an interrupted write leaves it on the previous row")
        last[0] = -1
        dataset.createVariable("frame_iteration", "i8", ("frame",))
        dataset.createVariable("positions", "f4", ("frame", "walker", "atom", "spatial"),
                               zlib=False)
        if has_box:
            dataset.createVariable("box", "f8", ("frame", "walker", "cell", "spatial"))
        dataset.sync()
        dataset.close()
        return cls(path, mode="a")

    # -- writing ------------------------------------------------------------------------------------

    def write_ladder(self, tau):
        self.dataset.variables["tau"][:] = np.asarray(tau, dtype=float)
        self.dataset.sync()

    def write_iteration(self, iteration, *, segment, time_ps, is_exchange, state_to_walker,
                        proposed, accepted, u, u_evaluated, reservoir=None):
        """One committed row. `last_iteration` is written last, deliberately."""
        variables = self.dataset.variables
        variables["segment"][iteration] = int(segment)
        variables["time_ps"][iteration] = float(time_ps)
        variables["is_exchange"][iteration] = 1 if is_exchange else 0
        variables["state_to_walker"][iteration, :] = np.asarray(state_to_walker, dtype=np.int32)
        variables["proposed"][iteration, :, :] = np.asarray(proposed, dtype=np.int64)
        variables["accepted"][iteration, :, :] = np.asarray(accepted, dtype=np.int64)
        variables["u"][iteration, :, :] = np.asarray(u, dtype=float)
        variables["u_evaluated"][iteration, :, :] = np.asarray(u_evaluated, dtype=np.int8)
        state, frame, outcome = (-1, -1, -1) if reservoir is None else reservoir
        variables["reservoir_state"][iteration] = int(state)
        variables["reservoir_frame"][iteration] = int(frame)
        variables["reservoir_accepted"][iteration] = int(outcome)
        self.dataset.sync()
        variables["last_iteration"][0] = int(iteration)
        self.dataset.sync()

    def write_frame(self, iteration, configurations):
        """One stored coordinate frame: every walker's positions, and its box where there is one."""
        variables = self.dataset.variables
        index = int(self.dataset.dimensions["frame"].size)
        variables["frame_iteration"][index] = int(iteration)
        positions = np.array([c.positions for c in configurations], dtype=np.float32)
        variables["positions"][index, :, :, :] = positions
        if "box" in variables:
            boxes = np.array([np.zeros((3, 3)) if c.box is None else c.box
                              for c in configurations], dtype=float)
            variables["box"][index, :, :, :] = boxes
        self.dataset.sync()
        return index

    # -- reading -------------------------------------------------------------------------------------

    @property
    def schema(self):
        return getattr(self.dataset, "schema", None)

    @property
    def identity(self):
        raw = getattr(self.dataset, "identity_json", None)
        return None if raw is None else json.loads(raw)

    @property
    def metadata(self):
        raw = getattr(self.dataset, "metadata_json", None)
        return None if raw is None else json.loads(raw)

    def last_iteration(self):
        return int(self.dataset.variables["last_iteration"][0])

    def rewind(self, iteration):
        """Disown every committed row after `iteration`.

        A resume continues from the last CHECKPOINT, because that is the newest point with
        configurations to continue from. Rows committed after it describe iterations whose
        configurations were lost with the process, so leaving `last_iteration` past the checkpoint
        would leave the storage claiming history the run is about to produce differently.

        The rows are not deleted -- NetCDF has no cheap truncation and they are about to be
        overwritten anyway -- but they stop being committed, and every reader honours
        `last_iteration`.
        """
        current = self.last_iteration()
        if int(iteration) > current:
            raise StorageError(
                f"cannot rewind to iteration {iteration}: only {current} is committed")
        self.dataset.variables["last_iteration"][0] = int(iteration)
        self.dataset.sync()
        return current

    def n_states(self):
        return int(self.dataset.dimensions["state"].size)

    def tau(self):
        return np.array(self.dataset.variables["tau"][:], dtype=float)

    def mapping(self, upto=None):
        last = self.last_iteration() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0, self.n_states()), dtype=int)
        return np.array(self.dataset.variables["state_to_walker"][:last + 1, :], dtype=int)

    def statistics(self, upto=None):
        """Every proposal and acceptance row up to and including the last committed iteration."""
        last = self.last_iteration() if upto is None else int(upto)
        if last < 0:
            n = self.n_states()
            return np.zeros((0, n, n), dtype=np.int64), np.zeros((0, n, n), dtype=np.int64)
        proposed = np.array(self.dataset.variables["proposed"][:last + 1, :, :], dtype=np.int64)
        accepted = np.array(self.dataset.variables["accepted"][:last + 1, :, :], dtype=np.int64)
        return accepted, proposed

    def reduced_potentials(self, iteration):
        u = np.array(self.dataset.variables["u"][int(iteration), :, :], dtype=float)
        mask = np.array(self.dataset.variables["u_evaluated"][int(iteration), :, :], dtype=np.int8)
        return u, mask

    def reservoir_events(self, upto=None):
        last = self.last_iteration() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0, 3), dtype=int)
        state = np.array(self.dataset.variables["reservoir_state"][:last + 1], dtype=int)
        frame = np.array(self.dataset.variables["reservoir_frame"][:last + 1], dtype=int)
        outcome = np.array(self.dataset.variables["reservoir_accepted"][:last + 1], dtype=int)
        return np.stack([state, frame, outcome], axis=1)

    def times(self, upto=None):
        last = self.last_iteration() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0,), dtype=float)
        return np.array(self.dataset.variables["time_ps"][:last + 1], dtype=float)

    def frames(self):
        return int(self.dataset.dimensions["frame"].size)

    def close(self):
        try:
            self.dataset.close()
        except Exception:
            pass


class ReplicaCheckpoint:
    """The last committed configurations, and everything else a continuation needs.

    Written whole and replaced atomically, so a checkpoint is never half a state. It carries the
    complete configuration of every walker plus the mapping, every RNG state, the rule's own
    persistent state, the iteration reached and the original budget.

    OpenMM's own context checkpoints are deliberately NOT stored: they are platform-specific and
    would make a checkpoint unusable on a different device. Positions, velocities and box vectors
    are the complete logical state of a Langevin walker, and continuing from them is exact up to
    the platform's own arithmetic -- which is stated rather than implied.
    """

    def __init__(self, path):
        self.path = Path(path)

    def write(self, *, iteration, segment, configurations, state_to_walker, rng_states,
              rule_state, budget_segments, identity, extra=None):
        import netCDF4

        temporary = self.path.with_name(f"{self.path.name}.partial.{os.getpid()}")
        n_walkers = len(configurations)
        n_atoms = configurations[0].n_atoms
        has_box = configurations[0].box is not None
        dataset = netCDF4.Dataset(str(temporary), "w")
        try:
            dataset.schema = SCHEMA_VERSION
            dataset.kind = "checkpoint"
            dataset.written_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            dataset.iteration = int(iteration)
            dataset.segment = int(segment)
            dataset.budget_segments = int(budget_segments)
            dataset.identity_json = json.dumps(identity, sort_keys=True)
            dataset.rng_states_json = json.dumps(rng_states, sort_keys=True)
            dataset.rule_state_json = json.dumps(rule_state or {}, sort_keys=True)
            dataset.extra_json = json.dumps(extra or {}, sort_keys=True)
            dataset.createDimension("walker", n_walkers)
            dataset.createDimension("atom", n_atoms)
            dataset.createDimension("spatial", 3)
            dataset.createDimension("cell", 3)
            dataset.createVariable("state_to_walker", "i4", ("walker",))[:] = \
                np.asarray(state_to_walker, dtype=np.int32)
            dataset.createVariable("positions", "f8", ("walker", "atom", "spatial"))[:] = \
                np.array([c.positions for c in configurations], dtype=float)
            dataset.createVariable("velocities", "f8", ("walker", "atom", "spatial"))[:] = \
                np.array([c.velocities for c in configurations], dtype=float)
            if has_box:
                dataset.createVariable("box", "f8", ("walker", "cell", "spatial"))[:] = \
                    np.array([c.box for c in configurations], dtype=float)
            dataset.sync()
        finally:
            dataset.close()
        os.replace(temporary, self.path)

    def read(self):
        import netCDF4

        from replica_engine import Configuration

        if not self.path.is_file():
            raise StorageError(f"{self.path} does not exist; there is no checkpoint to continue")
        try:
            dataset = netCDF4.Dataset(str(self.path), "r")
        except Exception as failure:
            raise StorageError(
                f"{self.path} could not be opened as NetCDF ({type(failure).__name__}: "
                f"{failure}). A truncated checkpoint fails here.") from None
        try:
            positions = np.array(dataset.variables["positions"][:], dtype=float)
            velocities = np.array(dataset.variables["velocities"][:], dtype=float)
            boxes = (np.array(dataset.variables["box"][:], dtype=float)
                     if "box" in dataset.variables else None)
            configurations = [
                Configuration(positions[i], velocities[i],
                              None if boxes is None else boxes[i])
                for i in range(positions.shape[0])]
            return {
                "iteration": int(dataset.iteration),
                "segment": int(dataset.segment),
                "budget_segments": int(dataset.budget_segments),
                "state_to_walker": [int(x) for x in dataset.variables["state_to_walker"][:]],
                "configurations": configurations,
                "rng_states": json.loads(dataset.rng_states_json),
                "rule_state": json.loads(dataset.rule_state_json),
                "identity": json.loads(dataset.identity_json),
                "extra": json.loads(getattr(dataset, "extra_json", "{}")),
            }
        finally:
            dataset.close()
