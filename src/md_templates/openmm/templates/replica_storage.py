#!/usr/bin/env python
"""The replica-exchange storage this repository OWNS. One documented NetCDF schema.

Copied verbatim into every generated replica project. Depends on `netCDF4` and nothing else.

FOUR STREAMS, FOUR SCHEDULES
    The previous schema had one row per propagation segment and wrote coordinates only on the
    whole-output stride, so the "frequent solute stream" it documented did not exist. The schedules
    are now genuinely independent and each stream carries its own ABSOLUTE STEP:

        exchange[e]        one row per exchange attempt: mapping, proposals, acceptances, the
                           reduced potentials the decision used, and any reservoir event
        frame[f]           complete coordinates for every walker
        solute_frame[s]    the solute subset only, typically far more often
        (checkpoint)       a separate file; see ReplicaCheckpoint

    Nothing is inferred from anything else. A frame records the exchange index in force when it was
    written, so the mapping is reconstructable at every observation without assuming a cadence.

INDEXING
    Coordinate arrays are WALKER-indexed: `positions[frame, walker, atom, spatial]`. A walker is a
    continuous trajectory; a thermodynamic state is a Hamiltonian that walkers pass through. The
    mapping `state_to_walker[exchange, state]` converts, and `replica_statistics.walker_view`
    inverts it. Both directions are tested against each other.

COMMIT MARKERS
    Each stream has a `last_*` marker written only AFTER every array for that record. A file
    truncated by a crash reads back as exactly what was fully committed, and a resume can rewind to
    the last checkpoint without inventing history.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

#: Bumped when the meaning of the schema changes. Written into the file and checked on read.
SCHEMA_VERSION = "md-templates-replica-exchange/v2"

#: Fields on the `exchange` dimension that were added to v2 AFTER v2 was first published.
#:
#: A file written by an earlier build of this same schema legitimately lacks them, and that is not
#: corruption: v2's meaning did not change, it gained a record. Readers therefore tolerate absence
#: and report `absent_value`, and rank 0 adds the field once when it opens such a file for append.
#: Bumping to v3 would have been the heavier answer to a strictly additive change, and would have
#: made every existing v2 file unreadable by a runtime that can in fact read it.
#:
#: `absent_value` is what the field MEANS for rows written before it existed -- not a placeholder.
#: Old v2 files predate working Maxwell refreshes, and a stored-velocity refresh draws no momenta,
#: so -1 is the true statement "no velocity seed was used for this exchange row".
OPTIONAL_EXCHANGE_FIELDS = {
    "reservoir_velocity_seed": {
        "dtype": "i8",
        "dimensions": ("exchange",),
        "absent_value": -1,
        "added_in": "the velocity-policy provenance work",
        "meaning": ("the seed the momenta were drawn from under `velocity_policy: maxwell`; "
                    "-1 where no refresh happened and where the policy was `stored`, which draws "
                    "nothing"),
    },
}

#: Where the cumulative migration history lives INSIDE the analysis file.
#:
#: The file is the durable authority for its own schema history, and it has to be: the event and
#: the mutation it describes are committed in one `sync()`, so a process killed immediately after
#: migrating still leaves a file that says it was migrated. Recording the event only in the run
#: state or the manifest would leave a window in which storage had been changed and nothing said
#: so, and the next continuation would have no way to know.
MIGRATION_HISTORY_ATTRIBUTE = "storage_migrations_json"

#: The key an event is identified by. Two records of the SAME event -- one copied into a
#: completion manifest, one into a run-state sidecar -- carry the same id and collapse to one.
#: Two genuinely different migrations differ in their content and stay separate.
MIGRATION_EVENT_ID = "event_id"


def migration_event_id(event):
    """A deterministic identity for one migration event.

    Over the canonical JSON of everything except the id itself, so an event copied between records
    keeps its identity and two distinct migrations keep theirs.
    """
    payload = {key: value for key, value in dict(event).items() if key != MIGRATION_EVENT_ID}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def normalise_migration_event(event):
    """One event in canonical form, or None if it records no change to storage.

    A migration that added nothing is not history. Keeping such an event would let a later no-op
    continuation look exactly like the real one that preceded it, which is how the singular field
    lost the real event in the first place.
    """
    if not isinstance(event, dict):
        return None
    if not event.get("fields_added"):
        return None
    canonical = dict(event)
    canonical[MIGRATION_EVENT_ID] = canonical.get(MIGRATION_EVENT_ID) or migration_event_id(event)
    return canonical


def migration_history_of(record):
    """The history carried by any record, accepting the legacy singular field.

    Older records wrote `storage_migration:` as a single value. A meaningful one becomes a
    one-event history; a no-op one is dropped, because it never described a change.
    """
    if not isinstance(record, dict):
        return []
    history = record.get("storage_migrations")
    if history is None:
        singular = record.get("storage_migration")
        history = [singular] if singular is not None else []
    if isinstance(history, dict):
        history = [history]
    return [event for event in (normalise_migration_event(e) for e in history or []) if event]


def merge_migration_histories(*histories):
    """Every distinct event, in first-seen order.

    Order is preserved rather than sorted: it is the order the migrations happened in, and no
    timestamp in these records is guaranteed to be comparable across machines.
    """
    merged, seen = [], set()
    for history in histories:
        for event in history or []:
            canonical = normalise_migration_event(event)
            if canonical is None:
                continue
            identity = canonical[MIGRATION_EVENT_ID]
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(canonical)
    return merged


#: The completed-run manifest format.
MANIFEST_FORMAT = "md-templates-replica-restart/v2"

#: The atomic run-state sidecar format.
RUN_STATE_FORMAT = "md-templates-replica-runstate/v2"

#: The exact line a completed run prints, and nothing else on that line.
COMPLETION_MARKER = "run_status: completed"


class StorageError(RuntimeError):
    """The storage cannot be opened, written or continued."""


def run_state_path(storage):
    """The sidecar beside the analysis NetCDF: `rest2.nc` -> `rest2.runstate.json`."""
    storage = Path(storage)
    return storage.with_name(storage.stem + ".runstate.json")


def solute_path(storage):
    """The solute stream beside the analysis NetCDF: `rest2.nc` -> `rest2.solute.nc`."""
    storage = Path(storage)
    return storage.with_name(storage.stem + ".solute.nc")


def write_atomic(path, text):
    """Write through a temporary in the same directory, then rename."""
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
    write_atomic(run_state_path(storage), json.dumps(record, indent=2, default=str) + "\n")
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
    """Writes and reads the analysis file and the solute stream. The only thing knowing the schema.

    Opened for writing ONLY on rank 0. Every other rank that needs to read does so in mode "r".
    """

    def __init__(self, path, *, mode="r"):
        import netCDF4

        self.path = Path(path)
        self.mode = mode
        try:
            self.dataset = netCDF4.Dataset(str(self.path), mode)
        except Exception as failure:
            raise StorageError(
                f"{self.path} could not be opened as NetCDF ({type(failure).__name__}: "
                f"{failure}). A truncated or corrupt file fails here.") from None
        self._solute = None

    # -- creation ---------------------------------------------------------------------------------

    @classmethod
    def create(cls, path, *, n_states, n_atoms, n_solute_atoms, has_box, identity, metadata):
        """Create the file and write everything fixed for the whole run.

        The scientific identity goes in HERE, before a single step is propagated, so an interrupted
        run carries in its own storage everything a continuation must be checked against.
        """
        import netCDF4

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = netCDF4.Dataset(str(path), "w")
        dataset.schema = SCHEMA_VERSION
        dataset.created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset.identity_json = json.dumps(identity, sort_keys=True, default=str)
        dataset.metadata_json = json.dumps(metadata, sort_keys=True, default=str)
        dataset.coordinate_indexing = "walker"
        dataset.position_unit = "nanometer"

        dataset.createDimension("exchange", None)
        dataset.createDimension("frame", None)
        dataset.createDimension("state", n_states)
        dataset.createDimension("walker", n_states)
        dataset.createDimension("atom", n_atoms)
        dataset.createDimension("spatial", 3)
        dataset.createDimension("cell", 3)

        tau = dataset.createVariable("tau", "f8", ("state",))
        tau.long_name = "tau[state] is the REST2 source parameter of that thermodynamic state"

        # -- the exchange stream ------------------------------------------------------------------
        step = dataset.createVariable("exchange_step", "i8", ("exchange",))
        step.long_name = "absolute integration step at which this exchange was attempted"
        dataset.createVariable("exchange_time_ps", "f8", ("exchange",))
        mapping = dataset.createVariable("state_to_walker", "i4", ("exchange", "state"))
        mapping.long_name = ("state_to_walker[exchange][state] is the walker occupying that state "
                             "AFTER this exchange; every row is a permutation")
        dataset.createVariable("proposed", "i8", ("exchange", "state", "state"))
        dataset.createVariable("accepted", "i8", ("exchange", "state", "state"))
        u = dataset.createVariable("u", "f8", ("exchange", "state", "state"))
        u.long_name = ("u[exchange][i][j] is the reduced potential of walker j's sample evaluated "
                       "in the Hamiltonian of state i")
        evaluated = dataset.createVariable("u_evaluated", "i1", ("exchange", "state", "state"))
        evaluated.long_name = "1 where u was actually computed"
        dataset.createVariable("reservoir_state", "i4", ("exchange",))
        dataset.createVariable("reservoir_frame", "i4", ("exchange",))
        dataset.createVariable("reservoir_source_step", "i8", ("exchange",))
        dataset.createVariable("reservoir_accepted", "i1", ("exchange",))
        # The optional v2 fields. A NEW file always gets them here, so only a file written before
        # they existed ever needs the migration in `ensure_optional_exchange_fields()`.
        for name, spec in OPTIONAL_EXCHANGE_FIELDS.items():
            dataset.createVariable(name, spec["dtype"], spec["dimensions"])
        last_exchange = dataset.createVariable("last_exchange", "i8")
        last_exchange.long_name = "the last FULLY committed exchange row"
        last_exchange[0] = -1

        # -- the whole-system stream --------------------------------------------------------------
        frame_step = dataset.createVariable("frame_step", "i8", ("frame",))
        frame_step.long_name = "absolute integration step at which this frame was stored"
        dataset.createVariable("frame_time_ps", "f8", ("frame",))
        frame_exchange = dataset.createVariable("frame_exchange", "i8", ("frame",))
        frame_exchange.long_name = ("the exchange index in force when this frame was written, so "
                                    "the mapping is reconstructable without assuming a cadence")
        positions = dataset.createVariable("positions", "f4",
                                           ("frame", "walker", "atom", "spatial"))
        positions.units = "nanometer"
        positions.long_name = "positions[frame][walker][atom][xyz]; WALKER-indexed"
        if has_box:
            box = dataset.createVariable("box", "f8", ("frame", "walker", "cell", "spatial"))
            box.units = "nanometer"
        last_frame = dataset.createVariable("last_frame", "i8")
        last_frame.long_name = "the last FULLY committed whole-system frame"
        last_frame[0] = -1

        dataset.sync()
        dataset.close()
        reporter = cls(path, mode="a")
        reporter._create_solute(n_states=n_states, n_solute_atoms=n_solute_atoms,
                               identity=identity)
        return reporter

    def _create_solute(self, *, n_states, n_solute_atoms, identity):
        """The solute stream is its own file: it is written far more often and read separately."""
        import netCDF4

        path = solute_path(self.path)
        dataset = netCDF4.Dataset(str(path), "w")
        dataset.schema = SCHEMA_VERSION
        dataset.kind = "solute"
        dataset.identity_json = json.dumps(identity, sort_keys=True, default=str)
        dataset.coordinate_indexing = "walker"
        dataset.position_unit = "nanometer"
        dataset.createDimension("solute_frame", None)
        dataset.createDimension("walker", n_states)
        dataset.createDimension("solute_atom", int(n_solute_atoms))
        dataset.createDimension("spatial", 3)
        dataset.createVariable("solute_step", "i8", ("solute_frame",))
        dataset.createVariable("solute_time_ps", "f8", ("solute_frame",))
        dataset.createVariable("solute_exchange", "i8", ("solute_frame",))
        positions = dataset.createVariable(
            "solute_positions", "f4", ("solute_frame", "walker", "solute_atom", "spatial"))
        positions.units = "nanometer"
        last = dataset.createVariable("last_solute_frame", "i8")
        last[0] = -1
        dataset.sync()
        dataset.close()
        self._solute = netCDF4.Dataset(str(path), "a")

    def _open_solute(self, mode="r"):
        import netCDF4

        if self._solute is not None:
            return self._solute
        path = solute_path(self.path)
        if not path.is_file():
            return None
        try:
            self._solute = netCDF4.Dataset(str(path), mode)
        except Exception as failure:
            raise StorageError(
                f"{path} could not be opened as NetCDF ({type(failure).__name__}: {failure})"
            ) from None
        return self._solute

    # -- writing ------------------------------------------------------------------------------------

    def write_ladder(self, tau):
        self.dataset.variables["tau"][:] = np.asarray(tau, dtype=float)
        self.dataset.sync()

    def write_exchange(self, index, *, step, time_ps, state_to_walker, proposed, accepted, u,
                       u_evaluated, reservoir=None):
        """One committed exchange row. `last_exchange` is written last, deliberately."""
        variables = self.dataset.variables
        variables["exchange_step"][index] = int(step)
        variables["exchange_time_ps"][index] = float(time_ps)
        variables["state_to_walker"][index, :] = np.asarray(state_to_walker, dtype=np.int32)
        variables["proposed"][index, :, :] = np.asarray(proposed, dtype=np.int64)
        variables["accepted"][index, :, :] = np.asarray(accepted, dtype=np.int64)
        variables["u"][index, :, :] = np.asarray(u, dtype=float)
        variables["u_evaluated"][index, :, :] = np.asarray(u_evaluated, dtype=np.int8)
        state, frame, source_step, outcome, velocity_seed = (
            (-1, -1, -1, -1, -1) if reservoir is None else reservoir)
        variables["reservoir_state"][index] = int(state)
        variables["reservoir_frame"][index] = int(frame)
        variables["reservoir_source_step"][index] = int(source_step)
        variables["reservoir_accepted"][index] = int(outcome)
        if "reservoir_velocity_seed" not in variables:
            raise StorageError(
                f"{self.path} has no `reservoir_velocity_seed` variable. It is an optional v2 "
                f"field that rank 0 adds once, before propagation, when it opens an older file "
                f"for append. Reaching a row write without it means the continuation path skipped "
                f"that step -- refusing here rather than dropping the record for this exchange.")
        variables["reservoir_velocity_seed"][index] = int(velocity_seed)
        self.dataset.sync()
        variables["last_exchange"][0] = int(index)
        self.dataset.sync()

    def write_frame(self, *, step, time_ps, exchange_index, configurations):
        """One whole-system frame: every walker's positions, and its box where there is one."""
        variables = self.dataset.variables
        index = int(variables["last_frame"][0]) + 1
        variables["frame_step"][index] = int(step)
        variables["frame_time_ps"][index] = float(time_ps)
        variables["frame_exchange"][index] = int(exchange_index)
        variables["positions"][index, :, :, :] = np.array(
            [c.positions for c in configurations], dtype=np.float32)
        if "box" in variables:
            variables["box"][index, :, :, :] = np.array(
                [np.zeros((3, 3)) if c.box is None else c.box for c in configurations],
                dtype=float)
        self.dataset.sync()
        variables["last_frame"][0] = index
        self.dataset.sync()
        return index

    def write_solute_frame(self, *, step, time_ps, exchange_index, configurations,
                           solute_indices):
        """One solute frame. Its own file, its own schedule, its own commit marker."""
        dataset = self._open_solute("a")
        if dataset is None:
            raise StorageError("the solute stream was never created")
        variables = dataset.variables
        index = int(variables["last_solute_frame"][0]) + 1
        variables["solute_step"][index] = int(step)
        variables["solute_time_ps"][index] = float(time_ps)
        variables["solute_exchange"][index] = int(exchange_index)
        subset = np.array([c.positions[solute_indices] for c in configurations],
                          dtype=np.float32)
        variables["solute_positions"][index, :, :, :] = subset
        dataset.sync()
        variables["last_solute_frame"][0] = index
        dataset.sync()
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

    def last_exchange(self):
        return int(self.dataset.variables["last_exchange"][0])

    def last_frame(self):
        return int(self.dataset.variables["last_frame"][0])

    def last_solute_frame(self):
        dataset = self._open_solute("r")
        if dataset is None:
            return -1
        return int(dataset.variables["last_solute_frame"][0])

    def n_states(self):
        return int(self.dataset.dimensions["state"].size)

    # How many rows each stream ACTUALLY holds, as opposed to how many its completion marker
    # claims. The two are compared when an output is validated: a marker ahead of the data is a
    # file that disagrees with itself.
    def n_exchange_rows(self):
        return int(self.dataset.dimensions["exchange"].size)

    def n_whole_frames(self):
        return int(self.dataset.dimensions["frame"].size)

    def n_solute_frames(self):
        dataset = self._open_solute("r")
        if dataset is None:
            return 0
        return int(dataset.dimensions["solute_frame"].size)

    def tau(self):
        return np.array(self.dataset.variables["tau"][:], dtype=float)

    def mapping(self, upto=None):
        last = self.last_exchange() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0, self.n_states()), dtype=int)
        return np.array(self.dataset.variables["state_to_walker"][:last + 1, :], dtype=int)

    def statistics(self, upto=None):
        last = self.last_exchange() if upto is None else int(upto)
        n = self.n_states()
        if last < 0:
            return np.zeros((0, n, n), dtype=np.int64), np.zeros((0, n, n), dtype=np.int64)
        return (np.array(self.dataset.variables["accepted"][:last + 1], dtype=np.int64),
                np.array(self.dataset.variables["proposed"][:last + 1], dtype=np.int64))

    def reduced_potentials(self, index):
        return (np.array(self.dataset.variables["u"][int(index)], dtype=float),
                np.array(self.dataset.variables["u_evaluated"][int(index)], dtype=np.int8))

    def reservoir_events(self, upto=None):
        last = self.last_exchange() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0, 4), dtype=int)
        variables = self.dataset.variables
        return np.stack([
            np.array(variables["reservoir_state"][:last + 1], dtype=int),
            np.array(variables["reservoir_frame"][:last + 1], dtype=int),
            np.array(variables["reservoir_source_step"][:last + 1], dtype=int),
            np.array(variables["reservoir_accepted"][:last + 1], dtype=int)], axis=1)

    # -- optional v2 fields --------------------------------------------------------------------

    def inspect_optional_exchange_fields(self):
        """READ-ONLY. What each optional field is: present and usable, absent, or malformed.

        Separated from the migration on purpose. A continuation has to know what it is about to do
        while it is still only reading, so that every other continuation check can refuse first and
        leave the file untouched.
        """
        report = {}
        for name, spec in OPTIONAL_EXCHANGE_FIELDS.items():
            variable = self.dataset.variables.get(name)
            if variable is None:
                report[name] = {"state": "absent", "problem": None}
                continue
            problem = None
            if tuple(variable.dimensions) != tuple(spec["dimensions"]):
                problem = (
                    f"{name} has dimensions {tuple(variable.dimensions)}, expected "
                    f"{tuple(spec['dimensions'])}")
            elif np.dtype(variable.dtype).kind not in "iu":
                problem = (
                    f"{name} has dtype {np.dtype(variable.dtype)!s}, which cannot hold the "
                    f"recorded integer seeds")
            report[name] = {"state": "present", "problem": problem}
        return report

    def ensure_optional_exchange_fields(self):
        """Add any optional v2 field this file predates, and say what was done.

        Rank 0 only, on a file already opened for append, and only AFTER every continuation check
        has passed -- a file must never be modified merely because it could be opened. Existing
        rows are initialised to the field's `absent_value`, which is its true meaning for a row
        written before the field existed rather than a filler.

        An existing field is validated and never deleted or replaced: a definition this runtime
        cannot use is a refusal, because silently rewriting one would destroy a record whose
        provenance we do not know.
        """
        if self.mode not in ("a", "r+", "w"):
            raise StorageError(
                f"{self.path} is open in mode {self.mode!r}; the optional-field migration writes "
                f"and must be given a file opened for append. Nothing was changed.")

        inspection = self.inspect_optional_exchange_fields()
        malformed = [entry["problem"] for entry in inspection.values() if entry["problem"]]
        if malformed:
            raise StorageError(
                "this file cannot be continued because an optional field is defined in a way this "
                "runtime cannot use:\n  - " + "\n  - ".join(malformed) +
                "\n  It is left exactly as it is. Refusing rather than deleting or redefining a "
                "variable whose provenance is unknown.")

        rows = self.n_exchange_rows()
        added, present = [], []
        for name, entry in inspection.items():
            if entry["state"] == "present":
                present.append(name)
                continue
            spec = OPTIONAL_EXCHANGE_FIELDS[name]
            variable = self.dataset.createVariable(name, spec["dtype"], spec["dimensions"])
            if rows:
                # Every row that already exists gets the value the field MEANS for it. The
                # dimension is unlimited, so this both fills and sizes the new variable.
                variable[:rows] = np.full(rows, spec["absent_value"], dtype=np.int64)
            added.append(name)

        event = {
            "schema": SCHEMA_VERSION,
            "fields_added": added,
            "fields_already_present": present,
            "rows_initialised": int(rows) if added else 0,
            "initialised_to": {name: OPTIONAL_EXCHANGE_FIELDS[name]["absent_value"]
                               for name in added},
            "previous_committed_exchanges": int(self.last_exchange()) + 1,
            "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "note": ("an older v2 file predates this field; the rows it already held record that "
                     "no velocity seed was used, which is what they mean"),
        }
        if added:
            # The event is written into the SAME file the mutation changed, and both are committed
            # by the sync below. A process killed here leaves a file that already says it was
            # migrated, so the next continuation reads the truth rather than having to infer it.
            event[MIGRATION_EVENT_ID] = migration_event_id(event)
            history = merge_migration_histories(self.migration_history(), [event])
            self.dataset.setncattr(MIGRATION_HISTORY_ATTRIBUTE,
                                   json.dumps(history, sort_keys=True, default=str))
        self.dataset.sync()
        return event

    def migration_history(self):
        """The cumulative migration history this FILE carries. Read-only, and never fabricated."""
        raw = getattr(self.dataset, MIGRATION_HISTORY_ATTRIBUTE, None)
        if raw is None:
            return []
        try:
            recorded = json.loads(raw)
        except ValueError:
            raise StorageError(
                f"{self.path} carries a {MIGRATION_HISTORY_ATTRIBUTE} attribute that is not "
                f"valid JSON. Refusing to guess what happened to this file.") from None
        return merge_migration_histories(recorded)

    def reservoir_velocity_seeds(self, upto=None):
        """The Maxwell seed per exchange row; -1 where nothing was drawn.

        Kept out of `reservoir_events` on purpose: that array is the statistics contract, and
        widening it would change what every consumer indexes.
        """
        last = self.last_exchange() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0,), dtype=int)
        if "reservoir_velocity_seed" not in self.dataset.variables:
            return np.full(last + 1, -1, dtype=int)
        return np.array(self.dataset.variables["reservoir_velocity_seed"][:last + 1], dtype=int)

    def exchange_steps(self, upto=None):
        last = self.last_exchange() if upto is None else int(upto)
        if last < 0:
            return np.zeros((0,), dtype=np.int64)
        return np.array(self.dataset.variables["exchange_step"][:last + 1], dtype=np.int64)

    def frame_steps(self):
        last = self.last_frame()
        if last < 0:
            return np.zeros((0,), dtype=np.int64)
        return np.array(self.dataset.variables["frame_step"][:last + 1], dtype=np.int64)

    def solute_steps(self):
        dataset = self._open_solute("r")
        last = -1 if dataset is None else int(dataset.variables["last_solute_frame"][0])
        if last < 0:
            return np.zeros((0,), dtype=np.int64)
        return np.array(dataset.variables["solute_step"][:last + 1], dtype=np.int64)

    def rewind(self, *, exchange, frame, solute_frame):
        """Disown every committed record after a checkpoint's position in each stream.

        A resume continues from the last CHECKPOINT, because that is the newest point with a
        complete state to continue from. Records committed after it describe work whose state was
        lost, so leaving the markers past the checkpoint would let the storage claim history the
        run is about to produce differently -- and would duplicate rows and frames on restart.
        """
        variables = self.dataset.variables
        if int(exchange) > self.last_exchange() or int(frame) > self.last_frame():
            raise StorageError(
                f"cannot rewind forwards: asked for exchange {exchange}/frame {frame} but only "
                f"{self.last_exchange()}/{self.last_frame()} are committed")
        variables["last_exchange"][0] = int(exchange)
        variables["last_frame"][0] = int(frame)
        self.dataset.sync()
        dataset = self._open_solute("a")
        if dataset is not None:
            dataset.variables["last_solute_frame"][0] = int(solute_frame)
            dataset.sync()

    def close(self):
        for dataset in (self._solute, self.dataset):
            try:
                if dataset is not None:
                    dataset.close()
            except Exception:
                pass
        self._solute = None


class ReplicaCheckpoint:
    """The last committed state, and everything else a continuation needs.

    Written whole and replaced atomically, so a checkpoint is never half a state. It carries every
    walker's complete phase-space sample, the mapping, the RNG states, the rule's persisted state,
    the absolute step, the event counters and the budget.

    OpenMM's own context checkpoints are deliberately NOT stored: they are platform-specific and
    would make a checkpoint unusable on a different device. Positions, velocities and box are the
    complete logical state of a Langevin walker.
    """

    def __init__(self, path):
        self.path = Path(path)

    def write(self, *, step, exchange_index, frame_index, solute_frame_index, configurations,
              state_to_walker, rng_states, rule_state, schedule, identity, extra=None):
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
            dataset.step = int(step)
            dataset.exchange_index = int(exchange_index)
            dataset.frame_index = int(frame_index)
            dataset.solute_frame_index = int(solute_frame_index)
            dataset.identity_json = json.dumps(identity, sort_keys=True, default=str)
            dataset.rng_states_json = json.dumps(rng_states, sort_keys=True, default=str)
            dataset.rule_state_json = json.dumps(rule_state or {}, sort_keys=True, default=str)
            dataset.schedule_json = json.dumps(schedule, sort_keys=True, default=str)
            dataset.extra_json = json.dumps(extra or {}, sort_keys=True, default=str)
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
                Configuration(positions[i], velocities[i], None if boxes is None else boxes[i])
                for i in range(positions.shape[0])]
            return {
                "step": int(dataset.step),
                "exchange_index": int(dataset.exchange_index),
                "frame_index": int(dataset.frame_index),
                "solute_frame_index": int(dataset.solute_frame_index),
                "state_to_walker": [int(x) for x in dataset.variables["state_to_walker"][:]],
                "configurations": configurations,
                "rng_states": json.loads(dataset.rng_states_json),
                "rule_state": json.loads(dataset.rule_state_json),
                "schedule": json.loads(dataset.schedule_json),
                "identity": json.loads(dataset.identity_json),
                "extra": json.loads(getattr(dataset, "extra_json", "{}")),
            }
        finally:
            dataset.close()
