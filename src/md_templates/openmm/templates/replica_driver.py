#!/usr/bin/env python
"""The replica-exchange loop: propagate to the next event, act, record. Copied into every project.

The only module that knows about all the others. It owns nothing scientific: the ladder comes from
`replica_protocol`, the Hamiltonians from `rest2_scaling`, the decisions from an exchange rule, the
schedule from `replica_schedule` and the storage schema from `replica_storage`.

EVENT-DRIVEN, ABSOLUTE STEPS
    There is no propagation "segment". The loop asks the schedule for the next step at which
    anything happens, advances exactly that many steps, and performs whatever falls there in the
    documented order: exchange, whole frame, solute frame, checkpoint. Simultaneous events happen
    once each.

MPI: ONE AUTHORITY, NO STALE BROADCAST
    Rank 0 decides. Every rank applies the same decision to the same walker-indexed state. The rank
    that owns a state installs it. Then all ranks compare a digest before continuing.

    The previous implementation broadcast rank 0's configuration list AFTER the owning rank had
    installed a reservoir sample, so whenever the top rung was not owned by rank 0 the new velocity
    was silently discarded and the run continued with the ladder's old momenta. Nothing here
    broadcasts a list that another rank has already modified.

WRITABLE STORAGE IS ROOT-OWNED
    Only rank 0 opens the analysis NetCDF for writing, on every path -- new, resume and extend. The
    previous `_continue` opened it in append mode on every rank and closed the non-root handles
    afterwards, which is a concurrent-writer window on a file format that does not tolerate one.
"""
import datetime
import hashlib
import json
import platform as platform_module
import signal
import socket
import sys
from pathlib import Path

import numpy as np

import hamiltonian_identity
import replica_storage as storage
from exchange_rules import (ExchangeContext, NeighbouringExchangeRule, builtin_rule_identity,
                            load_rule)
from replica_engine import (Configuration, ReplicaEngine, resolve_platform,
                            select_device_for_rank, visible_cuda_devices)


class DriverError(RuntimeError):
    """The run cannot proceed as configured."""


class IdentityError(DriverError):
    """The run being continued is not the run that was started."""


class Coordinator:
    """Rank, size, and the few collectives the driver needs. No MPI import when not under MPI."""

    def __init__(self):
        self.comm = None
        self.rank, self.size = 0, 1
        try:
            from mpi4py import MPI
        except ImportError:
            return
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            self.comm, self.rank, self.size = comm, comm.Get_rank(), comm.Get_size()

    @property
    def is_root(self):
        return self.rank == 0

    def allgather(self, value):
        return [value] if self.comm is None else self.comm.allgather(value)

    def bcast(self, value):
        return value if self.comm is None else self.comm.bcast(value, root=0)

    def barrier(self):
        if self.comm is not None:
            self.comm.barrier()

    def any_true(self, flag):
        """True on EVERY rank if it is true on any. How a termination request becomes collective."""
        if self.comm is None:
            return bool(flag)
        return bool(max(self.comm.allgather(bool(flag))))

    def agree(self, value, *, what):
        """Every rank must present the same value. Used to prove shared state is actually shared."""
        if self.comm is None:
            return
        values = self.comm.allgather(value)
        if len(set(values)) != 1:
            raise DriverError(
                f"the ranks disagree about {what}: {sorted(set(values))[:4]}. Continuing would "
                f"propagate different states under one record.")


def owned_states(protocol, coordinator):
    """Which states this process drives, under the stated policy."""
    if coordinator.size == 1:
        return list(range(protocol.n_states))
    if coordinator.size != protocol.n_states:
        raise DriverError(
            f"this ladder has {protocol.n_states} states but MPI world size is "
            f"{coordinator.size}. The supported policy is one process for the whole ladder, or "
            f"exactly one rank per state.")
    return [coordinator.rank]


def configuration_digest(configurations):
    """A compact digest of every walker's phase-space sample, for cross-rank agreement.

    Positions AND velocities AND boxes: a digest over positions alone would agree happily in
    exactly the case this exists to catch -- a reservoir refresh whose velocity was lost.
    """
    digest = hashlib.sha256()
    for configuration in configurations:
        digest.update(np.ascontiguousarray(configuration.positions, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(configuration.velocities, dtype=np.float64).tobytes())
        if configuration.box is not None:
            digest.update(np.ascontiguousarray(configuration.box, dtype=np.float64).tobytes())
    return digest.hexdigest()


class _Interruption:
    """A termination request, recorded rather than raised.

    Raising inside a signal handler under MPI leaves the other ranks blocked in a collective. The
    handler only sets a flag; the loop checks it at an event boundary, where every rank checks
    together, and they stop as one.
    """

    def __init__(self):
        self.requested = False
        self.signal = None
        self._previous = {}

    def install(self):
        def stop(signum, frame):
            self.requested = True
            self.signal = int(signum)

        for name in ("SIGINT", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                self._previous[number] = signal.signal(number, stop)
            except (ValueError, OSError):
                pass
        return self

    def restore(self):
        for number, handler in self._previous.items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                pass
        self._previous = {}


class ReplicaRun:
    """One coordinated replica-exchange calculation."""

    def __init__(self, *, protocol, files, base_system, topology, solute_indices,
                 excluded_bonds=(), platform=None, precision=None, rule_path=None,
                 reservoir_declaration=None, identity_extra=None):
        self.protocol = protocol
        self.files = files
        self.base_system = base_system
        self.topology = topology
        self.solute_indices = [int(i) for i in solute_indices]
        self.excluded_bonds = list(excluded_bonds)
        self.platform_request = platform
        self.precision = precision
        self.rule_path = rule_path
        self.reservoir_declaration = reservoir_declaration
        self.identity_extra = dict(identity_extra or {})

        self.coordinator = Coordinator()
        self.owned = owned_states(protocol, self.coordinator)
        self.engine = None
        self.reporter = None
        self.reservoir = None
        self._audit = None
        self._run_context = {}
        self._periodic = bool(base_system.usesPeriodicBoundaryConditions())

    # -- identity ---------------------------------------------------------------------------------

    def scientific_identity(self, rule_identity):
        """Everything a continuation must agree with, fixed before propagation begins."""
        payload = dict(self.protocol.describe())
        payload.update({
            "format": "md-templates-replica-identity/v2",
            "solute_atoms": len(self.solute_indices),
            # The UNSCALED reference every rung is derived from, recorded as such. Labelling this
            # digest with tau_max would describe one object with another object's tau: the ladder
            # is fixed by (reference system, tau list), and the tau list is already in
            # `protocol.describe()` above.
            "hamiltonian": hamiltonian_identity.identity_record(
                self.base_system, tau=None,
                temperature_k=self.protocol.temperature_k,
                ensemble="NVT" if self.protocol.pressure_bar is None else "NPT",
                solute_indices=self.solute_indices, excluded_bonds=self.excluded_bonds),
            "exchange_rule": rule_identity,
        })
        payload.update(self.identity_extra)
        return payload

    @staticmethod
    def compare_identity(before, now):
        keys = (set(before) | set(now)) - {"format"}
        return [key for key in sorted(keys) if before.get(key) != now.get(key)]

    # -- the run -------------------------------------------------------------------------------------

    def run(self, *, resume=False, extend=0):
        rule, rule_identity = self._load_rule()
        identity = self.scientific_identity(rule_identity)
        started = datetime.datetime.now(datetime.timezone.utc)

        continuing = bool(resume or extend)
        if continuing and not Path(self.files.trajectory).exists():
            raise DriverError(
                f"{self.files.trajectory} does not exist, so there is nothing to "
                f"{'extend' if extend else 'resume'}.")

        self._resolve_platform()
        context = self._run_context
        print(f"# platform           : {context['platform']} device={context['device_index']} "
              f"({context['device_policy']}), precision {context['precision']}")
        print(f"# this process drives: state(s) {context['owned_states']} of "
              f"{self.protocol.n_states}")
        print(f"# host               : {context['hostname']}")
        sys.stdout.flush()

        systems, self._audit = self.protocol.build_systems(
            self.base_system, self.solute_indices, self.excluded_bonds)

        interruption = _Interruption().install()
        try:
            state = self._begin(identity, systems, rule_identity, resume=resume, extend=extend)
            self._open_reservoir(systems)
            if self.reservoir is not None:
                self.reservoir.check_box_matches(state["configurations"][0].box)
            self._loop(state, rule, interruption)
        except BaseException as failure:
            if self.coordinator.is_root:
                interrupted = isinstance(failure, KeyboardInterrupt)
                storage.write_run_state(
                    self.files.trajectory, "interrupted" if interrupted else "failed",
                    identity=identity,
                    reason=f"{type(failure).__name__}: {failure}"[:400],
                    note=("the analysis NetCDF holds every committed record and this run can be "
                          "continued with --resume; no completion manifest exists"))
            raise
        finally:
            interruption.restore()

        if state.get("interrupted"):
            return self._record_interruption(state, identity)
        return self._finish(state, identity, rule_identity, started)

    # -- setup ------------------------------------------------------------------------------------------

    def _load_rule(self):
        if self.rule_path:
            return load_rule(self.rule_path)
        rule = NeighbouringExchangeRule()
        return rule, builtin_rule_identity(rule)

    def _open_reservoir(self, systems):
        if not self.reservoir_declaration:
            return
        from rrest2_reservoir import PreparedReservoir

        # The reservoir refreshes the TOP rung, so the Hamiltonian it must match is the top rung's
        # SCALED system -- not the unscaled reference. Comparing against the reference rejected
        # every correctly prepared source, and would have accepted a source recorded at tau = 0,
        # which is the pairing that actually breaks the probability-one rule.
        self.reservoir = PreparedReservoir.open(
            self.reservoir_declaration, protocol=self.protocol,
            topology_path=self.files.topology, periodic=self._periodic,
            system=systems[-1], solute_indices=self.solute_indices,
            excluded_bonds=self.excluded_bonds, coordinator=self.coordinator)
        print(f"# reservoir          : {self.reservoir.n_frames} phase-space sample(s), "
              f"velocity_policy={self.reservoir.velocity_policy}")
        sys.stdout.flush()

    def _resolve_platform(self):
        name = self.platform_request
        device, policy = None, "not a CUDA platform"
        if name in (None, "automatic"):
            from openmm import Platform
            available = {Platform.getPlatform(i).getName()
                         for i in range(Platform.getNumPlatforms())}
            name = "CUDA" if "CUDA" in available else "CPU"
        if name == "CUDA":
            devices = visible_cuda_devices(probe=(self.coordinator.size > 1))
            if self.coordinator.size > 1:
                device, policy = select_device_for_rank(
                    self.coordinator.rank, self.coordinator.size, devices)
                if device is None:
                    raise DriverError(
                        "the CUDA platform was selected under MPI but no CUDA device is visible "
                        "to this rank. Refusing rather than letting every rank fall onto one GPU.")
            else:
                policy = "single process: OpenMM selects the device"
        self._platform, self._properties = resolve_platform(
            name, precision=self.precision, device_index=device)
        self._run_context = {
            "platform": name, "device_index": device, "device_policy": policy,
            "precision": self._properties.get("Precision"),
            "mpi_rank": self.coordinator.rank, "mpi_size": self.coordinator.size,
            "hostname": socket.gethostname(), "owned_states": list(self.owned),
        }

    def _begin(self, identity, systems, rule_identity, *, resume, extend):
        seed = int(self.protocol.random_seed or 20260830)
        self.engine = ReplicaEngine(
            self.protocol, systems, self.topology, owned=self.owned,
            platform=self._platform, properties=self._properties, seed=seed)

        if resume or extend:
            return self._continue(identity, extend=extend)

        start = self._read_initial_configuration()
        configurations = [start.copy() for _ in range(self.protocol.n_states)]
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "initialized", identity=identity,
                total_steps=self.protocol.total_steps,
                note="scientific identity fixed; no propagation has happened yet")
            self.reporter = storage.ReplicaReporter.create(
                self.files.trajectory, n_states=self.protocol.n_states,
                n_atoms=configurations[0].n_atoms,
                n_solute_atoms=len(self.solute_indices),
                has_box=configurations[0].box is not None,
                identity=identity,
                metadata={"created_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                    "versions": environment_versions(),
                    "execution": self._run_context,
                    "exchange_rule": rule_identity})
            self.reporter.write_ladder(self.protocol.tau)
        self.coordinator.barrier()

        state = {
            "step": 0, "exchange_index": -1, "frame_index": -1, "solute_frame_index": -1,
            "state_to_walker": list(range(self.protocol.n_states)),
            "configurations": configurations,
            "schedule": self.protocol.schedule,
            "rng": np.random.default_rng(_stream_seed(seed, "exchange")),
            "rule_state": {}, "resumed_from_step": None, "interrupted": False,
            # A file this process just created needs no migration and must not be given a
            # fabricated event. Empty is the honest record.
            "storage_migrations": [],
        }
        self._equilibrate(state)
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "running", identity=identity,
                total_steps=self.protocol.total_steps,
                note="propagation started; the analysis NetCDF is authoritative for progress")
        return state

    def _previous_manifest(self):
        """The completion manifest of the run being continued, if it wrote one.

        Read-only and forgiving: an interrupted run never wrote one, and that is the ordinary
        case for a resume rather than a problem.
        """
        path = getattr(self.files, "restart", None)
        if not path or not Path(path).is_file():
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _continue(self, identity, *, extend):
        """Rank 0 validates and decides; every other rank receives the decision.

        No non-root rank opens the analysis file at all, in any mode. Rank 0 reads the checkpoint,
        compares the identity, rewinds the streams and broadcasts a payload the others simply
        adopt.
        """
        payload = None
        if self.coordinator.is_root:
            # Read the stored output and check it BEFORE opening it for writing. A file whose
            # completion markers lead its data, or whose streams have duplicated steps, must not
            # be continued: the continuation would read rows that were never written and then
            # append to them. Nothing here trusts the previous process to have exited cleanly.
            import replica_validate

            check = replica_validate.validate_replica_output(
                analysis=self.files.trajectory, checkpoint=self.files.checkpoint,
                expect_completed=False)
            if not check.ok:
                raise storage.StorageError(
                    replica_validate.format_report(
                        check, title="this run cannot be continued"))

            # PHASE 1 -- READ ONLY. Every reason to refuse is established while the file is open
            # for reading and nothing can be written. A file must never be modified merely because
            # it could be opened, and the optional-field migration below is a modification.
            probe = storage.ReplicaReporter(self.files.trajectory, mode="r")
            try:
                stored = probe.identity
                committed = probe.last_exchange()
                optional_fields = probe.inspect_optional_exchange_fields()
                file_history = probe.migration_history()
            finally:
                probe.close()

            # Every authoritative record that could carry migration history, all read-only. The
            # file is the durable authority for its own schema, but a run migrated by an earlier
            # build recorded the event only in its manifest, and an interrupted one only in its
            # run state -- so all three are consulted and merged rather than any one trusted to be
            # complete.
            prior_history = storage.merge_migration_histories(
                file_history,
                storage.migration_history_of(self._previous_manifest()),
                storage.migration_history_of(storage.read_run_state(self.files.trajectory)))

            malformed = [entry["problem"] for entry in optional_fields.values()
                         if entry["problem"]]
            if malformed:
                raise storage.StorageError(
                    "this run cannot be continued because an optional field is defined in a way "
                    "this runtime cannot use:\n  - " + "\n  - ".join(malformed) +
                    "\n  The file is left exactly as it is.")

            if stored is None:
                raise IdentityError(
                    f"{Path(self.files.trajectory).name} carries no scientific identity, so what "
                    f"it was created with cannot be established. Refusing rather than appending "
                    f"samples whose Hamiltonian cannot be checked.")
            differences = self.compare_identity(stored, identity)
            if differences:
                lines = "\n".join(f"    {k}: was {stored.get(k)!r}, now {identity.get(k)!r}"
                                  for k in differences)
                raise IdentityError(
                    f"the scientific configuration changed since this run was created, so "
                    f"continuing it would append samples from a different calculation:\n{lines}\n"
                    f"  Running longer is a legitimate extension; changing the physics is not.")

            checkpoint = storage.ReplicaCheckpoint(self.files.checkpoint).read()
            resume_exchange = int(checkpoint["exchange_index"])
            total = int(checkpoint["schedule"]["total_steps"])
            if extend:
                if int(checkpoint["step"]) < total:
                    raise IdentityError(
                        f"--extend lengthens a run that reached its budget, but this one stopped "
                        f"at step {checkpoint['step']} of {total}. Use --resume to finish it "
                        f"first; extending an unfinished run would add attempts while abandoning "
                        f"the ones never run.")
                # Extend the budget this run ACTUALLY reached, not the one in the protocol file.
                # A run that was already extended once stores a larger budget than the protocol
                # describes; extending the protocol's budget instead produced a new total BELOW
                # the current step, so the loop ran nothing and the summary reported "220000 of
                # 210000" while exiting successfully.
                already = ((total - self.protocol.schedule.total_steps)
                           // self.protocol.schedule.exchange_steps)
                schedule = self.protocol.schedule.extended(already + int(extend))
                if schedule.total_steps <= int(checkpoint["step"]):
                    raise IdentityError(
                        f"--extend {int(extend)} would set the budget to {schedule.total_steps} "
                        f"step(s), which is not beyond the {checkpoint['step']} already run. "
                        f"Refusing rather than reporting a run that propagated nothing as "
                        f"extended.")
            else:
                schedule = self.protocol.schedule
                if schedule.total_steps != total:
                    # The checkpoint's budget is authoritative for a resume: the protocol may have
                    # been generated with a different number_of_exchanges, and a resume finishes
                    # the run that was started.
                    schedule = self.protocol.schedule.extended(
                        (total - self.protocol.schedule.total_steps)
                        // self.protocol.schedule.exchange_steps)

            # PHASE 2 -- every check has passed, so now, and only now, the file is opened for
            # append and may be changed.
            reporter = storage.ReplicaReporter(self.files.trajectory, mode="a")
            self.reporter = reporter

            # An older v2 file predates the optional exchange fields. Adding them is done ONCE,
            # here, before a single step is integrated -- never lazily at the first exchange, where
            # a schema error would surface only after new dynamics had already been produced.
            migration = reporter.ensure_optional_exchange_fields()
            # Append-only. A continuation that changed nothing contributes nothing, so it cannot
            # overwrite the event that a real migration recorded -- which is exactly what the
            # singular field did, losing the fact that a legacy file had ever been migrated.
            migrations = storage.merge_migration_histories(prior_history, [migration])
            if migration["fields_added"]:
                print(f"# storage migrated     : added {migration['fields_added']} to this v2 "
                      f"file and set {migration['rows_initialised']} existing row(s) to "
                      f"{migration['initialised_to']}; "
                      f"{migration['previous_committed_exchanges']} exchange(s) were committed "
                      f"before this continuation")
                sys.stdout.flush()

            if committed > resume_exchange:
                reporter.rewind(exchange=resume_exchange,
                                frame=int(checkpoint["frame_index"]),
                                solute_frame=int(checkpoint["solute_frame_index"]))
                print(f"# rewound {committed - resume_exchange} exchange row(s) with no "
                      f"checkpoint behind them; continuing from step {checkpoint['step']}")

            payload = {
                "storage_migrations": migrations,
                "step": int(checkpoint["step"]),
                "exchange_index": resume_exchange,
                "frame_index": int(checkpoint["frame_index"]),
                "solute_frame_index": int(checkpoint["solute_frame_index"]),
                "state_to_walker": list(checkpoint["state_to_walker"]),
                "rng": checkpoint["rng_states"]["exchange"],
                "rule_state": dict(checkpoint["rule_state"]),
                "total_steps": schedule.total_steps,
                "number_of_exchanges": schedule.number_of_exchanges,
                "configurations": [(c.positions, c.velocities, c.box)
                                   for c in checkpoint["configurations"]],
            }
        payload = self.coordinator.bcast(payload)

        rng = np.random.default_rng()
        rng.bit_generator.state = payload["rng"]
        schedule = self.protocol.schedule
        if schedule.number_of_exchanges != payload["number_of_exchanges"]:
            schedule = schedule.extended(
                payload["number_of_exchanges"] - schedule.number_of_exchanges)
        state = {
            "step": payload["step"], "exchange_index": payload["exchange_index"],
            "frame_index": payload["frame_index"],
            "solute_frame_index": payload["solute_frame_index"],
            "state_to_walker": list(payload["state_to_walker"]),
            "configurations": [Configuration(p, v, b) for p, v, b in payload["configurations"]],
            "schedule": schedule,
            "rng": rng, "rule_state": dict(payload["rule_state"]),
            "resumed_from_step": payload["step"], "interrupted": False,
            "storage_migrations": list(payload.get("storage_migrations") or []),
        }
        self.coordinator.agree(configuration_digest(state["configurations"]),
                               what="the continued configurations")

        if self.coordinator.is_root:
            # The continuation provenance says what this process did to the storage before it
            # propagated anything, including how many exchanges the file already held. A reader
            # who finds `-1` in the early rows of a seed history can tell from here whether those
            # rows predate the field or were written by a `stored`-policy run.
            history = state["storage_migrations"]
            storage.write_run_state(
                self.files.trajectory, "running", identity=identity,
                total_steps=state["schedule"].total_steps,
                resumed_from_step=int(state["step"]),
                storage_migrations=history,
                note=("continuation started; the analysis NetCDF is authoritative for progress"
                      + (f". This file carries {len(history)} recorded schema migration(s); the "
                         f"most recent added {history[-1]['fields_added']} with "
                         f"{history[-1]['previous_committed_exchanges']} exchange(s) already "
                         f"committed" if history else "")))
        return state

    def _read_initial_configuration(self):
        from openmm import XmlSerializer, unit

        text = Path(self.files.coordinates).read_text(encoding="utf-8")
        opened = XmlSerializer.deserialize(text)
        positions = opened.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        try:
            velocities = opened.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond)
        except Exception:
            velocities = np.zeros_like(np.asarray(positions))
        box = None
        if self._periodic:
            box = np.array(opened.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                unit.nanometer), dtype=float)
        return Configuration(positions, velocities, box)

    def _equilibrate(self, state):
        """Per-state relaxation before any exchange. Not counted as production."""
        steps = self.protocol.equilibration_steps
        if not steps:
            return
        print(f"# equilibration      : {steps} step(s) per state, NOT counted as production")
        sys.stdout.flush()
        for index in self.owned:
            self.engine.set_configuration(index, state["configurations"][index])
            self.engine.propagate(index, steps)
        state["configurations"] = self._gather_configurations(state)

    # -- the loop ------------------------------------------------------------------------------------

    def _gather_configurations(self, state):
        """Every walker's sample, on every rank, walker-indexed."""
        local = {}
        for index in self.owned:
            walker = state["state_to_walker"][index]
            local[walker] = self.engine.get_configuration(index)
        if self.coordinator.size == 1:
            return [local[w] for w in range(self.protocol.n_states)]
        assembled = {}
        for piece in self.coordinator.allgather(
                {w: (c.positions, c.velocities, c.box) for w, c in local.items()}):
            for walker, (positions, velocities, box) in piece.items():
                assembled[walker] = Configuration(positions, velocities, box)
        return [assembled[w] for w in range(self.protocol.n_states)]

    def _reduced_potential_matrix(self, configurations):
        """u[i][w]: walker w's sample in state i's Hamiltonian. Rank i computes row i."""
        n = self.protocol.n_states
        rows = {index: [self.engine.reduced_potential_of(index, configurations[w])
                        for w in range(n)] for index in self.owned}
        if self.coordinator.size > 1:
            for piece in self.coordinator.allgather(rows):
                rows.update(piece)
        return np.array([rows[i] for i in range(n)], dtype=float)

    def _install_owned(self, state):
        for index in self.owned:
            self.engine.set_configuration(
                index, state["configurations"][state["state_to_walker"][index]])

    def _loop(self, state, rule, interruption):
        schedule = state["schedule"]
        self._install_owned(state)

        while state["step"] < schedule.total_steps:
            target = schedule.next_event_step(state["step"])
            if target is None:
                break
            span = target - state["step"]
            for index in self.owned:
                self.engine.propagate(index, span)
            state["step"] = target
            state["configurations"] = self._gather_configurations(state)

            events = schedule.events_at(target)
            reservoir_event = None
            if "exchange" in events:
                reservoir_event = self._exchange(state, rule, target, schedule)
            if self.coordinator.is_root:
                if "exchange" in events:
                    pass    # already written by _exchange
                if "whole" in events:
                    state["frame_index"] = self.reporter.write_frame(
                        step=target, time_ps=schedule.step_to_ps(target),
                        exchange_index=state["exchange_index"],
                        configurations=state["configurations"])
                if "solute" in events:
                    state["solute_frame_index"] = self.reporter.write_solute_frame(
                        step=target, time_ps=schedule.step_to_ps(target),
                        exchange_index=state["exchange_index"],
                        configurations=state["configurations"],
                        solute_indices=self.solute_indices)
                if "checkpoint" in events:
                    self._write_checkpoint(state, schedule)
            self.coordinator.barrier()

            # A termination request becomes collective HERE, at an event boundary where every rank
            # is together. One rank raising inside its signal handler would leave the others in a
            # collective forever.
            if self.coordinator.any_true(interruption.requested):
                if self.coordinator.is_root and "checkpoint" not in events:
                    self._write_checkpoint(state, schedule)
                state["interrupted"] = True
                state["interrupt_signal"] = interruption.signal
                self.coordinator.barrier()
                return

    def _exchange(self, state, rule, step, schedule):
        n = self.protocol.n_states
        matrix = self._reduced_potential_matrix(state["configurations"])
        state["exchange_index"] += 1

        payload = None
        if self.coordinator.is_root:
            context = ExchangeContext(
                iteration=state["exchange_index"], segment=state["step"], protocol=self.protocol,
                state_to_walker=state["state_to_walker"],
                reduced_potential=lambda i, w: matrix[i][w],
                rng=state["rng"], reservoir=self.reservoir, rule_state=state["rule_state"],
                exchange_index=state["exchange_index"])
            outcome = rule.propose(context)
            payload = {"proposals": outcome.proposals, "swaps": outcome.swaps,
                       "reservoir_refresh": outcome.reservoir_refresh,
                       "rule_state": outcome.rule_state, "diagnostics": outcome.diagnostics}
        payload = self.coordinator.bcast(payload)

        proposed = np.zeros((n, n), dtype=np.int64)
        accepted = np.zeros((n, n), dtype=np.int64)
        for state_i, state_j, _log_alpha, was_accepted in payload["proposals"]:
            proposed[state_i, state_j] += 1
            proposed[state_j, state_i] += 1
            if was_accepted:
                accepted[state_i, state_j] += 1
                accepted[state_j, state_i] += 1
        for state_i, state_j in payload["swaps"]:
            mapping = state["state_to_walker"]
            mapping[state_i], mapping[state_j] = mapping[state_j], mapping[state_i]
        state["rule_state"] = payload["rule_state"] or state["rule_state"]

        reservoir_event = None
        if payload["reservoir_refresh"] is not None:
            reservoir_event = self._apply_reservoir(state, payload["reservoir_refresh"])

        self._install_owned(state)
        self.coordinator.agree(configuration_digest(state["configurations"]),
                               what="the configurations after this exchange")

        if self.coordinator.is_root:
            self.reporter.write_exchange(
                state["exchange_index"], step=step, time_ps=schedule.step_to_ps(step),
                state_to_walker=state["state_to_walker"], proposed=proposed, accepted=accepted,
                u=matrix, u_evaluated=np.ones((n, n), dtype=np.int8),
                reservoir=reservoir_event)
        return reservoir_event

    def _apply_reservoir(self, state, refresh):
        """Install one prepared phase-space sample into the top state. Single authority, no stale
        broadcast.

        Every rank reads the SAME sample from the SAME prepared file using the index rank 0 chose,
        so nothing large travels over MPI and no rank's copy can be stale. The rank owning the
        state installs it into its Context; every rank updates the walker-indexed list identically;
        and the caller then verifies a digest across ranks.

        This replaces a flow that installed the sample on the owning rank and then overwrote the
        list with rank 0's copy -- discarding the new velocity whenever the top rung was not rank
        0's.
        """
        state_index = int(refresh["state"])
        frame_index = int(refresh["frame"])
        walker = state["state_to_walker"][state_index]
        current = state["configurations"][walker]

        positions, velocities, box, source_step, _source_time = self.reservoir.sample(frame_index)

        if self.reservoir.velocity_policy == "stored":
            # The recorded momentum, installed unchanged. This is what the probability-one rule is
            # stated for: a phase-space sample is a point in phase space.
            installed_velocities = np.asarray(velocities, dtype=float)
        else:
            # `maxwell` was asked for explicitly. Drawn on every rank from the same recorded seed
            # via the owning rank, below.
            installed_velocities = None

        if self._periodic and box is None:
            raise DriverError(
                "the reservoir sample carries no box but this is an explicit-solvent run; a "
                "configuration without its box is at an undefined density.")
        # The ladder's basis is kept: the lattices were proven equal before propagation, so this is
        # physically identical and stops the stored box flickering between equivalent bases.
        keep_box = current.box

        if installed_velocities is None:
            # Maxwell: the owning rank draws, then shares the drawn momenta so every rank holds the
            # same sample. Nothing else is broadcast.
            drawn = None
            if state_index in self.owned:
                self.engine.set_configuration(
                    state_index, Configuration(positions, current.velocities, keep_box))
                self.engine.set_velocities_to_temperature(
                    state_index, int(refresh["velocity_seed"]))
                drawn = self.engine.get_configuration(state_index).velocities
            if self.coordinator.size > 1:
                for piece in self.coordinator.allgather(drawn):
                    if piece is not None:
                        drawn = piece
                        break
            installed_velocities = np.asarray(drawn, dtype=float)

        replacement = Configuration(positions, installed_velocities, keep_box)
        state["configurations"][walker] = replacement
        if state_index in self.owned:
            self.engine.set_configuration(state_index, replacement)
        # The seed is recorded only when it was actually used. Under `stored` nothing was drawn,
        # and writing the rule's unused seed there would suggest a draw that never happened.
        drawn_from = (int(refresh["velocity_seed"])
                      if self.reservoir.velocity_policy == "maxwell" else -1)
        return (state_index, frame_index, int(source_step),
                1 if refresh.get("accepted", True) else 0, drawn_from)

    def _write_checkpoint(self, state, schedule):
        storage.ReplicaCheckpoint(self.files.checkpoint).write(
            step=state["step"], exchange_index=state["exchange_index"],
            frame_index=state["frame_index"],
            solute_frame_index=state["solute_frame_index"],
            configurations=state["configurations"],
            state_to_walker=state["state_to_walker"],
            rng_states={"exchange": _encode_rng(state["rng"])},
            rule_state=state["rule_state"], schedule=schedule.describe(),
            identity=self.reporter.identity,
            extra={"configuration_digest": configuration_digest(state["configurations"])})

    # -- finishing ------------------------------------------------------------------------------------

    def _record_interruption(self, state, identity):
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "interrupted", identity=identity,
                step=state["step"], total_steps=state["schedule"].total_steps,
                signal=state.get("interrupt_signal"),
                storage_migrations=list(state.get("storage_migrations") or []),
                note=("stopped at an event boundary with a complete checkpoint; --resume "
                      "continues it. No completion manifest exists."))
            print(f"# interrupted at step {state['step']} of {state['schedule'].total_steps}; "
                  f"a checkpoint was committed and --resume will continue it")
        return {"run_status": "interrupted", "step": state["step"]}

    def _finish(self, state, identity, rule_identity, started):
        completed = state["step"]
        expected = state["schedule"].total_steps
        rank = self.coordinator.rank
        if rank != 0:
            print(f"# rank {rank} finished its share of the propagation; rank 0 owns the manifest")
            print(f"rank_status: completed rank={rank}")
            return {"run_status": "completed_by_rank", "rank": rank}

        if completed < expected:
            storage.write_run_state(self.files.trajectory, "interrupted", identity=identity,
                                    step=completed, note="stopped before the budget; resumable")
            raise DriverError(
                f"the run stopped at step {completed} of {expected}. No completion manifest was "
                f"written; --resume will continue it.")

        from replica_statistics import lifetime_statistics, round_trip_report
        accepted, proposed = self.reporter.statistics()
        events = self.reporter.reservoir_events()
        mapping = self.reporter.mapping()
        stats = lifetime_statistics(
            accepted, proposed, tau=self.protocol.tau, reservoir_events=events,
            reservoir_velocity_seeds=self.reporter.reservoir_velocity_seeds())
        trips = round_trip_report(mapping, n_states=self.protocol.n_states)

        record = {
            "format": storage.MANIFEST_FORMAT,
            "run_status": "completed",
            "steps_expected": int(expected),
            "steps_completed": int(completed),
            "exchanges_committed": int(self.reporter.last_exchange()) + 1,
            "whole_frames": int(self.reporter.last_frame()) + 1,
            "solute_frames": int(self.reporter.last_solute_frame()) + 1,
            "production_ps_per_replica": state["schedule"].step_to_ps(completed),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "resumed_from_step": state.get("resumed_from_step"),
            # Durable provenance for the optional-field migration. The run-state sidecar records
            # it while the continuation is running, but that file is rewritten on completion, so a
            # reader of the finished run would otherwise have no record that this file was
            # written before the field existed -- which is exactly what explains a leading run of
            # -1 in its seed history.
            "storage_migrations": list(state.get("storage_migrations") or []),
            "schedule": state["schedule"].describe(),
            "storage": {
                "analysis_netcdf": Path(self.files.trajectory).name,
                "solute_netcdf": storage.solute_path(self.files.trajectory).name,
                "checkpoint_netcdf": Path(self.files.checkpoint).name,
                "run_state": storage.run_state_path(self.files.trajectory).name,
                "schema": storage.SCHEMA_VERSION,
                "authoritative": "analysis_netcdf",
                "coordinate_indexing": "walker",
            },
            "scientific_identity": identity,
            "exchange_rule": rule_identity,
            "lifetime_statistics": stats,
            "round_trips": trips,
            "hamiltonian": {
                "forces_scaled": [n for _, n in self._audit["scaled"]],
                "forces_unscaled_by_convention": [
                    n for _, n in self._audit["unscaled_by_convention"]],
                "forces_energy_free": [n for _, n in self._audit["energy_free"]],
            },
            "versions": environment_versions(),
            "execution": self._run_context,
            "final_state_to_walker": list(state["state_to_walker"]),
        }
        if self.reservoir is not None:
            record["reservoir"] = self.reservoir.describe()
        storage.write_atomic(self.files.restart,
                             json.dumps(record, indent=2, default=str) + "\n")
        # The completed sidecar carries the history too. It is one of the sources a later
        # continuation merges, and a `completed` record that reported none would be a source
        # claiming this file had never been migrated.
        storage.write_run_state(self.files.trajectory, "completed", identity=identity,
                                step=completed,
                                storage_migrations=list(state.get("storage_migrations") or []),
                                note="manifest written; the storage remains authoritative")
        self.reporter.close()
        return record


def _stream_seed(base, name):
    value = int(base)
    for byte in str(name).encode("utf-8"):
        value = (value * 1000003 + byte) & 0xFFFFFFFF
    return (value % (2 ** 31 - 1)) or 1


def _encode_rng(generator):
    return json.loads(json.dumps(generator.bit_generator.state,
                                 default=lambda o: int(o) if hasattr(o, "__int__") else str(o)))


def environment_versions():
    import netCDF4
    import openmm
    record = {"python": platform_module.python_version(),
              "openmm": openmm.version.version,
              "openmm_short": openmm.version.short_version,
              "netCDF4": netCDF4.__version__, "numpy": np.__version__}
    for name in ("mdtraj", "mpi4py"):
        try:
            record[name] = __import__(name).__version__
        except ImportError:
            record[name] = None
    try:
        import openmmtools
        record["openmmtools_present_but_unused"] = openmmtools.__version__
    except ImportError:
        record["openmmtools_present_but_unused"] = None
    return record
