#!/usr/bin/env python
"""The replica-exchange loop: propagate, evaluate, decide, record. Copied into every project.

This is the only module that knows about all the others. It owns nothing scientific: the ladder
comes from `replica_protocol`, the Hamiltonians from `rest2_scaling`, the decisions from an
exchange rule, and the storage schema from `replica_storage`.

THE PARALLEL POLICY, CHOSEN AND STATED
    World size must be 1 or exactly the number of states. Nothing in between is supported, because
    a policy that silently packed several states onto one rank would make the device assignment and
    the timing unreproducible, and "it ran" would stop meaning "it ran the way the record says".

        size == 1            one process owns every state
        size == n_states     rank r owns state r, and drives one device

    Configurations are allgathered after propagation, so every rank can evaluate its own state's
    Hamiltonian against every walker. Decisions are made on rank 0 from a recorded RNG and
    broadcast, so a run does not depend on every rank drawing the same numbers.

THE ENERGY MATRIX, AND WHAT IT COSTS
    At each exchange iteration the FULL n x n reduced-potential matrix is evaluated: rank r
    computes row r, which parallelises exactly. A neighbouring sweep needs only the adjacent
    blocks, so this does more work than the minimum -- n^2 evaluations instead of about 4n -- and
    the reason is that it makes the rule contract a pure lookup with no communication, and leaves
    a complete matrix in storage for later analysis. For a six-state ladder exchanging every fifth
    segment that is a small fraction of propagation, and it is measured rather than assumed.

    Every entry is evaluated INDEPENDENTLY by installing the configuration and asking OpenMM. No
    cross energy is ever inferred by scaling another one: the scaled Hamiltonians differ by more
    than a single factor, so such a shortcut would be a different number that merely looks
    plausible.
"""
import datetime
import json
import platform as platform_module
import socket
import sys
from pathlib import Path

import numpy as np

import replica_storage as storage
from exchange_rules import (ExchangeContext, NeighbouringExchangeRule, builtin_rule_identity,
                            load_rule)
from replica_engine import Configuration, ReplicaEngine, resolve_platform, select_device_for_rank, \
    visible_cuda_devices


class DriverError(RuntimeError):
    """The run cannot proceed as configured."""


class IdentityError(DriverError):
    """The run being continued is not the run that was started."""


class Coordinator:
    """Rank, size, and the three collectives the driver needs. No MPI import when not under MPI."""

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
        if self.comm is None:
            return [value]
        return self.comm.allgather(value)

    def bcast(self, value):
        if self.comm is None:
            return value
        return self.comm.bcast(value, root=0)

    def barrier(self):
        if self.comm is not None:
            self.comm.barrier()


def owned_states(protocol, coordinator):
    """Which states this process drives, under the stated policy."""
    if coordinator.size == 1:
        return list(range(protocol.n_states))
    if coordinator.size != protocol.n_states:
        raise DriverError(
            f"this ladder has {protocol.n_states} states but MPI world size is "
            f"{coordinator.size}. The supported policy is one process for the whole ladder, or "
            f"exactly one rank per state. Packing several states onto a rank is refused rather "
            f"than done silently, because the device assignment and the timing in the record "
            f"would stop describing what ran.")
    return [coordinator.rank]


class ReplicaRun:
    """One coordinated replica-exchange calculation."""

    def __init__(self, *, protocol, files, base_system, topology, solute_indices,
                 excluded_bonds=(), platform=None, precision=None, rule_path=None,
                 reservoir=None, identity_extra=None):
        self.protocol = protocol
        self.files = files
        self.base_system = base_system
        self.topology = topology
        self.solute_indices = list(solute_indices)
        self.excluded_bonds = list(excluded_bonds)
        self.platform_request = platform
        self.precision = precision
        self.rule_path = rule_path
        self.reservoir = reservoir
        self.identity_extra = dict(identity_extra or {})

        self.coordinator = Coordinator()
        self.owned = owned_states(protocol, self.coordinator)
        self.engine = None
        self.reporter = None
        self._audit = None
        self._run_context = {}

    # -- identity ---------------------------------------------------------------------------------

    def scientific_identity(self, rule_identity):
        """Everything a continuation must agree with, fixed before propagation begins."""
        import hashlib

        def digest(path):
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        payload = dict(self.protocol.describe())
        payload.update({
            "format": "md-templates-replica-identity/v1",
            "solute_atoms": len(self.solute_indices),
            "enhanced_region_sha256": hashlib.sha256(json.dumps(
                {"solute": [int(i) for i in self.solute_indices],
                 "omega_excluded_bonds": [[int(a), int(b)] for a, b in self.excluded_bonds]},
                sort_keys=True).encode("utf-8")).hexdigest(),
            "topology_sha256": digest(self.files.topology),
            "system_sha256": digest(self.files.system),
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
                f"{'extend' if extend else 'resume'}. The analysis NetCDF is authoritative.")

        self._resolve_platform()
        systems, self._audit = self.protocol.build_systems(
            self.base_system, self.solute_indices, self.excluded_bonds)

        state = self._begin(identity, systems, rule, rule_identity, resume=resume, extend=extend)
        handlers = _install_interrupt_handlers()
        try:
            self._loop(state, rule)
        except BaseException as failure:
            if self.coordinator.is_root:
                status = "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
                storage.write_run_state(
                    self.files.trajectory, status, identity=identity,
                    reason=f"{type(failure).__name__}: {failure}"[:400],
                    iteration=state["iteration"],
                    note=("the analysis NetCDF holds every committed iteration and this run can "
                          "be continued with --resume; no completion manifest exists"))
            raise
        finally:
            _restore_interrupt_handlers(handlers)
        return self._finish(state, identity, rule, rule_identity, started)

    # -- setup ------------------------------------------------------------------------------------------

    def _load_rule(self):
        if self.rule_path:
            rule, identity = load_rule(self.rule_path)
            return rule, identity
        rule = NeighbouringExchangeRule()
        return rule, builtin_rule_identity(rule)

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
            "hostname": socket.gethostname(),
            "owned_states": list(self.owned),
        }

    def _begin(self, identity, systems, rule, rule_identity, *, resume, extend):
        seed = int(self.protocol.random_seed or 20260830)
        self.engine = ReplicaEngine(
            self.protocol, systems, self.topology, owned=self.owned,
            platform=self._platform, properties=self._properties, seed=seed)

        if resume or extend:
            return self._continue(identity, rule_identity, extend=extend)

        # A brand-new run. The identity goes into the storage BEFORE anything propagates.
        start = self._read_initial_configuration()
        configurations = [start.copy() for _ in range(self.protocol.n_states)]
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "initialized", identity=identity,
                budget_segments=self.protocol.total_segments,
                note="scientific identity fixed; no propagation has happened yet")
            self.reporter = storage.ReplicaReporter.create(
                self.files.trajectory, n_states=self.protocol.n_states,
                n_atoms=configurations[0].n_atoms,
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
            "iteration": -1, "segment": 0,
            "state_to_walker": list(range(self.protocol.n_states)),
            "configurations": configurations,
            "budget_segments": self.protocol.total_segments,
            "rng": np.random.default_rng(_stream_seed(seed, "exchange")),
            "rule_state": {},
            "exchange_index": 0,
            "resumed_from": None,
        }
        self._equilibrate(state)
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "running", identity=identity,
                budget_segments=state["budget_segments"],
                note="propagation started; the analysis NetCDF is authoritative for progress")
        return state

    def _continue(self, identity, rule_identity, *, extend):
        reporter = storage.ReplicaReporter(self.files.trajectory, mode="a")
        self.reporter = reporter if self.coordinator.is_root else None
        stored = reporter.identity
        if stored is None:
            raise IdentityError(
                f"{Path(self.files.trajectory).name} carries no scientific identity, so what it "
                f"was created with cannot be established. Refusing rather than appending samples "
                f"whose Hamiltonian cannot be checked.")
        differences = self.compare_identity(stored, identity)
        if differences:
            lines = "\n".join(f"    {k}: was {stored.get(k)!r}, now {identity.get(k)!r}"
                              for k in differences)
            raise IdentityError(
                f"the scientific configuration changed since this run was created, so continuing "
                f"it would append samples from a different calculation:\n{lines}\n"
                f"  Running longer is a legitimate extension; changing the physics is not.")

        checkpoint = storage.ReplicaCheckpoint(self.files.checkpoint).read()
        committed = reporter.last_iteration()
        # A run is continued from the last CHECKPOINT, not from the last committed row. The
        # reporter commits every iteration while checkpoints are written every
        # whole_output_stride, so an interruption typically leaves committed rows with no
        # configurations behind them. Continuing from `last_iteration` would carry the
        # checkpoint's OLDER configurations forward under the newer iteration number, and the
        # segment counter and the exchange schedule would drift apart from then on.
        last = int(checkpoint["iteration"])
        if committed > last and self.coordinator.is_root:
            reporter.rewind(last)
            print(f"# rewound {committed - last} committed row(s) with no checkpoint behind them; "
                  f"continuing from the last checkpoint at iteration {last}")
        budget = int(checkpoint["budget_segments"])
        if not self.coordinator.is_root:
            reporter.close()

        if extend:
            if last + 1 < budget:
                raise IdentityError(
                    f"--extend lengthens a run that reached its budget, but this one stopped at "
                    f"segment {last + 1} of {budget}. Use --resume to finish it first; extending "
                    f"an unfinished run would add segments while abandoning the ones never run.")
            budget = budget + int(extend) * self.protocol.exchange_stride

        rng = np.random.default_rng()
        rng.bit_generator.state = _decode_rng(checkpoint["rng_states"]["exchange"])
        return {
            "iteration": last, "segment": int(last) + 1,
            "state_to_walker": list(checkpoint["state_to_walker"]),
            "configurations": checkpoint["configurations"],
            "budget_segments": budget,
            "rng": rng,
            "rule_state": dict(checkpoint["rule_state"]),
            # Carried across a resume so the odd/even schedule continues rather than restarting.
            "exchange_index": int(checkpoint["extra"].get("exchange_index", 0)),
            "resumed_from": last,
        }

    def _read_initial_configuration(self):
        from openmm import XmlSerializer, unit

        text = Path(self.files.coordinates).read_text(encoding="utf-8")
        state = XmlSerializer.deserialize(text)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        try:
            velocities = state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond)
        except Exception:
            velocities = np.zeros_like(np.asarray(positions))
        box = None
        vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        if np.any(np.asarray(vectors, dtype=float)):
            box = np.array(vectors, dtype=float)
        return Configuration(positions, velocities, box)

    def _equilibrate(self, state):
        """Per-state relaxation before any exchange. Not counted as production."""
        if not self.protocol.equilibration_segments:
            return
        for index in self.owned:
            self.engine.set_configuration(index, state["configurations"][index])
        for _ in range(self.protocol.equilibration_segments):
            for index in self.owned:
                self.engine.propagate(index, self.protocol.steps_per_segment)
        state["configurations"] = self._gather_configurations()

    # -- the loop ------------------------------------------------------------------------------------

    def _gather_configurations(self):
        """Every walker's configuration, on every rank.

        `state_to_walker` says which walker each state holds, so a rank reads back the state it
        owns and the allgather assembles the rest.
        """
        local = {index: self.engine.get_configuration(index) for index in self.owned}
        if self.coordinator.size == 1:
            return [local[i] for i in range(self.protocol.n_states)]
        packed = self.coordinator.allgather(
            {i: (c.positions, c.velocities, c.box) for i, c in local.items()})
        assembled = {}
        for piece in packed:
            for index, (positions, velocities, box) in piece.items():
                assembled[index] = Configuration(positions, velocities, box)
        return [assembled[i] for i in range(self.protocol.n_states)]

    def _reduced_potential_matrix(self, configurations):
        """u[i][w]: walker w's configuration in state i's Hamiltonian. Rank i computes row i."""
        n = self.protocol.n_states
        rows = {}
        for index in self.owned:
            rows[index] = [self.engine.reduced_potential_of(index, configurations[w])
                           for w in range(n)]
        if self.coordinator.size > 1:
            for piece in self.coordinator.allgather(rows):
                rows.update(piece)
        return np.array([rows[i] for i in range(n)], dtype=float)

    def _loop(self, state, rule):
        protocol = self.protocol
        n = protocol.n_states
        # Install the configurations this rank's states hold.
        for index in self.owned:
            walker = state["state_to_walker"][index]
            self.engine.set_configuration(index, state["configurations"][walker])

        if state["segment"] != state["iteration"] + 1:
            raise DriverError(
                f"segment {state['segment']} and iteration {state['iteration']} are out of step; "
                f"segment N is iteration N-1 by construction, and a continuation that broke that "
                f"would record every later row under the wrong physical time.")
        while state["segment"] < state["budget_segments"]:
            state["segment"] += 1
            state["iteration"] += 1
            for index in self.owned:
                self.engine.propagate(index, protocol.steps_per_segment)
            configurations = self._gather_configurations()
            # `configurations` is indexed by STATE here; re-key it by walker.
            by_walker = [None] * n
            for index, walker in enumerate(state["state_to_walker"]):
                by_walker[walker] = configurations[index]
            state["configurations"] = by_walker

            is_exchange = (state["segment"] % protocol.exchange_stride == 0)
            proposed = np.zeros((n, n), dtype=np.int64)
            accepted = np.zeros((n, n), dtype=np.int64)
            u = np.zeros((n, n), dtype=float)
            evaluated = np.zeros((n, n), dtype=np.int8)
            reservoir_event = None

            if is_exchange:
                matrix = self._reduced_potential_matrix(by_walker)
                u[:, :] = matrix
                evaluated[:, :] = 1
                outcome = self._decide(state, rule, matrix)
                state["exchange_index"] += 1
                for state_i, state_j, log_alpha, was_accepted in outcome.proposals:
                    proposed[state_i, state_j] += 1
                    proposed[state_j, state_i] += 1
                    if was_accepted:
                        accepted[state_i, state_j] += 1
                        accepted[state_j, state_i] += 1
                for state_i, state_j in outcome.swaps:
                    mapping = state["state_to_walker"]
                    mapping[state_i], mapping[state_j] = mapping[state_j], mapping[state_i]
                state["rule_state"] = outcome.rule_state or state["rule_state"]
                if outcome.reservoir_refresh is not None:
                    reservoir_event = self._apply_reservoir(state, outcome.reservoir_refresh)
                # Re-install whatever each owned state now holds.
                for index in self.owned:
                    self.engine.set_configuration(
                        index, state["configurations"][state["state_to_walker"][index]])

            time_ps = state["segment"] * protocol.segment_ps
            if self.coordinator.is_root:
                self.reporter.write_iteration(
                    state["iteration"], segment=state["segment"], time_ps=time_ps,
                    is_exchange=is_exchange, state_to_walker=state["state_to_walker"],
                    proposed=proposed, accepted=accepted, u=u, u_evaluated=evaluated,
                    reservoir=reservoir_event)
                if state["segment"] % protocol.whole_output_stride == 0:
                    self.reporter.write_frame(state["iteration"], state["configurations"])
                    storage.ReplicaCheckpoint(self.files.checkpoint).write(
                        iteration=state["iteration"], segment=state["segment"],
                        configurations=state["configurations"],
                        state_to_walker=state["state_to_walker"],
                        rng_states={"exchange": _encode_rng(state["rng"])},
                        rule_state=state["rule_state"],
                        budget_segments=state["budget_segments"],
                        identity=self.reporter.identity,
                        extra={"exchange_index": state["exchange_index"]})
            self.coordinator.barrier()

    def _decide(self, state, rule, matrix):
        """Rule decisions are made on rank 0 and broadcast, so ranks cannot diverge."""
        if self.coordinator.is_root:
            context = ExchangeContext(
                iteration=state["iteration"], segment=state["segment"], protocol=self.protocol,
                state_to_walker=state["state_to_walker"],
                reduced_potential=lambda i, w: matrix[i][w],
                rng=state["rng"], reservoir=self.reservoir,
                rule_state=state["rule_state"],
                exchange_index=state["exchange_index"])
            outcome = rule.propose(context)
            payload = {"proposals": outcome.proposals, "swaps": outcome.swaps,
                       "reservoir_refresh": outcome.reservoir_refresh,
                       "rule_state": outcome.rule_state, "diagnostics": outcome.diagnostics}
        else:
            payload = None
        payload = self.coordinator.bcast(payload)
        from exchange_rules import ExchangeOutcome
        return ExchangeOutcome(**payload)

    def _apply_reservoir(self, state, refresh):
        """Replace the configuration in one state from the prepared reservoir.

        Not a swap and never recorded as one: nothing is displaced into the reservoir, and the
        walker that was there simply ceases to carry its previous configuration.
        """
        state_index = int(refresh["state"])
        frame_index = int(refresh["frame"])
        walker = state["state_to_walker"][state_index]
        positions, box = self.reservoir.configuration(frame_index)
        current = state["configurations"][walker]
        if box is None and current.box is not None:
            raise DriverError(
                "the reservoir frame carries no box but this is an explicit-solvent run; a "
                "configuration without its box is at an undefined density.")
        replacement = Configuration(positions, current.velocities,
                                    None if current.box is None else np.asarray(box, dtype=float))
        state["configurations"][walker] = replacement
        if state_index in self.owned:
            self.engine.set_configuration(state_index, replacement)
            # A DCD carries no velocities. Fresh Maxwell momenta at the one common temperature,
            # from a recorded seed -- a configuration is not a restart.
            self.engine.set_velocities_to_temperature(state_index, int(refresh["velocity_seed"]))
            state["configurations"][walker] = self.engine.get_configuration(state_index)
        state["configurations"] = self.coordinator.bcast(state["configurations"]) \
            if self.coordinator.size > 1 else state["configurations"]
        return (state_index, frame_index, 1 if refresh.get("accepted", True) else 0)

    # -- finishing ------------------------------------------------------------------------------------

    def _finish(self, state, identity, rule, rule_identity, started):
        completed = state["segment"]
        expected = state["budget_segments"]
        if not self.coordinator.is_root:
            print(f"# rank {self.coordinator.rank} finished its share of the propagation; "
                  f"rank 0 owns the manifest and the storage")
            print(f"rank_status: completed rank={self.coordinator.rank}")
            return {"run_status": "completed_by_rank", "rank": self.coordinator.rank}

        if completed < expected:
            storage.write_run_state(self.files.trajectory, "interrupted", identity=identity,
                                    iteration=state["iteration"],
                                    note="stopped before the requested budget; resumable")
            raise DriverError(
                f"the run stopped at segment {completed} of {expected}. No completion manifest "
                f"was written; --resume will continue it.")

        from replica_statistics import lifetime_statistics, round_trip_report
        accepted, proposed = self.reporter.statistics()
        events = self.reporter.reservoir_events()
        mapping = self.reporter.mapping()
        stats = lifetime_statistics(accepted, proposed, tau=self.protocol.tau,
                                    exchange_stride=self.protocol.exchange_stride,
                                    reservoir_events=events)
        trips = round_trip_report(mapping, n_states=self.protocol.n_states)

        record = {
            "format": storage.MANIFEST_FORMAT,
            "run_status": "completed",
            "segments_expected": int(expected),
            "segments_completed": int(completed),
            "iterations_committed": int(self.reporter.last_iteration()) + 1,
            "exchange_attempts_expected": int(expected) // self.protocol.exchange_stride,
            "production_ps_per_replica": self.protocol.production_ps(completed),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "resumed_from_iteration": state.get("resumed_from"),
            "storage": {
                "analysis_netcdf": Path(self.files.trajectory).name,
                "checkpoint_netcdf": Path(self.files.checkpoint).name,
                "run_state": storage.run_state_path(self.files.trajectory).name,
                "schema": storage.SCHEMA_VERSION,
                "authoritative": "analysis_netcdf",
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
        storage.write_atomic(self.files.restart, json.dumps(record, indent=2) + "\n")
        storage.write_run_state(self.files.trajectory, "completed", identity=identity,
                                iteration=state["iteration"],
                                note="manifest written; the storage remains authoritative")
        self.reporter.close()
        return record


def _install_interrupt_handlers():
    """Turn SIGINT and SIGTERM into KeyboardInterrupt, so an interruption is RECORDED.

    Without this a SIGTERM ends the process outright: the storage is still resumable, because the
    reporter writes `last_iteration` after every committed row, but the run-state sidecar would be
    left saying `running` and nothing would say what happened. The loop checkpoints between
    iterations, so the raise lands at an iteration boundary and the last committed row is intact.

    Handlers can only be installed on the main thread; a run driven from elsewhere keeps whatever
    handling it already had rather than failing here.
    """
    import signal

    previous = {}

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.signal(number, stop)
        except (ValueError, OSError):
            pass
    return previous


def _restore_interrupt_handlers(previous):
    import signal

    for number, handler in (previous or {}).items():
        try:
            signal.signal(number, handler)
        except (ValueError, OSError):
            pass


def _stream_seed(base, name):
    value = int(base)
    for byte in str(name).encode("utf-8"):
        value = (value * 1000003 + byte) & 0xFFFFFFFF
    return (value % (2 ** 31 - 1)) or 1


def _encode_rng(generator):
    """A numpy Generator state as JSON-safe data, so a resume continues the same stream."""
    state = generator.bit_generator.state
    return json.loads(json.dumps(state, default=lambda o: int(o) if hasattr(o, "__int__")
                                 else str(o)))


def _decode_rng(state):
    return state


def environment_versions():
    import netCDF4
    import openmm
    record = {"python": platform_module.python_version(),
              "openmm": openmm.version.version,
              "openmm_short": openmm.version.short_version,
              "netCDF4": netCDF4.__version__,
              "numpy": np.__version__}
    for name in ("mdtraj", "mpi4py"):
        try:
            record[name] = __import__(name).__version__
        except ImportError:
            record[name] = None
    # OpenMMTools is deliberately NOT required by generated production code. It is recorded only
    # so a reader can see whether the optional test oracle was even present.
    try:
        import openmmtools
        record["openmmtools_present_but_unused"] = openmmtools.__version__
    except ImportError:
        record["openmmtools_present_but_unused"] = None
    return record
