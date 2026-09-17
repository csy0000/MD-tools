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
import os
import platform as platform_module
import signal
import socket
import sys
from pathlib import Path

import numpy as np

from md_tools.rest2 import identity as hamiltonian_identity
from . import storage
from . import rem_log as rem_log
from . import state_trajectories as state_trajectories
from .rules import (ExchangeContext, NeighbouringExchangeRule, builtin_rule_identity,
                            load_rule)
from .mpi import Coordination
from .core import stream_seed
from .engine import Configuration, ReplicaEngine
from md_tools.rest2 import require_compatible_implementation


class DriverError(RuntimeError):
    """The run cannot proceed as configured."""


class IdentityError(DriverError):
    """The run being continued is not the run that was started."""


class Coordinator(Coordination):
    """The driver's view of the world. A NAME for `md_tools.remd.mpi.Coordination`, not a copy.

    It used to be a copy, and the copy had its own policy:

        try:
            from mpi4py import MPI
        except ImportError:
            return                      # rank 0 of 1, whatever the launcher said

    Under `mpirun -n 8` on a machine with a broken mpi4py that produced eight processes each
    believing it was the only one, all eight driving every state, all eight writing the same
    files. The failure is invisible precisely because it only happens when MPI is broken.

    There is one MPI authority now. This subclass exists so `driver.Coordinator()` keeps working
    for callers, and it adds nothing.
    """

    def __init__(self):
        opened = Coordination.open()
        super().__init__(MPI=opened.MPI, rank=opened.rank, size=opened.size)


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
    exactly the case this exists to catch -- an installed configuration whose velocity was lost.
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
                 identity_extra=None, explicit_cpu=False,
                 prepared=None):
        self.protocol = protocol
        self.files = files
        self.base_system = base_system
        self.topology = topology
        self.solute_indices = [int(i) for i in solute_indices]
        self.excluded_bonds = list(excluded_bonds)
        self.platform_request = platform
        self.precision = precision
        # `--cpu` is the ONLY way a ladder runs on the CPU. Carried here rather than inferred from
        # `platform`, so the record can say a person chose it.
        self.explicit_cpu = bool(explicit_cpu)
        self.rule_path = rule_path
        self.identity_extra = dict(identity_extra or {})
        #: The `LadderPreflight` this run was validated by. Its platform, device and machine
        #: settings are CONSUMED; nothing here re-resolves them. See `_build_platform`.
        self.prepared = prepared

        # THE RESOLVED RESTRAINTS, consumed the same way. `_protocol.py` names the definition
        # file, because that is what a generated helper can carry; which four atoms each restraint
        # holds, at what centre and what force constant, is resolved by the preflight against the
        # collective-variable definition -- and it is what the rungs were actually built with.
        # Without this the record said "restrained by umbrella.yaml" beside an EMPTY restraint
        # list, which understates a biased run in the one file a reader trusts.
        prepared_restraints = tuple(getattr(prepared, "ladder_restraints", ()) or ())
        if prepared_restraints and not getattr(protocol, "umbrella_restraints", ()):
            protocol.umbrella_restraints = tuple(dict(entry) for entry in prepared_restraints)

        self.coordinator = prepared.coordination if prepared is not None else Coordinator()
        self.owned = owned_states(protocol, self.coordinator)
        self.engine = None
        self.reporter = None
        # The N per-state Amber trajectories. Root-only: only the root writes frames, so only the
        # root holds writers, and every other rank leaves this None for the whole run.
        self.trajectories = None
        self.solute_trajectories = None
        #: Per-thermodynamic-state collective-variable series. Root-only, like the trajectories:
        #: only the root holds the gathered configurations every state's row is written from.
        self.cv_states = None
        #: Test-seam bookkeeping: how many times each armed boundary has been crossed.
        self._fault_crossings = {}
        self._audit = None
        self._run_context = {}
        self._acceleration = {}
        self._periodic = bool(base_system.usesPeriodicBoundaryConditions())

    # -- identity ---------------------------------------------------------------------------------

    def scientific_identity(self, rule_identity):
        """Everything a continuation must agree with, fixed before propagation begins."""
        payload = dict(self.protocol.describe())
        payload.update({
            "format": "md-tools-replica-identity/v2",
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

    def run(self, *, resume=False, extend=0, extend_from=None):
        rule, rule_identity = self._load_rule()
        identity = self.scientific_identity(rule_identity)
        started = datetime.datetime.now(datetime.timezone.utc)

        # An out-of-place extension continues its PARENT, not the file named by -x: that one is
        # new and must not exist yet. The existence rule below is about the file being continued.
        continuing = bool(resume or extend) and not extend_from
        if continuing and not Path(self.files.trajectory).exists():
            raise DriverError(
                f"{self.files.trajectory} does not exist, so there is nothing to "
                f"{'extend' if extend else 'resume'}.")

        self._build_platform()
        context = self._run_context
        print(f"# platform           : {context['platform']} device={context['device_index']} "
              f"({context['device_policy']}), precision {context['precision']}")
        print(f"# this process drives: state(s) {context['owned_states']} of "
              f"{self.protocol.n_states}")
        print(f"# host               : {context['hostname']}")
        sys.stdout.flush()

        systems, self._audit = self._rung_systems()

        interruption = _Interruption().install()
        # Read between per-tau equilibration stages as well as in `_loop`.
        self._interruption = interruption
        state = None
        result = None
        try:
            state = self._begin(identity, systems, rule_identity, resume=resume,
                                extend=extend, extend_from=extend_from)
            if state.get("per_tau_interrupted"):
                # Stopped between two per-tau stages, collectively, before the ladder took a
                # step: there is no checkpoint, and the record says so by name.
                return self._record_per_tau_interruption(state, identity)
            self._loop(state, rule, interruption)
            if state.get("interrupted"):
                # A CLEAN, COLLECTIVELY REACHED interruption: every rank left `_loop` together via
                # `coordinator.any_true`, not because one of them raised. It keeps its own distinct
                # resumable status rather than falling into the failure handling below -- this is
                # `_finish`'s sibling, not a failure, and `_finish` is deliberately not reached.
                result = self._record_interruption(state, identity)
            else:
                # Manifest writing belongs INSIDE this try. It used to run after it, unguarded: an
                # exception here on rank 0 left every other rank -- which had already returned
                # from `_finish`'s `if rank != 0: return` above and exited normally -- reporting
                # "completed" while the coordinated run wrote no manifest at all.
                result = self._finish(state, identity, rule_identity, started)
        except BaseException as failure:
            self._fail_closed(state, identity, failure)
            raise
        finally:
            interruption.restore()
            if self.cv_states is not None:
                # Closed on EVERY exit -- completion, interruption and failure alike. A series
                # left open on the failure path would lose whatever the last buffer held, which is
                # exactly the region a person reads to find out what went wrong.
                self.cv_states.close()
        return result

    def _rung_systems(self):
        """The N Systems this ladder propagates, and the force audit describing them.

        CONSUMED from the preflight when it prepared them -- one System per rung, built before any
        output existed and already validated there. Rebuilding them here was the defect: the
        preflight validated ONLY the top rung, so an unclassifiable force in an intermediate rung
        surfaced at this point, with every output file already created.

        No clone. These are handed straight to the engine, so the Systems propagated are the exact
        objects the preflight audited; a copy would be one more construction that could differ
        from what was checked, which is the failure mode being removed.
        """
        prepared_rungs = tuple(getattr(self.prepared, "rung_systems", ()) or ())
        if not prepared_rungs:
            # No prepared plan: a direct caller that constructed this object itself. Same
            # function, same Systems -- `Protocol.build_systems` delegates to the one the
            # preflight uses, so this branch cannot build a different ladder.
            return self.protocol.build_systems(
                self.base_system, self.solute_indices, self.excluded_bonds)
        if len(prepared_rungs) != self.protocol.n_states:
            raise DriverError(
                f"the preflight prepared {len(prepared_rungs)} rung System(s) and this protocol "
                f"describes {self.protocol.n_states} states. The plan and the ladder must be the "
                f"same ladder; neither is inferred from the other.")
        return list(prepared_rungs), getattr(self.prepared, "force_audit", None)

    def _open_cv_states(self, *, committed_rows=0, committed=None):
        """Open one collective-variable series per thermodynamic state, on the root.

        Root-only for the same reason the trajectories are: only the root holds the gathered
        configurations from which every state's row is written. A non-root rank opening these
        would be N processes writing one set of paths.
        """
        if not self.coordinator.is_root:
            return None
        interval = getattr(self.protocol.schedule, "cv_steps", None)
        if not interval:
            return None
        definition = getattr(self.prepared, "cv_definition", None)
        if definition is None:
            return None
        from .cv_states import StateCVSet

        return StateCVSet(
            Path(self.files.trajectory).parent, definition,
            taus=self.protocol.tau, interval_steps=int(interval),
            fingerprint=getattr(self.prepared, "fingerprint", None),
        ).open(committed_rows=int(committed_rows), committed=committed)

    def _cv_prefix_record(self):
        """The committed CV prefix for this generation, or None when reporting is disabled."""
        if self.cv_states is None:
            return None
        from .cv_states import prefix_records

        return prefix_records(
            Path(self.files.trajectory).parent, self.cv_states.definition,
            taus=self.protocol.tau, interval_steps=int(self.cv_states.interval_steps),
            rows=self.cv_states.rows_written(), cost=self.cv_states.cost(),
            per_state_cost=[series.cost() for series in self.cv_states.series])

    def _continue_cv_states(self, checkpoint):
        """Validate the per-state CV series against this run, then reopen at the committed count.

        Returns None when this run has no CV reporting enabled -- in which case a series left by
        an EARLIER, CV-enabled run of the same ladder is not continued and not silently extended;
        it is stale output, and the overwrite policy owns it.
        """
        interval = getattr(self.protocol.schedule, "cv_steps", None)
        definition = getattr(self.prepared, "cv_definition", None)
        if not interval or definition is None:
            return None

        from .cv_states import CVContinuationError, validate_for_continuation

        extra = checkpoint.get("extra") or {}
        directory = Path(self.files.trajectory).parent
        try:
            # Structure and identity first -- files present, columns and sidecars agreeing with
            # this run's definition -- then the committed prefix itself, by digest. Both are
            # read-only: a continuation that has already truncated cannot decide afterwards that
            # it should have refused.
            validate_for_continuation(
                directory, definition, taus=self.protocol.tau,
                interval_steps=int(interval), committed_rows=extra.get("cv_rows"))
            from .cv_states import truncate_to, validate_prefixes

            validated = validate_prefixes(
                directory, definition, taus=self.protocol.tau,
                interval_steps=int(interval), block=extra.get("cv_prefix"),
                # The checkpoint's OWN progress, so the committed count is checked against
                # something the record being checked cannot influence. Without it a coherent
                # but too-short prefix truncated committed observations and resumed past them.
                committed_step=checkpoint.get("step"),
                committed_rows=extra.get("cv_rows"))
        except CVContinuationError as refusal:
            raise storage.StorageError(str(refusal)) from None
        # Only now is anything cut: everything past the committed prefix belongs to steps whose
        # dynamics are about to be repeated. The count comes from the VALIDATED object, not from
        # the raw block -- the two used to be read separately, so the number that cut the files
        # was not necessarily the number that had been checked.
        truncate_to(directory, taus=self.protocol.tau, rows=validated.rows)
        # Rows AND cost, together, and from that same validated object. Restoring rows alone left
        # a resumed ladder whose series was correct and whose cumulative counters had silently
        # reset to this segment's work.
        from .cv_states import committed_prefixes

        return self._open_cv_states(
            committed=committed_prefixes(validated, self.protocol.n_states))

    def _cv_manifest(self, state):
        """The completion manifest's record of this ladder's CV series, or None if disabled."""
        if self.cv_states is None:
            return None
        from .cv_states import manifest_entries

        return manifest_entries(
            Path(self.files.trajectory).parent, self.cv_states.definition,
            taus=self.protocol.tau,
            interval_steps=int(self.cv_states.interval_steps),
            total_steps=int(state["schedule"].total_steps),
            cost=self.cv_states.cost())

    def _fail_closed(self, state, identity, failure):
        """Report what happened if that is possible, and stop the whole communicator regardless.

        THE ORDER, AND WHY IT IS THIS ORDER

            By the time this runs, every other rank is blocked in a collective this one will never
            reach, so the clock on a hung job is already running. The per-rank report is worth a
            bounded attempt -- it is the only record of WHY the run stopped -- but it is never
            worth the job. So the report is best-effort and wrapped, and the abort happens whether
            it succeeded, failed, or was skipped entirely.

            The wrapping is not defensive habit. A second failure while writing the report is the
            LIKELY case, not the exotic one: a full disk or a stalled filesystem is often the
            reason the first failure happened, and both of those surface again here. Letting that
            escape would skip the abort and leave the communicator waiting on a rank that has
            already given up -- turning a stopped job into a hung one at exactly the moment the
            code was trying to be helpful.

        Only the root writes the run-state record (it owns that file); every rank aborts.
        """
        if self.coordinator.is_root:
            try:
                # This record REPLACES whatever run state exists, so writing one without the
                # migration history would erase the only note that a legacy file had been
                # migrated -- from the very path that runs when something went wrong.
                if isinstance(state, dict):
                    history = list(state.get("storage_migrations") or [])
                else:
                    # `_begin` never returned, so there is no in-memory history to use.
                    history = self._recover_history_readonly()
                if history is not None:
                    interrupted = isinstance(failure, KeyboardInterrupt)
                    # `_begin` may never have returned, so `state` is not always a dict here --
                    # which is also why an absent `extends` is read as "not an extension" rather
                    # than as unknown: the in-place advice is the safe default, and the only run
                    # that gets the extension advice is one whose state proves it is one.
                    extending = isinstance(state, dict) and bool(state.get("extends"))
                    storage.write_run_state(
                        self.files.trajectory, "interrupted" if interrupted else "failed",
                        identity=identity,
                        reason=f"{type(failure).__name__}: {failure}"[:400],
                        storage_migrations=history,
                        note=("the analysis NetCDF holds every committed record, but "
                              + self._EXTENSION_REDO + "; no completion manifest exists")
                        if extending else
                        ("the analysis NetCDF holds every committed record and this run can "
                         "be continued with --resume; no completion manifest exists"))
                # If it could not be read, the existing run state is left exactly as it is. An
                # incomplete claim about provenance is worse than a slightly stale true one, and
                # the exception the caller re-raises is the thing that actually needs reporting.
            except BaseException as while_reporting:        # noqa: BLE001 - never masks the abort
                print(f"[rank {self.coordinator.rank}/{self.coordinator.size}] could not record "
                      f"the failure state: "
                      f"{type(while_reporting).__name__}: {while_reporting}",
                      file=sys.stderr, flush=True)

        if self.coordinator.size > 1:
            # ANY unexpected exception in a plural launch -- during Context construction,
            # propagation, energy evaluation, exchange, trajectory reporting,
            # checkpointing, or manifest writing -- invokes the ONE shared communicator
            # fail/abort authority. It used to just `raise`: the exception unwound out of `run()`
            # untouched, and whatever called it converted it into a local integer status code with
            # no `Abort()` anywhere on that path. A rank that fails inside `_begin`, before the
            # barrier at the end of its non-continuation branch, shows why that is not good
            # enough: every OTHER rank is already blocked in that barrier waiting for a
            # participant that will never call it again, and nothing rescues them until something,
            # somewhere, decides the run failed and calls `Abort()`.
            #
            # `fail` prints the reason from the rank that has it, calls `Abort`, and does not
            # return -- so the caller's `raise` is the serial path's behaviour, unchanged.
            self.coordinator.fail(f"{type(failure).__name__}: {failure}")

    # -- setup ------------------------------------------------------------------------------------------

    def _load_rule(self):
        """The transition rule this ladder runs under: an explicit `--exchange-rule`, or the
        neighbouring-pair Metropolis rule REST2 is defined by."""
        if self.rule_path:
            return load_rule(self.rule_path)
        rule = NeighbouringExchangeRule()
        return rule, builtin_rule_identity(rule)

    def _build_platform(self):
        """CONSUME the platform the preflight resolved. Do not resolve a second one.

        This function used to reload `machine_openmm_settings()` and re-run the whole platform
        and device decision, here, after `solute.yaml`, `_protocol.py`, the group file and the
        logs already existed. That is a second authority for a policy that had already run: the
        two agreed only for as long as nobody edited one of them, and the one that decided where
        the ladder actually ran was this one -- the one no preflight refusal could reach.

        Before that it read `"CUDA" if "CUDA" in available else "CPU"`, so a ladder fell silently
        onto the CPU where a stage refused. There is no fallback in either direction now, and
        there is no second resolver either.
        """
        if self.prepared is None:
            raise DriverError(
                "this ladder was constructed without a validated preflight result. The platform, "
                "the device and the machine settings are decided by `preflight_ladder` before any "
                "output exists, and consumed here; resolving them again at this point would put "
                "the decision after the files it is supposed to guard.")

        prepared = self.prepared
        self._platform = prepared.acceleration.platform
        self._properties = prepared.acceleration.properties
        self._acceleration = prepared.record()
        self._run_context = {
            "acceleration": self._acceleration,
            "platform": prepared.platform_name,
            "device_index": prepared.device_index,
            "device_policy": prepared.device_policy_detail,
            "precision": self._properties.get("Precision"),
            # THE CPU THREAD COUNT, when this ran on the CPU platform. OpenMM's CPU platform
            # sums its force reductions in thread-completion order, so it is reproducible only
            # at a fixed pool size: two runs of the same ladder, same seed, same inputs, on the
            # same machine, diverge by the first observation when the pool differs. Without this
            # field nothing in the record says whether two runs were even comparable, and a
            # divergence between them looks like a defect in the ladder rather than a difference
            # in how many threads OpenMM was given. Absent on CUDA, which has no such property.
            "cpu_threads": self._cpu_threads(),
            "mpi_rank": self.coordinator.rank, "mpi_size": self.coordinator.size,
            "hostname": socket.gethostname(), "owned_states": list(self.owned),
        }

    def _cpu_threads(self):
        """The resolved CPU thread pool size, or None when this is not the CPU platform.

        Asked of the PLATFORM rather than read from `OPENMM_CPU_THREADS`: the environment
        variable is one of several inputs OpenMM resolves, and the resolved value is the one
        that determines whether two runs can be compared.
        """
        if self._platform is None or str(self._platform.getName()) != "CPU":
            return None
        try:
            if "Threads" not in self._platform.getPropertyNames():
                return None
            return int(self._properties.get("Threads")
                       or self._platform.getPropertyDefaultValue("Threads"))
        except Exception:      # noqa: BLE001 - a missing property is not a run failure
            return None

    def _begin(self, identity, systems, rule_identity, *, resume, extend,
               extend_from=None):
        seed = int(self.protocol.random_seed or 20260830)
        self.engine = ReplicaEngine(
            self.protocol, systems, self.topology, owned=self.owned,
            platform=self._platform, properties=self._properties, seed=seed)

        if extend_from:
            return self._extend_from(identity, Path(extend_from), extend=extend)
        if resume or extend:
            return self._continue(identity, extend=extend)

        start = self._read_initial_configuration()
        configurations = [start.copy() for _ in range(self.protocol.n_states)]
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "initialized", identity=identity,
                # A file this run created needs no migration; empty is the truthful record.
                storage_migrations=[],
                
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
            # Created here, with the reporter, because the two are one commit: the marker in the
            # authoritative record counts rows in these files, so neither may exist without the
            # other.
            self.trajectories = state_trajectories.StateTrajectorySet.create(
                Path(self.files.trajectory).parent,
                taus=self.protocol.tau, n_atoms=configurations[0].n_atoms,
                temperature_k=self.protocol.temperature_k, periodic=self._periodic,
                program_version=environment_versions().get("md_tools", "0"))
            # THE SOLUTE SET, one AMBER trajectory per state, on its own schedule.
            #
            # It replaces a single `<run>.solute.nc` whose `solute_positions` had a WALKER axis --
            # (frame, walker, atom, xyz). That shape has no place in the AMBER convention, so no
            # standard reader opened it: mdtraj looks for `coordinates` and found none. The data
            # was there and unreachable without writing a bespoke reader for it.
            #
            # Same set, same commit, same mapping: after an accepted exchange the configuration
            # now in state 2 is written to state 2's solute file exactly as it is to state 2's
            # whole file, so the two streams cannot disagree about which state they describe.
            self.solute_trajectories = state_trajectories.StateTrajectorySet.create(
                Path(self.files.trajectory).parent,
                taus=self.protocol.tau, n_atoms=configurations[0].n_atoms,
                temperature_k=self.protocol.temperature_k, periodic=self._periodic,
                program_version=environment_versions().get("md_tools", "0"),
                atom_indices=list(self.solute_indices), content="solute")
            self.cv_states = self._open_cv_states()
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
        if self._equilibrate_per_tau(state, systems):
            return state
        self._equilibrate(state)
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "running", identity=identity,
                # A file this run created needs no migration; empty is the truthful record.
                storage_migrations=[],
                
                total_steps=self.protocol.total_steps,
                note="propagation started; the analysis NetCDF is authoritative for progress")
        return state

    def _recover_history_readonly(self):
        """Best effort, read-only: the cumulative history from every authoritative record.

        Returns None when it cannot be read. That is not the same as "there is none", and the
        caller must treat it differently: replacing a truthful run state with an empty claim is
        worse than leaving the old record in place.
        """
        try:
            probe = storage.ReplicaReporter(self.files.trajectory, mode="r")
            try:
                file_history = probe.migration_history()
            finally:
                probe.close()
            return storage.merge_migration_histories(
                file_history,
                storage.migration_history_of(self._previous_manifest()),
                storage.migration_history_of(storage.read_run_state(self.files.trajectory)))
        except Exception:                                   # noqa: BLE001 - see the docstring
            # Deliberately swallowed. This runs inside an exception handler, and a failure to read
            # provenance must never replace the exception that brought us here.
            return None

    def read_only_probe(self):
        """Everything the continuation needs to know before it may write anything.

        Extracted so it can be exercised directly: the ordering here IS the safety property, and a
        test that only searched the source for a keyword would not notice it moving.

        The pending record is validated STRICTLY, read-only, and then handed to the field
        inspection. Without it the inspection cannot tell an unfinished transaction from a corrupt
        file -- a migration killed after `createVariable` leaves the field physically present
        holding HDF5 fill values, and inspecting it blind classified that as malformed and refused.
        Recovery then depended on whether the unsynced variable had happened to reach disk. It is
        the same validator `--verify-only` and the reconciliation use: one contract, not three.
        """
        probe = storage.ReplicaReporter(self.files.trajectory, mode="r")
        try:
            pending = probe.validated_pending_migration()
            return {
                "identity": probe.identity,
                "committed": probe.last_exchange(),
                "pending": pending,
                "optional_fields": probe.inspect_optional_exchange_fields(pending=pending),
                "file_history": probe.migration_history(),
            }
        finally:
            probe.close()

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

    #: The one name an extension assumes. Everything else it needs is READ OUT of this file:
    #: a completion manifest records the analysis and checkpoint names it was written with, and
    #: those are the truth. Assuming them instead would work only for runs whose outputs happen
    #: to be named the way the documentation example names them -- the generated ladders call
    #: theirs `rest2.nc` and `rest2_checkpoint.nc` -- and would fail confusingly for the rest.
    PARENT_MANIFEST = "restart.json"

    @classmethod
    def parent_files(cls, parent):
        """The parent's manifest, and the analysis and checkpoint files it names."""
        parent = Path(parent)
        manifest_path = parent / cls.PARENT_MANIFEST
        if not manifest_path.is_file():
            raise DriverError(
                f"{parent} holds no {cls.PARENT_MANIFEST}, so it is not a completed run. An "
                f"extension continues a finished parent; a run that never wrote a completion "
                f"manifest is resumed in place with --resume instead.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = manifest.get("storage") or {}
        analysis = names.get("analysis_netcdf")
        checkpoint = names.get("checkpoint_netcdf")
        if not analysis or not checkpoint:
            raise DriverError(
                f"{manifest_path} does not record which files it was written with "
                f"(storage.analysis_netcdf / storage.checkpoint_netcdf). Guessing conventional "
                f"names here could open a different run's storage, so it is refused.")
        return manifest, {"manifest": manifest_path, "analysis": parent / analysis,
                          "checkpoint": parent / checkpoint}

    @classmethod
    def validate_extension_parent(cls, parent):
        """Every READ-ONLY reason to refuse extending `parent`, established from the parent alone.

        Returns `(manifest, files, checkpoint, stored_identity)`. Separate from `_extend_from`
        because the runtime has to ask it BEFORE `-odir` holds anything: `replica_main` publishes
        `_protocol.py`, `solute.yaml` and opens the `.out`/`.log` before the driver runs, so a
        parent refused only here left a new directory that read as a started extension. The one
        check not made here is the comparison with THIS run's identity, which needs the prepared
        ladder and stays in `_extend_from`.
        """
        from . import validate as replica_validate

        parent = Path(parent)
        manifest, files = cls.parent_files(parent)
        for name in ("analysis", "checkpoint"):
            if not files[name].is_file():
                raise DriverError(
                    f"{files['manifest']} names {files[name].name} as its {name}, and that "
                    f"file is not in {parent}. The parent is incomplete: refused read-only, "
                    f"before anything here is created.")

        # The parent's CV series must still be what its manifest says. An extension
        # concatenates onto that column, so a parent whose series was truncated or edited
        # after completion produces an extended series that is the old measurement joined to
        # a new one, with nothing in the file marking the join. Read-only, and before
        # anything local exists.
        if manifest.get("collective_variables") is not None:
            from .cv_states import verify_manifest_entries

            damaged = verify_manifest_entries(Path(parent),
                                              manifest["collective_variables"])
            if damaged:
                raise DriverError(
                    f"{parent} cannot be extended: its collective-variable output no longer "
                    f"matches the completion manifest that describes it.\n  - "
                    + "\n  - ".join(damaged))

        # PHASE 1 -- READ ONLY, and complete. Nothing local exists yet, so a refusal here
        # leaves no half-created extension directory behind to be mistaken for a run.
        check = replica_validate.validate_replica_output(
            analysis=str(files["analysis"]), checkpoint=str(files["checkpoint"]),
            expect_completed=True, reconcilable=False)
        if not check.ok:
            raise storage.StorageError(replica_validate.format_report(
                check, title=f"{parent} cannot be extended"))

        if manifest.get("run_status") != "completed":
            raise DriverError(
                f"{files['manifest']} records run_status "
                f"{manifest.get('run_status')!r}, not 'completed'. An extension continues a "
                f"finished parent; an unfinished one is resumed in place instead, with "
                f"--resume, so that its own budget is met before anything is added to it.")

        stored = manifest.get("scientific_identity")
        if stored is None:
            raise IdentityError(
                f"{files['manifest']} carries no scientific identity, so what the parent was "
                f"created with cannot be established. Refusing rather than continuing a "
                f"Hamiltonian that cannot be checked.")
        require_compatible_implementation(
            (stored.get("rest2_implementation") or {}),
            what=f"the parent run in {parent}")

        checkpoint = storage.ReplicaCheckpoint(str(files["checkpoint"])).read()
        if int(checkpoint["step"]) != int(manifest["steps_completed"]):
            raise DriverError(
                f"{parent} is inconsistent: the completion manifest says "
                f"{manifest['steps_completed']} steps but the terminal checkpoint holds "
                f"{checkpoint['step']}. The two must agree before anything continues from "
                f"either.")

        return manifest, files, checkpoint, stored

    def _extend_from(self, identity, parent, *, extend):
        """Continue a COMPLETED parent into a NEW output set, leaving the parent untouched.

        This is not `--resume`, which reopens the parent's own files for append. Here the parent
        is opened read-only, every reason to refuse is established before anything local is
        created, and the new segment is written to its own storage. What carries across the
        boundary is the physical state -- positions, velocities, box, the mapping, the RNG, the
        rule's state, the absolute step and the exchange count -- and nothing else.

        The parent is never opened for writing on any path in this method. That is the whole
        contract: an extension that could modify its parent is not an extension, it is an
        in-place append with extra directories.
        """
        payload = None
        if self.coordinator.is_root:
            manifest, files, checkpoint, stored = self.validate_extension_parent(parent)
            differences = self.compare_identity(stored, identity)
            if differences:
                raise IdentityError(
                    "the extension does not describe the same run as its parent, so it would not "
                    "be a continuation of it. These differ:\n  - " + "\n  - ".join(
                        f"{key}: parent {stored.get(key)!r} vs here {identity.get(key)!r}"
                        for key in differences))

            inherited = {
                "path": str(parent.resolve()),
                "manifest": {"name": files["manifest"].name,
                             "sha256": _sha256_of(files["manifest"])},
                "checkpoint": {"name": files["checkpoint"].name,
                               "sha256": _sha256_of(files["checkpoint"])},
                "analysis": {"name": files["analysis"].name,
                             "sha256": _sha256_of(files["analysis"])},
                "state_trajectories": _state_trajectory_digests(
                    parent, n_states=self.protocol.n_states),
                "exchanges_committed": int(manifest["exchanges_committed"]),
                "steps_completed": int(checkpoint["step"]),
                "whole_frames": int(manifest["whole_frames"]),
                "production_ps": float(manifest["production_ps_per_replica"]),
                "completion_report": manifest.get("completion_report"),
                "versions": manifest.get("versions"),
                "execution": manifest.get("execution"),
                "storage_migrations": list(manifest.get("storage_migrations") or []),
            }

            # PHASE 2 -- the parent has been accepted, so now the LOCAL set is created. The
            # schedule is absolute: its total is the parent's plus the new segment, so steps and
            # times in this directory carry on from the parent's terminal values rather than
            # restarting at zero.
            already = int(checkpoint["schedule"]["number_of_exchanges"])
            schedule = self.protocol.schedule.extended(
                already + int(extend) - self.protocol.schedule.number_of_exchanges)

            storage.write_run_state(
                self.files.trajectory, "initialized", identity=identity,
                storage_migrations=[], total_steps=schedule.total_steps,
                note=f"extension of {parent}; no new propagation has happened yet")
            self.reporter = storage.ReplicaReporter.create(
                self.files.trajectory, n_states=self.protocol.n_states,
                n_atoms=checkpoint["configurations"][0].n_atoms,
                n_solute_atoms=len(self.solute_indices),
                has_box=checkpoint["configurations"][0].box is not None,
                identity=identity,
                metadata={"created_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                    "versions": environment_versions(),
                    "execution": self._run_context,
                    "exchange_rule": identity.get("exchange_rule"),
                    "extends": json.dumps(inherited, default=str)})
            self.reporter.write_ladder(self.protocol.tau)
            self.trajectories = state_trajectories.StateTrajectorySet.create(
                Path(self.files.trajectory).parent,
                taus=self.protocol.tau,
                n_atoms=checkpoint["configurations"][0].n_atoms,
                temperature_k=self.protocol.temperature_k, periodic=self._periodic,
                program_version=environment_versions().get("md_tools", "0"))
            # The solute set travels with the whole set at EVERY site that makes one -- fresh,
            # extension, and continuation. Creating it in only one of the three is how a resumed
            # or extended ladder reaches `write_frame` on an attribute that is still None.
            self.solute_trajectories = state_trajectories.StateTrajectorySet.create(
                Path(self.files.trajectory).parent,
                taus=self.protocol.tau,
                n_atoms=checkpoint["configurations"][0].n_atoms,
                temperature_k=self.protocol.temperature_k, periodic=self._periodic,
                program_version=environment_versions().get("md_tools", "0"),
                atom_indices=list(self.solute_indices), content="solute")

            print(f"# extending          : {parent}")
            print(f"#   parent           : {inherited['steps_completed']} step(s), "
                  f"{inherited['exchanges_committed']} exchange(s), "
                  f"{inherited['production_ps']} ps, left unchanged")
            print(f"#   this segment adds : {int(extend)} exchange(s) to step "
                  f"{schedule.total_steps}")
            sys.stdout.flush()

            payload = {
                "extends": inherited,
                "storage_migrations": [],
                "step": int(checkpoint["step"]),
                "exchange_index": -1,
                "frame_index": -1,
                "solute_frame_index": -1,
                "parent_exchange_index": int(checkpoint["exchange_index"]),
                "state_to_walker": list(checkpoint["state_to_walker"]),
                "rng": checkpoint["rng_states"]["exchange"],
                "rule_state": dict(checkpoint["rule_state"]),
                "total_steps": schedule.total_steps,
                "number_of_exchanges": schedule.number_of_exchanges,
                "configurations": [(c.positions, c.velocities, c.box)
                                   for c in checkpoint["configurations"]],
                "context_blobs": checkpoint.get("context_blobs"),
                "context_platform": checkpoint.get("context_platform"),
            }
        payload = self.coordinator.bcast(payload)

        rng = np.random.default_rng()
        rng.bit_generator.state = payload["rng"]
        schedule = self.protocol.schedule
        if schedule.number_of_exchanges != payload["number_of_exchanges"]:
            schedule = schedule.extended(
                payload["number_of_exchanges"] - schedule.number_of_exchanges)

        # The local counters start at -1: this storage holds the NEW segment only. The parent's
        # counts are carried alongside, not added in, so a segment-local figure stays a
        # segment-local figure and the chain total is an explicit sum rather than an accident.
        state = {
            "step": payload["step"], "exchange_index": -1,
            "frame_index": -1, "solute_frame_index": -1,
            "state_to_walker": list(payload["state_to_walker"]),
            "configurations": [Configuration(p, v, b) for p, v, b in payload["configurations"]],
            "restore_contexts": self._restorable_contexts(payload),
            "announce_restore": True,
            "schedule": schedule,
            "rng": rng, "rule_state": dict(payload["rule_state"]),
            "resumed_from_step": payload["step"], "interrupted": False,
            "storage_migrations": [],
            "extends": payload["extends"],
            "parent_exchange_index": payload["parent_exchange_index"],
        }
        self.coordinator.agree(configuration_digest(state["configurations"]),
                               what="the extended configurations")
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "running", identity=identity,
                total_steps=schedule.total_steps, resumed_from_step=int(state["step"]),
                storage_migrations=[],
                note=(f"extension of {payload['extends']['path']} started; that parent is "
                      f"complete and is not written to"))
        return state

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
            from . import validate as replica_validate

            check = replica_validate.validate_replica_output(
                analysis=self.files.trajectory, checkpoint=self.files.checkpoint,
                expect_completed=False,
                # A pending migration is what this path exists to finish. Treating it as a reason
                # to refuse would deadlock the only route that can reconcile it.
                reconcilable=True)
            if not check.ok:
                raise storage.StorageError(
                    replica_validate.format_report(
                        check, title="this run cannot be continued"))

            # PHASE 1 -- READ ONLY. Every reason to refuse is established while the file is open
            # for reading and nothing can be written. A file must never be modified merely because
            # it could be opened, and the optional-field migration below is a modification.
            probed = self.read_only_probe()
            stored = probed["identity"]
            committed = probed["committed"]
            pending = probed["pending"]
            optional_fields = probed["optional_fields"]
            file_history = probed["file_history"]

            if pending:
                print(f"# pending migration   : transaction {pending['transaction_id'][:12]} for "
                      f"{pending['fields']} is unfinished and will be reconciled before any step "
                      f"is propagated")
                sys.stdout.flush()

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

            # The state trajectories are continued from the SAME marker the reporter was rewound
            # to, so the two records agree about how many frames exist before a single new step is
            # propagated. Rows past it were written before a crash moved the marker: they are
            # uncommitted, and they are overwritten from here on.
            committed_frames = int(checkpoint["frame_index"]) + 1
            self.trajectories = state_trajectories.StateTrajectorySet.continue_from(
                Path(self.files.trajectory).parent,
                taus=self.protocol.tau,
                n_atoms=int(checkpoint["configurations"][0].n_atoms),
                committed_frames=committed_frames)
            # The solute set travels with the whole set at EVERY site that makes one -- fresh,
            # extension, and continuation. Creating it in only one of the three is how a resumed
            # or extended ladder reaches `write_frame` on an attribute that is still None.
            self.solute_trajectories = state_trajectories.StateTrajectorySet.continue_from(
                Path(self.files.trajectory).parent,
                taus=self.protocol.tau,
                n_atoms=int(checkpoint["configurations"][0].n_atoms),
                committed_frames=int(checkpoint["solute_frame_index"]) + 1,
                atom_indices=list(self.solute_indices), content="solute")

            # THE CV SERIES, reopened at the count the checkpoint vouches for.
            #
            # `_continue` used to leave `cv_states` as None, so a resumed ladder wrote no CV rows
            # at all after the interruption: the run completed, every other stream continued
            # correctly, and the CV files silently stopped at the crash. Validated read-only
            # FIRST -- a continuation that has already truncated cannot decide afterwards that it
            # should have refused -- then opened at the committed count, which truncates whatever
            # was written after the last commit and appends from there.
            self.cv_states = self._continue_cv_states(checkpoint)

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
                "context_blobs": checkpoint.get("context_blobs"),
                "context_platform": checkpoint.get("context_platform"),
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
            "restore_contexts": self._restorable_contexts(payload),
            "announce_restore": True,
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

    def _equilibrate_per_tau(self, state, systems):
        """`rest2.equilibration_per_tau`: the equilibration stages on every rung, under its own tau.

        Runs on a fresh start only, before `_equilibrate` and before the step-0 observation, so the
        first CV row and the first exchange both see each rung's own equilibrated state. Resume and
        extension never repeat it. What each stage does is `rung_equilibration.run_stage` -- the
        function an exported bundle calls too -- on a restrained COPY of the rung: the Systems in
        `systems` are the ones the ladder propagates and are never modified.

        Stage by stage across the rungs this process owns, so every rank reaches the same
        boundaries and an interruption can stop them together. Returns True when it did.
        """
        plan = getattr(self.protocol, "per_tau_equilibration", None)
        if not plan:
            return False
        from openmm.app import PDBFile

        from .rung_equilibration import (RECORD_NAME, RESTRAINT_PARAMETER, handoff_name,
                                         restrained_clone, run_stage)

        # The seed every stream of this ladder is derived from; see `_begin`.
        seed = int(self.protocol.random_seed or 20260830)
        # The stage chain restrains towards the topology's coordinates (`-p`), not towards the
        # starting state, and so does this.
        reference = PDBFile(str(self.files.topology)).positions
        print("# per-tau equilibration: "
              + ", ".join(f"{stage['name']} {stage['steps']} step(s)" for stage in plan)
              + " on every rung, under its own tau, NOT counted as production")
        sys.stdout.flush()

        clones = {index: restrained_clone(systems[index], reference, self.solute_indices)
                  for index in self.owned}
        current = {index: state["configurations"][index] for index in self.owned}
        records = {index: [] for index in self.owned}
        finals = {}
        interruption = getattr(self, "_interruption", None)
        for stage in plan:
            if self.coordinator.any_true(bool(interruption is not None
                                              and interruption.requested)):
                state.update(per_tau_interrupted=True, per_tau_stage=stage["name"],
                             interrupt_signal=getattr(interruption, "signal", None))
                return True
            self._fail_at("per-tau-equilibration")
            for index in self.owned:
                current[index], record, finals[index] = run_stage(
                    clones[index], current[index], stage, state_index=index, seed=seed,
                    temperature_k=self.protocol.temperature_k,
                    friction_per_ps=self.protocol.friction_per_ps,
                    timestep_fs=self.protocol.timestep_fs,
                    constraint_tolerance=self.protocol.constraint_tolerance,
                    platform=self._platform, properties=self._properties)
                records[index].append(record)
            print(f"# per-tau {stage['name']:<16}: {stage['steps']} step(s) on state(s) "
                  f"{self.owned}")
            sys.stdout.flush()
        del clones

        for index in self.owned:
            self.engine.set_configuration(index, current[index])
        state["configurations"] = self._gather_configurations(state)
        boxes = [configuration.box for configuration in state["configurations"]]
        if boxes[0] is not None and any(not np.array_equal(box, boxes[0]) for box in boxes[1:]):
            raise DriverError(
                "after per-tau equilibration the rungs do not share one box. Every per-tau stage "
                "is NVT from the ladder's one starting box, so this cannot come from the physics; "
                "exchanging configurations of different volume would not be REST2, so the ladder "
                "is refused.")

        local = {index: (finals[index], records[index]) for index in self.owned}
        if self.coordinator.size > 1:
            gathered = {}
            for piece in self.coordinator.allgather(local):
                gathered.update(piece)
            local = gathered
        if self.coordinator.is_root:
            directory = Path(self.files.trajectory).parent
            entries = []
            for index in range(self.protocol.n_states):
                text, stages_run = local[index]
                name = handoff_name(index)
                storage.write_atomic(directory / name, text)
                entries.append({"state_index": index, "tau": float(self.protocol.tau[index]),
                                "stages": stages_run, "file": name,
                                "sha256": _sha256_of(directory / name)})
            record = {
                "format": "md-tools-per-tau-equilibration/v1",
                "stages": [dict(stage) for stage in plan],
                "order": ("these stages on every rung under its own tau, then "
                          "equilibration_steps, then the first exchange; none of it production"),
                "started_from": {"path": Path(self.files.coordinates).name,
                                 "sha256": _sha256_of(self.files.coordinates)},
                "restraint": {"atoms": len(self.solute_indices),
                              "reference": f"{Path(self.files.topology).name} coordinates",
                              "parameter": RESTRAINT_PARAMETER},
                "integrator": {"temperature_k": float(self.protocol.temperature_k),
                               "friction_per_ps": float(self.protocol.friction_per_ps),
                               "timestep_fs": float(self.protocol.timestep_fs),
                               "constraint_tolerance": float(
                                   self.protocol.constraint_tolerance)},
                "seed_base": seed,
                "seeds": "derive_seed(seed_base, stage, 'state<i>')",
                "velocities": "carried from the starting state, never redrawn",
                "box_shared": None if boxes[0] is None else True,
                "states": entries,
            }
            storage.write_atomic(directory / RECORD_NAME,
                                 json.dumps(record, indent=2, default=str) + "\n")
        self.coordinator.barrier()
        return False

    def _record_per_tau_interruption(self, state, identity):
        """A clean, collective stop between two per-tau stages. Nothing to resume from."""
        from .rung_equilibration import INTERRUPTED_PHASE

        stage = state.get("per_tau_stage")
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "interrupted", identity=identity, step=0,
                total_steps=self.protocol.total_steps, signal=state.get("interrupt_signal"),
                phase=INTERRUPTED_PHASE, stage=stage, storage_migrations=[],
                note=(f"stopped during per-tau equilibration, before its stage {stage!r} and "
                      f"before the ladder's first step. There is no checkpoint, so --resume "
                      f"refuses this run: rerun with --overwrite. The equilibration is "
                      f"deterministic from the recorded seeds."))
            print(f"# interrupted during per-tau equilibration, before {stage}; no checkpoint "
                  f"exists, so rerun with --overwrite")
            for handle in (self.trajectories, self.solute_trajectories):
                if handle is not None:
                    handle.close()
            if self.reporter is not None:
                self.reporter.close()
        return {"run_status": "interrupted", "step": 0, "phase": INTERRUPTED_PHASE}

    def _per_tau_manifest(self, state):
        """The completion manifest's record of per-tau equilibration, verified, or None if off."""
        if not getattr(self.protocol, "per_tau_equilibration", None):
            return None
        from .rung_equilibration import RECORD_NAME

        directory = Path(self.files.trajectory).parent
        path = directory / RECORD_NAME
        if not path.is_file():
            if state.get("extends"):
                return {"performed_here": False,
                        "why": "an out-of-place extension continues its parent's ladder; the "
                               "per-tau equilibration ran there and is in the parent's manifest"}
            raise DriverError(
                f"this ladder was configured with rest2.equilibration_per_tau, and {path.name} -- "
                f"the record of it -- is not in {directory}. A completion manifest must not "
                f"describe an equilibration it cannot show.")
        record = json.loads(path.read_text(encoding="utf-8"))
        problems = []
        for entry in record.get("states") or []:
            handoff = directory / str(entry.get("file"))
            if not handoff.is_file():
                problems.append(f"{entry.get('file')} is missing")
            elif _sha256_of(handoff) != entry.get("sha256"):
                problems.append(f"{entry.get('file')} no longer has the sha256 recorded for it")
        if problems:
            raise DriverError("the per-tau equilibration record does not verify:\n  - "
                              + "\n  - ".join(problems))
        return dict(record, record_file=RECORD_NAME, record_sha256=_sha256_of(path))

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

    def _reduced_potential_matrix(self, configurations, *, state_to_walker=None):
        """u[i][w]: walker w's sample in state i's Hamiltonian. Rank i computes row i.

        THE DIAGONAL IS NOT RE-EVALUATED. `_gather_configurations` read walker
        `state_to_walker[i]`'s configuration OUT OF state `i`'s own Context, so `u[i][that
        walker]` is the energy of what that Context is already holding. Installing it again to
        measure it is a round trip to the device for a number already in hand -- and it is the one
        Amber takes for free: `hamiltonian_exchange` sets `my_ene_temp%energy_1 = my_pot_ene_tot`
        from the value the last dynamics step already produced, and pays one force call only for
        the CROSS term it names `my pot ene with THEIR coordinates`.

        Not an inference and not a shortcut of the kind `reduced_potential_of` refuses: that
        refusal is about deriving one Hamiltonian's energy from another's by scaling, which is
        wrong because the scale factors differ per term. This is the same configuration in the
        same Hamiltonian, converted by the same `reduced_potential`, so the number is identical
        rather than merely close.

        It is read FIRST, before any cross energy, so it is measured on the Context as propagation
        left it -- never on one that a cross evaluation has saved and restored.

        The matrix stays DENSE. `rem_log.block_rows` indexes `u[state, state]` and
        `u[state, mate]`, `exchange_free_energies` needs both cross terms of every proposed pair,
        and `_write_exchange_csv` indexes `u[state][walker]` -- two different conventions over
        three renderers. Evaluating only what a rule declares would leave holes that some of them
        index, so `required_entries` is recorded as the rules' contract and not used to skip work
        here.
        """
        from .engine import reduced_potential

        n = self.protocol.n_states
        rows = {}
        for index in self.owned:
            own = None if state_to_walker is None else int(state_to_walker[index])
            row = [None] * n
            if own is not None:
                box = configurations[own].box
                volume = None if box is None else float(abs(np.linalg.det(box)))
                row[own] = reduced_potential(
                    self.engine.potential_energy(index), self.protocol.beta,
                    pressure_bar=self.protocol.pressure_bar, volume_nm3=volume)
            for walker in range(n):
                if row[walker] is None:
                    row[walker] = self.engine.reduced_potential_of(
                        index, configurations[walker])
            rows[index] = row
        if self.coordinator.size > 1:
            for piece in self.coordinator.allgather(rows):
                rows.update(piece)
        return np.array([rows[i] for i in range(n)], dtype=float)

    def _install_owned(self, state):
        """Put every owned rung's walker into its context.

        `restore_contexts` is the bit-for-bit path. When the checkpoint carried OpenMM context
        checkpoints AND they were written by the platform this process is running on, loading one
        restores the integrator's pseudo-random stream along with the coordinates, so the
        continuation reproduces the uninterrupted trajectory exactly. Positions and velocities
        alone cannot: they are the complete PHYSICAL state, and the stream position is not part of
        it, so a walker resumed from them is correct but different.

        The fallback is never an error. A checkpoint from another device, or from a build that
        wrote no blobs, resumes from positions and velocities as it always did.
        """
        blobs = state.pop("restore_contexts", None)
        if state.pop("announce_restore", False):
            print(f"# continuation       : {self.context_restore}")
            sys.stdout.flush()
        for index in self.owned:
            if blobs is not None and blobs[index]:
                self.engine.load_integrator_state(index, blobs[index])
                continue
            self.engine.set_configuration(
                index, state["configurations"][state["state_to_walker"][index]])

    def _context_signature(self):
        """What a context checkpoint is bound to. Anything else must not load one.

        OpenMM refuses a checkpoint from a different platform, and a checkpoint from the same
        platform at a different precision is worse than a refusal -- it can load and mean
        something subtly different. Both are compared before a blob is trusted.
        """
        return {
            "platform": str(self._platform.getName()) if self._platform is not None else None,
            "precision": self._properties.get("Precision"),
            "n_states": int(self.protocol.n_states),
        }

    #: How the last continuation restored its walkers: "bitwise: ..." when every rung's OpenMM
    #: context checkpoint was loaded and the trajectory therefore reproduces an uninterrupted
    #: reference exactly, or "physical: <why not>" when it resumed from coordinates alone, which
    #: is correct and statistically exact but is a DIFFERENT trajectory from here on. None on a
    #: fresh run, which continues nothing.
    context_restore = None

    def _restorable_contexts(self, payload):
        """The blobs this process may load, or None when it must resume from coordinates.

        Refusing is the safe answer and costs only bit-for-bit reproducibility, so every doubt --
        no blobs, a different platform, a different precision, a different ladder width, a short
        or empty entry -- resolves to None and the ordinary physical restart.
        """
        blobs = payload.get("context_blobs")
        recorded = payload.get("context_platform") or {}
        mine = self._context_signature()
        if not blobs:
            self.context_restore = "physical: the checkpoint carries no context checkpoints"
        elif recorded != mine:
            self.context_restore = (
                f"physical: the context checkpoints were written under {recorded} "
                f"and this process runs {mine}")
        elif len(blobs) != int(self.protocol.n_states) or not all(blobs):
            self.context_restore = (
                f"physical: {sum(1 for b in blobs if b)} of {self.protocol.n_states} rungs "
                f"carry a context checkpoint")
        else:
            self.context_restore = "bitwise: every rung's OpenMM context checkpoint was restored"
            return list(blobs)
        return None

    def _context_blobs(self, state):
        """Every rung's OpenMM context checkpoint, on every rank, state-indexed.

        COLLECTIVE. Each rank can only serialise the contexts it owns, and the checkpoint is
        written by the root, so the blobs have to be gathered exactly as the configurations are.
        Call it from a point every rank reaches -- never from inside an `is_root` branch.
        """
        local = {index: self.engine.integrator_state(index) for index in self.owned}
        if self.coordinator.size > 1:
            assembled = {}
            for piece in self.coordinator.allgather(local):
                assembled.update(piece)
            local = assembled
        return [local.get(index) for index in range(self.protocol.n_states)]

    #: Test seam: raise a rank-local RuntimeError inside the dynamics loop on named ranks, as
    #: "0,3". This is what a real mid-run failure looks like -- a NaN energy, a device error, a
    #: disk full on a checkpoint write -- and it cannot be provoked reliably any other way in a
    #: real multi-rank launch. Never set in normal use.
    FAIL_PROPAGATION_ENVIRONMENT = "MD_TOOLS_FAIL_PROPAGATION_ON_RANKS"

    #: Test seam: how many times a named boundary may be crossed before it raises. Without it a
    #: fault always lands on the FIRST crossing, so no generation is ever committed and there is
    #: nothing to resume from -- which makes every continuation test untestable.
    FAIL_AFTER_ENVIRONMENT = "MD_TOOLS_FAIL_PROPAGATION_AFTER"

    #: Test seam: which boundary inside the ladder loop raises, as a name. A crash between a CV
    #: row reaching the disk and the checkpoint that vouches for it is the case the committed
    #: counts exist for, and it cannot be provoked any other way in a real launch.
    FAIL_BOUNDARY_ENVIRONMENT = "MD_TOOLS_FAIL_LADDER_AT"
    FAIL_BOUNDARIES = ("propagation", "before-cv-row", "after-cv-row",
                       "before-checkpoint", "after-checkpoint", "per-tau-equilibration")

    def _fail_at(self, boundary):
        """Raise a rank-local failure at a named boundary, if the tests armed it. Never in use."""
        import os

        if boundary == "propagation":
            wanted = os.environ.get(self.FAIL_PROPAGATION_ENVIRONMENT)
            armed = os.environ.get(self.FAIL_BOUNDARY_ENVIRONMENT, "propagation") == "propagation"
        else:
            wanted = os.environ.get(self.FAIL_PROPAGATION_ENVIRONMENT)
            armed = os.environ.get(self.FAIL_BOUNDARY_ENVIRONMENT) == boundary
        if not wanted or not armed:
            return
        if self.coordinator.rank not in {int(part) for part in wanted.replace(",", " ").split()}:
            return
        allowed = int(os.environ.get(self.FAIL_AFTER_ENVIRONMENT) or 0)
        self._fault_crossings[boundary] = self._fault_crossings.get(boundary, 0) + 1
        if self._fault_crossings[boundary] <= allowed:
            return
        raise RuntimeError(
            f"{self.FAIL_PROPAGATION_ENVIRONMENT} names this rank: simulating a rank-local "
            f"failure at {boundary} on rank {self.coordinator.rank} of "
            f"{self.coordinator.size}")

    def _fail_propagation_if_asked(self):
        self._fail_at("propagation")

    def _loop(self, state, rule, interruption):
        schedule = state["schedule"]
        self._install_owned(state)

        # STEP 0, exactly once, before a single step is propagated.
        #
        # The contract is universal: every enabled CV series contains step 0 and the final step
        # exactly once, with the intermediate rows on the declared cadence. This ladder used to
        # observe only at schedule events, and `events_at` never returns anything at step 0, so a
        # 40-step run at interval 5 wrote 5..40 and silently omitted the initial configuration --
        # the one every later row is a displacement from.
        #
        # `exchange_attempt = -1` because no attempt has been made; the mapping is the identity in
        # a fresh run; and `trajectory_frame_index` is empty because no state frame is written at
        # step 0. Guarded on the row count rather than on a "is this a resume" flag, so it is
        # idempotent: a continuation reopens at its committed count, which is already nonzero.
        if (self.cv_states is not None and self.coordinator.is_root
                and self.cv_states.rows_written() == 0):
            self.cv_states.observe(
                step=int(state["step"]), time_ps=schedule.step_to_ps(int(state["step"])),
                exchange_attempt=-1,
                state_to_walker=state["state_to_walker"],
                configurations=state["configurations"],
                frame_index=None)

        while state["step"] < schedule.total_steps:
            target = schedule.next_event_step(state["step"])
            if target is None:
                break
            span = target - state["step"]
            self._fail_propagation_if_asked()
            for index in self.owned:
                self.engine.propagate(index, span)
            state["step"] = target
            state["configurations"] = self._gather_configurations(state)

            events = schedule.events_at(target)
            # BEFORE the exchange, deliberately. `cv` precedes `exchange` in EVENT_ORDER and this
            # call precedes `_exchange` here: a row landing on an exchange boundary describes the
            # configuration the walker actually PROPAGATED to this step, not one that arrived
            # from another rung and was never integrated at this tau. Writing it after the swap
            # would put values from trajectories that never visited a state into that state's
            # series. See `md_tools.remd.cv_states`.
            # THE PRE-EXCHANGE CONFIGURATIONS, frozen here, before `_exchange` can permute the
            # mapping. The row's VALUES and its `walker_index` are the pre-exchange ones -- that
            # convention is unchanged -- but the row is WRITTEN after the exchange, because
            # whether it may name a state trajectory frame is only knowable once the exchange has
            # decided.
            observing = ("cv" in events and self.cv_states is not None
                         and self.coordinator.is_root)
            mapping_before = list(state["state_to_walker"]) if observing else None
            # THE CONFIGURATIONS THEMSELVES, not merely the mapping: a deep copy, so nothing the
            # exchange installs can mutate the positions or box this row is computed from. `Configuration.copy()` is the existing primitive and copies
            # velocities too, which a torsion does not need -- clarity over saving one array on a
            # path that runs once per CV observation, not once per step.
            configurations_before = ([configuration.copy()
                                      for configuration in state["configurations"]]
                                     if observing else None)

            if "exchange" in events:
                self._exchange(state, rule, target, schedule)

            if observing:
                self._fail_at("before-cv-row")
                # Which states may name the frame about to be written, and which may not.
                #
                # The frame is written from the POST-exchange occupant; the CV row describes the
                # PRE-exchange one. For a state the exchange did not move, those are the same
                # configuration. For a state that was swapped, they are different, and naming the
                # frame would attribute a value to coordinates it was not measured on -- which is
                # exactly what happened before: the row named a frame holding another walker's
                # configuration, and the number was plausible, in range, and wrong.
                #
                # Empty is the honest answer there. It says "no frame in this file holds what this
                # row measured", which is true, and it is never -1.
                forthcoming = int(state["frame_index"]) + 1
                named = [
                    (forthcoming
                     if ("whole" in events
                         and mapping_before[index] == state["state_to_walker"][index])
                     else None)
                    for index in range(self.protocol.n_states)]
                self.cv_states.observe(
                    step=target, time_ps=schedule.step_to_ps(target),
                    exchange_attempt=state["exchange_index"],
                    state_to_walker=mapping_before,
                    configurations=configurations_before,
                    frame_index_for_state=named)
                self._fail_at("after-cv-row")

            if "checkpoint" in events:
                # COLLECTIVE, and so deliberately outside the root-only block below: every rank
                # holds only its own rungs' contexts, and gathering them is a collective the root
                # cannot perform alone.
                state["context_blobs"] = self._context_blobs(state)

            if self.coordinator.is_root:
                if "exchange" in events:
                    pass    # already written by _exchange
                if "whole" in events:
                    # One logical commit across N+1 files. The state trajectories take the
                    # row
                    # first and are synced; only then does the marker in the authoritative record
                    # move. A crash between the two leaves rows nothing counts, which a
                    # continuation ignores -- a crash after it leaves a set every file agrees on.
                    self.trajectories.write_frame(
                        step=target, time_ps=schedule.step_to_ps(target),
                        state_to_walker=state["state_to_walker"],
                        configurations=state["configurations"])
                    self.trajectories.sync()
                    promised = int(state["frame_index"]) + 1
                    state["frame_index"] = self.reporter.write_frame(
                        step=target, time_ps=schedule.step_to_ps(target),
                        exchange_index=state["exchange_index"])
                    if observing and int(state["frame_index"]) != promised:
                        # The CV row written a moment ago named `promised`. If the writer
                        # disagrees, a CV value is attributed to a frame holding a different
                        # configuration -- silently, and in a file that reads perfectly.
                        raise DriverError(
                            f"the collective-variable row at step {target} names state trajectory "
                            f"frame {promised}, and the frame just written is "
                            f"{state['frame_index']}. Refusing rather than leaving a CV value "
                            f"attributed to a configuration it was not measured on.")
                if "solute" in events:
                    self.solute_trajectories.write_frame(
                        step=target, time_ps=schedule.step_to_ps(target),
                        state_to_walker=state["state_to_walker"],
                        configurations=state["configurations"])
                    self.solute_trajectories.sync()
                    state["solute_frame_index"] = self.reporter.write_solute_frame(
                        step=target, time_ps=schedule.step_to_ps(target),
                        exchange_index=state["exchange_index"],
                        configurations=state["configurations"],
                        solute_indices=self.solute_indices)
                if "checkpoint" in events:
                    self._fail_at("before-checkpoint")
                    self._write_checkpoint(state, schedule)
                    self._fail_at("after-checkpoint")
                    # Kept in step with the checkpoint: both describe committed rows.
                    self._write_rem_log()
            self.coordinator.barrier()

            # A termination request becomes collective HERE, at an event boundary where every rank
            # is together. One rank raising inside its signal handler would leave the others in a
            # collective forever.
            if self.coordinator.any_true(interruption.requested):
                if "checkpoint" not in events:
                    state["context_blobs"] = self._context_blobs(state)
                if self.coordinator.is_root and "checkpoint" not in events:
                    self._write_checkpoint(state, schedule)
                state["interrupted"] = True
                state["interrupt_signal"] = interruption.signal
                self.coordinator.barrier()
                return

    def _exchange(self, state, rule, step, schedule):
        n = self.protocol.n_states
        matrix = self._reduced_potential_matrix(
            state["configurations"], state_to_walker=state["state_to_walker"])
        state["exchange_index"] += 1

        payload = None
        if self.coordinator.is_root:
            context = ExchangeContext(
                iteration=state["exchange_index"], segment=state["step"], protocol=self.protocol,
                state_to_walker=state["state_to_walker"],
                reduced_potential=lambda i, w: matrix[i][w],
                rng=state["rng"], rule_state=state["rule_state"],
                exchange_index=state["exchange_index"])
            outcome = rule.propose(context)
            payload = {"proposals": outcome.proposals, "swaps": outcome.swaps,
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

        self._install_owned(state)
        self.coordinator.agree(configuration_digest(state["configurations"]),
                               what="the configurations after this exchange")

        if self.coordinator.is_root:
            self.reporter.write_exchange(
                state["exchange_index"], step=step, time_ps=schedule.step_to_ps(step),
                state_to_walker=state["state_to_walker"], proposed=proposed, accepted=accepted,
                # OBSERVED, not asserted. `np.ones(...)` stated that every entry was computed;
                # it happened to be true, but a claim that cannot be wrong also cannot catch a
                # day when it stops being true. `u_evaluated` means "1 where u was actually
                # computed", so it is derived from the matrix itself.
                u=matrix, u_evaluated=np.isfinite(matrix).astype(np.int8))

    def _write_rem_log(self):
        """Regenerate `rem.log` in full from committed exchange rows, and replace it atomically.

        A projection, never an authority: every row comes from `exchange.nc`, so the log cannot
        drift from the record it describes. Rewriting rather than appending is what makes it
        crash-safe -- a torn append would disagree with the record and nothing in the file would
        say which half was true.
        """
        path = getattr(self.files, "rem", None)
        if not path or not self.coordinator.is_root:
            return None
        last = self.reporter.last_exchange()
        if last < 0:
            return None
        # THREE bulk reads, not 2N indexed ones. `statistics` was already a slice; the reduced
        # potentials were not, and gathering them record by record made every regeneration cost
        # O(N) in the history it was regenerating from. The rows built here are identical either
        # way -- `test_rem_log_bulk_read.py` renders both and compares the bytes.
        accepted, proposed = self.reporter.statistics()
        u_history, _ = self.reporter.reduced_potentials_upto(last)
        exchanges = [(proposed[i], accepted[i], u_history[i]) for i in range(last + 1)]
        blocks = rem_log.build(n_states=self.protocol.n_states, exchanges=exchanges,
                               beta=self.protocol.beta,
                               temperature_k=self.protocol.temperature_k)
        written = rem_log.write(path, blocks, remlog_name=Path(path).name)
        self._write_exchange_csv(last, accepted, proposed, u_history)
        return written

    def _write_exchange_csv(self, last, accepted, proposed, u_history):
        """`exchange.csv`: the exchange history as a table, beside `rem.log`.

        A SECOND PROJECTION OF THE SAME AUTHORITY, not a second record. `exchange.nc` remains the
        thing every row is derived from; this file and `rem.log` are two renderings of it, both
        rebuilt in full whenever the log is, so neither can drift from the record or from each
        other.

        Why both. `rem.log` is Amber H-REMD format, which cpptraj reads and a person does not.
        This is the same history in the shape anything else reads without a parser -- pandas,
        awk, a spreadsheet.

        ONE ROW PER STATE PER EXCHANGE, which is the grain the record is kept at. What is NOT
        here is the full N-by-N reduced-potential matrix: that is quadratic in the ladder size and
        is what `exchange.nc` exists to hold. A reweighting reads the NetCDF; this file answers
        "which walker was in which state, and did the swap take".
        """
        import csv as _csv

        path = getattr(self.files, "rem", None)
        if not path or not self.coordinator.is_root:
            return
        target = Path(path).with_name("exchange.csv")
        mapping = self.reporter.mapping(upto=last)
        steps = self.reporter.exchange_steps(upto=last)
        times = self.reporter.exchange_times(upto=last)
        taus = list(self.protocol.tau)

        import io

        buffer = io.StringIO()
        writer = _csv.writer(buffer)
        # `potential_energy_kj_per_mol` beside the reduced potential, because THIS FILE IS THE
        # LADDER'S STATE TABLE and a state table reports energy in energy units. u = beta * U and
        # beta is one number for the whole ladder (every rung is at the same temperature), so the
        # conversion is exact rather than a reconstruction -- but a reader should not have to know
        # that, nor go looking for beta, to answer "what was the energy of state 2".
        #
        # This is why a ladder writes no `mdout.csv`: it would be this table with one column
        # renamed. What `info_printout` governs on a ladder is its equilibration stages, which do
        # write `mdout_<stage>.csv`; the production span is recorded per exchange, here.
        beta = float(self.protocol.beta)
        writer.writerow(["exchange", "step", "time_ps", "state", "tau", "walker",
                         "reduced_potential", "potential_energy_kj_per_mol",
                         "proposed_with_next", "accepted_with_next"])
        for n in range(last + 1):
            for state in range(len(taus)):
                walker = int(mapping[n][state])
                nxt = state + 1
                reduced = float(u_history[n][state][walker])
                writer.writerow([
                    n, int(steps[n]), f"{float(times[n]):.6f}", state, f"{taus[state]:.6f}",
                    walker, f"{reduced:.6f}", f"{reduced / beta:.6f}",
                    int(proposed[n][state][nxt]) if nxt < len(taus) else "",
                    int(accepted[n][state][nxt]) if nxt < len(taus) else ""])
        # Atomic, for the same reason the log is: a reader arriving mid-rewrite must not find a
        # short file that is still valid CSV.
        staging = target.with_name(target.name + ".partial")
        staging.write_text(buffer.getvalue(), encoding="utf-8")
        os.replace(staging, target)

    def _write_checkpoint(self, state, schedule):
        storage.ReplicaCheckpoint(self.files.checkpoint).write(
            context_blobs=state.get("context_blobs"),
            context_platform=self._context_signature(),
            step=state["step"], exchange_index=state["exchange_index"],
            frame_index=state["frame_index"],
            solute_frame_index=state["solute_frame_index"],
            configurations=state["configurations"],
            state_to_walker=state["state_to_walker"],
            rng_states={"exchange": _encode_rng(state["rng"])},
            rule_state=state["rule_state"], schedule=schedule.describe(),
            identity=self.reporter.identity,
            extra={"configuration_digest": configuration_digest(state["configurations"]),
                   # The per-state CV files are coordinated appendable streams, so a generation
                   # has to vouch for their length exactly as it does for the frame index --
                   # otherwise a continuation cannot tell a committed row from one written after
                   # the checkpoint by a process that then died. `rows_written` refuses if the
                   # state files have drifted apart, so one number describes the whole set.
                   "cv_rows": (self.cv_states.rows_written()
                               if self.cv_states is not None else None),
                   # PER STATE: a row count catches a truncation, and a per-state digest catches
                   # a committed row edited in place -- and, unlike one combined hash over all
                   # states, it also catches two states' files being swapped, which leaves any
                   # aggregate unchanged while every series becomes another's.
                   "cv_prefix": self._cv_prefix_record()})

    # -- finishing ------------------------------------------------------------------------------------

    def _extension_provenance(self, state, report):
        """What this segment inherited, and what belongs to it alone.

        Section 14 allows a cumulative figure for the whole chain and requires the segment-local
        counts to stay distinguishable. Both are here, separately, and the chain figure is a
        stated sum of two stated segments rather than a number whose provenance has to be
        reconstructed.
        """
        parent = dict(state["extends"])
        parent_report = parent.get("completion_report") or {}
        parent_overall = parent_report.get("overall") or {}
        segment = report["overall"]

        accepted = int(segment["accepted"]) + int(parent_overall.get("accepted") or 0)
        proposed = int(segment["proposed"]) + int(parent_overall.get("proposed") or 0)
        chain = {"accepted": accepted, "proposed": proposed,
                 "acceptance": (accepted / proposed) if proposed else None,
                 "complete": bool(parent_overall)}
        return {
            "format": "md-tools-extension/v1",
            "parent": parent,
            "segment": {
                "steps": int(state["step"]) - int(parent["steps_completed"]),
                "starts_after_step": int(parent["steps_completed"]),
                "exchanges": int(self.reporter.last_exchange()) + 1,
                "first_exchange_number": int(parent["exchanges_committed"]) + 1,
                "production_ps": (state["schedule"].step_to_ps(state["step"])
                                  - float(parent["production_ps"])),
                "neighbouring_acceptance": segment,
            },
            "chain": {
                "steps": int(state["step"]),
                "exchanges": int(parent["exchanges_committed"])
                             + int(self.reporter.last_exchange()) + 1,
                "production_ps": state["schedule"].step_to_ps(state["step"]),
                "segments": 2 if not parent.get("chain_length") else
                            int(parent["chain_length"]) + 1,
                "neighbouring_acceptance": chain,
            },
        }

    #: What an interrupted OUT-OF-PLACE EXTENSION must be told, and why it is not `--resume`.
    #:
    #: An extension segment is atomic, and atomic by OMISSION rather than by a guard: `_continue`
    #: builds no `extends` key, and nothing reads one back from storage, so a resumed segment
    #: finishes as an ordinary run -- no parent pinning, no segment-local counts, no chain
    #: accounting, and with per-tau equilibration it cannot even write a manifest. No code
    #: refuses that, which is precisely why the advice has to carry the answer instead.
    _EXTENSION_REDO = (
        "an out-of-place extension CANNOT be resumed: a resume carries no `extends`, so this "
        "segment would finish as an ordinary run and lose its parent pinning and its segment "
        "and chain accounting. Re-run the whole segment with --extend-from into a FRESH "
        "directory -- re-running into this one is refused")

    def _interruption_note(self, state):
        """The run-state note for an interruption, which differs by what this run IS."""
        if isinstance(state, dict) and state.get("extends"):
            return ("stopped at an event boundary with a complete checkpoint, but "
                    + self._EXTENSION_REDO + ". No completion manifest exists.")
        return ("stopped at an event boundary with a complete checkpoint; --resume "
                "continues it. No completion manifest exists.")

    def _record_interruption(self, state, identity):
        if self.coordinator.is_root:
            storage.write_run_state(
                self.files.trajectory, "interrupted", identity=identity,
                step=state["step"], total_steps=state["schedule"].total_steps,
                signal=state.get("interrupt_signal"),
                storage_migrations=list(state.get("storage_migrations") or []),
                note=self._interruption_note(state))
            if isinstance(state, dict) and state.get("extends"):
                print(f"# interrupted at step {state['step']} of "
                      f"{state['schedule'].total_steps}; a checkpoint was committed, but this is "
                      f"an out-of-place extension and --resume will NOT continue it: re-run the "
                      f"segment with --extend-from into a fresh directory")
            else:
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
            extending = bool(state.get("extends"))
            storage.write_run_state(
                self.files.trajectory, "interrupted", identity=identity,
                storage_migrations=list(state.get('storage_migrations') or []),
                step=completed,
                note=("stopped before the budget; " + self._EXTENSION_REDO) if extending
                else "stopped before the budget; resumable")
            raise DriverError(
                f"the run stopped at step {completed} of {expected}. No completion manifest was "
                + (f"written, and {self._EXTENSION_REDO}." if extending
                   else "written; --resume will continue it."))

        from .statistics import (completion_report, lifetime_statistics,
                                        mapping_is_permutation_every_iteration)
        accepted, proposed = self.reporter.statistics()
        mapping = self.reporter.mapping()
        stats = lifetime_statistics(accepted, proposed, tau=self.protocol.tau)
        # The permutation check is storage integrity, not analysis: every committed row must be
        # a bijection state->walker, and a row that is not means the mapping was corrupted.
        permuted, offending_rows = mapping_is_permutation_every_iteration(
            mapping, n_states=self.protocol.n_states)
        report = completion_report(stats)

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
            # Section 4: the resolved state records. Deliberately here and not in the scientific
            # identity -- `effective_temperature_k` is derived and reporting-only, and a
            # continuation compares the identity key by key, so a reporting field in it would
            # refuse every file written before the field existed.
            "states": self.protocol.state_records(),
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
            "completion_report": report,
            "mapping_integrity": {
                "rows": int(len(mapping)),
                "every_row_is_a_permutation": bool(permuted),
                "offending_rows": offending_rows,
            },
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
        # THE CV SERIES, as authoritative outputs of this run.
        #
        # Recorded explicitly as `None` when reporting was disabled rather than omitted, so a
        # reader can tell "this run had no collective variables" from "this manifest predates the
        # field" -- and so a CV-disabled run cannot inherit a previous run's series by silence.
        record["collective_variables"] = self._cv_manifest(state)
        if record["collective_variables"] is not None:
            from .cv_states import verify_manifest_entries

            problems = verify_manifest_entries(Path(self.files.trajectory).parent,
                                               record["collective_variables"])
            if problems:
                # BEFORE the manifest lands. A completion record that claims files it has not
                # checked is exactly the record a later reader trusts.
                raise DriverError(
                    "this ladder's collective-variable output is not a completed set:\n  - "
                    + "\n  - ".join(problems))

        # PER-TAU EQUILIBRATION, verified like the CV series: `None` when the ladder did not use
        # it, stated rather than omitted for the same reason.
        record["per_tau_equilibration"] = self._per_tau_manifest(state)

        if state.get("extends"):
            record["extends"] = self._extension_provenance(state, report)
        storage.write_atomic(self.files.restart,
                             json.dumps(record, indent=2, default=str) + "\n")
        # The completed sidecar carries the history too. It is one of the sources a later
        # continuation merges, and a `completed` record that reported none would be a source
        # claiming this file had never been migrated.
        self._write_rem_log()
        storage.write_run_state(self.files.trajectory, "completed", identity=identity,
                                step=completed,
                                storage_migrations=list(state.get("storage_migrations") or []),
                                note="manifest written; the storage remains authoritative")
        if self.trajectories is not None:
            self.trajectories.close()
        self.reporter.close()
        return record


#: The seed derivation now lives in `core`, which carries no md_tools imports and so can be
#: copied verbatim into an exported reference bundle. It had to move: it decides which random
#: stream an exchange acceptance draws from, so a bundle deriving it independently would propose
#: the same swaps and accept a different set of them.
_stream_seed = stream_seed


def _sha256_of(path, *, chunk=1 << 20):
    """The file exactly as it is on disk. Provenance that names a parent without pinning its
    contents records only where someone looked, not what they found."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_trajectory_digests(directory, *, n_states):
    """The parent's per-state trajectories, pinned. A missing one is recorded as missing rather
    than skipped: a silent gap in this list would read as a parent that had fewer states."""
    from .amber_trajectory import state_trajectory_name

    records = []
    for index in range(int(n_states)):
        path = Path(directory) / state_trajectory_name(index)
        records.append({
            "index": index, "name": path.name,
            "sha256": _sha256_of(path) if path.is_file() else None,
            "present": path.is_file(),
        })
    return records


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
    # READ FROM DISTRIBUTION METADATA RATHER THAN BY IMPORTING IT. `import openmmtools` pulls in
    # PyMBAR, which writes a timeseries caveat and a JAX 64-bit banner to STDERR at import time.
    # The grouped executor deliberately redirects BOTH streams into the run's `.out` -- that is
    # how a real runtime failure reaches the report file it then checks for a completion marker
    # -- so a provenance document ended up carrying another library's advice about statistical
    # inefficiency. The version is a fact about what is installed; establishing it does not
    # require executing the package.
    try:
        from importlib.metadata import version as _distribution_version

        record["openmmtools_present_but_unused"] = _distribution_version("openmmtools")
    except Exception:                                     # noqa: BLE001 - absent, or no metadata
        record["openmmtools_present_but_unused"] = None
    return record
