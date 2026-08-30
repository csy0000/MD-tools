#!/usr/bin/env python
"""The REST2 engine a generated `rest2.py` imports. Copied verbatim into the project.

The user-facing protocol file states the science and nothing else:

    from rest2_runtime import REST2

    def run(files):
        REST2(files, tau_min=0.0, tau_max=0.5, number_of_replicas=6, ...).run(
            number_of_exchanges=1000)

Everything below is implementation. It carries no path of its own: every concrete file arrives in
`files`, which `openmm-rest2` resolved from the command line.

WHO OWNS WHAT
    OpenMMTools owns propagation, reduced potentials, the exchange decision under the default
    `swap-all` scheme, thermodynamic-state assignment, multistate NetCDF storage, checkpointing,
    restart and the walker-to-state mapping. See `rest2_openmmtools.py` for the one thing this
    repository owns about mixing -- WHEN it is attempted -- and for the different, explicitly
    stated ownership under `swap-neighbors`.

    This module owns the REST2 Hamiltonian (through `rest2_scaling`, this repository's convention
    and not OpenMMTools'), the physical-time arithmetic, the run's scientific identity, and the
    records that say what happened.

THE OMEGA CONVENTION
    Omega-selective REST2: torsions about a peptide omega bond are LEFT UNSCALED, so the hot rungs
    cannot rotate a peptide bond and sample cis/trans interconversion the cold rung never sees.
    A deliberate departure from a plain textbook REST2 Hamiltonian, recorded in every provenance
    record this module writes, and not to be described as an unmodified standard REST2.

THREE RECORDS, THREE JOBS
    reporter metadata   the scientific identity, written INTO the NetCDF before propagation
                        begins. This is what makes an interrupted run resumable, because it
                        survives in the authoritative storage whatever happens to the process.
    <stem>.runstate.json   an atomic sidecar tracking initialized -> running -> completed /
                        interrupted / failed. Rank 0 only. Never claims completion.
    restart.json        the COMPLETED-run manifest, written atomically after the final iteration
                        is committed. It is evidence of completion; it is NOT the source of
                        resume identity, and a run interrupted before it exists is still resumable.

TIME
    The ITERATION is the solute output interval. Exchange is attempted every `exchange_stride`
    iterations and full coordinates are written every `checkpoint_interval` iterations, so

        solute frames  : one per iteration            (real configurations, never repeated)
        whole frames   : one per checkpoint_interval  (OpenMMTools checkpoint storage)
        exchanges      : one attempt per exchange_stride iterations

    Every conversion to steps must be exact. A duration that does not divide is refused, never
    rounded: a rounded interval means the reported physical time is not the one simulated.
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
import yaml
from openmm import Platform, XmlSerializer, unit
from openmm.app import PDBFile

from rest2_scaling import (audit_force_classes, build_scaled_system, linear_tau_ladder,
                           scale_factor_for_tau)
from rest2_openmmtools import (DEFAULT_MIXING_SCHEME, TESTED_OPENMMTOOLS_VERSION,
                               VERSION_OVERRIDE_ENVIRONMENT, exchange_ownership,
                               exchange_stride_for, installed_openmmtools_version,
                               sampler_class_for)
from rest2_statistics import (adjacent_pairs, lifetime_mixing_statistics,
                              mapping_is_permutation_every_iteration, round_trips)

#: Written by `run()` into the `.out` once, and only after the manifest is on disk.
COMPLETION_MARKER = "run_status: completed"

#: The scientific identity stored in reporter metadata. Bumped when its MEANING changes.
IDENTITY_FORMAT = "md-templates-rest2-identity/v1"

#: The atomic running-state sidecar.
RUN_STATE_FORMAT = "md-templates-rest2-runstate/v1"

#: The completed-run manifest `-r` carries.
RESTART_FORMAT = "md-templates-rest2-restart/v1"


# --- small helpers -----------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_steps(duration_ps, timestep_fs, *, what):
    """`duration_ps` as a whole number of steps, or an error naming the offender."""
    timestep_ps = timestep_fs / 1000.0
    ratio = duration_ps / timestep_ps
    steps = int(round(ratio))
    if abs(ratio - steps) > 1e-9:
        raise ValueError(
            f"{what} ({duration_ps} ps) is not a whole number of {timestep_fs} fs steps "
            f"({ratio} steps). Rejected rather than rounded.")
    if steps < 1:
        raise ValueError(f"{what} ({duration_ps} ps) is shorter than one {timestep_fs} fs step.")
    return steps


def _write_atomic(path, text):
    """Write through a temporary file in the same directory, then rename.

    A half-written record after a crash would be a claim about a run that cannot be checked, so no
    record this module writes is ever observable in a partial state. The temporary carries this
    process's pid so two writers can never collide on it.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def run_state_path(storage):
    """The sidecar beside the analysis NetCDF: `rest2.nc` -> `rest2.runstate.json`.

    Derived from the STORAGE rather than from `-r`, because the storage is what a resume is given
    and what it is authoritative about. A run interrupted before any manifest exists can still be
    found from `-x` alone.
    """
    storage = Path(storage)
    return storage.with_name(storage.stem + ".runstate.json")


def mpi_rank_and_size():
    """(rank, size) if this process is under MPI, else (0, 1). Never imports MPI needlessly."""
    try:
        from mpi4py import MPI
    except ImportError:
        return 0, 1
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def _versions():
    import netCDF4
    import openmm
    record = {
        "python": platform_module.python_version(),
        "openmm": openmm.version.version,
        "openmm_short": openmm.version.short_version,
        "openmmtools": installed_openmmtools_version(),
        "openmmtools_tested_against": TESTED_OPENMMTOOLS_VERSION,
        "netCDF4": netCDF4.__version__,
        "numpy": np.__version__,
    }
    for name, module in (("pymbar", "pymbar"), ("mpi4py", "mpi4py")):
        try:
            record[name] = __import__(module).__version__
        except ImportError:
            record[name] = None
    try:
        import mpiplus                                    # noqa: F401
        record["mpiplus"] = "present"
    except ImportError:
        record["mpiplus"] = None
    return record


def count_cuda_devices(limit=64):
    """How many CUDA devices this process can actually open, by opening them.

    OpenMM exposes no device count: `getPropertyDefaultValue("DeviceIndex")` is a default index,
    not a total, and reading it as one reported zero devices on a nine-GPU machine.
    """
    from openmm import Context, Platform as _Platform, System, VerletIntegrator
    try:
        platform = _Platform.getPlatformByName("CUDA")
    except Exception:
        return 0
    found = 0
    for index in range(limit):
        system = System()
        system.addParticle(1.0)
        try:
            context = Context(system, VerletIntegrator(0.001), platform,
                              {"DeviceIndex": str(index)})
        except Exception:
            break
        del context
        found += 1
    return found


def visible_cuda_devices(*, probe=True):
    """The device ordinals this process may use.

    `CUDA_VISIBLE_DEVICES` renumbers devices from 0 for the process, so the ordinals OpenMM wants
    are 0..n-1 -- NOT the values in the variable, which are the driver's.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is not None:
        entries = [part.strip() for part in value.split(",") if part.strip() != ""]
        return list(range(len(entries)))
    if not probe:
        return []
    return list(range(count_cuda_devices()))


def select_device_for_rank(rank, size, devices):
    """Which CUDA device this rank drives, and the policy that decided it.

    mpiplus does no device binding whatsoever -- it distributes replicas across ranks and leaves
    the platform entirely to the caller. Without this, every rank creates its Context on the
    default device and the whole ladder runs on GPU 0, silently.
    """
    if not devices:
        return None, "no visible CUDA device"
    if size <= 1:
        return str(devices[0]), "single process: first visible device"
    if len(devices) >= size:
        return str(devices[rank]), f"one rank per device ({size} ranks, {len(devices)} devices)"
    chosen = devices[rank % len(devices)]
    return str(chosen), (
        f"round-robin: {size} ranks share {len(devices)} device(s), "
        f"{-(-size // len(devices))} rank(s) per device")


class Rest2IdentityError(ValueError):
    """The run being continued is not the run that was started."""


# --- the engine --------------------------------------------------------------------------------

class REST2:
    """One REST2 ladder on `openmmtools.multistate.ReplicaExchangeSampler`."""

    def __init__(self, files, *, tau_min, tau_max, number_of_replicas,
                 exchange_interval_ps, temperature_k, timestep_fs,
                 whole_output_interval_ps, solute_output_interval_ps,
                 equilibration_duration_ps=0.0, pressure_bar=None,
                 friction_per_ps=1.0, random_seed=None, platform=None, precision=None,
                 replica_mixing_scheme=DEFAULT_MIXING_SCHEME, constraint_tolerance=1.0e-8,
                 hydrogen_mass_amu=None):
        self.files = files
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.number_of_replicas = int(number_of_replicas)
        self.exchange_interval_ps = float(exchange_interval_ps)
        self.solute_output_interval_ps = float(solute_output_interval_ps)
        self.whole_output_interval_ps = float(whole_output_interval_ps)
        self.equilibration_duration_ps = float(equilibration_duration_ps)
        self.temperature_k = float(temperature_k)
        self.pressure_bar = None if pressure_bar is None else float(pressure_bar)
        self.timestep_fs = float(timestep_fs)
        self.friction_per_ps = float(friction_per_ps)
        self.random_seed = random_seed
        self.platform_name = platform
        self.precision = precision
        self.replica_mixing_scheme = replica_mixing_scheme
        self.constraint_tolerance = float(constraint_tolerance)
        self.hydrogen_mass_amu = hydrogen_mass_amu

        if self.number_of_replicas < 2:
            raise ValueError(f"a REST2 ladder needs at least 2 replicas; got {number_of_replicas}")
        # Refuses an unknown scheme here, before anything is read or built.
        self.exchange_decision_owner = exchange_ownership(self.replica_mixing_scheme)

        # -- exact physical-time arithmetic, before anything is read or built ---------------------
        self.exchange_stride = exchange_stride_for(
            exchange_interval_ps=self.exchange_interval_ps,
            solute_output_interval_ps=self.solute_output_interval_ps)
        self.checkpoint_interval = exchange_stride_for(
            exchange_interval_ps=self.whole_output_interval_ps,
            solute_output_interval_ps=self.solute_output_interval_ps)
        self.steps_per_iteration = exact_steps(
            self.solute_output_interval_ps, self.timestep_fs, what="the solute output interval")
        self.equilibration_iterations = 0
        if self.equilibration_duration_ps > 0:
            steps = exact_steps(self.equilibration_duration_ps, self.timestep_fs,
                                what="the per-tau equilibration duration")
            if steps % self.steps_per_iteration:
                raise ValueError(
                    f"the per-tau equilibration duration ({self.equilibration_duration_ps} ps) "
                    f"must be a whole multiple of the solute output interval "
                    f"({self.solute_output_interval_ps} ps).")
            self.equilibration_iterations = steps // self.steps_per_iteration

        self.taus = linear_tau_ladder(self.tau_min, self.tau_max, self.number_of_replicas)
        self.scale_factors = [scale_factor_for_tau(t) for t in self.taus]

        self._sampler = None
        self._reporter = None
        self._interrupted = False

    # -- paths ------------------------------------------------------------------------------------

    @property
    def storage(self):
        return self.files.trajectory

    @property
    def checkpoint_storage(self):
        return self.files.checkpoint

    @property
    def run_state(self):
        return str(run_state_path(self.storage))

    # -- inputs ------------------------------------------------------------------------------------

    def _load_solute(self):
        document = yaml.safe_load(Path(self.files.solute).read_text(encoding="utf-8"))
        count = int(document["n_solute_atoms"])
        span = document.get("solute_atom_range")
        if span and document.get("solute_atom_indices_are_contiguous", False):
            indices = list(range(int(span[0]), int(span[1]) + 1))
            if len(indices) != count:
                raise ValueError(
                    f"{self.files.solute} says {count} solute atoms but its range {span} spans "
                    f"{len(indices)}.")
        else:
            indices = list(range(count))
        excluded = [tuple(int(a) for a in pair)
                    for pair in (document.get("rest2") or {}).get("omega_excluded_bonds", [])]
        return indices, excluded, document

    def _resolve_platform(self):
        """Platform, and for CUDA the device this rank drives. Never a silent GPU 0."""
        rank, size = mpi_rank_and_size()
        name = self.platform_name
        if name in (None, "automatic"):
            available = {Platform.getPlatform(i).getName()
                         for i in range(Platform.getNumPlatforms())}
            name = "CUDA" if "CUDA" in available else "CPU"
        properties = {}
        device, policy = None, "not a CUDA platform"
        if name == "CUDA":
            devices = visible_cuda_devices(probe=(size > 1))
            if size > 1:
                device, policy = select_device_for_rank(rank, size, devices)
                if device is None:
                    raise RuntimeError(
                        "the CUDA platform was selected under MPI but no CUDA device is visible "
                        "to this rank. Listing the platform is not having a device; refusing "
                        "rather than letting every rank fall onto one GPU.")
                properties["DeviceIndex"] = device
            else:
                device, policy = None, "single process: OpenMM selects the device"
            properties["Precision"] = self.precision or "mixed"
        elif name == "OpenCL":
            properties["Precision"] = self.precision or "mixed"
        return Platform.getPlatformByName(name), properties, {
            "platform": name, "device_index": device, "device_policy": policy,
            "precision": properties.get("Precision"),
            "mpi_rank": rank, "mpi_size": size, "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    # -- identity ------------------------------------------------------------------------------

    def scientific_identity(self, solute_indices, excluded_bonds):
        """Everything a continuation must agree with.

        Built BEFORE the sampler is created and stored in reporter metadata, so it lives inside the
        authoritative storage rather than in a manifest that only a completed run writes. The atom
        indices and omega exclusions are hashed rather than listed: a solvated solute is thousands
        of integers and these records are meant to stay readable.
        """
        return {
            "format": IDENTITY_FORMAT,
            "taus": [float(t) for t in self.taus],
            "scale_factors": [float(s) for s in self.scale_factors],
            "number_of_replicas": self.number_of_replicas,
            "temperature_k": self.temperature_k,
            "pressure_bar": self.pressure_bar,
            "timestep_fs": self.timestep_fs,
            "friction_per_ps": self.friction_per_ps,
            "constraint_tolerance": self.constraint_tolerance,
            "exchange_interval_ps": self.exchange_interval_ps,
            "solute_output_interval_ps": self.solute_output_interval_ps,
            "whole_output_interval_ps": self.whole_output_interval_ps,
            "exchange_stride_iterations": self.exchange_stride,
            "checkpoint_interval_iterations": self.checkpoint_interval,
            "steps_per_iteration": self.steps_per_iteration,
            "equilibration_duration_ps": self.equilibration_duration_ps,
            "replica_mixing_scheme": self.replica_mixing_scheme,
            "exchange_decision_owner": self.exchange_decision_owner,
            "random_seed": self.random_seed,
            "enhanced_region_sha256": sha256_text(
                json.dumps({"solute": [int(i) for i in solute_indices],
                            "omega_excluded_bonds": [[int(a), int(b)]
                                                     for a, b in excluded_bonds]},
                           sort_keys=True)),
            "topology_sha256": sha256_file(self.files.topology),
            "system_sha256": sha256_file(self.files.system),
            "solute_definition_sha256": sha256_file(self.files.solute),
        }

    @staticmethod
    def compare_identity(before, now):
        """Fields that differ between a stored identity and the current request."""
        keys = set(before) | set(now)
        keys.discard("format")
        return [key for key in sorted(keys) if before.get(key) != now.get(key)]

    def _require_same_identity(self, before, now, *, source):
        differences = self.compare_identity(before, now)
        if differences:
            lines = "\n".join(f"    {key}: was {before.get(key)!r}, now {now.get(key)!r}"
                              for key in differences)
            raise Rest2IdentityError(
                f"the scientific configuration changed since this run was created, so continuing "
                f"it would append samples from a different Hamiltonian to the same storage "
                f"(identity read from {source}):\n{lines}\n"
                f"  Start a new run instead. Running longer is a legitimate extension; changing "
                f"the physics is not.")

    # -- the run-state sidecar ---------------------------------------------------------------------

    def write_run_state(self, status, **extra):
        """Atomically record what this run is doing. Rank 0 only; never claims completion early."""
        rank, size = mpi_rank_and_size()
        if rank != 0:
            return None
        record = {
            "format": RUN_STATE_FORMAT,
            "status": status,
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pid": os.getpid(),
            "mpi_size": size,
            "storage": {
                "analysis_netcdf": os.path.basename(self.storage),
                "checkpoint_netcdf": (None if not self.checkpoint_storage
                                      else os.path.basename(self.checkpoint_storage)),
                "manifest": os.path.basename(self.files.restart),
            },
        }
        record.update(extra)
        _write_atomic(self.run_state, json.dumps(record, indent=2) + "\n")
        return record

    def read_run_state(self):
        path = Path(self.run_state)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # -- the run ---------------------------------------------------------------------------------

    def run(self, number_of_exchanges, *, resume=None, extend=None):
        """Start, resume or extend the ladder, then write the completion manifest.

        `number_of_exchanges` is the TOTAL exchange budget for the run, not an increment;
        `extend=N` adds N mixing events to a run that already reached its budget.

        `resume` and `extend` default to whatever `openmm-rest2` put on `files`, so `--resume` and
        `--extend N` work without the protocol file mentioning them: the scientific protocol says
        how long the calculation is, the command line says which invocation this is.
        """
        if resume is None:
            resume = bool(getattr(self.files, "resume", False))
        if extend is None:
            extend = int(getattr(self.files, "extend", 0) or 0)

        number_of_exchanges = int(number_of_exchanges)
        if number_of_exchanges < 1:
            raise ValueError(f"number_of_exchanges must be >= 1; got {number_of_exchanges}")

        started = datetime.datetime.now(datetime.timezone.utc)
        solute_indices, excluded_bonds, solute_document = self._load_solute()
        identity = self.scientific_identity(solute_indices, excluded_bonds)

        continuing = bool(resume or extend)
        if continuing and not Path(self.storage).exists():
            raise FileNotFoundError(
                f"{self.storage} does not exist, so there is nothing to "
                f"{'extend' if extend else 'resume'}. The NetCDF storage is authoritative; a "
                f"missing one means the run never started here.")

        handlers = self._install_interrupt_handlers()
        try:
            if continuing:
                sampler, reporter, total_iterations = self._continue(
                    identity, number_of_exchanges, extend=extend)
            else:
                sampler, reporter, total_iterations = self._start(
                    identity, number_of_exchanges, solute_indices, excluded_bonds)
            self._sampler, self._reporter = sampler, reporter
            sampler.run()
        except BaseException as failure:                     # noqa: BLE001 - recorded, re-raised
            status = "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
            # The identity goes back in. The sidecar is the FALLBACK source a resume uses when
            # reporter metadata is unreadable, so a failure write that dropped it would remove the
            # very record the failure makes valuable.
            self.write_run_state(status, identity=identity,
                                 reason=f"{type(failure).__name__}: {failure}"[:400],
                                 note=("OpenMMTools storage is preserved and this run can be "
                                       "continued with --resume; no completion manifest exists."))
            raise
        finally:
            self._restore_interrupt_handlers(handlers)

        return self._finalise(identity, solute_document, excluded_bonds, total_iterations, started)

    # -- interruption ------------------------------------------------------------------------------

    def _install_interrupt_handlers(self):
        """Turn SIGINT/SIGTERM into KeyboardInterrupt so the sidecar records an interruption.

        OpenMMTools decorates its per-iteration reporting with `mpiplus.delayed_termination`, so a
        signal arriving mid-write is deferred until that iteration's records are complete. The
        storage a resume reads is therefore never the half-written one.

        Signal handlers can only be installed on the main thread; a run driven from elsewhere keeps
        whatever handling it already had rather than failing here.
        """
        previous = {}

        def stop(signum, frame):
            self._interrupted = True
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

    @staticmethod
    def _restore_interrupt_handlers(previous):
        for number, handler in (previous or {}).items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                pass

    # -- starting a new run --------------------------------------------------------------------------

    def _start(self, identity, number_of_exchanges, solute_indices, excluded_bonds):
        from openmmtools import cache, mcmc, states
        from openmmtools.multistate import MultiStateReporter

        openmm_platform, properties, run_context = self._resolve_platform()
        base_system = XmlSerializer.deserialize(
            Path(self.files.system).read_text(encoding="utf-8"))
        audit = audit_force_classes(base_system, where="REST2 ladder construction")
        self._audit = audit
        PDBFile(self.files.topology)                     # refuses an unreadable topology early

        total_iterations = number_of_exchanges * self.exchange_stride
        implicit = self.pressure_bar is None
        pressure = None if implicit else self.pressure_bar * unit.bar

        # The identity exists BEFORE anything propagates, and the sidecar says so.
        self.write_run_state("initialized", identity=identity,
                             iterations_requested=total_iterations,
                             exchange_attempts_requested=number_of_exchanges,
                             note="scientific identity fixed; no propagation has happened yet")

        self._announce(run_context, excluded_bonds)

        reporter = MultiStateReporter(
            self.storage,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_storage=(None if not self.checkpoint_storage
                                else Path(self.checkpoint_storage).name),
            analysis_particle_indices=tuple(int(i) for i in solute_indices))

        thermodynamic_states = [
            states.ThermodynamicState(
                system=build_scaled_system(base_system, solute_indices, tau,
                                           excluded_bonds=excluded_bonds),
                temperature=self.temperature_k * unit.kelvin, pressure=pressure)
            for tau in self.taus]

        common = XmlSerializer.deserialize(
            Path(self.files.coordinates).read_text(encoding="utf-8"))
        box = common.getPeriodicBoxVectors()
        # Positions, velocities AND box travel together: they are one configuration, and a replica
        # that kept its positions but not its box would be at a different density.
        sampler_states = [
            states.SamplerState(positions=common.getPositions(asNumpy=True),
                                velocities=common.getVelocities(asNumpy=True),
                                box_vectors=(None if implicit else box))
            for _ in self.taus]

        move = mcmc.LangevinDynamicsMove(
            timestep=self.timestep_fs * unit.femtosecond,
            collision_rate=self.friction_per_ps / unit.picosecond,
            n_steps=self.steps_per_iteration,
            constraint_tolerance=self.constraint_tolerance,
            reassign_velocities=False)

        sampler = sampler_class_for(self.replica_mixing_scheme)(
            mcmc_moves=move, number_of_iterations=total_iterations,
            replica_mixing_scheme=self.replica_mixing_scheme,
            exchange_stride=self.exchange_stride)

        context_cache = cache.ContextCache(capacity=None, time_to_live=None,
                                           platform=openmm_platform)
        for attribute in ("energy_context_cache", "sampler_context_cache"):
            setattr(sampler, attribute, context_cache)
        if properties:
            # ContextCache does not forward platform properties, so they are set on the Platform
            # itself -- which is what actually decides the device this rank uses.
            for key, value in properties.items():
                openmm_platform.setPropertyDefaultValue(key, value)

        # `create(metadata=...)` is the supported way to put a dictionary into the storage; it adds
        # `title` and hands the rest to `MultiStateReporter.write_dict('metadata', ...)`.
        sampler.create(thermodynamic_states=thermodynamic_states,
                       sampler_states=sampler_states, storage=reporter,
                       metadata=self._metadata(identity, total_iterations, number_of_exchanges,
                                               run_context))

        if self.equilibration_iterations:
            print(f"# per-tau equilibration: {self.equilibration_iterations} iteration(s) "
                  f"= {self.equilibration_duration_ps} ps per replica, NOT counted toward "
                  f"production")
            sys.stdout.flush()
            sampler.equilibrate(self.equilibration_iterations)

        self.write_run_state("running", identity=identity,
                             iterations_requested=total_iterations,
                             exchange_attempts_requested=number_of_exchanges,
                             note="propagation started; storage is authoritative for progress")
        self._run_context = run_context
        return sampler, reporter, total_iterations

    def _metadata(self, identity, total_iterations, number_of_exchanges, run_context):
        """What goes into reporter metadata. `title` is required by `write_dict`."""
        return {
            "title": (f"REST2 ladder, {self.number_of_replicas} replicas, tau "
                      f"{self.tau_min} to {self.tau_max}, generated by md-templates"),
            "rest2_identity": identity,
            "rest2_plan": {
                "iterations_requested": int(total_iterations),
                "exchange_attempts_requested": int(number_of_exchanges),
                "analysis_netcdf": os.path.basename(self.storage),
                "checkpoint_netcdf": (None if not self.checkpoint_storage
                                      else os.path.basename(self.checkpoint_storage)),
                "manifest": os.path.basename(self.files.restart),
            },
            "rest2_generator": _project_identity(self.files),
            "rest2_versions": _versions(),
            "rest2_created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rest2_hostname": run_context.get("hostname"),
        }

    # -- continuing an existing run ---------------------------------------------------------------

    def stored_identity(self, reporter):
        """The identity this storage was created with, from reporter metadata.

        Falls back to the run-state sidecar, which is an independently written record of the same
        identity. Returns (identity, source) or (None, reason).
        """
        try:
            metadata = reporter.read_dict("metadata")
        except Exception as failure:                     # noqa: BLE001 - a reason, not a crash
            metadata = None
            reason = f"reporter metadata unreadable ({type(failure).__name__}: {failure})"
        else:
            reason = "reporter metadata carries no rest2_identity"
        if isinstance(metadata, dict) and isinstance(metadata.get("rest2_identity"), dict):
            return metadata["rest2_identity"], "reporter metadata"

        sidecar = self.read_run_state()
        if isinstance(sidecar, dict) and isinstance(sidecar.get("identity"), dict):
            return sidecar["identity"], f"run-state sidecar {os.path.basename(self.run_state)}"
        return None, reason

    def _continue(self, identity, number_of_exchanges, *, extend):
        """Resume to the original budget, or extend beyond it. The two are kept distinct."""
        sampler_class = sampler_class_for(self.replica_mixing_scheme)
        sampler = sampler_class.from_storage(self.storage)
        sampler.exchange_stride = self.exchange_stride
        reporter = sampler._reporter

        before, source = self.stored_identity(reporter)
        if before is None:
            raise Rest2IdentityError(
                f"neither reporter metadata nor a run-state sidecar can establish what "
                f"{os.path.basename(self.storage)} was created with ({source}), so this run "
                f"cannot be continued on trust. Refusing rather than appending samples whose "
                f"Hamiltonian cannot be checked.")
        self._require_same_identity(before, identity, source=source)

        stored_openmmtools = None
        try:
            metadata = reporter.read_dict("metadata")
            stored_openmmtools = (metadata.get("rest2_versions") or {}).get("openmmtools")
        except Exception:
            metadata = None
        current_openmmtools = installed_openmmtools_version()
        if stored_openmmtools and stored_openmmtools != current_openmmtools:
            raise Rest2IdentityError(
                f"this storage was written by openmmtools {stored_openmmtools} and the installed "
                f"version is {current_openmmtools}. Exchange scheduling and restart determinism "
                f"are version-dependent; refusing to continue across a change.")

        last = int(reporter.read_last_iteration(last_checkpoint=False))
        last_checkpoint = int(reporter.read_last_iteration(last_checkpoint=True))
        budget = int(sampler.number_of_iterations)

        if extend:
            if last < budget:
                raise Rest2IdentityError(
                    f"--extend adds mixing events to a run that reached its budget, but this one "
                    f"stopped at iteration {last} of {budget}. Use --resume to finish it first; "
                    f"the two are different operations and conflating them would silently shorten "
                    f"the original request.")
            added = int(extend) * self.exchange_stride
            sampler.extend(added)
            total_iterations = int(sampler.number_of_iterations)
            operation = f"extended by {extend} mixing event(s) (+{added} iterations)"
        else:
            if last >= budget:
                print(f"# nothing to resume: iteration {last} already reached the requested "
                      f"budget of {budget}")
            total_iterations = budget
            operation = f"resumed at iteration {last} toward the original budget of {budget}"

        _, _, run_context = self._resolve_platform()
        self._run_context = run_context
        self._audit = audit_force_classes(
            XmlSerializer.deserialize(Path(self.files.system).read_text(encoding="utf-8")),
            where="REST2 continuation")
        print(f"# {operation}")
        print(f"# identity verified against {source}")
        print(f"# last committed iteration {last}, last checkpoint {last_checkpoint}")
        sys.stdout.flush()

        self.write_run_state("running", identity=identity,
                             iterations_requested=total_iterations,
                             exchange_attempts_requested=(
                                 number_of_exchanges + int(extend or 0)),
                             resumed_from_iteration=last,
                             note=operation)
        return sampler, reporter, total_iterations

    # -- announcing ---------------------------------------------------------------------------------

    def _announce(self, run_context, excluded_bonds):
        print(f"# REST2 on openmmtools {installed_openmmtools_version()} "
              f"(private-method extension tested against {TESTED_OPENMMTOOLS_VERSION})")
        print(f"# ladder tau       : {[round(t, 6) for t in self.taus]}")
        print(f"# ladder s=(1-tau)^2: {[round(s, 6) for s in self.scale_factors]}")
        print(f"# one temperature  : {self.temperature_k} K for every rung "
              f"(Hamiltonian scaling, not temperature REMD)")
        print(f"# omega convention : omega-selective REST2 -- "
              f"{len(excluded_bonds)} omega bond(s) left UNSCALED")
        print(f"# mixing scheme    : {self.replica_mixing_scheme}; the accept/reject decision is "
              f"made by {self.exchange_decision_owner}")
        print(f"# iteration        : {self.solute_output_interval_ps} ps "
              f"= {self.steps_per_iteration} steps at {self.timestep_fs} fs")
        print(f"# exchange every   : {self.exchange_stride} iteration(s) "
              f"= {self.exchange_interval_ps} ps")
        print(f"# whole coords every: {self.checkpoint_interval} iteration(s) "
              f"= {self.whole_output_interval_ps} ps")
        print(f"# platform         : {run_context['platform']} "
              f"device={run_context['device_index']} ({run_context['device_policy']})")
        print(f"# mpi              : rank {run_context['mpi_rank']}/{run_context['mpi_size']} "
              f"on {run_context['hostname']}")
        sys.stdout.flush()

    # -- finishing -----------------------------------------------------------------------------------

    def _finalise(self, identity, solute_document, excluded_bonds, total_iterations, started):
        sampler, reporter = self._sampler, self._reporter
        completed = int(sampler.iteration)
        finished = datetime.datetime.now(datetime.timezone.utc)
        rank, size = mpi_rank_and_size()

        # Only rank 0 holds an OPEN reporter. OpenMMTools opens the storage in append mode on node
        # 0 alone (`mpiplus.run_single_node(0, reporter.open, mode='a')`), so a non-zero rank
        # reading it back gets `'NoneType' object has no attribute 'variables'`. The derived
        # reports belong to the run, and the run's records belong to rank 0.
        if rank != 0:
            print(f"# rank {rank} of {size} finished its share of the propagation; rank 0 owns "
                  f"{os.path.basename(self.files.restart)} and the NetCDF storage")
            print(f"rank_status: completed rank={rank}")
            return {"run_status": "completed_by_rank", "rank": rank,
                    "iterations_completed": completed}

        statistics = self.lifetime_statistics(reporter)
        trips = self.round_trip_report(reporter)

        record = {
            "format": RESTART_FORMAT,
            "run_status": "completed" if completed >= total_iterations else "incomplete",
            "iterations_expected": int(total_iterations),
            "iterations_completed": completed,
            "exchange_attempts_expected": int(total_iterations) // self.exchange_stride,
            "exchange_attempts_recorded": statistics["mixing_events"],
            "production_per_replica_ps": completed * self.solute_output_interval_ps,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "storage": {
                "analysis_netcdf": os.path.basename(self.storage),
                "checkpoint_netcdf": (None if not self.checkpoint_storage
                                      else os.path.basename(self.checkpoint_storage)),
                "run_state": os.path.basename(self.run_state),
                "authoritative": "analysis_netcdf",
                "note": ("OpenMMTools owns these files. This manifest references them and adds "
                         "project-level provenance; it never duplicates the arrays they hold."),
            },
            "scientific_identity": identity,
            "omega_convention": {
                "name": "omega-selective REST2",
                "omega_bonds_left_unscaled": len(excluded_bonds),
                "detection_method": (solute_document.get("rest2") or {}).get(
                    "omega_detection_method"),
                "statement": ("Torsions about a peptide omega bond are NOT scaled. This is this "
                              "repository's convention and is not an unmodified standard REST2 "
                              "Hamiltonian."),
            },
            "hamiltonian": {
                "scaling": "s = (1 - tau)^2 solute-solute, sqrt(s) = 1 - tau solute-environment",
                "single_temperature_k": self.temperature_k,
                "is_temperature_remd": False,
                "forces_scaled": [name for _, name in self._audit["scaled"]],
                "forces_unscaled_by_convention": [
                    n for _, n in self._audit["unscaled_by_convention"]],
                "forces_energy_free": [n for _, n in self._audit["energy_free"]],
                "solvation": "implicit" if self.pressure_bar is None else "explicit",
                "hydrogen_mass_amu": self.hydrogen_mass_amu,
            },
            "exchange": {
                "replica_mixing_scheme": self.replica_mixing_scheme,
                "decision_owner": self.exchange_decision_owner,
                "schedule_owner": "md-templates",
                "schedule_note": ("md-templates decides WHICH iterations attempt exchange (the "
                                  "stride); it does not decide any swap under swap-all."),
                "exchange_stride_iterations": self.exchange_stride,
            },
            "move": {
                "class": "openmmtools.mcmc.LangevinDynamicsMove",
                "integrator": "openmm.LangevinMiddleIntegrator",
                "splitting": "BAOAB (Langevin middle), as implemented by LangevinMiddleIntegrator",
                "collision_rate_per_ps": self.friction_per_ps,
                "timestep_fs": self.timestep_fs,
                "constraint_tolerance": self.constraint_tolerance,
                "n_steps_per_iteration": self.steps_per_iteration,
                "reassign_velocities": False,
            },
            "lifetime_mixing_statistics": statistics,
            "round_trips": trips,
            "versions": _versions(),
            "execution": self._run_context,
            "replica_thermodynamic_states_final": [
                int(x) for x in sampler._replica_thermodynamic_states],
            "inputs": {
                "topology": os.path.basename(self.files.topology),
                "system": os.path.basename(self.files.system),
                "coordinates": os.path.basename(self.files.coordinates),
                "solute": os.path.basename(self.files.solute),
                "protocol": os.path.basename(self.files.input),
            },
            "project": _project_identity(self.files),
        }

        if record["run_status"] != "completed":
            self.write_run_state("interrupted", identity=identity,
                                 iterations_requested=int(total_iterations),
                                 iterations_completed=completed,
                                 note="stopped before the requested budget; resumable")
            raise RuntimeError(
                f"the sampler stopped at iteration {completed} of {total_iterations}. No "
                f"completion manifest was written; the NetCDF storage holds what did run and "
                f"--resume will continue it.")

        # Rank 0 only reaches here; every other rank returned above.
        _write_atomic(self.files.restart,
                      json.dumps(record, indent=2, sort_keys=False) + "\n")
        self.write_run_state("completed", identity=identity,
                             iterations_requested=int(total_iterations),
                             iterations_completed=completed,
                             note="manifest written; storage remains authoritative")
        self._print_summary(record, trips)
        print(COMPLETION_MARKER)
        return record

    # -- derived reports -------------------------------------------------------------------------

    def lifetime_statistics(self, reporter):
        """Lifetime mixing statistics from the complete stored history.

        Read through the public reporter API, so this is correct across an interrupted resume and
        an extension: the storage is authoritative, not the process that happened to write it.
        """
        accepted, proposed = reporter.read_mixing_statistics(slice(None))
        return lifetime_mixing_statistics(accepted, proposed,
                                          scheme=self.replica_mixing_scheme,
                                          exchange_stride=self.exchange_stride)

    def round_trip_report(self, reporter):
        mapping = np.asarray(reporter.read_replica_thermodynamic_states())
        ok, offending = mapping_is_permutation_every_iteration(
            mapping, n_states=self.number_of_replicas)
        return {
            "definition": ("cold state visited, then the hot state later, then the cold state "
                           "again. The starting position earns nothing: a walker beginning at the "
                           "hot state must first reach cold, then hot, then cold."),
            "cold_state": 0,
            "hot_state": self.number_of_replicas - 1,
            "mapping_rows": int(mapping.shape[0]),
            "every_row_is_a_permutation": bool(ok),
            "offending_rows": offending,
            "by_walker": round_trips(mapping, n_states=self.number_of_replicas),
        }

    def _print_summary(self, record, trips):
        statistics = record["lifetime_mixing_statistics"]
        print()
        print("# --- REST2 summary (readable; the NetCDF is authoritative) -------------------")
        print(f"# iterations completed : {record['iterations_completed']}")
        print(f"# mixing events        : {statistics['mixing_events']} "
              f"(expected {record['exchange_attempts_expected']})")
        print(f"# production/replica   : {record['production_per_replica_ps']} ps")
        print(f"# mixing scheme        : {statistics['scheme']} -- decision owned by "
              f"{record['exchange']['decision_owner']}")
        print(f"# final replica->state : {record['replica_thermodynamic_states_final']}")
        print("# LIFETIME acceptance by adjacent state pair "
              f"(iterations {statistics['iteration_range'][0]}-{statistics['iteration_range'][1]}):")
        for pair in adjacent_pairs(statistics):
            rate = pair["acceptance"]
            shown = "n/a" if rate is None else f"{rate:.3f}"
            i, j = pair["state_pair"]
            print(f"#   states {i}-{j} (tau {self.taus[i]:.3f}-{self.taus[j]:.3f}): "
                  f"{pair['accepted']}/{pair['proposed']} = {shown}")
        if statistics["scheme"] == "swap-all":
            print(f"#   (swap-all also proposes non-adjacent pairs; overall off-diagonal "
                  f"acceptance "
                  f"{'n/a' if statistics['overall_acceptance'] is None else format(statistics['overall_acceptance'], '.3f')}"
                  f", self-swaps on the diagonal: {statistics['self_proposals_on_diagonal']})")
        print(f"# round trips (cold->hot->cold): "
              f"{[w['round_trips'] for w in trips['by_walker']]}")
        print(f"#   walkers started at states  : "
              f"{[w['started_at_state'] for w in trips['by_walker']]}")
        print(f"# manifest             : {os.path.basename(self.files.restart)}")
        print("# ---------------------------------------------------------------------------")


def _project_identity(files):
    """The generating project and template identity, read from files the project already has."""
    here = Path(files.input).resolve().parent
    for parent in [here, *here.parents]:
        for name in ("config.yaml", "md.config.yaml"):
            candidate = parent / name
            if candidate.is_file():
                try:
                    document = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                except Exception:
                    return {"source": None, "note": "project configuration could not be parsed"}
                provenance = document.get("provenance") or {}
                return {
                    "source": name,
                    "templates_commit": (provenance.get("git_commit")
                                         or document.get("templates_commit")),
                    "templates_version": (provenance.get("version")
                                          or document.get("templates_version")),
                    "system": document.get("system") or document.get("system_id"),
                }
    return {"source": None, "note": "no project configuration found above the protocol file"}
