#!/usr/bin/env python
"""The REST2 engine a generated `rest2.py` imports. Copied verbatim into the project.

The user-facing protocol file states the science and nothing else:

    from rest2_runtime import REST2

    def run(files):
        REST2(files, tau_min=0.0, tau_max=0.5, number_of_replicas=6,
              exchange_interval_ps=10.0, ...).run(number_of_exchanges=1000)

Everything below is implementation. It carries no path of its own: every concrete file arrives in
`files`, which `openmm-rest2` resolved from the command line.

WHO OWNS WHAT
    OpenMMTools owns replica exchange -- propagation, reduced potentials, exchange decisions,
    thermodynamic-state assignment, multistate NetCDF storage, checkpointing, restart, extension
    and the walker-to-state mapping. This module owns the REST2 Hamiltonian (through
    `rest2_scaling`, which is this repository's convention and not OpenMMTools'), the physical-time
    arithmetic, the input identity checks and the completion manifest.

THE OMEGA CONVENTION
    This is omega-selective REST2: torsions about a peptide omega bond are LEFT UNSCALED, so the
    hot rungs cannot rotate a peptide bond and sample cis/trans interconversion the cold rung never
    sees. That is a deliberate departure from a plain textbook REST2 Hamiltonian, it is recorded in
    every provenance record this module writes, and it should not be described as an unmodified
    standard REST2.

TIME
    The ITERATION is the solute output interval. Exchanges are attempted every `exchange_stride`
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
import socket
import sys
from pathlib import Path

import numpy as np
import yaml
from openmm import Platform, XmlSerializer, unit
from openmm.app import PDBFile

from rest2_scaling import (audit_force_classes, build_scaled_system, linear_tau_ladder,
                           scale_factor_for_tau)
from rest2_openmmtools import (StridedREST2Sampler, TESTED_OPENMMTOOLS_VERSION,
                               VERSION_OVERRIDE_ENVIRONMENT, exchange_stride_for,
                               require_tested_openmmtools)

#: Written by `run()` into the `.out` once, and only after the manifest is on disk.
COMPLETION_MARKER = "run_status: completed"

#: The manifest format `-r` carries. Bumped when its meaning changes, never when a field is added.
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

    The manifest is the completion evidence. A half-written one after a crash would be a claim
    that a run finished when it did not, so it is never observable in a partial state.
    """
    path = Path(path)
    # The temporary carries this process's pid so two writers can never collide on it. Under MPI
    # only rank 0 should reach here at all, but a shared scratch name is the kind of thing that
    # works until the day it does not.
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _versions():
    import netCDF4
    import openmm
    import openmmtools
    record = {
        "python": platform_module.python_version(),
        "openmm": openmm.version.version,
        "openmm_short": openmm.version.short_version,
        "openmmtools": openmmtools.__version__,
        "openmmtools_tested_against": TESTED_OPENMMTOOLS_VERSION,
        "netCDF4": netCDF4.__version__,
        "numpy": np.__version__,
    }
    try:
        import pymbar
        record["pymbar"] = pymbar.__version__
    except ImportError:
        record["pymbar"] = None
    try:
        import mpi4py
        record["mpi4py"] = mpi4py.__version__
    except ImportError:
        record["mpi4py"] = None
    try:
        import mpiplus                                    # noqa: F401
        record["mpiplus"] = "present"
    except ImportError:
        record["mpiplus"] = None
    return record


def mpi_rank_and_size():
    """(rank, size) if this process is under MPI, else (0, 1). Never imports MPI needlessly."""
    try:
        from mpi4py import MPI
    except ImportError:
        return 0, 1
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def count_cuda_devices(limit=64):
    """How many CUDA devices this process can actually open, by opening them.

    OpenMM exposes no device count: `getPropertyDefaultValue("DeviceIndex")` is a default index,
    not a total, and reading it as one is how a machine with nine GPUs came to report none. The
    only answer OpenMM will give is whether a Context on device N can be created, so that is what
    is asked, on a one-particle System that costs nothing.
    """
    from openmm import Context, Platform, System, VerletIntegrator
    try:
        platform = Platform.getPlatformByName("CUDA")
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

    `CUDA_VISIBLE_DEVICES` renumbers devices from 0 for the process, so when it is set the
    ordinals OpenMM wants are 0..n-1 -- NOT the values in the variable, which are the driver's.
    Passing the driver ordinals through would address the wrong device whenever the variable does
    not start at 0, which is exactly the case a per-rank binding creates.
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
    default device and the whole ladder runs on GPU 0 at a fraction of the speed, silently.
    """
    if not devices:
        return None, "no visible CUDA device"
    if size <= 1:
        return str(devices[0]), "single process: first visible device"
    if len(devices) >= size:
        return str(devices[rank]), f"one rank per device ({size} ranks, {len(devices)} devices)"
    # Fewer devices than ranks: round-robin, stated rather than stumbled into.
    chosen = devices[rank % len(devices)]
    return str(chosen), (
        f"round-robin: {size} ranks share {len(devices)} device(s), "
        f"{-(-size // len(devices))} rank(s) per device")


# --- the engine --------------------------------------------------------------------------------

class REST2:
    """One REST2 ladder on `openmmtools.multistate.ReplicaExchangeSampler`.

    Parameters are physical. Nothing here is a path; `files` carries those.
    """

    def __init__(self, files, *, tau_min, tau_max, number_of_replicas,
                 exchange_interval_ps, temperature_k, timestep_fs,
                 whole_output_interval_ps, solute_output_interval_ps,
                 equilibration_duration_ps=0.0, pressure_bar=None,
                 friction_per_ps=1.0, random_seed=None, platform=None, precision=None,
                 replica_mixing_scheme="swap-neighbors", constraint_tolerance=1.0e-8,
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
        self._started = None

    # -- inputs ---------------------------------------------------------------------------------

    @property
    def storage(self):
        return self.files.trajectory

    @property
    def checkpoint_storage(self):
        return self.files.checkpoint

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
            # A single process needs no binding: there is nothing to collide with, and forcing an
            # index would override a deliberate CUDA_VISIBLE_DEVICES. Under MPI the binding is
            # mandatory -- mpiplus does none, so without it every rank lands on the same device.
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
        elif name in ("OpenCL",):
            properties["Precision"] = self.precision or "mixed"
        return Platform.getPlatformByName(name), properties, {
            "platform": name, "device_index": device, "device_policy": policy,
            "precision": properties.get("Precision"),
            "mpi_rank": rank, "mpi_size": size, "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    # -- identity ------------------------------------------------------------------------------

    def _scientific_identity(self, solute_indices, excluded_bonds):
        """Everything a resume must agree with. Hashed so a change is one comparison, not twenty.

        The atom indices and omega exclusions are hashed rather than listed here: a solvated
        solute is thousands of integers, and the manifest is meant to stay readable.
        """
        return {
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

    # -- the run -------------------------------------------------------------------------------

    def run(self, number_of_exchanges, *, resume=None, extend=None):
        """Run, resume or extend the ladder, then write the completion manifest.

        `number_of_exchanges` is the TOTAL exchange budget for the run, not an increment;
        `extend=N` adds N attempts to a finished one.

        `resume` and `extend` default to whatever `openmm-rest2` put on `files`, so `--resume` and
        `--extend N` work without the protocol file mentioning them. The scientific protocol says
        how long the calculation is; the command line says which invocation this is.
        """
        if resume is None:
            resume = bool(getattr(self.files, "resume", False))
        if extend is None:
            extend = int(getattr(self.files, "extend", 0) or 0)
        from openmmtools import cache, mcmc, states
        from openmmtools.multistate import MultiStateReporter

        started = datetime.datetime.now(datetime.timezone.utc)
        installed_openmmtools = require_tested_openmmtools()

        number_of_exchanges = int(number_of_exchanges)
        if number_of_exchanges < 1:
            raise ValueError(f"number_of_exchanges must be >= 1; got {number_of_exchanges}")

        solute_indices, excluded_bonds, solute_document = self._load_solute()
        identity = self._scientific_identity(solute_indices, excluded_bonds)
        openmm_platform, properties, run_context = self._resolve_platform()

        pdb = PDBFile(self.files.topology)
        base_system = XmlSerializer.deserialize(
            Path(self.files.system).read_text(encoding="utf-8"))
        # Refuse an energy-bearing force this repository cannot place, BEFORE any Context exists.
        audit = audit_force_classes(base_system, where="REST2 ladder construction")

        implicit = self.pressure_bar is None
        pressure = None if implicit else self.pressure_bar * unit.bar

        print(f"# REST2 on openmmtools {installed_openmmtools} "
              f"(tested against {TESTED_OPENMMTOOLS_VERSION})")
        print(f"# ladder tau       : {[round(t, 6) for t in self.taus]}")
        print(f"# ladder s=(1-tau)^2: {[round(s, 6) for s in self.scale_factors]}")
        print(f"# one temperature  : {self.temperature_k} K for every rung "
              f"(Hamiltonian scaling, not temperature REMD)")
        print(f"# omega convention : omega-selective REST2 -- "
              f"{len(excluded_bonds)} omega bond(s) left UNSCALED")
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

        move = mcmc.LangevinDynamicsMove(
            timestep=self.timestep_fs * unit.femtosecond,
            collision_rate=self.friction_per_ps / unit.picosecond,
            n_steps=self.steps_per_iteration,
            constraint_tolerance=self.constraint_tolerance,
            reassign_velocities=False)

        total_iterations = number_of_exchanges * self.exchange_stride

        storage_exists = Path(self.storage).exists()
        if (resume or extend) and not storage_exists:
            raise FileNotFoundError(
                f"{self.storage} does not exist, so there is nothing to "
                f"{'extend' if extend else 'resume'}. The NetCDF storage is authoritative; a "
                f"missing one means the run never started here.")

        if storage_exists and (resume or extend):
            sampler = StridedREST2Sampler.from_storage(self.storage)
            sampler.exchange_stride = self.exchange_stride
            reporter = sampler._reporter
            self._verify_resume(reporter, identity)
            print(f"# resumed from {self.storage} at iteration {sampler.iteration}")
            if extend:
                added = int(extend) * self.exchange_stride
                sampler.extend(added)
                total_iterations = sampler.number_of_iterations
            else:
                sampler.run()
        else:
            reporter = MultiStateReporter(
                self.storage,
                checkpoint_interval=self.checkpoint_interval,
                checkpoint_storage=(None if not self.checkpoint_storage
                                    else Path(self.checkpoint_storage).name),
                analysis_particle_indices=tuple(int(i) for i in solute_indices))

            thermodynamic_states = []
            for tau in self.taus:
                scaled = build_scaled_system(base_system, solute_indices, tau,
                                             excluded_bonds=excluded_bonds)
                thermodynamic_states.append(states.ThermodynamicState(
                    system=scaled, temperature=self.temperature_k * unit.kelvin,
                    pressure=pressure))

            common = XmlSerializer.deserialize(
                Path(self.files.coordinates).read_text(encoding="utf-8"))
            positions = common.getPositions(asNumpy=True)
            velocities = common.getVelocities(asNumpy=True)
            box = common.getPeriodicBoxVectors()
            # Positions, velocities AND box travel together: they are one configuration, and a
            # replica that kept its positions but not its box would be at a different density.
            sampler_states = [
                states.SamplerState(positions=positions, velocities=velocities,
                                    box_vectors=(None if implicit else box))
                for _ in self.taus]

            sampler = StridedREST2Sampler(
                mcmc_moves=move, number_of_iterations=total_iterations,
                replica_mixing_scheme=self.replica_mixing_scheme,
                exchange_stride=self.exchange_stride)
            context_cache = cache.ContextCache(capacity=None, time_to_live=None,
                                               platform=openmm_platform)
            for attribute in ("energy_context_cache", "sampler_context_cache"):
                setattr(sampler, attribute, context_cache)
            if properties:
                # ContextCache does not forward platform properties, so they are set on the
                # Platform itself -- which is what actually decides the device this rank uses.
                for key, value in properties.items():
                    openmm_platform.setPropertyDefaultValue(key, value)

            sampler.create(thermodynamic_states=thermodynamic_states,
                           sampler_states=sampler_states, storage=reporter)

            if self.equilibration_iterations:
                print(f"# per-tau equilibration: {self.equilibration_iterations} iteration(s) "
                      f"= {self.equilibration_duration_ps} ps per replica, "
                      f"NOT counted toward production")
                sys.stdout.flush()
                sampler.equilibrate(self.equilibration_iterations)

            sampler.run()

        self._sampler, self._reporter = sampler, reporter
        completed_iterations = int(sampler.iteration)
        expected = total_iterations
        finished = datetime.datetime.now(datetime.timezone.utc)

        summary = self._summary(sampler, reporter)
        record = {
            "format": RESTART_FORMAT,
            "run_status": "completed" if completed_iterations >= expected else "incomplete",
            "iterations_expected": expected,
            "iterations_completed": completed_iterations,
            "exchange_attempts_expected": expected // self.exchange_stride,
            "exchange_attempts_recorded": summary["exchange_attempts_recorded"],
            "production_per_replica_ps": completed_iterations * self.solute_output_interval_ps,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "storage": {
                "analysis_netcdf": os.path.basename(self.storage),
                "checkpoint_netcdf": (None if not self.checkpoint_storage
                                      else os.path.basename(self.checkpoint_storage)),
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
                "forces_scaled": [name for _, name in audit["scaled"]],
                "forces_unscaled_by_convention": [n for _, n in audit["unscaled_by_convention"]],
                "forces_energy_free": [n for _, n in audit["energy_free"]],
                "solvation": "implicit" if implicit else "explicit",
                "hydrogen_mass_amu": self.hydrogen_mass_amu,
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
            "extension": {
                "module": "rest2_openmmtools.py",
                "class": "StridedREST2Sampler",
                "overrides": ["_mix_replicas", "_mix_neighboring_replicas", "_attempt_swap",
                              "equilibrate"],
                "reasons": [
                    "openmmtools 0.26.0 swap-neighbors raises TypeError on numpy >= 1.25 "
                    "(np.where tuple indexing makes the acceptance argument a 2-d array)",
                    "exchange attempted every exchange_stride iterations so solute coordinates "
                    "can be stored more often than exchanges without fabricating frames",
                    "equilibrate() must not exchange: each replica relaxes at its own tau",
                ],
                "pinned_to": TESTED_OPENMMTOOLS_VERSION,
                "version_override_env": VERSION_OVERRIDE_ENVIRONMENT,
                "version_override_active": bool(os.environ.get(VERSION_OVERRIDE_ENVIRONMENT)),
            },
            "versions": _versions(),
            "execution": run_context,
            "acceptance": summary["acceptance"],
            "replica_thermodynamic_states_final": summary["final_mapping"],
            "inputs": {
                "topology": os.path.basename(self.files.topology),
                "system": os.path.basename(self.files.system),
                "coordinates": os.path.basename(self.files.coordinates),
                "solute": os.path.basename(self.files.solute),
                "protocol": os.path.basename(self.files.input),
            },
        }
        record["project"] = _project_identity(self.files)

        if record["run_status"] != "completed":
            raise RuntimeError(
                f"the sampler stopped at iteration {completed_iterations} of {expected}. No "
                f"completion manifest was written; the NetCDF storage holds what did run.")

        # The manifest, like the NetCDF, belongs to the RUN and not to a rank. Every rank writing
        # it is a race whose winner is arbitrary, and the loser reports a failed run that in fact
        # succeeded. Rank 0 owns it, exactly as OpenMMTools makes rank 0 own the storage.
        rank, size = mpi_rank_and_size()
        if rank == 0:
            _write_atomic(self.files.restart,
                          json.dumps(record, indent=2, sort_keys=False) + "\n")
            self._print_summary(record, summary)
            print(COMPLETION_MARKER)
        else:
            print(f"# rank {rank} of {size} finished its share of the propagation; "
                  f"rank 0 owns {os.path.basename(self.files.restart)} and the NetCDF storage")
            print(f"rank_status: completed rank={rank}")
        return record

    # -- resume verification ---------------------------------------------------------------------

    def _verify_resume(self, reporter, identity):
        """Refuse a changed scientific configuration before appending to a run.

        The comparison is against the identity the ORIGINAL run recorded in its manifest, not
        against the NetCDF: OpenMMTools stores what it owns, and the REST2 convention, the omega
        exclusions and the input digests are this repository's to check.
        """
        manifest_path = Path(self.files.restart)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} does not exist, so the scientific configuration this storage "
                f"was created with cannot be established. Refusing to append to "
                f"{self.storage} on trust.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        before = previous.get("scientific_identity") or {}
        differences = [key for key in sorted(set(before) | set(identity))
                       if before.get(key) != identity.get(key)]
        if differences:
            lines = "\n".join(
                f"    {key}: was {before.get(key)!r}, now {identity.get(key)!r}"
                for key in differences)
            raise ValueError(
                f"the scientific configuration changed since this run was created, so continuing "
                f"it would append samples from a different Hamiltonian to the same storage:\n"
                f"{lines}\n"
                f"  Start a new run instead. Running longer is a legitimate extension; changing "
                f"the physics is not.")
        stored = previous.get("versions", {}).get("openmmtools")
        current = _versions()["openmmtools"]
        if stored and stored != current:
            raise ValueError(
                f"this storage was written by openmmtools {stored} and the installed version is "
                f"{current}. Exchange scheduling and restart determinism are version-dependent; "
                f"refusing to continue across a change.")

    # -- reporting --------------------------------------------------------------------------------

    def _summary(self, sampler, reporter):
        proposed = np.asarray(sampler._n_proposed_matrix)
        accepted = np.asarray(sampler._n_accepted_matrix)
        pairs = []
        for i in range(self.number_of_replicas - 1):
            j = i + 1
            n_proposed = int(proposed[i, j])
            n_accepted = int(accepted[i, j])
            pairs.append({
                "state_pair": [i, j], "tau_pair": [self.taus[i], self.taus[j]],
                "proposed": n_proposed, "accepted": n_accepted,
                "acceptance": (n_accepted / n_proposed) if n_proposed else None,
            })
        try:
            mapping = reporter.read_replica_thermodynamic_states()
            recorded = int(mapping.shape[0])
            round_trips = _state_round_trips(np.asarray(mapping), self.number_of_replicas)
        except Exception:
            mapping, recorded, round_trips = None, None, None
        return {
            "acceptance": {"by_neighbouring_pair": pairs,
                           "note": ("counts are for the LAST exchange attempt only: OpenMMTools "
                                    "resets the proposal matrices each mixing call. The NetCDF "
                                    "holds the full per-iteration history.")},
            "exchange_attempts_recorded": int(sampler.iteration) // self.exchange_stride,
            "final_mapping": [int(x) for x in sampler._replica_thermodynamic_states],
            "mapping_rows": recorded,
            "round_trips": round_trips,
        }

    def _print_summary(self, record, summary):
        print()
        print("# --- REST2 summary (readable; the NetCDF is authoritative) -------------------")
        print(f"# iterations completed : {record['iterations_completed']}")
        print(f"# exchange attempts    : {record['exchange_attempts_recorded']}")
        print(f"# production/replica   : {record['production_per_replica_ps']} ps")
        print(f"# final replica->state : {record['replica_thermodynamic_states_final']}")
        if summary["round_trips"] is not None:
            print(f"# state round trips    : {summary['round_trips']}")
        print("# neighbouring-pair acceptance (last attempt):")
        for pair in summary["acceptance"]["by_neighbouring_pair"]:
            rate = pair["acceptance"]
            shown = "n/a" if rate is None else f"{rate:.3f}"
            print(f"#   states {pair['state_pair'][0]}-{pair['state_pair'][1]} "
                  f"(tau {pair['tau_pair'][0]:.3f}-{pair['tau_pair'][1]:.3f}): "
                  f"{pair['accepted']}/{pair['proposed']} = {shown}")
        print(f"# manifest             : {os.path.basename(self.files.restart)}")
        print("# ---------------------------------------------------------------------------")


def _state_round_trips(mapping, n_states):
    """How many times each walker travelled from state 0 to the top and back."""
    trips = []
    top = n_states - 1
    for walker in range(mapping.shape[1]):
        series = mapping[:, walker]
        seen_end, count = None, 0
        for value in series:
            if value == 0:
                if seen_end == "top":
                    count += 1
                seen_end = "bottom"
            elif value == top:
                seen_end = "top" if seen_end != "top" else seen_end
        trips.append(int(count))
    return trips


def _project_identity(files):
    """The generating project and template identity, read from the files the project already has.

    Read rather than recomputed: `md.config.yaml` / `config.yaml` recorded the commit at
    generation, and a run months later must not try to re-derive it from whatever is checked out.
    """
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
