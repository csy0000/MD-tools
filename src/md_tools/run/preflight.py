"""Everything that must be true before a run may write anything at all.

ONE model, CONSUMED by every authoritative entry point:

    md_tools.run.main.md_run_main          the CLI dispatcher
    md_tools.md.stage.stage_main           generated stage scripts
    md_tools.build.md.build_scripts        the whole chain, at generation time
    md_tools.remd.generated.replica_main   generated REST2.py
    md_tools.ais.run.ais_main              generated AIS.py

That list is the reason this module exists. The guards began life inside `md-run`, which made them
a property of ONE entry point rather than of the runtime -- and the generated scripts call the
runtimes directly. A generated `REST2.py` could create its output directory, write `solute.yaml`,
`_protocol.py` and a group file, open a log, and only then discover that the machine configuration
was malformed. A safe outer wrapper over an unsafe runtime is worse than no wrapper, because it
makes the unsafe path look tested.

CONSUMED, not merely called. Each runtime uses the result -- the coordination, the machine
settings, the resolved platform -- rather than resolving any of it a second time. Two
implementations of one policy is how the two come to disagree, and the one that runs is never the
one that was fixed.

ORDER MATTERS AS MUCH AS THE CHECKS. Nothing below touches the filesystem, and all of it happens
before `mkdir`, a `SimulationOutput`, a `LogWriter`, `resolved.config`, `solute.yaml`,
`_protocol.py`, a group file, a checkpoint, a trajectory or a restart:

     1. flags parsed with `allow_abbrev=False`; contradictions and mandatory values checked
     2. MPI coordination opened for the parallel modes, failing closed without `mpi4py`
     3. paths normalised
     4. required inputs are regular files
     5. a declared trajectory's format is read from its CONTENT
     6. output-output and input-output collisions, on RESOLVED paths
     7. the machine configuration, loaded strictly -- absent is not invalid
     8. device policy and rank placement
     9. the requested OpenMM platform exists
    10. a Context is actually created on it
    11. cheap topology/System compatibility, so a mismatched pair is refused here

Reading, hashing, importing and opening an ephemeral one-particle CUDA Context are not mutations.

A refusal leaves the filesystem exactly as it was -- not merely "creates no new directory", but
does not touch what is already there. A `-odir` holding a `resolved.config` is indistinguishable
from a run that started, and a rerun that damages a previous run's results while refusing has
destroyed the thing it was protecting.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

__all__ = ["PreflightError", "ExecutionPreflight", "StagePreflight", "LadderPreflight",
           "AISPreflight", "preflight_stage", "preflight_ladder", "preflight_ais",
           "check_output_collisions", "check_input_files", "resolve_launch",
           "check_topology_matches_system", "LoadedInputs", "load_inputs", "PendingParent",
           "check_ensemble", "check_scaling_plan", "report_check", "collectively",
           "OutputInventory", "check_existing_outputs", "reject_plural_launch",
           "reject_contradictory_continuation"]


class PreflightError(SystemExit):
    """A run that cannot start. A `SystemExit`, so it stops the process without a traceback."""


# ---------------------------------------------------------------------------------------------
# the individual checks
# ---------------------------------------------------------------------------------------------

#: Inventory roles that ARE command-line flags, so a refusal can spell them as the person typed
#: them. Everything else is a derived output and is named plainly -- "-state_trajectory_0" would
#: be a flag that does not exist.
FLAG_ROLES = frozenset({"o", "out", "log", "x", "trajectory", "r", "restart", "c", "chk",
                        "checkpoint", "p", "s", "source-traj", "groupfile"})


def _role(name: str) -> str:
    return f"-{name}" if name in FLAG_ROLES else name


def check_output_collisions(*, outputs: dict[str, str | None],
                            inputs: dict[str, str | None] | None = None) -> None:
    """No two named outputs may be the same file, and no output may be an input.

    Compared as RESOLVED paths, so `sub/../run.out` and `./run.out` are recognised as one file.
    String comparison misses that, and the two would then be opened twice with one writer
    truncating the other -- leaving a file whose surviving half nobody can identify.

    The input case is the sharper one: `-log ../built.xml` writes a log over the System the run is
    about to read.

    `resolve(strict=False)` because outputs do not exist yet, and requiring them to would disable
    the check exactly when it matters.
    """
    seen: dict[Path, str] = {}
    for name, value in outputs.items():
        if not value:
            continue
        resolved = Path(value).resolve(strict=False)
        if resolved in seen:
            first, second = sorted((seen[resolved], name))
            raise PreflightError(
                f"{_role(first)} and {_role(second)} are the same file ({resolved}). They are "
                f"different outputs with different readers; one file cannot be both, and opening "
                f"it twice would have one of them truncate the other.")
        seen[resolved] = name

    for name, value in (inputs or {}).items():
        if not value:
            continue
        resolved = Path(value).resolve(strict=False)
        if resolved in seen:
            raise PreflightError(
                f"{_role(seen[resolved])} would be written over the input {_role(name)} "
                f"({resolved}). The run reads that file; writing an output onto it destroys what "
                f"the run needs.")


def check_input_files(**paths: str | None) -> None:
    """Every named input must exist and be a regular file. A directory is named as one."""
    for name, value in paths.items():
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            continue
        # `is_file`, not `exists`: a path written with a trailing slash names a directory, and
        # `exists` would accept it and fail obscurely several layers down.
        what = "is a directory, not a file" if path.is_dir() else "does not exist"
        raise PreflightError(f"-{name} {path} {what}")


def check_topology_matches_system(topology: str | Path, system: str | Path) -> int:
    """The PDB and the serialised System must describe the same particles. Returns the count.

    Cheap, and worth doing here: a mismatched pair produces a Context that cannot be built, deep
    inside OpenMM, after the log and the output directory already exist.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    try:
        atoms = PDBFile(str(topology)).topology.getNumAtoms()
    except Exception as unreadable:                       # noqa: BLE001 - reported below
        raise PreflightError(f"-p {topology} could not be read as a PDB: "
                             f"{type(unreadable).__name__}: {unreadable}") from None
    try:
        particles = XmlSerializer.deserialize(
            Path(system).read_text(encoding="utf-8")).getNumParticles()
    except Exception as unreadable:                       # noqa: BLE001 - reported below
        raise PreflightError(f"-s {system} could not be read as a serialised OpenMM System: "
                             f"{type(unreadable).__name__}: {unreadable}") from None

    if atoms != particles:
        raise PreflightError(
            f"-p {Path(topology).name} has {atoms} atom(s) but -s {Path(system).name} has "
            f"{particles} particle(s). These two files describe different systems; every "
            f"coordinate the run reads would be attributed to the wrong particle.")
    return atoms


def resolve_launch(*, number_of_groups: int | None = None, replicas: int | None = None,
                   protocol: str = "this run"):
    """The MPI launch, validated. Returns the one `Coordination` the run will use.

    Delegates entirely to `md_tools.remd.mpi`, the only MPI authority. Nothing here decides what
    to do about a missing `mpi4py`; that module refuses, and this passes the refusal on.
    """
    from ..remd.mpi import Coordination

    return Coordination.open(number_of_groups=number_of_groups, replicas=replicas,
                             protocol=protocol)


def _check_command_line(*, cpu: bool, device: Any, number_of_groups: int | None) -> None:
    """Contradictions knowable from the command line and the environment alone.

    These come first because they need neither the filesystem nor a library: somebody debugging a
    launch script should be told about the launch, not about their paths.
    """
    if cpu and device is not None:
        raise PreflightError(
            "--cpu and --device are contradictory: the CPU platform has no device to place. "
            "--device says which GPU a CUDA run goes to, never whether it is one.")

    from ..remd.mpi import launcher_rank_and_size

    _, launcher_size = launcher_rank_and_size()
    if number_of_groups is not None and int(number_of_groups) > 1 and launcher_size == 1:
        raise PreflightError(
            f"-ng {number_of_groups} was requested but this process was not started by an MPI "
            f"launcher: the world size is 1.\n"
            f"  -ng says how many processes coordinate; it does not create them.\n"
            f"  mpirun -n {number_of_groups} ... -ng {number_of_groups}")


def _resolve_machine(machine_config: str | Path | None) -> dict[str, Any]:
    """The machine settings, strictly. An ABSENT file is valid; an INVALID one is fatal."""
    from ..registry.errors import RegistrationError
    from ..registry.userconfig import machine_openmm_settings

    try:
        return machine_openmm_settings(machine_config)
    except RegistrationError as invalid:
        # Re-raised as a PreflightError so a person sees the message rather than a traceback: a
        # broken configuration is their mistake in a file they edited, and a stack trace makes it
        # look like ours.
        raise PreflightError(str(invalid)) from None


def _resolve_platform(machine: dict[str, Any], *, cpu: bool, device: Any,
                      plan) -> tuple[Any, Any, str]:
    """Platform and device, CONSUMED from the launch plan. Proves a Context can be created.

    The device was decided by `md_tools.openmm.placement` from facts every rank gathered; this
    only turns the decision into a Platform. Deciding it again here, per rank, is how round-robin
    and a planner would come to disagree about where rank 3 runs.
    """
    from ..openmm.platform_policy import PlatformRequest, resolve_platform_request

    request = PlatformRequest.from_machine(machine, cpu=cpu)
    mine = plan.for_rank()
    index = mine["device"] if request.name == "CUDA" else None
    if request.name != "CUDA":
        detail = "not a CUDA platform"
    else:
        detail = plan.placement
        if plan.workers > 1 and index is not None:
            detail += (f" (rank {plan.this_rank} of {plan.workers}, local rank "
                       f"{mine['local_rank']} on {mine['host']}, {mine['co_tenants']} worker(s) "
                       f"on this device)")
    threads = None
    if request.name == "CPU" and plan.workers > 1 and not os.environ.get("OPENMM_CPU_THREADS"):
        # A bound worker running a pool sized for the whole host oversubscribes its own block.
        threads = len(mine["cpus"])
    return (resolve_platform_request(request, device_index=index, cpu_threads=threads),
            index, detail)


def _plan_placement(coordination, machine: dict[str, Any], *, cpu: bool, device: Any,
                    loaded, topology, system, protocol: str):
    """CPUs, devices and MPS for every worker of this launch. Collective; refuses before output.

    In order, because each step needs the one before:

      1. every rank reads its own facts -- affinity, cgroup quota, visible devices, MPS
      2. the facts are gathered and the CPU rule checked: identical arithmetic on every rank, so
         a refusal is every rank's refusal without a hang
      3. each rank binds itself to its block, before any Context exists, so CUDA's host threads
         inherit the binding and the measurement below runs on the CPUs the run will use
      4. the first rank on each host measures that host's devices with THIS run's System
      5. the plan is made from the gathered measurement, identically on every rank
      6. a rank on a shared device proves it is an MPS client, or the launch is refused
    """
    from ..openmm import placement as placing
    from ..openmm.platform_policy import PlatformRequest
    from ..remd.engine import visible_cuda_devices

    request = PlatformRequest.from_machine(machine, cpu=cpu)
    policy = str(machine.get("device_policy") or "local_rank")
    cuda = request.name == "CUDA"

    def _facts():
        _fail_here_if_asked(coordination)
        count = (len(visible_cuda_devices(probe=coordination.size > 1))
                 if cuda and device is None and policy != "openmm" else 0)
        return placing.read_worker_facts(coordination.rank, visible_devices=count)

    facts = coordination.allgather(collectively(coordination, _facts,
                                                what=f"the {protocol} placement facts"))
    try:
        cpus = placing.plan_cpus(facts, where=protocol)
    except placing.PlacementError as refused:
        raise PreflightError(str(refused)) from None

    def _bind():
        try:
            return placing.bind_worker(cpus[coordination.rank]["cpus"])
        except OSError as failure:
            raise PreflightError(f"{protocol}: this rank could not be bound to CPUs "
                                 f"{cpus[coordination.rank]['cpus']}: {failure}") from None

    collectively(coordination, _bind, what=f"the {protocol} CPU binding")

    hosts = placing.needs_measurement(facts, platform=request.name, device_policy=policy,
                                      explicit_device=device)
    mine = cpus[coordination.rank]

    def _measure():
        if mine["host"] not in hosts or mine["local_rank"] != 0:
            return None
        if loaded is None and not (topology and system):
            raise PreflightError(
                f"{protocol}: placement by measured throughput needs this run's System to "
                f"measure with, and none was given to the preflight.")
        pair = loaded if loaded is not None else load_inputs(topology, system)
        try:
            return placing.measure_device_throughput(
                pair.system, pair.pdb.positions, devices=hosts[mine["host"]],
                precision=request.precision)
        except Exception as failure:                       # noqa: BLE001 - reported as refusal
            raise PreflightError(
                f"{protocol}: measuring device throughput on {mine['host']} failed: "
                f"{type(failure).__name__}: {failure}. Placement is by measured throughput; "
                f"without a measurement there is nothing to place by.") from None

    measured = {}
    for rank, piece in enumerate(coordination.allgather(
            collectively(coordination, _measure, what=f"the {protocol} device measurement"))):
        if piece is not None:
            measured[cpus[rank]["host"]] = piece
    try:
        plan = placing.plan_launch(
            facts, platform=request.name, device_policy=policy,
            explicit_device=None if device is None else int(device),
            throughput={host: m["steps_per_second"] for host, m in measured.items()} or None,
            where=protocol)
    except placing.PlacementError as refused:
        raise PreflightError(str(refused)) from None
    plan = replace(plan, this_rank=coordination.rank, measurement=measured or None)

    mps = next(f.mps for f in facts if f.rank == coordination.rank)
    if cuda and plan.shared_devices:
        def _verify():
            status = _verify_mps(plan.for_rank()["device"], request.precision, mps)
            try:
                placing.refuse_unverified_sharing(plan, status, rank=coordination.rank,
                                                  where=protocol)
            except placing.PlacementError as refused:
                raise PreflightError(str(refused)) from None
            return status

        mps = collectively(coordination, _verify, what=f"the {protocol} MPS verification")
    return replace(plan, mps=mps)


def _verify_mps(device, precision: str, status):
    """Hold a Context on this rank's device and ask the driver whether we are an MPS client."""
    import os as _os

    from openmm import Context, Platform, System, VerletIntegrator, unit

    from ..openmm import placement as placing

    if _os.environ.get(placing.FORCE_MPS_VERDICT):
        verdict = _os.environ[placing.FORCE_MPS_VERDICT]
        return status.with_verification({"verified": True, "not-a-client": False}.get(verdict),
                                        f"{placing.FORCE_MPS_VERDICT}={verdict} (test seam)")
    system = System()
    system.addParticle(1.0 * unit.amu)
    properties = {"Precision": precision}
    if device is not None:
        properties["DeviceIndex"] = str(device)
    try:
        context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("CUDA"),
                          properties)
    except Exception as failure:                           # noqa: BLE001 - reported below
        return status.with_verification(None, f"no Context to verify MPS with: {failure}")
    try:
        verdict, why = placing.verify_mps_client(_os.getpid(), xml=placing.nvidia_smi_xml())
    finally:
        del context
    return status.with_verification(verdict, why)


# ---------------------------------------------------------------------------------------------
# loading the pair once, and the refusals that need it
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedInputs:
    """The topology and the System, deserialised ONCE and handed to whoever needs them.

    Every refusal that has to look inside the System -- the barostat, the masses behind a 4 fs
    timestep, the force classification, the atom counts -- needs the deserialised object. Loading
    it here and passing it on is what lets those refusals happen before the first mkdir instead
    of after it, and it also stops the runtime deserialising the same megabytes a second time
    only to reach a different conclusion.
    """

    topology_path: Path
    system_path: Path
    pdb: Any
    system: Any
    particles: int
    implicit: bool
    barostats: int

    @property
    def periodic(self) -> bool:
        return not self.implicit


def load_inputs(topology, system, *, flags=("-p", "-s")) -> LoadedInputs:
    """Read `-p` and `-s`, and refuse a pair that does not describe the same particles.

    `flags` names the pair in a refusal. AIS reads a second pair, `-p2`/`-s2`, and a message about
    the wrong flag sends a person to fix a file that is fine.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from ..md._stages import count_barostats

    topology_path, system_path = Path(topology), Path(system)
    try:
        pdb = PDBFile(str(topology_path))
    except Exception as broken:
        raise PreflightError(f"{flags[0]} {topology_path} is not a PDB this build can read: "
                             f"{type(broken).__name__}: {broken}") from None
    try:
        base = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
    except Exception as broken:
        raise PreflightError(f"{flags[1]} {system_path} is not a serialised OpenMM System: "
                             f"{type(broken).__name__}: {broken}") from None

    atoms, particles = pdb.topology.getNumAtoms(), base.getNumParticles()
    if atoms != particles:
        raise PreflightError(
            f"{topology_path.name} has {atoms} atom(s) but {system_path.name} has {particles} "
            f"particle(s). They must describe the same system; a mismatched pair produces "
            f"coordinates assigned to the wrong particles and no error at all.")
    return LoadedInputs(topology_path=topology_path, system_path=system_path, pdb=pdb,
                        system=base, particles=particles,
                        implicit=not base.usesPeriodicBoundaryConditions(),
                        barostats=count_barostats(base))


@dataclass(frozen=True)
class OutputInventory:
    """EVERY file and directory a run will create or modify, named before any of it exists.

    `check_output_collisions` compared the handful of paths that arrive as flags. That is not the
    inventory: a ladder also writes `solute.yaml`, `_protocol.py`, a group file, `rem.log`, N
    per-state trajectories and a restart manifest, and an AIS run writes a global work table, a
    summary, a selected-frames table, a run identity, and per path a directory, a staged
    trajectory, two CSVs, a checkpoint generation tree and a completion manifest. None of those
    were checked against anything, so `--overwrite` -- which is supposed to say "yes, replace what
    is there" -- governed `resolved.config` alone and every other file was replaced silently
    whether it was asked for or not.

    `roles` maps a short name to the path, so a refusal can say WHICH output is in the way rather
    than printing a list and leaving the reader to work it out.
    """

    roles: dict[str, Path]
    #: Paths whose existence is expected and harmless -- a directory a person made themselves, a
    #: checkpoint tree a `--resume` is going to read. Existing-output policy skips these.
    resumable: frozenset[str] = frozenset()

    def existing(self) -> dict[str, Path]:
        return {role: path for role, path in sorted(self.roles.items())
                if role not in self.resumable and Path(path).exists()}

    def record(self) -> dict[str, str]:
        return {role: str(path) for role, path in sorted(self.roles.items())}


def check_existing_outputs(inventory: OutputInventory, *, overwrite: bool = False,
                           resume: bool = False, where: str = "this run") -> None:
    """Refuse to overwrite an existing run unless told to, for the WHOLE inventory.

    A `-odir` that already holds outputs is a run that happened. Writing into it produces a
    directory that is half one run and half another, and every file in it looks equally current.
    `--resume` is the other legitimate answer: it says the caller means to continue THAT run, and
    the identity checks downstream then decide whether they may.
    """
    if overwrite or resume:
        return
    present = inventory.existing()
    if not present:
        return
    listed = "\n".join(f"    {role:<20} {path}" for role, path in list(present.items())[:8])
    more = f"\n    ... and {len(present) - 8} more" if len(present) > 8 else ""
    raise PreflightError(
        f"{where}: {len(present)} output(s) already exist:\n{listed}{more}\n"
        f"  This directory already holds a run. Writing into it would leave a tree that is half "
        f"one run and half another, with every file looking equally current.\n"
        f"  Pass --overwrite to replace them, --resume to continue that run, or choose another "
        f"-odir.")


@dataclass(frozen=True)
class PendingParent:
    """A `-c` that does not exist yet BECAUSE an earlier stage of this same chain will write it.

    The one legitimate missing-continuation case, and it is represented rather than inferred. The
    old rule was "a `-c` that does not exist is not checked", which is the same sentence as "a
    typo in `-c` is not checked": under `--check` it was the intended leniency and under a real
    run it silently dropped the restart the stage was supposed to continue from, so the stage
    started from the topology's coordinates and finished successfully.

    A caller that constructs one of these is stating WHICH stage produces the file. Nothing else
    excuses a missing `-c`.
    """

    path: Path
    produced_by: str


def check_ensemble(loaded: LoadedInputs, *, ensemble: str | None, tau: float = 0.0,
                   where: str = "this stage") -> None:
    """Pressure coupling is refused where it has no meaning, before any output exists.

    Two cases, both of which used to be found after the output directory had been created:

      implicit solvent has no box, so there is no volume to control and no barostat to control it;
      a scaled run (tau > 0) is fixed-volume throughout, because a Monte Carlo volume move under a
      scaled Hamiltonian is not the move the ensemble is defined by.
    """
    if ensemble is None:
        return
    ensemble = str(ensemble).upper()
    if ensemble not in ("NVT", "NPT", "NVE"):
        raise PreflightError(f"{where}: ensemble {ensemble!r} is not one of NVT, NPT, NVE")
    if ensemble != "NPT":
        return
    if loaded.implicit:
        raise PreflightError(
            f"{where} asks for NPT, but the System has no periodic box: implicit solvent has no "
            f"volume to control. Implicit stages are NVT and are named for it; an NPT stage with "
            f"the pressure quietly ignored reports an ensemble it did not sample.")
    if float(tau) > 0.0:
        raise PreflightError(
            f"{where} asks for NPT at tau = {tau}. A scaled run is fixed-volume throughout, "
            f"equilibration included: the barostat's volume move is defined for the unscaled "
            f"Hamiltonian, and accepting it at tau > 0 samples neither ensemble.")


def check_scaling_plan(loaded: LoadedInputs, *, solute_indices, excluded_bonds=(), tau=0.0,
                       where: str = "this run"):
    """Classify every force and BUILD the scaled System, here, where failing costs nothing.

    `audit_force_classes` refuses an energy-bearing force the convention cannot place, and
    `build_scaled_system` is where a CustomGBForce over a partial enhanced region, an unreadable
    CMAP map or any other construction failure surfaces. Both used to happen after the run
    directory, the log and `solute.yaml` existed, so the refusal arrived attached to a tree that
    looks exactly like a run that started.

    The constructed System is returned, so the runtime uses this one rather than building a second
    that could differ.
    """
    from ..rest2.scaler import UnclassifiedForceError, audit_force_classes, build_scaled_system

    try:
        audit = audit_force_classes(loaded.system, where=where)
    except UnclassifiedForceError as unknown:
        raise PreflightError(str(unknown)) from None
    try:
        scaled = build_scaled_system(loaded.system, solute_indices, float(tau), excluded_bonds)
    except Exception as broken:
        raise PreflightError(
            f"{where}: the scaled System at tau = {tau} could not be constructed: "
            f"{type(broken).__name__}: {broken}") from None
    return audit, scaled


# ---------------------------------------------------------------------------------------------
# the results: one type per mode, so a field's meaning never depends on who is reading it
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionPreflight:
    """What every mode establishes. A run holding one of these may create output; nothing else may.

    Deliberately not a dictionary with optional keys. A `machine` that is sometimes absent, or a
    `coordination` that is sometimes None-meaning-serial and sometimes None-meaning-unchecked, is
    a type whose meaning depends on the caller -- and the caller who reads it wrong is the one
    running eight ranks over one set of files.
    """

    coordination: Any
    machine: dict[str, Any]
    acceleration: Any
    device_index: Any
    device_policy: str
    device_policy_detail: str
    #: Every worker's CPUs and device, the measurement they were placed by, and MPS. From
    #: `md_tools.openmm.placement`, the one placement authority.
    placement: Any = None
    particles: int | None = None
    #: The deserialised pair, when the mode needed to look inside it. Handed to the runtime so it
    #: consumes this System rather than deserialising a second one that could differ.
    loaded: LoadedInputs | None = None
    #: The resolved timestep record, including how `auto` was decided and what the masses proved.
    timestep: dict[str, Any] | None = None
    #: Every file and directory this run will create or modify.
    inventory: OutputInventory | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def platform_name(self) -> str:
        return self.acceleration.name

    def record(self) -> dict[str, Any]:
        """The acceleration block for the run's machine record, already assembled."""
        from ..openmm.platform_policy import acceleration_record

        return dict(
            acceleration_record(
                self.acceleration, mpi_rank=self.coordination.rank,
                mpi_size=self.coordination.size,
                # The rank on this NODE. This field used to carry the device index.
                local_rank=(self.placement.for_rank()["local_rank"]
                            if self.placement is not None else None)),
            device_policy_detail=self.device_policy_detail,
            placement=self.placement.record() if self.placement is not None else None)


@dataclass(frozen=True)
class StagePreflight(ExecutionPreflight):
    """One conventional stage: a trajectory, a restart, a checkpoint, an out and a log.

    Carries the FULLY PREPARED System -- scaled for fixed tau if asked, restrained, and
    barostatted -- so `stage_main` constructs a Simulation from it rather than rebuilding the
    scientific policy after its reports are open. Every refusal that construction can produce
    then happens before the first file exists.
    """

    trajectory: Path | None = None
    #: The System a Simulation is built from: `build_scaled_system` at this stage's tau, then the
    #: positional restraint, then the barostat -- in that order, which is the order the REST2
    #: ladder uses. Scaling last would scale the restraint and the barostat, and a fixed-tau
    #: walker would construct a different System from the rung it is meant to match.
    prepared_system: Any = None
    solute: tuple = ()
    excluded_bonds: tuple = ()
    implicit: bool = False
    seed: int = 0
    restrained: bool = False
    #: The molecular Hamiltonian's identity, taken BEFORE the restraint and barostat are added,
    #: for the phase-space stream it records.
    hamiltonian_identity: dict[str, Any] | None = None


@dataclass(frozen=True)
class LadderPreflight(ExecutionPreflight):
    """A REST2 ladder: one process per thermodynamic state."""

    replicas: int = 0
    #: How every energy-bearing force was classified, and the scaled System built from it. Both
    #: are the products of `check_scaling_plan`, which is where an unplaceable force is refused.
    force_audit: dict[str, Any] | None = None
    scaled_system: Any = None
    excluded_bonds: tuple = ()
    #: The solute selection this ladder scales, as resolved here. Carried as a field rather than
    #: left in `notes` so the executor can CONSUME it instead of re-deriving it from `solute.yaml`.
    solute_indices: tuple = ()
    #: The ladder's exact tau list, rung 0 first. Nothing carried this before, so "which
    #: Hamiltonian schedule is this" was answerable only by rebuilding it.
    tau_list: tuple = ()
    #: ONE PREPARED SYSTEM PER RUNG, in tau order, built before any output exists and consumed by
    #: the driver. The preflight used to build ONLY the top rung -- to validate the force layout --
    #: and the driver then called `protocol.build_systems` again at runtime, so rungs 0..N-2 were
    #: never validated until propagation was about to start, with every output file already
    #: created. An unclassifiable force in a middle rung surfaced attached to a tree that looks
    #: exactly like a run that started.
    rung_systems: tuple = ()
    #: The torsion restraints every rung carries, resolved here against the collective-variable
    #: definition and already built into `rung_systems`. Carried so the run RECORDS what it was
    #: biased by: `_protocol.py` names the definition file, and the resolved torsions -- which
    #: four atoms, which centre, which force constant -- exist only here. Empty unless declared.
    ladder_restraints: tuple = ()
    #: The validated collective-variable definition, parsed before any output exists so a
    #: malformed cv.yaml refuses the run rather than failing at the first observation.
    cv_definition: Any = None


@dataclass(frozen=True)
class AISPreflight(ExecutionPreflight):
    """AIS switching paths: independent workers, and a source ensemble that already exists."""

    source: Path | None = None
    source_format: str | None = None
    #: Everything the path loop needs, decided before the run directory exists.
    schedule: dict[str, Any] | None = None
    #: V1, `-p2`/`-s2`, loaded and checked against V0 (`loaded`).
    end_state_1: Any = None
    #: `md_tools.ais.two_state.TwoStateHamiltonian`: V0 and V1 mixed, built once, audited.
    hamiltonian: Any = None
    #: Whether the source ensemble was shown to be V0's: "verified", or "asserted" with the reason.
    source_ensemble: dict[str, Any] | None = None
    chosen_frames: tuple = ()
    eligible_frames: int = 0
    source_frames: int = 0
    source_atoms: int | None = None
    #: The source trajectory's digest and size, computed ONCE here. The runtime reuses it for the
    #: record and for the fingerprint rather than reading a production-sized file a second time.
    source_facts: dict[str, Any] | None = None
    #: The per-path checkpoint fingerprint and the whole-directory identity document, both built
    #: before anything exists -- see `disposition`.
    fingerprint: str | None = None
    identity: dict[str, Any] | None = None
    #: What this invocation is: "fresh", "verify", "resume", "overwrite". Decided by inspecting
    #: the output directory READ-ONLY and agreed across ranks, so an incompatible run refuses
    #: without having changed a byte. "verify" is a compatible run whose every selected path
    #: already verifies complete -- distinct from "resume" so the boolean handed to
    #: `run_one_path` is never "resume" when `--resume` was not actually given.
    disposition: str = "fresh"
    #: The validated collective-variable definition, parsed before `-odir` exists. Its own field
    #: rather than a key in `schedule`, which is serialised into the run records.
    cv_definition: Any = None
    #: The ALREADY-VALIDATED `AIS_run.json` this directory held, when one did. Carried forward so
    #: `clear_run_directory` cleans up against the identity that was verified here, rather than
    #: re-reading (and re-verifying) the file a second time after `-odir` exists.
    previous_identity: dict[str, Any] | None = None


# ---------------------------------------------------------------------------------------------
# the mode-aware entry points
# ---------------------------------------------------------------------------------------------

def _continuation_inputs(coordinates, pending: PendingParent | None, *, where: str):
    """`-c`, checked. A missing one is an error unless a named earlier stage will write it.

    The old rule tested `Path(c).exists()` and skipped the check when it did not. That reads as
    leniency for a chain whose parents do not exist yet, and it is also complete leniency for a
    typo: a
    real run given `-c eq_npt_fre.xml` found nothing to check, started from the topology's
    coordinates, and completed. The pending case is now something a caller STATES, naming the
    stage that produces the file, so nothing else can fall through it.
    """
    if not coordinates:
        return {}
    path = Path(coordinates)
    if path.exists():
        return {"c": coordinates}
    if pending is not None and Path(pending.path) == path:
        return {}                       # produced by `pending.produced_by`, earlier in this chain
    raise PreflightError(
        f"{where}: -c {path} does not exist. A continuation names the restart this run starts "
        f"from; if it is missing the run does not continue anything, it silently starts from the "
        f"coordinates in -p and finishes looking successful."
        + (f"\n  ({pending.path} is the one file this chain may still be missing, and it is not "
           f"this one.)" if pending is not None else ""))


#: Test seam: fail the platform probe on exactly these ranks, as a comma-separated list. A rank
#: whose GPU is held by another job, or whose CUDA context cannot initialise, fails alone and
#: exactly here -- and that is the case whose collective handling has to be provable without
#: arranging for a real GPU to be unavailable on one rank of a live launch.
FAIL_RANKS_ENVIRONMENT = "MD_TOOLS_FAIL_PREFLIGHT_ON_RANKS"


def collectively(coordination, thunk, *, what: str):
    """Run a rank-local check, then AGREE about it. One rank's refusal becomes every rank's.

    This is the difference between a stopped job and a hung one. Every check below is rank-local
    -- a GPU held by another job, a device that will not initialise, a path visible on one node
    and not another -- so without an agreement step the failing rank raises and exits while every
    other rank walks on to the next collective and waits there for a participant that has already
    gone. The launcher then reports nothing, and the job occupies its GPUs until a wall clock
    kills it.

    `allgather` rather than a bare `all_agree`: every rank ends up holding every rank's message,
    so the error a person reads names WHICH ranks failed and why, from whichever rank's output
    they happen to look at first.
    """
    error = None
    result = None
    try:
        result = thunk()
    except PreflightError as refusal:
        error = str(refusal)

    if coordination.size <= 1:
        if error:
            raise PreflightError(error)
        return result

    reports = coordination.allgather(error)
    failed = [(rank, message) for rank, message in enumerate(reports) if message]
    if not failed:
        return result
    if len(failed) == len(reports):
        # Everyone hit it: it is not rank-local, so say it once rather than N times.
        raise PreflightError(failed[0][1])
    detail = "\n".join(f"  rank {rank} of {coordination.size}: {message}"
                        for rank, message in failed)
    raise PreflightError(
        f"{what} failed on {len(failed)} of {coordination.size} rank(s), so the whole launch is "
        f"refused:\n{detail}\n"
        f"  Every rank stops. A rank that failed alone would leave the rest of the ladder waiting "
        f"at the next collective for a participant that has already exited.")


def _fail_here_if_asked(coordination) -> None:
    """Honour the rank-failure test seam. A no-op in every normal run."""
    import os

    wanted = os.environ.get(FAIL_RANKS_ENVIRONMENT)
    if not wanted:
        return
    ranks = {int(part) for part in str(wanted).replace(",", " ").split()}
    if coordination.rank in ranks:
        raise PreflightError(
            f"{FAIL_RANKS_ENVIRONMENT} names this rank: simulating a rank-local platform failure "
            f"on rank {coordination.rank} of {coordination.size}.")


def _common(*, topology, system, outputs, inputs, cpu, device, number_of_groups, replicas,
            protocol, machine_config, check_particles=True, load=False, serial=False):
    """Steps 1-11, in order. Shared by every mode; each mode adds only its own inputs."""
    _check_command_line(cpu=cpu, device=device, number_of_groups=number_of_groups)

    coordination = resolve_launch(number_of_groups=number_of_groups, replicas=replicas,
                                  protocol=protocol)

    check_input_files(p=topology, s=system, **dict(inputs or {}))
    check_output_collisions(outputs=outputs or {},
                            inputs={"p": topology, "s": system, **(inputs or {})})

    machine = _resolve_machine(machine_config)

    def _inputs():
        # Loaded once when the mode has refusals that need to look inside the System; the
        # particle comparison then comes from the loaded pair rather than a second parse.
        # NO SYSTEM TO LOAD OR TO COMPARE AGAINST on a grouped launch.
        #
        # A REST2 ladder driven by a group file names one System PER LINE -- each rung's
        # own pre-scaled Hamiltonian -- so there is no single `-s` for this launch, and
        # `remd.executor` validates those paths where it reads them. Passing `None` down here
        # would have `check_topology_matches_system` call `PDBFile(str(None))` and report a
        # missing file for a flag nobody was asked to give.
        prepared = load_inputs(topology, system) if (load and system) else None
        if prepared is not None:
            count = prepared.particles
        elif check_particles and system:
            count = check_topology_matches_system(topology, system)
        else:
            count = None
        return prepared, count

    loaded, particles = collectively(coordination, _inputs, what=f"the {protocol} inputs")
    if serial:
        # A serial protocol under a plural launch is refused before it is placed: the placement
        # would measure GPUs for workers that must not exist.
        reject_plural_launch(coordination, what=protocol)

    # THE COLLECTIVE POINTS. Everything above is a property of the command line or of files every
    # rank sees identically; placement and the platform are rank-local -- this rank's CPUs, this
    # rank's device, this rank's CUDA context -- and each is made collective.
    plan = _plan_placement(coordination, machine, cpu=cpu, device=device, loaded=loaded,
                           topology=topology, system=system, protocol=protocol)
    acceleration, index, detail = collectively(
        coordination, lambda: _resolve_platform(machine, cpu=cpu, device=device, plan=plan),
        what=f"the {protocol} preflight")
    return coordination, machine, acceleration, index, detail, particles, loaded, plan


def preflight_stage(*, topology, system, coordinates=None, trajectory=None, restart=None,
                    checkpoint=None, output=None, log=None, cpu=False, device=None,
                    machine_config=None, protocol="this stage", pending_parent=None,
                    timestep_fs=None, ensemble=None, tau=0.0, stage=None,
                    number_of_groups=None, groupfile=None, whole=None, segment=1,
                    source_trajectory=None, system2=None, topology2=None) -> StagePreflight:
    """A conventional stage, run on its own or planned as one link of a chain."""
    from ..md.stage import check_trajectory_suffix

    _reject_flags_outside_their_protocol(
        protocol_name="a cMD stage", number_of_groups=number_of_groups, groupfile=groupfile,
        source_trajectory=source_trajectory, system2=system2, topology2=topology2)

    if trajectory:
        check_trajectory_suffix(Path(trajectory))

    inputs = _continuation_inputs(coordinates, pending_parent, where=protocol)
    inventory = _stage_inventory(output=output, log=log, trajectory=trajectory, whole=whole,
                                 restart=restart, segment=segment,
                                 checkpoint=checkpoint, stage=stage)
    coordination, machine, acceleration, index, detail, particles, loaded, plan = _common(
        topology=topology, system=system,
        outputs=inventory.roles,
        inputs=inputs, cpu=cpu, device=device, number_of_groups=None, replicas=None,
        protocol=protocol, machine_config=machine_config, load=timestep_fs is not None,
        # A cMD stage is serial by construction. `_common` refuses a plural launch after the
        # coordination is open and before placement or anything is created, so it stops at the
        # preflight with every rank agreeing rather than partway through with N writers.
        serial=True)

    resolved_timestep = None
    prepared: dict[str, Any] = {}
    if timestep_fs is not None:
        # 4 fs on hydrogens that were never repartitioned, and every other mass/timestep
        # incompatibility, refused HERE. It used to be found after the `.out` and the `.log` had
        # been created, which is a directory that reads as a run that started.
        resolved_timestep = _resolve_timestep(loaded, timestep_fs, where=protocol)
        check_ensemble(loaded, ensemble=ensemble, tau=tau, where=protocol)
    if stage is not None:
        prepared = _prepare_stage(loaded, stage=stage, name=stage.get("name") or protocol,
                                  where=protocol)

    return StagePreflight(coordination=coordination, machine=machine, acceleration=acceleration,
                          device_index=index,
                          device_policy=str(machine.get("device_policy") or "local_rank"),
                          device_policy_detail=detail, placement=plan, particles=particles,
                          loaded=loaded,
                          timestep=resolved_timestep, inventory=inventory,
                          trajectory=Path(trajectory) if trajectory else None, **prepared)


def validate_generated_chain(*, topology, system, stages: list[dict[str, Any]],
                             where: str = "build-md") -> None:
    """Every stage of a chain, checked when the chain is GENERATED rather than when it runs.

    This is the whole-chain guarantee `--all-in-one` used to carry and nothing else did. A split
    chain preflights one stage at a time, each as it starts, so a configuration whose LAST stage
    asks for 4 fs on unrepartitioned masses or declares NPT on an implicit System ran every
    earlier stage to completion and then refused -- hours of queue time spent on a chain that
    could never finish, and a run directory that cannot be resumed or cleanly restarted. Every
    refusal below is knowable from the configuration and the built System alone, so there is no
    reason to wait until a stage opens to raise it.

    DEVICE-FREE, DELIBERATELY. This does not resolve a platform, place a device, open a CUDA
    Context or consult the MPI world, and it must not: generation routinely happens on a login
    node, in CI, or simply on a different machine from the run, so a device answered here would
    be an answer about the wrong computer. Those checks stay in `preflight_stage`, where the
    machine actually is. What is checked here is what a machine cannot change -- the masses in
    the System, the ensemble against the solvent, the forces the scaling convention has to place,
    and the paths the chain writes.
    """
    check_input_files(p=topology, s=system)
    loaded_by_system = {str(system): load_inputs(topology, system)}

    prepared_plans: dict[str, Any] = {}
    previous_name = previous_restart = None
    for entry in stages:
        name = str(entry["name"])
        stage_where = f"{where}: stage {name}"
        # A hot stage names its own System -- the saved scaled state -- and is checked against it.
        stage_system = entry.get("system") or system
        if str(stage_system) not in loaded_by_system:
            check_input_files(p=topology, s=stage_system)
            loaded_by_system[str(stage_system)] = load_inputs(topology, stage_system)
        loaded = loaded_by_system[str(stage_system)]

        # THE PARENT, STATED RATHER THAN INFERRED FROM ITS ABSENCE. At generation time every
        # stage after the first is missing its parent BY CONSTRUCTION -- nothing has run -- which
        # is exactly the one legitimate missing continuation `PendingParent` exists to name. It
        # matters that this goes through `_continuation_inputs` rather than skipping the check:
        # the rule is that a missing `-c` is refused unless a NAMED earlier stage will write it,
        # and a chain that simply did not look would be the silent leniency that rule replaced.
        _continuation_inputs(
            str(previous_restart) if previous_restart is not None else None,
            (PendingParent(path=previous_restart, produced_by=previous_name)
             if previous_restart is not None else None),
            where=stage_where)
        if entry.get("timestep_fs") is not None:
            _resolve_timestep(loaded, entry.get("timestep_fs"), where=stage_where)
        check_ensemble(loaded, ensemble=entry.get("ensemble"),
                       tau=float(entry.get("tau") or 0.0), where=stage_where)
        # THE FORCE AUDIT, which is the refusal that most deserves to arrive early:
        # `build_scaled_system` refuses a System carrying a force the scaling convention cannot
        # place, and that is a property of the built System, not of the run.
        if entry.get("stage") is not None:
            _prepare_stage(loaded, stage=entry["stage"], name=name, where=stage_where)

        inventory = _stage_inventory(
            output=entry.get("output"), log=entry.get("log"),
            trajectory=entry.get("trajectory"), restart=entry.get("restart"),
            checkpoint=entry.get("checkpoint"), stage=entry["stage"])
        # An output of this stage may not be one of the shared inputs every stage reads.
        check_output_collisions(outputs=inventory.roles,
                                inputs={"p": topology, "s": stage_system})
        prepared_plans[name] = SimpleNamespace(inventory=inventory)
        previous_name, previous_restart = name, entry.get("restart")

    # TWO STAGES WRITING ONE PATH, which no single stage's own inventory can see.
    from ..md.stage import cross_stage_collision

    collision = cross_stage_collision(prepared_plans)
    if collision is not None:
        raise PreflightError(f"{where}: {collision}")


def _prepare_stage(loaded: LoadedInputs, *, stage: dict[str, Any], name: str,
                   where: str) -> dict[str, Any]:
    """Build the System a stage will actually integrate, and refuse here if it cannot be built.

    Everything below used to run AFTER the `.out` and the `.log` were open: the solute selection,
    the saved-state check, the force audit, the implicit/NPT and fixed-tau/NPT checks, the restraint, and the barostat. Each is a refusal that arrived attached
    to a directory that reads as a run that started -- and `build_scaled_system` in particular
    refuses a System carrying a force the convention cannot place, which is not a rare case on a
    hand-built System.
    """
    from ..md._stages import add_barostat, add_positional_restraint, count_barostats, derive_seed
    from ..md.stage import solute_atom_indices

    system = loaded.system
    implicit = loaded.implicit
    solute = solute_atom_indices(loaded.pdb.topology)
    seed = derive_seed(int(stage["seed"]), name)
    tau = float(stage.get("tau") or 0.0)
    excluded: list = []

    if implicit and stage.get("ensemble") not in (None, "NVT"):
        raise PreflightError(
            f"{where} declares ensemble {stage['ensemble']}, but the System is not periodic. "
            f"Implicit solvent has no volume to control, so there is no NPT here.")

    # A STAGE NEVER SCALES. `dynamics.tau` is a CLAIM about the System it was given, checked
    # against the record `build-top --rest2-scaler` wrote beside it (user, 2026-09-16). Scaling in
    # memory here made the same Hamiltonian a second way, saved it nowhere, and could not tell a
    # System that was already scaled from one that was not: a saved state given with tau > 0 was
    # scaled AGAIN, to (1-tau)^4, and one given with tau = 0 ran NPT on a scaled Hamiltonian.
    # Minimisation claims nothing: it minimises the file it is given, and `min/` is shared.
    from ..rest2.scaler import clone_system
    from ..rest2.states import ScaledStateError, scaled_state_identity

    unscaled_impropers = True
    if name != "min":
        try:
            identity = scaled_state_identity(loaded.system_path)
        except ScaledStateError as refusal:
            raise PreflightError(f"{where}: {refusal}") from None
        if identity is not None:
            if abs(float(identity["tau"]) - tau) > 1e-9:
                raise PreflightError(
                    f"{where} claims tau = {tau}, but -s {Path(loaded.system_path).name} is state "
                    f"{identity['state']} of {identity['record']} at tau = {identity['tau']}. A "
                    f"stage scales nothing: the tau it declares must be the tau of the System it "
                    f"is given, or its ensemble and its records describe a different Hamiltonian "
                    f"from the one it integrates.")
            from ..rest2.states import load_scaler_record

            section = load_scaler_record(identity["record"])["unscaled_torsions"]
            excluded = [tuple(int(a) for a in bond)
                        for bond in section.get("unscaled_central_bonds", [])]
            unscaled_impropers = bool(section.get("unscaled_impropers", True))
        elif tau > 0.0:
            raise PreflightError(
                f"{where} claims tau = {tau}, but -s {Path(loaded.system_path).name} is not a "
                f"saved scaled state (no scaler.yaml beside it names it). A stage no longer "
                f"scales in memory. Build the state with `md-openmm build-top --rest2-scaler` and "
                f"pass that file as -s, e.g. build/cMD/system_state0.xml.")
    if tau > 0.0 and name != "min" and stage.get("ensemble") != "NVT":
        raise PreflightError(
            f"{where} runs at tau={tau} but declares ensemble {stage.get('ensemble')}. A "
            f"scaled run samples the fixed-volume ensemble of the ladder rung it sits at; a "
            f"barostat would sample a different distribution.")
    # The force audit still runs: a System carrying a force the convention cannot place is refused
    # here, before any output, whether or not it is scaled.
    from ..rest2.scaler import UnclassifiedForceError, audit_force_classes

    try:
        audit_force_classes(system, where=where)
    except UnclassifiedForceError as unknown:
        raise PreflightError(str(unknown)) from None
    # A copy, so a plan never hands the runtime the object `LoadedInputs` holds: the restraint
    # and the barostat below mutate it in place, and two plans built from one `LoadedInputs`
    # would otherwise accumulate each other's machinery.
    system = clone_system(system)

    hamiltonian_identity = None
    if stage.get("phase_space_interval_steps"):
        from ..rest2 import identity_record

        # Taken BEFORE the restraint and barostat: they are properties of how this stage is run,
        # not terms of the energy the ensemble is defined by. An identity taken after them claims
        # a CustomExternalForce the scaled rung's own ensemble does not have.
        hamiltonian_identity = identity_record(
            system, tau=tau, temperature_k=float(stage["temperature_K"]),
            ensemble=stage.get("ensemble"), solute_indices=solute, excluded_bonds=excluded,
            unscaled_impropers=unscaled_impropers)

    add_positional_restraint(system, loaded.pdb.positions, solute)
    if not implicit:
        active = stage.get("ensemble") == "NPT"
        add_barostat(system, float(stage["pressure_bar"]), float(stage["temperature_K"]),
                     derive_seed(int(stage["seed"]), name, "barostat"),
                     frequency=int(stage["barostat_interval_steps"]) if active else 0)
    if implicit and count_barostats(system):
        raise PreflightError(f"{where}: implicit solvent must carry no barostat")

    return {"prepared_system": system, "solute": tuple(int(i) for i in solute),
            "excluded_bonds": tuple(excluded), "implicit": bool(implicit), "seed": int(seed),
            "restrained": float(stage.get("restraint_kcal_per_mol_A2") or 0.0) > 0.0,
            "hamiltonian_identity": hamiltonian_identity}


def cv_sidecar_path(csv_path) -> Path:
    """The interpretation sidecar beside a CV series: `<name>.cv.csv` -> `<name>.cv.json`.

    ONE definition of the name, used by the writer, the inventory, the completion check and the
    registration path alike. They disagreed before -- the inventory said `.cv.yaml` and the writer
    produced `.cv.json` -- which is the whole failure mode a shared helper removes.
    """
    csv_path = Path(csv_path)
    return csv_path.with_suffix(".json") if csv_path.suffix == ".csv" \
        else Path(str(csv_path) + ".json")


def _stage_inventory(*, output, log, trajectory, restart, checkpoint,
                     whole=None, segment=1, stage=None) -> OutputInventory:
    """A stage's complete inventory, including the outputs it derives rather than is given.

    The phase-space stream and the checkpoint GENERATION TREE are the two that were invisible:
    neither arrives as a flag, both are written, and `--overwrite` therefore governed neither.
    """
    roles: dict[str, Path] = {}
    for role, value in (("out", output), ("log", log), ("trajectory", trajectory),
                        ("restart", restart)):
        if value:
            roles[role] = Path(value)
    if whole:
        roles["whole_trajectory"] = Path(whole)
    # THE WHOLE STREAM'S sidecar, because that is whose atoms it holds. Derived from
    # `trajectory` -- the solute stream since the rename -- it named a file the run never wrote,
    # so `--overwrite` left the real one behind: reporting turned off, the stale
    # `whole_prod1.phase_space.nc` still sitting there looking like an output of the new run.
    phase_space_source = whole or trajectory
    if phase_space_source:
        roles["phase_space"] = Path(phase_space_source).with_suffix(".phase_space.nc")
    if log:
        # The StateDataReporter CSV sits beside the log, is appended to, and is one of the three
        # streams a checkpoint commits counts for -- so it belongs in the inventory that decides
        # collisions and what `--overwrite` governs.
        from ..md._stages import info_csv_name
        roles["state_csv"] = Path(log).parent / info_csv_name(Path(log).stem, segment)
    if trajectory:
        # The collective-variable series and the resolved definition beside it. Both are written
        # by the stage, neither arrives as a flag, and an inventory that omits them is an
        # `--overwrite` that leaves a previous cv.yaml's columns in place beside new ones.
        # `.cv.json`, which is what `CVSeries` actually writes. The inventory named `.cv.yaml`
        # -- a file that never existed -- so the real sidecar was in no inventory at all: not
        # collision-checked, not removed by `--overwrite`, and free to survive a definition change
        # and describe the new CSV with the old atom mapping.
        #
        # Three different files are deliberately not conflated: the INPUT `cv.yaml` a person
        # writes, the content-addressed copy `build-md` puts in the generated directory, and this
        # OUTPUT sidecar, which says how to read the CSV beside it.
        #
        # NAMED BY `cv_csv_path`, the one authority the stage writes through: `<key>.cv.csv`.
        # This derived `<trajectory stem>.cv.csv` -- `solute_prod1.cv.csv`, a file no stage writes
        # since a stage has two trajectories -- so a completed resumed stage was refused as
        # missing its series, and `--overwrite` left the real `cMD.cv.csv` and `.cv.json` behind.
        from ..md.stage import cv_csv_path

        series = cv_csv_path(Path(trajectory), stage)
        roles["collective_variables"] = series
        roles["collective_variables_definition"] = cv_sidecar_path(series)
    if checkpoint:
        checkpoint = Path(checkpoint)
        roles["checkpoints"] = checkpoint.parent / f"{checkpoint.stem}.checkpoints"
    # The checkpoint tree is what a `--resume` reads, so its presence is never itself the
    # objection -- `check_existing_outputs` is about a FINISHED run being in the way.
    return OutputInventory(roles=roles, resumable=frozenset({"checkpoints"}))


def report_check(result, *, what: str, extra=()) -> int:
    """What `--check` prints. To STDOUT, because `--check` creates nothing.

    `--check` answers "would this run start?", and it used to answer it by starting: it created
    `-odir`, the `.out` and the `.log`, validated, and wrote `status: checked` into the log. The
    directory it left behind is indistinguishable from a run that began -- which is precisely the
    thing the caller was asking about. Every fact below comes from the preflight result, so
    nothing has to be opened to report it.
    """
    import sys as _sys

    lines = [f"{what}: --check passed. Nothing was created.",
             f"  platform          {result.platform_name} "
             f"({result.record()['platform_selection']})",
             f"  device            {result.device_index if result.device_index is not None else '-'}"
             f"  [{result.device_policy_detail}]",
             f"  mpi               rank {result.coordination.rank} of {result.coordination.size}",
             *_placement_lines(result.placement),
             f"  particles         {result.particles}"]
    if result.timestep:
        lines.append(f"  timestep          {result.timestep['timestep_fs']} fs "
                     f"({result.timestep['basis']})")
    lines.extend(f"  {label:<18}{value}" for label, value in extra)
    print("\n".join(lines), file=_sys.stdout)
    return 0


def _placement_lines(plan) -> list[str]:
    if plan is None:
        return []
    mine = plan.for_rank()
    node = next(n for n in plan.nodes if n["host"] == mine["host"])
    lines = [f"  cpus              {len(mine['cpus'])} bound ({_ranges(mine['cpus'])}) of "
             f"{node['usable_cpus']} usable on {mine['host']}, {len(node['ranks'])} worker(s)"]
    measured = (plan.measurement or {}).get(mine["host"])
    if measured:
        rates = ", ".join(f"{rate:.0f}" for rate in measured["steps_per_second"])
        lines.append(f"  throughput        {rates} steps/s by device")
    if plan.mps is not None:
        lines.append(f"  mps               {plan.mps.status}"
                     + (" (required: a device is shared)" if plan.shared_devices else ""))
    return lines


def _ranges(cpus) -> str:
    """`0-5,24-29` rather than twelve numbers."""
    ordered, spans = sorted(int(c) for c in cpus), []
    for cpu in ordered:
        if spans and cpu == spans[-1][1] + 1:
            spans[-1][1] = cpu
        else:
            spans.append([cpu, cpu])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)


def _resolve_timestep(loaded: LoadedInputs, requested, *, where: str) -> dict[str, Any]:
    """`md_tools.openmm.timestep` is the rule; this is where it is applied early enough to help."""
    from ..openmm.timestep import resolve_timestep_fs

    try:
        return resolve_timestep_fs(requested, loaded.system, loaded.pdb.topology)
    except (ValueError, SystemExit) as refusal:
        raise PreflightError(f"{where}: {refusal}") from None


def reject_contradictory_continuation(*, resume: bool, overwrite: bool, what: str) -> None:
    """`--resume` and `--overwrite` together ask for opposite things.

    One says continue the run that is there; the other says there is no run to continue, replace
    it. Silently letting either win is how a person who typed both gets the one they did not mean
    -- and the two outcomes are "your previous work continues" and "your previous work is gone",
    which is not a difference to resolve by precedence.
    """
    if resume and overwrite:
        raise PreflightError(
            f"{what}: --resume and --overwrite contradict each other. --resume continues the run "
            f"already in this directory; --overwrite deletes it and starts a new one. Pass one.")


def reject_plural_launch(coordination, *, what: str) -> None:
    """A serial protocol launched under `mpirun -n N` is N runs over ONE set of output paths.

    Nothing coordinates them, nothing partitions them, and every rank opens the same trajectory,
    the same state CSV and the same checkpoint tree. The files that result are interleaved from N
    simulations and there is no record anywhere that says so -- the run "completes", and the
    trajectory is a splice of N different walkers.

    Refused rather than tolerated, and refused for the WHOLE world: a rank that exited alone would
    leave the others writing.
    """
    if coordination.size <= 1:
        return
    raise PreflightError(
        f"{what} was launched under a plural MPI world ({coordination.size} ranks), and it is a "
        f"serial protocol.\n"
        f"  Nothing here partitions work across ranks, so all {coordination.size} would integrate "
        f"the same stage and write the same trajectory, state table and checkpoint -- producing "
        f"one set of files interleaved from {coordination.size} independent simulations, with "
        f"nothing in them saying so.\n"
        f"  Run it in one process, or use a protocol that coordinates: REST2 place one "
        f"rank per thermodynamic state, and AIS distributes paths by `paths_for_rank`.")


def _reject_flags_outside_their_protocol(*, protocol_name, number_of_groups=None, groupfile=None,
                                         source_trajectory=None, trajectory=None,
                                         coordinates=None, restart=None, checkpoint=None,
                                         system2=None, topology2=None):
    """Every accepted flag must do its documented job here, or be refused before any output.

    A flag that a protocol parses and then ignores is worse than one it rejects: the run
    completes, the record shows the flag was given, and nothing anywhere did what it says. These
    are the pairings where that was true.
    """
    if number_of_groups is not None:
        raise PreflightError(
            f"-ng is a REST2 flag and {protocol_name} has no replicas to group. It was "
            f"accepted and ignored; refusing it instead, because a run launched under "
            f"`mpirun -n {number_of_groups}` with -ng silently ignored is N processes writing "
            f"over one set of files.")
    if groupfile is not None:
        raise PreflightError(
            f"-groupfile describes one line per replica and {protocol_name} has one process. "
            f"Refusing it rather than reading a file whose contents can have no effect.")
    if source_trajectory is not None:
        raise PreflightError(
            f"-source-traj names the equilibrium ensemble AIS draws its starting frames from, "
            f"and {protocol_name} draws none. Refusing it rather than accepting a trajectory "
            f"nothing will read.")
    for value, flag in ((system2, "-s2"), (topology2, "-p2")):
        if value is not None:
            raise PreflightError(
                f"{flag} names the second end state of an AIS transformation, and "
                f"{protocol_name} has one Hamiltonian. Refusing it rather than accepting a file "
                f"nothing will read.")
    for value, flag, what in ((trajectory, "-x", "its own per-path trajectory names"),
                              (coordinates, "-c", "no continuation"),
                              (restart, "-r", "no single output restart"),
                              (checkpoint, "-chk", "no single checkpoint")):
        if value is not None:
            raise PreflightError(
                f"{flag} has no meaning for {protocol_name}, which has {what}. Refusing it "
                f"rather than accepting a path nothing writes to.")



def _ligand_sdf_beside(system):
    """The prepared molecule `build-top` retained beside the System, or None.

    Two layouts, read in this order:

    * `<system stem>.sdf` -- `built.sdf`, which every build before 0.5.4 wrote;
    * `<RESNAME>.sdf` -- what `build-top` writes now, named for the solute's residue. RESNAME is
      read from `<system stem>.pdb`: the ONE residue that is neither solvent nor an ion. A
      topology with several such residues is a peptide or a complex, not a molecule `build-top`
      named, so it resolves to nothing.

    Returns None for a peptide build, which writes no SDF -- so this cannot turn a protein run
    into a ligand run, and the absence is as meaningful as the presence.
    """
    if not system:
        return None
    candidate = Path(system).with_suffix(".sdf")
    if candidate.is_file():
        return candidate
    topology = Path(system).with_suffix(".pdb")
    if not topology.is_file():
        return None
    from ..md.stage import SOLVENT_RESIDUES

    # Residue names straight from the records, columns 18-20 keyed by chain and number: reading a
    # solvated box through OpenMM's PDB parser to learn one name would cost seconds per call.
    solute: dict[str, str] = {}
    with topology.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                name = line[17:20].strip()
                if name.upper() not in SOLVENT_RESIDUES:
                    solute.setdefault(line[21:27], name)
    names = set(solute.values())
    if len(solute) != 1 or len(names) != 1:
        return None
    named = Path(system).with_name(f"{names.pop()}.sdf")
    return named if named.is_file() else None

def preflight_ladder(*, topology, system, replicas, coordinates=None, groupfile=None,
                     trajectory=None, restart=None, checkpoint=None, output=None, log=None,
                     number_of_groups=None, cpu=False, device=None, machine_config=None,
                     protocol="this ladder", pending_parent=None, timestep_fs=None,
                     ensemble=None, tau=0.0, source_trajectory=None, system2=None,
                     topology2=None, ladder=None, out_dir=None) -> LadderPreflight:
    """A REST2 ladder. `-ng`, the configured state count and the world must agree.

    Since 0.5.4 a ladder reads `-s` ONLY from its group file, and every line must name a saved
    scaled state (`md-openmm build-top --rest2-scaler`). Refused here, in the shared runtime guard,
    whichever surface launched it.
    """
    # Flags of another protocol first, by name: they are wrong whatever else is.
    if source_trajectory is not None or system2 is not None or topology2 is not None:
        _reject_flags_outside_their_protocol(protocol_name="a REST2 ladder",
                                             source_trajectory=source_trajectory,
                                             system2=system2, topology2=topology2)
    if system is not None:
        raise PreflightError(
            f"{protocol}: -s {system} was given. A REST2 ladder reads -s only from its "
            f"group file (0.5.4): each line names one saved scaled state, "
            f"build/REST2/system_state<n>.xml. Pass --groupfile and no -s.")
    if not groupfile:
        raise PreflightError(
            f"{protocol}: no group file was given. A REST2 ladder reads -s only from its "
            f"group file (0.5.4); `build-md` writes remd_groupfile.<segment>.")

    inputs: dict[str, Any] = dict(_continuation_inputs(coordinates, pending_parent,
                                                       where=protocol))
    if groupfile:
        inputs["groupfile"] = groupfile

    # NO `-s`, BUT STILL A SYSTEM TO CHECK: the group file's FIRST line names one.
    #
    # A grouped ladder carries no single `-s` -- each line names its own pre-scaled rung -- and
    # the checks below genuinely need a System: the timestep is resolved from the MASSES actually
    # serialised in it, which is what refuses a 4 fs step that HMR does not support. Skipping
    # those checks when `-s` is absent would drop that refusal for every ladder, silently.
    #
    # The first line is representative, and not by assumption. `parse_group_file` validates every
    # line and requires `system` on each; scaling changes force CONSTANTS, not masses, so all N
    # rungs of a ladder carry identical masses (measured: three rungs of a 22-particle solute,
    # byte-identical mass signatures). `_preflight_from_groups` already reads the launch's inputs
    # from this same first line for the same reason.
    #
    # Its paths resolve against the GROUP FILE rather than the working directory, so
    # `remd0/build_state0.xml` is found wherever the launch was started from.
    # AND THE STARTING STATE, for the same reason and from the same place.
    #
    # A GROUP FILE IS THE AMBER-STYLE DESCRIPTION OF THE LAUNCH (`-rem 3`, the only REMD type this
    # build supports), and the per-replica INPUT commands live on its lines: `-i`, `-p`, `-s` and
    # `-c` are validated there, not on the run-level command line, which carries only the outputs.
    # `remd.executor.resolve` already exempts exactly those four when a group file is in force;
    # this function did not, so `rest2.equilibration_per_tau` refused a perfectly well described
    # launch with "equilibrates every rung FROM the ladder's starting state, and no -c was given"
    # while every line of the group file named that state.
    #
    # Line 0 is representative BY CONSTRUCTION, not by assumption: `coordinates` is in
    # `HOMOGENEOUS_GROUP_FIELDS`, so `_require_homogeneous_groups` refuses a file whose lines
    # disagree about it before this point -- a ladder's rungs are N states of ONE system and share
    # one starting configuration. `system` is read the same way, and is the one field deliberately
    # NOT homogeneous: each rung is its own pre-scaled Hamiltonian, and the timestep is resolved
    # from the MASSES, which scaling does not change (measured: identical mass signatures across
    # three rungs). Skipping that check when `-s` is absent would drop the HMR refusal silently.
    #
    # Paths resolve against the GROUP FILE rather than the working directory, so
    # `remd0/build_state0.xml` and `eq/eq_3.xml` are found wherever the launch was started from.
    if groupfile and not (system and coordinates):
        from ..remd.executor import GroupFileError, parse_group_file

        try:
            first = parse_group_file(groupfile)[0]
        except GroupFileError as refusal:
            raise PreflightError(f"{protocol}: {refusal}") from None
        system = system or first.get("system")
        # An EXTENSION legitimately has no `-c` on its lines -- it takes the physical state from
        # the parent's checkpoint -- so a group file without one leaves this as it was, and the
        # per-tau check below refuses it by name rather than being quietly satisfied.
        if not coordinates and first.get("coordinates"):
            coordinates = first["coordinates"]
            inputs = dict(_continuation_inputs(coordinates, pending_parent, where=protocol),
                          groupfile=groupfile)
    inventory = _ladder_inventory(protocol=protocol, replicas=int(replicas), output=output,
                                  log=log, trajectory=trajectory, restart=restart,
                                  checkpoint=checkpoint, groupfile=groupfile,
                                  per_tau=bool((ladder or {}).get("per_tau_equilibration")))

    coordination, machine, acceleration, index, detail, particles, loaded, plan = _common(
        topology=topology, system=system,
        outputs=inventory.roles,
        inputs=inputs, cpu=cpu, device=device, number_of_groups=number_of_groups,
        replicas=int(replicas), protocol=protocol, machine_config=machine_config,
        # ALWAYS loaded: the saved states' solute record, force audit and per-state equilibration
        # all need the topology and the state System.
        load=True)

    resolved_timestep = audit = scaled = None
    if timestep_fs is not None:
        resolved_timestep = _resolve_timestep(loaded, timestep_fs, where=protocol)
    if loaded is not None:
        # A REST2 runtime is NVT by contract, so `ensemble` is normally not passed; when it is,
        # the same two impossibilities are refused as for a stage.
        check_ensemble(loaded, ensemble=ensemble, tau=tau, where=protocol)
    solute_record = None
    # SAVED SCALED STATES. When every line of the group file names a state of ONE `scaler.yaml`,
    # in state order, the ladder integrates those files AS THEY ARE: `build-top --rest2-scaler`
    # scaled them, and nothing here scales or classifies again (docs/amber-like-fix/
    # REST2-scaler.md, step 4). Before this, a grouped ladder loaded its first line's System and
    # rebuilt every rung from it at run time -- re-classifying torsions from an SDF it looked for
    # beside `remd0/build_state0.xml`, where none ever is, which is how 0.5.3 refused a ligand.
    saved_states = _saved_state_ladder(groupfile, replicas=int(replicas), ladder=ladder,
                                       where=protocol, out_dir=out_dir)
    if saved_states is None:
        raise PreflightError(
            f"{protocol}: {groupfile} names no saved scaled states. A ladder scales nothing "
            f"(0.5.4): each line must name build/REST2/system_state<n>.xml, written by "
            f"`md-openmm build-top --rest2-scaler`, and `build-md` writes such a group file.")
    solute_indices = list(saved_states["solute_indices"])
    excluded_bonds = list(saved_states["excluded_bonds"])
    if loaded is not None:
        solute_record = saved_states["solute_record"](loaded)
    if loaded is not None:
        from ..rest2.scaler import UnclassifiedForceError, audit_force_classes

        try:
            audit = audit_force_classes(loaded.system, where=protocol)
        except UnclassifiedForceError as unknown:
            raise PreflightError(str(unknown)) from None

    # -- EVERY rung, not just the top one -------------------------------------------------------
    #
    # `check_scaling_plan` above validates the force layout using one System at `tau`. That is not
    # the ladder: a ladder is N Systems, and the driver used to build all of them itself, at
    # runtime, after every output file existed. So a force that classifies at tau_max and fails at
    # an intermediate rung was found in the worst possible place.
    #
    # These are built through `build_rung_systems` -- the SAME function `Protocol.build_systems`
    # delegates to -- so the preflight and the driver cannot build two different ladders. The
    # driver consumes exactly these.
    # -- restraints on every rung, resolved BEFORE the rungs are built --------------------------
    #
    # A ladder may carry torsion restraints -- the same ones on every rung, which is what keeps
    # the bias out of the exchange criterion. They have to be resolved here, before
    # `build_rung_systems`, because that is where they are added to each scaled System; and here
    # rather than in the driver, because a restraint naming an atom selector that does not resolve
    # must refuse before `-odir` exists.
    #
    # The collective-variable definition is loaded for this even when reporting is off: a
    # restraint names a CV to get its four atoms, which is a different question from whether the
    # series is written. The reporting block further down loads it again for its own reasons; two
    # reads of one file are tolerable here only because neither can bias anything -- the FORCE is
    # built from this one alone.
    ladder_restraints = ()
    umbrella_file = ((ladder or {}).get("umbrella") or {}).get("file") if ladder else None
    if umbrella_file and loaded is not None:
        def _beside_config(name):
            path = Path(name)
            if path.is_absolute():
                return path
            beside = (ladder or {}).get("resolved_config")
            base = Path(beside).parent if beside else (Path(out_dir) if out_dir else Path("."))
            return base / path

        def _restraints():
            from ..cv import CVDefinitionError, load_cv_definition
            from ..umbrella import UmbrellaError, load_umbrella_definition

            cv_file = ((ladder or {}).get("collective_variables") or {}).get("file")
            if not cv_file:
                raise PreflightError(
                    f"{protocol}: umbrella.file = {umbrella_file!r} restrains collective "
                    f"variables, but collective_variables.file is not set, so there is nothing to "
                    f"resolve the restraint against. Nothing has been written.")
            try:
                definition = load_cv_definition(
                    _beside_config(cv_file), topology=loaded.pdb.topology,
                    particles=loaded.system.getNumParticles())
            except CVDefinitionError as refusal:
                raise PreflightError(f"{protocol}: {refusal}") from None
            try:
                entries = load_umbrella_definition(_beside_config(umbrella_file), definition)
            except UmbrellaError as refusal:
                raise PreflightError(f"{protocol}: {refusal}") from None
            return tuple(entry.record() for entry in entries)

        ladder_restraints = collectively(coordination, _restraints,
                                         what="the ladder's torsion restraints")

    rung_systems = ()
    rungs_tau = ()
    if saved_states is not None:
        from ..remd.protocol import apply_ladder_restraints

        rungs_tau = tuple(saved_states["taus"])
        systems = [system_ for system_ in saved_states["systems"]]
        if ladder_restraints:
            for system_ in systems:
                apply_ladder_restraints(system_, ladder_restraints)
        rung_systems = tuple(systems)

    # -- per-tau equilibration, checked HERE ---------------------------------------------------
    #
    # The plan, the solute it restrains and the restrained copy of the hottest rung, all before
    # any output exists. The driver builds one such copy per rung; a Force that cannot be added
    # would otherwise be found with the analysis file and every trajectory already created.
    per_tau = (ladder or {}).get("per_tau_equilibration") or []
    if per_tau and loaded is not None:
        def _per_tau():
            from ..remd.rung_equilibration import restrained_clone, validate_plan

            try:
                validate_plan(per_tau)
            except ValueError as refusal:
                raise PreflightError(f"{protocol}: rest2.equilibration_per_tau: {refusal}") \
                    from None
            if not solute_indices:
                raise PreflightError(
                    f"{protocol}: rest2.equilibration_per_tau restrains the solute during its "
                    f"restrained stages, and this System has no solute atoms.")
            if not coordinates:
                raise PreflightError(
                    f"{protocol}: rest2.equilibration_per_tau equilibrates every rung FROM the "
                    f"ladder's starting state, and no -c was given.")
            try:
                restrained_clone(rung_systems[-1] if rung_systems else loaded.system,
                                 loaded.pdb.positions, solute_indices)
            except Exception as broken:
                raise PreflightError(
                    f"{protocol}: rest2.equilibration_per_tau could not restrain the top rung: "
                    f"{type(broken).__name__}: {broken}") from None
            return None

        collectively(coordination, _per_tau, what="the per-tau equilibration plan")

    # -- the collective-variable definition ------------------------------------------------------
    #
    # Parsed HERE, before -odir exists. Left to the driver it would be read after the run
    # directory, both logs and every helper were on disk, and a typo in an atom selector would
    # refuse a ladder that had already written output.
    cv_definition = None
    cv_block = (ladder or {}).get("collective_variables") or {}
    if cv_block.get("file") and int(cv_block.get("interval_steps") or 0) > 0 and loaded is not None:
        def _definition():
            from ..cv import CVDefinitionError, load_cv_definition

            path = Path(cv_block["file"])
            if not path.is_absolute():
                # Relative to the directory holding `resolved.config` -- the GENERATED directory,
                # where `build-md` put the content-addressed copy. Not `-odir`, which is where
                # this run's output goes and holds no definition, and not the working directory,
                # which only happens to be right when the script is launched from beside itself.
                beside = (ladder or {}).get("resolved_config")
                base = Path(beside).parent if beside else (
                    Path(out_dir) if out_dir else Path("."))
                path = base / path
            try:
                return load_cv_definition(path, topology=loaded.pdb.topology,
                                          particles=loaded.system.getNumParticles())
            except CVDefinitionError as refusal:
                raise PreflightError(f"{protocol}: {refusal}") from None

        cv_definition = collectively(coordination, _definition,
                                     what="the collective-variable definition")

        # The CADENCE, checked here too. `EventSchedule` enforces it, but it is constructed when
        # the generated `_protocol.py` is loaded -- which `--check` never reaches, so an interval
        # that does not divide the exchange interval passed `--check` and then failed the run.
        # A preflight that cannot refuse what the run will refuse is not a preflight.
        def _cadence():
            from ..cv import CVScheduleError, check_divides

            exchange = int((ladder or {}).get("exchange_interval_steps") or 0)
            if exchange <= 0:
                return None
            try:
                check_divides(int(cv_block["interval_steps"]), exchange,
                              where=protocol,
                              what=f"exchange_interval_steps ({exchange})")
            except CVScheduleError as refusal:
                raise PreflightError(f"{protocol}: {refusal}") from None
            return None

        collectively(coordination, _cadence, what="the collective-variable cadence")

    # WHAT THIS INVOCATION INTENDS TO CONTINUE, validated here -- before `-odir`, the `.out`, the
    # `.log`, the run-state record or `resolved.config` is created or replaced. Every protocol
    # used to check its committed CV records only once those files were already open, so a
    # refused continuation had overwritten the prior run's machine-readable provenance and
    # completion status with a description of an invocation that never started.
    if out_dir is not None and cv_definition is not None:
        from .continuation import ContinuationError, validate_ladder_continuation

        try:
            validate_ladder_continuation(
                out_dir, definition=cv_definition, taus=rungs_tau,
                interval_steps=_cv_interval_of(ladder),
                checkpoint_path=(checkpoint or Path(out_dir) / f"{protocol}_checkpoint.nc"))
        except ContinuationError as refusal:
            raise PreflightError(str(refusal)) from None

    return LadderPreflight(coordination=coordination, machine=machine,
                           acceleration=acceleration, device_index=index,
                           device_policy=str(machine.get("device_policy") or "local_rank"),
                           device_policy_detail=detail, placement=plan, particles=particles,
                           loaded=loaded,
                           timestep=resolved_timestep, replicas=int(replicas),
                           inventory=inventory,
                           force_audit=audit, scaled_system=scaled,
                           solute_indices=tuple(int(i) for i in (solute_indices or ())),
                           tau_list=rungs_tau, rung_systems=rung_systems,
                           excluded_bonds=tuple(tuple(int(a) for a in b) for b in excluded_bonds),
                           cv_definition=cv_definition,
                           ladder_restraints=tuple(ladder_restraints),
                           notes={"solute_document": solute_record} if solute_record else {})


def _saved_state_ladder(groupfile, *, replicas: int, ladder, where: str, out_dir=None):
    """The ladder's rungs as SAVED SCALED STATES, or None when the group file names none.

    Every line must name a state of the same `scaler.yaml`, line i state i, and the states must be
    exactly the ladder: the configured count, and -- when the ladder is described -- the configured
    tau list. A group file mixing saved states with other Systems is refused: half a ladder scaled
    at build time and half at run time is two derivations of one Hamiltonian.
    """
    from openmm import XmlSerializer

    from ..remd.executor import GroupFileError, parse_group_file
    from ..rest2.states import ScaledStateError, load_scaler_record, scaled_state_identity

    try:
        groups = parse_group_file(groupfile)
    except GroupFileError as refusal:
        raise PreflightError(f"{where}: {refusal}") from None
    identities = []
    for group in groups:
        try:
            identities.append(scaled_state_identity(group["system"]) if group.get("system")
                              else None)
        except ScaledStateError as refusal:
            raise PreflightError(f"{where}: {refusal}") from None
    if all(identity is None for identity in identities):
        return None
    if any(identity is None for identity in identities):
        raise PreflightError(
            f"{where}: {groupfile} names saved scaled states on some lines and other Systems on "
            f"others. A ladder's rungs are either all built by `build-top --rest2-scaler` or "
            f"none are.")
    records = {identity["record"] for identity in identities}
    if len(records) != 1:
        raise PreflightError(f"{where}: {groupfile} names states from {len(records)} different "
                             f"scaler records ({sorted(records)}); a ladder is one schedule.")
    record = load_scaler_record(next(iter(records)))
    if len(identities) != replicas or len(record["states"]) != replicas:
        raise PreflightError(
            f"{where}: the ladder has {replicas} state(s), {groupfile} names {len(identities)} and "
            f"{identities[0]['record']} holds {len(record['states'])}. They must be one number.")
    for group, identity in zip(groups, identities):
        if int(identity["state"]) != int(group.get("group_index", -1)):
            raise PreflightError(
                f"{where}: {groupfile}:{group['line']} gives --group-index "
                f"{group.get('group_index')} the saved state {identity['state']}. Each line must "
                f"name its own state, or the ladder runs one state's Hamiltonian under another's "
                f"identity.")
    taus = [float(identity["tau"]) for identity in identities]
    if ladder is not None:
        from ..remd.generated import tau_ladder

        claimed = [float(t) for t in tau_ladder(int(ladder["n_states"]), float(ladder["tau_max"]))]
        if any(abs(a - b) > 1e-9 for a, b in zip(claimed, taus)):
            raise PreflightError(
                f"{where}: the ladder is configured as tau {claimed} (rest2.number_of_replicas, "
                f"rest2.tau_max) but its saved states are at tau {taus}. A ladder scales nothing: "
                f"its configuration must describe the states it is given.")
    section = record["unscaled_torsions"]
    solute = [int(i) for i in record["solute"]["atom_indices"]]
    excluded = [tuple(int(a) for a in bond) for bond in section["unscaled_central_bonds"]]
    systems = [XmlSerializer.deserialize(Path(group["system"]).read_text(encoding="utf-8"))
               for group in groups]

    def solute_record(loaded):
        from ..openmm.builders import _solute_document

        document = _solute_document(
            loaded.pdb.topology, solute,
            {"unscaled_central_bonds": excluded, "central_bonds": section["central_bonds"],
             "unscaled_impropers": section["unscaled_impropers"],
             "proline_like_scaled_bonds": section["proline_like_scaled_bonds"],
             "detection_method": section["method"], "unclassified": []},
            route="saved-state", system=loaded.system)
        # RELATIVE to the directory `solute.yaml` is written into, as every path a run records
        # about its inputs is: an absolute path stops resolving when the run tree is moved, and
        # `solute.yaml` is content-addressed, so it would also differ between two copies of one run.
        record_path = identities[0]["record"]
        if out_dir is not None:
            record_path = os.path.relpath(Path(record_path).resolve(), Path(out_dir).resolve())
        document["rest2"]["scaled_states"] = {"record": record_path,
                                             "states": [i["system_sha256"] for i in identities]}
        return document

    return {"taus": taus, "systems": systems, "solute_indices": solute,
            "excluded_bonds": excluded, "solute_record": solute_record}


def _cv_interval_of(document) -> int:
    """The configured CV cadence for a resolved protocol document, or 0 when reporting is off."""
    block = (document or {}).get("collective_variables") or {}
    try:
        return int(block.get("interval_steps") or 0)
    except (TypeError, ValueError):
        return 0


def _ladder_inventory(*, protocol, replicas, output, log, trajectory, restart, checkpoint,
                      groupfile, per_tau=False) -> OutputInventory:
    """A ladder's complete inventory: the run-level files AND every per-state and per-rank one.

    The per-state trajectories -- `solute_state<i>_prod<N>.nc`, and `whole_state<i>_prod<N>.nc`
    when a whole-system cadence is set -- are the scientific result and were not in any inventory
    at all: one per thermodynamic state, written by the root, and silently replaceable. So were
    `solute.yaml`, `_protocol.py`, the group file and `rem.log`.

    Nor were the ones a PLURAL launch writes. A ladder of N states runs on N ranks and each keeps
    its own `.rankNN` report; listing only rank 0's meant `--overwrite` after a size change left
    rank 05's report from a six-state ladder sitting beside a two-state one, looking equally
    current -- and it is the file a rank that failed to bind its device writes into. The
    checkpoint GENERATION TREE and the completion and provenance manifests were missing for the
    same reason: nothing named them, so nothing could check them.
    """
    from ..remd.executor import report_path_for_rank

    roles: dict[str, Path] = {}
    for role, value in (("trajectory", trajectory), ("restart", restart),
                        ("checkpoint", checkpoint)):
        if value:
            roles[role] = Path(value)
    # One entry per rank, named the way the rank actually names it. A ladder's world size is its
    # state count (or one process for the whole ladder), so N states is N possible reports.
    for role, value in (("out", output), ("log", log)):
        if not value:
            continue
        for rank in range(max(int(replicas), 1)):
            name = role if rank == 0 else f"{role}_rank{rank:02d}"
            roles[name] = Path(report_path_for_rank(str(value), rank))
    directory = Path(output).parent if output else Path(".")
    roles["solute"] = directory / "solute.yaml"
    roles["protocol_helper"] = directory / "_protocol.py"
    if not groupfile:
        # A group file the caller NAMED is an INPUT -- it is read, not written -- and listing it
        # among the outputs made it collide with itself.
        roles["group_file"] = directory / f"{protocol}.group"
    roles["rem_log"] = directory / "rem.log"
    roles["provenance"] = directory / "machine.yaml"
    from ..remd.amber_trajectory import state_trajectory_name

    for state in range(int(replicas)):
        # BOTH per-state streams, under the names the driver actually writes. The inventory
        # still said `remd{state}.nc`, a file no ladder produces any more, so `--overwrite`
        # governed nothing: a six-state ladder rerun over a four-state one left two stale
        # `whole_state{4,5}_prod1.nc` and every solute file of the previous run in place,
        # looking exactly as current as the new ones. The solute stream was never in any
        # inventory even under its old name.
        roles[f"state_trajectory_{state}"] = directory / state_trajectory_name(
            state, content="whole")
        roles[f"state_solute_trajectory_{state}"] = directory / state_trajectory_name(
            state, content="solute")
        # The per-state CV series and its interpretation sidecar. In no inventory before, so a
        # `--overwrite` left a previous CV-enabled run's `cv_stateN.csv` sitting beside the new
        # ladder's output -- and a CV-DISABLED rerun left them there permanently, describing a
        # calculation that no longer exists, with nothing in the directory saying so.
        roles[f"state_cv_{state}"] = directory / f"cv_state{state}.csv"
        roles[f"state_cv_definition_{state}"] = directory / f"cv_state{state}.json"
    if per_tau:
        # `rest2.equilibration_per_tau`: each rung's equilibrated state and the record naming
        # them. Listed only when the setting is on, so `--overwrite` governs them and a ladder
        # that does not use it has the inventory it always had.
        from ..remd.rung_equilibration import RECORD_NAME, handoff_name

        for state in range(int(replicas)):
            roles[f"per_tau_state_{state}"] = directory / handoff_name(state)
        roles["per_tau_record"] = directory / RECORD_NAME
    if checkpoint:
        # What a `--resume` READS. Its presence is never itself the reason to refuse a
        # continuation, which is what `resumable` says.
        stem = Path(checkpoint)
        roles["checkpoints"] = stem.parent / f"{stem.stem}.checkpoints"
    return OutputInventory(roles=roles, resumable=frozenset({"checkpoints"}))


def preflight_ais(*, topology, system, source, topology2=None, system2=None,
                  number_of_groups=None, output=None, log=None,
                  cpu=False, device=None, machine_config=None, dynamics=None, ais=None,
                  reporting=None, source_config=None, groupfile=None, trajectory=None,
                  coordinates=None, restart=None, checkpoint=None,
                  out_dir=None, resolved_config=None, overwrite=False,
                  resume=False, collective_variables=None) -> AISPreflight:
    """AIS switching paths between two end states. The source is validated by CONTENT.

    `-p`/`-s` are V0, the state the source ensemble was sampled from; `-p2`/`-s2` are V1. When the
    resolved configuration is supplied -- which the runtime always does -- every remaining refusal
    happens here too: the timestep against the masses, the end-state pair, the barostat, the
    switching and reporting divisibility, the source window, the source's own atom count and
    System digest, and the frame selection. The result carries all of it, so `ais_main` runs the
    schedule it validated rather than building a second one.
    """
    from ..openmm.trajectory import check_trajectory_declaration

    # -x is refused for AIS: path trajectories are named by `path_trajectory_name(index, total)`,
    # so a single -x could only be accepted and ignored. The rest have no AIS meaning either.
    _reject_flags_outside_their_protocol(
        protocol_name="AIS", groupfile=groupfile, trajectory=trajectory,
        coordinates=coordinates, restart=restart, checkpoint=checkpoint)
    for value, flag, what in ((system2, "-s2", "serialised System"),
                              (topology2, "-p2", "topology (PDB)")):
        if value is None:
            raise PreflightError(
                f"AIS needs {flag}, the {what} of the second end state V1. AIS transforms V0 "
                f"(-p/-s, the state the source ensemble was sampled from) into V1 as "
                f"V(lambda) = (1 - lambda) V0 + lambda V1.")

    coordination, machine, acceleration, index, detail, particles, loaded, plan = _common(
        topology=topology, system=system,
        outputs={"o": output, "log": log},
        inputs={"source-traj": source, "s2": system2, "p2": topology2}, cpu=cpu, device=device,
        number_of_groups=number_of_groups, replicas=None, protocol="AIS",
        machine_config=machine_config, load=dynamics is not None)

    # After existence, before any output: a source whose suffix and contents disagree is neither.
    source_format = check_trajectory_declaration(source, what="-source-traj")

    if dynamics is None:
        return AISPreflight(coordination=coordination, machine=machine, acceleration=acceleration,
                            device_index=index,
                            device_policy=str(machine.get("device_policy") or "local_rank"),
                            device_policy_detail=detail, placement=plan, particles=particles,
                            source=Path(source), source_format=source_format)

    prepared = _prepare_ais(
        loaded, topology2=Path(topology2), system2=Path(system2),
        source=Path(source), dynamics=dynamics, ais=ais,
        reporting=reporting, source_config=source_config,
        collective_variables=collective_variables,
        # Beside `resolved.config` -- the generated directory holding the content-addressed
        # definition copy, not `-odir` and not the working directory.
        config_directory=(Path(resolved_config).parent if resolved_config else None))

    # THE IDENTITY, HERE. It used to be built after `-odir` and both rank reports existed, so an
    # incompatible source or schedule was refused by a run that had already created the directory
    # it was refusing to write into -- and on a plain rerun that directory is indistinguishable
    # from a run that started.
    #
    # Everything below is read-only: digests of files that already exist, a document derived from
    # them, and one `is_file()` on the output directory.
    from ..ais.run import decide_run_disposition, run_identity_document

    directory = Path(out_dir) if out_dir is not None else (
        Path(output).parent if output else Path("."))
    prepared["inventory"] = _ais_inventory(output=output, log=log,
                                           paths=len(prepared["chosen_frames"]),
                                           ranks=coordination.size)
    facts = prepared.pop("_facts")
    fingerprint = _ais_fingerprint(facts, prepared, dynamics=dynamics,
                                   resolved_config=resolved_config)
    identity = run_identity_document(
        fingerprint=fingerprint, end_state_facts=facts,
        source_facts=prepared["source_facts"], source_format=source_format,
        schedule=prepared["schedule"], ais=ais, dynamics=dynamics,
        chosen=list(prepared["chosen_frames"]), reporting=reporting,
        resolved_config=resolved_config)

    def _decide():
        # Read-only: an absent directory is fresh, an orphaned one (owned-looking artefacts with
        # no readable identity to prove what owns them) is refused, a compatible one with an
        # unfinished path requires --resume, and --resume against nothing is refused. See
        # `decide_run_disposition` for the complete state machine.
        try:
            return decide_run_disposition(
                directory, identity, resume=resume, overwrite=overwrite,
                chosen=list(prepared["chosen_frames"]), fingerprint=fingerprint,
                schedule=prepared["schedule"])
        except SystemExit as refusal:
            raise PreflightError(str(refusal)) from None

    disposition, previous_identity = collectively(coordination, _decide,
                                                  what="the AIS run-identity check")

    # EVERY SELECTED PATH, before work begins on any of them -- and before `-odir`, the `.out`,
    # the `.log` or `resolved.config` is created or replaced. Validating lazily meant an invalid
    # path 3 was discovered once path 0 had already been rewritten, and a refusal had by then
    # replaced the prior campaign's provenance with a description of a run that never started.
    if out_dir is not None and prepared.get("cv_definition") is not None:
        from .continuation import ContinuationError, validate_ais_continuation

        try:
            validate_ais_continuation(
                out_dir, definition=prepared["cv_definition"], schedule=prepared["schedule"],
                chosen=list(prepared["chosen_frames"]),
                selected=range(len(prepared["chosen_frames"])))
        except ContinuationError as refusal:
            raise PreflightError(str(refusal)) from None

    return AISPreflight(coordination=coordination, machine=machine, acceleration=acceleration,
                        device_index=index,
                        device_policy=str(machine.get("device_policy") or "local_rank"),
                        device_policy_detail=detail, placement=plan, particles=particles,
                        loaded=loaded,
                        source=Path(source), source_format=source_format,
                        fingerprint=fingerprint, identity=identity,
                        previous_identity=previous_identity, disposition=disposition,
                        **prepared)


def _ais_fingerprint(facts, prepared, *, dynamics, resolved_config) -> str:
    """What a mid-path checkpoint has to match before it may be resumed from.

    Built here rather than in the runtime so the identity that decides whether this directory may
    be written to is the same string the checkpoints are stamped with. Both end states are in it:
    a checkpoint taken under one V1 continued under another would add work along a path that
    changed destination half way.
    """
    import hashlib
    import json as _json

    from ..ais.two_state import TWO_STATE_SCHEMA

    schedule = prepared["schedule"]
    return hashlib.sha256(_json.dumps({
        "end_states": facts,
        "source": prepared["source_facts"]["sha256"],
        "schedule": {k: v for k, v in schedule.items()
                     if k not in ("observations", "lambdas", "note")},
        "ais_schema": [TWO_STATE_SCHEMA["name"], TWO_STATE_SCHEMA["version"]],
        "temperature_K": float(dynamics["temperature_K"]),
        "friction_per_ps": float(dynamics["friction_per_ps"]),
        "seed": int(dynamics["seed"]),
        "resolved_config": resolved_config,
    }, sort_keys=True).encode()).hexdigest()


def _ais_inventory(*, output, log, paths: int, ranks: int = 1) -> OutputInventory:
    """AIS's complete inventory: the global tables AND every per-path artefact.

    A path directory holds a staged trajectory, two CSVs, a checkpoint generation tree and a
    completion manifest, and the run root holds one published trajectory per path plus four
    tables. None of it was named anywhere, so nothing could be checked against it.
    """
    from ..ais import path_trajectory_name

    directory = Path(output).parent if output else Path(".")
    from ..remd.executor import report_path_for_rank

    roles: dict[str, Path] = {}
    # The reports as they are ACTUALLY named. Every rank keeps its own, suffixed `.rankNN`, and
    # listing only rank 0's meant `--overwrite` after a size change left rank 05's report from a
    # six-rank run sitting beside a two-rank one, looking equally current.
    for role, value in (("out", output), ("log", log)):
        if not value:
            continue
        for rank in range(max(int(ranks), 1)):
            name = f"{role}" if rank == 0 else f"{role}_rank{rank:02d}"
            roles[name] = Path(report_path_for_rank(str(value), rank))
    roles["run_identity"] = directory / "AIS_run.json"
    roles["work_table"] = directory / "AIS_work.csv"
    roles["work_summary"] = directory / "AIS_paths.csv"
    roles["hs_table"] = directory / "AIS_hs.csv"
    # The aggregate CV table. In no inventory before, so a CV-disabled rerun over a CV-enabled
    # directory left the old `AIS_cv.csv` in place, describing paths the new run did not measure.
    roles["cv_table"] = directory / "AIS_cv.csv"
    roles["selected_frames"] = directory / "selected_source_frames.csv"
    for index in range(int(paths)):
        roles[f"path_{index:04d}"] = directory / f"path_{index:04d}"
        roles[f"trajectory_{index:04d}"] = directory / path_trajectory_name(index, int(paths))
    # Every per-path artefact is resumable by design: a completed path is skipped and an
    # interrupted one is continued, which is what `--resume` is for and what the run identity
    # and the completion manifests police. What must not be silently replaced is the RUN.
    return OutputInventory(
        roles=roles,
        resumable=frozenset({role for role in roles
                             if role.startswith(("path_", "trajectory_"))}
                            | {"run_identity", "work_table", "work_summary", "hs_table",
                               "cv_table", "selected_frames"}))


def _topology_differences(first, second) -> list[str]:
    """Where two topologies disagree about which atom is which, in words. Empty when they agree."""
    atoms0, atoms1 = list(first.atoms()), list(second.atoms())
    if len(atoms0) != len(atoms1):
        return [f"-p has {len(atoms0)} atom(s) and -p2 has {len(atoms1)}"]
    for position, (a, b) in enumerate(zip(atoms0, atoms1)):
        if (a.name, a.residue.name, a.residue.index) != (b.name, b.residue.name, b.residue.index):
            return [f"atom {position} is {a.residue.name}{a.residue.index}:{a.name} in -p and "
                    f"{b.residue.name}{b.residue.index}:{b.name} in -p2"]
    return []


def _source_ensemble_evidence(source: Path, system_sha256: str) -> dict[str, Any]:
    """Whether the source frames were sampled from V0, from what the trajectory records ITSELF.

    A stage writes the digest of the System it integrated into its AMBER NetCDF. When the source
    carries one, it must be V0's: frames from any other Hamiltonian make every work value the cost
    of a switch that did not start where the record says. A source that records none -- a DCD, a
    foreign file, a trajectory from before the attribute existed -- is still legitimate, and the
    run says it ASSERTED the ensemble rather than claiming to have verified it.
    """
    from ..remd.source_ensemble import trajectory_identity

    recorded, reason = trajectory_identity(source)
    digest = (recorded or {}).get("system_sha256")
    if digest is None:
        return {"status": "asserted",
                "reason": reason or f"{Path(source).name} records no System digest"}
    if digest != system_sha256:
        raise PreflightError(
            f"-source-traj {Path(source).name} records that it was sampled from a System with "
            f"sha256 {digest[:16]}..., and -s hashes to {system_sha256[:16]}.... The source "
            f"ensemble must be V0's, or every work value measures a switch from a state the frames "
            f"were never in. Pass the System that run integrated as -s, or point -source-traj at "
            f"the run you meant.")
    return {"status": "verified", "system_sha256": digest}


def _ais_schedule_claim(*, ais, source_config, dynamics, where, system0, system1):
    """The lambda schedule, with a tau-linear claim checked against the end-state files.

    tau-linear follows REST2's solute-solute scaling from tau0 to 0, so it is only meaningful when
    V0 IS a saved scaled state at tau0 and V1 is the unscaled System that state was scaled from.
    The record beside V0 is read to CHECK that claim; the lambdas themselves come from the
    configuration's number, never from the record.
    """
    from ..build.md import ais_lambda_schedule
    from ..build.record import file_facts
    from ..build.strict import ConfigError
    from ..rest2.states import ScaledStateError, load_scaler_record, scaled_state_identity

    try:
        kind, tau0 = ais_lambda_schedule({"ais": ais, "ais_source": source_config,
                                          "dynamics": dynamics})
    except ConfigError as refusal:
        raise PreflightError(f"{where}: {refusal}") from None
    if kind != "tau-linear" or tau0 is None:
        return kind, tau0
    try:
        identity = scaled_state_identity(system0)
    except ScaledStateError as refusal:
        raise PreflightError(f"{where}: -s {Path(system0).name}: {refusal}") from None
    if identity is None:
        raise PreflightError(
            f"{where}: ais.lambda_schedule is tau-linear, but -s {Path(system0).name} is not a "
            f"saved scaled state (no scaler.yaml beside it). tau-linear follows REST2 scaling from "
            f"tau0 to 0, which means nothing for an arbitrary V0. Use lambda_schedule: linear, or "
            f"pass the state built by `md-openmm build-top --rest2-scaler`.")
    if abs(float(identity["tau"]) - float(tau0)) > 1e-9:
        raise PreflightError(
            f"{where}: ais.lambda_schedule_tau0 is {tau0}, but -s {Path(system0).name} is the "
            f"saved state at tau {identity['tau']} ({identity['record']}). The schedule would "
            f"follow a scaling V0 does not have.")
    source_sha = load_scaler_record(Path(identity["record"]))["source"]["system_sha256"]
    if source_sha != file_facts(system1)["sha256"]:
        raise PreflightError(
            f"{where}: ais.lambda_schedule is tau-linear, but -s2 {Path(system1).name} is not the "
            f"System -s was scaled from ({identity['record']} records sha256 "
            f"{source_sha[:16]}...). tau-linear switches the scaling off, so V1 must be the "
            f"unscaled source of V0.")
    return kind, tau0


def _prepare_ais(loaded: LoadedInputs, *, topology2: Path, system2: Path, source: Path, dynamics,
                 ais, reporting, source_config, collective_variables=None, config_directory=None):
    """Every AIS refusal that needs the end states or the source file, before any output exists."""
    import mdtraj

    from ..ais.run import _source_atom_count, choose_frames
    from ..ais.schedule import switching_schedule
    from ..ais.two_state import EndStateError, TwoStateHamiltonian
    from ..build.record import file_facts

    where = "AIS"
    timestep = _resolve_timestep(loaded, dynamics["timestep_fs"], where=where)

    # THE SECOND END STATE, and the pair. Every way two Systems can fail to be a parameter-only
    # pair is named at once -- particles, masses, constraints, virtual sites, a barostat in either,
    # the force layout, the long-range treatment -- because each produces a plausible work table.
    end_state_1 = load_inputs(topology2, system2, flags=("-p2", "-s2"))
    mismatch = _topology_differences(loaded.pdb.topology, end_state_1.pdb.topology)
    if mismatch:
        raise PreflightError(
            f"{where}: -p and -p2 do not describe the same atoms: {mismatch[0]}. V0 and V1 are "
            f"mixed particle by particle, so the two topologies must agree on every atom.")
    try:
        hamiltonian = TwoStateHamiltonian(loaded.system, end_state_1.system)
    except EndStateError as refusal:
        raise PreflightError(f"{where}: {refusal}") from None

    schedule_kind, tau0 = _ais_schedule_claim(
        ais=ais, source_config=source_config, dynamics=dynamics, where=where,
        system0=loaded.system_path, system1=end_state_1.system_path)

    try:
        schedule = switching_schedule(
            switching_steps=int(ais["switching_steps"]),
            parameter_update_interval_steps=int(ais["parameter_update_interval_steps"]),
            observation_interval_steps=int(ais["observation_interval_steps"]),
            timestep_fs=float(timestep["timestep_fs"]),
            trajectory_interval_steps=int(reporting["crd_printout_solute"]),
            state_interval_steps=int(reporting["info_printout"]),
            checkpoint_interval_steps=int(reporting["checkpoint_printout"]),
            cv_interval_steps=int((collective_variables or {}).get("interval_steps") or 0),
            lambda_schedule=schedule_kind, lambda_schedule_tau0=tau0)
    except (ValueError, SystemExit) as refusal:
        raise PreflightError(f"{where}: {refusal}") from None

    # Parsed here, before -odir exists. Carried on the preflight and passed to `run_one_path` as
    # its own argument -- NOT inside `schedule`, which is serialised into the run records, where a
    # parsed object makes every write fail.
    cv_definition = None
    cv_block = collective_variables or {}
    if cv_block.get("file") and int(cv_block.get("interval_steps") or 0) > 0:
        from ..cv import CVDefinitionError, load_cv_definition

        cv_path = Path(cv_block["file"])
        if not cv_path.is_absolute():
            cv_path = (Path(config_directory) if config_directory else Path(".")) / cv_path
        try:
            cv_definition = load_cv_definition(
                cv_path, topology=loaded.pdb.topology,
                particles=loaded.system.getNumParticles())
        except CVDefinitionError as refusal:
            raise PreflightError(f"{where}: {refusal}") from None

    facts = {"V0": {"system": file_facts(loaded.system_path)["sha256"],
                    "topology": file_facts(loaded.topology_path)["sha256"]},
             "V1": {"system": file_facts(end_state_1.system_path)["sha256"],
                    "topology": file_facts(end_state_1.topology_path)["sha256"]}}

    # -- the source ensemble, read before anything is written ----------------------------------
    source_ensemble = _source_ensemble_evidence(source, facts["V0"]["system"])
    top = mdtraj.Topology.from_openmm(loaded.pdb.topology)
    try:
        n_frames = sum(chunk.n_frames for chunk in mdtraj.iterload(str(source), top=top, chunk=50))
    except Exception as broken:
        raise PreflightError(
            f"-source-traj {source.name} could not be read against {loaded.topology_path.name}: "
            f"{type(broken).__name__}: {broken}") from None

    last = source_config["last_frame"]
    last = n_frames - 1 if last is None else int(last)
    first = int(source_config["first_frame"])
    stride = int(source_config["frame_stride"])
    if last >= n_frames:
        raise PreflightError(f"ais_source.last_frame = {last} but {source.name} holds "
                             f"{n_frames} frame(s) (0..{n_frames - 1})")
    if first > last:
        raise PreflightError(f"ais_source.first_frame = {first} is past last_frame = {last}: "
                             f"the source window is empty, so no path has a starting frame.")
    eligible = list(range(first, last + 1, stride))

    source_atoms = _source_atom_count(source)
    if source_atoms is not None and source_atoms != loaded.particles:
        raise PreflightError(
            f"-source-traj {source.name} holds {source_atoms} atom(s) but "
            f"{loaded.topology_path.name} and {loaded.system_path.name} describe "
            f"{loaded.particles}. The source ensemble must be of the same system the paths are "
            f"run in; a trajectory of a different one reads without error and produces work "
            f"values that mean nothing.")

    try:
        chosen = choose_frames(eligible=eligible, count=int(ais["number_of_paths"]),
                               selection=source_config["selection"],
                               allow_repeats=bool(source_config["allow_repeated_frames"]),
                               seed=int(dynamics["seed"]))
    except (ValueError, SystemExit) as refusal:
        raise PreflightError(f"{where}: {refusal}") from None

    # Hashed ONCE, here. A production source is large and is already read frame by frame above;
    # digesting it twice would double that for a number that has one value -- and the runtime
    # needs the same number for the record AND for the fingerprint.
    return {"timestep": timestep, "schedule": schedule, "cv_definition": cv_definition,
            "end_state_1": end_state_1, "hamiltonian": hamiltonian,
            "source_ensemble": source_ensemble, "chosen_frames": tuple(chosen),
            "eligible_frames": len(eligible), "source_frames": n_frames,
            "source_atoms": source_atoms, "source_facts": file_facts(source),
            "_facts": facts,
            "notes": {"first_frame": first, "last_frame": last, "frame_stride": stride}}
