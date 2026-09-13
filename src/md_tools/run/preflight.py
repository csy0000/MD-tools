"""Everything that must be true before a run may write anything at all.

ONE model, CONSUMED by every authoritative entry point:

    md_tools.run.main.md_run_main          the CLI dispatcher
    md_tools.md.stage.stage_main           generated split stages and all-in-one workflows
    md_tools.remd.generated.replica_main   generated REST2.py and rREST2.py
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
from dataclasses import dataclass, field
from pathlib import Path
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
                      coordination) -> tuple[Any, Any, str]:
    """Platform, device and the policy that placed it. Proves a Context can be created."""
    from ..openmm.platform_policy import (PlatformRequest, device_index_for,
                                          resolve_platform_request)

    policy = str(machine.get("device_policy") or "local_rank")
    request = PlatformRequest.from_machine(machine, cpu=cpu)

    index = device
    detail = "named on the command line (--device)"
    if index is None:
        if request.name != "CUDA":
            detail = "not a CUDA platform"
        elif policy == "openmm":
            detail = "machine.openmm.device_policy: openmm -- OpenMM selects"
        else:
            from ..remd.engine import visible_cuda_devices

            index = device_index_for(policy=policy, rank=coordination.rank,
                                     size=coordination.size,
                                     devices=visible_cuda_devices(probe=coordination.size > 1))
            detail = (f"machine.openmm.device_policy: local_rank "
                      f"(rank {coordination.rank} of {coordination.size})")
            if index is None and coordination.size > 1:
                raise PreflightError(
                    "the CUDA platform was selected under MPI with device_policy: local_rank, "
                    "but no CUDA device is visible to this rank. Refusing rather than letting "
                    "every rank fall onto one GPU.")

    return resolve_platform_request(request, device_index=index), index, detail


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


def load_inputs(topology, system) -> LoadedInputs:
    """Read `-p` and `-s`, and refuse a pair that does not describe the same particles."""
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from ..md._stages import count_barostats

    topology_path, system_path = Path(topology), Path(system)
    try:
        pdb = PDBFile(str(topology_path))
    except Exception as broken:
        raise PreflightError(f"-p {topology_path} is not a PDB this build can read: "
                             f"{type(broken).__name__}: {broken}") from None
    try:
        base = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
    except Exception as broken:
        raise PreflightError(f"-s {system_path} is not a serialised OpenMM System: "
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
        scaled = build_scaled_system(loaded.system, solute_indices, float(tau),
                                     excluded_bonds, prepare_for_switching=True)
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
            acceleration_record(self.acceleration, mpi_rank=self.coordination.rank,
                                mpi_size=self.coordination.size, local_rank=self.device_index),
            device_policy_detail=self.device_policy_detail)


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
    #: for a phase-space stream a reservoir will later be built from.
    hamiltonian_identity: dict[str, Any] | None = None


@dataclass(frozen=True)
class LadderPreflight(ExecutionPreflight):
    """A REST2 or rREST2 ladder: one process per thermodynamic state."""

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
    #: rREST2 only. The exact `reservoir.yaml` text rank 0 will publish, built and validated
    #: BEFORE `-odir` exists -- see `preflight_ladder`. Carried here so the source is opened once,
    #: by the preflight, rather than reopened after output creation to rediscover the same facts.
    reservoir_declaration: str | None = None
    #: sha256 of `reservoir_declaration`, so every rank can verify it reads the bytes rank 0 wrote.
    reservoir_digest: str | None = None
    #: What the source phase-space file actually proved: its path, frame count and time window.
    reservoir_source: dict[str, Any] | None = None
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
    solute: tuple = ()
    excluded_bonds: tuple = ()
    switcher: Any = None
    chosen_frames: tuple = ()
    eligible_frames: int = 0
    source_frames: int = 0
    source_atoms: int | None = None
    force_audit: dict[str, Any] | None = None
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
    leniency for the all-in-one `--check` chain, and it is also complete leniency for a typo: a
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
            protocol, machine_config, check_particles=True, load=False):
    """Steps 1-11, in order. Shared by every mode; each mode adds only its own inputs."""
    _check_command_line(cpu=cpu, device=device, number_of_groups=number_of_groups)

    coordination = resolve_launch(number_of_groups=number_of_groups, replicas=replicas,
                                  protocol=protocol)

    check_input_files(p=topology, s=system, **dict(inputs or {}))
    check_output_collisions(outputs=outputs or {},
                            inputs={"p": topology, "s": system, **(inputs or {})})

    machine = _resolve_machine(machine_config)

    def _platform_and_inputs():
        _fail_here_if_asked(coordination)
        resolved = _resolve_platform(machine, cpu=cpu, device=device, coordination=coordination)
        # Loaded once when the mode has refusals that need to look inside the System; the
        # particle comparison then comes from the loaded pair rather than a second parse.
        prepared = load_inputs(topology, system) if load else None
        if prepared is not None:
            count = prepared.particles
        else:
            count = check_topology_matches_system(topology, system) if check_particles else None
        return resolved, prepared, count

    # THE COLLECTIVE POINT. Everything above is a property of the command line or of files every
    # rank sees identically; everything inside is rank-local -- this rank's device, this rank's
    # view of the filesystem, this rank's CUDA context.
    (acceleration, index, detail), loaded, particles = collectively(
        coordination, _platform_and_inputs, what=f"the {protocol} preflight")
    return coordination, machine, acceleration, index, detail, particles, loaded


def preflight_stage(*, topology, system, coordinates=None, trajectory=None, restart=None,
                    checkpoint=None, output=None, log=None, cpu=False, device=None,
                    machine_config=None, protocol="this stage", pending_parent=None,
                    timestep_fs=None, ensemble=None, tau=0.0, stage=None,
                    number_of_groups=None, groupfile=None, whole=None, segment=1,
                    source_trajectory=None) -> StagePreflight:
    """A conventional stage, including every stage of an all-in-one workflow."""
    from ..md.stage import check_trajectory_suffix

    _reject_flags_outside_their_protocol(
        protocol_name="a cMD stage", number_of_groups=number_of_groups, groupfile=groupfile,
        source_trajectory=source_trajectory)

    if trajectory:
        check_trajectory_suffix(Path(trajectory))

    inputs = _continuation_inputs(coordinates, pending_parent, where=protocol)
    inventory = _stage_inventory(output=output, log=log, trajectory=trajectory, whole=whole,
                                 restart=restart, segment=segment,
                                 checkpoint=checkpoint)
    coordination, machine, acceleration, index, detail, particles, loaded = _common(
        topology=topology, system=system,
        outputs=inventory.roles,
        inputs=inputs, cpu=cpu, device=device, number_of_groups=None, replicas=None,
        protocol=protocol, machine_config=machine_config, load=timestep_fs is not None)

    # A cMD stage is serial by construction. Checked HERE, after the coordination is open and
    # before anything is created, so a plural launch stops at the preflight with every rank
    # agreeing rather than partway through with N writers.
    reject_plural_launch(coordination, what=protocol)

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
                          device_policy_detail=detail, particles=particles, loaded=loaded,
                          timestep=resolved_timestep, inventory=inventory,
                          trajectory=Path(trajectory) if trajectory else None, **prepared)


def _prepare_stage(loaded: LoadedInputs, *, stage: dict[str, Any], name: str,
                   where: str) -> dict[str, Any]:
    """Build the System a stage will actually integrate, and refuse here if it cannot be built.

    Everything below used to run AFTER the `.out` and the `.log` were open: the solute selection,
    the omega classification, `build_scaled_system` with its force audit, the implicit/NPT and
    fixed-tau/NPT checks, the restraint, and the barostat. Each is a refusal that arrived attached
    to a directory that reads as a run that started -- and `build_scaled_system` in particular
    refuses a System carrying a force the convention cannot place, which is not a rare case on a
    hand-built System.
    """
    from ..md._stages import add_barostat, add_positional_restraint, count_barostats, derive_seed
    from ..md.stage import solute_atom_indices
    from ..openmm.system import classify_omega_bonds

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

    # SCALE FIRST, on the bare System, then restrain, then add the barostat. The scaler audits
    # every force and refuses one it cannot classify; the restraint and the barostat are stage
    # machinery rather than terms of the molecular Hamiltonian, so neither may be scaled. Scaling
    # last would scale them, and a fixed-tau walker would then build a different System from the
    # ladder rung it is supposed to match.
    if tau > 0.0:
        if stage.get("ensemble") != "NVT":
            raise PreflightError(
                f"{where} runs at tau={tau} but declares ensemble {stage.get('ensemble')}. A "
                f"scaled run samples the fixed-volume ensemble of the ladder rung it sits at; a "
                f"barostat would sample a different distribution.")
        omega = classify_omega_bonds(loaded.pdb.topology, solute, route="peptide", ligand_sdf=None)
        excluded = [tuple(int(a) for a in bond)
                    for bond in omega.get("omega_unscaled_bonds", [])]
        _audit, system = check_scaling_plan(loaded, solute_indices=solute,
                                            excluded_bonds=excluded, tau=tau,
                                            where=f"{where} fixed-tau scaling")
    else:
        from ..rest2.scaler import clone_system

        # A copy, so a plan never hands the runtime the object `LoadedInputs` holds: the restraint
        # and the barostat below mutate it in place, and two plans built from one `LoadedInputs`
        # would otherwise accumulate each other's machinery.
        system = clone_system(system)

    hamiltonian_identity = None
    if stage.get("phase_space_interval_steps"):
        from ..rest2 import identity_record

        # Taken BEFORE the restraint and barostat: they are properties of how this stage is run,
        # not terms of the energy the ensemble is defined by. An identity taken after them claims
        # a CustomExternalForce the ladder rung a reservoir refreshes does not have.
        hamiltonian_identity = identity_record(
            system, tau=tau, temperature_k=float(stage["temperature_K"]),
            ensemble=stage.get("ensemble"), solute_indices=solute, excluded_bonds=excluded)

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
                     whole=None, segment=1) -> OutputInventory:
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
        roles["collective_variables"] = Path(trajectory).with_suffix(".cv.csv")
        roles["collective_variables_definition"] = cv_sidecar_path(
            Path(trajectory).with_suffix(".cv.csv"))
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
             f"  particles         {result.particles}"]
    if result.timestep:
        lines.append(f"  timestep          {result.timestep['timestep_fs']} fs "
                     f"({result.timestep['basis']})")
    lines.extend(f"  {label:<18}{value}" for label, value in extra)
    print("\n".join(lines), file=_sys.stdout)
    return 0


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
        f"  Run it in one process, or use a protocol that coordinates: REST2/rREST2 place one "
        f"rank per thermodynamic state, and AIS distributes paths by `paths_for_rank`.")


def _reject_flags_outside_their_protocol(*, protocol_name, number_of_groups=None, groupfile=None,
                                         source_trajectory=None, trajectory=None,
                                         coordinates=None, restart=None, checkpoint=None):
    """Every accepted flag must do its documented job here, or be refused before any output.

    A flag that a protocol parses and then ignores is worse than one it rejects: the run
    completes, the record shows the flag was given, and nothing anywhere did what it says. These
    are the pairings where that was true.
    """
    if number_of_groups is not None:
        raise PreflightError(
            f"-ng is a REST2/rREST2 flag and {protocol_name} has no replicas to group. It was "
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
    for value, flag, what in ((trajectory, "-x", "its own per-path trajectory names"),
                              (coordinates, "-c", "no continuation"),
                              (restart, "-r", "no single output restart"),
                              (checkpoint, "-chk", "no single checkpoint")):
        if value is not None:
            raise PreflightError(
                f"{flag} has no meaning for {protocol_name}, which has {what}. Refusing it "
                f"rather than accepting a path nothing writes to.")



def _ligand_sdf_beside(system):
    """`<system stem>.sdf`, when `build-top` retained one for a SMILES-built solute.

    Returns None for a peptide build, which writes no SDF -- so this cannot turn a protein run
    into a ligand run, and the absence is as meaningful as the presence.
    """
    if not system:
        return None
    candidate = Path(system).with_suffix(".sdf")
    return candidate if candidate.is_file() else None

def preflight_ladder(*, topology, system, replicas, coordinates=None, groupfile=None,
                     trajectory=None, restart=None, checkpoint=None, output=None, log=None,
                     number_of_groups=None, cpu=False, device=None, machine_config=None,
                     protocol="this ladder", pending_parent=None, timestep_fs=None,
                     ensemble=None, tau=0.0, solute_indices=None, excluded_bonds=(),
                     route=None, source_trajectory=None,
                     reservoir=False, ladder=None, out_dir=None) -> LadderPreflight:
    """A REST2 or rREST2 ladder. `-ng`, the configured state count and the world must agree."""
    if source_trajectory is not None:
        _reject_flags_outside_their_protocol(protocol_name="a REST2/rREST2 ladder",
                                             source_trajectory=source_trajectory)

    inputs: dict[str, Any] = dict(_continuation_inputs(coordinates, pending_parent,
                                                       where=protocol))
    if groupfile:
        inputs["groupfile"] = groupfile
    inventory = _ladder_inventory(protocol=protocol, replicas=int(replicas), output=output,
                                  log=log, trajectory=trajectory, restart=restart,
                                  checkpoint=checkpoint, groupfile=groupfile,
                                  reservoir=bool(reservoir),
                                  per_tau=bool((ladder or {}).get("per_tau_equilibration")))

    coordination, machine, acceleration, index, detail, particles, loaded = _common(
        topology=topology, system=system,
        outputs=inventory.roles,
        inputs=inputs, cpu=cpu, device=device, number_of_groups=number_of_groups,
        replicas=int(replicas), protocol=protocol, machine_config=machine_config,
        load=timestep_fs is not None or solute_indices is not None or route is not None)

    resolved_timestep = audit = scaled = None
    if timestep_fs is not None:
        resolved_timestep = _resolve_timestep(loaded, timestep_fs, where=protocol)
    if loaded is not None:
        # A REST2 runtime is NVT by contract, so `ensemble` is normally not passed; when it is,
        # the same two impossibilities are refused as for a stage.
        check_ensemble(loaded, ensemble=ensemble, tau=tau, where=protocol)
    solute_record = None
    if loaded is not None and solute_indices is None and route is not None:
        # Derived here rather than by the writer that used to do it. The omega classification
        # carries its own refusal -- an amide that is neither ordinary nor proline-like -- and it
        # used to fire from inside `write_solute_document`, after `-odir` and both logs existed.
        from ..remd.generated import solute_document

        # THE SDF BESIDE THE SYSTEM, and the route that follows from it.
        #
        # `build-top` writes `<system stem>.sdf` when and only when it built the solute from
        # SMILES through the small-molecule route. Its presence is therefore not a guess about
        # what this system is -- it is the build recording what it did, in the one place a later
        # run already has a path to. A peptide build writes none, so the default is unchanged and
        # every existing ladder classifies exactly as before.
        #
        # Resolved here rather than left to a flag the operator must remember: forgetting
        # `--route ligand` produced a refusal listing every backbone amide as unclassifiable,
        # which is safe but reads like a chemistry problem rather than a missing argument.
        ligand_sdf = _ligand_sdf_beside(system)
        if ligand_sdf is not None and route == "peptide":
            route = "ligand"

        try:
            solute_record = solute_document(loaded.pdb.topology, loaded.system, route=route,
                                            ligand_sdf=ligand_sdf)
        except SystemExit as refusal:
            raise PreflightError(f"{protocol}: {refusal}") from None
        span = solute_record.get("solute_atom_range")
        if span and solute_record.get("solute_atom_indices_are_contiguous", False):
            solute_indices = list(range(int(span[0]), int(span[1]) + 1))
        else:
            solute_indices = list(range(int(solute_record["n_solute_atoms"])))
        excluded_bonds = [tuple(int(a) for a in pair) for pair in
                          (solute_record.get("rest2") or {}).get("omega_excluded_bonds", [])]

    if solute_indices is not None:
        # The force classification, the omega handling and the scaled-System construction, all
        # before `solute.yaml`, `_protocol.py` or a group file exists. An unclassifiable force
        # used to be found once the run tree was already on disk.
        audit, scaled = check_scaling_plan(loaded, solute_indices=solute_indices,
                                           excluded_bonds=excluded_bonds, tau=tau, where=protocol)

    # -- EVERY rung, not just the top one -------------------------------------------------------
    #
    # `check_scaling_plan` above validates the force layout using one System at `tau`. That is not
    # the ladder: a ladder is N Systems, and the driver used to build all of them itself, at
    # runtime, after every output file existed. So a force that classifies at tau_max and fails at
    # an intermediate rung was found in the worst possible place.
    #
    # These are built through `build_rung_systems` -- the SAME function `Protocol.build_systems`
    # delegates to -- and not through `check_scaling_plan`, whose `prepare_for_switching=True`
    # makes a different rung 0. The driver consumes exactly these.
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
    if solute_indices is not None and ladder is not None and loaded is not None:
        from ..remd.generated import tau_ladder

        rungs_tau = tuple(float(t) for t in
                          tau_ladder(int(replicas), float(ladder["tau_max"])))

        def _rungs():
            from ..remd.protocol import ProtocolError, build_rung_systems

            try:
                built, full_audit = build_rung_systems(
                    loaded.system, list(solute_indices), rungs_tau,
                    excluded_bonds=list(excluded_bonds),
                    # NOT the config's `dynamics.pressure_bar`. A REST2/rREST2 runtime is NVT by
                    # contract and the `Protocol` this ladder becomes carries `pressure_bar=None`;
                    # the config block may still name a pressure for the equilibration stages that
                    # precede the ladder. Reading it here refused every healthy explicit-solvent
                    # ladder for requesting a barostat nobody had asked the ladder for.
                    pressure_bar=None,
                    # The same restraints on every rung, added after scaling. Empty unless
                    # `umbrella.file` is set, so every existing ladder builds exactly as before.
                    restraints=ladder_restraints)
            except ProtocolError as refusal:
                raise PreflightError(f"{protocol}: {refusal}") from None
            except Exception as broken:
                raise PreflightError(
                    f"{protocol}: the ladder's rung Systems could not be constructed: "
                    f"{type(broken).__name__}: {broken}") from None
            return built, full_audit

        built_systems, audit = collectively(coordination, _rungs,
                                            what="the ladder's rung Systems")
        rung_systems = tuple(built_systems)

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

    # -- the rREST2 reservoir, validated HERE ---------------------------------------------------
    #
    # It used to be built at helper-publication time, after `out.mkdir()`: a reservoir that did
    # not exist, held no frames, or could not be read as phase space was discovered with the run
    # directory already created. Worse, the same source was then opened AGAIN by the driver, after
    # `_begin` had created the analysis file and the per-state trajectories, to rediscover facts
    # this step had already established -- two readings of one file, either of which could be the
    # one that refuses.
    #
    # Everything predictable about the source is therefore settled before a single byte of output
    # exists, and the exact text rank 0 will publish is carried out of here with its digest.
    declaration = digest = source_facts = None
    if reservoir and ladder is not None:
        def _declare():
            from ..remd.generated import reservoir_declaration_text

            try:
                return reservoir_declaration_text(ladder, Path(out_dir) if out_dir else Path("."))
            except SystemExit as refusal:
                raise PreflightError(f"{protocol}: {refusal}") from None

        # Collectively: a reservoir every rank can see is a different failure from one only some
        # ranks can, and a plural launch must refuse as a whole rather than have rank 3 alone walk
        # into a barrier the others already left.
        declaration = collectively(coordination, _declare, what="the rREST2 reservoir")
        digest = hashlib.sha256(declaration.encode("utf-8")).hexdigest()
        source_facts = (yaml.safe_load(declaration) or {}).get("source")

        # And the DEEPER checks, still read-only, still before any output: that the source records
        # the same Hamiltonian as the top rung this ladder will refresh, holds the velocities the
        # policy needs, and describes this molecule. Those used to run only from the driver's
        # `_prepare`, after `_begin` had created the analysis file and the per-state
        # trajectories -- so a source recorded at another tau failed a ladder that had already
        # written output. `_prepare` still checks them (it is what materialises the reservoir, and
        # a check that runs only elsewhere can be bypassed); this is the same code called early.
        if scaled is not None:
            def _validate_source():
                from ..remd.reservoir import ReservoirError, validate_source_read_only

                try:
                    return validate_source_read_only(
                        yaml.safe_load(declaration),
                        # The declaration records `phase_space` relative to the run directory's
                        # PARENT, which is what `_prepare` resolves it against too.
                        declaration_directory=(Path(out_dir).parent if out_dir else Path(".")),
                        system=scaled, tau_max=float(tau),
                        temperature_k=float(ladder["dynamics"]["temperature_K"]),
                        solute_indices=solute_indices or (),
                        excluded_bonds=excluded_bonds or (),
                        periodic=bool(scaled.usesPeriodicBoundaryConditions()))
                except ReservoirError as refusal:
                    raise PreflightError(f"{protocol}: {refusal}") from None

            source_facts = dict(source_facts or {},
                                verified=collectively(coordination, _validate_source,
                                                      what="the rREST2 reservoir source"))

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
                           device_policy_detail=detail, particles=particles, loaded=loaded,
                           timestep=resolved_timestep, replicas=int(replicas),
                           inventory=inventory,
                           force_audit=audit, scaled_system=scaled,
                           solute_indices=tuple(int(i) for i in (solute_indices or ())),
                           tau_list=rungs_tau, rung_systems=rung_systems,
                           excluded_bonds=tuple(tuple(int(a) for a in b) for b in excluded_bonds),
                           reservoir_declaration=declaration, reservoir_digest=digest,
                           reservoir_source=source_facts, cv_definition=cv_definition,
                           ladder_restraints=tuple(ladder_restraints),
                           notes={"solute_document": solute_record} if solute_record else {})


def _cv_interval_of(document) -> int:
    """The configured CV cadence for a resolved protocol document, or 0 when reporting is off."""
    block = (document or {}).get("collective_variables") or {}
    try:
        return int(block.get("interval_steps") or 0)
    except (TypeError, ValueError):
        return 0


def _ladder_inventory(*, protocol, replicas, output, log, trajectory, restart, checkpoint,
                      groupfile, reservoir=False, per_tau=False) -> OutputInventory:
    """A ladder's complete inventory: the run-level files AND every per-state and per-rank one.

    `remd0.nc .. remdN-1.nc` are the scientific result and were not in any inventory at all --
    one per thermodynamic state, written by the root, and silently replaceable. So were
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
    if reservoir:
        # Written by rank 0 and read by every rank a moment later; an rREST2 launch that found a
        # stale one from another ladder would draw its probability-one transfers from it.
        roles["reservoir_declaration"] = directory / "reservoir.yaml"
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


def preflight_ais(*, topology, system, source, number_of_groups=None, output=None, log=None,
                  cpu=False, device=None, machine_config=None, dynamics=None, ais=None,
                  reporting=None, source_config=None, groupfile=None, trajectory=None,
                  coordinates=None, restart=None, checkpoint=None,
                  out_dir=None, resolved_config=None, overwrite=False,
                  resume=False, collective_variables=None) -> AISPreflight:
    """AIS switching paths. The source ensemble is validated by CONTENT, not by suffix.

    When the resolved configuration is supplied -- which the runtime always does -- every
    remaining refusal happens here too: the timestep against the masses, the barostat, the
    switching and reporting divisibility, the source window, the source's own atom count, the
    force classification, the scaled System, and the frame selection. The result carries all of
    it, so `ais_main` runs the schedule it validated rather than building a second one.
    """
    from ..openmm.trajectory import check_trajectory_declaration

    # -x is refused for AIS: path trajectories are named by `path_trajectory_name(index, total)`,
    # so a single -x could only be accepted and ignored. The rest have no AIS meaning either.
    _reject_flags_outside_their_protocol(
        protocol_name="AIS", groupfile=groupfile, trajectory=trajectory,
        coordinates=coordinates, restart=restart, checkpoint=checkpoint)

    coordination, machine, acceleration, index, detail, particles, loaded = _common(
        topology=topology, system=system,
        outputs={"o": output, "log": log},
        inputs={"source-traj": source}, cpu=cpu, device=device,
        number_of_groups=number_of_groups, replicas=None, protocol="AIS",
        machine_config=machine_config, load=dynamics is not None)

    # After existence, before any output: a source whose suffix and contents disagree is neither.
    source_format = check_trajectory_declaration(source, what="-source-traj")

    if dynamics is None:
        return AISPreflight(coordination=coordination, machine=machine, acceleration=acceleration,
                            device_index=index,
                            device_policy=str(machine.get("device_policy") or "local_rank"),
                            device_policy_detail=detail, particles=particles,
                            source=Path(source), source_format=source_format)

    prepared = _prepare_ais(
        loaded, source=Path(source), dynamics=dynamics, ais=ais,
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
    fingerprint = _ais_fingerprint(loaded, prepared, dynamics=dynamics,
                                   resolved_config=resolved_config)
    identity = run_identity_document(
        fingerprint=fingerprint,
        topology_facts=prepared["_topology_facts"], system_facts=prepared["_system_facts"],
        source_facts=prepared["source_facts"], source_format=source_format,
        schedule=prepared["schedule"], ais=ais, dynamics=dynamics,
        chosen=list(prepared["chosen_frames"]), reporting=reporting,
        resolved_config=resolved_config)

    def _decide():
        # Read-only: an absent directory is fresh, an orphaned one (owned-looking artefacts with
        # no readable identity to prove what owns them) is refused, a compatible one with an
        # unfinished path requires --resume, and --resume against nothing is refused. See
        # `decide_run_disposition` for the complete state machine this single call replaces --
        # it used to be "if overwrite: overwrite; elif the identity file exists: resume;
        # else: fresh", which adopted an unidentified directory as fresh and labelled every
        # compatible directory "resume" whether or not --resume was actually given.
        try:
            return decide_run_disposition(
                directory, identity, resume=resume, overwrite=overwrite,
                chosen=list(prepared["chosen_frames"]), fingerprint=fingerprint,
                schedule=prepared["schedule"])
        except SystemExit as refusal:
            raise PreflightError(str(refusal)) from None

    disposition, previous_identity = collectively(coordination, _decide,
                                                  what="the AIS run-identity check")

    prepared.pop("_topology_facts")
    prepared.pop("_system_facts")

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
                        device_policy_detail=detail, particles=particles, loaded=loaded,
                        source=Path(source), source_format=source_format,
                        fingerprint=fingerprint, identity=identity,
                        previous_identity=previous_identity, disposition=disposition,
                        **prepared)


def _ais_fingerprint(loaded, prepared, *, dynamics, resolved_config) -> str:
    """What a mid-path checkpoint has to match before it may be resumed from.

    Built here rather than in the runtime so the identity that decides whether this directory may
    be written to is the same string the checkpoints are stamped with. Two derivations of one
    fingerprint is two chances to disagree, and the one that decides is whichever runs later.
    """
    import hashlib
    import json as _json

    schedule = prepared["schedule"]
    return hashlib.sha256(_json.dumps({
        "system": prepared["_system_facts"]["sha256"],
        "topology": prepared["_topology_facts"]["sha256"],
        "source": prepared["source_facts"]["sha256"],
        "schedule": {k: v for k, v in schedule.items()
                     if k not in ("observations", "taus", "note")},
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


def _prepare_ais(loaded: LoadedInputs, *, source: Path, dynamics, ais, reporting, source_config,
                 collective_variables=None, config_directory=None):
    """Every AIS refusal that needs the System or the source file, before any output exists."""
    import mdtraj

    from ..ais.run import _source_atom_count, choose_frames
    from ..ais.schedule import switching_schedule
    from ..md.stage import solute_atom_indices
    from ..openmm.system import classify_omega_bonds

    where = "AIS"
    timestep = _resolve_timestep(loaded, dynamics["timestep_fs"], where=where)

    # FIXED VOLUME, decided before the run directory exists. A pressure-volume term in the work
    # would make the path measure something the Jarzynski/Crooks relations are not written for.
    if loaded.barostats:
        raise PreflightError(
            "the prepared System carries a barostat. AIS switches at FIXED VOLUME: each path "
            "keeps the box of the frame it started from, and no pressure-volume term enters the "
            "work. Build the System without a barostat.")

    try:
        schedule = switching_schedule(
            tau_start=float(ais["tau_start"]), tau_end=float(ais["tau_end"]),
            switching_steps=int(ais["switching_steps"]),
            parameter_update_interval_steps=int(ais["parameter_update_interval_steps"]),
            observation_interval_steps=int(ais["observation_interval_steps"]),
            timestep_fs=float(timestep["timestep_fs"]),
            trajectory_interval_steps=int(reporting["crd_printout_solute"]),
            state_interval_steps=int(reporting["info_printout"]),
            checkpoint_interval_steps=int(reporting["checkpoint_printout"]),
            cv_interval_steps=int((collective_variables or {}).get("interval_steps") or 0))
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

    solute = solute_atom_indices(loaded.pdb.topology)
    omega = classify_omega_bonds(loaded.pdb.topology, solute, route="peptide", ligand_sdf=None)
    excluded = tuple(tuple(int(a) for a in bond)
                     for bond in omega.get("omega_unscaled_bonds", []))
    audit, _scaled = check_scaling_plan(loaded, solute_indices=solute, excluded_bonds=excluded,
                                        tau=float(ais["tau_start"]), where="AIS tau switching")
    from ..rest2.scaler import TauSwitcher

    switcher = TauSwitcher(loaded.system, solute, excluded)

    # -- the source ensemble, read before anything is written ----------------------------------
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
    from ..build.record import file_facts

    return {"timestep": timestep, "schedule": schedule, "solute": tuple(int(i) for i in solute),
            "cv_definition": cv_definition,
            "excluded_bonds": excluded, "switcher": switcher, "chosen_frames": tuple(chosen),
            "eligible_frames": len(eligible), "source_frames": n_frames,
            "source_atoms": source_atoms, "force_audit": audit,
            "source_facts": file_facts(source),
            "_topology_facts": file_facts(loaded.topology_path),
            "_system_facts": file_facts(loaded.system_path),
            "notes": {"omega": omega, "first_frame": first, "last_frame": last,
                      "frame_stride": stride}}
