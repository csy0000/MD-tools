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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["PreflightError", "ExecutionPreflight", "StagePreflight", "LadderPreflight",
           "AISPreflight", "preflight_stage", "preflight_ladder", "preflight_ais",
           "check_output_collisions", "check_input_files", "resolve_launch",
           "check_topology_matches_system"]


class PreflightError(SystemExit):
    """A run that cannot start. A `SystemExit`, so it stops the process without a traceback."""


# ---------------------------------------------------------------------------------------------
# the individual checks
# ---------------------------------------------------------------------------------------------

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
                f"-{first} and -{second} are the same file ({resolved}). They are different "
                f"outputs with different readers; one file cannot be both, and opening it twice "
                f"would have one of them truncate the other.")
        seen[resolved] = name

    for name, value in (inputs or {}).items():
        if not value:
            continue
        resolved = Path(value).resolve(strict=False)
        if resolved in seen:
            raise PreflightError(
                f"-{seen[resolved]} would be written over the input -{name} ({resolved}). The "
                f"run reads that file; writing an output onto it destroys what the run needs.")


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
    """One conventional stage: a trajectory, a restart, a checkpoint, an out and a log."""

    trajectory: Path | None = None


@dataclass(frozen=True)
class LadderPreflight(ExecutionPreflight):
    """A REST2 or rREST2 ladder: one process per thermodynamic state."""

    replicas: int = 0


@dataclass(frozen=True)
class AISPreflight(ExecutionPreflight):
    """AIS switching paths: independent workers, and a source ensemble that already exists."""

    source: Path | None = None
    source_format: str | None = None


# ---------------------------------------------------------------------------------------------
# the mode-aware entry points
# ---------------------------------------------------------------------------------------------

def _common(*, topology, system, outputs, inputs, cpu, device, number_of_groups, replicas,
            protocol, machine_config, check_particles=True):
    """Steps 1-11, in order. Shared by every mode; each mode adds only its own inputs."""
    _check_command_line(cpu=cpu, device=device, number_of_groups=number_of_groups)

    coordination = resolve_launch(number_of_groups=number_of_groups, replicas=replicas,
                                  protocol=protocol)

    check_input_files(p=topology, s=system, **dict(inputs or {}))
    check_output_collisions(outputs=outputs or {},
                            inputs={"p": topology, "s": system, **(inputs or {})})

    machine = _resolve_machine(machine_config)
    acceleration, index, detail = _resolve_platform(machine, cpu=cpu, device=device,
                                                    coordination=coordination)
    particles = check_topology_matches_system(topology, system) if check_particles else None
    return coordination, machine, acceleration, index, detail, particles


def preflight_stage(*, topology, system, coordinates=None, trajectory=None, restart=None,
                    checkpoint=None, output=None, log=None, cpu=False, device=None,
                    machine_config=None, protocol="this stage") -> StagePreflight:
    """A conventional stage, including every stage of an all-in-one workflow."""
    from ..md.stage import check_trajectory_suffix

    if trajectory:
        check_trajectory_suffix(Path(trajectory))

    # A continuation that does not exist yet is not an error here: `--check` validates a whole
    # chain before any of it has run, and every parent after the first is missing by construction.
    inputs = {"c": coordinates} if coordinates and Path(coordinates).exists() else {}
    coordination, machine, acceleration, index, detail, particles = _common(
        topology=topology, system=system,
        outputs={"o": output, "log": log, "x": trajectory, "r": restart, "chk": checkpoint},
        inputs=inputs, cpu=cpu, device=device, number_of_groups=None, replicas=None,
        protocol=protocol, machine_config=machine_config)

    return StagePreflight(coordination=coordination, machine=machine, acceleration=acceleration,
                          device_index=index,
                          device_policy=str(machine.get("device_policy") or "local_rank"),
                          device_policy_detail=detail, particles=particles,
                          trajectory=Path(trajectory) if trajectory else None)


def preflight_ladder(*, topology, system, replicas, coordinates=None, groupfile=None,
                     trajectory=None, restart=None, output=None, log=None,
                     number_of_groups=None, cpu=False, device=None, machine_config=None,
                     protocol="this ladder") -> LadderPreflight:
    """A REST2 or rREST2 ladder. `-ng`, the configured state count and the world must agree."""
    inputs: dict[str, Any] = {}
    if coordinates and Path(coordinates).exists():
        inputs["c"] = coordinates
    if groupfile:
        inputs["groupfile"] = groupfile

    coordination, machine, acceleration, index, detail, particles = _common(
        topology=topology, system=system,
        outputs={"o": output, "log": log, "x": trajectory, "r": restart},
        inputs=inputs, cpu=cpu, device=device, number_of_groups=number_of_groups,
        replicas=int(replicas), protocol=protocol, machine_config=machine_config)

    return LadderPreflight(coordination=coordination, machine=machine,
                           acceleration=acceleration, device_index=index,
                           device_policy=str(machine.get("device_policy") or "local_rank"),
                           device_policy_detail=detail, particles=particles,
                           replicas=int(replicas))


def preflight_ais(*, topology, system, source, number_of_groups=None, output=None, log=None,
                  cpu=False, device=None, machine_config=None) -> AISPreflight:
    """AIS switching paths. The source ensemble is validated by CONTENT, not by suffix."""
    from ..openmm.trajectory import check_trajectory_declaration

    coordination, machine, acceleration, index, detail, particles = _common(
        topology=topology, system=system,
        outputs={"o": output, "log": log},
        inputs={"source-traj": source}, cpu=cpu, device=device,
        number_of_groups=number_of_groups, replicas=None, protocol="AIS",
        machine_config=machine_config)

    # After existence, before any output: a source whose suffix and contents disagree is neither.
    source_format = check_trajectory_declaration(source, what="-source-traj")

    return AISPreflight(coordination=coordination, machine=machine, acceleration=acceleration,
                        device_index=index,
                        device_policy=str(machine.get("device_policy") or "local_rank"),
                        device_policy_detail=detail, particles=particles,
                        source=Path(source), source_format=source_format)
