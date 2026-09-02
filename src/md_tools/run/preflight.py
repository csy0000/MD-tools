"""Everything that must be true before a run may write anything at all.

ONE model, called from every entry point -- `md-openmm md-run`, `stage_main`, `replica_main`,
`ais_main`, and the generated `min.py` / `REST2.py` / `AIS.py` that call those directly.

That last clause is the reason this module exists. The guards used to live in `md-run`, which made
them a property of one entry point rather than of the runtime; a generated wrapper reached the same
code with none of them. A safe outer wrapper hiding an unsafe runtime is worse than no wrapper,
because it makes the unsafe path look tested.

ORDER MATTERS AS MUCH AS THE CHECKS. Everything here runs before `-odir`, `resolved.config`, the
`.out`, the `.log`, a group file, a trajectory, a checkpoint or a restart exists:

     1. the flags name the KIND of thing they claim
     2. the output paths do not collide, compared as RESOLVED paths
     3. the run input parses and resolves, duplicate keys and all
     4. the topology, System and AIS source exist and are what they claim to be
     5. the MPI launch can coordinate itself, and every count agrees
     6. the machine configuration is loadable and valid -- absent is not invalid
     7. `--cpu` and `--device` are compatible
     8. the device policy resolves to a device
     9. the requested OpenMM platform exists
    10. a CUDA Context can actually be created on it
    11. the trajectory suffix suits the protocol

A refusal at any step leaves the filesystem exactly as it was. A `-odir` holding a
`resolved.config` and a `.log` is indistinguishable from a run that happened, and the next person
to look will read it as one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Preflight", "PreflightError", "check_output_collisions", "check_input_files",
           "resolve_launch"]


class PreflightError(SystemExit):
    """A run that cannot start. A `SystemExit` so it stops the process without a traceback."""


def check_output_collisions(**paths: str | None) -> None:
    """No two named outputs may be the same file.

    Compared as RESOLVED paths, so `sub/../run.out` and `./run.out` are recognised as one file.
    String comparison misses that, and the two would then be opened twice with one of them
    silently truncating the other.

    `resolve(strict=False)` because these are outputs: they do not exist yet, and requiring them
    to would make the check useless exactly when it matters.
    """
    seen: dict[Path, str] = {}
    for name, value in paths.items():
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


def check_input_files(**paths: str | None) -> None:
    """Every named input must exist and be a file. Directories are named as such."""
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


def resolve_launch(*, number_of_groups: int | None = None, replicas: int | None = None,
                   protocol: str = "this run"):
    """The MPI launch, validated. Returns the one `Coordination` the run will use.

    Delegates entirely to `md_tools.remd.mpi`, which is the only MPI authority. Nothing here
    decides what to do about a missing `mpi4py`; it refuses, and this passes the refusal on.
    """
    from ..remd.mpi import Coordination

    return Coordination.open(number_of_groups=number_of_groups, replicas=replicas,
                             protocol=protocol)


@dataclass
class Preflight:
    """The result of a successful preflight: everything the run needs, already validated.

    Built by `Preflight.run(...)`. A run that holds one of these may create output; nothing else
    may, which is the invariant the ordering above exists to keep.
    """

    coordination: Any
    machine: dict[str, Any]
    acceleration: Any
    device_index: Any = None
    device_policy: str = "local_rank"
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def run(cls, *, topology: str | Path, system: str | Path,
            source: str | Path | None = None,
            outputs: dict[str, str | None] | None = None,
            trajectory_format: str | None = None,
            cpu: bool = False, device: int | str | None = None,
            number_of_groups: int | None = None, replicas: int | None = None,
            protocol: str = "this run",
            machine_config: str | Path | None = None,
            probe_cuda: bool = True) -> "Preflight":
        """Every check, in the order the module docstring gives, before anything is written."""
        from ..openmm.platform_policy import (PlatformRequest, device_index_for,
                                              resolve_platform_request)
        from ..openmm.trajectory import check_trajectory_declaration
        from ..registry.userconfig import machine_openmm_settings

        # 1-2. the command line describes something coherent
        check_output_collisions(**(outputs or {}))
        if cpu and device is not None:
            raise PreflightError(
                "--cpu and --device are contradictory: the CPU platform has no device to place. "
                "--device says which GPU a CUDA run goes to, never whether it is one.")

        # 3-4. the inputs exist and are what they claim
        check_input_files(p=topology, s=system)
        source_format = None
        if source is not None:
            source_format = check_trajectory_declaration(source, what="-source-traj")
        if trajectory_format is not None:
            from ..md.stage import check_trajectory_suffix

            check_trajectory_suffix(Path(trajectory_format))

        # 5. the launch. Before the machine configuration, because a launch that cannot coordinate
        #    is wrong on every machine, and its message should not be preceded by one about YAML.
        coordination = resolve_launch(number_of_groups=number_of_groups, replicas=replicas,
                                      protocol=protocol)

        # 6. the machine. An ABSENT configuration is legitimate and resolves to the built-in
        #    defaults; an existing INVALID one is fatal and is not replaced by them.
        machine = machine_openmm_settings(machine_config)

        # 7-9. what platform, and where
        policy = str(machine.get("device_policy") or "local_rank")
        request = PlatformRequest.from_machine(machine, cpu=cpu)
        index = device
        if index is None and request.name == "CUDA":
            from ..remd.engine import visible_cuda_devices

            index = device_index_for(policy=policy, rank=coordination.rank,
                                     size=coordination.size,
                                     devices=visible_cuda_devices(
                                         probe=coordination.size > 1))
            if index is None and policy == "local_rank" and coordination.size > 1:
                raise PreflightError(
                    "the CUDA platform was selected under MPI with device_policy: local_rank, but "
                    "no CUDA device is visible to this rank. Refusing rather than letting every "
                    "rank fall onto one GPU.")

        # 10. a Context, actually created. Listing the platform is not the same as having a device.
        acceleration = resolve_platform_request(request, device_index=index,
                                                probe=probe_cuda)

        return cls(coordination=coordination, machine=machine, acceleration=acceleration,
                   device_index=index, device_policy=policy,
                   notes={"source_format": source_format})
