"""Which OpenMM platform a run uses, decided in ONE place for every runtime.

The policy, in one sentence: **CUDA unless the user wrote `--cpu`.**

`md-openmm md-run` is meant to be read as `pmemd.cuda`. A silent fall back to the CPU still
finishes, still writes a trajectory, still says `status: completed` — two orders of magnitude
later, on a machine whose GPU was simply not visible to the process. The result is not obviously
wrong, which is what makes it expensive: it is found weeks later, if at all.

So there is no automatic fallback to CPU, OpenCL or Reference. CUDA that cannot be initialised is
an error *before* dynamics, and `--cpu` is the only public way to ask for a CPU run.

This module exists because the policy was previously decided twice. An ordinary stage refused to
fall back; a replica ladder, given no explicit platform, chose `"CUDA" if available else "CPU"`.
Two implementations of one decision, and the one that got fixed was not the one that ran. Stages,
REMD and AIS all resolve through `resolve_platform_request` now, so they cannot drift again.

What this does NOT cover: `build-top`. Assigning parameters with OpenFF and AmberTools is CPU work
that does not create an OpenMM Context, and calling it GPU-accelerated would be false. The policy
applies to Contexts — minimisation, energy evaluation, integration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "PlatformRequest",
    "PlatformResolution",
    "PlatformUnavailable",
    "resolve_platform_request",
    "acceleration_record",
]


class PlatformUnavailable(SystemExit):
    """CUDA was required and could not be initialised. A `SystemExit` so it stops a run cleanly."""


@dataclass(frozen=True)
class PlatformRequest:
    """What the user asked for, before anything is checked against the machine.

    Deliberately two fields rather than one string: `explicit_cpu` records that a person *chose*
    the CPU, and the run record must be able to say so. A CPU result that cannot be distinguished
    from an unnoticed CUDA fallback is the failure this whole module exists to prevent.
    """

    name: str = "CUDA"
    explicit_cpu: bool = False
    device_index: Optional[int] = None
    precision: str = "mixed"

    @classmethod
    def default(cls) -> "PlatformRequest":
        """CUDA. Not "CUDA if it happens to be there"."""
        return cls()

    @classmethod
    def from_flags(cls, *, cpu: bool = False, platform: str | None = None,
                   device_index: int | None = None,
                   precision: str | None = None) -> "PlatformRequest":
        """Build a request from command-line flags, refusing contradictions.

        A contradiction is refused rather than resolved by precedence: `--cpu --platform CUDA` is
        not a preference to rank, it is two incompatible instructions, and picking one silently is
        how a run ends up on hardware nobody asked for.
        """
        if cpu and platform and platform.upper() != "CPU":
            raise PlatformUnavailable(
                f"--cpu and --platform {platform} are contradictory. Ask for one: --cpu for an "
                f"explicit CPU run, or --platform {platform} for that platform. This is refused "
                f"rather than resolved by precedence, because either choice would silently "
                f"discard half of what was asked for.")
        if cpu:
            return cls(name="CPU", explicit_cpu=True,
                       precision=precision or "mixed")
        if platform:
            return cls(name=platform.upper() if platform.upper() in ("CUDA", "CPU") else platform,
                       explicit_cpu=platform.upper() == "CPU",
                       device_index=device_index, precision=precision or "mixed")
        return cls(device_index=device_index, precision=precision or "mixed")


@dataclass(frozen=True)
class PlatformResolution:
    """What the machine actually gave, and everything a record needs to say about it."""

    platform: Any                      # openmm.Platform
    properties: dict[str, str]
    request: PlatformRequest
    device_index: Optional[int]
    visible_devices: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.platform.getName()


def _cuda_device_names() -> tuple[str, ...]:
    """The CUDA devices ON THE HOST, by name, for the record.

    Not "the devices this process can see": `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so this
    is the host's inventory. The record carries `CUDA_VISIBLE_DEVICES` alongside it, and those two
    together say what was available and what this process was restricted to.

    Best-effort: a run must not fail because the diagnostic that describes it could not be
    produced. Absent names are reported as absent rather than guessed.
    """
    try:
        import subprocess

        done = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
        if done.returncode == 0:
            return tuple(line.strip() for line in done.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return ()


def resolve_platform_request(request: PlatformRequest | None = None, *,
                             device_index: int | None = None) -> PlatformResolution:
    """Turn a request into a real OpenMM Platform, or refuse before any dynamics happen.

    `device_index` overrides the request's own, so an MPI rank can be placed on its own device
    without rebuilding the request it shares with every other rank.
    """
    from openmm import Platform

    request = request or PlatformRequest.default()
    available = {Platform.getPlatform(i).getName()
                 for i in range(Platform.getNumPlatforms())}

    if request.name not in available:
        if request.name == "CUDA":
            raise PlatformUnavailable(
                f"CUDA is required and this OpenMM build does not provide it (it has: "
                f"{', '.join(sorted(available))}).\n"
                f"  There is no automatic fall back to CPU, OpenCL or Reference: a run that "
                f"quietly moved to the CPU would finish, write a trajectory and report success, "
                f"two orders of magnitude later.\n"
                f"  For an intentional CPU run, pass --cpu.")
        raise PlatformUnavailable(
            f"OpenMM offers no {request.name!r} platform; available: {sorted(available)}")

    platform = Platform.getPlatformByName(request.name)
    properties: dict[str, str] = {}
    index = device_index if device_index is not None else request.device_index
    if request.name in ("CUDA", "OpenCL"):
        properties["Precision"] = request.precision
        if index is not None:
            properties["DeviceIndex"] = str(index)

    if request.name == "CUDA":
        # Listing the platform is not the same as having a usable device: conda-forge ships the
        # plugin unconditionally, so a machine with no driver still reports CUDA. The only honest
        # check is to build a Context, and doing it HERE means the failure lands before a
        # minimisation, an output file, or an hour of somebody's time.
        _prove_cuda_initialises(platform, properties)

    return PlatformResolution(platform=platform, properties=properties, request=request,
                              device_index=index, visible_devices=_cuda_device_names())


def _prove_cuda_initialises(platform, properties: dict[str, str]) -> None:
    """Open and discard a one-particle Context. Cheap, and it either works or it does not."""
    from openmm import Context, System, VerletIntegrator, unit

    system = System()
    system.addParticle(1.0 * unit.amu)
    integrator = VerletIntegrator(0.001 * unit.picosecond)
    try:
        context = Context(system, integrator, platform, properties)
    except Exception as failure:                                   # noqa: BLE001 - reported below
        device = properties.get("DeviceIndex", "the default device")
        raise PlatformUnavailable(
            f"CUDA is required but a Context could not be created on {device}: "
            f"{type(failure).__name__}: {failure}\n"
            f"  Check that a GPU is visible to this process (nvidia-smi, CUDA_VISIBLE_DEVICES) "
            f"and that the driver matches the CUDA runtime OpenMM was built against.\n"
            f"  Nothing has been integrated. For an intentional CPU run, pass --cpu.") from None
    del context


def acceleration_record(resolution: PlatformResolution, *,
                        mpi_rank: int | None = None,
                        mpi_size: int | None = None,
                        local_rank: int | None = None) -> dict[str, Any]:
    """Everything a run record must say about how it was accelerated.

    A CPU result must never be mistakable for an unnoticed CUDA fallback, so the record carries
    the REQUEST as well as the outcome: `requested_policy` is `default-cuda` or `explicit-cpu`,
    and they are different facts about the same run.
    """
    import openmm

    request = resolution.request
    record: dict[str, Any] = {
        "requested_policy": "explicit-cpu" if request.explicit_cpu else "default-cuda",
        "requested_platform": request.name,
        "explicit_cpu": bool(request.explicit_cpu),
        "resolved_platform": resolution.name,
        "cuda_device_index": resolution.device_index,
        "cuda_precision": resolution.properties.get("Precision"),
        "gpus_on_host": list(resolution.visible_devices),
        "openmm_version": openmm.version.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        from openmm import Platform

        plugin = Platform.getPluginLoadFailures()
        record["openmm_plugin_load_failures"] = list(plugin) if plugin else []
    except Exception:                                              # noqa: BLE001 - diagnostics
        record["openmm_plugin_load_failures"] = None
    if mpi_size is not None:
        record["mpi"] = {"rank": mpi_rank, "size": mpi_size, "local_rank": local_rank}
    return record
