"""Which GPUs a run may use, and which replica lands on which one.

REST2 holds one OpenMM `Context` per replica, so the useful number of devices is

    N_use = min(N_replica, N_available)

and never `max`. More devices than replicas leaves the surplus idle; more replicas than devices is
legal and means some devices carry several replicas. Both directions are handled here rather than
being left to the caller, because the wrong one is silent: a ladder that quietly uses one GPU looks
exactly like a ladder that uses six, only slower.

Two index spaces exist and confusing them puts work on the wrong card:

* **physical** -- what `nvidia-smi` prints, and what a person means by "GPU 5";
* **logical**  -- what OpenMM's `DeviceIndex` property takes, which counts only the devices
  `CUDA_VISIBLE_DEVICES` exposes.

With `CUDA_VISIBLE_DEVICES=3,5,7` the machine's GPU 5 is logical index 1. Passing 5 there would
select physical GPU 7. Every selection here therefore carries both, and the mapping is recorded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Sequence

__all__ = [
    "GpuDiscoveryError",
    "NoUsableGpuError",
    "busy_physical_devices",
    "describe_selection",
    "discover_devices",
    "select_devices",
    "visible_devices",
]

#: How long to wait for `nvidia-smi`. It normally answers in milliseconds; a hang here would block
#: a run before it starts, which is worse than falling back to "assume nothing is busy".
_SMI_TIMEOUT_S = 10


class GpuDiscoveryError(RuntimeError):
    """Device discovery could not be completed."""


class NoUsableGpuError(GpuDiscoveryError):
    """A CUDA run was requested and no usable device exists.

    Raised BEFORE any replica `System` is constructed. Discovering this after building six Systems
    wastes the expensive part of setup and leaves half-initialised Contexts behind.
    """


def _run_smi(args: Sequence[str]) -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=_SMI_TIMEOUT_S, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout if out.returncode == 0 else None


def _all_physical_devices() -> list[dict]:
    """Every device the driver reports, ignoring `CUDA_VISIBLE_DEVICES`."""
    text = _run_smi(["--query-gpu=index,uuid,name,memory.total",
                     "--format=csv,noheader,nounits"])
    if not text:
        return []
    devices = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        devices.append({"physical_index": index, "uuid": parts[1], "name": parts[2],
                        "memory_total_mib": parts[3]})
    return devices


def busy_physical_devices() -> set[str]:
    """UUIDs of devices with a compute process on them.

    Matched by UUID rather than index: an index means different things under different
    `CUDA_DEVICE_ORDER` settings, whereas a UUID identifies one card.
    """
    text = _run_smi(["--query-compute-apps=gpu_uuid", "--format=csv,noheader"])
    if not text:
        return set()
    return {line.strip() for line in text.strip().splitlines() if line.strip()}


def visible_devices() -> list[dict]:
    """The devices this process may use, in logical order, honouring `CUDA_VISIBLE_DEVICES`.

    Each entry carries `logical_index` (what OpenMM's `DeviceIndex` takes) and `physical_index`
    (what `nvidia-smi` prints). An unset variable means every device is visible and the two indices
    coincide.

    A `CUDA_VISIBLE_DEVICES` entry naming a device the driver does not report is dropped rather than
    guessed at -- CUDA itself stops at the first invalid entry, and quietly inventing a device would
    place work somewhere nobody asked for.
    """
    physical = _all_physical_devices()
    by_index = {d["physical_index"]: d for d in physical}
    by_uuid = {d["uuid"]: d for d in physical}

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return [dict(d, logical_index=i, visible_via="all devices visible")
                for i, d in enumerate(physical)]
    if raw.strip() == "":
        return []                      # an empty variable hides everything; that is a valid state

    out: list[dict] = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            continue
        found = by_uuid.get(token)
        if found is None:
            try:
                found = by_index.get(int(token))
            except ValueError:
                found = None
        if found is None:
            continue                   # names a device the driver does not report
        out.append(dict(found, logical_index=len(out), visible_via=f"CUDA_VISIBLE_DEVICES={raw}"))
    return out


def discover_devices(*, exclude_busy: bool = True) -> dict:
    """The usable device set, with everything needed to explain the choice afterwards."""
    visible = visible_devices()
    busy = busy_physical_devices() if exclude_busy else set()
    usable = [d for d in visible if d["uuid"] not in busy]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "visible": visible,
        "busy_uuids": sorted(busy),
        "excluded_busy": [d for d in visible if d["uuid"] in busy],
        "usable": usable,
        "n_visible": len(visible),
        "n_usable": len(usable),
    }


def select_devices(n_replicas: int, *, platform: str = "CUDA",
                   explicit: Optional[Sequence[int]] = None,
                   exclude_busy: bool = True) -> dict:
    """Choose the devices for `n_replicas`, automatically unless `explicit` is given.

    Returns a record carrying the discovery, the selection, the physical/logical mapping and the
    replica-to-device assignment -- everything the run manifest has to state so that a reader can
    tell which card did which work.

    `explicit` is used EXACTLY as given after validation: a device the operator named and a device
    this module chose are different claims, and silently substituting one for the other would make
    the manifest a record of something that did not happen. Busy devices are still reported when
    named explicitly, but they are not removed.
    """
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be positive; got {n_replicas}")

    discovery = discover_devices(exclude_busy=exclude_busy)
    if str(platform).upper() not in ("CUDA", "OPENCL"):
        return {"platform": platform, "devices": None, "discovery": discovery,
                "replica_devices": None,
                "note": f"{platform} has no device selection; every replica shares the CPU build"}

    if explicit is not None:
        wanted = [int(d) for d in explicit]
        if not wanted:
            raise ValueError("an explicit device list may not be empty")
        by_logical = {d["logical_index"]: d for d in discovery["visible"]}
        # An explicit index is a LOGICAL index -- the same space OpenMM's DeviceIndex uses -- so
        # that what the operator writes is what OpenMM receives.
        hidden = [d for d in wanted if d not in by_logical]
        if hidden and discovery["visible"]:
            raise NoUsableGpuError(
                f"devices {hidden} were requested but are not visible to this process. "
                f"CUDA_VISIBLE_DEVICES={discovery['cuda_visible_devices']!r} exposes logical "
                f"indices {sorted(by_logical)}. A hidden device cannot be selected, and choosing a "
                f"different one instead would put the work somewhere nobody asked for."
            )
        chosen = [by_logical.get(d, {"logical_index": d, "physical_index": d,
                                     "uuid": None, "name": None}) for d in wanted]
        source = "explicit"
    else:
        usable = discovery["usable"]
        if not usable:
            raise NoUsableGpuError(
                f"a {platform} run was requested but no usable device was found. "
                f"visible={discovery['n_visible']}, busy={len(discovery['busy_uuids'])}, "
                f"CUDA_VISIBLE_DEVICES={discovery['cuda_visible_devices']!r}, "
                f"nvidia-smi available={discovery['nvidia_smi_available']}. "
                f"Refusing before any replica System is built -- discovering this afterwards wastes "
                f"the expensive part of setup."
            )
        # min, never max: one Context per replica, so surplus devices cannot be used, and surplus
        # replicas share.
        n_use = min(n_replicas, len(usable))
        chosen = usable[:n_use]
        source = "automatic"

    from .tau import map_replicas_to_devices

    logical = [int(d["logical_index"]) for d in chosen]
    replica_devices = map_replicas_to_devices(n_replicas, logical)
    per_device: dict[int, list[int]] = {}
    for replica, dev in enumerate(replica_devices):
        per_device.setdefault(dev, []).append(replica)

    return {
        "platform": platform,
        "selection_source": source,
        "n_replicas": n_replicas,
        "n_devices_selected": len(chosen),
        "rule": "min(n_replicas, n_available)" if source == "automatic" else "operator-specified",
        "devices": logical,
        "device_details": chosen,
        "physical_of_logical": {int(d["logical_index"]): d.get("physical_index") for d in chosen},
        "uuid_of_logical": {int(d["logical_index"]): d.get("uuid") for d in chosen},
        "replica_devices": replica_devices,
        "replicas_per_device": {int(k): v for k, v in sorted(per_device.items())},
        "shared_devices": sorted(k for k, v in per_device.items() if len(v) > 1),
        "discovery": discovery,
    }


def describe_selection(record: dict) -> str:
    """One log-friendly line per device, so the mapping is visible without reading JSON."""
    if not record.get("devices"):
        return f"[gpu] {record.get('note') or 'no device selection'}"
    lines = [
        f"[gpu] {record['selection_source']}: {record['n_devices_selected']} device(s) for "
        f"{record['n_replicas']} replica(s)  rule={record['rule']}"
    ]
    for logical in record["devices"]:
        physical = record["physical_of_logical"].get(logical)
        uuid = record["uuid_of_logical"].get(logical)
        replicas = record["replicas_per_device"].get(logical, [])
        lines.append(f"[gpu]   logical {logical} -> physical {physical}  {uuid}  "
                     f"replicas {replicas}")
    if record["shared_devices"]:
        lines.append(f"[gpu]   shared devices {record['shared_devices']}: their replicas propagate "
                     f"sequentially on that device")
    return "\n".join(lines)
