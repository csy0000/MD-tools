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
import re
import shutil
import subprocess
import sys
from typing import Optional, Sequence

#: `GPU 0: NVIDIA RTX A5000 (UUID: GPU-7a14ba65-...)`
_GPU_LINE = re.compile(r"^GPU (?P<index>\d+): (?P<name>.+?) \(UUID: (?P<uuid>GPU-[0-9a-f-]+)\)$")
#: `  MIG 1g.5gb Device 0: (UUID: MIG-...)`
_MIG_LINE = re.compile(r"^MIG (?P<name>\S+) Device (?P<dev>\d+): \(UUID: (?P<uuid>MIG-[0-9A-Za-z-]+)\)$")

__all__ = [
    "GpuDiscoveryError",
    "NoUsableGpuError",
    "busy_physical_devices",
    "describe_selection",
    "discover_devices",
    "ensure_pci_bus_id_order",
    "mig_devices",
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
    text = _run_smi(["--query-gpu=index,uuid,name,memory.total,pci.bus_id",
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
                        "memory_total_mib": parts[3],
                        # PCI bus ID is the only identifier that is stable across CUDA_DEVICE_ORDER
                        # settings AND meaningful to a human reading `lspci`. Older drivers may not
                        # report it, so it is optional rather than assumed.
                        "pci_bus_id": parts[4] if len(parts) > 4 else None})
    return devices


def mig_devices() -> list[dict]:
    """MIG instances, parsed from `nvidia-smi -L`, keyed to their parent GPU.

    MIG UUIDs may appear in `CUDA_VISIBLE_DEVICES` (`MIG-<uuid>`), and they are NOT in the
    `--query-gpu` output, so a MIG token would otherwise look like a device the driver does not
    report and be dropped. Dropping it silently would run the ladder on the wrong devices.
    """
    text = _run_smi(["-L"])
    if not text:
        return []
    out: list[dict] = []
    parent_uuid = None
    parent_index = None
    for line in text.splitlines():
        stripped = line.strip()
        gpu = _GPU_LINE.match(stripped)
        if gpu:
            parent_index = int(gpu.group("index"))
            parent_uuid = gpu.group("uuid")
            continue
        mig = _MIG_LINE.match(stripped)
        if mig and parent_uuid is not None:
            out.append({"uuid": mig.group("uuid"), "name": mig.group("name"),
                        "mig": True, "parent_uuid": parent_uuid,
                        "physical_index": parent_index,
                        "mig_device_index": int(mig.group("dev"))})
    return out


def ensure_pci_bus_id_order() -> dict:
    """Make CUDA enumerate devices in the same order `nvidia-smi` prints them.

    This is the correctness problem the release review named. `nvidia-smi` numbers devices by PCI
    bus order, but CUDA's DEFAULT is `CUDA_DEVICE_ORDER=FASTEST_FIRST`, which sorts by capability.
    On a machine with mixed cards the two orders genuinely differ, so "GPU 5" in nvidia-smi and
    logical index 5 in OpenMM can be different cards -- and every busy-device exclusion computed
    from the first would then be applied to the second.

    Setting the variable only has effect BEFORE the CUDA driver initialises, so this reports what
    it actually managed to do rather than claiming success:

    * `already_pci`    -- the caller had set it correctly; nothing to do
    * `set`            -- set here, in time
    * `conflicting`    -- the caller explicitly asked for a different order; left alone, because
                          overriding an explicit choice is worse than reporting it
    """
    current = os.environ.get("CUDA_DEVICE_ORDER")
    if current == "PCI_BUS_ID":
        return {"action": "already_pci", "cuda_device_order": current, "trustworthy": True}
    if current:
        return {"action": "conflicting", "cuda_device_order": current, "trustworthy": False,
                "note": (f"CUDA_DEVICE_ORDER={current!r} was set by the caller. nvidia-smi indices "
                         "and CUDA ordinals may disagree; identity is matched by UUID, which is "
                         "unaffected, but physical_index is reported as nvidia-smi sees it.")}
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return {"action": "set", "cuda_device_order": "PCI_BUS_ID",
            "trustworthy": not _cuda_already_initialised(),
            "note": ("set here; it takes effect only if the CUDA driver had not already "
                     "initialised in this process")}


def _cuda_already_initialised() -> bool:
    """Whether a CUDA context already exists, in which case the order variable is too late.

    Checked without importing anything heavy: if OpenMM has not been imported, CUDA cannot have
    been initialised by it.
    """
    module = sys.modules.get("openmm")
    if module is None:
        return False
    try:
        return any(module.Platform.getPlatform(i).getName() == "CUDA"
                   and module.Platform.getPlatform(i).getPropertyDefaultValue("DeviceIndex") != ""
                   for i in range(module.Platform.getNumPlatforms()))
    except Exception:                              # noqa: BLE001
        return False


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
    # MIG instances are absent from --query-gpu, so a `MIG-...` token would look like a device the
    # driver does not report and be dropped -- silently running the ladder on the wrong devices.
    # They are only looked up if a token needs them, because `nvidia-smi -L` is a second call.
    mig_by_uuid: dict[str, dict] = {}
    if raw_mentions_mig := ("MIG-" in (os.environ.get("CUDA_VISIBLE_DEVICES") or "")):
        mig_by_uuid = {d["uuid"]: d for d in mig_devices()}

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
        found = by_uuid.get(token) or mig_by_uuid.get(token)
        if found is None and token.startswith("MIG-"):
            # A MIG token the -L listing did not resolve: keep it rather than drop it, because a
            # dropped entry shifts every later logical index and moves work to another card.
            found = {"uuid": token, "name": None, "mig": True, "physical_index": None,
                     "pci_bus_id": None, "memory_total_mib": None, "unresolved": True}
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
    # Do this BEFORE enumerating, so the ordinals reported here are the ordinals OpenMM will use.
    order = ensure_pci_bus_id_order()
    visible = visible_devices()
    busy = busy_physical_devices() if exclude_busy else set()
    # Busy-matching is by UUID, and for a MIG instance by its PARENT's UUID: compute apps are
    # reported against the parent, so comparing a MIG UUID to them would never match and a busy
    # card would be handed out as free.
    def _busy(d: dict) -> bool:
        return d.get("uuid") in busy or (d.get("parent_uuid") in busy if d.get("mig") else False)

    usable = [d for d in visible if not _busy(d)]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "device_order_action": order,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "visible": visible,
        "busy_uuids": sorted(busy),
        "excluded_busy": [d for d in visible if _busy(d)],
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
        "pci_bus_id_of_logical": {int(d["logical_index"]): d.get("pci_bus_id") for d in chosen},
        # The full identity per selected device, which is what gpu_selection.json must carry: an
        # ordinal alone is meaningless once CUDA_VISIBLE_DEVICES or CUDA_DEVICE_ORDER differ.
        "device_identity": [
            {"logical_index": int(d["logical_index"]),
             "physical_index": d.get("physical_index"),
             "uuid": d.get("uuid"),
             "pci_bus_id": d.get("pci_bus_id"),
             "name": d.get("name"),
             "mig": bool(d.get("mig")),
             "parent_uuid": d.get("parent_uuid")}
            for d in chosen],
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
        pci = record.get("pci_bus_id_of_logical", {}).get(logical)
        lines.append(f"[gpu]   logical {logical} -> physical {physical}  pci {pci}  {uuid}  "
                     f"replicas {replicas}")
    if record["shared_devices"]:
        lines.append(f"[gpu]   shared devices {record['shared_devices']}: their replicas propagate "
                     f"sequentially on that device")
    return "\n".join(lines)
