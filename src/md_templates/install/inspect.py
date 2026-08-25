"""What this machine is, written once into `$MD_STACK/machine.yaml`.

One file, human-readable, containing only what a person or a later command actually uses: how many
cores are available, whether there is a usable GPU and which CUDA it exposes, where the stack
lives, and what has been installed into it.

No fingerprint, no registry, no second manifest. A field that cannot be determined is written as
null rather than omitted, so "not detected" is distinguishable from "never asked".
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml

SCHEMA_VERSION = 1
SUBDIRECTORIES = ("envs", "packages", "logs")


def _run(args: list[str], timeout: int = 15) -> Optional[str]:
    exe = shutil.which(args[0])
    if exe is None:
        return None
    try:
        result = subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _cpu_model() -> Optional[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _physical_cores() -> Optional[int]:
    """Physical cores, which is what decides how many simulations fit -- not hyperthreads."""
    text = _run(["lscpu", "-p=Core,Socket"])
    if text:
        pairs = {line for line in text.splitlines() if line and not line.startswith("#")}
        if pairs:
            return len(pairs)
    return None


def _memory_gb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return None


def gpu_information() -> dict[str, Any]:
    """NVIDIA devices, by UUID rather than by index.

    An index means different things under different `CUDA_DEVICE_ORDER` settings; a UUID names one
    card. `CUDA_VISIBLE_DEVICES` is honoured because a device this process cannot see is not a
    device it can use.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {"available": False, "driver_version": None, "cuda_version": None, "devices": []}

    header = _run(["nvidia-smi"]) or ""
    driver = None
    cuda = None
    match = re.search(r"Driver Version:\s*([\d.]+)", header)
    if match:
        driver = match.group(1)
    match = re.search(r"CUDA Version:\s*([\d.]+)", header)
    if match:
        cuda = match.group(1)

    text = _run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                 "--format=csv,noheader,nounits"]) or ""
    devices = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            memory_gb = round(float(parts[3]) / 1024, 1)
        except ValueError:
            memory_gb = None
        devices.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                        "memory_gb": memory_gb})

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        wanted = {t.strip() for t in visible.split(",") if t.strip()}
        devices = [d for d in devices
                   if str(d["index"]) in wanted or d["uuid"] in wanted]

    return {"available": bool(devices), "driver_version": driver, "cuda_version": cuda,
            "devices": devices}


def machine_document(stack: Path) -> dict[str, Any]:
    stack = Path(stack).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "machine": {
            "hostname": platform.node() or None,
            "operating_system": f"{platform.system()} {platform.release()}".strip() or None,
            "architecture": platform.machine() or None,
            "cpu_model": _cpu_model(),
            "physical_cpu_cores": _physical_cores(),
            "logical_cpu_cores": os.cpu_count(),
            "memory_gb": _memory_gb(),
        },
        "gpu": gpu_information(),
        "paths": {
            "md_stack": str(stack),
            "environments": str(stack / "envs"),
            "packages": str(stack / "packages"),
            "logs": str(stack / "logs"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
        },
        "installed": {"openmm": None},
    }


def initialise(stack: Path) -> Path:
    """Create the directory layout and write `machine.yaml`.

    An existing `machine.yaml` keeps its `installed:` section: re-running `init` re-inspects the
    hardware, and must not forget what has been installed into the stack.
    """
    stack = Path(stack).resolve()
    for name in SUBDIRECTORIES:
        (stack / name).mkdir(parents=True, exist_ok=True)

    path = stack / "machine.yaml"
    document = machine_document(stack)
    if path.is_file():
        try:
            previous = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(previous.get("installed"), dict):
                document["installed"] = previous["installed"]
        except yaml.YAMLError:
            pass
    path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
                    encoding="utf-8")
    return path


def load_machine(stack: Path) -> dict[str, Any]:
    path = Path(stack) / "machine.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run `md-template init --target-dir {stack}` first.")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_machine(stack: Path, document: dict[str, Any]) -> Path:
    path = Path(stack) / "machine.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
                    encoding="utf-8")
    return path
