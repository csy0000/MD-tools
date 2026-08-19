#!/usr/bin/env python3
"""Read-only hardware and toolchain inventory for Amber/OpenMM installation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS = (
    "gcc", "g++", "gfortran", "clang", "cmake", "make", "ninja", "git", "tar", "bzip2",
    "python3", "conda", "mamba", "micromamba", "nvidia-smi", "nvcc", "clinfo",
    "mpicc", "mpicxx", "mpifort", "mpirun", "mpiexec", "srun",
)


def run(command: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    output = (result.stdout + "\n" + result.stderr).strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": "\n".join(output.splitlines()[:12]),
    }


def version_of(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False}
    probes = ([path, "--version"], [path, "-version"], [path, "-V"])
    result: dict[str, Any] = {"available": True, "path": str(Path(path).resolve())}
    for probe in probes:
        checked = run(list(probe), timeout=8)
        if checked.get("output"):
            result["version"] = checked["output"].splitlines()[0][:300]
            result["version_probe_returncode"] = checked.get("returncode")
            break
    return result


def memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def nearest_existing(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def disk_report(path: Path) -> dict[str, Any]:
    base = nearest_existing(path)
    try:
        usage = shutil.disk_usage(base)
    except OSError as exc:
        return {"checked_path": str(base), "error": str(exc)}
    return {
        "requested_install_dir": str(path.expanduser()),
        "checked_existing_parent": str(base.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def nvidia_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    query = run([
        executable,
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not query.get("ok"):
        query = run([
            executable,
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ])
    rows = []
    for line in query.get("output", "").splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) >= 4:
            row = {
                "index": values[0], "name": values[1], "driver_version": values[2],
                "memory_mib": values[3],
            }
            if len(values) >= 5:
                row["compute_capability"] = values[4]
            rows.append(row)
    return {"available": query.get("ok", False), "gpus": rows, "diagnostic": query}


def cuda_report() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    candidates = []
    if nvcc:
        resolved = Path(nvcc).resolve()
        candidates.append(str(resolved.parent.parent))
    for path in (Path("/usr/local/cuda"), Path("/opt/cuda")):
        if path.exists():
            candidates.append(str(path.resolve()))
    ordered = list(dict.fromkeys(candidates))
    return {
        "nvcc_available": bool(nvcc),
        "nvcc_path": str(Path(nvcc).resolve()) if nvcc else None,
        "nvcc_version": run([nvcc, "--version"]) if nvcc else None,
        "candidate_toolkit_roots": ordered,
    }


def recommendations(report: dict[str, Any]) -> dict[str, str]:
    gpu = bool(report["nvidia"].get("available") and report["nvidia"].get("gpus"))
    nvcc = bool(report["cuda"].get("nvcc_available"))
    tools = report["tools"]
    mpi_compilers = all(tools[name].get("available") for name in ("mpicc", "mpifort"))
    mpi_launcher = any(tools[name].get("available") for name in ("mpirun", "mpiexec", "srun"))
    gpu_count = len(report["nvidia"].get("gpus", []))
    if gpu:
        openmm = "CUDA preferred; verify the packaged CUDA runtime against the detected driver"
    else:
        openmm = "CPU preferred; offer OpenCL only if the user explicitly requests it"
    if gpu and nvcc:
        amber = "CUDA candidate; verify Amber26 supports the detected toolkit before enabling pmemd.cuda"
    elif gpu:
        amber = "CPU build until a supported CUDA toolkit/nvcc is provided; do not replace drivers automatically"
    else:
        amber = "CPU build"
    if gpu and nvcc and mpi_compilers and mpi_launcher:
        amber_mpi = (
            "CUDA+MPI candidate for pmemd.cuda.MPI; verify Amber26/MPI compatibility and "
            f"the intended rank-to-GPU layout against {gpu_count} visible GPU(s)"
        )
    elif gpu and nvcc:
        amber_mpi = (
            "pmemd.cuda.MPI is not ready: provide compatible MPI compiler wrappers and a launcher"
        )
    else:
        amber_mpi = "pmemd.cuda.MPI is not ready until Amber CUDA prerequisites are satisfied"
    return {
        "openmm_backend": openmm,
        "amber_backend": amber,
        "amber_cuda_mpi_backend": amber_mpi,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-dir", required=True,
                        help="Candidate installation prefix; it is inspected but not created")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    install_dir = Path(args.install_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu_logical_count": os.cpu_count(),
            "memory_bytes": memory_bytes(),
        },
        "disk": disk_report(install_dir),
        "tools": {name: version_of(name) for name in TOOLS},
        "nvidia": nvidia_report(),
        "cuda": cuda_report(),
    }
    report["recommendation"] = recommendations(report)

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
