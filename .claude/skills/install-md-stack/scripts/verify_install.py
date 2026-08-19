#!/usr/bin/env python3
"""Verify an Amber26/AmberTools26 and OpenMM 8.5.2 installation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AMBERTOOLS_PROGRAMS = ("tleap", "sander", "cpptraj")
PMEMD_PROGRAMS = ("pmemd", "pmemd.cuda", "pmemd.cuda.MPI")


def run(command: list[str], *, env: dict[str, str] | None = None,
        timeout: int = 300) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "command": command,
                "error": f"{type(exc).__name__}: {exc}"}
    output = (result.stdout + "\n" + result.stderr).strip()
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "output": "\n".join(output.splitlines()[:80]),
    }


def openmm_probe(python: Path) -> dict[str, Any]:
    code = r'''
import json
import openmm
platforms = []
for index in range(openmm.Platform.getNumPlatforms()):
    platform = openmm.Platform.getPlatform(index)
    platforms.append({"name": platform.getName(), "speed": platform.getSpeed()})
print(json.dumps({"version": openmm.__version__, "platforms": platforms}))
'''
    result = run([str(python), "-c", code], timeout=60)
    if result.get("ok"):
        try:
            result["data"] = json.loads(result["output"].splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            result["ok"] = False
            result["error"] = f"could not parse OpenMM probe: {exc}"
    return result


def amber_probe(amberhome: Path, name: str) -> dict[str, Any]:
    executable = amberhome / "bin" / name
    result: dict[str, Any] = {
        "path": str(executable),
        "exists": executable.is_file(),
        "executable": os.access(executable, os.X_OK),
    }
    if result["exists"] and result["executable"]:
        probe = run([str(executable), "-h"], timeout=30)
        # Several Amber programs return nonzero for help; presence plus readable output is useful.
        result["help_returncode"] = probe.get("returncode")
        result["help_output"] = probe.get("output", "")[:4000]
    return result


def mpi_launcher_probe(requested: str | None) -> dict[str, Any]:
    candidates = [requested] if requested else ["mpirun", "mpiexec", "srun"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser() if "/" in candidate else None
        executable = (
            str(path.resolve())
            if path and path.is_file() and os.access(path, os.X_OK)
            else shutil.which(candidate)
        )
        if executable:
            version = run([executable, "--version"], timeout=30)
            return {
                "available": True,
                "requested": requested,
                "path": str(Path(executable).resolve()),
                "version": version,
            }
    return {"available": False, "requested": requested, "candidates": candidates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amberhome", required=True)
    parser.add_argument(
        "--pmemdhome",
        help="Licensed PMEMD install prefix; defaults to AMBERHOME for a combined installation",
    )
    parser.add_argument("--python", required=True,
                        help="Python executable from the OpenMM 8.5.2 environment")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--require-cuda-mpi", action="store_true",
        help="Require pmemd.cuda.MPI and an MPI/scheduler launcher; also implies --require-cuda",
    )
    parser.add_argument(
        "--mpi-launcher",
        help="MPI or scheduler launcher name/path; auto-detect mpirun, mpiexec, then srun",
    )
    parser.add_argument("--skip-openmm-self-test", action="store_true")
    parser.add_argument("--output", help="Optional JSON manifest path")
    args = parser.parse_args()

    amberhome = Path(args.amberhome).expanduser().resolve()
    pmemdhome = (
        Path(args.pmemdhome).expanduser().resolve() if args.pmemdhome else amberhome
    )
    python = Path(args.python).expanduser().resolve()
    programs = {name: amber_probe(amberhome, name) for name in AMBERTOOLS_PROGRAMS}
    programs.update({name: amber_probe(pmemdhome, name) for name in PMEMD_PROGRAMS})
    mpi_launcher = mpi_launcher_probe(args.mpi_launcher)
    openmm = openmm_probe(python)
    platform_names = {item["name"] for item in (openmm.get("data") or {}).get("platforms", [])}

    errors = []
    if not amberhome.is_dir():
        errors.append(f"AMBERHOME is not a directory: {amberhome}")
    if not pmemdhome.is_dir():
        errors.append(f"PMEMDHOME is not a directory: {pmemdhome}")
    for name in AMBERTOOLS_PROGRAMS:
        if not (programs[name]["exists"] and programs[name]["executable"]):
            errors.append(f"missing executable: {amberhome / 'bin' / name}")
    if not (programs["pmemd"]["exists"] and programs["pmemd"]["executable"]):
        errors.append(f"missing executable: {pmemdhome / 'bin' / 'pmemd'}")
    if not openmm.get("ok"):
        errors.append("OpenMM import/platform probe failed")
    elif (openmm.get("data") or {}).get("version") != "8.5.2":
        errors.append(f"OpenMM version is {(openmm.get('data') or {}).get('version')!r}, expected '8.5.2'")
    require_cuda = args.require_cuda or args.require_cuda_mpi
    if require_cuda:
        if not (programs["pmemd.cuda"]["exists"] and programs["pmemd.cuda"]["executable"]):
            errors.append("CUDA selected but pmemd.cuda is missing")
        if "CUDA" not in platform_names:
            errors.append("CUDA selected but OpenMM does not expose the CUDA platform")
    if args.require_cuda_mpi:
        if not (
            programs["pmemd.cuda.MPI"]["exists"]
            and programs["pmemd.cuda.MPI"]["executable"]
        ):
            errors.append("CUDA+MPI selected but pmemd.cuda.MPI is missing")
        if not mpi_launcher["available"]:
            errors.append("CUDA+MPI selected but no MPI or scheduler launcher was found")

    self_test = None
    if openmm.get("ok") and not args.skip_openmm_self_test:
        self_test = run([str(python), "-m", "openmm.testInstallation"], timeout=900)
        if not self_test.get("ok"):
            errors.append("python -m openmm.testInstallation failed")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amberhome": str(amberhome),
        "pmemdhome": str(pmemdhome),
        "openmm_python": str(python),
        "require_cuda": require_cuda,
        "require_cuda_mpi": args.require_cuda_mpi,
        "amber_programs": programs,
        "mpi_launcher": mpi_launcher,
        "openmm": openmm,
        "openmm_self_test": self_test,
        "ok": not errors,
        "errors": errors,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
