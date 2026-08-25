"""Install OpenMM into `$MD_STACK/envs/openmm-<version>/`.

Uses whichever of micromamba, mamba or conda is already present. It does not install a package
manager, a CUDA driver, compilers, MPI, or another MD engine: those are the machine's business, and
a script that installs a driver behind someone's back is worse than one that stops and says what is
missing.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .inspect import load_machine, save_machine

#: In preference order. micromamba first: it is a single binary, solves fastest, and needs no base
#: installation of its own.
PACKAGE_MANAGERS = ("micromamba", "mamba", "conda")


class InstallError(RuntimeError):
    """Installation cannot proceed, with an actionable reason."""


def find_package_manager() -> tuple[str, str]:
    for name in PACKAGE_MANAGERS:
        found = shutil.which(name)
        if found:
            return name, found
    raise InstallError(
        "no conda-compatible package manager found. Tried: "
        + ", ".join(PACKAGE_MANAGERS) + ".\n"
        "Install one, for example:\n"
        "    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba\n"
        "This command deliberately does not install one for you.")


def _log_path(stack: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = Path(stack) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / f"install-openmm-{stamp}.log"


def _cuda_wanted(machine: dict[str, Any]) -> bool:
    return bool((machine.get("gpu") or {}).get("available"))


def install_openmm(stack: Path, version: str = "8.6.0", *,
                   dry_run: bool = False) -> dict[str, Any]:
    """Create the environment, install OpenMM, verify its platforms, and record all of it."""
    stack = Path(stack).resolve()
    machine = load_machine(stack)
    manager, executable = find_package_manager()

    prefix = stack / "envs" / f"openmm-{version}"
    log = _log_path(stack)
    cuda = _cuda_wanted(machine)

    command = [executable, "create", "--yes", "--prefix", str(prefix),
               "-c", "conda-forge", f"openmm={version}", "python=3.12"]
    if cuda:
        # conda-forge builds OpenMM against a CUDA version; letting the solver pick the build that
        # matches the driver is more reliable than pinning a toolkit here.
        command.append("cuda-version>=11.8")

    lines = [f"# {datetime.now(timezone.utc).isoformat()}",
             f"# package manager: {manager} ({executable})",
             f"# cuda requested: {cuda}",
             "$ " + " ".join(command), ""]

    if dry_run:
        log.write_text("\n".join(lines + ["# dry run: nothing was executed"]), encoding="utf-8")
        return {"prefix": str(prefix), "log": str(log), "dry_run": True}

    result = subprocess.run(command, capture_output=True, text=True)
    lines += [result.stdout, result.stderr]
    log.write_text("\n".join(lines), encoding="utf-8")
    if result.returncode != 0:
        raise InstallError(
            f"{manager} failed to create {prefix}. The full output is in {log}.\n"
            + (result.stderr or result.stdout)[-1500:])

    check = verify_environment(prefix)
    lines += ["", "# platform check", repr(check)]
    log.write_text("\n".join(lines), encoding="utf-8")

    machine.setdefault("installed", {})["openmm"] = {
        "version": check.get("openmm_version") or version,
        "environment": str(prefix),
        "python_version": check.get("python_version"),
        "platforms": check.get("platforms"),
        "cuda_available": check.get("cuda_available"),
        "cuda_check": check.get("cuda_check"),
        "installed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "log": str(log),
    }
    save_machine(stack, machine)
    return {"prefix": str(prefix), "log": str(log), **check}


PLATFORM_PROBE = r"""
import json, sys
import openmm
platforms = [openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())]
out = {"openmm_version": openmm.version.short_version,
       "python_version": sys.version.split()[0],
       "platforms": platforms,
       "cuda_available": "CUDA" in platforms,
       "plugin_load_failures": list(openmm.Platform.getPluginLoadFailures())}
if out["cuda_available"]:
    # Build a two-particle System and take a step: a CUDA platform that is listed but cannot
    # allocate a context is worse than one that is absent, because it fails at run time.
    try:
        system = openmm.System()
        system.addParticle(1.0); system.addParticle(1.0)
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
        context.setPositions([(0, 0, 0), (0, 0, 0.1)])
        context.getState(getEnergy=True)
        integrator.step(1)
        out["cuda_check"] = "ok"
    except Exception as exc:
        out["cuda_check"] = f"{type(exc).__name__}: {exc}"
else:
    out["cuda_check"] = "no CUDA platform"
print(json.dumps(out))
"""


def verify_environment(prefix: Path) -> dict[str, Any]:
    """Import OpenMM in the new environment and report what it can actually do."""
    import json

    python = Path(prefix) / "bin" / "python"
    if not python.is_file():
        raise InstallError(f"{python} does not exist; the environment was not created")
    result = subprocess.run([str(python), "-c", PLATFORM_PROBE],
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise InstallError(
            f"OpenMM could not be imported from {prefix}:\n{result.stderr[-1500:]}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise InstallError(f"unreadable platform check output:\n{result.stdout[-800:]}")
