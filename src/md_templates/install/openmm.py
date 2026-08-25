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

#: What the advertised workflows actually import, one entry per reason. Derived from the imports in
#: `src/`, not from a wish list: an environment that installs only `openmm` cannot run `sys-gen`,
#: because building a system needs OpenFF, openmmforcefields, AmberTools, ParmEd and RDKit.
CONDA_PACKAGES = (
    "python=3.12",
    "pyyaml",                 # every configuration file this package reads or writes
    "numpy",                  # box geometry in solvation.py and system.py
    "openff-toolkit",         # Molecule, GLOBAL_TOOLKIT_REGISTRY -- ligand parameterisation
    "openff-nagl-models",     # the `am1bcc_nagl` charge option in defaults.py
    "openmmforcefields",      # SMIRNOFFTemplateGenerator, the ligand template generator
    "ambertools",             # sqm/antechamber for AM1-BCC, tleap for implicit peptide topologies
    "parmed",                 # prmtop/rst7 <-> OpenMM, used by the implicit route
    "rdkit",                  # SMILES -> 3D conformer for the ligand route
)

#: Imports that must succeed, and the package that provides each. A missing one is named rather
#: than surfacing later as an ImportError halfway through a build.
REQUIRED_IMPORTS = (
    ("openmm", "openmm"),
    ("yaml", "pyyaml"),
    ("numpy", "numpy"),
    ("openff.toolkit", "openff-toolkit"),
    ("openmmforcefields", "openmmforcefields"),
    ("parmed", "parmed"),
    ("rdkit", "rdkit"),
)

#: AmberTools executables the two routes shell out to.
REQUIRED_EXECUTABLES = ("sqm", "antechamber", "tleap")


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
               "-c", "conda-forge", f"openmm={version}", *CONDA_PACKAGES]
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

    try:
        check = verify_environment(prefix)
    except InstallError as error:
        lines += ["", "# validation failed", str(error)]
        log.write_text("\n".join(lines), encoding="utf-8")
        raise
    lines += ["", "# validation", repr(check)]
    log.write_text("\n".join(lines), encoding="utf-8")

    record = record_environment(stack, prefix, check, log=log, version=version)
    return {"prefix": str(prefix), "log": str(log), "record": record, **check}


def record_environment(stack: Path, prefix: Path, check: dict[str, Any], *,
                       log: Optional[Path] = None, version: str = "") -> dict[str, Any]:
    """Write the package versions and validation results into `machine.yaml`.

    Provenance, not decoration: which OpenFF and AmberTools produced a system is part of what the
    system is, and reading it back off the machine months later is the only way to know.
    """
    stack = Path(stack).resolve()
    machine = load_machine(stack)
    record = {
        "version": check.get("openmm_version") or version,
        "environment": str(prefix),
        "python_version": check.get("python_version"),
        "package_versions": check.get("versions"),
        "executables": check.get("executables"),
        "openff_toolkits": check.get("openff_toolkits"),
        "am1bcc_ready": check.get("am1bcc_ready"),
        "platforms": check.get("platforms"),
        "cuda_available": check.get("cuda_available"),
        "reference_check": check.get("reference_check"),
        "cpu_check": check.get("cpu_check"),
        "cuda_check": check.get("cuda_check"),
        "plugin_load_failures": check.get("plugin_load_failures"),
        "problems": check.get("problems"),
        "warnings": check.get("warnings"),
        "validated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if log is not None:
        record["log"] = str(log)
    machine.setdefault("installed", {})["openmm"] = record
    save_machine(stack, machine)
    return record


def validate_existing(stack: Path, prefix: Path) -> dict[str, Any]:
    """Validate an environment somebody else built, and record it the same way.

    The check is the same either way. An environment this command did not create is not less
    obliged to be able to run the commands this package advertises.
    """
    check = verify_environment(Path(prefix).resolve())
    record = record_environment(stack, Path(prefix).resolve(), check)
    return {"prefix": str(Path(prefix).resolve()), "record": record, **check}


#: Runs INSIDE the environment being validated, so it must import nothing from this package and
#: must print exactly one line of JSON. It reports rather than judges; `verify_environment` below
#: decides what counts as a failure.
ENVIRONMENT_PROBE = r"""
import importlib, json, shutil, sys

out = {"python_version": sys.version.split()[0], "python": sys.executable,
       "versions": {}, "import_errors": {}, "executables": {}}

for module in ("openmm", "yaml", "numpy", "openff.toolkit", "openff.nagl_models",
               "openmmforcefields", "parmed", "rdkit"):
    try:
        loaded = importlib.import_module(module)
        version = getattr(loaded, "__version__", None)
        if module == "openmm":
            version = loaded.version.short_version
        elif module == "rdkit":
            version = importlib.import_module("rdkit.rdBase").rdkitVersion
        out["versions"][module] = str(version) if version else "unknown"
    except Exception as exc:
        out["import_errors"][module] = f"{type(exc).__name__}: {exc}"

# AmberTools ships executables, not a Python package: AM1-BCC shells out to sqm through
# antechamber, and the implicit peptide route builds its topology with tleap.
for name in ("sqm", "antechamber", "tleap"):
    out["executables"][name] = shutil.which(name)

# The OpenFF side of the same question: a registered AmberTools wrapper is what makes
# `partial_charges='am1bcc'` resolvable at all.
try:
    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
    out["openff_toolkits"] = [type(t).__name__
                              for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits]
except Exception as exc:
    out["openff_toolkits"] = []
    out["import_errors"]["openff.toolkit.utils.toolkits"] = f"{type(exc).__name__}: {exc}"
out["am1bcc_ready"] = (any("AmberTools" in name for name in out.get("openff_toolkits", []))
                       and bool(out["executables"].get("sqm"))
                       and bool(out["executables"].get("antechamber")))

if "openmm" in out["import_errors"]:
    print(json.dumps(out))
    raise SystemExit(0)

import openmm
out["platforms"] = [openmm.Platform.getPlatform(i).getName()
                    for i in range(openmm.Platform.getNumPlatforms())]
out["cuda_available"] = "CUDA" in out["platforms"]
out["plugin_load_failures"] = list(openmm.Platform.getPluginLoadFailures())
out["openmm_version"] = openmm.version.short_version


def one_step(platform_name):
    # Build a two-particle System and take a step. A platform that is listed but cannot allocate
    # a context is worse than one that is absent, because it fails at run time instead of here.
    try:
        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(system, integrator,
                                 openmm.Platform.getPlatformByName(platform_name))
        context.setPositions([(0, 0, 0), (0, 0, 0.1)])
        context.getState(getEnergy=True)
        integrator.step(1)
        del context, integrator
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# The CPU platform is what a GPU-less machine runs on; Reference is the fallback that always
# exists. Whichever is present is exercised, so "installed" means "took a step".
out["cpu_check"] = one_step("CPU") if "CPU" in out["platforms"] else "no CPU platform"
out["reference_check"] = (one_step("Reference") if "Reference" in out["platforms"]
                          else "no Reference platform")
out["cuda_check"] = one_step("CUDA") if out["cuda_available"] else "no CUDA platform"
print(json.dumps(out))
"""


def verify_environment(prefix: Path, *, strict: bool = True) -> dict[str, Any]:
    """Run every advertised workflow's dependencies in `prefix` and report what actually works.

    `strict` raises when the environment cannot run the commands this package advertises. That is
    the point of the check: an environment holding only `openmm` imports fine and then fails
    halfway through `sys-gen`, which is a worse place to discover it.
    """
    import json

    python = Path(prefix) / "bin" / "python"
    if not python.is_file():
        raise InstallError(f"{python} does not exist; there is no environment at {prefix}")
    result = subprocess.run([str(python), "-c", ENVIRONMENT_PROBE],
                            capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise InstallError(
            f"the environment at {prefix} could not be probed:\n{result.stderr[-1500:]}")
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise InstallError(f"unreadable validation output:\n{result.stdout[-800:]}")

    report["problems"] = _problems(report)
    report["warnings"] = warnings_for(report)
    if strict and report["problems"]:
        raise InstallError(
            f"the environment at {prefix} cannot run the advertised OpenMM workflows:\n"
            + "\n".join(f"  - {problem}" for problem in report["problems"]))
    return report


def _problems(report: dict[str, Any]) -> list[str]:
    """What is missing, named by the package or command that would fix it.

    CUDA is deliberately NOT required: a CPU-only machine is a supported machine. A CUDA platform
    that is present but cannot take a step IS a problem, because that one fails at run time.
    """
    problems = []
    errors = report.get("import_errors") or {}
    for module, package in REQUIRED_IMPORTS:
        if module in errors:
            problems.append(f"cannot import {module} ({errors[module]}) -- install {package}")

    executables = report.get("executables") or {}
    missing = [name for name in REQUIRED_EXECUTABLES if not executables.get(name)]
    if missing:
        problems.append(f"AmberTools executable(s) not on PATH: {', '.join(missing)} "
                        "-- install ambertools")
    if not report.get("am1bcc_ready") and not missing:
        problems.append("standard AM1-BCC is not available: OpenFF has no registered AmberTools "
                        f"toolkit (registered: {report.get('openff_toolkits')})")

    for name in ("cpu_check", "reference_check"):
        value = report.get(name)
        if value and value not in ("ok",) and not str(value).startswith("no "):
            problems.append(f"{name}: {value}")
    cuda = report.get("cuda_check")
    if report.get("cuda_available") and cuda != "ok":
        problems.append(f"a CUDA platform is present but unusable: {cuda}")
    # Plugin load failures are usually a conda-forge build shipping plugins for hardware this
    # machine does not have -- the HIP ones on an NVIDIA box, for instance. That is not a broken
    # environment, so it is reported rather than refused. A CUDA plugin failing IS fatal, because
    # CUDA is what runs production.
    for failure in report.get("plugin_load_failures") or []:
        if "CUDA" in failure or "Cuda" in failure:
            problems.append(f"the CUDA plugin failed to load: {failure}")
    return problems


def warnings_for(report: dict[str, Any]) -> list[str]:
    """Worth saying, not worth refusing over."""
    notes = []
    for failure in report.get("plugin_load_failures") or []:
        if "CUDA" not in failure and "Cuda" not in failure:
            notes.append(f"plugin not loaded (hardware absent?): {failure.splitlines()[0]}")
    if not report.get("cuda_available"):
        notes.append("no CUDA platform: generated runs default to CUDA and will refuse to start "
                     "unless MD_PLATFORM names another platform")
    if "openff.nagl_models" in (report.get("import_errors") or {}):
        notes.append("openff-nagl-models missing: the `am1bcc_nagl` charge option is unavailable "
                     "(standard am1bcc is unaffected)")
    return notes
