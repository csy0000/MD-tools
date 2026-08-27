"""Install OpenMM into `$MD_STACK/envs/openmm-<version>/`.

Uses whichever of micromamba, mamba or conda is already present. It does not install a package
manager, a CUDA driver, compilers, MPI, or another MD engine: those are the machine's business, and
a script that installs a driver behind someone's back is worse than one that stops and says what is
missing.
"""
from __future__ import annotations

import json
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
    "mdtraj",                 # reads the source DCD and its box vectors for the AIS method
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
    # AIS starts from an existing equilibrium trajectory. `md-openmm md-gen --method AIS` writes a
    # runtime that reads a whole-system DCD and its periodic box vectors, so an environment without
    # mdtraj can generate an AIS project it cannot run.
    ("mdtraj", "mdtraj"),
)

#: AmberTools executables the two routes shell out to.
REQUIRED_EXECUTABLES = ("sqm", "antechamber", "tleap")


#: Versions this repository advertises and therefore holds to the exact stable release.
EXACT_RELEASE_VERSIONS = ("8.6.0",)


def md_data_requirement() -> str:
    """The pinned MD-data validator this repository is compatible with.

    Read from `md_data_contract`, which is the ONE place that identity is maintained. A second
    copy of a URL and a commit here is a second thing to forget to update, and the failure it
    produces -- an environment validated against a contract the package does not target -- is
    silent.
    """
    from ..openmm.md_data_contract import md_data_requirement as spec

    return spec()


def md_data_pin() -> dict[str, str]:
    """Repository, commit and contract version of the pinned validator, for the record."""
    from ..openmm import md_data_contract as contract

    return {"repository": contract.MD_DATA_REPOSITORY,
            "commit": contract.MD_DATA_COMMIT,
            "contract_version": contract.MD_DATA_CONTRACT_VERSION,
            "requirement": contract.md_data_requirement()}


def _md_data_record(pin: dict[str, str], **fields: Any) -> dict[str, Any]:
    """One shape for every MD-data outcome, so a caller never has to ask which keys exist.

    The bug this closes: a failed pip install returned `{"installed": False, "error": ...}` with no
    `reasons` and no `contract_support_ready`, while `warnings_for()` reads `reasons` to tell the
    user what went wrong. The failure was recorded and then displayed as an empty list.
    """
    record = {
        **pin,
        "attempted": None,
        "command": None,
        "returncode": None,
        "installed": None,
        "installed_version": None,
        "installed_commit": None,
        "installed_source": None,
        "source_kind": None,
        "commit_verified": False,
        "contract_support_ready": False,
        "reasons": [],
        "error": None,
        "note": None,
    }
    record.update(fields)
    return record


def install_md_data(prefix: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Install the pinned MD-data validator into a created environment.

    Optional by design: `dataset.enabled` is off by default and an unregistered local project
    never needs it. But a user who follows the documented installation and then meets "md-data is
    not installed" at `sys-gen`, an hour into preparing a system, has been told the wrong thing
    about what was installed. So it is attempted, its outcome is recorded either way in ONE shape,
    and a failure is a WARNING rather than a failed installation.
    """
    pin = md_data_pin()
    command = [str(Path(prefix) / "bin" / "pip"), "install", pin["requirement"]]
    if dry_run:
        # Neither success nor failure: nothing was installed, so readiness is not a question that
        # has an answer yet. Saying `contract_support_ready: false` here would read as a failure.
        return _md_data_record(
            pin, attempted=False, command=command, contract_support_ready=None,
            reasons=["dry run: the pinned specification was constructed but not executed, so "
                     "contract readiness was not evaluated"],
            note="dry run: readiness not evaluated")

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error = (result.stderr or result.stdout)[-800:].strip()
        # The repository may be private, or the machine offline. Neither makes the OpenMM
        # environment unusable, and neither is something to paper over -- so the reason lands in
        # `reasons`, which is what the console and machine.yaml actually read.
        return _md_data_record(
            pin, attempted=True, command=command, returncode=result.returncode,
            installed=False, commit_verified=False, contract_support_ready=False,
            error=error,
            reasons=[f"installing the pinned validator failed (pip exit {result.returncode}): "
                     f"{error.splitlines()[-1] if error else 'no output'}",
                     f"install it manually with: pip install '{pin['requirement']}'"])

    verified = verify_md_data(prefix)
    return _md_data_record(pin, attempted=True, command=command,
                           returncode=result.returncode, installed=True,
                           **{k: v for k, v in verified.items() if k not in pin})


#: The probe, run inside the TARGET environment. It reports what is installed there, including
#: the installed distribution's own PEP 610 record -- which is the only thing that can say which
#: source the package actually came from. Echoing the intended pin would describe our wish, not
#: the installation.
_MD_DATA_PROBE = r"""
import json
out = {"import_ok": False, "validator_available": False, "version": None,
       "contract_version": None, "installed_commit": None, "installed_source": None,
       "source_kind": "unknown"}
try:
    from importlib import metadata
    distribution = metadata.distribution("md-data")
    out["version"] = distribution.version
    try:
        record = json.loads(distribution.read_text("direct_url.json") or "{}")
    except Exception:
        record = {}
    out["installed_source"] = record.get("url")
    vcs = record.get("vcs_info") or {}
    if vcs.get("commit_id"):
        out["installed_commit"] = str(vcs["commit_id"]).strip().lower()
        out["source_kind"] = "vcs"
    elif record.get("dir_info") is not None:
        out["source_kind"] = "local directory"
    elif record.get("archive_info") is not None:
        out["source_kind"] = "archive"
    elif record:
        out["source_kind"] = "other"
    else:
        out["source_kind"] = "index or wheel (no direct_url.json)"
except Exception as error:
    out["metadata_error"] = "%s: %s" % (type(error).__name__, error)
try:
    import md_data
    import md_data.storage as storage
    out["import_ok"] = True
    out["version"] = getattr(md_data, "__version__", out["version"])
    out["contract_version"] = getattr(md_data, "CONTRACT_VERSION", None)
    out["validator_available"] = (hasattr(md_data, "validate_dataset")
                                  and hasattr(storage, "check_dataset_tree"))
except Exception as error:
    out["error"] = "%s: %s" % (type(error).__name__, error)
print(json.dumps(out))
"""


def probe_md_data(prefix: Path) -> dict[str, Any]:
    """What is ACTUALLY installed: version, contract version, validator, and source commit."""
    result = subprocess.run([str(Path(prefix) / "bin" / "python"), "-c", _MD_DATA_PROBE],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return {"import_ok": False, "validator_available": False,
                "error": (result.stderr or result.stdout)[-500:].strip()}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as error:
        return {"import_ok": False, "validator_available": False,
                "error": f"unreadable probe output ({type(error).__name__}: {error})"}


def verify_md_data(prefix: Path) -> dict[str, Any]:
    """Whether contract support is READY, and if not, exactly why.

    "Ready" is a conjunction, and every term is checked against the installation rather than
    against the intention:

    * the package imports;
    * both validator entry points exist;
    * its contract version equals the one this repository targets;
    * its installed source is the pinned commit, PROVED from the distribution's own
      `direct_url.json`.

    That last one is the reason this function exists. Recording the pin we asked for and calling it
    verification describes our wish, not the bytes on disk -- and a package installed from a local
    checkout, an index, or a different commit imports and validates perfectly well while being a
    different contract.
    """
    pin = md_data_pin()
    probe = probe_md_data(prefix)
    reasons = []

    if not probe.get("import_ok"):
        reasons.append(f"md_data does not import ({probe.get('error', 'no detail')})")
    elif not probe.get("validator_available"):
        reasons.append("md_data imports but validate_dataset / storage.check_dataset_tree are "
                       "not both available")
    installed_contract = probe.get("contract_version")
    if probe.get("import_ok") and str(installed_contract) != pin["contract_version"]:
        reasons.append(f"installed contract version {installed_contract!r} is not the "
                       f"{pin['contract_version']!r} this repository targets")

    installed_commit = (probe.get("installed_commit") or "").lower()
    if not installed_commit:
        reasons.append(
            f"the installed md-data records no source commit (installed from "
            f"{probe.get('source_kind')}: {probe.get('installed_source')}), so it cannot be shown "
            f"to be the pinned {pin['commit'][:12]}")
    elif installed_commit != pin["commit"].lower():
        reasons.append(f"the installed md-data is from commit {installed_commit[:12]}, not the "
                       f"pinned {pin['commit'][:12]}")

    return {
        **pin,
        "probe": probe,
        "installed_version": probe.get("version"),
        "installed_commit": probe.get("installed_commit"),
        "installed_source": probe.get("installed_source"),
        "source_kind": probe.get("source_kind"),
        "commit_verified": bool(installed_commit and installed_commit == pin["commit"].lower()),
        "contract_support_ready": not reasons,
        "reasons": reasons,
    }


class InstallError(RuntimeError):
    """Installation cannot proceed, with an actionable reason."""


def _kind_of(version_string: str) -> str:
    """`release`, `development`, `prerelease` or `unknown`, from a version string alone."""
    text = str(version_string or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    if ".dev" in lowered or "dev-" in lowered:
        return "development"
    tail = lowered.split(".")[-1]
    if any(marker in tail for marker in ("rc", "alpha", "beta")) or "+" in lowered:
        return "prerelease"
    return "release"


def release_status(*, package: Optional[dict], python_version: str, short_version: str,
                   requested: str) -> dict[str, Any]:
    """Whether the OpenMM in an environment is the exact stable release that was asked for.

    **The obvious check is the wrong one, and this is why.** OpenMM's own
    `openmm.version.version` is not a reliable release marker. conda-forge's *release* package
    `openmm 8.6.0 py312hdfcc665_0` reports

        openmm.version.version       = "8.6.0.dev-c6173db"
        openmm.version.short_version = "8.6.0"
        openmm.version.git_revision  = "c6173db6e8edd705eb59172bd21e9ce69c572405"

    because upstream stamps the build commit into the string and conda-forge builds the release
    from that commit. So `.dev` in the Python string does NOT mean a development build, and
    `short_version` cannot separate a release from a prerelease either -- it is the same triple for
    both. Judging on either one alone gets the answer wrong in one direction or the other.

    The authoritative fact is the INSTALLED PACKAGE identity: `conda-meta/openmm-*.json` records
    the version, build string and channel that the solver actually resolved, and a prerelease is a
    different package version there (`8.6.0rc1`, not `8.6.0`). That is what decides. The Python
    strings are still reported, and the disagreement between them is recorded rather than hidden,
    because a reader comparing a provenance file against `openmm.version.version` will otherwise
    conclude something is wrong.

    Falls back to the Python string when no package record exists -- a pip or source install, where
    there is no package metadata to consult and the string is all there is.
    """
    requested = str(requested or "").strip()
    python_version = str(python_version or "").strip()
    package = package or None

    if package and package.get("version"):
        authority = "conda package metadata"
        authoritative = str(package["version"]).strip()
    else:
        authority = "openmm.version.version (no package metadata found)"
        authoritative = python_version

    kind = _kind_of(authoritative)
    return {
        "requested_version": requested,
        # What the environment is, decided by the package that was installed.
        "authoritative_version": authoritative,
        "authority": authority,
        "build_kind": kind,
        "is_exact_release": bool(requested) and authoritative == requested and kind == "release",
        "exact_release_required": requested in EXACT_RELEASE_VERSIONS,
        # What OpenMM itself reports at runtime, which is what a provenance record carries.
        "runtime_version": python_version,
        "runtime_short_version": str(short_version or "").strip(),
        "package": package,
        # True in the ordinary conda-forge case above. Recorded so the difference is a documented
        # fact rather than a discrepancy somebody has to rediscover.
        "runtime_string_carries_build_marker": (
            bool(python_version) and _kind_of(python_version) != "release"),
    }


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


def driver_cuda_ceiling() -> Optional[str]:
    """The highest CUDA runtime this machine's DRIVER can load, from `nvidia-smi`.

    `nvidia-smi`'s header reports `CUDA Version: X.Y`, which is the driver's ceiling and not the
    toolkit that happens to be installed. It has to be an upper bound on the conda solve because
    the solver otherwise takes the newest `cuda-version` available, and a build compiled for a
    newer toolkit than the driver supports fails at the point a real kernel is loaded:

        openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION

    which is a run-time failure in the middle of a simulation, not an install-time one. Observed on
    driver 580.173.02 (ceiling 13.0) with a solve that chose cuda-version 13.3.

    Returns None when nvidia-smi is absent or unparseable, in which case no ceiling is applied and
    the solver behaves as before.
    """
    import re

    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    found = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", result.stdout)
    return found.group(1) if found else None


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
    ceiling = driver_cuda_ceiling() if cuda else None
    if cuda:
        # conda-forge builds OpenMM against a CUDA version, and the solver takes the newest one it
        # can unless it is bounded. Bounded ABOVE by what the DRIVER supports, because a build
        # newer than the driver installs cleanly, passes a trivial context check, and then fails
        # with CUDA_ERROR_UNSUPPORTED_PTX_VERSION the first time a real force kernel is loaded.
        command.append("cuda-version>=11.8")
        if ceiling:
            command.append(f"cuda-version<={ceiling}")

    lines = [f"# {datetime.now(timezone.utc).isoformat()}",
             f"# package manager: {manager} ({executable})",
             f"# cuda requested: {cuda}",
             f"# driver cuda ceiling: {ceiling or 'unknown (no bound applied)'}",
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
        # The version that was asked for is what the exact-release rule is applied against: an
        # install of 8.6.0 that lands a development build has not installed what it advertised.
        check = verify_environment(prefix, version=version)
    except InstallError as error:
        lines += ["", "# validation failed", str(error)]
        log.write_text("\n".join(lines), encoding="utf-8")
        raise
    lines += ["", "# validation", repr(check)]

    # The MD-data validator the contract-managed workflow needs. Attempted here so the documented
    # installation delivers what the documentation advertises, and recorded either way.
    # `install_md_data` already verifies on success and carries its own reasons on failure, so
    # there is nothing to re-run or patch up here.
    md_data = install_md_data(prefix)
    lines += ["", "# md-data (pinned)", "$ " + " ".join(md_data.get("command") or []),
              repr(md_data)]
    check["md_data"] = md_data
    # Three states, never collapsed into one. A researcher who only wants unregistered local
    # simulation has a working installation even when contract support is unavailable, and saying
    # otherwise would either block them or promise them something they did not get.
    check["capabilities"] = capability_summary(md_data)
    log.write_text("\n".join(lines), encoding="utf-8")

    record = record_environment(stack, prefix, check, log=log, version=version)
    return {"prefix": str(prefix), "log": str(log), "record": record, **check}


def capability_summary(md_data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The three states, in one place, so installation and validation cannot disagree.

    `openmm_runtime_ready` is separate on purpose: a researcher doing unregistered local
    simulation has a working environment whether or not the contract validator is available, and
    collapsing the two would either block them or promise them something they did not get.
    """
    md_data = md_data or {}
    return {
        "openmm_runtime_ready": True,
        "md_data_contract_support_ready": bool(md_data.get("contract_support_ready")),
        "md_data_unavailable_reasons": list(md_data.get("reasons") or []),
    }


def _yaml_safe(value: Any) -> Any:
    """Plain data only. A machine.yaml is read months later by something that is not this code.

    Subprocess results and exceptions stringify into something unparseable and unstable, so
    anything that is not a plain container, string, number, bool or None becomes its `str()`.
    """
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
        # The whole MD-data result and the capability summary, not a selection of them. Dropping
        # these meant the one durable record of what this environment can do said nothing about
        # whether contract-managed generation would work.
        "md_data": _yaml_safe(check.get("md_data")),
        "capabilities": _yaml_safe(check.get("capabilities")),
        "validated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if log is not None:
        record["log"] = str(log)
    machine.setdefault("installed", {})["openmm"] = record
    save_machine(stack, machine)
    return record


def validate_existing(stack: Path, prefix: Path, *, version: str = "") -> dict[str, Any]:
    """Validate an environment somebody else built, and record it the same way.

    The check is the same either way. An environment this command did not create is not less
    obliged to be able to run the commands this package advertises.

    `version` is empty unless the caller says which release this environment is supposed to be. It
    stays empty by default deliberately: a user validating an environment they built around a
    different OpenMM gets the facts reported, and nothing refused, which is what
    `--validate` is for.
    """
    prefix = Path(prefix).resolve()
    check = verify_environment(prefix, version=version)
    # Validation answers the same question installation does, so it records the same shape. An
    # environment validated later must not produce a machine.yaml a reader has to special-case.
    md_data = verify_md_data(prefix)
    check["md_data"] = md_data
    check["capabilities"] = capability_summary(md_data)
    check["warnings"] = list(check.get("warnings") or []) + [
        note for note in warnings_for(check) if "MD-DATA" in note]
    record = record_environment(stack, prefix, check)
    return {"prefix": str(prefix), "record": record, **check}


#: Runs INSIDE the environment being validated, so it must import nothing from this package and
#: must print exactly one line of JSON. It reports rather than judges; `verify_environment` below
#: decides what counts as a failure.
ENVIRONMENT_PROBE = r"""
import importlib, json, os, shutil, sys

out = {"python_version": sys.version.split()[0], "python": sys.executable,
       "versions": {}, "import_errors": {}, "executables": {}}


def detect_nvidia():
    # Whether this MACHINE should be able to run CUDA at all, asked without OpenMM's help.
    # A CPU-only CI runner ships the CUDA plugin and cannot load it, because libcuda.so.1 is part
    # of the driver and there is no driver. That is the expected state of a machine with no GPU,
    # not a broken environment -- but on a machine that DOES have the hardware, the same failure
    # means CUDA is genuinely broken. These signals are what separates the two.
    import ctypes, glob, subprocess

    signals = {"device_nodes": sorted(glob.glob("/dev/nvidia[0-9]*"))}
    signals["nvidia_smi"] = bool(shutil.which("nvidia-smi"))
    signals["nvidia_smi_lists_gpus"] = False
    if signals["nvidia_smi"]:
        try:
            listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                    timeout=60)
            signals["nvidia_smi_lists_gpus"] = (listed.returncode == 0
                                                and bool(listed.stdout.strip()))
        except Exception as exc:
            signals["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    try:
        ctypes.CDLL("libcuda.so.1")
        signals["libcuda"] = True
    except OSError as exc:
        signals["libcuda"] = False
        signals["libcuda_error"] = str(exc)
    # The driver library is the decisive one: without it nothing CUDA can run, and with it the
    # machine is expected to.
    signals["present"] = bool(signals["device_nodes"]) or signals["nvidia_smi_lists_gpus"] \
        or signals["libcuda"]
    return signals


out["nvidia"] = detect_nvidia()

for module in ("openmm", "yaml", "numpy", "openff.toolkit", "openff.nagl_models",
               "openmmforcefields", "parmed", "rdkit", "mdtraj"):
    try:
        loaded = importlib.import_module(module)
        version = getattr(loaded, "__version__", None)
        if module == "openmm":
            # The FULL string. `short_version` is "8.6.0" for the stable release AND for
            # 8.6.0.dev-c6173db, so reporting only the short form cannot distinguish a release
            # from a development build -- which is exactly the gap this records.
            version = loaded.version.version
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
# Three separate facts, because they answer different questions. `openmm_version` is the full
# string OpenMM reports and is what a run is actually using; `openmm_short_version` is the
# marketing triple, which a development build shares with the release it precedes; the git
# revision is empty for a release build and set for a development one.
# Four separate facts, because they answer different questions and the first two disagree in the
# ordinary case. `openmm_version` is what OpenMM reports at runtime and is what lands in a
# provenance record; `openmm_short_version` is the marketing triple, shared by a release and any
# prerelease of it; the git revision is the commit the binary was built from; and the PACKAGE is
# what the solver actually installed, which is the only one of the four that separates a release
# from a release candidate. See `release_status`.
out["openmm_version"] = openmm.version.version
out["openmm_short_version"] = openmm.version.short_version
out["openmm_git_revision"] = getattr(openmm.version, "git_revision", "") or ""


def conda_package(name):
    # The conda-meta record for `name` in this prefix, or None if it was not conda-installed.
    # (A comment, not a docstring: this whole probe lives inside a triple-quoted string.)
    import glob

    records = sorted(glob.glob(os.path.join(sys.prefix, "conda-meta", name + "-*.json")))
    for path in records:
        try:
            with open(path) as handle:
                data = json.load(handle)
        except Exception:
            continue
        if data.get("name") == name:
            return {"version": data.get("version"), "build": data.get("build"),
                    "channel": data.get("channel"), "url": data.get("url"),
                    "record": os.path.basename(path)}
    return None


out["openmm_package"] = conda_package("openmm")
out["mdtraj_package"] = conda_package("mdtraj")


def one_step(platform_name):
    # Build a small System WITH FORCES and take a step. A platform that is listed but cannot
    # allocate a context is worse than one that is absent, because it fails at run time instead of
    # here -- and a System of two free particles is not enough to find that out. A CUDA build
    # compiled for a newer toolkit than the driver supports creates a context for free particles
    # happily and then fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION the moment a real force
    # kernel is loaded, which is the first molecular System a user builds. So the probe carries a
    # NonbondedForce and a HarmonicBondForce: the kernels an actual run compiles.
    try:
        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nonbonded = openmm.NonbondedForce()
        nonbonded.addParticle(0.1, 0.3, 0.5)
        nonbonded.addParticle(-0.1, 0.3, 0.5)
        system.addForce(nonbonded)
        bonds = openmm.HarmonicBondForce()
        bonds.addBond(0, 1, 0.1, 1000.0)
        system.addForce(bonds)
        integrator = openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
        context = openmm.Context(system, integrator,
                                 openmm.Platform.getPlatformByName(platform_name))
        context.setPositions([(0, 0, 0), (0, 0, 0.1)])
        context.getState(getEnergy=True, getForces=True)
        integrator.step(2)
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


def verify_environment(prefix: Path, *, strict: bool = True,
                       version: str = "") -> dict[str, Any]:
    """Run every advertised workflow's dependencies in `prefix` and report what actually works.

    `strict` raises when the environment cannot run the commands this package advertises. That is
    the point of the check: an environment holding only `openmm` imports fine and then fails
    halfway through `sys-gen`, which is a worse place to discover it.

    `version` is the version that was ASKED for. When it is one this repository advertises as an
    exact stable release (see `EXACT_RELEASE_VERSIONS`), an environment carrying a development or
    prerelease build of the same triple is a problem rather than a note. Validating some other
    version deliberately -- `--validate` on an environment a user built themselves -- is untouched:
    the facts are still reported, and nothing is refused.
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

    report["release"] = release_status(package=report.get("openmm_package"),
                                       python_version=report.get("openmm_version", ""),
                                       short_version=report.get("openmm_short_version", ""),
                                       requested=version)
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
    release = report.get("release") or {}
    if release.get("exact_release_required") and not release.get("is_exact_release"):
        package = release.get("package") or {}
        installed = (f"{package.get('version')} (build {package.get('build')}, "
                     f"channel {package.get('channel')})" if package.get("version")
                     else f"{release.get('authoritative_version') or 'no version'}")
        problems.append(
            f"OpenMM {release['requested_version']} was requested, but this environment has "
            f"{installed} -- a {release['build_kind']} build, decided from "
            f"{release.get('authority')}. Install the release: "
            f"`conda install -c conda-forge openmm={release['requested_version']}`.")

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

    # Reference and CPU are required everywhere, including the GPU-less release runner: they are
    # what the smoke tests run on.
    for name in ("cpu_check", "reference_check"):
        value = report.get(name)
        if value and value not in ("ok",) and not str(value).startswith("no "):
            problems.append(f"{name}: {value}")

    # Whether a CUDA failure is a fault depends on whether this machine has a GPU at all.
    #
    # conda-forge ships the CUDA plugin unconditionally. On a CPU-only runner it cannot load,
    # because libcuda.so.1 belongs to the driver and there is no driver -- the expected state of a
    # machine without a GPU, and what failed the openmm-v0.2.0 release for no real reason. On a
    # machine that HAS the hardware, the identical message means CUDA is genuinely broken and a
    # production run would land on the CPU or die.
    nvidia = report.get("nvidia") or {}
    cuda_expected = bool(nvidia.get("present"))
    cuda = report.get("cuda_check")
    if cuda_expected:
        if not report.get("cuda_available"):
            problems.append(
                "this machine has NVIDIA hardware or driver "
                f"({_nvidia_evidence(nvidia)}) but OpenMM offers no CUDA platform")
        elif cuda != "ok":
            problems.append(f"a CUDA platform is present but unusable: {cuda}")
        for failure in report.get("plugin_load_failures") or []:
            if "CUDA" in failure or "Cuda" in failure:
                problems.append(f"the CUDA plugin failed to load: {failure}")
    elif report.get("cuda_available") and cuda not in ("ok", None) \
            and not str(cuda).startswith("no "):
        # No driver detected, yet OpenMM lists CUDA and it does not work. Worth naming, because
        # a run that asks for CUDA here will fail rather than fall back.
        problems.append(f"OpenMM lists a CUDA platform that cannot take a step: {cuda}")
    return problems


def _nvidia_evidence(nvidia: dict[str, Any]) -> str:
    """Which signal said there is a GPU, so a fatal CUDA verdict can be argued with."""
    found = []
    if nvidia.get("device_nodes"):
        found.append(f"{len(nvidia['device_nodes'])} /dev/nvidia* node(s)")
    if nvidia.get("nvidia_smi_lists_gpus"):
        found.append("nvidia-smi lists GPUs")
    if nvidia.get("libcuda"):
        found.append("libcuda.so.1 loads")
    return ", ".join(found) or "no signal"


def warnings_for(report: dict[str, Any]) -> list[str]:
    """Worth saying, not worth refusing over."""
    notes = []
    nvidia = report.get("nvidia") or {}
    cuda_expected = bool(nvidia.get("present"))
    for failure in report.get("plugin_load_failures") or []:
        is_cuda = "CUDA" in failure or "Cuda" in failure
        if is_cuda and cuda_expected:
            continue                       # already fatal in _problems; not also a note
        notes.append(f"plugin not loaded (hardware absent?): {failure.splitlines()[0]}")
    if not cuda_expected:
        notes.append("no NVIDIA driver or device detected, so CUDA is not required here; "
                     "unavailable CUDA plugins are expected on this machine")
    if not report.get("cuda_available"):
        notes.append("no CUDA platform: generated runs default to CUDA and will refuse to start "
                     "unless MD_PLATFORM names another platform")
    if "openff.nagl_models" in (report.get("import_errors") or {}):
        notes.append("openff-nagl-models missing: the `am1bcc_nagl` charge option is unavailable "
                     "(standard am1bcc is unaffected)")

    # Contract support is a separate capability from the OpenMM runtime, and stating it plainly is
    # the whole point: a researcher doing unregistered local simulation is unaffected, and one who
    # intended `dataset.enabled: true` needs to know now rather than at sys-gen.
    capabilities = report.get("capabilities") or {}
    if capabilities and not capabilities.get("md_data_contract_support_ready"):
        reasons = capabilities.get("md_data_unavailable_reasons") or ["no reason recorded"]
        notes.append(
            "MD-DATA CONTRACT SUPPORT UNAVAILABLE: the OpenMM runtime is ready and unregistered "
            "local simulation works, but `dataset.enabled: true` will fail before building a "
            "system. Reason: " + reasons[0])
        for extra in reasons[1:]:
            notes.append(f"  also: {extra}")
    return notes
