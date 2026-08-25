"""`validate-env` — decide whether this machine can do the work, before it starts doing it.

Preparation costs AM1-BCC charges and a 2 ns equilibration; production costs far more. Both fail
at the point of use if a dependency is missing, which is the most expensive place to find out. So
this check runs first and reports everything a user would otherwise have to discover one traceback
at a time.

A check is either OK, a WARNING (works, but something is worth knowing), or an ERROR (the work
cannot proceed). Only errors make the command fail — and `prepare`/`rest2` run the same checks and
refuse to start when there are any.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional

from . import provenance

MIN_PYTHON = (3, 11)

#: Import name -> what it is needed for. Checked by import, not by metadata: a distribution can be
#: present and the module still be unimportable, and it is the import that the run performs.
REQUIRED_IMPORTS = {
    "openmm": "building and propagating the System",
    "yaml": "reading manifests",
    "numpy": "numerics",
}
#: Needed only for the route that uses them, so their absence is an error only in context.
ROUTE_IMPORTS = {
    "rdkit": "smiles route: parsing, embedding and canonical identity",
    "openff.toolkit": "smiles route: Sage parameterisation",
    "openmmforcefields": "smiles route: the SMIRNOFF/GAFF template generator",
}


@dataclass
class Check:
    name: str
    level: str          # "ok" | "warning" | "error"
    detail: str

    def line(self) -> str:
        tag = {"ok": "  ok  ", "warning": " warn ", "error": " FAIL "}[self.level]
        return f"[{tag}] {self.name}: {self.detail}"


def _import_check(module: str, why: str, *, level_if_missing: str) -> Check:
    try:
        import importlib

        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", None)
        return Check(module, "ok", f"{version or 'present'} ({why})")
    except Exception as exc:                       # noqa: BLE001
        return Check(module, level_if_missing, f"unavailable — {exc} ({why})")


def run_checks(
    *,
    platform: Optional[str] = None,
    device: Optional[str] = None,
    precision: str = "mixed",
    route: str = "smiles",
) -> list[Check]:
    """Every check, in the order they matter. `platform=None` skips the platform-specific ones."""
    checks: list[Check] = []

    v = sys.version_info
    checks.append(Check(
        "python",
        "ok" if v[:2] >= MIN_PYTHON else "error",
        f"{v.major}.{v.minor}.{v.micro} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    ))

    pkg_version = provenance.package_version()
    checks.append(Check(
        "md-templates",
        "ok" if pkg_version != "unknown" else "error",
        f"{pkg_version} at {provenance.package_location()}",
    ))
    install = provenance.installed_from_wheel()
    checks.append(Check(
        "install", "ok",
        f"kind={install['install_kind']}"
        + (f" wheel={install['wheel']} sha256={install['wheel_sha256']}"
           if install["wheel"] else ""),
    ))

    for module, why in REQUIRED_IMPORTS.items():
        checks.append(_import_check(module, why, level_if_missing="error"))
    for module, why in ROUTE_IMPORTS.items():
        # a pdb-route run genuinely does not need RDKit or the OpenFF stack
        checks.append(_import_check(
            module, why, level_if_missing="error" if route == "smiles" else "warning"
        ))

    # Version identity before platforms: a platform list from the wrong OpenMM build is a report
    # about the wrong software. A mismatch is a warning rather than an error because the package
    # runs on neighbouring releases -- what must not happen is running on one and recording another.
    version_ok, version_msg = provenance.check_openmm_version()
    checks.append(Check(
        "openmm version",
        "ok" if version_ok else "warning",
        version_msg,
    ))

    # The machine itself, reported so a run can be PLANNED rather than guessed at: how many
    # replicas will fit, how many workers a test run should use, whether a GPU is already busy.
    # A checklist that says "ok" without saying what it is ok on leaves the operator to go and
    # look, which is where wrong device choices come from.
    hw = provenance.hardware()
    checks.append(Check(
        "cpu",
        "ok",
        f"{hw.get('cpu_count')} logical cores on {hw.get('node')} "
        f"({hw.get('machine')}, {hw.get('system')} {hw.get('release')})",
    ))

    from .gpus import discover_devices

    try:
        found = discover_devices()
    except Exception as exc:                       # noqa: BLE001 - reporting must never fail a run
        found = None
        checks.append(Check("gpu", "warning", f"device discovery failed: {type(exc).__name__}: {exc}"))
    if found is not None:
        visible, usable = found["n_visible"], found["n_usable"]
        busy = len(found["busy_uuids"])
        if not found["nvidia_smi_available"]:
            checks.append(Check("gpu", "warning", "nvidia-smi not found; no CUDA device inventory"))
        elif visible == 0:
            checks.append(Check("gpu", "warning",
                                f"no visible CUDA device "
                                f"(CUDA_VISIBLE_DEVICES={found['cuda_visible_devices']!r})"))
        else:
            checks.append(Check(
                "gpu",
                "ok" if usable else "warning",
                f"{usable} usable of {visible} visible ({busy} busy)"
                + (f", CUDA_VISIBLE_DEVICES={found['cuda_visible_devices']}"
                   if found["cuda_visible_devices"] else "")))
            # `gpu` and not `device`: `device` is this function's PARAMETER, and shadowing it here
            # left it holding the last GPU dict instead of None. The CPU branch below then saw
            # `device is not None`, emitted a fatal "--device is meaningless for CPU", and that dict
            # reached `int()` in main() -- so `md-openmm prepare --config` failed on every CPU run.
            busy_uuids = set(found["busy_uuids"])
            for gpu in found["visible"]:
                state = "BUSY" if gpu["uuid"] in busy_uuids else "free"
                checks.append(Check(
                    f"  gpu {gpu['logical_index']}",
                    "ok" if state == "free" else "warning",
                    f"physical {gpu['physical_index']}  {gpu['name']}  "
                    f"{gpu['memory_total_mib']} MiB  {state}  {gpu['uuid']}"))
            # what that inventory means for a ladder, since that is the decision it feeds
            checks.append(Check(
                "  rest2 capacity", "ok",
                f"a ladder of N replicas will use min(N, {usable}) device(s); "
                f"more replicas than that share, propagating sequentially per device"))

    platforms = provenance.openmm_platforms()
    checks.append(Check(
        "openmm platforms",
        "ok" if platforms else "error",
        ", ".join(platforms) if platforms else "none reported — the OpenMM install is broken",
    ))

    sqm = provenance.sqm_path()
    checks.append(Check(
        "sqm",
        "ok" if sqm else ("error" if route == "smiles" else "warning"),
        sqm or "not on PATH — AM1-BCC charges cannot be computed (AmberTools)",
    ))

    if platform:
        checks.extend(_platform_checks(platform, device, precision, platforms))
    return checks


def _platform_checks(platform: str, device: Optional[str], precision: str,
                     platforms: list[str]) -> list[Check]:
    out: list[Check] = []
    if platform not in platforms:
        out.append(Check(
            f"platform {platform}", "error",
            f"not available; OpenMM reports {platforms or 'nothing'}",
        ))
        return out
    out.append(Check(f"platform {platform}", "ok", "available"))

    if platform == "CPU":
        if precision != "mixed":
            out.append(Check("precision", "warning",
                             f"the CPU platform ignores precision={precision!r}"))
        if device is not None:
            out.append(Check("device", "error", "--device is meaningless for CPU"))
        return out

    # CUDA / OpenCL: precision is a real property, and a requested device must exist
    try:
        import openmm

        plat = openmm.Platform.getPlatformByName(platform)
        names = [plat.getPropertyName(i) for i in range(plat.getNumPropertyNames())]
        if "Precision" in names:
            out.append(Check("precision", "ok", f"{precision} requested; property supported"))
        else:
            out.append(Check("precision", "warning",
                             f"{platform} exposes no Precision property"))
    except Exception as exc:                       # noqa: BLE001
        out.append(Check("precision", "warning", f"could not query {platform}: {exc}"))

    if platform == "CUDA":
        devices = provenance.hardware().get("cuda_devices")
        if devices is None:
            out.append(Check("cuda runtime", "warning",
                             "nvidia-smi not available; cannot enumerate devices"))
        else:
            out.append(Check("cuda runtime", "ok", f"{len(devices)} device(s): "
                                                   + "; ".join(devices)))
            if device is not None:
                try:
                    idx = int(device)
                except ValueError:
                    out.append(Check("device", "error", f"--device {device!r} is not an integer"))
                    return out
                if idx < 0 or idx >= len(devices):
                    out.append(Check(
                        "device", "error",
                        f"--device {idx} does not exist; this machine has {len(devices)}",
                    ))
                else:
                    out.append(Check("device", "ok", f"{idx}: {devices[idx]}"))
        out.append(Check("CUDA_DEVICE_ORDER", "ok",
                         "set to PCI_BUS_ID at launch so --device matches nvidia-smi numbering"))
    return out


def report(checks: list[Check]) -> str:
    return "\n".join(c.line() for c in checks)


def errors(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.level == "error"]


def require_ok(*, platform: Optional[str] = None, device: Optional[str] = None,
               precision: str = "mixed", route: str = "smiles") -> None:
    """Raise before any expensive work if the environment cannot support it."""
    checks = run_checks(platform=platform, device=device, precision=precision, route=route)
    bad = errors(checks)
    if bad:
        raise EnvironmentError(
            "this environment cannot run the requested work:\n"
            + "\n".join(c.line() for c in bad)
            + "\n\nRun `md-openmm validate-env` for the full report."
        )


def as_dict(checks: list[Check]) -> dict[str, Any]:
    return {
        "ok": not errors(checks),
        "checks": [{"name": c.name, "level": c.level, "detail": c.detail} for c in checks],
    }
