"""Write a runnable MD project from a built system and `md.config.yaml`.

One directory per stage, in the order they depend on each other:

    MD/
      minimization/                                       the common chain
      eq/nvt_1kcal/  eq/npt_1kcal/  eq/npt_free/          equilibration, grouped
      cMD/                                                production, both branching from
      REST2/                                              the LAST common stage -- siblings

Each common stage reads its parent's `final_state.xml` and writes its own. `cMD` and `REST2` both
branch from the LAST common stage, so neither has to run before the other and REST2 never repeats
the minimisation or the NVT/NPT preparation.

The generated project contains ordinary OpenMM scripts and the configuration they read. It does not
import this package at run time and does not carry a copy of it: the classification work was done
by `sys-gen` and is in `inputs/solute.yaml`, and the stage helper plus the REST2 scaling arithmetic
travel as two small readable modules beside the scripts that use them.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import ConfigError, check_timestep_against_masses, resolve_md_config, \
    sha256_of_document, write_yaml
from .defaults import canonical_method
from .provenance_min import implementation_identity, package_provenance, sha256_file
from .stages import stage_plan

TEMPLATES = Path(__file__).resolve().parent / "templates"

#: What sys-gen writes and md-gen needs. A missing one is named rather than discovered later.
REQUIRED_INPUTS = ("system.xml", "topology.pdb", "solute.pdb", "initial_state.xml", "solute.yaml",
                   "resolved_sys.config.yaml")


def _executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _interpreter() -> str:
    """The python that generated this project, recorded so run.sh finds OpenMM by default.

    run.sh falls back to whatever `python3` resolves to if this path stops existing, so recording
    it is a convenience rather than a dependency -- a moved project still runs.
    """
    return sys.executable


def generate_md(*, input_folder: Path, config_path: Path, output_folder: Path) -> dict[str, Any]:
    inputs = Path(input_folder).resolve()
    out = Path(output_folder).resolve()
    config_path = Path(config_path).resolve()

    missing = [name for name in REQUIRED_INPUTS if not (inputs / name).is_file()]
    if missing:
        raise ConfigError(
            f"{inputs} is missing {', '.join(missing)}. Build the system first:\n"
            f"    md-openmm sys-gen -i <structure> --config sys.config.yaml -of {inputs}")

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys_resolved = yaml.safe_load((inputs / "resolved_sys.config.yaml").read_text())
    implicit = sys_resolved.get("solvation") == "implicit"

    resolved = resolve_md_config(document, implicit=implicit)
    check_timestep_against_masses(resolved, sys_resolved)
    methods = [canonical_method(m) for m in resolved["methods"]]

    out.mkdir(parents=True, exist_ok=True)
    # The inputs folder is addressed RELATIVE to the generated project, so moving `inputs/` and
    # `MD/` together needs no edit. An absolute path is recorded only if the two are not on a
    # shared root, in which case relativity would be a lie.
    try:
        relative_inputs = os.path.relpath(inputs, out)
    except ValueError:                             # different drives (Windows)
        relative_inputs = str(inputs)
    resolved["paths"] = {"inputs_folder": relative_inputs}

    plan = stage_plan(resolved, implicit=implicit)
    resolved["paths"]["common_stages"] = [stage["path"] for stage in plan]
    resolved["paths"]["common_final_stage"] = plan[-1]["path"]
    # Both production methods read this one file. Written into the config rather than recomputed by
    # each script, so "where does production start" has exactly one answer in the project. cMD/ and
    # REST2/ sit one level under MD/, so `../` reaches the grouped stage directory.
    resolved["paths"]["common_final_state"] = f"../{plan[-1]['path']}/final_state.xml"
    # Recorded once, read by every generated script. REST2's per-tau equilibration used to look
    # for a key that was never written and recorded null.
    # Carried INTO the project so runtime records can name the implementation that wrote them.
    # The scripts cannot import md_templates to ask, and a run months later should not have to
    # guess which version produced it.
    identity = implementation_identity()
    resolved["provenance"] = {
        "template_commit": identity["git_commit"],
        "md_templates_version": identity["version"],
        "installed_fingerprint": identity["installed_fingerprint"]["value"],
    }

    write_yaml(out / "md.config.yaml", resolved,
               header="# Resolved protocol, read by every run.py in this project.\n")

    # One copy for the whole project. Every script locates MD/ by looking for md.config.yaml above
    # itself, so stages at different depths all find the same helper.
    shutil.copy2(TEMPLATES / "md_stages.py", out / "md_stages.py")

    seed = _base_seed(resolved)
    for index, stage in enumerate(plan):
        directory = out / stage["path"]
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEMPLATES / "stage_run.py", directory / "run.py")

        stage_document = dict(stage)
        stage_document["input_state"] = (
            stage["input_state"] if index else
            os.path.join(os.path.relpath(inputs, directory), "initial_state.xml"))
        stage_document.update({
            "temperature_kelvin": resolved["common"]["temperature_kelvin"],
            "timestep_fs": resolved["common"]["timestep_fs"],
            "friction_per_ps": resolved["common"]["friction_per_ps"],
            "integrator_seed": _seed(seed, stage["name"], "integrator"),
            "velocity_seed": _seed(seed, stage["name"], "velocities"),
            "barostat_seed": _seed(seed, stage["name"], "barostat"),
            # Only the first stage after minimisation assigns fresh velocities; every later stage
            # inherits them through its parent's final_state.xml.
            "assign_velocities": False,
            "template_commit": _template_commit(),
        })
        write_yaml(directory / "stage.yaml", stage_document,
                   header=f"# Stage {index + 1} of {len(plan)}. Read by run.py beside this file.\n")
        _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh", stage["path"])

    for method in methods:
        directory = out / method
        directory.mkdir(parents=True, exist_ok=True)
        if method == "cMD":
            shutil.copy2(TEMPLATES / "cmd_run.py", directory / "run.py")
            # cMD carries the scaling module because it may run at tau > 0. It is the SAME file
            # REST2 gets, so a fixed-tau walker cannot drift from the ladder it is meant to match.
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh", "cMD production")
        else:
            shutil.copy2(TEMPLATES / "rest2_run.py", directory / "run.py")
            shutil.copy2(TEMPLATES / "rest2_equilibrate.py", directory / "equilibrate.py")
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh",
                            "REST2 exchange production")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "equilibrate.sh",
                            "REST2 per-tau equilibration", script="equilibrate.py")
            shutil.copy2(TEMPLATES / "extend.sh", directory / "extend.sh")
            _executable(directory / "extend.sh")
            for replica in range(int(resolved["REST2"]["number_of_replicas"])):
                for phase in ("equilibration", "production"):
                    (directory / f"replica_{replica:02d}" / phase).mkdir(parents=True,
                                                                        exist_ok=True)

    _write_run_all(out, plan, methods)

    write_yaml(out / "provenance.yaml", _md_provenance(
        out=out, inputs=inputs, relative_inputs=relative_inputs, resolved=resolved,
        sys_resolved=sys_resolved, plan=plan, methods=methods, seed=seed))
    manifest = write_generated_manifest(out)
    if manifest is not None:
        pass
    return {"output_folder": str(out), "methods": methods, "implicit": implicit,
            "common_stages": [stage["path"] for stage in plan]}


MD_PROVENANCE_FORMAT = "md-templates-md-provenance/v1"
#: What `md-gen` itself wrote. NOT the dataset checksum manifest: trajectories, checkpoints and
#: final states do not exist yet when this is written, and MD-data computes those at archival.
GENERATED_MANIFEST = "generated-files.sha256"


def implicit_route(sys_resolved: dict) -> bool:
    return sys_resolved.get("solvation") == "implicit"


def _md_provenance(*, out: Path, inputs: Path, relative_inputs: str, resolved: dict,
                   sys_resolved: dict, plan: list, methods: list, seed: int) -> dict[str, Any]:
    """Which implementation generated this project, from which prepared system, with which seeds.

    The lineage fields -- the three hashes of the parent `inputs/` records -- are what let a reader
    confirm that this MD/ belongs to that inputs/, rather than to a different bundle that happens
    to sit beside it.
    """
    from .provenance_min import environment_versions, implementation_identity, sha256_file

    def parent_hash(name: str) -> Optional[str]:
        path = inputs / name
        return sha256_file(path) if path.is_file() else None

    stage_seeds = {}
    for stage in plan:
        stage_seeds[stage["path"]] = {
            "integrator": _seed(seed, stage["name"], "integrator"),
            "velocities": _seed(seed, stage["name"], "velocities"),
            "barostat": _seed(seed, stage["name"], "barostat"),
        }
    replica_seeds = {}
    if "REST2" in methods:
        for replica in range(int((resolved.get("REST2") or {}).get("number_of_replicas", 0))):
            replica_seeds[f"replica_{replica:02d}"] = {
                name: _seed(seed, "REST2", replica, name)
                for name in ("integrator", "velocities", "barostat")}

    common = resolved.get("common") or {}
    constraints = (sys_resolved.get("constraints") or {})
    timestep_fs = float(common.get("timestep_fs", 2.0))
    frequency = common.get("barostat_frequency_steps")
    return {
        "format": MD_PROVENANCE_FORMAT,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": list(sys.argv),
        "implementation": implementation_identity(),
        "environment": environment_versions(),
        # The protocol as RESOLVED, recorded here rather than left to be read back out of the
        # configuration. `null` means "does not apply to this solvation route", which is why the
        # implicit case writes null for pressure and barostat rather than omitting them.
        "protocol": {
            "thermostat": {
                "integrator": "openmm.LangevinMiddleIntegrator",
                "temperature_kelvin": common.get("temperature_kelvin"),
                "friction_per_ps": common.get("friction_per_ps"),
                "friction_note": ("OpenMM collision rate in ps^-1; 1.0 ps^-1 is a nominal 1 ps "
                                  "damping time"),
            },
            "timestep_fs": timestep_fs,
            "constraints": {
                "type": constraints.get("type"),
                "rigid_water": constraints.get("rigid_water"),
                "hydrogen_mass_amu": constraints.get("hydrogen_mass_amu"),
                "hydrogen_mass_repartitioning": constraints.get("hydrogen_mass_amu") is not None,
            },
            "pressure_coupling": None if implicit_route(sys_resolved) else {
                "barostat": "openmm.MonteCarloBarostat",
                "pressure_bar": common.get("pressure_bar"),
                "frequency_steps": frequency,
                "interval_ps": (round(float(frequency) * timestep_fs / 1000.0, 6)
                                if frequency else None),
                "active_in_stages": [s["path"] for s in plan if s.get("barostat_active")],
                "present_but_inactive_in_stages": [s["path"] for s in plan
                                                   if not s.get("barostat_active")],
            },
        },
        "parent_system": {
            "inputs_path": relative_inputs,
            "provenance_sha256": parent_hash("provenance.yaml"),
            "forcefield_sha256": parent_hash("forcefield.json"),
            "checksums_sha256": parent_hash("SHA256SUMS"),
            "input_hashes": {name: sha256_file(inputs / name) for name in REQUIRED_INPUTS},
        },
        "sys_config_hash": sha256_of_document(sys_resolved),
        # The RESOLVED protocol this project runs, hashed from the bytes written to
        # MD/md.config.yaml -- resolution fills in stage paths and reconciles the ensemble with
        # the solvent, so the input document is a different file.
        "md_config_hash": sha256_file(out / "md.config.yaml"),
        "methods": list(methods),
        "common_stage_plan": [{"path": s["path"], "kind": s["kind"], "ensemble": s["ensemble"],
                               "parent": s.get("parent_path"),
                               "input_state": s["input_state"]} for s in plan],
        "seeds": {"base": seed, "common_stages": stage_seeds, "rest2_replicas": replica_seeds},
        "generated_paths": sorted(
            p.relative_to(out).as_posix() for p in out.rglob("*")
            if p.is_file() and p.name != GENERATED_MANIFEST),
        "checksum_manifest": GENERATED_MANIFEST,
        "checksum_manifest_scope": (
            "files written by md-gen before any dynamics: scripts, launchers, stage.yaml and "
            "md.config.yaml. Trajectories, checkpoints and final states are produced later and "
            "are checksummed by MD-data at archival, not here."),
    }


def write_generated_manifest(out: Path) -> Path:
    """Hash what md-gen produced, deterministically, excluding the manifest itself."""
    from .provenance_min import sha256_file

    out = Path(out)
    lines = []
    for path in sorted(out.rglob("*"), key=lambda p: p.relative_to(out).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(out).as_posix()
        if relative == GENERATED_MANIFEST:
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    manifest = out / GENERATED_MANIFEST
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _template_commit() -> str | None:
    """Which revision of the run-script templates this project was written from."""
    return (package_provenance().get("md_templates") or {}).get("git_commit")


def _base_seed(resolved: dict[str, Any]) -> int:
    value = (resolved.get("common") or {}).get("random_seed")
    return int(value) if value is not None else 20260825


def _seed(base: int, *purpose: Any) -> int:
    """The same derivation the generated scripts use, so a stage.yaml and a run agree."""
    value = int(base)
    for part in purpose:
        for byte in str(part).encode("utf-8"):
            value = (value * 1000003 + byte) & 0xFFFFFFFF
    seed = value % (2 ** 31 - 1)
    return seed or 1


def _write_launcher(template: Path, path: Path, label: str, *, script: str = "run.py") -> None:
    text = (template.read_text()
            .replace("__PYTHON__", _interpreter())
            .replace("__STAGE__", label)
            .replace("__SCRIPT__", script)
            .replace("__LOG__", Path(script).stem + ".log"))
    path.write_text(text, encoding="utf-8")
    _executable(path)


def _write_run_all(out: Path, plan: list[dict[str, Any]], methods: list[str]) -> None:
    """The convenience wrapper. It calls the stage scripts; it does not reimplement them."""
    lines = []
    if "cMD" in methods:
        lines += ['echo "== cMD production =="', '( cd cMD && ./run.sh )']
    if "REST2" in methods:
        lines += ['echo "== REST2 per-tau equilibration =="',
                  '( cd REST2 && ./equilibrate.sh )',
                  'echo "== REST2 exchange production =="',
                  '( cd REST2 && ./run.sh )']
    text = ((TEMPLATES / "run_all.sh").read_text()
            .replace("__COMMON_STAGES__", " ".join(stage["path"] for stage in plan))
            .replace("__PRODUCTION__", "\n".join(lines)))
    (out / "run_all.sh").write_text(text, encoding="utf-8")
    _executable(out / "run_all.sh")
