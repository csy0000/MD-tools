"""Write a runnable MD project from a built system and `md.config.yaml`.

One directory per stage, in the order they depend on each other:

    MD/
      minimization/  eq1_nvt_1kcal/  eq2_npt_1kcal/  eq3_npt_free/     the common chain
      cMD/                                                            production, from the last
      REST2/                                                          common stage -- siblings

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
from typing import Any

import yaml

from .config import ConfigError, check_timestep_against_masses, resolve_md_config, \
    sha256_of_document, write_yaml
from .defaults import canonical_method
from .provenance_min import package_provenance, sha256_file
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

        document = dict(stage)
        document["input_state"] = (
            stage["input_state"] if index else
            os.path.join(os.path.relpath(inputs, directory), "initial_state.xml"))
        document.update({
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
        write_yaml(directory / "stage.yaml", document,
                   header=f"# Stage {index + 1} of {len(plan)}. Read by run.py beside this file.\n")
        _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh", stage["path"])

    for method in methods:
        directory = out / method
        directory.mkdir(parents=True, exist_ok=True)
        if method == "cMD":
            shutil.copy2(TEMPLATES / "cmd_run.py", directory / "run.py")
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

    write_yaml(out / "provenance.yaml", {
        **package_provenance(),
        "generated": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_hashes": {name: sha256_file(inputs / name) for name in REQUIRED_INPUTS},
            "sys_config_hash": sha256_of_document(sys_resolved),
            # The RESOLVED protocol this project actually runs, hashed from the bytes written to
            # MD/md.config.yaml. Hashing the input document instead recorded a hash for a file
            # that is not in the project: resolution fills in stage paths, drops the solvent block
            # that does not apply, and reconciles the ensemble with the solvent.
            "md_config_hash": sha256_file(out / "md.config.yaml"),
        },
    })
    return {"output_folder": str(out), "methods": methods, "implicit": implicit,
            "common_stages": [stage["path"] for stage in plan]}


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
