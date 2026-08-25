"""Write a runnable MD project from a built system and `md.config.yaml`.

The generated project contains ordinary OpenMM scripts and the configuration they read. It does not
import this package at run time and does not carry a copy of it: the classification work was done
by `sys-gen` and is in `inputs/solute.yaml`, and the REST2 scaling arithmetic travels as one small
readable module beside the script that uses it.
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

    write_yaml(out / "md.config.yaml", resolved,
               header="# Resolved protocol, read by every run.py in this project.\n")

    for method in methods:
        directory = out / method
        directory.mkdir(parents=True, exist_ok=True)
        source = "cmd_run.py" if method == "cMD" else "rest2_run.py"
        shutil.copy2(TEMPLATES / source, directory / "run.py")
        # Both scripts import it beside themselves; a project without it cannot run at all.
        shutil.copy2(TEMPLATES / "md_stages.py", directory / "md_stages.py")
        if method == "REST2":
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")

        run_sh = (TEMPLATES / "run.sh").read_text()
        run_sh = run_sh.replace("__PYTHON__", _interpreter()).replace("__METHOD__", method)
        (directory / "run.sh").write_text(run_sh, encoding="utf-8")
        _executable(directory / "run.sh")

        if method == "REST2":
            shutil.copy2(TEMPLATES / "extend.sh", directory / "extend.sh")
            _executable(directory / "extend.sh")

    write_yaml(out / "provenance.yaml", {
        **package_provenance(),
        "generated": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_hashes": {name: sha256_file(inputs / name) for name in REQUIRED_INPUTS},
            "sys_config_hash": sha256_of_document(sys_resolved),
            "md_config_hash": sha256_of_document(document),
        },
    })
    return {"output_folder": str(out), "methods": methods, "implicit": implicit}
