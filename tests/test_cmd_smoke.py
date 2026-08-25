"""A tiny cMD run completes, and a second invocation continues rather than restarting."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from .conftest import ALA_PDB, run_cli

pytestmark = pytest.mark.slow


def _tiny_project(work, *, solvent="OPC", padding=0.9):
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "--solvent", solvent, cwd=work)
    sys_config = work / "sys.config.yaml"
    document = yaml.safe_load(sys_config.read_text())
    if solvent == "OPC":
        document["solvent"]["padding_nm"] = padding
    sys_config.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    md_config = work / "md.config.yaml"
    protocol = yaml.safe_load(md_config.read_text())
    protocol["minimization"]["max_iterations"] = 25
    protocol["equilibration"]["nvt_duration_ps"] = 0.02
    protocol["equilibration"]["npt_duration_ps"] = 0.02
    protocol["cMD"].update({"duration_ns": 0.0002, "checkpoint_interval_ps": 0.1,
                            "whole_system_interval_ps": 0.1, "solute_interval_ps": 0.1})
    md_config.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work / "MD" / "cMD"


def _run(directory, platform="Reference"):
    environment = dict(os.environ, MD_PLATFORM=platform)
    return subprocess.run([sys.executable, "run.py"], cwd=str(directory),
                          capture_output=True, text=True, env=environment, timeout=1800)


def test_a_tiny_cmd_run_completes_on_the_reference_platform(tmp_path):
    directory = _tiny_project(tmp_path)
    result = _run(directory)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "complete" in result.stdout
    assert (directory / "production.chk").is_file()
    assert (directory / "final_state.xml").is_file()


def test_rerunning_continues_from_the_checkpoint_rather_than_restarting(tmp_path):
    """The failure this guards against reports a plausible run that is not one trajectory."""
    directory = _tiny_project(tmp_path)
    first = _run(directory)
    assert first.returncode == 0, first.stderr[-1500:]

    # ask for twice the length, then run again: it must CONTINUE, not begin at zero
    config = directory.parent / "md.config.yaml"
    document = yaml.safe_load(config.read_text())
    document["cMD"]["duration_ns"] = 0.0004
    config.write_text(yaml.safe_dump(document, sort_keys=False))

    second = _run(directory)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert "resuming from production.chk" in second.stdout, second.stdout[-800:]

    rows = list(csv.DictReader((directory / "production.csv").open()))
    steps = [int(row['#"Step"'] if '#"Step"' in row else row["Step"]) for row in rows]
    assert steps == sorted(steps), "the step counter must increase across the restart"
    assert max(steps) > 100, steps
