"""A short REST2 run completes, and extends from its checkpoints.

Explicit solvent deliberately: the NPT exchange is the part of this repository that OpenMM does not
provide, and its reduced potential carries a pV term that only exists with a box. The box is as
small as the cutoff allows, because what is under test is the exchange bookkeeping, not a size.
"""
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


@pytest.fixture(scope="module")
def rest2_project(tmp_path_factory):
    work = tmp_path_factory.mktemp("rest2smoke")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "REST2", "--solvent", "OPC", cwd=work)

    sys_config = work / "sys.config.yaml"
    document = yaml.safe_load(sys_config.read_text())
    document["solvent"]["padding_nm"] = 0.5
    document["solvent"]["cutoff_nm"] = 0.5
    sys_config.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    md_config = work / "md.config.yaml"
    protocol = yaml.safe_load(md_config.read_text())
    protocol["minimization"]["max_iterations"] = 25
    protocol["equilibration"] = {"nvt_duration_ps": 0.0, "npt_duration_ps": 0.0,
                                 "restraint_k_kcal_mol_a2": 1.0}
    # THREE replicas, not two: with two, the alternating phases give a pair on phase 0 and none
    # on phase 1, so an extension would legitimately record nothing and the continuation could not
    # be observed. Three is the smallest ladder where both phases exchange.
    protocol["REST2"].update({"number_of_replicas": 3, "duration_per_segment_ps": 0.02,
                              "number_of_exchanges": 1, "tau_max": 0.05})
    md_config.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work / "MD" / "REST2"


def _run(directory, script="run.py", args=()):
    environment = dict(os.environ, MD_PLATFORM="Reference")
    return subprocess.run([sys.executable, script, *args], cwd=str(directory),
                          capture_output=True, text=True, env=environment, timeout=3600)


def _rows(directory):
    return list(csv.DictReader((directory / "exchange_attempts.csv").open()))


def test_a_short_rest2_segment_completes(rest2_project):
    result = _run(rest2_project)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "segment complete" in result.stdout

    rows = _rows(rest2_project)
    assert len(rows) == 1, rows                     # 1 round, phase 0 -> the single pair (0,1)
    assert rows[0]["replica_i"] == "0" and rows[0]["replica_j"] == "1"
    for replica in range(3):
        assert (rest2_project / f"replica_{replica:02d}.chk").is_file()


def test_the_ladder_scaling_is_the_amber_convention(rest2_project):
    """s = (1-tau)^2 for solute-solute and (1-tau) for solute-environment."""
    sys.path.insert(0, str(rest2_project))
    import rest2_scaling

    assert rest2_scaling.scale_factor_for_tau(0.0) == 1.0
    assert rest2_scaling.scale_factor_for_tau(0.5) == pytest.approx(0.25)
    assert rest2_scaling.scale_factor_for_tau(0.2) == pytest.approx(0.64)


def test_extending_continues_the_exchange_sequence(rest2_project):
    """A second invocation that restarted the sequence would look healthy and be two runs."""
    before = _rows(rest2_project)
    assert before, "run the first segment before extending"

    result = _run(rest2_project)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "resuming" in result.stdout

    after = _rows(rest2_project)
    assert len(after) > len(before), "the extension recorded no new attempt"
    attempts = [int(row["attempt_index"]) for row in after]
    assert attempts == list(range(len(after))), "attempt indices must continue, not restart"
    steps = [int(row["step"]) for row in after]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), steps
    # the earlier rows must be untouched: the history is appended, never rewritten
    assert after[:len(before)] == before
