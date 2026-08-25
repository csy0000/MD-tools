"""A tiny cMD run: the stages the configuration asks for actually happen, and a rerun continues."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from .conftest import ALA_PDB, dcd_header, run_cli

pytestmark = pytest.mark.slow


def _tiny_project(work, *, solvent="OPC", padding=0.5):
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "--solvent", solvent, cwd=work)
    sys_config = work / "sys.config.yaml"
    document = yaml.safe_load(sys_config.read_text())
    if solvent == "OPC":
        # As small as the cutoff allows. What is under test is the stage sequence and the
        # bookkeeping, and a box big enough to be physically interesting costs minutes per run.
        document["solvent"]["padding_nm"] = padding
        document["solvent"]["cutoff_nm"] = 0.5
    sys_config.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    md_config = work / "md.config.yaml"
    protocol = yaml.safe_load(md_config.read_text())
    protocol["minimization"]["max_iterations"] = 25
    protocol["minimization"]["restraint_k_kcal_mol_a2"] = 1.0
    protocol["equilibration"]["nvt_duration_ps"] = 0.02
    protocol["equilibration"]["npt_duration_ps"] = 0.02
    protocol["equilibration"]["restraint_k_kcal_mol_a2"] = 2.0
    # Two DIFFERENT intervals, so a trajectory written at the wrong one is visible.
    protocol["cMD"].update({"duration_ns": 0.0002, "checkpoint_interval_ps": 0.1,
                            "whole_system_interval_ps": 0.1, "solute_interval_ps": 0.05})
    md_config.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work / "MD" / "cMD"


def _run(directory, platform="CPU"):
    environment = dict(os.environ, MD_PLATFORM=platform)
    return subprocess.run([sys.executable, "run.py"], cwd=str(directory),
                          capture_output=True, text=True, env=environment, timeout=1800)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """One completed explicit-solvent run, inspected by several tests."""
    directory = _tiny_project(tmp_path_factory.mktemp("cmdsmoke"))
    result = _run(directory)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return directory, result


@pytest.fixture(scope="module")
def implicit(tmp_path_factory):
    directory = _tiny_project(tmp_path_factory.mktemp("cmdgbn2"), solvent="GBn2")
    result = _run(directory)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return directory, result


def test_a_tiny_cmd_run_completes_without_a_gpu(finished):
    directory, result = finished
    assert "complete" in result.stdout
    assert (directory / "production.chk").is_file()
    assert (directory / "final_state.xml").is_file()


def test_a_fresh_run_minimises_then_nvt_then_npt_then_produces(finished):
    """A stage the configuration asks for and the script skips is worse than one that is absent."""
    _, result = finished
    text = result.stdout
    assert (text.index("minimising") < text.index("NVT")
            < text.index("NPT") < text.index("equilibration complete")), text


def test_the_nvt_stage_has_no_barostat_and_npt_and_production_have_exactly_one(finished):
    directory, _ = finished
    record = yaml.safe_load((directory / "equilibration.yaml").read_text())
    assert record["barostats_active_minimization"] == 0
    assert record["barostats_active_nvt"] == 0, "a barostat during NVT is not NVT"
    assert record["barostats_active_npt"] == 1
    assert record["barostats_active_production"] == 1


def test_the_solute_is_restrained_through_equilibration_and_free_in_production(finished):
    directory, _ = finished
    record = yaml.safe_load((directory / "equilibration.yaml").read_text())
    # 1 kcal/mol/A^2 = 418.4 kJ/mol/nm^2, and the two configured constants are both honoured.
    assert record["restraint_kj_nm2"] == pytest.approx(418.4)
    assert record["restraint_kj_nm2_equilibration"] == pytest.approx(836.8)
    assert record["restraint_kj_nm2_production"] == 0.0, "production must be unrestrained"


def test_the_whole_system_and_solute_trajectories_use_their_own_intervals(finished):
    directory, _ = finished
    inputs = directory.parent.parent / "inputs"
    solute_atoms = int(yaml.safe_load((inputs / "solute.yaml").read_text())["n_solute_atoms"])

    whole = dcd_header(directory / "whole_system.dcd")
    solute = dcd_header(directory / "solute.dcd")
    # 0.2 ps of production at 2 fs = 100 steps; 0.1 ps = every 50, 0.05 ps = every 25.
    assert whole["interval"] == 50 and whole["frames"] == 2, whole
    assert solute["interval"] == 25 and solute["frames"] == 4, solute
    assert whole["atoms"] > solute["atoms"], "the subset trajectory is not a subset"
    assert solute["atoms"] == solute_atoms


def test_the_solute_trajectory_matches_the_topology_sys_gen_wrote_for_it(finished):
    """A subset DCD read against the whole-system PDB silently mis-assigns every atom."""
    from openmm.app import PDBFile

    directory, _ = finished
    inputs = directory.parent.parent / "inputs"
    solute_pdb = PDBFile(str(inputs / "solute.pdb"))
    assert dcd_header(directory / "solute.dcd")["atoms"] == solute_pdb.topology.getNumAtoms()


def test_an_implicit_system_never_contains_a_barostat(implicit):
    """There is no box to control, so a barostat here would be sampling nothing."""
    from openmm import XmlSerializer

    directory, result = implicit
    record = yaml.safe_load((directory / "equilibration.yaml").read_text())
    assert record["barostats_active_npt"] == 0
    assert record["barostats_active_production"] == 0
    assert "NPT" not in result.stdout, result.stdout

    system = XmlSerializer.deserialize(
        (directory.parent.parent / "inputs" / "system.xml").read_text())
    barostats = [system.getForce(i) for i in range(system.getNumForces())
                 if "Barostat" in type(system.getForce(i)).__name__]
    assert not barostats


def test_rerunning_continues_from_the_checkpoint_and_appends_its_trajectories(tmp_path):
    """The failure this guards against reports a plausible run that is not one trajectory."""
    directory = _tiny_project(tmp_path)
    first = _run(directory)
    assert first.returncode == 0, first.stderr[-1500:]
    before = {name: dcd_header(directory / name)["frames"]
              for name in ("whole_system.dcd", "solute.dcd")}

    # ask for twice the length, then run again: it must CONTINUE, not begin at zero
    config = directory.parent / "md.config.yaml"
    document = yaml.safe_load(config.read_text())
    document["cMD"]["duration_ns"] = 0.0004
    config.write_text(yaml.safe_dump(document, sort_keys=False))

    second = _run(directory)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert "resuming from production.chk" in second.stdout, second.stdout[-800:]
    assert "minimising" not in second.stdout, "a resumed run must not re-equilibrate"
    assert "1 active barostat" in second.stdout, "the resumed run lost its barostat"

    rows = list(csv.DictReader((directory / "production.csv").open()))
    steps = [int(row['#"Step"'] if '#"Step"' in row else row["Step"]) for row in rows]
    assert steps == sorted(steps), "the step counter must increase across the restart"
    assert max(steps) > 100, steps

    for name, count in before.items():
        assert dcd_header(directory / name)["frames"] > count, f"{name} did not append"
