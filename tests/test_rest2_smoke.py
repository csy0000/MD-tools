"""A short REST2 run: the ladder equilibrates, the durations mean what the YAML says, and it
continues rather than restarting.

Explicit solvent deliberately: the NPT exchange is the part of this repository that OpenMM does not
provide. The box is as small as the cutoff allows, because what is under test is the bookkeeping,
not a size.
"""
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

SEGMENT_PS = 0.02
EXCHANGES = 3
STEPS_PER_SEGMENT = 10                      # 0.02 ps at 2 fs
REPLICAS = 3


@pytest.fixture(scope="module")
def built_inputs(tmp_path_factory):
    """One tiny solvated ALA system, shared by every REST2 project below."""
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
    return work


def _project(work, folder, *, replicas):
    """A REST2 project with `replicas` rungs, generated from the shared inputs."""
    config = work / f"md.{folder}.yaml"
    protocol = yaml.safe_load((work / "md.config.yaml").read_text())
    protocol["minimization"]["max_iterations"] = 25
    protocol["minimization"]["restraint_k_kcal_mol_a2"] = 1.0
    protocol["equilibration"] = {"nvt_duration_ps": 0.02, "npt_duration_ps": 0.02,
                                 "restraint_k_kcal_mol_a2": 2.0}
    # Two DIFFERENT trajectory intervals, so one written at the other's interval is visible.
    protocol["REST2"].update({"number_of_replicas": replicas,
                              "duration_per_segment_ps": SEGMENT_PS,
                              "number_of_exchanges": EXCHANGES, "tau_max": 0.05,
                              "checkpoint_interval_ps": 0.02,
                              "whole_system_interval_ps": 0.04,
                              "solute_interval_ps": 0.02})
    config.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", config.name,
                        "-of", f"./{folder}/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work / folder / "REST2"


def _run(directory, script="run.py", args=()):
    environment = dict(os.environ, MD_PLATFORM="CPU")
    return subprocess.run([sys.executable, script, *args], cwd=str(directory),
                          capture_output=True, text=True, env=environment, timeout=3600)


def _rows(directory):
    return list(csv.DictReader((directory / "exchange_attempts.csv").open()))


@pytest.fixture(scope="module")
def ladder(built_inputs):
    """Three rungs, run once. Three is the smallest ladder where both phases offer a pair."""
    directory = _project(built_inputs, "MD", replicas=REPLICAS)
    result = _run(directory)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return directory, result


def test_a_short_rest2_segment_completes(ladder):
    directory, result = ladder
    assert "segment complete" in result.stdout
    for replica in range(REPLICAS):
        assert (directory / f"replica_{replica:02d}.chk").is_file()


def test_a_segment_is_the_time_between_exchanges_not_the_whole_invocation(ladder):
    """`duration_per_segment_ps: 10` with `number_of_exchanges: 1000` is 10 ns, not 10 ps.

    The runner used to divide one segment among the exchanges, which is the same configuration
    running a thousandth of the time it claims.
    """
    directory, result = ladder
    expected = STEPS_PER_SEGMENT * EXCHANGES
    assert f"{expected:,} steps per replica" in result.stdout, result.stdout[-1500:]

    rows = _rows(directory)
    assert max(int(row["step"]) for row in rows) == expected
    assert sorted({int(row["attempt_index"]) for row in rows}) == list(range(EXCHANGES))


def test_every_replica_is_equilibrated_before_the_ladder_produces(ladder):
    """The equilibration block used to be read for cMD and ignored here."""
    directory, _ = ladder
    for replica in range(REPLICAS):
        record = yaml.safe_load(
            (directory / f"replica_{replica:02d}_equilibration.yaml").read_text())
        assert record["restraint_kj_nm2"] == pytest.approx(418.4)
        assert record["restraint_kj_nm2_equilibration"] == pytest.approx(836.8)
        assert record["restraint_kj_nm2_production"] == 0.0
        assert record["barostats_active_nvt"] == 0
        assert record["barostats_active_npt"] == 1
        assert record["barostats_active_production"] == 1


def test_each_replica_has_its_own_integrator_velocity_and_barostat_seed(ladder):
    directory, _ = ladder
    seeds = []
    for replica in range(REPLICAS):
        record = yaml.safe_load(
            (directory / f"replica_{replica:02d}_equilibration.yaml").read_text())
        seeds.extend(record["seeds"].values())
        assert set(record["seeds"]) == {"integrator", "velocities", "barostat"}
    assert len(set(seeds)) == len(seeds), f"seeds repeat across the ladder: {seeds}"


def test_every_replica_writes_a_whole_system_and_a_solute_trajectory(ladder):
    from openmm.app import PDBFile

    directory, _ = ladder
    solute_pdb = PDBFile(str(directory.parent.parent / "inputs" / "solute.pdb"))
    for replica in range(REPLICAS):
        whole = dcd_header(directory / f"replica_{replica:02d}_whole.dcd")
        solute = dcd_header(directory / f"replica_{replica:02d}_solute.dcd")
        assert whole["interval"] == 20 and solute["interval"] == 10, (whole, solute)
        assert solute["frames"] > whole["frames"], "the two intervals are not distinct"
        assert solute["atoms"] == solute_pdb.topology.getNumAtoms()
        assert whole["atoms"] > solute["atoms"]
        assert (directory / f"replica_{replica:02d}.csv").is_file()


def test_the_ladder_scaling_is_the_amber_convention(ladder):
    """s = (1-tau)^2 for solute-solute and (1-tau) for solute-environment."""
    directory, _ = ladder
    sys.path.insert(0, str(directory))
    import rest2_scaling

    assert rest2_scaling.scale_factor_for_tau(0.0) == 1.0
    assert rest2_scaling.scale_factor_for_tau(0.5) == pytest.approx(0.25)
    assert rest2_scaling.scale_factor_for_tau(0.2) == pytest.approx(0.64)


def test_extending_continues_the_exchange_sequence_and_appends_trajectories(ladder):
    """A second invocation that restarted the sequence would look healthy and be two runs."""
    directory, _ = ladder
    before = _rows(directory)
    assert before, "run the first segment before extending"
    frames_before = dcd_header(directory / "replica_00_solute.dcd")["frames"]

    result = _run(directory)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "resuming" in result.stdout
    assert "minimising" not in result.stdout, "a resumed ladder must not re-equilibrate"

    after = _rows(directory)
    assert len(after) > len(before), "the extension recorded no new attempt"
    attempts = sorted({int(row["attempt_index"]) for row in after})
    assert attempts == list(range(len(attempts))), "attempt indices must continue, not restart"
    steps = [int(row["step"]) for row in after]
    assert steps == sorted(steps), steps
    assert max(steps) == STEPS_PER_SEGMENT * EXCHANGES * 2
    # the earlier rows must be untouched: the history is appended, never rewritten
    assert after[:len(before)] == before
    assert dcd_header(directory / "replica_00_solute.dcd")["frames"] > frames_before


def test_a_two_replica_ladder_exchanges_on_every_round(built_inputs):
    """Alternating phases give phase 1 no pair when there are two rungs.

    The old runner attempted an exchange on every other round only, and a run resumed on an odd
    attempt index could sit in that state indefinitely.
    """
    directory = _project(built_inputs, "MD2", replicas=2)
    result = _run(directory)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    rows = _rows(directory)
    assert len(rows) == EXCHANGES, rows
    assert all(row["replica_i"] == "0" and row["replica_j"] == "1" for row in rows)

    # and again, from the odd attempt index the old schedule got stuck on
    again = _run(directory)
    assert again.returncode == 0, again.stdout[-3000:] + again.stderr[-3000:]
    assert len(_rows(directory)) == 2 * EXCHANGES
