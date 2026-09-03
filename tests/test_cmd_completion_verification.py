"""A completed cMD stage is PROVEN complete, not observed to have a field saying so.

WHAT WAS WRONG

    The baseline check read `status: completed` from the log, compared the stage dictionary the
    log carried against the one the invocation resolved, and asked whether the final restart and
    (for dynamics) the trajectory EXIST.

    The recorded `fingerprint` was tested for presence and never recomputed or compared. So a run
    whose `built.xml`, `built.pdb` or `resolved.config` had changed since -- a rebuilt System, a
    topology from a different structure, an edited configuration -- was skipped, and the NEXT
    stage continued from its restart. The error propagates through the chain wearing the right
    file names.

    `exists()` is not `is the file we wrote`. Every test below damages one output in a way that
    leaves it existing, and every one of them was skipped as complete before this change.

Each test states the baseline behaviour it pins, and reaches the real verifier -- not argparse,
and not an earlier unrelated refusal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from md_tools.md.completion import verify_completed_stage


# --- a synthetic completed record, so the unit can be reached without integrating ----------------

def _record(tmp_path: Path, *, fingerprint: str = "f" * 64, **overrides):
    """A minimal log record of a completed dynamics stage, with real files behind it."""
    from md_tools.build.record import file_facts

    restart = tmp_path / "cMD.xml"
    restart.write_text("<State/>", encoding="utf-8")
    trajectory = tmp_path / "cMD.dcd"
    trajectory.write_bytes(b"\x54\x00\x00\x00CORD" + b"\0" * 200)
    state_csv = tmp_path / "cMD.csv"
    state_csv.write_text('#"Step","Potential Energy (kJ/mole)"\n1,-1.0\n2,-1.0\n', encoding="utf-8")

    stage = {"steps": 100, "trajectory_interval_steps": 50, "state_interval_steps": 50,
             "temperature_K": 300.0, "timestep_fs": 2.0, "ensemble": "NVT"}
    record = {
        "status": "completed",
        "fingerprint": fingerprint,
        "stage": dict(stage),
        "outputs": {
            "final_state": file_facts(restart),
            "trajectory": file_facts(trajectory),
            "state_csv": file_facts(state_csv),
        },
    }
    record.update(overrides)
    return record, stage, restart, trajectory, state_csv


def _verify(record, stage, tmp_path, restart, trajectory, **kwargs):
    return verify_completed_stage(
        record, stage=stage, name="cMD", restart=restart, trajectory=trajectory,
        log_path=tmp_path / "cMD.log", fingerprint=kwargs.pop("fingerprint", "f" * 64), **kwargs)


def test_a_healthy_record_verifies(tmp_path):
    """The control. Without this the tests below could pass by always refusing."""
    record, stage, restart, trajectory, _ = _record(tmp_path)
    assert _verify(record, stage, tmp_path, restart, trajectory) == []


def test_a_fingerprint_that_does_not_match_is_refused(tmp_path):
    """THE baseline failure: the recorded fingerprint was never compared to anything.

    A rebuilt `built.xml`, a topology from another structure, or an edited `resolved.config` all
    change what this stage IS while leaving the log's copy of the stage dictionary identical.
    The baseline skipped the stage; the next one then continued from a restart belonging to a
    different System.
    """
    record, stage, restart, trajectory, _ = _record(tmp_path, fingerprint="a" * 64)
    problems = _verify(record, stage, tmp_path, restart, trajectory, fingerprint="b" * 64)
    assert any("fingerprint does not match" in p for p in problems), problems


def test_a_truncated_trajectory_is_refused(tmp_path):
    """`exists()` said yes. The digest says the file is not the one that was written."""
    record, stage, restart, trajectory, _ = _record(tmp_path)
    trajectory.write_bytes(trajectory.read_bytes()[:64])
    problems = _verify(record, stage, tmp_path, restart, trajectory)
    assert any("cMD.dcd" in p for p in problems), problems


def test_a_state_csv_cut_in_half_is_refused(tmp_path):
    """The state CSV was not in the record at all, so nothing could look at it."""
    record, stage, restart, trajectory, state_csv = _record(tmp_path)
    state_csv.write_text('#"Step"\n1,-1.0\n', encoding="utf-8")
    problems = _verify(record, stage, tmp_path, restart, trajectory)
    assert any("cMD.csv" in p for p in problems), problems


def test_a_deleted_output_is_refused_by_name(tmp_path):
    record, stage, restart, trajectory, _ = _record(tmp_path)
    trajectory.unlink()
    problems = _verify(record, stage, tmp_path, restart, trajectory)
    assert any("missing" in p and "cMD.dcd" in p for p in problems), problems


def test_a_record_with_no_output_manifest_is_not_accepted(tmp_path):
    """A legacy log cannot be verified, and 'cannot be verified' is not 'verified'."""
    record, stage, restart, trajectory, _ = _record(tmp_path)
    del record["outputs"]
    problems = _verify(record, stage, tmp_path, restart, trajectory)
    assert any("no verifiable output manifest" in p for p in problems), problems


def test_a_configuration_that_now_writes_a_stream_the_record_lacks_is_refused(tmp_path):
    """Turning on phase-space or CV reporting makes the previous run a different shape."""
    record, stage, restart, trajectory, _ = _record(tmp_path)
    stage = dict(stage, phase_space_interval_steps=50)
    problems = _verify(record, stage, tmp_path, restart, trajectory)
    assert any("phase_space" in p for p in problems), problems


def test_a_restart_from_a_different_molecule_is_refused(tmp_path):
    """The particle count is compared against the System this invocation prepared."""
    from openmm import System, XmlSerializer, unit
    from openmm.app import Simulation
    import openmm

    system = System()
    for _ in range(3):
        system.addParticle(1.0 * unit.amu)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * unit.nanometer)
    state = context.getState(getPositions=True, getVelocities=True)

    record, stage, restart, trajectory, _ = _record(tmp_path)
    restart.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    from md_tools.build.record import file_facts

    record["outputs"]["final_state"] = file_facts(restart)

    problems = _verify(record, stage, tmp_path, restart, trajectory, particles=99)
    assert any("particles" in p for p in problems), problems


def test_a_checkpoint_short_of_the_step_count_is_not_completion(tmp_path):
    """A stage can hold a valid committed generation and still not have finished."""
    from md_tools.openmm.checkpoint import commit_generation

    checkpoints = tmp_path / "cMD.checkpoints"
    commit_generation(
        checkpoints,
        write_checkpoint=lambda path: Path(path).write_bytes(b"state"),
        state={"fingerprint": "f" * 64, "steps_done": 40, "stage": "cMD", "streams": {}})

    record, stage, restart, trajectory, _ = _record(tmp_path)
    problems = _verify(record, stage, tmp_path, restart, trajectory, checkpoints=checkpoints)
    assert any("stops at step 40" in p for p in problems), problems


def test_a_missing_committed_pointer_is_not_completion(tmp_path):
    record, stage, restart, trajectory, _ = _record(tmp_path)
    problems = _verify(record, stage, tmp_path, restart, trajectory,
                       checkpoints=tmp_path / "absent.checkpoints")
    assert any("no committed checkpoint pointer" in p for p in problems), problems


def test_a_checkpoint_binary_edited_since_commit_is_refused(tmp_path):
    """`read_committed` verifies the digest; the verifier must surface that, not swallow it."""
    from md_tools.openmm.checkpoint import commit_generation

    checkpoints = tmp_path / "cMD.checkpoints"
    commit_generation(
        checkpoints,
        write_checkpoint=lambda path: Path(path).write_bytes(b"state"),
        state={"fingerprint": "f" * 64, "steps_done": 100, "stage": "cMD", "streams": {}})
    binary = next((checkpoints / "checkpoints").glob("*.chk"))
    binary.write_bytes(b"tampered")

    record, stage, restart, trajectory, _ = _record(tmp_path)
    problems = _verify(record, stage, tmp_path, restart, trajectory, checkpoints=checkpoints)
    assert any("not usable" in p or "sha256" in p for p in problems), problems


def test_committed_stream_counts_are_compared_with_the_files(tmp_path):
    """The generation vouches for counts. A stream that disagrees is not a completed stage."""
    from md_tools.openmm.checkpoint import commit_generation

    record, stage, restart, trajectory, state_csv = _record(tmp_path)
    checkpoints = tmp_path / "cMD.checkpoints"
    commit_generation(
        checkpoints,
        write_checkpoint=lambda path: Path(path).write_bytes(b"state"),
        state={"fingerprint": "f" * 64, "steps_done": 100, "stage": "cMD",
               "streams": {"state_csv": 99}})

    problems = _verify(record, stage, tmp_path, restart, trajectory, checkpoints=checkpoints,
                       streams={"state_csv": state_csv})
    assert any("vouches for 99" in p for p in problems), problems


def test_verification_is_read_only(tmp_path):
    """A verifier that repairs what it finds cannot be run twice with the same answer."""
    record, stage, restart, trajectory, state_csv = _record(tmp_path)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(tmp_path.iterdir())
              if p.is_file()}
    _verify(record, stage, tmp_path, restart, trajectory)
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(tmp_path.iterdir())
             if p.is_file()}
    assert before == after
