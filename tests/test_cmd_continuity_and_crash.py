"""cMD continuity identity, committed-output integrity, and crash/fallback recovery.

Findings 2, 4 and 5.

Every refusal test asserts the outputs are **byte-for-byte unchanged** afterwards. That is the
property that matters: a refusal that happens after the trajectory has been opened for append has
already done the damage it was supposed to prevent, and it would still look like a clean error.

The interruption tests kill a real subprocess at chosen points rather than simulating a crash in
process, because what is being tested is what survives on disk when the writer stops existing.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
INPUT_GEN = REPO_ROOT / "MD_input_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
               / "ace_ala_nme.pdb")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


def _tiny_config() -> dict:
    return {
        "profile": "implicit-md-peptide-v1",
        "protocol": {
            "integrator": {"kind": "langevin-middle", "timestep": "2 fs",
                           "temperature": "300 K", "friction": "1 /ps"},
            "equilibration": {"protocol": "simple", "minimize_max_iterations": 100,
                              "restrained": "1 ps"},
            "production": {"method": "md", "duration_per_segment": "2 ps"},
        },
        "randomness": {"master_seed": 20260821},
        "execution": {"platform": "CPU", "precision": "mixed",
                      "reporting": {"all_atom": "1 ps", "solute": "0.2 ps"}},
    }


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("cc_bundle") / "bundle"
    config = REPO_ROOT / "test" / "ala" / "cMD" / "implicit" / "system_config.json"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    if result.returncode != 0:
        pytest.skip(f"implicit preparation unavailable: {result.stderr[-300:]}")
    return out


@pytest.fixture(scope="module")
def committed_project(bundle, tmp_path_factory):
    """A project with ONE committed cMD segment, ready to be continued or broken."""
    root = tmp_path_factory.mktemp("cc_project")
    config = root / "md.json"
    config.write_text(json.dumps(_tiny_config()))
    project = root / "run"
    assert _run(INPUT_GEN, "--system", str(bundle / "system_manifest.json"),
                "-o", str(project), "--config", str(config)).returncode == 0

    for stage in ("min", "eq"):
        result = subprocess.run([str(project / stage / f"{stage}.sh")], capture_output=True,
                                text=True, cwd=str(project / stage))
        assert result.returncode == 0, f"{stage}: {result.stderr[-400:]}"
    result = subprocess.run([str(project / "cMD_1" / "cMD_1.sh")], capture_output=True, text=True,
                            cwd=str(project / "cMD_1"))
    assert result.returncode == 0, result.stderr[-600:]
    return project


def _fresh(committed_project, tmp_path) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(committed_project, copy)
    return copy


def _fingerprint(project: Path) -> dict:
    """Bytes of every output stream, so a refusal can be proven not to have touched them."""
    stage = project / "cMD_1"
    return {name: (stage / name).read_bytes()
            for name in ("cMD_1_all_atoms.dcd", "cMD_1_selected_atoms.dcd", "cMD_1.log")
            if (stage / name).is_file()}


def _continue(project: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(project / "cMD_1" / "cMD_1.sh")], capture_output=True, text=True,
                          cwd=str(project / "cMD_1"))


def _committed(project: Path) -> dict:
    path = glob.glob(str(project / "cMD_1" / "run" / "**" / "committed.json"), recursive=True)[0]
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------
# Finding 2 -- continuity binds identity
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_single_changed_byte_in_system_xml_refuses_before_touching_output(
        committed_project, tmp_path):
    """Counts do not identify a System. One byte does."""
    project = _fresh(committed_project, tmp_path)
    before = _fingerprint(project)

    system_xml = project / "inputs" / "system.xml"
    text = system_xml.read_text()
    marker = 'openmmVersion="'
    assert marker in text
    system_xml.write_text(text.replace(marker, 'openmmVersion="9', 1))

    result = _continue(project)
    assert result.returncode != 0, "a changed System was accepted as a continuation"
    assert _fingerprint(project) == before, "outputs were modified before the refusal"


@pytest.mark.slow
@pytest.mark.parametrize("field,value", [
    ("temperature", "310 K"),
    ("timestep", "4 fs"),
])
def test_changing_the_integrator_refuses_before_touching_output(committed_project, tmp_path,
                                                                field, value):
    project = _fresh(committed_project, tmp_path)
    before = _fingerprint(project)

    payload_path = project / "cMD_1" / "cMD_1.json"
    payload = json.loads(payload_path.read_text())
    payload["integrator"][field] = value
    payload_path.write_text(json.dumps(payload, indent=2))

    result = _continue(project)
    assert result.returncode != 0, f"a changed {field} was accepted"
    assert _fingerprint(project) == before


@pytest.mark.slow
def test_changing_the_reporting_cadence_refuses_before_touching_output(committed_project,
                                                                       tmp_path):
    project = _fresh(committed_project, tmp_path)
    before = _fingerprint(project)

    payload_path = project / "cMD_1" / "cMD_1.json"
    payload = json.loads(payload_path.read_text())
    payload["reporting"]["selected_atoms_interval_steps"] = 50
    payload_path.write_text(json.dumps(payload, indent=2))

    result = _continue(project)
    assert result.returncode != 0
    assert _fingerprint(project) == before


@pytest.mark.slow
def test_changing_the_segment_length_refuses_but_the_segment_count_does_not(committed_project,
                                                                            tmp_path):
    """Segment LENGTH is the calculation; segment COUNT is how long you run it."""
    project = _fresh(committed_project, tmp_path)
    payload_path = project / "cMD_1" / "cMD_1.json"
    payload = json.loads(payload_path.read_text())
    payload["steps"] = payload["steps"] * 2
    payload_path.write_text(json.dumps(payload, indent=2))
    assert _continue(project).returncode != 0, "a changed segment length was accepted"

    # the count lives in Bash and is not in the contract at all
    contract = _committed(committed_project)["continuity"]
    assert "number_of_segments" not in json.dumps(contract)

    clean = _fresh(committed_project, tmp_path / "clean")
    before_hash = _committed(clean)["continuity_hash"]
    assert _continue(clean).returncode == 0, "a plain continuation was refused"
    assert _committed(clean)["continuity_hash"] == before_hash, (
        "running another segment changed the continuity hash")


def test_the_atom_identity_fingerprint_distinguishes_a_permuted_selection():
    """Equal-length selections must not alias, which a length or a count cannot prevent."""
    from openmm import app

    from md_templates.openmm.stage import _atom_identity

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("ALA", chain)
    for name in ("N", "CA", "C"):
        topology.addAtom(name, app.element.nitrogen if name == "N" else app.element.carbon,
                         residue)

    straight = _atom_identity(topology, [0, 1, 2])
    permuted = _atom_identity(topology, [2, 1, 0])
    subset = _atom_identity(topology, [0, 1])
    assert straight != permuted, "a permuted selection produced the same fingerprint"
    assert straight != subset
    assert _atom_identity(topology, [0, 1, 2]) == straight, "the fingerprint is not deterministic"


# ---------------------------------------------------------------------------------------------
# Finding 4 -- short or missing committed output is corruption
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("victim", ["cMD_1_all_atoms.dcd", "cMD_1_selected_atoms.dcd",
                                    "cMD_1.log"])
def test_a_deleted_committed_stream_refuses_and_leaves_the_others_untouched(
        committed_project, tmp_path, victim):
    project = _fresh(committed_project, tmp_path)
    before = {k: v for k, v in _fingerprint(project).items() if k != victim}
    (project / "cMD_1" / victim).unlink()

    result = _continue(project)
    assert result.returncode != 0, f"a missing {victim} was accepted as a continuation"
    assert "missing history" in result.stderr or "absent" in result.stderr, result.stderr[-400:]

    after = {k: v for k, v in _fingerprint(project).items() if k != victim}
    assert after == before, "another stream was mutated while refusing"


@pytest.mark.slow
def test_a_truncated_committed_trajectory_refuses(committed_project, tmp_path):
    """Shorter than the watermark is missing history: the bytes are gone, not pending."""
    from md_templates.openmm.dcdtail import truncate_to_frames

    project = _fresh(committed_project, tmp_path)
    truncate_to_frames(project / "cMD_1" / "cMD_1_selected_atoms.dcd", 1)
    before = _fingerprint(project)

    result = _continue(project)
    assert result.returncode != 0
    assert "missing history" in result.stderr, result.stderr[-400:]
    assert _fingerprint(project) == before


@pytest.mark.slow
def test_a_malformed_committed_log_refuses(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    log = project / "cMD_1" / "cMD_1.log"
    log.write_text(log.read_text() + "not,a,valid,row\n")
    before = _fingerprint(project)

    result = _continue(project)
    assert result.returncode != 0
    assert _fingerprint(project) == before


def test_a_zero_watermark_allows_an_absent_stream(tmp_path):
    """Nothing committed yet is the one case where absence is correct."""
    from md_templates.openmm.cmd_segments import inspect_committed_outputs

    problems = inspect_committed_outputs([
        {"name": "all-atom", "path": tmp_path / "absent.dcd", "watermark": 0, "kind": "dcd",
         "n_atoms": 10},
    ])
    assert problems == []


# ---------------------------------------------------------------------------------------------
# Finding 5 -- one atomic authority, and the cMD State fallback
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_commit_record_carries_the_invocation_history(committed_project):
    """The invocation is published by the commit, so a crash cannot leave a phantom one."""
    record = _committed(committed_project)
    assert record["cmd_schema_version"] >= 2
    history = record["invocation_history"]
    assert len(history) == record["invocations_completed"] == record["generation"]
    entry = history[-1]
    assert entry["ended_absolute_step"] == record["absolute_step"]
    assert entry["started_absolute_step"] < entry["ended_absolute_step"]
    assert entry["restart_source"] in ("predecessor state", "checkpoint", "state")


@pytest.mark.slow
def test_a_forced_state_fallback_continues_and_says_so(committed_project, tmp_path):
    """The cMD-specific fallback test. A REST2 fallback test does not cover this path.

    The checkpoint is corrupted while the serialized State stays valid, which is exactly the
    situation the State exists for: a checkpoint is platform- and System-specific, so it is the
    form that goes bad first.
    """
    project = _fresh(committed_project, tmp_path)
    before = _committed(project)

    checkpoint = Path(glob.glob(str(project / "cMD_1" / "run" / "**" / "walker.chk"),
                                recursive=True)[0])
    state = checkpoint.with_name("walker.state.xml")
    assert state.is_file(), "the portable State was not written beside the checkpoint"
    checkpoint.write_bytes(b"this is not a checkpoint")

    result = _continue(project)
    assert result.returncode == 0, result.stderr[-800:]
    assert "falling back to the serialized State" in (result.stdout + result.stderr), (
        "the fallback was taken silently")

    after = _committed(project)
    assert after["generation"] == before["generation"] + 1
    assert after["restart_source_for_this_segment"] == "state"
    assert after["absolute_step"] > before["absolute_step"]
    assert after["invocation_history"][-1]["started_absolute_step"] == before["absolute_step"], (
        "the fallback segment did not start from the committed boundary")


@pytest.mark.slow
def test_an_uncommitted_tail_is_removed_and_the_run_continues(committed_project, tmp_path):
    """A crash after some frames but before the commit leaves frames nothing accounts for."""
    from md_templates.openmm.dcdtail import frame_offsets

    project = _fresh(committed_project, tmp_path)
    record = _committed(project)
    trajectory = project / "cMD_1" / "cMD_1_selected_atoms.dcd"

    # forge an uncommitted tail by appending a copy of the file's own last frame
    scan = frame_offsets(trajectory)
    committed_frames = record["watermarks"]["selected_atoms"]
    assert scan["complete_frames"] == committed_frames
    data = trajectory.read_bytes()
    last_frame = data[scan["boundaries"][-2]:scan["boundaries"][-1]]
    trajectory.write_bytes(data + last_frame)
    assert frame_offsets(trajectory)["complete_frames"] == committed_frames + 1

    result = _continue(project)
    assert result.returncode == 0, result.stderr[-800:]

    after = _committed(project)
    scan_after = frame_offsets(trajectory)
    assert scan_after["complete_frames"] == after["watermarks"]["selected_atoms"], (
        "the tail was appended after rather than removed first")


@pytest.mark.slow
@pytest.mark.parametrize("kill_after_seconds", [0.6, 1.2])
def test_killing_a_segment_mid_flight_leaves_the_committed_physics_intact(
        committed_project, tmp_path, kill_after_seconds):
    """A real subprocess is killed, because the question is what survives on disk.

    Whatever the timing hits -- mid-dynamics, after reporters close, between restart members -- the
    committed generation must still describe the last complete boundary, and the next invocation
    must continue from it without a duplicated or skipped index.
    """
    import time

    project = _fresh(committed_project, tmp_path)
    before = _committed(project)

    # The launcher is a shell that execs Python. Killing the Popen kills only the SHELL, and the
    # Python child keeps running -- which is how this test first produced a committed history of
    # [1000, 2000, 2000]: the "crashed" segment and its retry ran concurrently on one run
    # directory. A crash must take the whole process group with it.
    process = subprocess.Popen([str(project / "cMD_1" / "cMD_1.sh")],
                               cwd=str(project / "cMD_1"), start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(kill_after_seconds)
    os.killpg(os.getpgid(process.pid), 9)
    process.wait(timeout=30)

    during = _committed(project)
    assert during["generation"] == before["generation"], (
        "a killed segment published a generation it never finished")
    assert during["absolute_step"] == before["absolute_step"]

    result = _continue(project)
    assert result.returncode == 0, result.stderr[-800:]
    after = _committed(project)
    assert after["generation"] == before["generation"] + 1, "the retry duplicated or skipped an index"
    assert after["absolute_step"] == before["absolute_step"] + after["steps_per_segment"]
    steps = [entry["ended_absolute_step"] for entry in after["invocation_history"]]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), (
        f"invocation history is not monotonic: {steps}")


@pytest.mark.slow
def test_two_invocations_cannot_share_one_run_directory(committed_project, tmp_path):
    """The hazard the crash test exposed: concurrent writers on one committed history.

    Both would restore the same committed generation, both append to the same trajectories and both
    commit. Every file would still look individually well-formed, which is why this has to be
    prevented rather than detected afterwards.
    """
    from md_templates.openmm.cmd_segments import RunDirectoryBusy, hold_run_lock

    project = _fresh(committed_project, tmp_path)
    run_dir = project / "cMD_1" / "run"

    holder = hold_run_lock(run_dir)
    try:
        with pytest.raises(RunDirectoryBusy, match="another process is writing"):
            hold_run_lock(run_dir)
        # and a real second invocation refuses rather than interleaving
        result = _continue(project)
        assert result.returncode != 0
        assert "another process is writing" in result.stderr
    finally:
        holder.close()

    # once released, the run continues normally
    assert _continue(project).returncode == 0
