"""A restart is verified against the commit BEFORE anything about it is changed.

Item 3. The continuation used to do this:

    sim.currentStep = int(continuation["absolute_step"])
    verify_restart_matches_commit(sim, ...)

`Simulation.currentStep` is a property over `Context.getStepCount()` in OpenMM 8.5.2, so the
assignment called `Context.setStepCount` and overwrote the value the next line was about to check.
The verification then compared the committed step against itself and could never fail. A checkpoint
from the wrong generation -- an older one left in place, or one copied from another run -- loaded
without error and was accepted.

**A correction to the instruction's premise.** It states that an OpenMM State does not carry
`currentStep`. That was true of older OpenMM, where `Simulation.currentStep` was a plain attribute.
In OpenMM 8.5.2 `State` does expose `getStepCount()` and `Context.setState` restores it, which these
tests assert directly rather than take on trust. The step is therefore verified on BOTH paths, and
the commit is consulted for it only when a restart genuinely reports no step -- a case that is
handled and recorded rather than assumed away.
"""

from __future__ import annotations

import glob
import json
import os
import re
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
STREAMS = ("cMD_1_all_atoms.dcd", "cMD_1_selected_atoms.dcd", "cMD_1.log")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


@pytest.fixture(scope="module")
def two_generations(tmp_path_factory):
    """A run with TWO committed generations, so a stale restart is a real, available mistake."""
    bundle = tmp_path_factory.mktemp("rv_bundle") / "bundle"
    prepared = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(bundle), "--config",
                    str(REPO_ROOT / "test" / "ala" / "cMD" / "implicit" / "system_config.json"))
    if prepared.returncode != 0:
        pytest.skip(f"implicit preparation unavailable: {prepared.stderr[-300:]}")

    root = tmp_path_factory.mktemp("rv_project")
    (root / "md.json").write_text(json.dumps({
        "profile": "implicit-md-peptide-v1",
        "protocol": {
            "integrator": {"kind": "langevin-middle", "timestep": "2 fs",
                           "temperature": "300 K", "friction": "1 /ps"},
            "equilibration": {"protocol": "simple", "minimize_max_iterations": 100,
                              "restrained": "1 ps"},
            "production": {"method": "md", "duration_per_segment": "2 ps"}},
        "randomness": {"master_seed": 20260821},
        "execution": {"platform": "CPU",
                      "reporting": {"all_atom": "1 ps", "solute": "0.2 ps"}}}))
    project = root / "run"
    assert _run(INPUT_GEN, "--system", str(bundle / "system_manifest.json"),
                "-o", str(project), "--config", str(root / "md.json")).returncode == 0
    for stage in ("min", "eq"):
        assert subprocess.run([str(project / stage / f"{stage}.sh")], capture_output=True,
                              text=True, cwd=str(project / stage)).returncode == 0
    for _ in range(2):
        result = subprocess.run([str(project / "cMD_1" / "cMD_1.sh")], capture_output=True,
                                text=True, cwd=str(project / "cMD_1"))
        assert result.returncode == 0, result.stderr[-600:]
    return project


def _fresh(two_generations, tmp_path) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(two_generations, copy)
    return copy


def _fingerprint(project: Path) -> dict:
    stage = project / "cMD_1"
    return {name: (stage / name).read_bytes() for name in STREAMS if (stage / name).is_file()}


def _continue(project: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(project / "cMD_1" / "cMD_1.sh")], capture_output=True, text=True,
                          cwd=str(project / "cMD_1"))


def _committed(project: Path) -> dict:
    found = glob.glob(str(project / "cMD_1" / "run" / "**" / "committed.json"), recursive=True)
    return json.loads(Path(found[0]).read_text())


def _generation_dirs(project: Path) -> list[Path]:
    return sorted((project / "cMD_1" / "run" / "restart").glob("gen_*"))


# ---------------------------------------------------------------------------------------------
# what OpenMM actually restores -- asserted, not assumed
# ---------------------------------------------------------------------------------------------

def test_currentstep_is_a_property_over_the_context_step_count():
    """This is why assigning it before verifying destroyed the check."""
    from openmm import app

    assert isinstance(app.Simulation.__dict__.get("currentStep"), property)


def test_both_restart_forms_carry_step_and_time_in_this_openmm():
    """The premise the fallback logic rests on, measured rather than trusted."""
    import numpy as np
    import openmm
    from openmm import XmlSerializer, app, unit

    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0 * unit.dalton)
    nonbonded = openmm.NonbondedForce()
    for _ in range(3):
        nonbonded.addParticle(0.0, 0.3 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    system.addForce(nonbonded)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    for i in range(3):
        topology.addAtom(f"C{i}", app.element.carbon, residue)

    def build():
        integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                     0.002 * unit.picosecond)
        sim = app.Simulation(topology, system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
        sim.context.setPositions(np.random.RandomState(1).rand(3, 3) * unit.nanometer)
        return sim

    source = build()
    source.step(500)
    assert source.context.getStepCount() == 500

    from_state = build()
    assert from_state.context.getStepCount() == 0
    from_state.context.setState(XmlSerializer.deserialize(XmlSerializer.serialize(
        source.context.getState(getPositions=True, getVelocities=True, getParameters=True))))
    assert from_state.context.getStepCount() == 500, (
        "a serialized State did not restore the step count in this OpenMM build")


# ---------------------------------------------------------------------------------------------
# the verifier itself
# ---------------------------------------------------------------------------------------------

def test_the_verifier_reads_a_captured_position_rather_than_the_live_context():
    """So a caller cannot accidentally verify a value it has already overwritten."""
    from md_templates.openmm.cmd_segments import verify_restart_matches_commit

    commit = {"absolute_step": 2000, "absolute_time_ps": 4.0}
    good = {"loaded_step": 2000, "loaded_time_ps": 4.0}
    assert verify_restart_matches_commit(None, commit, position=good)["loaded_step"] == 2000

    for wrong in ({"loaded_step": 1000, "loaded_time_ps": 4.0},
                  {"loaded_step": 2000, "loaded_time_ps": 2.0}):
        with pytest.raises(RuntimeError, match="does not match the committed generation"):
            verify_restart_matches_commit(None, commit, position=wrong)


def test_a_restart_reporting_no_step_is_treated_as_carrying_none_not_as_a_mismatch():
    from md_templates.openmm.cmd_segments import verify_restart_matches_commit

    verified = verify_restart_matches_commit(
        None, {"absolute_step": 2000, "absolute_time_ps": 4.0},
        position={"loaded_step": 0, "loaded_time_ps": 4.0})
    assert verified["step_carried_by_restart"] is False
    # time still had to match: that is what makes trusting the commit for the step safe
    with pytest.raises(RuntimeError, match="time"):
        verify_restart_matches_commit(
            None, {"absolute_step": 2000, "absolute_time_ps": 4.0},
            position={"loaded_step": 0, "loaded_time_ps": 9.0})


def test_the_step_is_taken_from_the_commit_only_when_the_restart_carried_none():
    from md_templates.openmm.cmd_segments import restore_restart_step

    class FakeContext:
        def __init__(self):
            self.step = None

        def setStepCount(self, value):
            self.step = value

    class FakeSim:
        def __init__(self):
            self.context = FakeContext()

    carried = FakeSim()
    provenance = restore_restart_step(
        carried, {"expected_step": 2000, "step_carried_by_restart": True, "loaded_time_ps": 4.0},
        restart_source="checkpoint")
    assert carried.context.step is None, "the step was rewritten even though the restart carried it"
    assert provenance["step_origin"].startswith("restart")
    assert provenance["bitwise_continuation"] is True

    absent = FakeSim()
    provenance = restore_restart_step(
        absent, {"expected_step": 2000, "step_carried_by_restart": False, "loaded_time_ps": 4.0},
        restart_source="state")
    assert absent.context.step == 2000
    assert "committed.json" in provenance["step_origin"]
    assert provenance["bitwise_continuation"] is False
    assert "not bitwise identical" in provenance["note"].lower()


# ---------------------------------------------------------------------------------------------
# end to end: wrong restarts are refused before output is touched
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_loadable_but_stale_checkpoint_is_refused(two_generations, tmp_path):
    """The regression. This checkpoint loads perfectly; it is simply the previous generation."""
    project = _fresh(two_generations, tmp_path)
    generations = _generation_dirs(project)
    assert len(generations) >= 2, generations
    older, newest = generations[-2], generations[-1]
    checkpoint = next(p for p in newest.iterdir() if p.suffix == ".chk")
    shutil.copy2(next(p for p in older.iterdir() if p.suffix == ".chk"), checkpoint)

    before = _fingerprint(project)
    record = _committed(project)
    result = _continue(project)
    assert result.returncode != 0, "a checkpoint from the previous generation was accepted"
    assert "does not match the committed generation" in (result.stdout + result.stderr)
    assert _fingerprint(project) == before, "outputs were touched before the refusal"
    assert _committed(project)["invocations_completed"] == record["invocations_completed"]


@pytest.mark.slow
def test_a_state_with_the_wrong_time_is_refused(two_generations, tmp_path):
    """Checkpoint unusable so the State path is taken, and that State is from the wrong point."""
    project = _fresh(two_generations, tmp_path)
    newest = _generation_dirs(project)[-1]
    next(p for p in newest.iterdir() if p.suffix == ".chk").write_bytes(b"not a checkpoint")
    state_path = next(p for p in newest.iterdir() if p.suffix == ".xml")
    text = state_path.read_text()
    match = re.search(r'time="([0-9.eE+-]+)"', text)
    assert match, "the serialized State has no time attribute to corrupt"
    wrong = float(match.group(1)) + 17.0
    state_path.write_text(text.replace(match.group(0), f'time="{wrong}"', 1))

    before = _fingerprint(project)
    result = _continue(project)
    assert result.returncode != 0, "a State from the wrong point in time was accepted"
    assert "does not match the committed generation" in (result.stdout + result.stderr)
    assert _fingerprint(project) == before, "outputs were touched before the refusal"


@pytest.mark.slow
def test_a_corrupt_checkpoint_falls_back_to_the_state_and_records_the_provenance(
        two_generations, tmp_path):
    """The success case, with the exact step and time provenance asserted."""
    project = _fresh(two_generations, tmp_path)
    before = _committed(project)
    newest = _generation_dirs(project)[-1]
    next(p for p in newest.iterdir() if p.suffix == ".chk").write_bytes(b"not a checkpoint")

    result = _continue(project)
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]

    record = _committed(project)
    assert record["restart_source_for_this_segment"] == "state"
    assert record["invocations_completed"] == before["invocations_completed"] + 1

    provenance = record["restart_provenance"]
    assert provenance["restart_source"] == "state"
    assert provenance["bitwise_continuation"] is False
    assert "not bitwise identical" in provenance["note"].lower()
    assert provenance["started_absolute_step"] == before["absolute_step"]
    assert provenance["started_absolute_time_ps"] == pytest.approx(
        before["absolute_time_ps"], abs=1e-6)

    latest = record["invocation_history"][-1]
    assert latest["started_absolute_step"] == before["absolute_step"]
    assert latest["ended_absolute_step"] == record["absolute_step"]
    assert record["absolute_step"] > before["absolute_step"]
    assert "state" in (result.stdout + result.stderr).lower()


@pytest.mark.slow
def test_a_normal_continuation_prefers_the_checkpoint_and_is_bitwise(two_generations, tmp_path):
    project = _fresh(two_generations, tmp_path)
    assert _continue(project).returncode == 0
    record = _committed(project)
    assert record["restart_source_for_this_segment"] == "checkpoint"
    provenance = record["restart_provenance"]
    assert provenance["bitwise_continuation"] is True
    assert provenance["step_origin"].startswith("restart"), (
        "a healthy checkpoint should supply its own step, not have one written over it")
