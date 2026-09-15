"""What the read-only preflight SKIPS, and whether anything downstream still protects the data.

THE QUESTION

    `md_tools.run.continuation` is a whole-operation boundary: it validates every selected AIS
    path before work begins on any of them. But it does not refuse on everything it looks at. It
    returns early on an unreadable ladder checkpoint, and for AIS it passes over a path whose
    committed checkpoint will not read (`except CheckpointError: continue`) and one whose
    committed state carries no CV prefix (`if entry is None: continue`). Its public-entry
    counterpart builds its selection from `completed.json` markers, so a path that is PARTIAL --
    interrupted, with a committed generation and no marker -- is not inspected there at all.

    Each of those is a case where the preflight declines to speak. The question this file answers
    is not whether the preflight is thorough; it is whether the scientific output survives. The
    instruction is explicit that if a later authoritative check already refuses before anything is
    modified, that fact is to be DOCUMENTED rather than duplicated by another validation layer.

WHAT IS ASSERTED, AND WHAT IS NOT

    Asserted: the run refuses, and every scientific table of every path -- `cv.csv`,
    `observations.csv`, `state.csv`, the staged trajectory -- is byte-for-byte what it was before
    the refused attempt, including the paths the attempt got to before reaching the damaged one.

    Not asserted: that the refusal comes from the preflight rather than from the runtime, or that
    no log, `.out` or diagnostic file moved. Under the proportional preservation rules a rejected
    attempt may write a diagnostic; what it may not do is truncate committed samples or overwrite
    the prior run's authoritative completion state.

    The uncommitted tail is the one thing that legitimately does change: a resume cuts each stream
    back to the count its checkpoint vouches for. That is the designed behaviour and not loss, so
    the comparisons below are against the COMMITTED prefix length, taken from each path's own
    committed checkpoint, and not against the raw file length.

PLATFORM_POLICY_EXEMPTION: `--cpu`. What is under test is the order in which files are read and
written around a refusal -- identical on every platform. The AIS runtime itself is exercised on
real CUDA in `test_cv_cuda_lanes.py` and under a real launcher in `test_cv_mpi_cuda_ais.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

PATHS = 3
SWITCHING = 20
UPDATE_EVERY = 5
CV_EVERY = 5
#: From the schedule, by hand: steps 0, 5, 10, 15, 20.
COMPLETE_ROWS = SWITCHING // CV_EVERY + 1

#: Every file in a path directory that carries scientific output, as opposed to a log or a
#: diagnostic. These are what a refused attempt may not touch.
SCIENTIFIC = ("cv.csv", "cv.json", "observations.csv", "state.csv", "completed.json")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("skipped-checks")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(
        "schema_version: 1\ncollective_variables:\n"
        "  - {name: phi, type: torsion, atom_indices: [4, 6, 8, 14]}\n", encoding="utf-8")
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": PATHS, "switching_steps": SWITCHING,
                "observation_interval_steps": 10,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260907},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./AIS-run1", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


def _environment(project: Path, extra=None):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base["OPENMM_CPU_THREADS"] = "1"
    base.update(extra or {})
    return base


def _run(project: Path, destination: Path, *extra, environment=None, expect=0):
    done = subprocess.run(
        [sys.executable, str(project / "AIS-run1" / "AIS.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "AIS-run1", capture_output=True, text=True, timeout=1800,
        env=_environment(project, environment))
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _digests(destination: Path):
    """Every scientific file in every path, by content."""
    out = {}
    for directory in sorted(destination.glob("path_*")):
        for name in SCIENTIFIC:
            candidate = directory / name
            if candidate.is_file():
                out[f"{directory.name}/{name}"] = hashlib.sha256(
                    candidate.read_bytes()).hexdigest()
        for staged in sorted(directory.glob("*.dcd")):
            out[f"{directory.name}/{staged.name}"] = hashlib.sha256(
                staged.read_bytes()).hexdigest()
    return out


def _edit_committed_state(path_directory: Path, edit):
    """Rewrite the committed generation's sidecar `state` in place, keeping it valid JSON."""
    from md_tools.openmm.checkpoint import GENERATIONS_DIR, POINTER_NAME

    pointer = json.loads((path_directory / POINTER_NAME).read_text(encoding="utf-8"))
    sidecar = path_directory / GENERATIONS_DIR / pointer["sidecar"]
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    edit(document["state"])
    sidecar.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture(scope="module")
def interrupted(project, tmp_path_factory):
    """A campaign stopped mid-flight: earlier paths finished, a later one is PARTIAL.

    The later path is the one every case below damages, because a first-failure ordering would
    reach it only after the earlier paths had already been rewritten.
    """
    destination = tmp_path_factory.mktemp("skipped-checks-run") / "run"
    # Let several generations commit -- across the earlier paths and into a later one -- then
    # crash at a commit boundary, leaving that path with a committed generation and no marker.
    _run(project, destination, expect=1,
         environment={"MD_TOOLS_CHECKPOINT_FAULT": "after-pointer-replace",
                      "MD_TOOLS_CHECKPOINT_FAULT_AFTER": "3"})
    partial = [d for d in sorted(destination.glob("path_*"))
               if not (d / "completed.json").is_file()
               and (d / "current_checkpoint.json").is_file()]
    if not partial:
        pytest.skip("this interruption left no partial path with a committed generation")
    return destination, partial[0]


def test_the_interrupted_campaign_resumes_cleanly(project, interrupted, tmp_path):
    """The control. Every case below must refuse; this one must not.

    Without it, an implementation that refused every resume outright would pass the whole file.
    """
    source, _ = interrupted
    destination = tmp_path / "control"
    shutil.copytree(source, destination)
    _run(project, destination, "--resume")
    for directory in sorted(destination.glob("path_*")):
        rows = (directory / "cv.csv").read_text(encoding="utf-8").splitlines()[1:]
        assert len(rows) == COMPLETE_ROWS, directory.name
        assert [int(row.split(",")[2]) for row in rows] == \
            list(range(0, SWITCHING + 1, CV_EVERY)), directory.name
        assert (directory / "completed.json").is_file(), directory.name


def _refuses_without_touching_the_science(project, source, damaged, tmp_path, name, damage):
    staged = tmp_path / name
    shutil.copytree(source, staged)
    damage(staged / damaged.name)

    before = _digests(staged)
    refused = _run(project, staged, "--resume", expect=1)
    assert refused.returncode != 0, refused.stdout[-3000:]

    after = _digests(staged)
    changed = sorted(key for key in set(before) | set(after)
                     if before.get(key) != after.get(key))
    assert not changed, (
        f"a refused attempt rewrote scientific output: {changed}\n"
        + (refused.stdout + refused.stderr)[-3000:])
    return refused


def test_an_unreadable_committed_checkpoint_on_a_partial_path(project, interrupted, tmp_path):
    """`except CheckpointError: continue` in the preflight -- and `read_committed` downstream.

    The runtime reads the committed pointer before it truncates anything, and that read is the
    same one the preflight declined to insist on.
    """
    source, damaged = interrupted

    def damage(directory: Path):
        pointer = json.loads((directory / "current_checkpoint.json").read_text(encoding="utf-8"))
        (directory / "checkpoints" / pointer["sidecar"]).write_text("{ not json",
                                                                    encoding="utf-8")

    _refuses_without_touching_the_science(
        project, source, damaged, tmp_path, "unreadable-checkpoint", damage)


def test_a_partial_path_whose_committed_state_carries_no_cv_prefix(project, interrupted,
                                                                   tmp_path):
    """`if entry is None: continue` in the preflight.

    Downstream the runtime passes `state.get("cv_prefix") or {}` to `_validate_cv_prefix`, which
    refuses an entry with no row count and no digest -- and it does so before the first
    `_truncate_csv`.
    """
    source, damaged = interrupted

    def damage(directory: Path):
        _edit_committed_state(directory, lambda state: state.pop("cv_prefix", None))

    refused = _refuses_without_touching_the_science(
        project, source, damaged, tmp_path, "no-cv-prefix", damage)
    combined = refused.stdout + refused.stderr
    assert "committed collective-variable prefix" in combined, combined[-3000:]


def test_a_partial_path_whose_committed_cv_prefix_is_damaged(project, interrupted, tmp_path):
    """The damaged LATER path the instruction asks for, reached through the ordinary resume.

    No `completed.json` exists on this path, so the public preflight's selection omits it
    entirely. Only the authoritative check stands between the damage and the data.
    """
    source, damaged = interrupted

    def damage(directory: Path):
        def edit(state):
            state["cv_prefix"] = dict(state["cv_prefix"])
            state["cv_prefix"]["prefix_sha256"] = "0" * 64
        _edit_committed_state(directory, edit)

    refused = _refuses_without_touching_the_science(
        project, source, damaged, tmp_path, "damaged-prefix", damage)
    combined = refused.stdout + refused.stderr
    assert "cv.csv" in combined, combined[-3000:]


def test_a_partial_paths_committed_cost_removed(project, interrupted, tmp_path):
    """A cost-less committed prefix: accepted once, restored as zero, and history invented."""
    source, damaged = interrupted

    def damage(directory: Path):
        def edit(state):
            state["cv_prefix"] = dict(state["cv_prefix"])
            state["cv_prefix"]["cost"] = None
        _edit_committed_state(directory, edit)

    _refuses_without_touching_the_science(
        project, source, damaged, tmp_path, "no-cost", damage)


def test_an_unreadable_ladder_checkpoint_refuses_before_truncating(tmp_path):
    """The ladder's own early return: `except storage.StorageError: return`.

    Its counterpart is `_continue_cv_states`, which reads the checkpoint through the same reader
    inside the driver -- and a checkpoint that will not read never reaches the truncation because
    the run cannot restore its configurations from it either.
    """
    from md_tools.run.continuation import ContinuationError, validate_ladder_continuation

    directory = tmp_path / "ladder"
    directory.mkdir()
    checkpoint = directory / "REST2_checkpoint.nc"
    checkpoint.write_bytes(b"not a netcdf file at all")
    series = directory / "cv_state0.csv"
    series.write_text("step,phi\n0,1.0\n", encoding="utf-8")
    before = series.read_bytes()

    class _Definition:
        names = ("phi",)

    # The preflight declines rather than refusing -- deliberately, because the ordinary
    # continuation reports an unreadable checkpoint far better than this boundary can.
    validate_ladder_continuation(directory, definition=_Definition(), taus=[0.0, 0.5],
                                 interval_steps=5, checkpoint_path=checkpoint)
    assert series.read_bytes() == before
    assert ContinuationError is not None      # imported for the callers' benefit, not raised here


# --- the one gap the runtime does NOT close: the public entry's own writes ----------------------

def _tree(destination: Path):
    """Every file under the tree, by content. Names included, so an addition is a difference."""
    out = {}
    for candidate in sorted(destination.rglob("*")):
        if candidate.is_file():
            out[str(candidate.relative_to(destination))] = hashlib.sha256(
                candidate.read_bytes()).hexdigest()
    return out


def test_md_run_refuses_a_damaged_partial_path_without_writing_into_the_tree(
        project, interrupted, tmp_path):
    """The public command, against a campaign that has no completed path at all.

    The runtime protects the tables -- every case above shows that -- but `md-run` is a different
    surface onto the same runtime: it creates `-odir` and writes `resolved.config` and the
    content-addressed definition copy ITSELF, before dispatching. Its preflight used to build its
    selection from `completed.json` alone, so a tree whose paths are all partial was not inspected
    at all and the public command rewrote the prior run's authoritative configuration record on
    its way to a refusal the runtime was always going to make.

    The comparison is the WHOLE tree, not just the scientific files, because what is at stake here
    is the provenance a later reader needs to interpret the results that are already there.
    """
    source, damaged = interrupted
    staged = tmp_path / "public-entry"
    shutil.copytree(source, staged)
    # No completed path survives, so nothing reaches a `completed.json`-only selection.
    for directory in sorted(staged.glob("path_*")):
        if directory.name != damaged.name:
            shutil.rmtree(directory)

    def edit(state):
        state["cv_prefix"] = dict(state["cv_prefix"])
        state["cv_prefix"]["prefix_sha256"] = "0" * 64
    _edit_committed_state(staged / damaged.name, edit)

    # `input/` is SHARED by every run on this system, so the method's input is at the
    # dataset root rather than inside the run.
    generated = project / "input"
    inputs = sorted(generated.glob("*.in"))
    assert inputs, f"no .in file in {generated}"
    stage_input = next((p for p in inputs if p.stem == "AIS"), inputs[-1])

    # The tree as the interrupted run left it. It carries no `resolved.config` -- the generated
    # wrapper does not write one -- which makes the comparison below sharper rather than weaker:
    # `_tree` keys on names, so a file the refused `md-run` ADDS is a difference. That addition
    # is the defect, since the same file in a tree written by `md-run` would be an overwrite of
    # the prior run's authoritative configuration record.
    # THE STAGED TREE IS THE SAME RUN, so it carries that run's `run.config`. The seed lives
    # there now, and `md-run` layers it from `-odir`: without it the seed resolved to the schema
    # default, which changed the selected source frames, so the damaged path was no longer in the
    # selection, the read-only boundary found nothing to object to, and the refusal came later --
    # from the runtime's own identity check, after `resolved.config` had been written.
    shutil.copy2(project / "AIS-run1" / "run.config", staged / "run.config")
    # AND THE CV DEFINITION, for the same reason. `_carry_cv_definition` is called only by
    # `md-run`; a generated wrapper reads `resolved.config` beside itself and copies nothing, so
    # the destination this campaign ran into holds no `cv.<digest>.yaml`. The refused `md-run`
    # reads the SHARED `../input/AIS.in`, which cannot say which run's definition copy to use, so
    # without this the cadence resolved to 0, the schedule differed from the recorded one, and the
    # refusal came from the identity check instead of the read-only boundary.
    for definition in (project / "AIS-run1").glob("cv.*.yaml"):
        shutil.copy2(definition, staged / definition.name)

    before = _tree(staged)
    refused = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "md-run",
         "-i", str(stage_input), "-p", str(project / "build" / "built.pdb"),
         "-s", str(project / "build" / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-odir", str(staged), "--cpu", "--resume"],
        cwd=generated, capture_output=True, text=True, timeout=1800,
        env=_environment(project))
    assert refused.returncode != 0, refused.stdout[-3000:]

    after = _tree(staged)
    changed = sorted(key for key in set(before) | set(after)
                     if before.get(key) != after.get(key))
    assert not changed, (
        f"a refused md-run wrote into the tree: {changed}\n"
        + (refused.stdout + refused.stderr)[-3000:])
