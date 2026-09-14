"""Section 14: an extension writes a NEW output set and leaves its parent byte-for-byte unchanged.

`--extend N` lengthens a run in place. That is the right operation for "this run should have been
longer", and the wrong one for "here is a second segment, and the first must stay exactly as it
was". `--extend-from PARENT` is the second: the parent is opened read-only, everything that could
refuse is established before anything local is created, and the new dynamics land in their own
directory with absolute step and time coordinates continuing the parent's.

The whole contract in one line: an extension that could modify its parent is not an extension.

PLATFORM_POLICY_EXEMPTION: alanine dipeptide in vacuum on CPU, a few seconds. This tests the
boundary, not the physics; the scientific validation is the ALA production chain.
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

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"
#: The checkout's `src`, for subprocesses. NOT the package directory: putting that on
#: PYTHONPATH would let `remd/statistics.py` shadow the standard library.
SRC = Path(__file__).resolve().parents[1] / "src"
pytest.importorskip("netCDF4")
pytest.importorskip("openmm")

from md_tools.remd import amber_trajectory as amber
from md_tools.remd.executor import INTERRUPTED_STATUS

from .conftest import EXCHANGES, TAUS                               # noqa: E402

# Every output the parent owns. This list does two jobs -- it is what an in-place extension is
# seeded with, and it is what `_fingerprint` checks stayed byte-identical -- so a file missing
# from it is both uncopied and unchecked. The per-state names are DERIVED: spelling them out is
# how the solute streams came to be absent from both jobs after the rename, which made an
# in-place extension die on files the test had simply never copied.
PARENT_FILES = ("exchange.nc", "checkpoint.nc", "restart.json", "rem.log",
                "exchange.solute.nc", "exchange.runstate.json") + tuple(
    amber.state_trajectory_name(state, content=content)
    for content in ("whole", "solute") for state in range(len(TAUS)))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(directory):
    """Every output the parent owns, by content. This is what 'unchanged' is checked against."""
    return {name: _digest(Path(directory) / name) for name in PARENT_FILES
            if (Path(directory) / name).is_file()}


def _invoke(work, *extra):
    environment = dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", str(len(TAUS)),
         "-x", "exchange.nc", "-r", "restart.json", "--checkpoint", "checkpoint.nc",
         "-o", "run.out", "--rem", "rem.log", *extra],
        cwd=str(work), capture_output=True, text=True, timeout=1800, env=environment)


@pytest.fixture(scope="module")
def chain(prepared, tmp_path_factory):
    """A completed parent and one genuine extension of it, in separate directories."""
    source, n_atoms, _ = prepared
    root = tmp_path_factory.mktemp("chain")
    parent, extension = root / "REST2", root / "REST2_ext1"
    for directory in (parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)

    first = _invoke(parent)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    before = _fingerprint(parent)

    second = _invoke(extension, "--extend-from", str(parent), "--extend", str(EXCHANGES))
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    return parent, extension, before, n_atoms


# --- the parent is immutable ------------------------------------------------------------------

@pytest.mark.slow
def test_the_parent_is_byte_for_byte_unchanged(chain):
    parent, _, before, _ = chain
    after = _fingerprint(parent)
    assert after == before, {name for name in before if before[name] != after.get(name)}


@pytest.mark.slow
def test_the_extension_writes_its_own_complete_set(chain):
    _, extension, _, _ = chain
    for name in ("exchange.nc", "checkpoint.nc", "restart.json", "rem.log",
                 "whole_state0_prod1.nc", "whole_state1_prod1.nc"):
        assert (extension / name).is_file(), name


@pytest.mark.slow
def test_nothing_of_the_parent_was_copied_into_the_extension(chain):
    """A copied parent relabelled as an extension is exactly what section 14 forbids."""
    parent, extension, _, _ = chain
    for name in ("exchange.nc", "whole_state0_prod1.nc", "whole_state1_prod1.nc", "checkpoint.nc"):
        assert _digest(parent / name) != _digest(extension / name), name


# --- it is a continuation, not a rerun ----------------------------------------------------------

@pytest.mark.slow
def test_the_extension_holds_only_the_new_segment(chain):
    parent, extension, _, _ = chain
    parent_frames = amber.read_frames(parent / "whole_state0_prod1.nc")["n_frames"]
    extension_frames = amber.read_frames(extension / "whole_state0_prod1.nc")["n_frames"]
    assert extension_frames == parent_frames


@pytest.mark.slow
def test_step_and_time_coordinates_are_absolute_across_the_boundary(chain):
    """Section 14: the extension's trajectories hold only new frames, but their coordinates
    continue the parent's rather than restarting at zero."""
    parent, extension, _, _ = chain
    for index in range(len(TAUS)):
        parent_times = amber.read_frames(parent / f"whole_state{index}_prod1.nc")["times_ps"]
        child_times = amber.read_frames(extension / f"whole_state{index}_prod1.nc")["times_ps"]
        assert min(child_times) > max(parent_times), (parent_times[-1], child_times[0])
        step = parent_times[1] - parent_times[0]
        assert child_times[0] == pytest.approx(parent_times[-1] + step)


@pytest.mark.slow
def test_the_chain_is_twice_the_parent(chain):
    parent, extension, _, _ = chain
    before = json.loads((parent / "restart.json").read_text(encoding="utf-8"))
    after = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    assert after["extends"]["chain"]["production_ps"] == pytest.approx(
        2 * before["production_ps_per_replica"])
    assert after["extends"]["segment"]["production_ps"] == pytest.approx(
        before["production_ps_per_replica"])


@pytest.mark.slow
def test_the_new_segment_is_the_dynamics_an_in_place_extension_would_have_produced(
        prepared, tmp_path_factory):
    """The strongest statement of "genuine continuation" available: identical trajectories.

    `--extend N` in place is the operation whose correctness is already established. If the
    out-of-place extension truly continues the same physical state, with the same mapping, the
    same integrator seeding and the same exchange RNG, then the frames it produces must be the
    ones the in-place extension appends -- not merely similar, but the same numbers.

    A rerun from the input coordinates, a coordinate-only restart, or a lost RNG state each break
    this immediately, which is exactly what the comparison is for.
    """
    import numpy as np
    import netCDF4

    source, _, _ = prepared
    root = tmp_path_factory.mktemp("equivalence")
    inplace, parent, extension = root / "inplace", root / "REST2", root / "REST2_ext1"
    for directory in (inplace, parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)

    assert _invoke(parent).returncode == 0
    for name in PARENT_FILES:
        if (parent / name).is_file():
            shutil.copy(parent / name, inplace / name)

    assert _invoke(inplace, "--extend", str(EXCHANGES)).returncode == 0
    assert _invoke(extension, "--extend-from", str(parent),
                   "--extend", str(EXCHANGES)).returncode == 0

    for index in range(len(TAUS)):
        with netCDF4.Dataset(str(inplace / f"whole_state{index}_prod1.nc")) as handle:
            appended = np.array(handle.variables["coordinates"][:], dtype=float)
            appended_times = np.array(handle.variables["time"][:], dtype=float)
        with netCDF4.Dataset(str(extension / f"whole_state{index}_prod1.nc")) as handle:
            segment = np.array(handle.variables["coordinates"][:], dtype=float)
            segment_times = np.array(handle.variables["time"][:], dtype=float)
        tail = appended[-len(segment):]
        assert np.array_equal(tail, segment), (
            f"state {index}: the out-of-place segment is not the dynamics the in-place "
            f"extension produced (max difference {np.abs(tail - segment).max()})")
        assert np.array_equal(appended_times[-len(segment_times):], segment_times)


# --- provenance ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_extension_pins_its_parent_by_content(chain):
    parent, extension, _, _ = chain
    record = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    pinned = record["extends"]["parent"]
    assert pinned["path"] == str(parent.resolve())
    assert pinned["manifest"]["sha256"] == _digest(parent / "restart.json")
    assert pinned["checkpoint"]["sha256"] == _digest(parent / "checkpoint.nc")
    assert pinned["analysis"]["sha256"] == _digest(parent / "exchange.nc")
    for entry in pinned["state_trajectories"]:
        assert entry["present"]
        assert entry["sha256"] == _digest(parent / entry["name"])


@pytest.mark.slow
def test_segment_and_chain_acceptance_stay_distinguishable(chain):
    """Section 14 allows a cumulative figure and requires the segment-local one to survive."""
    parent, extension, _, _ = chain
    before = json.loads((parent / "restart.json").read_text(encoding="utf-8"))
    after = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    segment = after["extends"]["segment"]["neighbouring_acceptance"]
    whole = after["extends"]["chain"]["neighbouring_acceptance"]
    assert segment == after["completion_report"]["overall"]
    assert whole["proposed"] == (segment["proposed"]
                                 + before["completion_report"]["overall"]["proposed"])
    assert whole["accepted"] == (segment["accepted"]
                                 + before["completion_report"]["overall"]["accepted"])


@pytest.mark.slow
def test_the_exchange_numbering_across_the_boundary_is_recorded(chain):
    parent, extension, _, _ = chain
    before = json.loads((parent / "restart.json").read_text(encoding="utf-8"))
    after = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    assert (after["extends"]["segment"]["first_exchange_number"]
            == before["exchanges_committed"] + 1)


@pytest.mark.slow
def test_the_extension_rem_log_is_segment_local_and_self_consistent(chain):
    """Amber's convention: a restarted run's log numbers from 1 and `numexchg` counts its own
    blocks. cpptraj checks that count, so the file must agree with itself."""
    _, extension, _, _ = chain
    text = (extension / "rem.log").read_text(encoding="utf-8")
    assert "# exchange        1" in text
    declared = int([line for line in text.splitlines()
                    if line.startswith("# numexchg")][0].split()[-1])
    assert declared == text.count("# exchange ")


# --- cpptraj joins the two segments -------------------------------------------------------------

@pytest.mark.slow
def test_cpptraj_concatenates_the_segments_without_a_gap_or_a_duplicate(chain):
    parent, extension, _, n_atoms = chain
    cpptraj = shutil.which("cpptraj")
    if cpptraj is None:
        pytest.skip("cpptraj is not installed in this environment")
    work = parent.parent
    joined = work / "joined.nc"
    script = work / "join.in"
    script.write_text(
        f"parm {parent / 'topology.pdb'}\n"
        f"trajin {parent / 'whole_state0_prod1.nc'}\n"
        f"trajin {extension / 'whole_state0_prod1.nc'}\n"
        f"trajout {joined}\ngo\nquit\n", encoding="utf-8")
    result = subprocess.run([cpptraj, "-i", str(script)], capture_output=True, text=True,
                            cwd=str(work), timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    total = (amber.read_frames(parent / "whole_state0_prod1.nc")["n_frames"]
             + amber.read_frames(extension / "whole_state0_prod1.nc")["n_frames"])
    import netCDF4
    with netCDF4.Dataset(str(joined)) as handle:
        assert len(handle.dimensions["frame"]) == total


# --- refusals -----------------------------------------------------------------------------------

@pytest.mark.slow
def test_an_unfinished_parent_is_refused_before_anything_is_created(prepared, tmp_path_factory):
    """A parent that never completed is resumed in place, not extended. The refusal is read-only:
    no extension directory is left behind looking like a run."""
    source, _, _ = prepared
    root = tmp_path_factory.mktemp("unfinished")
    parent, extension = root / "REST2", root / "REST2_ext1"
    for directory in (parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)
    assert _invoke(parent).returncode == 0
    (parent / "restart.json").unlink()

    result = _invoke(extension, "--extend-from", str(parent), "--extend", str(EXCHANGES))
    assert result.returncode != 0
    assert not (extension / "exchange.nc").exists()
    assert not (extension / "whole_state0_prod1.nc").exists()


@pytest.mark.slow
def test_an_output_inside_the_parent_is_refused(prepared, tmp_path_factory):
    """The parent is immutable, so an output placed inside it would modify the run being
    continued. Caught in validation, before a byte is written anywhere."""
    source, _, _ = prepared
    root = tmp_path_factory.mktemp("inside")
    parent = root / "REST2"
    parent.mkdir()
    for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py", "ladder.group"):
        shutil.copy(source / name, parent / name)
        shutil.copy(source / name, root / name)
    assert _invoke(parent).returncode == 0
    before = _fingerprint(parent)

    environment = dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", str(len(TAUS)),
         "-x", str(parent / "inside.nc"), "-r", "restart2.json",
         "--checkpoint", "checkpoint2.nc", "-o", "run2.out",
         "--extend-from", str(parent), "--extend", str(EXCHANGES)],
        cwd=str(root), capture_output=True, text=True, timeout=900, env=environment)
    assert result.returncode != 0
    assert "immutable" in (result.stdout + result.stderr)
    assert _fingerprint(parent) == before


@pytest.mark.slow
def test_extend_from_without_extend_is_refused(prepared, tmp_path_factory):
    """How much new dynamics an extension adds is stated, never inherited."""
    source, _, _ = prepared
    root = tmp_path_factory.mktemp("nolength")
    parent, extension = root / "REST2", root / "REST2_ext1"
    for directory in (parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)
    assert _invoke(parent).returncode == 0

    result = _invoke(extension, "--extend-from", str(parent))
    assert result.returncode != 0
    assert "--extend N" in (result.stdout + result.stderr)


@pytest.mark.slow
def test_the_parent_file_names_are_read_from_its_manifest_not_assumed(prepared,
                                                                      tmp_path_factory):
    """A parent whose outputs are not named the way the documentation names them still extends.

    The generated ladders call theirs `rest2.nc` and `rest2_checkpoint.nc`. Assuming
    `exchange.nc` would make the documented example work and every real ladder fail, so the names
    come out of the parent's own completion manifest.
    """
    source, _, _ = prepared
    root = tmp_path_factory.mktemp("named")
    parent, extension = root / "REST2", root / "REST2_ext1"
    for directory in (parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)

    environment = dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1")
    first = subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", str(len(TAUS)),
         "-x", "rest2.nc", "-r", "restart.json", "--checkpoint", "rest2_checkpoint.nc",
         "-o", "run.out", "--rem", "rem.log"],
        cwd=str(parent), capture_output=True, text=True, timeout=1800, env=environment)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    before = {name: _digest(parent / name) for name in
              ("rest2.nc", "rest2_checkpoint.nc", "restart.json", "whole_state0_prod1.nc", "whole_state1_prod1.nc")}

    second = subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", str(len(TAUS)),
         "-x", "rest2.nc", "-r", "restart.json", "--checkpoint", "rest2_checkpoint.nc",
         "-o", "run.out", "--rem", "rem.log",
         "--extend-from", str(parent), "--extend", str(EXCHANGES)],
        cwd=str(extension), capture_output=True, text=True, timeout=1800, env=environment)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]

    assert {name: _digest(parent / name) for name in before} == before
    record = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    assert record["extends"]["parent"]["analysis"]["name"] == "rest2.nc"
    assert record["extends"]["parent"]["checkpoint"]["name"] == "rest2_checkpoint.nc"


# --- an interrupted extension is redone, not resumed --------------------------------------------


def _detach(work, *extra):
    """`_invoke`'s command, started in its own process group so it can be signalled mid-run.

    The signal goes to the GROUP rather than to the child. Here the executor integrates in
    process, but under `mpirun` the launcher is not the process doing the dynamics, and
    signalling only the parent leaves the ranks running -- a mistake already paid for outside
    the suite. Using the group in both cases means the idiom survives a change of launch.
    """
    environment = dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1")
    return subprocess.Popen(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", str(len(TAUS)),
         "-x", "exchange.nc", "-r", "restart.json", "--checkpoint", "checkpoint.nc",
         "-o", "run.out", "--rem", "rem.log", *extra],
        cwd=str(work), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True, env=environment)


def _interrupt_once_committed(process, checkpoint, *, timeout=600):
    """SIGINT the process group once `checkpoint` exists. Returns the collected output.

    Waiting for the checkpoint rather than for a duration is what keeps this test about the
    interruption instead of about how busy the machine is: a committed checkpoint is the
    precondition the whole resume question is asked under, and it is established BEFORE the
    signal is sent, so an escalation in the teardown cannot invalidate it.
    """
    import signal
    import time

    group = os.getpgid(process.pid)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(
                    "the extension reached its budget before a checkpoint was committed; "
                    "--extend is too short for this test to interrupt anything")
            if Path(checkpoint).is_file():
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"waited {timeout}s for {checkpoint} and it never appeared")
        os.killpg(group, signal.SIGINT)
        try:
            collected, _ = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            # SIGINT is a REQUEST: the driver commits at the next event boundary and then exits.
            # How long that takes is not what this test is about, and the thing it needs -- a
            # committed checkpoint -- was established above, before the signal.
            os.killpg(group, signal.SIGKILL)
            collected, _ = process.communicate(timeout=120)
        return collected or ""
    finally:
        if process.poll() is None:
            os.killpg(group, signal.SIGKILL)
            process.communicate()


@pytest.mark.slow
def test_an_interrupted_extension_is_redone_not_resumed(prepared, tmp_path_factory):
    """An extension SEGMENT is atomic: interrupted, it is redone into a fresh directory.

    This is the question a production campaign lost two hours of six-GPU time to, and the answer
    was in no test and no document. The shape of it: nothing REFUSES a mid-extension resume. An
    extension is atomic by OMISSION -- no resume path knows that `extends` exists -- so both
    things a person reaches for fail with messages about something else, and the more dangerous
    one does not fail at all:

      * `--resume` together with `--extend-from` is refused as two different operations, which
        reads as a flag-combination complaint rather than an answer.
      * `--resume` ALONE, on the segment's own directory, would physically continue it and write
        a `restart.json` with no `extends` block: the parent pinning, the segment-local counts
        and the chain accounting are silently gone, and what is left reads as an ordinary run.

    Which is why what the interruption SAYS is the thing under test. A note advising `--resume`
    is advice that produces the second case, so the segment must be named as something to re-run.
    """
    source, _, _ = prepared
    root = tmp_path_factory.mktemp("interrupted-extension")
    parent, extension = root / "REST2", root / "REST2_ext1"
    for directory in (parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)
    assert _invoke(parent).returncode == 0
    before = _fingerprint(parent)

    # Long enough that a checkpoint -- every third exchange at this protocol's cadence -- lands
    # well before the budget does, so there is a genuine mid-segment state to interrupt.
    process = _detach(extension, "--extend-from", str(parent), "--extend", "200")
    output = _interrupt_once_committed(process, extension / "checkpoint.nc")

    assert process.returncode == INTERRUPTED_STATUS, (process.returncode, output[-3000:])
    assert (extension / "checkpoint.nc").is_file()
    # An interruption is not a completion and never leaves the evidence of one.
    assert not (extension / "restart.json").exists()
    # And it did not touch its parent on the way out.
    assert _fingerprint(parent) == before

    state = json.loads((extension / "exchange.runstate.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"

    advice = (state.get("note") or "") + output
    assert "--extend-from" in advice, (
        "the interruption must name re-running the SEGMENT; advice was: " + advice[-2000:])
    for misleading in ("--resume continues it", "--resume will continue it"):
        assert misleading not in advice, (
            f"an interrupted extension advised {misleading!r}, which produces a restart.json "
            f"with no extends block: " + advice[-2000:])

    # The obvious next move -- re-run the same command into the partial directory -- is refused,
    # which is why the answer is a FRESH directory rather than a cleaned one.
    again = _invoke(extension, "--extend-from", str(parent), "--extend", "200")
    assert again.returncode != 0, again.stdout[-3000:]
    assert not (extension / "restart.json").exists()


@pytest.mark.slow
def test_resume_together_with_extend_from_is_refused(prepared, tmp_path_factory):
    """The first dead end, stated as a test so the next person does not have to guess it.

    Continuing in place and writing a new output set are different operations on different
    directories. The refusal is read-only: nothing of the extension is created.
    """
    source, _, _ = prepared
    root = tmp_path_factory.mktemp("resume-and-extend-from")
    parent, extension = root / "REST2", root / "REST2_ext1"
    for directory in (parent, extension):
        directory.mkdir()
        for name in ("system.xml", "topology.pdb", "coordinates.xml", "protocol.py",
                     "ladder.group"):
            shutil.copy(source / name, directory / name)
    assert _invoke(parent).returncode == 0

    result = _invoke(extension, "--extend-from", str(parent), "--extend", str(EXCHANGES),
                     "--resume")
    assert result.returncode != 0
    assert "different operations" in (result.stdout + result.stderr)
    assert not (extension / "exchange.nc").exists()
    assert not (extension / "restart.json").exists()
