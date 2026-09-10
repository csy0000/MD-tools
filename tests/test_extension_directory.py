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

from .conftest import EXCHANGES, TAUS                               # noqa: E402

PARENT_FILES = ("exchange.nc", "checkpoint.nc", "restart.json", "rem.log",
                "exchange.solute.nc", "exchange.runstate.json", "whole_state0_prod1.nc", "whole_state1_prod1.nc")


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
        f"trajin {parent / 'remd0.nc'}\n"
        f"trajin {extension / 'remd0.nc'}\n"
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
