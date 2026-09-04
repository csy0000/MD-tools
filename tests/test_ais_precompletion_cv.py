"""An AIS path's CV output, validated BEFORE its completion marker becomes authoritative.

WHY THIS FILE

    The full scientific validation of a path's CV series -- its row count against the schedule,
    its switching-step grid, the path and source-frame identity every row carries, the
    finiteness of every value, the column order, the tau schedule, the sidecar -- ran in exactly
    one place: when a LATER invocation skipped an already completed path. That is the wrong half
    of a path's life. By then `completed.json` is committed and authoritative, the trajectory is
    published under its stable name, and a series that was already wrong at the moment it was
    written has been recorded as finished. The refusal arrives one invocation too late to
    prevent anything.

    The same rules now run before `completed.json` is committed. Everything before that commit
    can be undone -- the staged generation is still on disk and a resume continues from it --
    so a refusal there costs a rerun instead of a wrong answer.

WHAT THE UNIT CASES PROVE

    Each doctors one aspect of an otherwise healthy completed path and asserts the validator
    refuses it. Because the validator is called with the STAGING marker, a refusal means
    `os.replace(staging, marker)` is never reached: no valid completion marker is committed.

PLATFORM_POLICY_EXEMPTION: the paths run under `--cpu`. What is under test is validation of
files already on disk, which reads no platform. The AIS runtime itself is exercised on real
CUDA in `test_cv_cuda_lanes.py`.
"""
from __future__ import annotations

import csv
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

SWITCHING = 40
UPDATE_EVERY = 5
OBSERVE_EVERY = 10
CV_EVERY = UPDATE_EVERY * 2
N_CV = 2
EXPECTED_ROWS = SWITCHING // CV_EVERY + 1

CV_TWO = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
  - name: psi
    type: torsion
    atom_indices: [6, 8, 14, 16]
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """One AIS tree with a two-torsion CV definition, reused by every case."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-precompletion")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_TWO, encoding="utf-8")

    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": SWITCHING,
                "observation_interval_steps": OBSERVE_EVERY,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": OBSERVE_EVERY, "system_printout": OBSERVE_EVERY,
                      "checkpoint_printout": UPDATE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./AIS", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


def _run(project: Path, destination: Path, *extra, environment=None, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    # See the ladder and cMD resume tests: OpenMM's CPU platform is only reproducible at a fixed
    # thread count, and the repaired path is compared to a reference value by value.
    base["OPENMM_CPU_THREADS"] = "1"
    base.update(environment or {})
    done = subprocess.run(
        [sys.executable, str(project / "AIS" / "AIS.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "AIS", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def finished(project, tmp_path_factory):
    """One completed run, kept whole. Every case copies it and doctors the copy."""
    destination = tmp_path_factory.mktemp("finished") / "run"
    _run(project, destination)
    return destination


def _copy(finished, tmp_path):
    destination = tmp_path / "run"
    shutil.copytree(finished, destination)
    return destination


def _check(directory: Path):
    """Run the pre-commit validation exactly as the committing process runs it."""
    from md_tools.ais.run import CV_CSV, _verify_cv_series

    path = directory / "path_0000"
    record = json.loads((path / "completed.json").read_text(encoding="utf-8"))
    schedule = {"cv_interval_steps": CV_EVERY, "switching_steps": SWITCHING}
    _verify_cv_series(path / CV_CSV, record, schedule=schedule, index=0, frame=None,
                      marker=path / "completed.json.staging")


def _cv(directory: Path) -> Path:
    return directory / "path_0000" / "cv.csv"


def _rewrite(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _edit(directory: Path, *, row: int, column: str, value: str):
    path = _cv(directory)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[row + 1].split(",")
    cells[header.index(column)] = value
    lines[row + 1] = ",".join(cells)
    _rewrite(path, lines)


def _frame(directory: Path) -> int:
    return int(_rows(_cv(directory))[0]["source_frame_index"])


def _expect_refusal(directory: Path, fragment: str):
    record = json.loads((directory / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    with pytest.raises(SystemExit) as refusal:
        from md_tools.ais.run import CV_CSV, _verify_cv_series

        _verify_cv_series(directory / "path_0000" / CV_CSV, record,
                          schedule={"cv_interval_steps": CV_EVERY,
                                    "switching_steps": SWITCHING},
                          index=0, frame=_frame(directory),
                          marker=directory / "path_0000" / "completed.json.staging")
    assert fragment in str(refusal.value), refusal.value


def test_a_healthy_path_is_accepted(finished, tmp_path):
    """The control. Without it every case below could pass by refusing everything."""
    directory = _copy(finished, tmp_path)
    _expect = json.loads(
        (directory / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert int(_expect["cv_rows"]) == EXPECTED_ROWS
    from md_tools.ais.run import CV_CSV, _verify_cv_series

    _verify_cv_series(directory / "path_0000" / CV_CSV, _expect,
                      schedule={"cv_interval_steps": CV_EVERY, "switching_steps": SWITCHING},
                      index=0, frame=_frame(directory),
                      marker=directory / "path_0000" / "completed.json.staging")


def test_a_truncated_series_is_refused(finished, tmp_path):
    """The last row lost after it was written: the count no longer matches the schedule."""
    directory = _copy(finished, tmp_path)
    lines = _cv(directory).read_text(encoding="utf-8").splitlines()
    _rewrite(_cv(directory), lines[:-1])
    _expect_refusal(directory, "data row")


def test_a_gap_in_the_switching_grid_is_refused(finished, tmp_path):
    """Right length, wrong grid: a step renumbered rather than a row dropped."""
    directory = _copy(finished, tmp_path)
    _edit(directory, row=2, column="protocol_step", value=str(SWITCHING + 1000))
    _expect_refusal(directory, "step grid is wrong")


def test_a_row_from_another_path_is_refused(finished, tmp_path):
    directory = _copy(finished, tmp_path)
    _edit(directory, row=1, column="path_index", value="7")
    _expect_refusal(directory, "path_index")


def test_a_row_from_another_source_frame_is_refused(finished, tmp_path):
    directory = _copy(finished, tmp_path)
    _edit(directory, row=1, column="source_frame_index", value="9999")
    _expect_refusal(directory, "source frame")


def test_a_non_finite_value_is_refused(finished, tmp_path):
    directory = _copy(finished, tmp_path)
    _edit(directory, row=1, column="phi", value="nan")
    _expect_refusal(directory, "non-finite")


def test_a_tau_schedule_running_the_wrong_way_is_refused(finished, tmp_path):
    """AIS anneals tau one way and never back. A wandering tau is not a switching path."""
    directory = _copy(finished, tmp_path)
    rows = _rows(_cv(directory))
    _edit(directory, row=1, column="tau", value=f"{float(rows[0]['tau']) + 0.25:.6f}")
    _expect_refusal(directory, "not monotonic")


def test_a_frame_reference_that_is_neither_empty_nor_an_index_is_refused(finished, tmp_path):
    """Empty is load-bearing: it says this step was not also a saved observation."""
    directory = _copy(finished, tmp_path)
    _edit(directory, row=1, column="coordinate_frame_index", value="-3")
    _expect_refusal(directory, "coordinate_frame_index")


def test_a_cumulative_cost_that_disagrees_with_the_rows_is_refused(finished, tmp_path):
    """The counters are pinned to the row count; a resume that lost a segment shows up here."""
    directory = _copy(finished, tmp_path)
    marker = directory / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["collective_variable_cost"]["cumulative"]["cv_observations"] = EXPECTED_ROWS - 1
    marker.write_text(json.dumps(record), encoding="utf-8")
    _expect_refusal(directory, "observation(s) and the series holds")


def test_a_scalar_count_that_is_not_rows_times_the_definition_is_refused(finished, tmp_path):
    """The misnomer this whole change exists to remove: calls counted as scalar evaluations."""
    directory = _copy(finished, tmp_path)
    marker = directory / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["collective_variable_cost"]["cumulative"]["cv_evaluations"] = EXPECTED_ROWS
    marker.write_text(json.dumps(record), encoding="utf-8")
    _expect_refusal(directory, "scalar evaluation(s)")


def test_a_crash_before_finalization_leaves_no_marker_and_resumes_to_the_reference(
        project, tmp_path):
    """The end-to-end promise, with the CV stream corrupted after its last write.

    The path is crashed part-way through switching, at a point where the CV writer has already
    written rows PAST the last committed checkpoint generation. Those rows are the uncommitted
    suffix, and the test then corrupts it further on disk -- a garbage row appended after the
    last real write, which is what a writer killed mid-flush leaves behind.

    The resume must trust none of it. It selects the last committed generation, discards
    everything after it, rebuilds the suffix, and lands on a path identical to one that was
    never interrupted, with no duplicated row.

    (Damaging a COMMITTED row is a different fault and is already refused: the checkpoint
    vouches for those rows, so losing one is data loss rather than a resumable state. This test
    deliberately damages only what no generation vouches for.)
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "reference"
    _run(project, reference)
    want = _rows(_cv(reference))
    assert len(want) == EXPECTED_ROWS

    destination = tmp_path / "resumed"
    crashed = _run(project, destination, expect=1,
                   environment={FAULT_ENVIRONMENT: "after-work-row",
                                FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0
    marker = destination / "path_0000" / "completed.json"
    assert not marker.exists(), (
        "a completion marker was committed for a path that crashed before finalization")

    # The corruption, confined to what no committed generation vouches for: a garbage row
    # appended after the last real write, as a writer killed mid-flush leaves behind.
    damaged = _cv(destination)
    assert damaged.is_file(), "the crashed path wrote no CV series to damage"
    lines = damaged.read_text(encoding="utf-8").splitlines()
    width = len(lines[0].split(","))
    _rewrite(damaged, lines + [",".join(["999999"] * width)])

    _run(project, destination, "--resume")
    assert marker.is_file(), "the resumed path committed no completion marker"

    got = _rows(_cv(destination))
    assert len(got) == EXPECTED_ROWS, "the repaired series is the wrong length"
    steps = [row["protocol_step"] for row in got]
    assert len(set(steps)) == len(steps), "the repair duplicated a row"
    assert got == want, (
        "the repaired path differs from the uninterrupted reference: the resume did not rebuild "
        "the uncommitted suffix, it produced a different path")


def test_the_validation_runs_before_the_commit_and_not_after():
    """The ORDER, asserted against the source.

    Every case above calls the validator directly, which proves the rules and says nothing about
    when they run -- and "when" is the entire correction: these same rules already existed and
    already refused every corruption here, one invocation too late, on the skip path. A
    validation that ran after `os.replace(staging_marker, marker)` would pass all of them while
    leaving a committed, authoritative marker on a bad path.

    (The repo already asserts an ordering against source text this way in
    `test_v2_append_compatibility.py`, for the same reason: no runtime observation distinguishes
    "checked first" from "checked second" once both have happened.)
    """
    import inspect

    from md_tools.ais import run as ais_run

    body = inspect.getsource(ais_run.run_one_path)
    checked = body.index("_verify_cv_series(")
    committed = body.index("os.replace(staging_marker, marker)")
    assert checked < committed, (
        "the CV series is validated after the completion marker is committed, which is the "
        "defect this change exists to remove")
