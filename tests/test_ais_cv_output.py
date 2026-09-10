"""AIS writes a CV series per path, aligned to the coordinates it actually saved.

THE ALIGNMENT CLAIM

    When a CV row carries an `observation_index` or a `coordinate_frame_index`, the values on it
    must have been measured on EXACTLY that saved coordinate -- not on a neighbouring one, and
    not interpolated. Attaching a CV measured at x_j to a different observation's coordinate is
    the same class of error the AIS two-probe separation exists to prevent, and it is invisible in
    the output: every number is plausible and in range.

    So the test recomputes each aligned row's torsion from the frame stored in the path's own
    NetCDF trajectory and requires agreement. A misalignment by one frame fails it.

PLATFORM_POLICY_EXEMPTION: the paths run under `--cpu`. What is under test is which coordinate a
row was measured on, which is bookkeeping and identical on every platform.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-cv")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "observation_interval_steps": 10, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        # On the update grid (multiple of 5) and dividing switching_steps (20).
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./AIS", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("ais-cv-run") / "run"
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    done = subprocess.run(
        [sys.executable, str(project / "AIS" / "AIS.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-odir", str(destination), "--cpu"],
        cwd=project / "AIS", capture_output=True, text=True, timeout=1800, env=base)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return destination


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_each_path_writes_its_own_series(completed):
    for index in (0, 1):
        assert (completed / f"path_{index:04d}" / "cv.csv").is_file()


def test_the_columns_are_the_documented_ones(completed):
    with (completed / "path_0000" / "cv.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["path_index", "source_frame_index", "protocol_step", "time_ps", "tau",
                      "observation_index", "coordinate_frame_index", "phi"]


def test_the_series_lands_on_the_update_grid_including_both_endpoints(completed):
    """20 switching steps at an interval of 5: five rows, 0 and 20 present exactly once."""
    steps = [int(row["protocol_step"]) for row in _rows(completed / "path_0000" / "cv.csv")]
    assert steps == [0, 5, 10, 15, 20], steps


def test_tau_moves_from_tau_start_to_tau_end_along_the_series(completed):
    taus = [float(row["tau"]) for row in _rows(completed / "path_0000" / "cv.csv")]
    assert taus[0] == 0.5 and taus[-1] == 0.0
    assert taus == sorted(taus, reverse=True), taus


def test_rows_off_the_observation_grid_have_empty_indices(completed):
    """Empty, never a nearest neighbour: a CV attached to another coordinate is invisible."""
    rows = _rows(completed / "path_0000" / "cv.csv")
    by_step = {int(row["protocol_step"]): row for row in rows}
    # Observations are every 10 steps; the CV cadence is every 5.
    assert by_step[5]["observation_index"] == "", by_step[5]
    assert by_step[15]["observation_index"] == "", by_step[15]
    assert by_step[10]["observation_index"] != "", by_step[10]
    assert by_step[20]["observation_index"] != "", by_step[20]


def test_an_aligned_row_was_measured_on_exactly_the_saved_coordinate(completed, project):
    """THE alignment claim, checked against the coordinates the path actually stored."""
    mdtraj = pytest.importorskip("mdtraj")

    trajectories = sorted(completed.glob("AIS_traj*.nc"))
    assert trajectories, "no published path trajectory to check alignment against"
    frames = mdtraj.load(str(trajectories[0]), top=str(project / "built.pdb"))

    checked = 0
    for row in _rows(completed / "path_0000" / "cv.csv"):
        index = row["coordinate_frame_index"]
        if index == "":
            continue
        expected = math.degrees(float(
            mdtraj.compute_dihedrals(frames[int(index)], [[4, 6, 8, 14]])[0][0]))
        reported = float(row["phi"])
        difference = abs((reported - expected + 180.0) % 360.0 - 180.0)
        assert difference < 1e-2, (row["protocol_step"], index, reported, expected)
        checked += 1
    assert checked >= 2, f"only {checked} aligned row(s) were available to check"


def test_the_aggregate_is_assembled_from_every_completed_path(completed):
    aggregate = completed / "AIS_cv.csv"
    assert aggregate.is_file(), "no aggregate CV table"
    rows = _rows(aggregate)
    assert {int(row["path_index"]) for row in rows} == {0, 1}
    # Sorted by path then step, so the table reads as a sequence of complete paths.
    order = [(int(r["path_index"]), int(r["protocol_step"])) for r in rows]
    assert order == sorted(order), "the aggregate is not in path/step order"
    assert len(rows) == 10, f"expected 5 rows from each of 2 paths, got {len(rows)}"


def test_a_path_without_a_verified_manifest_is_absent_from_the_aggregate(completed, tmp_path):
    """The aggregate gate is the completion manifest, not the presence of a cv.csv.

    A path that crashed mid-write leaves a plausible-looking series and no manifest; including it
    would put rows from an unfinished switch into the run's headline table.
    """
    import shutil

    from md_tools.ais.run import write_work_table

    staged = tmp_path / "staged"
    shutil.copytree(completed, staged)
    (staged / "path_0001" / "completed.json").unlink()

    summary = write_work_table(staged, [0, 1])
    rows = _rows(staged / "AIS_cv.csv")
    assert {int(row["path_index"]) for row in rows} == {0}, (
        "a path with no verified manifest reached the aggregate")
    assert summary["cv_rows"] == 5


def test_the_sidecar_records_the_path_and_the_alignment_rule(completed):
    body = json.loads((completed / "path_0000" / "cv.json").read_text(encoding="utf-8"))
    assert body["path_index"] == 0
    assert body["tau_start"] == 0.5 and body["tau_end"] == 0.0
    assert "exactly that saved coordinate" in body["observation_index_meaning"]


def test_a_cv_interval_off_the_update_grid_is_refused(project, tmp_path):
    """tau is piecewise constant across an update: a row between two would name a tau never held."""
    configuration = yaml.safe_load((project / "AIS.config").read_text(encoding="utf-8"))
    configuration["collective_variables"]["interval_steps"] = 4       # 4 % 5 != 0
    (tmp_path / "bad.config").write_text(yaml.safe_dump(configuration), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", str(tmp_path / "bad"),
               "--config", str(tmp_path / "bad.config")],
        cwd=project, capture_output=True, text=True, timeout=600)
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-3000:]
    assert "parameter update interval" in message, message[-3000:]
