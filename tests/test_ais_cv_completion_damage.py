"""A completed AIS path stops being skippable when its CV output changes.

WHAT WAS MISSING

    The path manifest hashed `cv.csv` and `cv.json`, which proves the bytes have not changed
    since completion. That says nothing about whether they were RIGHT when the manifest was
    written: a path whose series was short by a row, or whose grid skipped one, was committed and
    then skipped as complete for ever after.

    So completion now derives the expected shape from the SCHEDULE --
    `switching_steps / cv_interval_steps + 1` rows on the grid `0, interval, ..., switching_steps`
    -- rather than reading it from the file, and checks path and source-frame identity on every
    row. A file that is self-consistently wrong cannot satisfy a requirement it did not supply.

Each damage below leaves a file that still parses and still looks like a CV series.

PLATFORM_POLICY_EXEMPTION: paths run under `--cpu`. What is under test is whether a completed
path is skipped -- bookkeeping, identical on every platform; the AIS runtime is exercised on CUDA
in `test_cv_cuda_lanes.py`.
"""
from __future__ import annotations

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

SWITCHING = 20
UPDATE_EVERY = 5
CV_EVERY = 5


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-cv-damage")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(
        "schema_version: 1\ncollective_variables:\n"
        "  - {name: phi, type: torsion, atom_indices: [4, 6, 8, 14]}\n", encoding="utf-8")
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": SWITCHING,
                "observation_interval_steps": 10,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./AIS", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


def _run(project: Path, destination: Path, *extra, expect=0):
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
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "AIS", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("ais-cv-damage-run") / "run"
    _run(project, destination)
    return destination


def test_the_series_has_the_length_and_grid_its_schedule_implies(completed):
    """`switching_steps / cv_interval_steps + 1`, both endpoints included exactly once."""
    path = completed / "path_0000" / "cv.csv"
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    assert len(rows) == SWITCHING // CV_EVERY + 1
    assert [int(row.split(",")[2]) for row in rows] == list(range(0, SWITCHING + 1, CV_EVERY))
    record = json.loads((completed / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert record["cv_rows"] == len(rows)


def test_rerunning_a_completed_run_skips_every_path(completed, project):
    """The control: an untouched completed campaign is skipped, not redone."""
    done = _run(project, completed)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


DAMAGE = ["truncate", "mutate_value", "mutate_step", "delete_csv", "replace_sidecar",
          "mutate_path_index"]


@pytest.mark.parametrize("damage", DAMAGE)
def test_a_damaged_path_is_not_skipped_as_completed(completed, project, tmp_path, damage):
    staged = tmp_path / damage
    shutil.copytree(completed, staged)
    csv_path = staged / "path_0000" / "cv.csv"
    sidecar = staged / "path_0000" / "cv.json"
    lines = csv_path.read_text(encoding="utf-8").splitlines()

    if damage == "truncate":
        csv_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    elif damage == "mutate_value":
        parts = lines[-1].split(",")
        parts[-1] = f"{float(parts[-1]) + 9.0:.6f}"
        csv_path.write_text("\n".join(lines[:-1] + [",".join(parts)]) + "\n", encoding="utf-8")
    elif damage == "mutate_step":
        parts = lines[-1].split(",")
        parts[2] = str(int(parts[2]) + 1)
        csv_path.write_text("\n".join(lines[:-1] + [",".join(parts)]) + "\n", encoding="utf-8")
    elif damage == "mutate_path_index":
        parts = lines[-1].split(",")
        parts[0] = "7"
        csv_path.write_text("\n".join(lines[:-1] + [",".join(parts)]) + "\n", encoding="utf-8")
    elif damage == "delete_csv":
        csv_path.unlink()
    else:
        sidecar.write_text('{"schema_version": 1}', encoding="utf-8")

    done = _run(project, staged, expect=1)
    message = (done.stdout + done.stderr + "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(staged.rglob("AIS.out*")))).lower()
    assert "path 0" in message or "cv.csv" in message or "cv.json" in message, message[-3000:]


def test_the_aggregate_is_not_rebuilt_from_a_damaged_path(completed, project, tmp_path):
    """A refused path must not reach `AIS_cv.csv` through a rerun that keeps going."""
    staged = tmp_path / "aggregate"
    shutil.copytree(completed, staged)
    lines = (staged / "path_0000" / "cv.csv").read_text(encoding="utf-8").splitlines()
    (staged / "path_0000" / "cv.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    before = (staged / "AIS_cv.csv").read_text(encoding="utf-8")

    _run(project, staged, expect=1)
    assert (staged / "AIS_cv.csv").read_text(encoding="utf-8") == before, (
        "the refused run rewrote the aggregate from a path it had just rejected")
