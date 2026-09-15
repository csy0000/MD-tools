"""CV outputs are owned, governed and verified like every other scientific stream.

WHAT WAS WRONG

    The cMD inventory named `<stage>.cv.yaml` -- a file that never existed, because the writer
    produces `<stage>.cv.json`. So the real sidecar was in no inventory: not collision-checked,
    not removed by `--overwrite`, and free to survive a definition change and describe the new
    CSV with the old atom mapping.

    `cv_stateN.csv`, `cv_stateN.json` and `AIS_cv.csv` were in no inventory at all. A CV-DISABLED
    rerun over a CV-enabled directory therefore left them in place permanently, describing a
    calculation that no longer exists, with nothing in the directory saying so.

    AIS path completion recorded `cv_rows` but not the FILES, so a path could be skipped as
    complete while its CV output had been truncated, mutated or deleted: the count agreed with
    itself and nothing looked at the bytes.

PLATFORM_POLICY_EXEMPTION: runs are under `--cpu`. What is under test is which files a run owns
and which it refuses, which is bookkeeping and identical on every platform.
"""
from __future__ import annotations

import json
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


def _build(root: Path):
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")


def _cmd_config(root: Path, *, cv: bool):
    return {
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 20},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": ({"file": str(root / "cv.yaml"), "interval_steps": 5}
                                 if cv else {"file": None, "interval_steps": 0}),
        "dynamics": {"seed": 20260904},
    }


def _generate(root: Path, name: str, document: dict, *extra):
    (root / f"{name}.config").write_text(yaml.safe_dump(document), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", f"./{name}-run1", "--config", str(root / f"{name}.config"),
               *extra],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    # The RUN directory, which is what `-odir` just created. Returning `root / name` pointed at a
    # directory that no longer exists, so every caller died on `cwd` before reaching its subject.
    return root / f"{name}-run1"


def _environment(root: Path):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    return base


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    place = tmp_path_factory.mktemp("cv-ownership")
    _build(place)
    return place


def _run_cmd(root: Path, script_dir: Path, destination: Path, *extra, expect=0):
    done = subprocess.run(
        [sys.executable, str(script_dir / "md.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=script_dir, capture_output=True, text=True, timeout=1800, env=_environment(root))
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-3000:] + done.stderr[-3000:]
    return done


# --- the sidecar the inventory used to miss ----------------------------------------------------

def test_the_cmd_inventory_names_the_sidecar_that_is_actually_written(root, tmp_path):
    """`.cv.json`, not `.cv.yaml`. The inventory named a file that never existed."""
    from md_tools.run.preflight import cv_sidecar_path

    scripts = _generate(root, "cvA", _cmd_config(root, cv=True), "--all-in-one")
    destination = tmp_path / "run"
    _run_cmd(root, scripts, destination)

    series = sorted(destination.rglob("*.cv.csv"))
    assert len(series) == 1
    sidecar = cv_sidecar_path(series[0])
    assert sidecar.is_file(), f"the written sidecar is not {sidecar}"
    assert not sidecar.with_suffix(".yaml").is_file(), "a .cv.yaml was written after all"

    from md_tools.md.stage import cv_csv_path
    from md_tools.run.preflight import _stage_inventory

    inventory = _stage_inventory(
        output=destination / "cMD.out", log=destination / "cMD.log",
        trajectory=destination / "cMD.dcd", restart=destination / "cMD.xml",
        checkpoint=destination / "cMD.chk")
    named = inventory.roles["collective_variables_definition"]
    assert named.name.endswith(".cv.json"), named
    assert not str(named).endswith(".cv.yaml")


def test_the_completion_record_names_the_sidecar_that_exists(root, tmp_path):
    scripts = _generate(root, "cvB", _cmd_config(root, cv=True), "--all-in-one")
    destination = tmp_path / "run"
    _run_cmd(root, scripts, destination)

    from md_tools.build.record import read_record

    record = read_record(destination / "cMD.log")
    outputs = record.get("outputs") or {}
    assert "collective_variables" in outputs, "the completion record does not claim the series"
    facts = outputs["collective_variables_definition"]
    assert str(facts["path"]).endswith(".cv.json"), facts


# --- overwrite: no stale CV artefact may survive ------------------------------------------------

def test_overwrite_replaces_the_cv_series_and_its_sidecar(root, tmp_path):
    scripts = _generate(root, "cvC", _cmd_config(root, cv=True), "--all-in-one")
    destination = tmp_path / "run"
    _run_cmd(root, scripts, destination)

    series = sorted(destination.rglob("*.cv.csv"))[0]
    from md_tools.run.preflight import cv_sidecar_path

    sidecar = cv_sidecar_path(series)
    series.write_text("step,time_ps,trajectory_frame_index,phi\n999,9.9,,0.0\n", encoding="utf-8")
    sidecar.write_text('{"tampered": true}', encoding="utf-8")

    _run_cmd(root, scripts, destination, "--overwrite")
    rebuilt = series.read_text(encoding="utf-8")
    assert "999" not in rebuilt, "the previous run's CV rows survived --overwrite"
    assert json.loads(sidecar.read_text(encoding="utf-8")).get("units") == "degrees", (
        "the previous run's sidecar survived --overwrite")


def test_a_cv_disabled_overwrite_leaves_no_stale_cv_output(root, tmp_path):
    """THE stale-artefact case. A directory that stops reporting CVs must stop having them.

    Otherwise the tree keeps a CSV and a sidecar describing a calculation the run no longer
    performs, and nothing in the directory says which run they belong to.
    """
    enabled = _generate(root, "cvD_on", _cmd_config(root, cv=True), "--all-in-one")
    destination = tmp_path / "run"
    _run_cmd(root, enabled, destination)
    assert sorted(destination.rglob("*.cv.csv")), "the first run wrote no CV series"

    # A SEPARATE DATASET ROOT for the CV-disabled run, not `--overwrite` on this one.
    #
    # `eq_*.in` carry the cadence, so a CV-on and a CV-off run have different shared inputs and
    # the second is refused. `--overwrite` looks like the remedy the refusal names, and it is not:
    # it rewrites `input/eq_*.in` to the CV-disabled form, and the four CV-ENABLED generations
    # later in this module then refuse against THAT. Within one root the collision simply moves,
    # whichever order the generations run in.
    #
    # The two runs are the same experiment in every respect except what they observe -- reporting
    # adds no Force, and the System, force inventory and single-point energy are asserted
    # identical with it on and off -- so making them comparable on ONE system means making the
    # cadence a per-run value beside the seed. That widens `RUN_CONFIG_ALLOWED`, which is a
    # deliberate one-key schema, so it is recorded in docs/run-layout.md as open rather than
    # decided here.
    off_root = root.parent / "cv-ownership-off"
    if not (off_root / "build" / "built.xml").is_file():
        off_root.mkdir(exist_ok=True)
        _build(off_root)
    disabled = _generate(off_root, "cvD_off", _cmd_config(off_root, cv=False), "--all-in-one")
    _run_cmd(off_root, disabled, destination, "--overwrite")

    left = sorted(destination.rglob("*.cv.csv")) + sorted(destination.rglob("*.cv.json"))
    assert not left, f"a CV-disabled run left stale collective-variable output: {left}"


# --- completion refuses a damaged series --------------------------------------------------------

@pytest.mark.parametrize("damage", ["truncate", "mutate", "delete", "replace_sidecar"])
def test_a_damaged_cv_series_is_not_accepted_as_a_completed_stage(root, tmp_path, damage):
    """A completed stage may not be skipped when its CV output no longer matches the record."""
    from md_tools.run.preflight import cv_sidecar_path

    scripts = _generate(root, f"cvE_{damage}", _cmd_config(root, cv=True), "--all-in-one")
    destination = tmp_path / "run"
    _run_cmd(root, scripts, destination)

    series = sorted(destination.rglob("*.cv.csv"))[0]
    sidecar = cv_sidecar_path(series)
    lines = series.read_text(encoding="utf-8").splitlines()
    if damage == "truncate":
        series.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    elif damage == "mutate":
        parts = lines[-1].split(",")
        parts[-1] = f"{float(parts[-1]) + 5.0:.6f}"
        series.write_text("\n".join(lines[:-1] + [",".join(parts)]) + "\n", encoding="utf-8")
    elif damage == "delete":
        series.unlink()
    else:
        sidecar.write_text('{"schema_version": 1}', encoding="utf-8")

    from md_tools.md.completion import verify_completed_stage

    from md_tools.build.record import read_record

    record = read_record(destination / "cMD.log")
    problems = verify_completed_stage(
        record, stage=record["stage"], name="cMD",
        restart=destination / "cMD.xml", trajectory=destination / "cMD.dcd",
        log_path=destination / "cMD.log", fingerprint=record["fingerprint"])
    assert problems, f"a {damage}d CV series was accepted as a completed stage"
    assert any("cv" in problem.lower() for problem in problems), problems
