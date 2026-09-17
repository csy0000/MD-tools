"""A real AIS run under `tau-linear`: the lambdas it visits, the identity at saved frames, the
frames it started from, and what a changed schedule does to a second invocation.

The unit tests in `test_ais_schedules.py` pin the lambda table and the refusals. What they cannot
show is that the RUNTIME visits those lambdas, records the schedule it ran under, and refuses to
extend that directory under another one -- which is the whole point of putting the schedule in the
run identity. So this file runs the paths.

V0 is the REST2 state at tau = 0.5 of `built.xml` and V1 is `built.xml`, the pair `tau-linear` is
defined for. Implicit solvent, three short paths.

PLATFORM_POLICY_EXEMPTION: `--cpu`. What is under test is which lambdas the path visited and what
the identity records, which is identical on every platform. The CUDA evidence for AIS switching
itself lives in the gpu lane.
"""
from __future__ import annotations

import csv
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

PATHS = 3
SWITCHING = 40
UPDATE_EVERY = 5
OBSERVE_EVERY = 10
TAU0 = 0.5
#: The source trajectory holds 8 copies of the built structure; every frame is eligible.
SOURCE_FRAMES = 8


def _config(root: Path, *, schedule: str, tau0: float | None) -> dict:
    ais = {"number_of_paths": PATHS, "switching_steps": SWITCHING,
           "observation_interval_steps": OBSERVE_EVERY,
           "parameter_update_interval_steps": UPDATE_EVERY,
           "lambda_schedule": schedule}
    if tau0 is not None:
        ais["lambda_schedule_tau0"] = tau0
    return {
        "protocol": "AIS", "solvent": "implicit", "ais": ais,
        "ais_source": {"trajectory": "../source.dcd", "selection": "evenly_spaced"},
        "reporting": {"crd_printout_solute": OBSERVE_EVERY, "info_printout": OBSERVE_EVERY,
                      "checkpoint_printout": UPDATE_EVERY},
        "dynamics": {"seed": 20260917},
    }


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A dataset root with both end states, a source trajectory, and two generated runs."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-schedule-run")
    (root / "build").mkdir(exist_ok=True)
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0

    config = root / "AIS.config"
    config.write_text(yaml.safe_dump(_config(root, schedule="tau-linear", tau0=TAU0)),
                      encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./AIS-run1", "--config", str(config)],
        cwd=root, capture_output=True, text=True, timeout=900).returncode == 0

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * SOURCE_FRAMES).save_dcd(str(root / "source.dcd"))

    # V0 is a SAVED STATE, scaler.yaml and all: tau-linear is refused without one, because the
    # record is what the claim about tau0 and about V1 is checked against.
    from .conftest import make_scaled_state

    make_scaled_state(root, tau=TAU0, method="AIS")

    # THE LINEAR RUN NEEDS ITS OWN DATASET ROOT. `input/AIS.in` is shared by every run on a
    # system, so a second configuration resolving to different bytes is refused there -- which is
    # the correct behaviour and is why the schedule-change refusal has to be tested across two
    # roots rather than two run directories.
    import shutil

    other = root.parent / f"{root.name}-linear"
    shutil.copytree(root / "build", other / "build")
    shutil.copy2(root / "source.dcd", other / "source.dcd")
    linear = other / "AIS.config"
    linear.write_text(yaml.safe_dump(_config(other, schedule="linear", tau0=None)),
                      encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./AIS-run1", "--config", str(linear)],
        cwd=other, capture_output=True, text=True, timeout=900).returncode == 0
    return root, other


def _run(project: Path, destination: Path, *extra, expect=0):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    environment["MD_TOOLS_CONFIG"] = str(user)
    environment["OPENMM_CPU_THREADS"] = "1"
    done = subprocess.run(
        [sys.executable, str(project / "AIS-run1" / "AIS.py"),
         "-p", str(project / "build" / "built.pdb"),
         "-s", str(project / "build" / "AIS" / "system_state0.xml"),
         "-p2", str(project / "build" / "built.pdb"),
         "-s2", str(project / "build" / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "AIS-run1", capture_output=True, text=True, timeout=1800, env=environment)
    assert (done.returncode == 0) == (expect == 0), done.stdout[-4000:] + done.stderr[-4000:]
    return done


@pytest.fixture(scope="module")
def completed(project):
    root, other = project
    out = root / "run-tau-linear"
    _run(root, out)
    return other, out


def _lambda_table():
    from md_tools.ais.schedule import switching_schedule

    return switching_schedule(
        switching_steps=SWITCHING, parameter_update_interval_steps=UPDATE_EVERY,
        observation_interval_steps=OBSERVE_EVERY, timestep_fs=2.0,
        trajectory_interval_steps=OBSERVE_EVERY, state_interval_steps=OBSERVE_EVERY,
        checkpoint_interval_steps=UPDATE_EVERY,
        lambda_schedule="tau-linear", lambda_schedule_tau0=TAU0)


def test_the_work_table_visits_exactly_the_scheduled_lambdas(completed):
    """G1. Every work row's lambda is the table's value for the updates completed by that step."""
    _, out = completed
    lambdas = _lambda_table()["lambdas"]
    rows = list(csv.DictReader((out / "AIS_work.csv").open()))
    assert len(rows) == PATHS * (SWITCHING // OBSERVE_EVERY + 1)
    for row in rows:
        step = int(row["switch_step"])
        after = lambdas[step // UPDATE_EVERY]
        before = lambdas[max(step - OBSERVE_EVERY, 0) // UPDATE_EVERY]
        assert float(row["lambda_after"]) == pytest.approx(after, abs=1e-12), row
        assert float(row["lambda_before"]) == pytest.approx(before, abs=1e-12), row
    last = [row for row in rows if int(row["switch_step"]) == SWITCHING]
    assert len(last) == PATHS and all(float(row["lambda_after"]) == 1.0 for row in last)
    # Not the linear table: the schedule genuinely changed which lambdas ran.
    assert any(abs(float(row["lambda_after"]) - int(row["switch_step"]) / SWITCHING) > 1e-6
               for row in rows)


def test_every_saved_frame_row_satisfies_the_mixing_identity(completed):
    """G1. V(lambda) = (1 - lambda) V0 + lambda V1 at the coordinate the row saved."""
    _, out = completed
    rows = [row for row in csv.DictReader((out / "AIS_hs.csv").open())]
    assert rows, "the frame-aligned table is empty"
    for row in rows:
        lam = float(row["lambda"])
        v0 = float(row["potential_v0_kj_mol"])
        v1 = float(row["potential_v1_kj_mol"])
        assert float(row["potential_direct_kj_mol"]) == pytest.approx(
            (1.0 - lam) * v0 + lam * v1, abs=1e-4, rel=1e-9), row
    assert any(abs(float(row["potential_v1_kj_mol"]) - float(row["potential_v0_kj_mol"])) > 1.0
               for row in rows), "the end states are indistinguishable; the identity is vacuous"


def test_the_run_identity_records_the_schedule_its_tau0_and_the_lambda_digest(completed):
    """G1. What a second invocation is compared against."""
    _, out = completed
    document = json.loads((out / "AIS_run.json").read_text(encoding="utf-8"))
    assert document["lambda"]["schedule"] == "tau-linear"
    assert document["lambda"]["tau0"] == TAU0
    assert document["schedule"]["lambda_schedule"] == "tau-linear"
    assert document["schedule"]["lambda_schedule_tau0"] == TAU0
    assert document["schedule"]["lambda_sha256"] == _lambda_table()["lambda_sha256"]


def test_the_selected_frames_span_the_whole_eligible_window(completed):
    """G4. evenly_spaced reaches the last eligible frame, not only the first `count` of them."""
    _, out = completed
    rows = list(csv.DictReader((out / "selected_source_frames.csv").open()))
    frames = [int(row["source_frame_index"]) for row in rows]
    assert len(frames) == PATHS and len(set(frames)) == PATHS
    assert frames[0] == 0 and frames[-1] == SOURCE_FRAMES - 1


def test_a_second_invocation_under_another_schedule_is_refused_by_name(completed):
    """G2. The directory holds a tau-linear campaign; linear paths may not be added to it."""
    linear_root, out = completed
    before = (out / "AIS_run.json").read_text(encoding="utf-8")
    done = _run(linear_root, out, "--resume", expect=1)
    message = done.stdout + done.stderr
    assert "already holds a DIFFERENT AIS run" in message
    assert "schedule" in message or "lambda" in message, message
    assert (out / "AIS_run.json").read_text(encoding="utf-8") == before
