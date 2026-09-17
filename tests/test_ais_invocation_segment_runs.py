"""Tests B and C: what an AIS invocation reports it did, across real runs.

    B  a campaign completed across two invocations -- paths already finished contribute zero to
       the second invocation's segment, and only what invocation 2 actually evaluated counts;
    C  a completed campaign re-entered with `--resume` -- an EXACTLY zero segment, and not one
       byte of the scientific output rewritten.

WHY BOTH, WHEN TEST A ALREADY STATES THE RULE

    Test A proves the arithmetic. It supplies the contributions itself, so it cannot show that
    the runtime produces the right ones -- that a skipped path really records zero, that a
    resumed path records only its second half, that the dispositions match what actually
    happened. These do, through the real runtime, against an uninterrupted reference.

EXPECTED COUNTS ARE DERIVED FROM THE SCHEDULE

    `switching_steps / cv_interval_steps + 1` rows per path, two torsions, so a path is 5
    observations and 10 scalar evaluations and a three-path campaign is 15 and 30. Those numbers
    are written out below and never computed by the code under test.

PLATFORM_POLICY_EXEMPTION: these run under `--cpu`. What is under test is which invocation gets
credited with which work -- accounting across processes, identical on every platform. The same
accounting is exercised on real CUDA in `test_cv_cuda_lanes.py` and across a changed MPI world
size in `test_cv_mpi_cuda_ais.py`.
"""
from __future__ import annotations

import csv
import hashlib
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
CV_EVERY = 10
N_CV = 2

#: Derived from the schedule, by hand: 40 / 10 + 1.
ROWS_PER_PATH = 5
OBSERVATIONS_PER_PATH = ROWS_PER_PATH
EVALUATIONS_PER_PATH = ROWS_PER_PATH * N_CV
CAMPAIGN_OBSERVATIONS = PATHS * OBSERVATIONS_PER_PATH        # 15
CAMPAIGN_EVALUATIONS = PATHS * EVALUATIONS_PER_PATH          # 30

CV_YAML = """\
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
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-invocation")
    (root / "build").mkdir(exist_ok=True)
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": PATHS, "switching_steps": SWITCHING,
                "observation_interval_steps": OBSERVE_EVERY,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": OBSERVE_EVERY, "info_printout": OBSERVE_EVERY,
                      "checkpoint_printout": UPDATE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260905},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./AIS", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=900).returncode == 0

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))

    # The two end states: V0 is the REST2 state at tau = 0.5 of `built.xml`, V1 is `built.xml`.
    # What is under test is accounting, so any parameter-only pair would do; this one is the pair
    # the retired tau switch travelled.
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2.hamiltonian import build_scaled_system

    base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    solute = solute_atom_indices(PDBFile(str(root / "build" / "built.pdb")).topology)
    (root / "build" / "V0.xml").write_text(
        XmlSerializer.serialize(build_scaled_system(base, solute, 0.5)), encoding="utf-8")
    return root


def _run(project: Path, destination: Path, *extra, environment=None, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base["OPENMM_CPU_THREADS"] = "1"
    base.update(environment or {})
    done = subprocess.run(
        [sys.executable, str(project / "AIS" / "AIS.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "V0.xml"),
         "-p2", str(project / "build" / "built.pdb"), "-s2", str(project / "build" / "built.xml"),
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


def _aggregate_cost(destination: Path) -> dict:
    from md_tools.build.record import read_record

    record = read_record(destination / "AIS.log")
    cost = record.get("collective_variable_cost")
    assert cost is not None, "the run recorded no global collective-variable cost"
    return cost


def _dispositions(cost) -> dict:
    return {entry["path_index"]: entry["disposition"] for entry in cost["per_path"]}


def _tree_digest(root: Path) -> dict:
    """Every file under the run root, by digest. The comparison for 'nothing was rewritten'."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix not in (".out", ".log")}


# --- Test B: paths completed across invocations -------------------------------------------------

def test_b_paths_completed_across_invocations_credit_only_the_current_one(project, tmp_path):
    """Invocation 1 finishes a strict subset; invocation 2 finishes the rest.

    The paths already complete at the start of invocation 2 must contribute exactly nothing to
    its segment, and its segment must be exactly what it evaluated: two paths, ten observations,
    twenty scalar evaluations. The cumulative must still be the whole campaign.
    """
    reference = tmp_path / "reference"
    _run(project, reference)
    want_cv = {index: _rows(reference / f"path_{index:04d}" / "cv.csv") for index in range(PATHS)}
    want_work = _rows(reference / "AIS_work.csv")

    destination = tmp_path / "staged"
    # Invocation 1: path 0 only. `--paths` selects a subset, so no global table is written yet.
    _run(project, destination, "--paths", "0")
    assert (destination / "path_0000" / "completed.json").is_file()
    assert not (destination / "path_0001" / "completed.json").exists()

    # Invocation 2: the whole campaign. Path 0 is already complete and must be skipped.
    second = _run(project, destination, "--resume")
    assert "already completed and verified" in second.stdout, second.stdout[-2000:]

    cost = _aggregate_cost(destination)

    # Cumulative: the entire campaign, every path exactly once.
    assert cost["cumulative"]["cv_observations"] == CAMPAIGN_OBSERVATIONS
    assert cost["cumulative"]["cv_evaluations"] == CAMPAIGN_EVALUATIONS

    # Segment: ONLY the two paths this invocation evaluated.
    assert cost["segment"]["cv_observations"] == 2 * OBSERVATIONS_PER_PATH
    assert cost["segment"]["cv_evaluations"] == 2 * EVALUATIONS_PER_PATH
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"]

    # Dispositions say which is which, and path 0 contributes zero.
    dispositions = _dispositions(cost)
    assert dispositions[0] == "already_complete"
    assert dispositions[1] == "fresh_and_completed"
    assert dispositions[2] == "fresh_and_completed"
    zero = next(e for e in cost["per_path"] if e["path_index"] == 0)
    assert zero["segment"]["cv_observations"] == 0
    assert zero["segment"]["cv_evaluations"] == 0
    assert zero["segment"]["wall_seconds"] == 0.0
    assert zero["cumulative"]["cv_observations"] == OBSERVATIONS_PER_PATH

    # And the science is the uninterrupted campaign's, path for path.
    for index in range(PATHS):
        assert _rows(destination / f"path_{index:04d}" / "cv.csv") == want_cv[index], (
            f"path {index} differs from the uninterrupted reference")
        steps = [r["protocol_step"] for r in _rows(destination / f"path_{index:04d}" / "cv.csv")]
        assert len(set(steps)) == len(steps) == ROWS_PER_PATH, "a path was evaluated twice"
    # The work table too: same rows, and every scientific column identical. `mpi_rank` records
    # which worker produced a row and is allowed to differ; nothing else is.
    got_work = _rows(destination / "AIS_work.csv")
    assert len(got_work) == len(want_work)
    differing = {column for mine, theirs in zip(got_work, want_work)
                 for column in mine if mine[column] != theirs[column]}
    assert differing <= {"mpi_rank"}, (
        f"completing the campaign across two invocations changed {sorted(differing)}")


# --- Test C: completed no-op re-entry -----------------------------------------------------------

def test_c_re_entering_a_completed_campaign_reports_exactly_zero_segment(project, tmp_path):
    """Nothing was computed, so the segment is zero -- not "small", and not the stored history.

    This is the case the old rule got most wrongly: re-entering a finished campaign reported the
    whole campaign's work as though it had just been performed, and would do so again on every
    re-entry.
    """
    destination = tmp_path / "completed"
    _run(project, destination)
    first = _aggregate_cost(destination)
    assert first["segment"]["cv_observations"] == CAMPAIGN_OBSERVATIONS, (
        "a fresh campaign's segment is the whole campaign")

    before = _tree_digest(destination)

    assert _run(project, destination, "--resume").returncode == 0
    cost = _aggregate_cost(destination)

    assert cost["segment"]["cv_observations"] == 0
    assert cost["segment"]["cv_evaluations"] == 0
    assert cost["segment"]["wall_seconds"] == 0.0
    assert set(_dispositions(cost).values()) == {"already_complete"}

    # Cumulative is unchanged, numerically.
    assert cost["cumulative"] == first["cumulative"]

    # And nothing scientific was rewritten. The `.out`/`.log` are this invocation's own record
    # and are excluded by `_tree_digest`; everything else must be byte-identical.
    after = _tree_digest(destination)
    assert after == before, (
        "re-entering a completed campaign rewrote: "
        f"{sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))}")


def test_c_a_second_re_entry_still_reports_zero(project, tmp_path):
    """The old rule grew the reported segment on every re-entry; this one cannot."""
    destination = tmp_path / "twice-entered"
    _run(project, destination)
    _run(project, destination, "--resume")
    once = _aggregate_cost(destination)
    _run(project, destination, "--resume")
    twice = _aggregate_cost(destination)
    assert once["segment"]["cv_observations"] == 0
    assert twice["segment"]["cv_observations"] == 0
    assert twice["cumulative"] == once["cumulative"]


def test_the_invocation_id_changes_between_invocations_and_touches_no_science(project, tmp_path):
    """It labels the launch and nothing else.

    Two invocations of the same completed campaign must differ in exactly one recorded field and
    agree on every scientific one -- which is what makes it safe to put in the record at all.
    """
    destination = tmp_path / "labelled"
    _run(project, destination)
    first = _aggregate_cost(destination)
    frames_before = (destination / "selected_source_frames.csv").read_text(encoding="utf-8")
    cv_before = (destination / "AIS_cv.csv").read_text(encoding="utf-8")
    work_before = (destination / "AIS_work.csv").read_text(encoding="utf-8")

    _run(project, destination, "--resume")
    second = _aggregate_cost(destination)

    assert first["invocation_id"] != second["invocation_id"], (
        "two launches shared one invocation id")
    assert len(second["invocation_id"]) == 32
    assert (destination / "selected_source_frames.csv").read_text(encoding="utf-8") \
        == frames_before, "the source-frame selection moved with the invocation id"
    assert (destination / "AIS_cv.csv").read_text(encoding="utf-8") == cv_before
    assert (destination / "AIS_work.csv").read_text(encoding="utf-8") == work_before

    # It is not in any path's scientific identity either.
    record = json.loads(
        (destination / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert "invocation_id" not in record
    assert "invocation" not in json.dumps(record.get("fingerprint", ""))
