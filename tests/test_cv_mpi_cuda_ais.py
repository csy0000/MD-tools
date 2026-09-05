"""AIS with CV reporting and the work decomposition, under a REAL launcher on real CUDA.

WHY A SEPARATE FILE

    Two lanes existed and neither covered this. `test_cv_mpi_cuda_lanes.py` runs REST2 only.
    `test_md_run_mpi_gpu.py` runs a hundred AIS paths across four ranks on real devices with NO
    collective variables enabled. So distributed AIS was covered, CV-enabled AIS was covered
    serially under `--cpu`, and the intersection -- several ranks, real devices, CV reporting on,
    and an interruption in the middle -- was covered by neither, while the evidence report
    described the MPI+CUDA CV lane as though it were.

    The intersection is where the interesting failure lives. Paths are numbered GLOBALLY and
    distributed across ranks; each path writes its own CV series and completion manifest, and
    rank 0 alone assembles the aggregate from whatever finished. A resume may hand a path to a
    different rank than ran it the first time. Path identity, per-path cost and the aggregate
    must survive all of that, and none of it exists in a serial run.

WHAT IS COMPARED

    Everything, against an uninterrupted reference: every per-path CV CSV, the work table, each
    completion manifest, and the final aggregate. Not step grids -- values. A continuation that
    restored coordinates but not the integrator stream writes a different path onto a grid that
    lines up perfectly.

Every subprocess call carries a timeout, and a hang is a failure rather than a wait.
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
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

LAUNCH_TIMEOUT = 1800

RANKS = 2
PATHS = 4
SWITCHING = 40
UPDATE_EVERY = 5
OBSERVE_EVERY = 10
CV_EVERY = 10
N_CV = 2
EXPECTED_ROWS = SWITCHING // CV_EVERY + 1

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


def _require_mpi_and_cuda():
    import openmm

    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail("no CUDA platform is available; a CPU run is not CUDA evidence")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    _require_mpi_and_cuda()
    root = tmp_path_factory.mktemp("ais-mpi-cuda")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": PATHS, "switching_steps": SWITCHING,
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


def _launch(project: Path, destination: Path, *extra, ranks=RANKS, environment=None, expect=0):
    """No `--cpu`: if CUDA is unavailable this must fail rather than quietly use the CPU."""
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base.update(environment or {})
    done = subprocess.run(
        ["mpirun", "-n", str(ranks), sys.executable, str(project / "AIS" / "AIS.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-ng", str(ranks), "-odir", str(destination), *extra],
        cwd=project / "AIS", capture_output=True, text=True, timeout=LAUNCH_TIMEOUT, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _completion(destination: Path, path_index: int) -> dict:
    return json.loads(
        (destination / f"path_{path_index:04d}" / "completed.json").read_text(encoding="utf-8"))


#: Keys that record HOW a path was produced rather than WHAT it is, dropped before two runs are
#: compared. `*_seconds` are measured durations -- requiring two runs to agree on microseconds
#: asserts that the machine is idle, not that the science is right. `discarded_is_complete` says
#: whether this invocation threw away an uncommitted generation, which is true of a resume and
#: false of a run that was never interrupted, by definition. Everything else must match exactly.
EXECUTION_DETAIL = ("discarded_is_complete",)


def _scrub(value):
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()
                if not key.endswith("_seconds") and key not in EXECUTION_DETAIL}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def _record(destination: Path) -> dict:
    from md_tools.build.record import read_record

    return read_record(destination / "AIS.log")


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("ais-mpi-run") / "run"
    _launch(project, destination)
    return destination


def test_every_globally_numbered_path_ran_on_cuda_and_the_ranks_shared_them(completed):
    """Paths are numbered globally; the rank that ran one is an implementation detail it records."""
    ranks = set()
    for index in range(PATHS):
        record = _completion(completed, index)
        assert record["path_index"] == index, "a path directory holds another path's manifest"
        assert record["platform"] == "CUDA", (
            f"path {index} ran on {record['platform']}, and a CPU run is not CUDA evidence")
        ranks.add(int(record["mpi_rank"]))
    assert len(ranks) == RANKS, (
        f"every path was run by rank(s) {sorted(ranks)}: the paths were not distributed across "
        f"the {RANKS} ranks, so nothing about distribution was exercised")


def test_every_path_wrote_its_own_cv_series(completed):
    for index in range(PATHS):
        rows = _rows(completed / f"path_{index:04d}" / "cv.csv")
        assert len(rows) == EXPECTED_ROWS
        assert {"phi", "psi"} <= set(rows[0])
        assert all(int(row["path_index"]) == index for row in rows), (
            f"path {index}'s series carries another path's identity")
        assert [int(r["protocol_step"]) for r in rows] \
            == list(range(0, SWITCHING + 1, CV_EVERY))


def test_the_work_table_keeps_the_three_scaling_groups_for_hummer_szabo(completed):
    """Non-scaled, square-root-scaled and linearly-scaled, with units and accumulated totals.

    These are the components a later Hummer-Szabo reweighting needs: the basis
    `U = U_non_scaled + sqrt(lambda) U_sqrt_scaled + lambda U_lin_scaled` cannot be re-evaluated
    at another lambda from a total alone.
    """
    from md_tools.ais.decomposition import GROUPS

    rows = _rows(completed / "AIS_work.csv")
    assert rows, "the work table is empty"
    header = set(rows[0])

    for group in GROUPS:
        delta = f"delta_work_{group}_kj_mol"
        total = f"total_work_{group}_kj_mol"
        assert delta in header, f"{delta} is missing: the per-update component was not retained"
        assert total in header, f"{total} is missing: the accumulated component was not retained"
        # Units are in the column names, and the values are real numbers rather than blanks.
        assert delta.endswith("_kj_mol") and total.endswith("_kj_mol")
        assert all(row[total] not in ("", None) for row in rows)

    # The identity the schema states: the components sum to the total, per row.
    for row in rows[:20]:
        components = sum(float(row[f"delta_work_{group}_kj_mol"]) for group in GROUPS)
        assert abs(components - float(row["delta_work_kj_mol"])) < 1e-3, row


def test_per_path_and_aggregate_cv_cost_are_correct(completed):
    """Two scopes per path, and a documented, auditable sum over paths."""
    for index in range(PATHS):
        cost = _completion(completed, index)["collective_variable_cost"]
        assert cost["cumulative"]["cv_observations"] == EXPECTED_ROWS
        assert cost["cumulative"]["cv_evaluations"] == EXPECTED_ROWS * N_CV, (
            "an observation of a two-torsion definition is two scalar evaluations")
        assert cost["segment"] == cost["cumulative"], "a fresh path's scopes are equal"

    aggregate = _record(completed)["collective_variable_cost"]
    assert aggregate["aggregation"] == "sum over completed paths"
    assert len(aggregate["per_path"]) == PATHS, "the per-path records make the total auditable"
    assert aggregate["cumulative"]["cv_observations"] == PATHS * EXPECTED_ROWS
    assert aggregate["cumulative"]["cv_evaluations"] == PATHS * EXPECTED_ROWS * N_CV
    assert sum(e["cumulative"]["cv_observations"] for e in aggregate["per_path"]) \
        == aggregate["cumulative"]["cv_observations"]

    # And the aggregate table is named in the run's own output inventory.
    outputs = _record(completed)["outputs"]
    assert "collective_variables" in outputs, (
        "the aggregate CV table is not named in the run's output inventory")
    assert (completed / outputs["collective_variables"]["path"]).is_file()


def test_the_aggregate_table_holds_every_path(completed):
    rows = _rows(completed / "AIS_cv.csv")
    assert {int(row["path_index"]) for row in rows} == set(range(PATHS))
    assert len(rows) == PATHS * EXPECTED_ROWS


def test_interruption_and_resume_under_mpi_reproduce_every_output(project, tmp_path):
    """The whole promise, compared against an uninterrupted reference.

    Every per-path CV series, the work table, each completion manifest and the final aggregate.
    A resume may schedule a path onto a different rank than ran it first; path identity and the
    aggregate must not depend on which rank that was, so `mpi_rank` is the one field allowed to
    differ.
    """
    reference = tmp_path / "reference"
    _launch(project, reference)
    want_cv = {index: _rows(reference / f"path_{index:04d}" / "cv.csv") for index in range(PATHS)}
    want_work = _rows(reference / "AIS_work.csv")
    want_aggregate = _rows(reference / "AIS_cv.csv")
    want_manifest = {index: _completion(reference, index) for index in range(PATHS)}

    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "resumed"
    crashed = _launch(project, destination, expect=1,
                      environment={FAULT_ENVIRONMENT: "after-work-row",
                                   FAULT_AFTER_ENVIRONMENT: "2"})
    assert crashed.returncode != 0
    _launch(project, destination, "--resume")

    for index in range(PATHS):
        assert _rows(destination / f"path_{index:04d}" / "cv.csv") == want_cv[index], (
            f"path {index}'s CV series differs from the uninterrupted reference")

    assert _rows(destination / "AIS_work.csv") == want_work, "the work table differs"
    assert _rows(destination / "AIS_cv.csv") == want_aggregate, "the CV aggregate differs"

    # `mpi_rank` and `resumed` are allowed to differ: they describe HOW the path was produced,
    # not what it is. So is `wall_seconds` inside the CV cost -- it is a measured duration, and
    # requiring two runs to spend the same number of microseconds would be asserting that the
    # machine is idle rather than that the science is right. The COUNTERS in that block are
    # compared exactly, below.
    ignore = {"mpi_rank", "resumed", "outputs", "collective_variable_cost"}
    split: list[bool] = []
    for index in range(PATHS):
        got = _completion(destination, index)
        for field, value in want_manifest[index].items():
            if field in ignore:
                continue
            assert _scrub(got[field]) == _scrub(value), (
                f"path {index}: completion manifest field {field!r} differs from the reference")

        cost = got["collective_variable_cost"]
        wanted = want_manifest[index]["collective_variable_cost"]
        assert cost["cumulative"]["cv_observations"] \
            == wanted["cumulative"]["cv_observations"] == EXPECTED_ROWS
        assert cost["cumulative"]["cv_evaluations"] \
            == wanted["cumulative"]["cv_evaluations"] == EXPECTED_ROWS * N_CV
        assert cost["cumulative"]["wall_seconds"] >= 0.0
        assert cost["segment"]["cv_observations"] <= cost["cumulative"]["cv_observations"]
        split.append(cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"])

    # Only the path that was actually MID-FLIGHT when the crash landed has its work divided
    # between two invocations. The paths queued behind it ran wholly inside the resuming one, so
    # their segment legitimately equals their cumulative -- asserting otherwise would demand a
    # split that never happened. What must be true is that the interruption split at least one
    # path, and that the split path's earlier work was carried rather than restarted.
    assert any(split), (
        "no path's cost is divided between two invocations, so the interruption did not land "
        "mid-path and the carrying of earlier work was never exercised")


def test_a_resume_under_a_different_rank_count_keeps_path_identity_and_the_aggregate(
        project, tmp_path):
    """The rescheduling case: crash under two ranks, finish under four.

    Path ids are GLOBAL and a given id always owns the same file name, so a restart under a
    different worker count is supposed to land on the same trajectories rather than silently
    reshuffling which path is which -- the scheduling is an implementation detail of how the
    work was divided, not part of what the paths ARE. Nothing exercised that: every resume in
    the suite used the same rank count it crashed with, which is precisely the case where a
    reshuffle cannot show up.

    Every per-path series, the work table and the aggregate must match a reference produced in
    one uninterrupted two-rank run.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "reference"
    _launch(project, reference)
    want_cv = {index: _rows(reference / f"path_{index:04d}" / "cv.csv") for index in range(PATHS)}
    want_aggregate = _rows(reference / "AIS_cv.csv")
    want_work = _rows(reference / "AIS_work.csv")

    destination = tmp_path / "rescheduled"
    crashed = _launch(project, destination, ranks=2, expect=1,
                      environment={FAULT_ENVIRONMENT: "after-work-row",
                                   FAULT_AFTER_ENVIRONMENT: "2"})
    assert crashed.returncode != 0

    # FOUR ranks now, for four paths: one path each, a different division of the same work.
    _launch(project, destination, "--resume", ranks=4)

    ranks = {int(_completion(destination, index)["mpi_rank"]) for index in range(PATHS)}
    assert len(ranks) > 1, f"the resumed run put every path on rank(s) {sorted(ranks)}"

    for index in range(PATHS):
        assert _completion(destination, index)["path_index"] == index, (
            f"path {index}'s directory holds another path's manifest after rescheduling")
        assert _rows(destination / f"path_{index:04d}" / "cv.csv") == want_cv[index], (
            f"path {index} differs from the reference after being rescheduled onto another rank")

    assert _rows(destination / "AIS_cv.csv") == want_aggregate, (
        "the aggregate changed when the work was divided differently")

    # The work table carries an `mpi_rank` column, which is a record of WHICH worker produced a
    # row and must change when the work is divided differently -- that is the whole point of
    # rescheduling. Asserting that it is the ONLY column that changes is the stronger claim:
    # every measured quantity, every identifier and every accumulated total is independent of
    # how the paths were distributed.
    got_work = _rows(destination / "AIS_work.csv")
    assert len(got_work) == len(want_work), "the work table changed length"
    differing = {column
                 for mine, theirs in zip(got_work, want_work)
                 for column in mine
                 if mine[column] != theirs[column]}
    assert differing <= {"mpi_rank"}, (
        f"rescheduling changed {sorted(differing)} in the work table; only mpi_rank may differ")
