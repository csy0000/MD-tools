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
                "work_measurement": "components",
                "observation_interval_steps": OBSERVE_EVERY,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": OBSERVE_EVERY, "info_printout": OBSERVE_EVERY,
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


def _rank_scoped_wrapper(tmp_path: Path, rank: int, boundary: str, allowed: str) -> Path:
    """A launcher shim that arms the fault on ONE rank only.

    AIS has no rank-scoped fault seam of its own, and adding one to the product for a test would
    be the wrong trade. The checkpoint fault is process-wide, so arming it under `mpirun` fires
    it on every rank at once -- which tests "the whole job failed", not "one worker died and the
    others were left holding a collective". Open MPI publishes the rank in the environment
    before the process starts, so the shim can arm the fault for exactly one of them and nothing
    in `md_tools` needs to know this test exists.
    """
    script = tmp_path / "one_rank_fails.sh"
    script.write_text(
        "#!/bin/sh\n"
        f'if [ "$OMPI_COMM_WORLD_RANK" = "{rank}" ]; then\n'
        f'  MD_TOOLS_CHECKPOINT_FAULT={boundary}\n'
        f'  MD_TOOLS_CHECKPOINT_FAULT_AFTER={allowed}\n'
        "  export MD_TOOLS_CHECKPOINT_FAULT MD_TOOLS_CHECKPOINT_FAULT_AFTER\n"
        "fi\n"
        'exec "$@"\n', encoding="utf-8")
    script.chmod(0o755)
    return script


@pytest.mark.parametrize("rank", [0, 1])
def test_one_rank_dying_fails_the_campaign_and_invents_no_completed_paths(project, tmp_path,
                                                                         rank):
    """A worker dies mid-campaign. The job must fail, and the tables must stay honest.

    AIS had continuation coverage under MPI and no injected-failure coverage. The danger here is
    not the crash: it is what survives it. Rank 0 assembles the aggregate by READING completion
    manifests off disk rather than gathering over MPI, so a campaign that lost a worker can
    still write `AIS_work.csv` and `AIS_cv.csv` -- and those files must then describe exactly
    the paths that genuinely finished, never the campaign that was requested. A table quietly
    short of paths, presented as the run's result, would bias any reweighting built on it.

    So: the launch must fail, no path may carry a completion marker it cannot support, and every
    row in the aggregate must belong to a path that really completed.

    The timeout is an assertion too. A rank that dies alone must not leave the others blocked in
    a collective; a job that hangs burns its allocation and reports nothing.
    """
    destination = tmp_path / f"rank{rank}-died"
    shim = _rank_scoped_wrapper(tmp_path, rank, "after-work-row", "1")

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)

    done = subprocess.run(
        ["mpirun", "-n", str(RANKS), str(shim), sys.executable,
         str(project / "AIS" / "AIS.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-ng", str(RANKS), "-odir", str(destination)],
        cwd=project / "AIS", capture_output=True, text=True, timeout=LAUNCH_TIMEOUT, env=base)

    assert done.returncode != 0, (
        "a campaign that lost a worker reported success:\n" + done.stdout[-3000:])

    finished = {index for index in range(PATHS)
                if (destination / f"path_{index:04d}" / "completed.json").is_file()}
    assert len(finished) < PATHS, (
        "every path completed although a rank was killed, so nothing was actually injected")

    # Every completion marker that DOES exist must still describe its own path correctly: a
    # crash must not leave a half-written marker that a later invocation would skip on.
    for index in sorted(finished):
        record = json.loads(
            (destination / f"path_{index:04d}" / "completed.json").read_text(encoding="utf-8"))
        assert record["path_index"] == index
        assert record["status"] == "completed"

    # And the aggregate, if one was written at all, must contain exactly the paths that finished
    # -- never a row for work that was not measured.
    aggregate = destination / "AIS_cv.csv"
    if aggregate.is_file():
        listed = {int(row["path_index"]) for row in _rows(aggregate)}
        assert listed <= finished, (
            f"the aggregate lists path(s) {sorted(listed - finished)} that never completed")


# --- the independent oracle ---------------------------------------------------------------------
#
# Test D used to compute `expected_segment` by summing the RESUMED output's own per-path segment
# counts and then asserting the global segment equalled that sum. That is internal consistency:
# it shows the aggregate adds up, and says nothing whatever about how many observations the
# resumed invocation should have evaluated. An implementation that credited itself with twice
# the work, or none, would satisfy it exactly as well.
#
# The oracle below is computed BEFORE the resume, from two things the resumed run does not get to
# choose: the committed checkpoint generations left by the interrupted run, and the CV grid this
# campaign was configured with. No cost counter is consulted -- those are the numbers under test.

#: R = 1 + S/d. Written out from the configuration, not read back from any output.
COMPLETE_OBSERVATIONS = 1 + SWITCHING // CV_EVERY


def _committed_grid_points(directory: Path) -> int:
    """How many CV grid points a partial path's committed checkpoint vouches for.

    From the generation's PROGRESS -- the protocol step it committed -- and the configured CV
    interval, which is what defines the grid. `cv_rows` is read only as a cross-check below; a
    count that agreed with itself while disagreeing with the checkpoint's progress is exactly
    the corruption the accounting is supposed to notice.
    """
    from md_tools.openmm.checkpoint import CheckpointError, read_committed

    try:
        committed = read_committed(directory)
    except (CheckpointError, FileNotFoundError):
        return 0
    state = (committed or {}).get("state") or {}
    step = state.get("protocol_step")
    if step is None:
        return 0
    step = int(step)
    # CHECKPOINT CADENCE IS NOT CV CADENCE. Checkpoints commit every parameter-update interval
    # (5 steps here); observations are written every CV interval (10). A committed step therefore
    # lands BETWEEN grid points routinely -- step 35 commits observations at 0, 10, 20 and 30 and
    # owes the one at 40. Requiring the committed step to sit on the CV grid asserted a
    # coincidence of two independent cadences, which is precisely the assumption to avoid.
    grid_points = 1 + step // CV_EVERY

    # CROSS-CHECK, not the oracle: the committed CSV prefix must hold exactly those rows.
    rows = state.get("cv_rows")
    if rows is not None:
        assert int(rows) == grid_points, (
            f"the checkpoint commits step {step} -- {grid_points} grid point(s) -- and claims "
            f"{rows} committed CV row(s); the two disagree")
    return grid_points


def _starting_dispositions(destination: Path, paths: int):
    """Each path's state before the resume, and how many observations it still owes.

    completed  -> 0 new; it is skipped and contributes nothing to the current segment
    partial    -> R minus the grid points its checkpoint committed; an uncommitted tail is NOT
                  retained work and is not counted
    fresh      -> R
    """
    expected = {}
    disposition = {}
    for index in range(paths):
        directory = destination / f"path_{index:04d}"
        if (directory / "completed.json").is_file():
            disposition[index] = "already_complete"
            expected[index] = 0
        elif directory.is_dir():
            committed = _committed_grid_points(directory)
            disposition[index] = "resumed_and_completed" if committed else "fresh_and_completed"
            expected[index] = COMPLETE_OBSERVATIONS - committed
        else:
            disposition[index] = "fresh_and_completed"
            expected[index] = COMPLETE_OBSERVATIONS
    return disposition, expected


def _per_rank_invocation_ids(destination: Path):
    """Every rank's own record of which launch it belonged to.

    A single id in the aggregate proves rank 0 wrote one. Rank AGREEMENT needs each rank's own
    evidence, which is why every rank records it in its own log.
    """
    from md_tools.build.record import read_record

    found = {}
    for path in sorted(destination.glob("AIS.log*")):
        record = read_record(path)
        if record.get("invocation_id"):
            found[path.name] = (record["invocation_id"], record.get("mpi_rank"))
    return found


# --- Test D: an interrupted two-rank campaign resumed under four ranks, on real CUDA -------------

def test_d_two_rank_interruption_resumed_under_four_ranks_on_cuda(project, tmp_path):
    """The case the invocation accounting exists for, on devices, with the world size changed.

    Four paths under TWO ranks, interrupted so that at least one path is fully complete and at
    least one has a committed partial prefix, then finished under FOUR. Path identity does not
    depend on the worker count -- global path n always writes AIS_traj000n.nc from its own source
    frame under its own seeds -- so every scientific table must equal an uninterrupted
    two-rank reference, and only execution metadata may differ.

    The accounting assertion is the point: the four-rank invocation must credit itself with the
    CVs it actually evaluated, not with the work the two-rank invocation did before it. Under the
    old rule the finished path's stored segment would have been summed in here, reporting work
    from a process that had already exited.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "reference"
    _launch(project, reference, ranks=2)
    want_cv = {index: _rows(reference / f"path_{index:04d}" / "cv.csv") for index in range(PATHS)}
    want_work = _rows(reference / "AIS_work.csv")
    want_aggregate = _rows(reference / "AIS_cv.csv")
    want_frames = (reference / "selected_source_frames.csv").read_text(encoding="utf-8")
    want_manifests = {index: _completion(reference, index) for index in range(PATHS)}

    # TWO ranks, interrupted part-way. `after-work-row` with a count that lets the first path
    # finish and stops the next one mid-switch, so the resume meets both an already-complete
    # path and a committed partial prefix.
    destination = tmp_path / "resumed"
    crashed = _launch(project, destination, ranks=2, expect=1,
                      environment={FAULT_ENVIRONMENT: "after-work-row",
                                   FAULT_AFTER_ENVIRONMENT: "9"})
    assert crashed.returncode != 0

    complete_before = {index for index in range(PATHS)
                       if (destination / f"path_{index:04d}" / "completed.json").is_file()}
    partial_before = {index for index in range(PATHS)
                      if index not in complete_before
                      and (destination / f"path_{index:04d}" / "cv.csv").is_file()}
    assert complete_before, "the interruption left no completed path, so nothing is skipped later"
    assert partial_before, "the interruption left no partial path, so no prefix is carried later"

    # THE ORACLE, taken from the interrupted tree BEFORE the resume touches it.
    starting, expected_new = _starting_dispositions(destination, PATHS)
    assert sum(1 for d in starting.values() if d == "already_complete") >= 1
    assert sum(1 for d in starting.values() if d == "resumed_and_completed") >= 1, (
        "no path carried a committed partial prefix, so the partial-path arithmetic is untested")
    assert 0 < sum(expected_new.values()) < PATHS * COMPLETE_OBSERVATIONS
    # Printed so the derived expectations are readable evidence rather than an invisible
    # intermediate: R, each path's starting disposition, and what it owes.
    print(f"@@ORACLE R={COMPLETE_OBSERVATIONS} (1 + {SWITCHING}/{CV_EVERY}) "
          f"dispositions={starting} expected_new={expected_new} "
          f"global_expected_segment={sum(expected_new.values())}")

    # FOUR ranks now: one path each, a different division of the same campaign.
    _launch(project, destination, "--resume", ranks=4)

    # RANK AGREEMENT, from each rank's own log rather than from the aggregate alone.
    per_rank = _per_rank_invocation_ids(destination)
    assert len(per_rank) >= 2, f"only {len(per_rank)} rank(s) recorded an invocation id"
    assert len({identity for identity, _rank in per_rank.values()}) == 1, (
        f"ranks disagree about which launch they belonged to: {per_rank}")

    cost = _aggregate_cost_record(destination)

    # All ranks agreed on one invocation id, and it is the one the aggregate reports.
    assert isinstance(cost["invocation_id"], str) and len(cost["invocation_id"]) == 32
    assert {identity for identity, _rank in per_rank.values()} == {cost["invocation_id"]}

    # Cumulative counts every completed path exactly once.
    assert cost["cumulative"]["cv_observations"] == PATHS * EXPECTED_ROWS
    assert cost["cumulative"]["cv_evaluations"] == PATHS * EXPECTED_ROWS * N_CV

    # The segment counts ONLY what this four-rank invocation evaluated. Every path that was
    # already complete contributes exactly zero, and the total is strictly less than the
    # campaign -- which is what the old rule could not report.
    by_index = {entry["path_index"]: entry for entry in cost["per_path"]}
    for index in range(PATHS):
        assert by_index[index]["disposition"] == starting[index], (
            f"path {index} was {starting[index]} before the resume and reports "
            f"{by_index[index]['disposition']}")
    for index in sorted(complete_before):
        assert by_index[index]["disposition"] == "already_complete"
        assert by_index[index]["segment"]["cv_observations"] == 0
        assert by_index[index]["segment"]["cv_evaluations"] == 0
        assert by_index[index]["segment"]["wall_seconds"] == 0.0
    for index in sorted(partial_before):
        assert by_index[index]["disposition"] == "resumed_and_completed"
        assert 0 < by_index[index]["segment"]["cv_observations"] <= EXPECTED_ROWS

    # THE ORACLE, computed before the resume from the checkpoints and the configured grid.
    for index in range(PATHS):
        assert by_index[index]["segment"]["cv_observations"] == expected_new[index], (
            f"path {index} ({starting[index]}) evaluated "
            f"{by_index[index]['segment']['cv_observations']} observation(s); the checkpoint it "
            f"resumed from and the {CV_EVERY}-step grid say it owed {expected_new[index]}")
        assert by_index[index]["segment"]["cv_evaluations"] == 2 * expected_new[index]
        assert by_index[index]["cumulative"]["cv_observations"] == COMPLETE_OBSERVATIONS
        assert by_index[index]["cumulative"]["cv_evaluations"] == 2 * COMPLETE_OBSERVATIONS
        if expected_new[index] == 0:
            assert by_index[index]["segment"]["wall_seconds"] == 0.0, (
                f"path {index} was skipped and still reported evaluation time")
        else:
            seconds = by_index[index]["segment"]["wall_seconds"]
            assert seconds >= 0.0 and seconds == seconds and seconds != float("inf")

    expected_segment = sum(expected_new.values())
    assert cost["segment"]["cv_observations"] == expected_segment, (
        f"the four-rank invocation reports {cost['segment']['cv_observations']} observation(s); "
        f"the pre-resume checkpoints and the configured grid say {expected_segment}")
    assert cost["cumulative"]["cv_observations"] == PATHS * COMPLETE_OBSERVATIONS
    assert cost["segment"]["cv_observations"] < cost["cumulative"]["cv_observations"], (
        "the resumed invocation claimed the whole campaign's work as its own")
    assert cost["segment"]["cv_evaluations"] == 2 * cost["segment"]["cv_observations"], (
        "two torsions per observation")

    # And the science is the two-rank reference's, exactly.
    for index in range(PATHS):
        assert _rows(destination / f"path_{index:04d}" / "cv.csv") == want_cv[index], (
            f"path {index} changed when the campaign was divided differently")
    assert _rows(destination / "AIS_cv.csv") == want_aggregate
    assert (destination / "selected_source_frames.csv").read_text(encoding="utf-8") \
        == want_frames, "the source-frame selection moved with the worker count"

    got_work = _rows(destination / "AIS_work.csv")
    assert len(got_work) == len(want_work)
    differing = {column for mine, theirs in zip(got_work, want_work)
                 for column in mine if mine[column] != theirs[column]}
    assert differing <= {"mpi_rank"}, (
        f"resuming under four ranks changed {sorted(differing)} in the work table")

    # Every path kept its source frame, seeds, filenames and work values.
    ignore = {"mpi_rank", "resumed", "outputs", "collective_variable_cost"}
    for index in range(PATHS):
        got = _completion(destination, index)
        assert (destination / f"AIS_traj{index:04d}.nc").is_file(), (
            f"path {index} did not write its own trajectory name")
        for field, value in want_manifests[index].items():
            if field in ignore:
                continue
            assert _scrub(got[field]) == _scrub(value), (
                f"path {index}: {field!r} changed under a different worker count")
        assert got["platform"] == "CUDA", got["platform"]


def _aggregate_cost_record(destination: Path) -> dict:
    from md_tools.build.record import read_record

    for candidate in sorted(destination.glob("AIS.log*")):
        record = read_record(candidate)
        cost = record.get("collective_variable_cost")
        if cost is not None:
            return cost
    raise AssertionError(f"no global collective-variable cost recorded under {destination}")


def test_a_malformed_record_on_a_nonzero_rank_refuses_collectively_without_touching_the_tree(
        project, tmp_path):
    """A damaged record owned by a rank that is not rank 0, under real MPI on real CUDA.

    Rank 0 assembles the aggregate and writes the shared files, so a malformed record it owns is
    the easy case -- the rank that would do the writing is the rank that refuses. The dangerous
    case is a record belonging to some OTHER rank: the refusal has to become collective before
    rank 0 writes anything, and the job has to stop rather than leave the others blocked in a
    collective while one of them exits.

    So this damages a path that a four-rank launch assigns to a nonzero rank, then asserts three
    things: the launch fails, nothing under the protected tree moved, and it did not hang. The
    subprocess timeout is the hang assertion.
    """
    import hashlib
    import os

    from md_tools.ais import paths_for_rank

    destination = tmp_path / "nonzero"
    _launch(project, destination, ranks=2)

    # A path owned by a rank other than 0 when the world is four wide.
    victim = next(index for rank in range(1, 4) for index in paths_for_rank(rank, 4, PATHS))
    marker = destination / f"path_{victim:04d}" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["collective_variable_cost"]["cumulative"]["cv_observations"] = "5"   # a string
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def tree(root):
        state = {}
        for path in sorted(root.rglob("*")):
            key = str(path.relative_to(root))
            if path.is_symlink():
                state[key] = ("symlink", os.readlink(path))
            elif path.is_dir():
                state[key] = ("dir", None)
            else:
                stat = path.stat()
                state[key] = ("file", hashlib.sha256(path.read_bytes()).hexdigest(),
                              stat.st_ino, stat.st_mtime_ns)
        return state

    before = tree(destination)
    refused = _launch(project, destination, "--resume", ranks=4, expect=1)
    assert refused.returncode != 0, refused.stdout[-2000:]

    changed = sorted(k for k in set(before) | set(tree(destination))
                     if before.get(k) != tree(destination).get(k))
    assert not changed, f"a collectively refused launch modified the tree: {changed}"

    combined = refused.stdout + refused.stderr
    assert "cv_observations" in combined or "cost" in combined.lower(), combined[-3000:]
