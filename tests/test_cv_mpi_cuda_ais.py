"""AIS with CV reporting and the two-state Hamiltonian, under a REAL launcher on real CUDA.

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

    A path that was never interrupted is compared with nothing but itself: a CUDA run is not
    promised to be bit-identical to another process's. A path completed BEFORE an interruption
    must survive the resume byte for byte. A path interrupted mid-switch must continue from
    EXACTLY its committed generation -- every stream prefix it vouches for, lambda, the carried
    cumulative work, the counters -- and finish complete and valid. Past the resume point it is a
    new realisation of the same switching process: the two-state mixing force's inner Contexts
    keep atom-ordering state no checkpoint captures, so row-for-row equality with an
    uninterrupted CUDA reference is not a property this runtime has (decided 2026-09-16).

    Against the uninterrupted reference, what must match is IDENTITY and STRUCTURE: source frames,
    seeds, trajectory names, fingerprints, the step and lambda grids, which rows carry a saved
    coordinate, and the CV and work accounting.

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

from tests.test_cv_cuda_lanes import (  # noqa: E402 - the one definition of exact restoration
    assert_restored_and_complete, committed_instant, write_scaled_v0)

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
    (root / "build").mkdir(exist_ok=True)
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
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
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./AIS-run1", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    # V0 = the REST2 state at tau = 0.5 of built.xml, V1 = built.xml: the two end states.
    write_scaled_v0(root)
    return root


#: `-p/-s` is V0, `-p2/-s2` is V1, as absolute paths under the project root.
def _end_states(project: Path) -> list[str]:
    return ["-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "V0.xml"),
            "-p2", str(project / "build" / "built.pdb"),
            "-s2", str(project / "build" / "built.xml")]


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
        ["mpirun", "-n", str(ranks), sys.executable, str(project / "AIS-run1" / "AIS.py"),
         *_end_states(project),
         "-source-traj", str(project / "source.dcd"),
         "-ng", str(ranks), "-odir", str(destination), *extra],
        cwd=project / "AIS-run1", capture_output=True, text=True, timeout=LAUNCH_TIMEOUT, env=base)
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


def _record(destination: Path) -> dict:
    from md_tools.build.record import read_record

    return read_record(destination / "AIS.log")


#: What a path IS, independent of how it was integrated: compared exactly against a reference.
IDENTITY_FIELDS = ("status", "path_index", "source_frame_index", "observations", "frames",
                   "state_rows", "cv_rows", "ais_schema", "integrator_seed", "velocity_seed",
                   "trajectory", "fingerprint", "platform")

#: The work table's STRUCTURE: every column that does not carry a value a CUDA continuation
#: legitimately changes. Which rows name a saved coordinate is structure too, so the emptiness of
#: the potential cells is compared rather than their values.
WORK_STRUCTURE = ("path_id", "source_frame", "observation_index", "switch_step",
                  "coordinate_frame_index", "lambda_before", "lambda_after", "trajectory")


def _work_structure(rows):
    return [tuple(row[name] for name in WORK_STRUCTURE)
            + (row["potential_direct_kj_mol"] == "",) for row in rows]


def _cv_structure(rows):
    return [(row["path_index"], row["source_frame_index"], row["protocol_step"], row["lambda"],
             row["observation_index"], row["coordinate_frame_index"]) for row in rows]


def _path_files(destination: Path, index: int) -> dict:
    """A completed path's scientific files, by content, for 'this path was not touched'."""
    directory = destination / f"path_{index:04d}"
    names = ["completed.json", "observations.csv", "cv.csv", "system.csv"]
    found = {name: (directory / name).read_bytes() for name in names
             if (directory / name).is_file()}
    found["trajectory"] = (destination / f"AIS_traj{index:04d}.nc").read_bytes()
    return found


def _before_resume(destination: Path):
    """Each path's state before a resume: completed files, a committed instant, or nothing."""
    completed_files, instants = {}, {}
    for index in range(PATHS):
        directory = destination / f"path_{index:04d}"
        if (directory / "completed.json").is_file():
            completed_files[index] = _path_files(destination, index)
        else:
            instants[index] = committed_instant(directory)
    return completed_files, instants


def _assert_resumed_campaign(destination: Path, reference: Path, completed_files, instants):
    """The whole resumed campaign against what it was before the resume and the reference."""
    for index in range(PATHS):
        if index in completed_files:
            assert _path_files(destination, index) == completed_files[index], (
                f"path {index} was complete before the resume and was rewritten by it")
            assert_restored_and_complete(destination, index, None, switching_steps=SWITCHING)
        else:
            assert_restored_and_complete(destination, index, instants[index],
                                         switching_steps=SWITCHING)
        got, want = _completion(destination, index), _completion(reference, index)
        for field in IDENTITY_FIELDS:
            assert got[field] == want[field], (
                f"path {index}: {field!r} differs from the uninterrupted reference")
        assert _cv_structure(_rows(destination / f"path_{index:04d}" / "cv.csv")) \
            == _cv_structure(_rows(reference / f"path_{index:04d}" / "cv.csv")), (
                f"path {index}'s CV grid differs from the uninterrupted reference")
    assert _work_structure(_rows(destination / "AIS_work.csv")) \
        == _work_structure(_rows(reference / "AIS_work.csv")), "the work table's structure differs"
    assert _cv_structure(_rows(destination / "AIS_cv.csv")) \
        == _cv_structure(_rows(reference / "AIS_cv.csv")), "the CV aggregate's structure differs"
    # The aggregate tables are assembled from the per-path files, value for value.
    totals = {int(row["path_index"]): float(row["total_work_kj_mol"])
              for row in _rows(destination / "AIS_paths.csv")}
    for index in range(PATHS):
        assert totals[index] == float(_completion(destination, index)["total_work_kj_mol"])


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


def test_the_work_table_keeps_the_end_state_potentials_for_hummer_szabo(completed):
    """V0, V1 and the direct potential at every saved coordinate, with units, under MPI on CUDA.

    These are what a later Hummer-Szabo reweighting needs: the potential at another lambda is
    `(1 - lambda) V0 + lambda V1`, which a total alone cannot give. The mixture is checked against
    the direct CUDA potential on every aligned row, at the precision the run recorded; a row with
    no saved coordinate carries none of the three.
    """
    from md_tools.ais.two_state import OBSERVATION_POTENTIAL_COLUMNS, identity_tolerance

    rows = _rows(completed / "AIS_work.csv")
    assert rows, "the work table is empty"
    header = set(rows[0])
    assert set(OBSERVATION_POTENTIAL_COLUMNS) <= header, sorted(header)
    assert {"lambda_before", "lambda_after", "delta_work_kj_mol", "total_work_kj_mol"} <= header
    assert all(name.endswith("_kj_mol") for name in OBSERVATION_POTENTIAL_COLUMNS)

    precision = (_record(completed)["acceleration"].get("cuda_precision") or "mixed")
    aligned = 0
    for row in rows:
        assert row["total_work_kj_mol"] not in ("", None)
        if row["coordinate_frame_index"] == "":
            assert all(row[name] == "" for name in OBSERVATION_POTENTIAL_COLUMNS), row
            continue
        lam = float(row["lambda_after"])
        v0, v1 = float(row["potential_v0_kj_mol"]), float(row["potential_v1_kj_mol"])
        direct = float(row["potential_direct_kj_mol"])
        allowed = identity_tolerance(max(abs(v0), abs(v1), abs(direct)), precision=precision)
        assert abs((1.0 - lam) * v0 + lam * v1 - direct) <= allowed, row
        aligned += 1
    assert aligned >= PATHS, f"only {aligned} frame-aligned row(s) across {PATHS} paths"


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


def test_interruption_and_resume_under_mpi_restore_exactly_and_complete(project, tmp_path):
    """The whole promise, under MPI: exact restoration, complete paths, a consistent aggregate.

    A resume may schedule a path onto a different rank than ran it first; path identity and the
    aggregate must not depend on which rank that was. Each path is held to what it was just
    before the resume -- untouched if it had completed, continued from exactly its committed
    generation if it had not -- and to the uninterrupted reference's identity and structure.
    """
    reference = tmp_path / "reference"
    _launch(project, reference)
    want_manifest = {index: _completion(reference, index) for index in range(PATHS)}

    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "resumed"
    crashed = _launch(project, destination, expect=1,
                      environment={FAULT_ENVIRONMENT: "after-work-row",
                                   FAULT_AFTER_ENVIRONMENT: "2"})
    assert crashed.returncode != 0
    completed_files, instants = _before_resume(destination)
    assert any(instant is not None for instant in instants.values()), (
        "no interrupted path had a committed generation, so exact restoration is untested")
    _launch(project, destination, "--resume")

    _assert_resumed_campaign(destination, reference, completed_files, instants)

    split: list[bool] = []
    for index in range(PATHS):
        got = _completion(destination, index)
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

    Every path's identity, grid and structure must match a reference produced in one uninterrupted
    two-rank run, and every path must continue from exactly what it had committed.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "reference"
    _launch(project, reference)

    destination = tmp_path / "rescheduled"
    crashed = _launch(project, destination, ranks=2, expect=1,
                      environment={FAULT_ENVIRONMENT: "after-work-row",
                                   FAULT_AFTER_ENVIRONMENT: "2"})
    assert crashed.returncode != 0
    completed_files, instants = _before_resume(destination)

    # FOUR ranks now, for four paths: one path each, a different division of the same work.
    _launch(project, destination, "--resume", ranks=4)

    ranks = {int(_completion(destination, index)["mpi_rank"]) for index in range(PATHS)}
    assert len(ranks) > 1, f"the resumed run put every path on rank(s) {sorted(ranks)}"

    # Path identity after rescheduling, each path's exact continuation, and the aggregate's
    # structure. `mpi_rank` is a record of WHICH worker produced a row and must change when the
    # work is divided differently; it is deliberately outside every comparison.
    _assert_resumed_campaign(destination, reference, completed_files, instants)


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
         str(project / "AIS-run1" / "AIS.py"),
         *_end_states(project),
         "-source-traj", str(project / "source.dcd"),
         "-ng", str(RANKS), "-odir", str(destination)],
        cwd=project / "AIS-run1", capture_output=True, text=True, timeout=LAUNCH_TIMEOUT, env=base)

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
    frame under its own seeds -- so identity and structure must equal an uninterrupted two-rank
    reference, and every path must continue from exactly what it had committed. (On CUDA a
    continuation past its committed generation is a new realisation, so values after a resume
    point are not compared with the reference.)

    The accounting assertion is the point: the four-rank invocation must credit itself with the
    CVs it actually evaluated, not with the work the two-rank invocation did before it. Under the
    old rule the finished path's stored segment would have been summed in here, reporting work
    from a process that had already exited.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / "reference"
    _launch(project, reference, ranks=2)
    want_frames = (reference / "selected_source_frames.csv").read_text(encoding="utf-8")

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
    completed_files, instants = _before_resume(destination)

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

    # And the science: completed paths untouched, partial ones continued exactly, every path
    # complete, and identity and structure the two-rank reference's.
    _assert_resumed_campaign(destination, reference, completed_files, instants)
    assert (destination / "selected_source_frames.csv").read_text(encoding="utf-8") \
        == want_frames, "the source-frame selection moved with the worker count"
    for index in range(PATHS):
        assert (destination / f"AIS_traj{index:04d}.nc").is_file(), (
            f"path {index} did not write its own trajectory name")


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
