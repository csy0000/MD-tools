"""`md-openmm md-run` under `mpirun`, on real CUDA: a ladder, and a hundred AIS paths.

Everything here needs a launcher and a GPU, so none of it can be a fast test. It is also where the
defects actually were: dispatch, file races between ranks, box handling, device placement and path
identity are all properties of a *launch*, and every one of them passed the fast lane while being
broken.

Deliberately tiny -- a few hundred steps, ten-step switching paths. What is asserted is the
contract: which files exist, which rank owns which path, that the platform really was CUDA, and
that a rerun does not overwrite work that was already measured. Not a number that depends on the
system.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.build.record import read_record

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _require_mpi():
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")


def _cli(cwd: Path, *args, timeout=1800):
    return subprocess.run(CLI + list(args), cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def _md_run(cwd: Path, *args, ranks: int = 1, timeout=3600):
    """`md-openmm md-run`, launched the way the documentation says to launch it."""
    launch = ["mpirun", "-n", str(ranks)] if ranks > 1 else []
    return subprocess.run(launch + ["md-openmm", "md-run", *args],
                          cwd=cwd, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One implicit peptide, built once. Implicit so there is no box to solvate and no barostat."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    work = tmp_path_factory.mktemp("md-run-mpi")
    (work / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    done = _cli(work, "build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
                "-log", "built.log", "--config", str(work / "sys.config"))
    assert done.returncode == 0, done.stdout + done.stderr
    return work


# --- a two-rank REST2 ladder ------------------------------------------------------------------

@pytest.fixture(scope="module")
def ladder(built):
    _require_mpi()
    config = built / "rest2.config"
    config.write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "dynamics": {"seed": 3},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50},
        "rest2": {"number_of_replicas": 2, "tau_max": 0.3,
                  "exchange_interval_steps": 50, "number_of_exchanges": 4},
        "reporting": {"solute_printout": 25, "system_printout": 50,
                      "checkpoint_printout": 50}}, sort_keys=False), encoding="utf-8")
    assert _cli(built, "build-md", "-odir", "./rest2", "--config", str(config)).returncode == 0

    out = built / "rest2"
    previous = None
    for stage in ("min", "eq_nvt_posres", "eq_nvt_posres_2", "eq_nvt_free"):
        argv = ["-i", f"{stage}.in", "-p", "../built.pdb", "-x", "../built.xml",
                "-r", f"{stage}.xml", "-log", f"{stage}.log"]
        if previous:
            argv += ["-c", f"{previous}.xml"]
        done = _md_run(out, *argv)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        previous = stage

    done = _md_run(out, "-ng", "2", "-i", "REST2.in", "-p", "../built.pdb", "-x", "../built.xml",
                   "-c", f"{previous}.xml", "-log", "REST2.log", ranks=2)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return out


def test_a_two_rank_ladder_writes_one_trajectory_per_state(ladder):
    """Fixed thermodynamic states, not walkers: `remd0.nc` and `remd1.nc`, and no third."""
    states = sorted(p.name for p in ladder.glob("remd*.nc"))
    assert states == ["remd0.nc", "remd1.nc"], states
    assert (ladder / "rem.log").is_file()
    assert (ladder / "restart.json").is_file()


def test_each_rank_kept_its_own_record_and_ran_on_cuda(ladder):
    """Two ranks, two records, two devices. A rank that lost its GPU has to be able to say so."""
    assert (ladder / "REST2.log").is_file() and (ladder / "REST2.log.rank01").is_file()
    text = (ladder / "REST2.out").read_text(encoding="utf-8")
    other = (ladder / "REST2.out.rank01").read_text(encoding="utf-8")
    assert "platform           : CUDA" in text, text
    assert "platform           : CUDA" in other, other
    # Deterministic placement, recorded: not "a GPU" but which one, and by what rule.
    assert "one rank per device" in text, text
    assert "this process drives: state(s) [0]" in text, text
    assert "this process drives: state(s) [1]" in other, other


def test_a_world_that_is_not_the_state_count_is_refused_before_integrating(ladder):
    _require_mpi()
    done = _md_run(ladder, "-ng", "3", "-i", "REST2.in", "-p", "../built.pdb",
                   "-x", "../built.xml", "-log", "refused.log", ranks=3)
    assert done.returncode != 0
    message = done.stdout + done.stderr
    # All three numbers, so the reader knows which one to change.
    for number in ("replicas in the configuration : 2", "-ng on the command line       : 3",
                   "MPI world size                : 3"):
        assert number in message, message
    assert not (ladder / "refused.log").exists(), "a refused launch wrote a log"


# --- a hundred AIS paths ----------------------------------------------------------------------

def _ais_project(built: Path, name: str, paths: int) -> Path:
    config = built / f"{name}.config"
    config.write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "dynamics": {"seed": 5},
        "ais": {"number_of_paths": paths, "tau_start": 0.5, "tau_end": 0.0,
                "switching_steps": 10, "observation_interval_steps": 5},
        "ais_source": {"trajectory": "../source/cMD.dcd"}}, sort_keys=False), encoding="utf-8")
    assert _cli(built, "build-md", "-odir", f"./{name}", "--config", str(config)).returncode == 0
    return built / name


@pytest.fixture(scope="module")
def source(built):
    """A short fixed-tau run at tau = 0.5: the ensemble the paths anneal away from."""
    config = built / "source.config"
    config.write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "dynamics": {"tau": 0.5, "seed": 9},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
                   "production_steps": 2000},
        "reporting": {"solute_printout": 10, "system_printout": 100,
                      "checkpoint_printout": 1000}}, sort_keys=False), encoding="utf-8")
    assert _cli(built, "build-md", "-odir", "./source", "--config", str(config)).returncode == 0
    done = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml"],
                          cwd=built / "source", capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return built / "source" / "cMD.dcd"


@pytest.fixture(scope="module")
def hundred(built, source):
    _require_mpi()
    out = _ais_project(built, "ais100", 100)
    done = _md_run(out, "-ng", "4", "-i", "AIS.in", "-p", "../built.pdb", "-x", "../built.xml",
                   "-source-traj", "../source/cMD.dcd", "-odir", ".", "-log", "AIS.log", ranks=4)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return out


def test_a_hundred_paths_write_exactly_a_hundred_named_trajectories(hundred):
    """`AIS_traj0000.nc` through `AIS_traj0099.nc`. No more, no fewer, no gaps."""
    written = sorted(p.name for p in hundred.glob("AIS_traj*.nc"))
    assert written == [f"AIS_traj{i:04d}.nc" for i in range(100)], written[:5] + written[-5:]


def test_every_path_trajectory_holds_its_endpoints(hundred):
    """switching_steps=10 at an interval of 5 is 3 frames: steps 0, 5 and 10."""
    import mdtraj

    for path_id in (0, 42, 99):
        trajectory = mdtraj.load(str(hundred / f"AIS_traj{path_id:04d}.nc"),
                                 top=str(hundred.parent / "built.pdb"))
        assert trajectory.n_frames == 3, (path_id, trajectory.n_frames)
        rows = list(csv.DictReader((hundred / f"path_{path_id:04d}" / "observations.csv").open()))
        assert [int(r["protocol_step"]) for r in rows] == [0, 5, 10], rows
        assert float(rows[0]["tau"]) == 0.5 and float(rows[-1]["tau"]) == 0.0


def test_the_work_table_is_complete_uniquely_keyed_and_deterministically_ordered(hundred):
    rows = list(csv.DictReader((hundred / "AIS_work.csv").open()))
    keys = [(int(r["path_id"]), int(r["switch_step"])) for r in rows]
    assert len(keys) == len(set(keys)), "a (path, step) pair appears twice"
    assert keys == sorted(keys), "the table is not sorted by path then step"
    assert keys == [(path, step) for path in range(100) for step in (0, 5, 10)]
    # Every row names the trajectory its configuration is in, so the table is self-contained.
    assert all(r["trajectory"] == f"AIS_traj{int(r['path_id']):04d}.nc" for r in rows)


def test_the_paths_were_shared_out_and_ran_on_cuda(hundred):
    """Four workers, a hundred paths, and every one of them on a GPU by default."""
    rows = list(csv.DictReader((hundred / "AIS_paths.csv").open()))
    assert len(rows) == 100
    ranks = sorted({int(r["mpi_rank"]) for r in rows})
    assert ranks == [0, 1, 2, 3], ranks
    counts = [sum(1 for r in rows if int(r["mpi_rank"]) == rank) for rank in ranks]
    assert counts == [25, 25, 25, 25], counts

    record = read_record(hundred / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["acceleration"]["requested_policy"] == "default-cuda"
    assert record["acceleration"]["cuda_device_index"] is not None
    assert all(json.loads((hundred / f"path_{i:04d}" / "completed.json").read_text())["platform"]
               == "CUDA" for i in (0, 50, 99))


# --- worker count and restart -----------------------------------------------------------------

def test_path_identity_does_not_depend_on_the_worker_count(built, source):
    """Same path, same source frame, same seeds, same filename -- at one worker and at three.

    Only identity. The work VALUES are not asserted equal: CUDA mixed precision does not reduce in
    a fixed order, and the difference grows along a chaotic trajectory. Claiming bit-identity here
    would be claiming something about the hardware that is not true.
    """
    _require_mpi()
    tables = {}
    for ranks in (1, 3):
        out = _ais_project(built, f"ais_n{ranks}", 12)
        done = _md_run(out, "-ng", str(ranks), "-i", "AIS.in", "-p", "../built.pdb",
                       "-x", "../built.xml", "-source-traj", "../source/cMD.dcd",
                       "-odir", ".", "-log", "AIS.log", ranks=ranks)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        tables[ranks] = {r["path_index"]: r
                         for r in csv.DictReader((out / "AIS_paths.csv").open())}

    one, three = tables[1], tables[3]
    assert one.keys() == three.keys()
    for path_id in one:
        for field in ("source_frame_index", "trajectory", "integrator_seed", "velocity_seed"):
            assert one[path_id][field] == three[path_id][field], (path_id, field)
    assert {r["mpi_rank"] for r in three.values()} == {"0", "1", "2"}


def test_a_rerun_does_not_touch_a_path_that_already_completed(built, source):
    """A completed path is skipped, never appended to and never silently redone."""
    out = _ais_project(built, "ais_restart", 4)
    argv = ("-i", "AIS.in", "-p", "../built.pdb", "-x", "../built.xml",
            "-source-traj", "../source/cMD.dcd", "-odir", ".", "-log", "AIS.log")
    assert _md_run(out, *argv).returncode == 0

    def fingerprint():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.glob("AIS_traj*.nc"))}

    before = fingerprint()
    assert len(before) == 4
    frames_before = (out / "selected_source_frames.csv").read_text(encoding="utf-8")

    assert _md_run(out, *argv, "--overwrite").returncode == 0
    assert fingerprint() == before, "a completed path's trajectory was rewritten"
    assert (out / "selected_source_frames.csv").read_text(encoding="utf-8") == frames_before
