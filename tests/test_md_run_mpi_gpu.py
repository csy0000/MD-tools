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
        argv = ["-i", f"{stage}.in", "-p", "../built.pdb", "-s", "../built.xml",
                "-r", f"{stage}.xml", "-log", f"{stage}.log"]
        if previous:
            argv += ["-c", f"{previous}.xml"]
        done = _md_run(out, *argv)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        previous = stage

    done = _md_run(out, "-ng", "2", "-i", "REST2.in", "-p", "../built.pdb", "-s", "../built.xml",
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
                   "-s", "../built.xml", "-log", "refused.log", ranks=3)
    assert done.returncode != 0
    message = done.stdout + done.stderr
    # All four numbers, so the reader knows which one is the odd one out.
    for number in ("launcher world size           : 3", "MPI communicator size         : 3",
                   "-ng on the command line       : 3", "replicas in the configuration : 2"):
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
    done = _md_run(out, "-ng", "4", "-i", "AIS.in", "-p", "../built.pdb", "-s", "../built.xml",
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
                       "-s", "../built.xml", "-source-traj", "../source/cMD.dcd",
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
    argv = ("-i", "AIS.in", "-p", "../built.pdb", "-s", "../built.xml",
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


# --- the corrections: formats, cadences, resume, machine platform, output separation ----------

def test_a_conventional_stage_writes_a_genuine_dcd_and_refuses_a_netcdf_name(built):
    """The suffix is checked against the bytes, not against the name in the command."""
    import mdtraj

    from md_tools.openmm.trajectory import detect_trajectory_format

    config = built / "dcd.config"
    config.write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit", "dynamics": {"seed": 2},
        "stages": {"minimization_iterations": 10, "restrained_nvt_steps": 20,
                   "restrained_npt_steps": 20, "unrestrained_npt_steps": 20,
                   "production_steps": 100},
        "reporting": {"solute_printout": 20, "system_printout": 50,
                      "checkpoint_printout": 100}}, sort_keys=False), encoding="utf-8")
    assert _cli(built, "build-md", "-odir", "./dcd", "--config", str(config)).returncode == 0
    out = built / "dcd"

    done = _md_run(out, "-i", "cMD.in", "-p", "../built.pdb", "-s", "../built.xml",
                   "-x", "custom.dcd", "-r", "cMD.xml", "-o", "cMD.out", "-log", "cMD.log")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    # Genuine, by two independent checks: our own byte sniffer and a reader that knows nothing
    # about this project.
    assert detect_trajectory_format(out / "custom.dcd") == "dcd"
    trajectory = mdtraj.load(str(out / "custom.dcd"), top=str(built / "built.pdb"))
    assert trajectory.n_frames == 5, trajectory.n_frames

    refused = _md_run(out, "-i", "cMD.in", "-p", "../built.pdb", "-s", "../built.xml",
                      "-x", "cMD.nc", "-r", "other.xml", "-o", "other.out", "-log", "other.log")
    assert refused.returncode != 0
    assert "DCD" in refused.stderr, refused.stderr
    assert not (out / "cMD.nc").exists(), "the refused name was created anyway"


def test_output_and_log_are_two_files_with_two_kinds_of_content(built):
    """The separation, on a real run rather than at the parser."""
    out = built / "dcd"
    readable = (out / "cMD.out").read_text(encoding="utf-8")
    record = (out / "cMD.log").read_text(encoding="utf-8")

    # The .out is for a person: progress, energies, an ending.
    assert "Progress" in readable and "Potential Energy" in readable, readable[:400]
    assert "status               completed" in readable, readable[-400:]
    # The .log is for a machine, and `read_record` is what reads it.
    assert read_record(out / "cMD.log")["status"] == "completed"
    # Each names the other, so whichever one is opened first leads to the rest of the story.
    assert "cMD.log" in readable, readable[:400]
    assert "cMD.out" in record, "the record does not point at the readable output"


def test_ais_reads_a_genuine_dcd_source_and_writes_genuine_netcdf(hundred, built):
    """DCD in, NetCDF out, both verified from the bytes and by an independent reader."""
    import mdtraj

    from md_tools.openmm.trajectory import detect_trajectory_format

    assert detect_trajectory_format(built / "source" / "cMD.dcd") == "dcd"
    assert read_record(hundred / "AIS.log")["source"]["format"] == "dcd"

    for path_id in (0, 99):
        published = hundred / f"AIS_traj{path_id:04d}.nc"
        assert detect_trajectory_format(published) == "netcdf"
        trajectory = mdtraj.load(str(published), top=str(built / "built.pdb"))
        assert trajectory.n_frames == 3, (path_id, trajectory.n_frames)


def test_a_source_whose_suffix_and_contents_disagree_is_refused(built, source, tmp_path):
    """A DCD named `.nc` reads fine to somebody and is not what it claims."""
    mislabelled = built / "mislabelled.nc"
    mislabelled.write_bytes(source.read_bytes())
    out = _ais_project(built, "ais_mislabelled", 2)
    done = _md_run(out, "-i", "AIS.in", "-p", "../built.pdb", "-s", "../built.xml",
                   "-source-traj", "../mislabelled.nc", "-odir", ".",
                   "-o", "AIS.out", "-log", "AIS.log")
    assert done.returncode != 0
    assert ".nc" in done.stderr and "DCD" in done.stderr, done.stderr
    assert not list(out.glob("AIS_traj*.nc")), "a refused source still produced paths"


# --- four cadences and an exact mid-path resume -----------------------------------------------

def _cadence_project(built: Path, name: str) -> Path:
    """Four DIFFERENT cadences, so each one's effect is distinguishable from the others'."""
    config = built / f"{name}.config"
    config.write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit", "dynamics": {"seed": 5},
        "ais": {"number_of_paths": 2, "tau_start": 0.5, "tau_end": 0.0,
                "switching_steps": 1000, "parameter_update_interval_steps": 1,
                "observation_interval_steps": 100},
        "ais_source": {"trajectory": "../source/cMD.dcd"},
        "reporting": {"solute_printout": 200, "system_printout": 500,
                      "checkpoint_printout": 200}}, sort_keys=False), encoding="utf-8")
    assert _cli(built, "build-md", "-odir", f"./{name}", "--config", str(config)).returncode == 0
    return built / name


def test_each_ais_cadence_controls_its_own_stream(built, source):
    """Eleven work rows, six frames, three state rows: three cadences, three different counts.

    `system_printout` and `checkpoint_printout` were once validated and then never read. A setting
    that is accepted and inert is worse than one that is refused, because the person who wrote it
    believes it took effect.
    """
    import mdtraj

    out = _cadence_project(built, "cadences")
    done = _md_run(out, "-i", "AIS.in", "-p", "../built.pdb", "-s", "../built.xml",
                   "-source-traj", "../source/cMD.dcd", "-odir", ".",
                   "-o", "AIS.out", "-log", "AIS.log")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    observations = list(csv.DictReader((out / "path_0000" / "observations.csv").open()))
    states = list(csv.DictReader((out / "path_0000" / "system.csv").open()))
    frames = mdtraj.load(str(out / "AIS_traj0000.nc"), top=str(built / "built.pdb"))

    assert [int(r["protocol_step"]) for r in observations] == list(range(0, 1001, 100))
    assert [int(r["protocol_step"]) for r in states] == [0, 500, 1000]
    assert frames.n_frames == 6                                # steps 0, 200 … 1000

    # The state table carries physics, not just step numbers.
    row = states[1]
    for field in ("potential_energy_kj_mol", "kinetic_energy_kj_mol", "total_energy_kj_mol",
                  "temperature_kelvin"):
        assert row[field] not in ("", None), (field, row)
    assert 150.0 < float(row["temperature_kelvin"]) < 500.0, row["temperature_kelvin"]
    assert abs(float(row["total_energy_kj_mol"])
               - float(row["potential_energy_kj_mol"])
               - float(row["kinetic_energy_kj_mol"])) < 1e-6


def test_an_interrupted_path_resumes_exactly_and_duplicates_nothing(built, source):
    """Kill a run mid-path, resume it, and get the same outputs an uninterrupted run produces.

    The interruption is a real SIGKILL of the real command, not a simulated one: what is being
    tested is that a process which never got to clean up left a resumable state behind.
    """
    import mdtraj

    out = _cadence_project(built, "resume")
    argv = ("-i", "AIS.in", "-p", "../built.pdb", "-s", "../built.xml",
            "-source-traj", "../source/cMD.dcd", "-odir", ".", "-o", "AIS.out", "-log", "AIS.log")

    # Long enough to pass a checkpoint, short enough not to finish the second path.
    killed = subprocess.run(["timeout", "-s", "KILL", "8", "md-openmm", "md-run", *argv],
                            cwd=out, capture_output=True, text=True, timeout=120)
    assert killed.returncode != 0, "the run was meant to be interrupted"
    interrupted = [p for p in out.glob("path_*") if (p / "resume.json").is_file()]
    if not interrupted:
        pytest.skip("the run finished or died before its first checkpoint; timing-dependent")

    sidecar = json.loads((interrupted[0] / "resume.json").read_text())
    assert sidecar["protocol_step"] > 0 and sidecar["work_rows"] > 0
    # Nothing is published until a path is complete and validated.
    assert not (out / sidecar["trajectory"]).exists(), \
        "a half-written path was published under its final name"

    finished = _md_run(out, *argv, "--resume", "--overwrite")
    assert finished.returncode == 0, finished.stdout[-3000:] + finished.stderr[-3000:]
    assert "resuming at step" in finished.stdout, finished.stdout[-2000:]

    for path_id in (0, 1):
        directory = out / f"path_{path_id:04d}"
        observations = list(csv.DictReader((directory / "observations.csv").open()))
        states = list(csv.DictReader((directory / "system.csv").open()))
        frames = mdtraj.load(str(out / f"AIS_traj{path_id:04d}.nc"),
                             top=str(built / "built.pdb"))

        steps = [int(r["protocol_step"]) for r in observations]
        assert steps == list(range(0, 1001, 100)), (path_id, steps)
        assert [int(r["protocol_step"]) for r in states] == [0, 500, 1000]
        assert frames.n_frames == 6, (path_id, frames.n_frames)

        # The work integral is continuous across the boundary: cumulative is still the running
        # sum of the increments, which a duplicated or dropped row would break.
        running = 0.0
        for row in observations[1:]:
            running += float(row["incremental_work_kj_mol"])
            assert abs(running - float(row["cumulative_work_kj_mol"])) < 1e-6, (path_id, row)
        assert float(observations[0]["cumulative_work_kj_mol"]) == 0.0
        assert float(observations[-1]["tau"]) == 0.0

        # Identity survived: same source frame, same seeds, same filename.
        completion = json.loads((directory / "completed.json").read_text())
        assert completion["trajectory"] == f"AIS_traj{path_id:04d}.nc"
        assert completion["source_frame_index"] == int(observations[0]["source_frame_index"])
        # And the checkpoint is gone, so nothing invites a resume of a finished path.
        assert not (directory / "path.chk").exists()
        assert not (directory / "resume.json").exists()


def test_a_completed_path_is_not_touched_by_a_resume(built, source):
    out = built / "resume"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.glob("AIS_traj*.nc"))}
    assert before, "nothing to check"
    done = _md_run(out, "-i", "AIS.in", "-p", "../built.pdb", "-s", "../built.xml",
                   "-source-traj", "../source/cMD.dcd", "-odir", ".", "-o", "AIS.out",
                   "-log", "AIS.log", "--resume", "--overwrite")
    assert done.returncode == 0, done.stderr[-2000:]
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.glob("AIS_traj*.nc"))}
    assert after == before, "a completed path was rewritten"


# --- the machine platform ---------------------------------------------------------------------

def test_the_machine_configuration_decides_the_platform_and_cpu_overrides_it(built, tmp_path):
    """CUDA by default, CPU when the machine says so, and `--cpu` for one run.

    All three are recorded distinguishably. A CPU result that could be an unnoticed CUDA fallback
    is the failure the whole platform policy exists to prevent.
    """
    import os

    out = built / "dcd"
    # Each run gets its OWN checkpoint. An OpenMM checkpoint is binary and platform-specific, so
    # sharing one between a CUDA run and a CPU run is not a thing that can work -- and the
    # refusal for trying is asserted separately below.
    argv = ("-i", "cMD.in", "-p", "../built.pdb", "-s", "../built.xml")

    # 1. no user configuration at all -> built-in default, CUDA.
    environment = dict(os.environ, MD_TOOLS_CONFIG=str(tmp_path / "absent.config"))
    done = subprocess.run(["md-openmm", "md-run", *argv, "-r", "m1.xml", "-x", "m1.dcd",
                           "-chk", "m1.chk", "-o", "m1.out", "-log", "m1.log"],
                          cwd=out, capture_output=True, text=True, timeout=1800, env=environment)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    record = read_record(out / "m1.log")["acceleration"]
    assert record["resolved_platform"] == "CUDA"
    assert record["platform_origin"] == "built-in default"

    # 2. a machine configuration that chooses CPU -> CPU, recorded as the machine's choice.
    configuration = tmp_path / "cpu.config"
    configuration.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "user": {"person_id": "t", "name": "T", "orcid": None, "affiliation": None},
        "machine": {"md_data": str(tmp_path),
                    "openmm": {"platform": "CPU", "precision": "mixed",
                               "device_policy": "local_rank"}}}), encoding="utf-8")
    environment = dict(os.environ, MD_TOOLS_CONFIG=str(configuration))
    done = subprocess.run(["md-openmm", "md-run", *argv, "-r", "m2.xml", "-x", "m2.dcd",
                           "-chk", "m2.chk", "-o", "m2.out", "-log", "m2.log"],
                          cwd=out, capture_output=True, text=True, timeout=1800, env=environment)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    record = read_record(out / "m2.log")["acceleration"]
    assert record["resolved_platform"] == "CPU"
    assert record["platform_origin"] == "machine.openmm"

    # 3. --cpu against a CUDA machine -> CPU, recorded as a command-line override.
    done = subprocess.run(["md-openmm", "md-run", *argv, "-r", "m3.xml", "-x", "m3.dcd",
                           "-chk", "m3.chk", "-o", "m3.out", "-log", "m3.log", "--cpu"],
                          cwd=out, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    record = read_record(out / "m3.log")["acceleration"]
    assert record["resolved_platform"] == "CPU"
    assert record["platform_origin"] == "--cpu (command line)"


# --- MPI fails closed, for real ----------------------------------------------------------------

def test_a_multi_rank_launch_without_mpi4py_fails_before_any_output(built, tmp_path):
    """A real `mpirun -n 2` with mpi4py made unavailable. Nothing may be written.

    Not a mock: the command is launched by the real launcher and refuses on its own. Eight ranks
    with no coordination would each run a whole ladder over one set of output paths, and the
    result would look complete.
    """
    import os

    _require_mpi()
    out = built / "rest2"
    destination = tmp_path / "nothing"
    environment = dict(os.environ, MD_TOOLS_FORCE_NO_MPI4PY="1")
    done = subprocess.run(
        ["mpirun", "-n", "2", "md-openmm", "md-run", "-ng", "2", "-i", "REST2.in",
         "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
         "-o", str(destination / "x.out"), "-log", str(destination / "x.log")],
        cwd=out, capture_output=True, text=True, timeout=600, env=environment)
    assert done.returncode != 0
    assert "mpi4py" in done.stdout + done.stderr, done.stdout + done.stderr
    assert not destination.exists() or not list(destination.iterdir()), \
        sorted(p.name for p in destination.iterdir())


def test_a_serial_run_needs_no_mpi4py(built, tmp_path):
    """The other half of the rule: one rank coordinates with nobody, so it needs nothing."""
    import os

    out = built / "dcd"
    environment = dict(os.environ, MD_TOOLS_FORCE_NO_MPI4PY="1")
    done = subprocess.run(
        ["md-openmm", "md-run", "-i", "cMD.in", "-p", "../built.pdb", "-s", "../built.xml",
         "-x", "serial.dcd", "-r", "serial.xml", "-o", "serial.out", "-log", "serial.log"],
        cwd=out, capture_output=True, text=True, timeout=1800, env=environment)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert read_record(out / "serial.log")["status"] == "completed"
