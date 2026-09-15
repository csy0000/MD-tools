"""WORKED EXAMPLES: how to start a simulation with this repository.

Read this file top to bottom. Each example is a complete, self-contained run of one protocol,
written the way you would actually write it, and it EXECUTES -- so an example that stops being
true fails here instead of quietly misleading the next person.

    example 1   cMD              build a system, generate a workflow, run the whole chain
    example 2   REST2            the same, plus a replica-exchange ladder under a launcher
    example 3   interrupt+resume kill a run and finish it
    example 4   AIS              switching paths drawn from a cMD ensemble
    example 5   peptide-like     a cyclic peptide from SMILES, with mbondi3 corrections
    example 6   the shipped configs in `configs/` still resolve

THE SHAPE OF EVERY EXAMPLE

    Two commands to prepare, one script to run:

        md-openmm build-top   structure + chemistry  ->  built.xml, built.pdb, built.log
        md-openmm build-md    a protocol config      ->  a directory of .in files and run.sh
        ./run.sh                                     ->  every stage, in order

    `build-top` decides the PHYSICS: force field, charges, radii, constraints. It writes
    `built.xml`, which is the Hamiltonian. `build-md` decides the EXPERIMENT: stage lengths,
    intervals, the ladder. It never opens `built.xml` -- the timestep is resolved at run time from
    the masses actually serialised there, because a configuration that claims HMR is a request and
    the System is the fact.

    `resolved.config` beside the generated scripts is AUTHORITATIVE. The `.in` files resolve to
    it; editing an `.in` changes the run, but the resolved document is what is read.

WHAT YOU SUPPLY, AND WHAT IS SUPPLIED FOR YOU

    You supply: the structure, the chemistry, the schedule, and the hardware. In particular
    nothing binds MPI ranks to GPUs for you -- see example 2.

SIZES HERE ARE TINY ON PURPOSE

    Tens of steps, so the file runs in minutes. A real run changes the step counts and nothing
    else about the shape.

PLATFORM_POLICY_EXEMPTION: `--cpu` throughout. What these examples demonstrate is the COMMAND
FLOW -- which command produces which file, and what you type next. They are not GPU evidence and
are not offered as any; real CUDA and MPI evidence lives in `test_cv_mpi_cuda_lanes.py`,
`test_cmd_cuda_smoke.py`, `test_rrest2_cuda_smoke.py` and `test_md_run_mpi_gpu.py`. On real
hardware you drop `--cpu`: CUDA is the default and there is no automatic fallback.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

#: ONE worker for this module. Its module-scoped fixture builds a system and runs
#: dynamics; scattered across workers it is built once per worker that draws a test.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("examples")]


def _md_openmm(cwd, *args, timeout=1800):
    """`md-openmm ...`, as you would type it."""
    done = subprocess.run(CLI + [str(a) for a in args], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)
    assert done.returncode == 0, (
        f"md-openmm {' '.join(str(a) for a in args)} failed:\n"
        + done.stdout[-3000:] + done.stderr[-3000:])
    return done


def _run_sh(directory, *extra, timeout=3600, expect_success=True):
    """`./run.sh ../built.pdb ../built.xml`, the generated driver for a whole protocol."""
    done = subprocess.run(
        ["bash", "run.sh", "../build/built.pdb", "../build/built.xml", *extra],
        cwd=directory, capture_output=True, text=True, timeout=timeout)
    if expect_success:
        assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return done


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """EXAMPLE 0 -- build a system.

    Everything below starts here. `-i` is the input structure; `-os` the serialised System, which
    holds the physics; `-op` the topology; `-log` the machine-readable record of what was loaded.

        md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log \\
                            --config sys.config

    `sys.config` says what the solute IS and how to solvate it. Implicit GBn2 here, so there is no
    box, no water, no ions and no barostat anywhere downstream -- choosing it changes what the
    rest of the configuration is allowed to say.
    """
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    work = tmp_path_factory.mktemp("examples")
    (work / "sys.config").write_text(yaml.safe_dump({
        "solute": {"kind": "peptide"},          # a PDB of a peptide -> the protein force field
        "solvent": {"model": "GBn2"},           # implicit; use TIP3P or OPC for explicit water
        "constraints": {"type": "HBonds"},      # X-H lengths fixed, which is what permits 2 fs
        "hydrogen_mass_repartitioning": {"enabled": False},
    }), encoding="utf-8")
    _md_openmm(work, "build-top", "-i", ALA, "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", "sys.config")
    assert (work / "build" / "built.xml").is_file() and (work / "build" / "built.pdb").is_file()
    return work


@pytest.fixture(scope="module")
def system(built, tmp_path_factory):
    """A SYSTEM ROOT PER EXAMPLE, each sharing the one built system.

    `build/`, `min/` and `input/` belong to the SYSTEM and are shared by every run on it -- and
    the sharing is enforced, not assumed: `input/min.in` is refused if a second configuration
    resolves it differently, because the runs already beside it read that file.

    The examples below are deliberately DIFFERENT EXPERIMENTS, not repeats of one. Example 1
    minimises for 25 iterations and reports every 20 steps; example 3 minimises for 10; example 4
    runs at `tau = 0.5`. Generated into one root, the second one to arrive refused with

        input/min.in already exists and is not what this configuration resolves to.

    which is the layout being right about them. They are not comparable runs of one system, so
    each gets a system root of its own -- exactly the advice the refusal gives. `build/` is copied
    rather than rebuilt: the physics is identical and `build-top` is the slow part.
    """
    def make(name):
        root = tmp_path_factory.mktemp(f"example-{name}")
        shutil.copytree(built / "build", root / "build")
        return root

    return make


# --- example 1: cMD ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cmd_run(system):
    """EXAMPLE 1's output, produced by a FIXTURE rather than by another test.

    `test_example_1b` used to require that `test_example_1` had already run, and said so by
    skipping when the directory was absent. Under `-n 24` the two land on different workers with
    separate module-scoped fixtures, so the directory legitimately was not there and a real test
    quietly did not run -- CI's no-skip policy being satisfied by which worker drew which test.

    A fixture states the dependency instead of assuming it: any worker that needs this output
    builds it.
    """
    built = system("cmd")
    (built / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD",
        "solvent": "implicit",
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 20260908},
        # Step counts, not durations: the step count is the authoritative number and the log
        # prints the derived picoseconds beside it.
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 20,
                   "restrained_npt_steps": 20, "unrestrained_npt_steps": 20,
                   "production_steps": 100},
        "reporting": {"crd_printout_solute": 20, "info_printout": 50, "checkpoint_printout": 50},
    }), encoding="utf-8")
    _md_openmm(built, "build-md", "-odir", "./cMD-run1", "--config", "cMD.config")
    _run_sh(built / "cMD-run1", "--cpu")
    return built / "cMD-run1"


def test_example_1_plain_md_from_a_structure_to_a_trajectory(cmd_run):
    """The simplest complete run: minimise, equilibrate, produce.

        md-openmm build-md -odir ./cMD-run1 --config cMD.config
        cd cMD-run1 && ./run.sh ../build/built.pdb ../build/built.xml

    `build-md` writes one `.in` per stage, one thin `.py` entry point per stage, `run.sh` which
    calls them in order, and `resolved.config`. Each stage hands its final state to the next
    through `-c`, which `run.sh` wires up for you.

    WHAT IS WHERE, because it is no longer all one directory. The SYSTEM owns `build/`, `min/`
    and `input/`; the RUN owns its `eq/`, its output and its records. So the `.in` files are
    shared at `../input/`, the minimisation is the shared `../min/`, and what is in the run
    directory is what belongs to this run alone.
    """
    root = cmd_run.parent
    generated = {p.name for p in cmd_run.iterdir()}
    assert {"run.sh", "resolved.config", "run.config", "cMD.py"} <= generated, generated

    # The inputs are SHARED, so they are beside the system, not inside the run.
    shared = {p.name for p in (root / "input").iterdir()}
    assert {"min.in", "eq_1.in", "cMD.in"} <= shared, shared
    # The minimisation is shared too; the equilibration is this run's own.
    assert (root / "min" / "min.py").is_file()
    # The SCRIPTS are filed `eq_<k>.py`; only the restarts they write are stage-named. The two
    # spellings are deliberate and are not interchangeable.
    assert (cmd_run / "eq" / "eq_1.py").is_file()

    # What you get: a trajectory, a final state to continue from, a human-readable output and a
    # machine-readable provenance record, per stage.
    assert (cmd_run / "solute_prod1.nc").is_file()
    assert (cmd_run / "cMD.xml").is_file()
    assert "completed" in (cmd_run / "cMD.out").read_text(encoding="utf-8")


def test_example_1b_the_same_run_one_stage_at_a_time(cmd_run):
    """`run.sh` is a convenience, not a second interface. Any stage can be run directly.

    These three are the SAME run reaching the same installed code:

        ./run.sh
        md-openmm md-run -i min.in -p ../built.pdb -s ../built.xml -o min.out ...
        python min.py    -p ../built.pdb -s ../built.xml -o min.out ...
    """
    directory = cmd_run
    _md_openmm(directory, "md-run", "-i", "../input/min.in", "-p", "../build/built.pdb", "-s", "../build/built.xml",
               "-o", "min_again.out", "-x", "min_again.dcd", "-r", "min_again.xml",
               "-log", "min_again.log", "--cpu", "-odir", "./again")
    assert (directory / "again" / "min_again.out").is_file() or \
        (directory / "min_again.out").is_file()


# --- example 2: REST2 ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rest2_run(system):
    """EXAMPLE 2's output, as a fixture. Same reason as `cmd_run`: a test must not depend on
    another test having run, because under `-n 24` it may not have run HERE."""
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    built = system("rest2")
    _write_rest2_config(built)
    _md_openmm(built, "build-md", "-odir", "./REST2-run1", "--config", "REST2.config")
    _run_sh(built / "REST2-run1", "--cpu", timeout=7200)
    return built / "REST2-run1"


def _write_rest2_config(built):
    """The ladder configuration, written by whoever needs it first."""
    (built / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2",
        "solvent": "implicit",
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 20260908},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 20,
                   "restrained_npt_steps": 20, "unrestrained_npt_steps": 20,
                   "production_steps": 0},        # the ladder owns production, not a stage
        "reporting": {"crd_printout_solute": 50, "info_printout": 50, "checkpoint_printout": 50},
        "rest2": {
            "number_of_replicas": 3,              # -> mpirun -n 3
            "tau_max": 0.5,                       # ladder is linear 0 -> 0.5 over the states
            "exchange_interval_steps": 50,
            "number_of_exchanges": 4,             # production per state = 50 * 4 = 200 steps
            "equilibration_steps": 50,            # per state, at that state's own Hamiltonian
        },
    }), encoding="utf-8")


def test_example_2_a_rest2_ladder_under_a_launcher(rest2_run):
    """Replica exchange: N states of one system, differing only in Hamiltonian.

        mpirun -n 3 md-openmm md-run -ng 3 -i REST2.in -p ../built.pdb -s ../built.xml ...

    `run.sh` writes that line for you with `-n` equal to the state count. **Any other world size
    is refused**: a ladder run in fewer processes is a different schedule, not a smaller one.

    ON REAL HARDWARE YOU MUST PLACE THE RANKS YOURSELF:

        CUDA_VISIBLE_DEVICES=0,1,2 ./run.sh ../built.pdb ../built.xml

    Nothing binds ranks to devices automatically. Without it every rank builds its Context on the
    default device and the whole ladder runs on one GPU, silently and slowly.

    `equilibration_steps` is the one setting people miss. It relaxes each rung under ITS OWN
    scaled Hamiltonian before the first exchange, outside the production budget. Without it every
    rung starts from a tau = 0 configuration and the hot rungs spend their opening exchanges
    relaxing out of a distribution that is not theirs -- while every record calls those samples
    production.
    """
    run_sh = (rest2_run / "run.sh").read_text(encoding="utf-8")
    assert "mpirun -n 3" in run_sh and "-ng 3" in run_sh

    # THE LADDER'S OUTPUT IS PER SEGMENT, in `remd_records/`. `run.sh` names it
    # `-o remd_records/REST2_prod<N>.out`, because a ladder that is extended in place writes a
    # `_prod2` set beside the first rather than over it -- so there is no `REST2.out` at the run
    # root for this to read, and there was not one to read here.
    out = (rest2_run / "remd_records" / "REST2_prod1.out").read_text(encoding="utf-8")
    assert "run_status: completed" in out
    # One trajectory per fixed thermodynamic STATE -- never per walker, never tau-named.
    for state in range(3):
        assert (rest2_run / f"whole_state{state}_prod1.nc").is_file()
    # An Amber-style exchange history, and the per-pair acceptance report.
    assert (rest2_run / "rem.log").is_file()
    assert "acceptance" in out.lower()


# --- example 3: interrupt and resume ---------------------------------------------------------------


def _has_committed_checkpoint(directory):
    """A `.checkpoints` directory exists AND holds a committed generation.

    The directory is created before the first checkpoint lands in it, so its mere existence is
    not the event worth waiting for.
    """
    return directory.is_dir() and any(directory.iterdir())


def _interrupt_when(directory, condition, *, what, until_output=None, timeout=900):
    """Run `run.sh`, wait for something to be TRUE, then SIGINT it. Returns the output.

    The signal goes to the process GROUP. `run.sh` is a shell that execs `md-openmm` children,
    and signalling only the shell leaves the integrator running -- a mistake already paid for
    once outside the suite, where a campaign's stages kept going after their launcher had been
    told to stop. `start_new_session=True` gives the group a known leader to signal.
    """
    import os
    import signal
    import time

    process = subprocess.Popen(
        ["bash", "run.sh", "../build/built.pdb", "../build/built.xml", "--cpu"],
        cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    group = os.getpgid(process.pid)
    collected = []

    def _ready():
        if until_output is not None:
            return any(until_output in line for line in collected)
        return condition()

    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if until_output is not None:
                # Reading a line at a time so the condition can be about the output itself.
                line = process.stdout.readline()
                if line:
                    collected.append(line)
            else:
                time.sleep(0.25)
            if _ready():
                break
        else:
            raise AssertionError(f"waited {timeout}s for {what} and it did not happen")
        assert process.poll() is None, (
            f"the run finished on its own; it was meant to be interrupted while {what}")
        os.killpg(group, signal.SIGINT)
        try:
            remaining, _ = process.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            # SIGINT is a REQUEST: the stage commits its checkpoint and then exits, and on a
            # machine running 24 workers that can take a while. How long it takes is not what
            # this test is about, and the thing the test needs -- a committed checkpoint -- was
            # established by the wait above, BEFORE the signal was sent. So escalate and carry
            # on rather than failing.
            #
            # 300 seconds here used to be a hard failure, which made a passing test depend on how
            # busy the machine was. That is the same defect as the load-dependent skip this
            # helper replaced, moved from the premise to the teardown.
            os.killpg(group, signal.SIGKILL)
            remaining, _ = process.communicate(timeout=120)
        return "".join(collected) + (remaining or "")
    finally:
        if process.poll() is None:
            os.killpg(group, signal.SIGKILL)
            process.communicate()


def test_example_3_an_interrupted_cmd_chain_resumes_by_rerunning_the_same_command(system):
    """READ THIS BEFORE RUNNING A LONG cMD CHAIN ON A SCHEDULER.

    An interrupted cMD chain continues by RE-RUNNING THE SAME COMMAND. Each stage that already
    reports completion is skipped, and the stage that was interrupted picks up from its committed
    checkpoint -- which is what `run.sh`'s own header has always promised.

        ./run.sh ...                -> "min: already completed and verified (min.log); not
                                        rerunning." ... then the interrupted stage resumes.

    THIS TEST USED TO ASSERT THE OPPOSITE, and was written to fail the day the defect was fixed
    rather than document it for ever. What was wrong: `md-run` ran the output-collision check
    before `stage_main` could consult the completion record, so a completed stage was never
    skipped and the chain refused on its own first stage's outputs. The refusal then advised
    `--resume`, which the cMD layer rejects by name -- two messages from one command pointing at
    each other, with no way forward but `--overwrite`, which discards the finished stages.

    `--resume` is still not a cMD flag, and that is deliberate: a stage continues on the strength
    of a committed checkpoint, which is a fact about the directory, not a claim on the command
    line. **A ladder is different** -- it takes `--resume` properly (example 3b).
    """
    built = system("interrupt")
    directory = built / "interrupted"
    (built / "interrupt.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 20260908},
        "stages": {"minimization_iterations": 10, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 400_000},
        "reporting": {"crd_printout_solute": 500, "info_printout": 500,
                      "checkpoint_printout": 500},
    }), encoding="utf-8")
    _md_openmm(built, "build-md", "-odir", "./interrupted", "--config", "interrupt.config")

    # THE INTERRUPT LANDS ON A CONDITION, NOT ON A TIMER.
    #
    # This used to be `timeout -s INT 45`, and 45 seconds is a statement about the machine rather
    # than about the software: under `-n 24` the chain often had not reached production yet, so
    # the test skipped -- announcing "this machine is too loaded to demonstrate the resume". A
    # test that stops testing when the machine is busy stops testing exactly when a resume defect
    # would matter most, and CI's no-skip policy was then satisfied by luck.
    #
    # The premise is that the interrupt reaches PRODUCTION, the only stage here long enough to
    # hold a committed checkpoint short of its step count. So wait for that checkpoint and
    # interrupt then. Slow machines take longer; they no longer take a different code path.
    killed = _interrupt_when(
        directory, lambda: _has_committed_checkpoint(directory / "cMD.checkpoints"),
        what="the production stage to commit a checkpoint")
    assert list(directory.glob("*.checkpoints")), "nothing was committed"
    assert (directory / "cMD.checkpoints").is_dir(), (
        "the interrupt did not reach production, which this test now waits for rather than "
        "hoping for: " + killed[-2000:])

    # Re-running the same command is the route, and it works. The production stage is long on
    # purpose (400,000 steps), so this is interrupted again rather than run to the end: what is
    # being shown is that the chain PROCEEDS, not that it finishes.
    #
    # Interrupted on the line being asserted, for the same reason as above.
    output = _interrupt_when(
        directory, None, what="the chain to report a skipped stage and resume",
        until_output="already completed and verified")
    assert "already exist" not in output, (
        "the chain refused on outputs a completed stage of its own wrote:\n" + output[-3000:])
    assert "already completed and verified" in output, (
        "no stage was skipped, so nothing was carried over from the interrupted run:\n"
        + output[-3000:])

    # `--resume` remains refused BY NAME on a cMD chain, and the message must say so plainly
    # rather than being advertised by another part of the same command.
    resumed = _run_sh(directory, "--cpu", "--resume", expect_success=False)
    assert resumed.returncode != 0
    assert "--resume is not a cMD flag" in resumed.stdout + resumed.stderr


def test_example_3b_resuming_a_ladder_is_a_different_command(rest2_run):
    """A ladder takes `--resume`, but NOT through `run.sh`.

        # this does NOT work -- run.sh forwards its arguments to every stage, and the
        # cMD-style min/eq stages reject --resume before the ladder is reached:
        ./run.sh ../built.pdb ../built.xml --resume

        # this is how you resume a ladder: the launcher line, directly.
        mpirun -n 3 md-openmm md-run -ng 3 -i ../input/REST2.in \
            -p ../build/built.pdb -s ../build/built.xml -c eq/eq_3.xml \
            -x REST2.nc -r restart.json -o REST2.out -log REST2.log --resume

    It continues to the ORIGINAL budget and does not extend it. Run against a ladder that already
    finished it reports completion rather than repeating the work.
    """
    directory = rest2_run

    # The wrong way, demonstrated so nobody has to discover it.
    wrong = _run_sh(directory, "--cpu", "--resume", expect_success=False)
    assert wrong.returncode != 0
    assert "not a cMD flag" in wrong.stdout + wrong.stderr

    # The right way.
    before = (directory / "whole_state0_prod1.nc").stat().st_mtime_ns
    done = subprocess.run(
        # `../input/REST2.in`: the input is SHARED at the dataset root, not inside the run.
        ["mpirun", "-n", "3", *CLI, "md-run", "-ng", "3", "-i", "../input/REST2.in",
         "-p", "../build/built.pdb", "-s", "../build/built.xml", "-c", "eq/eq_3.xml",
         "-x", "REST2.nc", "-r", "restart.json", "-o", "REST2.out", "-log", "REST2.log",
         "--cpu", "--resume"],
        cwd=directory, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert (directory / "whole_state0_prod1.nc").stat().st_mtime_ns == before, \
        "a completed ladder was rerun instead of reporting completion"


# --- example 4: AIS ------------------------------------------------------------------------------

def test_example_4_switching_paths_from_a_fixed_tau_ensemble(system):
    """AIS: many short non-equilibrium paths, each starting from a frame of a FIXED-TAU cMD run.

    AIS is `protocol: AIS` in a `build-md` configuration, not a separate command. It needs a
    SOURCE trajectory -- the ensemble the paths anneal away from -- which you pass with
    `-source-traj`. `number_of_paths` is the GLOBAL total, not a count per rank.

    THE SOURCE MUST BE SAMPLED AT `ais.tau_start`, and this example used to break that rule. It
    took example 1's ordinary cMD, which runs at tau = 0, and left `ais.tau_start` at its default
    of 0.5 -- so every path claimed to begin in an ensemble that had never been sampled, and
    every work value measured a switch that did not start where it said. `tau_start`'s own
    documentation states the requirement ("It must equal the tau of the source ensemble"), and
    nothing could check it: the trajectory recorded no tau, and the log said so in as many words.

    It is checked now, from the `tau` attribute the source records for itself, so this example
    has to do what it always should have: run its own short cMD AT tau = 0.5 and anneal from
    that. The extra build is the point of the example, not overhead around it.
    """
    built = system("ais")
    hot_config = built / "hot.config"
    hot_config.write_text(yaml.safe_dump({
        "protocol": "cMD",
        "solvent": "implicit",
        # `tau: 0.5` -- the whole reason this run exists. It samples the scaled ensemble that
        # `ais.tau_start` below names.
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 20260908, "tau": 0.5},
        "stages": {"minimization_iterations": 10, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 100},
        # `crd_printout_whole`, because AIS builds a Context from the FULL System: the
        # solute-only stream is not a usable source, and this key defaults to 0.
        "reporting": {"crd_printout_solute": 10, "crd_printout_whole": 10,
                      "info_printout": 50, "checkpoint_printout": 100},
    }), encoding="utf-8")
    _md_openmm(built, "build-md", "-odir", "./hot-run1", "--config", "hot.config")
    _run_sh(built / "hot-run1", "--cpu")
    source = built / "hot-run1" / "whole_prod1.nc"
    assert source.is_file(), sorted(p.name for p in (built / "hot-run1").iterdir())

    (built / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS",
        "solvent": "implicit",
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 20260908},
        "stages": {"minimization_iterations": 0, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        # tau_start MATCHES the source above, and is checked against the tau that file records.
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "observation_interval_steps": 10, "parameter_update_interval_steps": 5,
                "tau_start": 0.5, "tau_end": 0.0},
        "ais_source": {"trajectory": str(source)},
    }), encoding="utf-8")
    _md_openmm(built, "build-md", "-odir", "./AIS-run1", "--config", "AIS.config")

    # `../input/AIS.in`: the input is SHARED at the dataset root, not inside the run.
    _md_openmm(built / "AIS-run1", "md-run", "-i", "../input/AIS.in",
               "-p", "../build/built.pdb", "-s", "../build/built.xml",
               "-source-traj", str(source), "-odir", "./out", "-log", "AIS.log", "--cpu",
               timeout=3600)

    out = built / "AIS-run1" / "out"
    # One directory per path, each with the work rows and the record that says it finished.
    assert (out / "path_0000" / "completed.json").is_file()
    # The work table rank 0 assembles from the per-path records on disk.
    assert (out / "AIS_work.csv").is_file()
    assert (out / "selected_source_frames.csv").is_file()


# --- example 5: a cyclic peptide from SMILES -------------------------------------------------------

def test_example_5_a_cyclic_peptide_built_from_smiles(tmp_path):
    """When the solute is a molecule rather than a PDB peptide: `-i` a `.smi`, and choose a kind.

        solute:
          kind: peptide-like          # peptide | peptide-like | ligand
          ligand_forcefield: sage-2.2.1
          ligand_charge_method: am1bcc

    `ligand` parameterises the whole molecule with the small-molecule force field and claims no
    peptide chemistry. `peptide-like` is the SAME route -- same force field, same charges -- plus
    a validated residue map, which is what lets residue-keyed corrections reach a solute that has
    no residue names. The concrete case is mbondi3: its Arg/Asp/Glu radius adjustments are
    selected by residue and atom NAME, so against a single made-up residue they match nothing and
    the radii silently reduce to mbondi2 while the build still says mbondi3.

    THIS IS THE SLOW EXAMPLE: AM1-BCC runs a real semi-empirical calculation on the CPU and is
    the longest part of a small-molecule build, minutes rather than seconds.
    """
    # cyclo(Gly-L-Asp-L-Arg): head-to-tail, and it carries both correctable groups.
    smiles = "O=C1NCC(=O)N[C@H](CC(=O)[O-])C(=O)N[C@H]1CCCNC(N)=[NH2+]"
    (tmp_path / "cyc.smi").write_text(f"{smiles} CYC\n", encoding="utf-8")
    (tmp_path / "sys.config").write_text(yaml.safe_dump({
        "solute": {"kind": "peptide-like",
                   "ligand_forcefield": "sage-2.2.1",
                   "ligand_charge_method": "am1bcc"},
        "solvent": {"model": "GBn2"},
        "constraints": {"type": "HBonds"},
        "hydrogen_mass_repartitioning": {"enabled": False},
    }), encoding="utf-8")
    _md_openmm(tmp_path, "build-top", "-i", "cyc.smi", "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", "sys.config", timeout=3600)

    # The build states what it actually assigned, including which atoms were corrected.
    log = (tmp_path / "build" / "built.log").read_text(encoding="utf-8")
    assert "peptide_like_mbondi3" in log
    assert "molecular_map_digest" in log
    # `built.sdf` is retained beside the System: bond orders are not recoverable from a topology,
    # and the omega classifier needs them for a ladder over this solute.
    assert (tmp_path / "build" / "built.sdf").is_file()

    # The map is reusable, and it is what you would define collective variables from.
    from md_tools.openmm.peptide_map import map_from_sdf

    mapped = map_from_sdf(tmp_path / "build" / "built.sdf")
    assert sorted(mapped.sequence) == ["ARG", "ASP", "GLY"]
    assert len(mapped.torsions()) == 9          # phi, psi, omega for each of three residues
    assert len(mapped.carboxylate_oxygens) == 2
    assert len(mapped.guanidinium_hydrogens) == 5


# --- example 6: the configurations shipped with the repository -------------------------------------

@pytest.mark.parametrize("name", ["cMD.config", "REST2.config", "rREST2.config", "AIS.config"])
def test_example_6_the_shipped_protocol_configs_are_usable_starting_points(name, tmp_path):
    """`configs/md/*.config` are documented starting points, and they still resolve.

    Copy one, change the step counts, and run it. This asserts they parse under the current
    schema -- a shipped example that no longer loads is worse than none.
    """
    from md_tools.build.md import resolve_md_config

    source = REPO / "configs" / "md" / name
    if not source.is_file():
        pytest.skip(f"{source} is not shipped")
    resolved = resolve_md_config(source)
    assert resolved["protocol"] in ("cMD", "REST2", "rREST2", "AIS")


def test_example_6b_the_shipped_build_top_config_is_usable(tmp_path):
    """`configs/sys/build-top.config` likewise."""
    from md_tools.build.top import resolve_build_config

    source = REPO / "configs" / "sys" / "build-top.config"
    if not source.is_file():
        pytest.skip(f"{source} is not shipped")
    resolved = resolve_build_config(source)
    assert resolved["solute"]["kind"] in ("peptide", "peptide-like", "ligand")


# --- the runnable script beside each method page ------------------------------------------------

def test_the_cmd_method_page_script_actually_runs(tmp_path):
    """`docs/openmm_methods/cMD/README.sh` is instructions AND a script; run it.

    A worked example nobody executes drifts from the code it documents, and the drift is invisible
    until someone follows it and it fails. `--quick` shrinks the documented example to an implicit
    solvent chain of a few thousand steps so this finishes in about a minute; everything else --
    the commands, the generated files, the order -- is what the page documents.
    """
    # `openmm_methods/cMD/` is a DOCS METHOD PAGE, not a run directory: it takes no `-run1`
    # suffix. A bulk rewrite gave it one, and since the test skips when the file is absent, both
    # of these stopped running instead of failing -- a silent pass is worse than a red test.
    script = REPO / "docs" / "openmm_methods" / "cMD" / "README.sh"
    if not script.is_file():
        pytest.skip(f"{script} is not present")
    assert script.stat().st_mode & 0o111, "README.sh is not executable"

    done = subprocess.run(
        ["bash", str(script), "--quick", "--cpu", "-o", str(tmp_path / "run")],
        cwd=script.parent, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    # `cMD-run1/`, not the retired `md_script/`: the run directory is `<method>-run<N>` and the
    # system's `build/`, `min/` and `input/` are its siblings.
    produced = tmp_path / "run" / "cMD-run1"
    assert (produced / "solute_prod1.nc").is_file()
    assert "completed" in (produced / "cMD.out").read_text(encoding="utf-8")


def test_the_method_page_script_refuses_to_delete_an_existing_directory(tmp_path):
    """It must never clear a path it was handed -- it stops and says so instead."""
    # `openmm_methods/cMD/` is a DOCS METHOD PAGE, not a run directory: it takes no `-run1`
    # suffix. A bulk rewrite gave it one, and since the test skips when the file is absent, both
    # of these stopped running instead of failing -- a silent pass is worse than a red test.
    script = REPO / "docs" / "openmm_methods" / "cMD" / "README.sh"
    if not script.is_file():
        pytest.skip(f"{script} is not present")

    occupied = tmp_path / "already-here"
    occupied.mkdir()
    keep = occupied / "precious.txt"
    keep.write_text("do not delete me", encoding="utf-8")

    done = subprocess.run(
        ["bash", str(script), "--quick", "--cpu", "-o", str(occupied)],
        cwd=script.parent, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "already exists" in done.stdout + done.stderr
    assert keep.read_text(encoding="utf-8") == "do not delete me", "the script deleted user data"
