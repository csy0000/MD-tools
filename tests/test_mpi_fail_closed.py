"""A plural launch either coordinates or stops, proved with a real launcher.

Two failure modes, and they are different:

* **no coordination** -- `mpi4py` is unavailable under `mpirun -n N`. Every rank must refuse, and
  none may write anything. The old behaviour was N uncoordinated simulations over one set of
  output paths, each believing it was the whole thing, producing a directory that looks complete.
* **one rank dies** -- a fatal error on a single rank. The whole communicator must stop. A rank
  that raises alone leaves the others blocked in the next collective, and the job hangs until a
  scheduler kills it, which is worse than failing: it burns the allocation and reports nothing.

Every test here has a TIMEOUT, and the timeout is the assertion. A hang is the failure being
tested for, so a test that waits forever cannot detect it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from .conftest import make_states_for

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = pytest.mark.slow

#: Long enough for a launcher to start N interpreters and for them to refuse; short enough that a
#: hang is a failure rather than a wait. A refusal takes well under a second.
LAUNCH_TIMEOUT = 120


def _require_mpirun():
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")


def _inputs(protocol):
    """What names the System for this protocol's launch.

    A REST2 ladder reads -s ONLY from its group file (0.5.4): each line names one saved scaled
    state, `build/REST2/system_state<i>.xml`, and continues from `-c eq/eq_3.xml`. `-s` beside it
    is refused by name, so passing it here would test that refusal instead of the MPI behaviour.
    AIS takes both end states on the command line: V0 as -s and V1 as -s2/-p2 (the saved tau-0.5
    state, a parameter-only edit of the same particles), or it refuses by name before MPI is
    reached -- which would test that refusal instead.
    """
    if protocol == "REST2":
        return ["--groupfile", "remd_groupfile.1"]
    return ["-s", "../build/built.xml", *AIS_V1]


#: V1 for every AIS launch here, built into `build/AIS/` by the `projects` fixture.
AIS_V1 = ["-p2", "../build/built.pdb", "-s2", "../build/AIS/system_state0.xml"]


def _launch_directory(projects, protocol, tmp_path):
    """Where a launch runs and what it names as `-odir`, and how to tell it wrote nothing.

    A REST2 ladder's group file names `-i _protocol.py` relative to ITSELF, and the runtime writes
    that helper into `-odir`, so the two must be one directory: any other `-odir` is refused before
    anything is written. So a ladder launches as `run.sh` does -- `-odir .` in its run directory --
    here in a fresh COPY of `REST2-run1` per case, a sibling of it so `../build/` still resolves.
    "Wrote nothing" is then the copy's file listing unchanged, rather than `-odir` not existing.
    AIS takes a separate `-odir` as before.
    """
    if protocol != "REST2":
        return projects / f"{protocol}-run1", tmp_path / "never"
    copy = projects / f"{tmp_path.name}-REST2"
    shutil.copytree(projects / "REST2-run1", copy)
    return copy, copy


def _listing(directory):
    return sorted(str(p.relative_to(directory)) for p in Path(directory).rglob("*"))


def _wrote_nothing(protocol, project, destination, before):
    if protocol == "REST2":
        assert _listing(project) == before, sorted(set(_listing(project)) - set(before))
    else:
        assert not destination.exists(), sorted(p.name for p in destination.iterdir())


def _odir(project, destination):
    return "." if destination == project else str(destination)


def _environment(**extra):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(extra)
    return base


@pytest.fixture(scope="module")
def projects(tmp_path_factory):
    """Generated REST2 and AIS directories, so the wrappers can be launched directly."""
    root = tmp_path_factory.mktemp("failclosed")

    # THE BUILT SYSTEM COMES FIRST NOW, and the order is load-bearing rather than tidy. A
    # ladder's rungs are scaled and serialised from `build/built.xml` at BUILD time, so
    # `build-md` for REST2 refuses outright if the System is not there yet. This fixture used to
    # generate first and build afterwards, which was harmless while scaling happened at run time.
    #
    # A REAL built pair and a REAL source trajectory.
    #
    # These were a one-line PDB, a `<System/>` stub and 200 zero bytes behind a DCD magic number,
    # on the reasoning that preflight only had to get as far as the MPI check. That stopped being
    # true when the preflight began deserialising the pair and reading the source: rank 0 now
    # fails on the stub before the rank-local case under test can be reached, and a test in which
    # EVERY rank fails cannot demonstrate what happens when ONE does.
    ala = REPO / "tests" / "data" / "ALA.pdb"
    if not ala.is_file():
        pytest.skip("no ALA fixture")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ala), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    # Enough eligible frames for the default 100 paths. With eight, the AIS preflight refused on
    # the frame count and every test using this fixture stopped BEFORE the runtime behaviour it
    # was written for -- passing on the return code while never reaching the phase under test.
    mdtraj.join([frames] * 128).save_dcd(str(root / "source.dcd"))

    # V1 for the AIS launches: the saved scaled state at tau = 0.5 of the same built System.
    from .conftest import make_scaled_state

    make_scaled_state(root, tau=0.5, method="AIS")

    # ...and only now the runs, each into its own `<method>-run<N>`.
    for protocol in ("REST2", "AIS"):
        configuration = root / f"{protocol}.config"
        # `implicit`, matching the GBn2 System built above: `build-md` validates the chain it
        # generates against that System, and an explicit chain's NPT stages are refused on a
        # System with no box.
        document = {"protocol": protocol, "solvent": "implicit"}
        if protocol == "REST2":
            document["rest2"] = {"number_of_replicas": 2}
        else:
            document["ais_source"] = {"trajectory": "../source.dcd"}
        configuration.write_text(yaml.safe_dump(document), encoding="utf-8")
        # A REST2 ladder integrates SAVED scaled states, and `build-md` refuses to generate one
        # until `build-top --rest2-scaler` has written them.
        make_states_for(root, configuration)
        done = subprocess.run(CLI + ["build-md", "-odir", str(root / f"{protocol}-run1"),
                                     "--config", str(configuration)],
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stdout + done.stderr

    # THE STATE EVERY GROUP LINE CONTINUES FROM, `REST2-run1/eq/eq_3.xml`: a real serialized State
    # with positions, velocities and the box. A ladder reads its coordinates only from its group
    # file now, so the refusal tests need it to exist -- or every rank would stop on the missing
    # file before reaching the rank-local case under test -- and the propagation test needs it to
    # propagate at all. It used to be passed as `-c initial_state.xml`.
    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    state = context.getState(getPositions=True, getVelocities=True,
                             enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    starting = root / "REST2-run1" / "eq" / "eq_3.xml"
    starting.parent.mkdir(parents=True, exist_ok=True)
    starting.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    return root


# --- no coordination available ------------------------------------------------------------------

@pytest.mark.parametrize("entry", ["md-run", "wrapper"])
@pytest.mark.parametrize("protocol", ["REST2", "AIS"])
def test_a_real_two_rank_launch_without_mpi4py_refuses_and_writes_nothing(entry, protocol,
                                                                         projects, tmp_path):
    """`mpirun -n 2`, mpi4py made unavailable, through both entry points.

    The wrapper case is the one that matters most: it calls the same runtime `md-run` calls, so a
    guard that lives only in `md-run` leaves it open. That is exactly what the baseline did.
    """
    _require_mpirun()
    project, destination = _launch_directory(projects, protocol, tmp_path)
    before = _listing(project)

    if entry == "md-run":
        argv = ["md-openmm", "md-run", "-i", f"../input/{protocol}.in",
                "-p", "../build/built.pdb", *_inputs(protocol),
                "-odir", _odir(project, destination)]
    else:
        argv = [sys.executable, str(project / f"{protocol}.py"),
                "-p", "../build/built.pdb", *_inputs(protocol),
                "-odir", _odir(project, destination)]
    if protocol == "AIS":
        argv += ["-source-traj", "../source.dcd"]

    done = subprocess.run(["mpirun", "-n", "2", *argv], cwd=project, capture_output=True,
                          text=True, timeout=LAUNCH_TIMEOUT,
                          env=_environment(MD_TOOLS_FORCE_NO_MPI4PY="1"))
    assert done.returncode != 0, done.stdout[-2000:]
    assert "mpi4py" in done.stdout + done.stderr, (done.stdout + done.stderr)[-2000:]
    _wrote_nothing(protocol, project, destination, before)


def test_a_serial_run_still_needs_no_mpi4py(projects, tmp_path):
    """The other half of the rule. World size 1 coordinates with nobody, so it imports nothing."""
    from md_tools.remd.mpi import Coordination

    environment = dict(os.environ)
    environment.pop("OMPI_COMM_WORLD_SIZE", None)
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment, MD_TOOLS_FORCE_NO_MPI4PY="1")
    try:
        coordination = Coordination.open()
        assert coordination.size == 1 and coordination.MPI is None
        coordination.barrier()                          # a genuine no-op, not a suppressed error
    finally:
        os.environ.clear()
        os.environ.update(previous)


# --- launcher and communicator disagree ---------------------------------------------------------

def test_a_launcher_size_that_the_communicator_contradicts_is_refused(projects, tmp_path):
    """`mpirun -n 2` with the environment claiming 4. The two must agree about the world."""
    _require_mpirun()
    project, destination = _launch_directory(projects, "REST2", tmp_path)
    before = _listing(project)
    done = subprocess.run(
        ["mpirun", "-n", "2", "md-openmm", "md-run", "-i", "../input/REST2.in",
         "-p", "../build/built.pdb", *_inputs("REST2"), "-ng", "4", "-odir", "."],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment())
    assert done.returncode != 0
    message = done.stdout + done.stderr
    assert "-ng on the command line" in message and "MPI communicator size" in message, message
    _wrote_nothing("REST2", project, destination, before)


# --- one rank dies ------------------------------------------------------------------------------

def test_a_rank_local_failure_stops_the_whole_communicator_without_a_hang(tmp_path):
    """One rank raises; the others must not sit in a collective until a scheduler kills them.

    The timeout IS the assertion. If `Coordination.fail` did not abort the communicator, the
    surviving rank would block in `barrier()` forever and this would time out rather than fail.
    """
    _require_mpirun()
    script = tmp_path / "one_rank_dies.py"
    script.write_text(textwrap.dedent('''
        import sys
        from md_tools.remd.mpi import Coordination

        coordination = Coordination.open()
        if coordination.rank == 1:
            # A rank-local fatal error: a device that will not bind, a file only this rank reads.
            coordination.fail("rank 1 cannot start", code=3)

        # Every other rank walks into a collective. Without a real Abort they wait here forever.
        coordination.barrier()
        print(f"rank {coordination.rank} SURVIVED THE FAILURE OF ANOTHER RANK")
    ''').strip() + "\n", encoding="utf-8")

    done = subprocess.run(["mpirun", "-n", "2", sys.executable, str(script)],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=LAUNCH_TIMEOUT, env=_environment())
    assert done.returncode != 0, done.stdout
    assert "SURVIVED" not in done.stdout, done.stdout
    assert "rank 1 cannot start" in done.stdout + done.stderr, done.stdout + done.stderr


def test_the_failing_rank_names_itself(tmp_path):
    """Which rank failed is the first thing anyone needs from a multi-rank failure."""
    _require_mpirun()
    script = tmp_path / "named.py"
    script.write_text(textwrap.dedent('''
        from md_tools.remd.mpi import Coordination

        coordination = Coordination.open()
        if coordination.rank == coordination.size - 1:
            coordination.fail("the last rank refuses")
        coordination.barrier()
    ''').strip() + "\n", encoding="utf-8")

    done = subprocess.run(["mpirun", "-n", "3", sys.executable, str(script)],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=LAUNCH_TIMEOUT, env=_environment())
    assert done.returncode != 0
    assert "[rank 2/3]" in done.stdout + done.stderr, done.stdout + done.stderr


# --- one rank fails during preflight -------------------------------------------------------------
#
# The case a real multi-GPU launch hits: rank 3's device is held by another job, or its CUDA
# context will not initialise. Everything before that point is a property of the command line and
# of files every rank sees identically, so rank 3 is the only one that fails -- and if it simply
# raises, it exits while ranks 0..2 walk on to the next collective and wait there for a
# participant that has already gone. The launcher reports nothing and the job holds its GPUs until
# a wall clock kills it.
#
# `MD_TOOLS_FAIL_PREFLIGHT_ON_RANKS` is the seam that makes this provable without arranging for a
# real GPU to be unavailable on exactly one rank of a live launch. It fails at precisely the point
# a rank-local platform failure fails.

FAIL_RANKS = "MD_TOOLS_FAIL_PREFLIGHT_ON_RANKS"


@pytest.mark.parametrize("entry", ["md-run", "wrapper"])
@pytest.mark.parametrize("protocol", ["REST2", "AIS"])
def test_one_rank_failing_preflight_stops_the_whole_launch_promptly(entry, protocol, projects,
                                                                    tmp_path):
    """Rank 1 of 2 fails. The launcher must return non-zero, and neither rank may survive.

    The TIMEOUT is the assertion. A hang is the defect being tested for, so a test that waits
    forever cannot detect it.
    """
    _require_mpirun()
    project, destination = _launch_directory(projects, protocol, tmp_path)
    before = _listing(project)

    if entry == "md-run":
        argv = ["md-openmm", "md-run", "-i", f"../input/{protocol}.in",
                "-p", "../build/built.pdb", *_inputs(protocol),
                "-odir", _odir(project, destination)]
    else:
        argv = [sys.executable, str(project / f"{protocol}.py"),
                "-p", "../build/built.pdb", *_inputs(protocol),
                "-odir", _odir(project, destination)]
    if protocol == "AIS":
        argv += ["-source-traj", "../source.dcd"]

    done = subprocess.run(["mpirun", "-n", "2", *argv], cwd=project, capture_output=True,
                          text=True, timeout=LAUNCH_TIMEOUT, env=_environment(**{FAIL_RANKS: "1"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    # Named as a rank-local failure, with the rank in it -- so a person reading either rank's
    # output learns which one failed rather than that "the launch failed".
    assert "rank 1" in message, message[-3000:]
    _wrote_nothing(protocol, project, destination, before)


def test_every_rank_failing_reports_the_reason_once_rather_than_n_times(projects, tmp_path):
    """A condition every rank hits is not rank-local, and repeating it N times only hides it."""
    _require_mpirun()
    project, destination = _launch_directory(projects, "REST2", tmp_path)
    before = _listing(project)
    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../build/built.pdb", *_inputs("REST2"), "-odir", "."],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{FAIL_RANKS: "0,1"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0
    assert "of 2 rank(s), so the whole launch is refused" not in message, (
        "a condition every rank hit was reported as a partial, rank-local failure:\n" + message)
    _wrote_nothing("REST2", project, destination, before)


def test_the_collective_agreement_is_the_one_in_the_mpi_authority():
    """`collectively` must go through `Coordination`, not import mpi4py for itself.

    A second `from mpi4py import MPI` anywhere is a second policy, and it will be the one that
    runs. The rule is stated in CLAUDE.md; this is it as a test.
    """
    import inspect

    from md_tools.run import preflight

    source = inspect.getsource(preflight)
    # The IMPORT, not the word: this module names mpi4py in prose, explaining why it does not
    # import it, and a test that forbade the word would forbid the explanation.
    for forbidden in ("from mpi4py", "import mpi4py"):
        assert forbidden not in source, f"the preflight has its own `{forbidden}`"
    assert "coordination.allgather" in source, (
        "the agreement does not go through the Coordination object")


# --- rank-zero preparation failures are collective ----------------------------------------------

def test_a_rank_zero_helper_failure_stops_every_rank_rather_than_hanging_them(projects,
                                                                              tmp_path):
    """`if rank == 0: prepare` then a barrier is a deadlock waiting for a bad input.

    Rank 0 raises inside the writer, exits, and every other rank waits at the barrier for a
    participant that has already gone. The launcher reports nothing and the job holds its GPUs
    until a wall clock kills it. The TIMEOUT is the assertion here.

    Provoked with a stale, incompatible `_protocol.py` that rank 0 refuses to overwrite -- the
    real reason a preparation fails, and one only rank 0 encounters.
    """
    _require_mpirun()
    project, _destination = _launch_directory(projects, "REST2", tmp_path)
    # A helper from another ladder: content-addressed, so rank 0 refuses it rather than replacing
    # it, and refuses it in the one place only rank 0 reaches. In the run directory, which is both
    # `-odir` and where the group file's `-i _protocol.py` resolves.
    (project / "_protocol.py").write_text("n_states = 99\n", encoding="utf-8")

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../build/built.pdb", *_inputs("REST2"), "-odir", "."],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment())
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "rank 0 could not prepare" in message or "generated from different content" in message, (
        message[-2500:])


# --- rank-local failures AFTER the preflight -----------------------------------------------------
#
# Preflight agreement is not enough. Everything after it is rank-local again -- creating a
# directory, opening a report, publishing a helper, running the ladder -- and each is a place ONE
# rank can fail while the others walk into the next collective and wait there for a participant
# that has already gone. The launcher then reports nothing and the job holds its GPUs until a wall
# clock kills it.
#
# The TIMEOUT is the assertion in every test here.

FAIL_PHASE = "MD_TOOLS_FAIL_PHASE"


@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize("phase", ["creating the output directory", "opening the rank report"])
def test_a_rank_local_failure_after_preflight_stops_the_whole_ladder(failing_rank, phase,
                                                                     projects, tmp_path):
    """Rank 0 AND a nonzero rank, at two phases. Neither may leave the other waiting."""
    _require_mpirun()
    project, destination = _launch_directory(projects, "REST2", tmp_path)

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../build/built.pdb", *_inputs("REST2"), "-odir", "."],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{FAIL_PHASE: f"REST2: {phase}:{failing_rank}"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert f"rank {failing_rank}" in message, message[-2500:]
    # No authoritative completion output survives a failed launch.
    assert not sorted(destination.glob("whole_state*_prod1.nc")), sorted(destination.glob("whole_state*_prod1.nc"))
    assert not (destination / "restart.json").exists()


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_a_rank_local_failure_after_preflight_stops_the_whole_ais_run(failing_rank, projects,
                                                                      tmp_path):
    _require_mpirun()
    project = projects / "AIS-run1"
    destination = tmp_path / f"ais-phase-{failing_rank}"

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "AIS.py"),
         "-p", "../build/built.pdb", "-s", "../build/built.xml", *AIS_V1,
         "-source-traj", "../source.dcd", "-odir", str(destination)],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{FAIL_PHASE: f"AIS: opening the rank reports:{failing_rank}"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert f"rank {failing_rank}" in message, message[-2500:]
    assert not sorted(destination.glob("AIS_traj*.nc"))
    assert not (destination / "AIS_hs.csv").exists()


# --- a rank-local failure INSIDE the dynamics loop itself -----------------------------------------
#
# Every phase above is guarded by `coordination.phase(...)` in the OUTER script (`generated.py`).
# The dynamics loop is not: it runs inside `ReplicaRun.run()`, called from `run_grouped`, called
# from `executor.main()` -- which used to catch every exception itself and turn it into a local
# integer status code, with no `Abort()` on the path a standalone `python -m md_tools.remd.executor`
# launch takes. A rank that fails there returned home from `main()` quietly while every other rank
# sat in whatever collective it reached next (a `barrier()`, an `allgather()`) waiting for a
# participant that would never call it again -- rescued only if and until something eventually
# noticed the bad status code and called `Abort()` itself. `ReplicaRun.run()` now calls
# `coordinator.fail()` itself, at the point the exception is caught, so nothing else has to.

PROPAGATION_FAIL_RANKS = "MD_TOOLS_FAIL_PROPAGATION_ON_RANKS"


@pytest.fixture(scope="module")
def initial_state(projects):
    """The real serialized State (positions, velocities, box) the ladder starts from.

    Propagation is never reached without one. The `projects` fixture writes it where every line of
    the generated group file names it, `REST2-run1/eq/eq_3.xml`, since a ladder reads `-c` from
    nowhere else (0.5.4).
    """
    path = projects / "REST2-run1" / "eq" / "eq_3.xml"
    assert path.is_file(), path
    return path


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_a_rank_local_failure_during_propagation_stops_the_whole_ladder_without_a_hang(
        failing_rank, projects, initial_state, tmp_path):
    """A real REST2 ladder, two ranks, one fails mid-loop. Neither rank may survive or hang."""
    _require_mpirun()
    project, destination = _launch_directory(projects, "REST2", tmp_path)
    assert (project / "eq" / "eq_3.xml").read_bytes() == initial_state.read_bytes()

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../build/built.pdb", *_inputs("REST2"), "-odir", "."],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{PROPAGATION_FAIL_RANKS: str(failing_rank)}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    # `Coordination.fail` prints from the failing rank and then calls `comm.Abort()`; Open MPI's
    # own report of WHICH rank called `Abort()` is what appears on the launcher's stdout/stderr --
    # `coordinator.fail`'s own message went to the rank's redirected `.out`/`.out.rankNN`, which
    # `executor.main` captures as the run's report, not to the process's real stderr.
    assert f"rank {failing_rank}" in message, message[-3000:]
    reports = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                        for path in sorted(destination.glob("REST2.out*")) if path.is_file())
    assert "MD_TOOLS_FAIL_PROPAGATION_ON_RANKS" in reports, reports[-3000:]
    # No authoritative completion manifest and no "completed" marker survive a failed launch --
    # the per-state trajectories exist (`_begin` creates them before propagation starts on every
    # run, healthy or not), but nothing claims the run they belong to finished.
    assert not (destination / "restart.json").exists()
    assert "run_status: completed" not in reports, reports[-3000:]


def test_the_phase_guard_is_the_only_mpi_authority():
    """`md_tools.remd.mpi` stays the one module that imports or controls mpi4py."""
    import inspect

    from md_tools.ais import run as ais_module
    from md_tools.remd import generated as ladder_module

    for module in (ais_module, ladder_module):
        source = inspect.getsource(module)
        for forbidden in ("from mpi4py", "import mpi4py"):
            assert forbidden not in source, f"{module.__name__} has its own `{forbidden}`"
        assert "coordination.phase(" in source, (
            f"{module.__name__} does not guard its post-preflight phases")


# --- CPU placement ------------------------------------------------------------------------------
#
# `--bind-to none` hands the launch the CPUs `taskset` gave it. Without it Open MPI binds each rank
# to one core by default, and those cores are then what the launch may use -- which is the rule
# working, but not the arithmetic under test here.

@pytest.mark.parametrize("protocol", ["REST2", "AIS"])
def test_a_cpu_count_that_does_not_divide_among_the_workers_is_refused_before_output(protocol,
                                                                                     projects,
                                                                                     tmp_path):
    """Three CPUs for two workers. Refused on every rank, naming the arithmetic, writing nothing."""
    _require_mpirun()
    if shutil.which("taskset") is None:
        pytest.skip("no taskset on PATH")
    project, destination = _launch_directory(projects, protocol, tmp_path)
    before = _listing(project)
    argv = ["md-openmm", "md-run", "-i", f"../input/{protocol}.in", "-p", "../build/built.pdb",
            *_inputs(protocol), "-odir", _odir(project, destination), "--cpu"]
    if protocol == "AIS":
        argv += ["-source-traj", "../source.dcd"]
    done = subprocess.run(["taskset", "-c", "0-2", "mpirun", "--bind-to", "none", "-n", "2",
                           *argv], cwd=project, capture_output=True, text=True,
                          timeout=LAUNCH_TIMEOUT, env=_environment())
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "3 = 2 x 1 + 1" in message, message[-3000:]
    assert "2 CPUs (1 per worker)" in message and "4 CPUs (2 per worker)" in message, message
    _wrote_nothing(protocol, project, destination, before)


def test_a_cpu_count_that_divides_binds_each_worker_to_its_own_block(projects, tmp_path):
    """Four CPUs for two workers: `--check` passes, and each rank reports its own two CPUs."""
    _require_mpirun()
    if shutil.which("taskset") is None:
        pytest.skip("no taskset on PATH")
    project, destination = _launch_directory(projects, "REST2", tmp_path)
    before = _listing(project)
    done = subprocess.run(
        ["taskset", "-c", "0-3", "mpirun", "--bind-to", "none", "-n", "2",
         "md-openmm", "md-run", "-i", "../input/REST2.in", "-p", "../build/built.pdb",
         *_inputs("REST2"), "-odir", ".", "--cpu", "--check"],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT, env=_environment())
    message = done.stdout + done.stderr
    assert done.returncode == 0, message[-3000:]
    bound = sorted(line.split("bound (")[1].split(")")[0]
                   for line in message.splitlines() if "cpus " in line and " bound (" in line)
    assert len(bound) == 2 and len(set(bound)) == 2, message[-3000:]
    assert "2 bound" in message and "of 4 usable" in message, message[-3000:]
    _wrote_nothing("REST2", project, destination, before)
