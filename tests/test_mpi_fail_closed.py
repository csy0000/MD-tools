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

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = pytest.mark.slow

#: Long enough for a launcher to start N interpreters and for them to refuse; short enough that a
#: hang is a failure rather than a wait. A refusal takes well under a second.
LAUNCH_TIMEOUT = 120


def _require_mpirun():
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")


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
    for protocol in ("REST2", "AIS"):
        configuration = root / f"{protocol}.config"
        document = {"protocol": protocol, "solvent": "explicit"}
        if protocol == "REST2":
            document["rest2"] = {"number_of_replicas": 2}
        else:
            document["ais_source"] = {"trajectory": "../source.dcd"}
        configuration.write_text(yaml.safe_dump(document), encoding="utf-8")
        done = subprocess.run(CLI + ["build-md", "-odir", str(root / protocol),
                                     "--config", str(configuration)],
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stdout + done.stderr
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
        CLI + ["build-top", "-i", str(ala), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    # Enough eligible frames for the default 100 paths. With eight, the AIS preflight refused on
    # the frame count and every test using this fixture stopped BEFORE the runtime behaviour it
    # was written for -- passing on the return code while never reaching the phase under test.
    mdtraj.join([frames] * 128).save_dcd(str(root / "source.dcd"))
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
    project = projects / protocol
    destination = tmp_path / "never"

    if entry == "md-run":
        argv = ["md-openmm", "md-run", "-i", f"{protocol}.in",
                "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)]
    else:
        argv = [sys.executable, str(project / f"{protocol}.py"),
                "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)]
    if protocol == "AIS":
        argv += ["-source-traj", "../source.dcd"]

    done = subprocess.run(["mpirun", "-n", "2", *argv], cwd=project, capture_output=True,
                          text=True, timeout=LAUNCH_TIMEOUT,
                          env=_environment(MD_TOOLS_FORCE_NO_MPI4PY="1"))
    assert done.returncode != 0, done.stdout[-2000:]
    assert "mpi4py" in done.stdout + done.stderr, (done.stdout + done.stderr)[-2000:]
    assert not destination.exists(), sorted(p.name for p in destination.iterdir())


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
    project = projects / "REST2"
    destination = tmp_path / "never"
    done = subprocess.run(
        ["mpirun", "-n", "2", "md-openmm", "md-run", "-i", "REST2.in",
         "-p", "../built.pdb", "-s", "../built.xml", "-ng", "4", "-odir", str(destination)],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment())
    assert done.returncode != 0
    message = done.stdout + done.stderr
    assert "-ng on the command line" in message and "MPI communicator size" in message, message
    assert not destination.exists()


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
    project = projects / protocol
    destination = tmp_path / "never"

    if entry == "md-run":
        argv = ["md-openmm", "md-run", "-i", f"{protocol}.in",
                "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)]
    else:
        argv = [sys.executable, str(project / f"{protocol}.py"),
                "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)]
    if protocol == "AIS":
        argv += ["-source-traj", "../source.dcd"]

    done = subprocess.run(["mpirun", "-n", "2", *argv], cwd=project, capture_output=True,
                          text=True, timeout=LAUNCH_TIMEOUT, env=_environment(**{FAIL_RANKS: "1"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    # Named as a rank-local failure, with the rank in it -- so a person reading either rank's
    # output learns which one failed rather than that "the launch failed".
    assert "rank 1" in message, message[-3000:]
    assert not destination.exists(), sorted(p.name for p in destination.iterdir())


def test_every_rank_failing_reports_the_reason_once_rather_than_n_times(projects, tmp_path):
    """A condition every rank hits is not rank-local, and repeating it N times only hides it."""
    _require_mpirun()
    project = projects / "REST2"
    destination = tmp_path / "never"
    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{FAIL_RANKS: "0,1"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0
    assert "of 2 rank(s), so the whole launch is refused" not in message, (
        "a condition every rank hit was reported as a partial, rank-local failure:\n" + message)
    assert not destination.exists()


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
    project = projects / "REST2"
    destination = tmp_path / "hung"
    destination.mkdir()
    # A helper from another ladder: content-addressed, so rank 0 refuses it rather than replacing
    # it, and refuses it in the one place only rank 0 reaches.
    (destination / "_protocol.py").write_text("n_states = 99\n", encoding="utf-8")

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment())
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "rank 0 could not prepare" in message or "generated from different content" in message, (
        message[-2500:])


def test_the_reservoir_declaration_is_written_by_rank_zero_alone():
    """Every rank used to write `reservoir.yaml`, then every rank read it.

    Eight processes truncating and rewriting one small YAML file while others parse it is an
    intermittent failure that looks like a corrupt configuration. It goes through the same
    rank-0-writes / everyone-verifies-the-digest path as the other helpers now, and the function
    that builds it returns TEXT so it cannot write anything by itself.
    """
    import inspect

    from md_tools.remd import generated

    source = inspect.getsource(generated.reservoir_declaration_text)
    for forbidden in ("write_text", "open(", "os.replace"):
        assert forbidden not in source, (
            f"reservoir_declaration_text writes to the filesystem ({forbidden}); it must return "
            f"text and leave the writing to the rank-0 helper path")
    assert "reservoir_file" in inspect.getsource(generated.replica_main)


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
    project = projects / "REST2"
    destination = tmp_path / f"phase-{failing_rank}-{phase.replace(' ', '-')}"

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination)],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{FAIL_PHASE: f"REST2: {phase}:{failing_rank}"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert f"rank {failing_rank}" in message, message[-2500:]
    # No authoritative completion output survives a failed launch.
    assert not sorted(destination.glob("remd*.nc")), sorted(destination.glob("remd*.nc"))
    assert not (destination / "restart.json").exists()


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_a_rank_local_failure_after_preflight_stops_the_whole_ais_run(failing_rank, projects,
                                                                      tmp_path):
    _require_mpirun()
    project = projects / "AIS"
    destination = tmp_path / f"ais-phase-{failing_rank}"

    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "AIS.py"),
         "-p", "../built.pdb", "-s", "../built.xml", "-source-traj", "../source.dcd",
         "-odir", str(destination)],
        cwd=project, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT,
        env=_environment(**{FAIL_PHASE: f"AIS: opening the rank reports:{failing_rank}"}))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert f"rank {failing_rank}" in message, message[-2500:]
    assert not sorted(destination.glob("AIS_traj*.nc"))
    assert not (destination / "AIS_hs.csv").exists()


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
