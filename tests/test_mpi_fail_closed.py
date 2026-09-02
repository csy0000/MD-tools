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
    # Files that exist, so preflight reaches the MPI check rather than refusing on the inputs.
    (root / "built.pdb").write_text("END\n", encoding="utf-8")
    (root / "built.xml").write_text("<System/>\n", encoding="utf-8")
    (root / "source.dcd").write_bytes(b"\x54\x00\x00\x00CORD" + b"\x00" * 200)
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
