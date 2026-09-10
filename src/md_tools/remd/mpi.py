"""MPI coordination that fails CLOSED.

The rule, in one sentence: **a launch of more than one rank either coordinates or stops.**

It used to fail open. `barrier()` caught `ImportError` and returned, so `mpirun -n 8` on a machine
without a working `mpi4py` ran eight processes that never met. That is not a slower ladder or a
partly-parallel one. It is eight independent simulations writing over one set of output paths,
each believing it is the whole thing, and the result is a directory of files that look complete.
The exchange record would describe a ladder that never exchanged.

So every collective here is real or fatal. There is no code path in which `barrier`, a broadcast
or a gather quietly becomes a no-op while `size > 1`.

WHAT IS CHECKED, AND WHY EACH ONE

    mpi4py imports, MPI initialised, not finalised   without this nothing else is meaningful
    launcher rank/size == communicator rank/size     the launcher and the library disagreeing
                                                     means the process is not in the world it
                                                     thinks it is in
    communicator size == -ng                         what was launched is what was asked for
    -ng == replica count (REST2/rREST2)              one process per thermodynamic state

All of it runs BEFORE any output directory, resolved configuration, log, Context or coordination
file is created, so a launch that cannot work leaves nothing behind that could be mistaken for a
run that did.

SERIAL IS NOT AFFECTED. World size 1 needs no mpi4py, imports nothing, and every collective is a
genuine no-op because there is genuinely nobody to wait for.
"""
from __future__ import annotations

import os
from typing import Any, Callable

__all__ = ["launcher_rank_and_size", "require_mpi", "check_launch_consistency",
           "MPI_RANK_ENVIRONMENT", "MPI_SIZE_ENVIRONMENT", "barrier", "abort", "Coordination"]

#: Variables an MPI launcher sets in every rank's environment. Read rather than importing mpi4py,
#: because the rank has to be known before anything decides whether to call MPI_Init at all.
MPI_RANK_ENVIRONMENT = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")
MPI_SIZE_ENVIRONMENT = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS")

#: Set by the fail-closed tests to make `mpi4py` unavailable in a child process without uninstalling
#: it. Never set in normal use; read here so the refusal path is exercised by the real command
#: rather than by a mock of it.
FORCE_NO_MPI4PY = "MD_TOOLS_FORCE_NO_MPI4PY"


def launcher_rank_and_size(environment=None) -> tuple[int, int]:
    """What the LAUNCHER says this process is. Not what the library says -- those are compared."""
    environment = os.environ if environment is None else environment
    rank, size = 0, 1
    for name in MPI_RANK_ENVIRONMENT:
        if name in environment:
            try:
                rank = int(environment[name])
                break
            except ValueError:
                pass
    for name in MPI_SIZE_ENVIRONMENT:
        if name in environment:
            try:
                size = int(environment[name])
                break
            except ValueError:
                pass
    return rank, size


def _import_mpi():
    """Import `mpi4py.MPI`, honouring the test switch that makes it unavailable."""
    if os.environ.get(FORCE_NO_MPI4PY):
        raise ImportError(f"{FORCE_NO_MPI4PY} is set: mpi4py is being treated as unavailable")
    from mpi4py import MPI                                # noqa: PLC0415 - deliberate, see module

    return MPI


def require_mpi(*, size: int, importer: Callable[[], Any] | None = None):
    """Return a live `MPI` module for a multi-rank launch, or refuse. None when `size == 1`.

    `importer` exists so the refusal can be tested without arranging for a broken install.
    """
    if int(size) <= 1:
        return None                                       # genuinely nobody to coordinate with

    importer = importer or _import_mpi
    try:
        MPI = importer()
    except Exception as failure:                          # noqa: BLE001 - reported below
        raise SystemExit(
            f"this command was launched with {size} ranks, but MD-tools cannot coordinate them: "
            f"mpi4py is unavailable ({type(failure).__name__}: {failure}).\n"
            f"  Refusing rather than continuing. {size} uncoordinated processes would each run a "
            f"whole simulation over the SAME output paths, and the result would look complete: a "
            f"ladder that never exchanged, or a set of paths written over one another.\n"
            f"  Install mpi4py into this environment, or run a single process without a "
            f"launcher.") from None

    if not MPI.Is_initialized():
        raise SystemExit(
            f"mpi4py imported but MPI is not initialised in this process, which was launched with "
            f"{size} ranks. Nothing can be coordinated; refusing before any output is written.")
    if MPI.Is_finalized():
        raise SystemExit(
            "MPI has already been finalised in this process; no collective can be issued. "
            "Refusing before any output is written.")
    return MPI


def check_launch_consistency(*, launcher_rank: int, launcher_size: int,
                             comm_rank: int, comm_size: int,
                             number_of_groups: int | None = None,
                             replicas: int | None = None,
                             protocol: str = "this run") -> None:
    """Every number that describes the launch must be the same number.

    Reported as a table of all of them rather than as the first pair that differed: the reader has
    to see which one is the odd one out, and naming two of four leaves them guessing.
    """
    rows = [
        ("launcher world size", launcher_size),
        ("MPI communicator size", comm_size),
    ]
    if number_of_groups is not None:
        rows.append(("-ng on the command line", int(number_of_groups)))
    if replicas is not None:
        rows.append(("replicas in the configuration", int(replicas)))

    values = {value for _, value in rows}
    if len(values) > 1:
        listing = "\n".join(f"  {name:<30}: {value}" for name, value in rows)
        raise SystemExit(
            f"{protocol} was launched with numbers that do not agree:\n{listing}\n"
            f"These must all be the same. Refusing before any output is written: a mismatch "
            f"leaves work either unowned or done twice, and the record would describe neither.")

    if int(launcher_rank) != int(comm_rank):
        raise SystemExit(
            f"the launcher says this process is rank {launcher_rank} but the MPI communicator "
            f"says it is rank {comm_rank}. The process is not in the world it thinks it is in; "
            f"refusing before any output is written.")


class Coordination:
    """A live world, or a genuine single process. Never a fake one.

    Every method is a real collective when `size > 1`. `require_mpi` has already refused the case
    where that is impossible, so there is no branch here in which a collective silently does
    nothing while other ranks are waiting.

    This is the ONLY coordinator. `md_tools.remd.driver.Coordinator` was a second one with its own
    permissive import, and `md_tools.remd.executor.barrier` a third; both answered "mpi4py is
    missing" with "then there is nobody to wait for", which is true only if the launcher agrees --
    and under `mpirun -n 8` it does not.
    """

    def __init__(self, *, MPI=None, rank: int = 0, size: int = 1) -> None:
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD if MPI is not None else None
        self.rank = int(rank)
        self.size = int(size)

    @classmethod
    def open(cls, *, number_of_groups: int | None = None, replicas: int | None = None,
             protocol: str = "this run") -> "Coordination":
        """The whole preflight: detect, require, cross-check. Call before creating any output."""
        launcher_rank, launcher_size = launcher_rank_and_size()
        MPI = require_mpi(size=launcher_size)
        if MPI is None:
            if number_of_groups is not None and int(number_of_groups) > 1:
                raise SystemExit(
                    f"-ng {number_of_groups} was requested but this process was not started by an "
                    f"MPI launcher: the world size is 1.\n"
                    f"  -ng says how many processes coordinate; it does not create them.\n"
                    f"  mpirun -n {number_of_groups} md-openmm md-run "
                    f"-ng {number_of_groups} ...")
            return cls()

        comm = MPI.COMM_WORLD
        check_launch_consistency(
            launcher_rank=launcher_rank, launcher_size=launcher_size,
            comm_rank=comm.Get_rank(), comm_size=comm.Get_size(),
            number_of_groups=number_of_groups, replicas=replicas, protocol=protocol)
        return cls(MPI=MPI, rank=comm.Get_rank(), size=comm.Get_size())

    def barrier(self) -> None:
        if self.comm is not None:
            self.comm.barrier()

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def allgather(self, value):
        return [value] if self.comm is None else self.comm.allgather(value)

    def bcast(self, value):
        return value if self.comm is None else self.comm.bcast(value, root=0)

    def all_agree(self, value: bool) -> bool:
        """True on every rank if it is true on any. Used to turn one rank's failure into all."""
        if self.comm is None:
            return bool(value)
        return bool(max(self.comm.allgather(bool(value))))

    #: The driver spells it `any_true`. One implementation, two names, rather than two
    #: implementations that could disagree about what "any" means under a partial failure.
    any_true = all_agree

    def agree(self, value, *, what: str) -> None:
        """Every rank must present the same value. Proves shared state is actually shared."""
        if self.comm is None:
            return
        values = self.comm.allgather(value)
        if len(set(values)) != 1:
            raise SystemExit(
                f"the ranks disagree about {what}: {sorted(set(values))[:4]}. Continuing would "
                f"propagate different states under one record.")

    #: Test seam: fail inside a named phase on named ranks, as `PHASE:0,3`. A rank-local failure
    #: after the preflight -- a directory that cannot be created, a report that cannot be opened,
    #: a reporter that cannot be constructed, a checkpoint that will not load -- is the case this
    #: machinery exists for, and it cannot be provoked reliably any other way.
    FAIL_PHASE_ENVIRONMENT = "MD_TOOLS_FAIL_PHASE"

    def phase(self, name: str):
        """A context manager that makes a rank-local failure everyone's failure.

        `if rank == 0: ...` followed by a collective is a deadlock waiting for a bad input: the
        failing rank raises, exits, and every other rank waits at the next collective for a
        participant that has already gone. The launcher then reports nothing and the job holds its
        GPUs until a wall clock kills it.

        Preflight agreement is not enough. Everything AFTER it is rank-local again -- creating a
        directory, opening a report, publishing a helper, constructing a reporter, loading a
        checkpoint, building a Simulation, finalising outputs -- and each is a place one rank can
        fail alone. Wrapping each phase in this is what turns those into a stopped job rather than
        a hung one.

        Every rank must enter the same phases in the same order: the `allgather` inside is itself
        a collective, so a phase entered by only some ranks would be the very deadlock it
        prevents.
        """
        import contextlib

        @contextlib.contextmanager
        def _guard():
            failure = None
            try:
                self._fail_here_if_asked(name)
                yield
            except BaseException as broken:                # noqa: BLE001 - reported collectively
                failure = f"{type(broken).__name__}: {broken}"
            if self.size > 1:
                reports = self.allgather(failure)
                bad = [(rank, message) for rank, message in enumerate(reports) if message]
                if bad:
                    detail = "; ".join(f"rank {rank}: {message}" for rank, message in bad[:4])
                    self.fail(f"{name} failed on {len(bad)} of {self.size} rank(s): {detail}")
            elif failure:
                raise SystemExit(f"{name}: {failure}")

        return _guard()

    def _fail_here_if_asked(self, name: str) -> None:
        """Honour the phase-failure test seam. A no-op in every normal run."""
        import os

        wanted = os.environ.get(self.FAIL_PHASE_ENVIRONMENT)
        if not wanted or ":" not in wanted:
            return
        # rpartition, not partition: a phase name contains a colon ("REST2: opening the rank
        # report"), and splitting on the first one made every request name the protocol and never
        # match a phase -- so the seam silently did nothing and the tests using it passed for the
        # wrong reason.
        phase_name, _, ranks = wanted.rpartition(":")
        if phase_name != name:
            return
        if self.rank in {int(part) for part in ranks.replace(",", " ").split()}:
            raise RuntimeError(
                f"{self.FAIL_PHASE_ENVIRONMENT} names this rank: simulating a rank-local failure "
                f"in {name!r} on rank {self.rank} of {self.size}")

    def fail(self, message: str, *, code: int = 1):
        """A rank-local fatal error, made collective.

        A rank that raises alone leaves the others waiting in the next collective forever. This
        prints the reason from the rank that has it and takes the whole communicator down, so a
        multi-rank failure is a stopped job rather than a hung one.
        """
        import sys as _sys

        text = f"[rank {self.rank}/{self.size}] fatal: {message}"
        print(text, file=_sys.stderr, flush=True)
        # AND THE TERMINAL. The executor runs the whole ladder inside
        # `contextlib.redirect_stderr(<protocol>.out)`, so `_sys.stderr` above is that file --
        # and `Abort` below ends the job without returning through the code that would have said
        # "see <protocol>.out". `mpirun` therefore printed "MPI_ABORT was invoked" and nothing
        # else, and the only way to read why a ladder refused was to re-run it single-rank.
        # `_sys.__stderr__` is the interpreter's own stderr and survives the redirect.
        original = getattr(_sys, "__stderr__", None)
        if original is not None and original is not _sys.stderr:
            try:
                print(text, file=original, flush=True)
            except (ValueError, OSError):                  # a closed or unusable original stream
                pass
        if self.comm is not None:
            self.comm.Abort(int(code))
        raise SystemExit(int(code))

    def abort(self, code: int = 1) -> None:
        """Stop the whole world. A rank that dies alone leaves the others integrating forever."""
        if self.comm is not None:
            self.comm.Abort(int(code))
        raise SystemExit(int(code))


def barrier(size: int | None = None, *, coordination: Coordination | None = None) -> None:
    """Wait for every rank. Fatal, not silent, when the world is plural and MPI is unusable."""
    if coordination is not None:
        coordination.barrier()
        return
    if size is None:
        _, size = launcher_rank_and_size()
    if int(size) <= 1:
        return
    MPI = require_mpi(size=int(size))
    MPI.COMM_WORLD.barrier()


def abort(code: int = 1, *, size: int | None = None) -> None:
    """Terminate the whole communicator. Used where a rank-local failure is unrecoverable."""
    if size is None:
        _, size = launcher_rank_and_size()
    if int(size) > 1:
        try:
            from mpi4py import MPI

            MPI.COMM_WORLD.Abort(int(code))
        except Exception:                                  # noqa: BLE001 - already failing
            pass
    raise SystemExit(int(code))
