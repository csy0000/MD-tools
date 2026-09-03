"""The driver's fail-closed handler, in process and without a launcher.

`test_mpi_fail_closed.py` proves the whole mechanism end to end under a real `mpirun`, and that
is the evidence that matters. It also needs a launcher, takes seconds per case, and cannot easily
provoke the one case that matters most: a SECOND failure while reporting the first.

These do, in microseconds, and they assert the ordering guarantee directly rather than through
its consequences -- so they run in the fast lane, on every commit, on a machine with no MPI at
all.
"""
from __future__ import annotations

import pytest


class _RecordingCoordinator:
    """A stand-in for `Coordination` that records the abort instead of performing it.

    A real `fail()` calls `MPI_Abort` and never returns, which cannot be observed from inside a
    test process. Recording it is the only way to assert that it WOULD have been called, and when.
    """

    def __init__(self, *, rank=0, size=4):
        self.rank = rank
        self.size = size
        self.failed = []

    @property
    def is_root(self):
        return self.rank == 0

    def fail(self, message, *, code=1):
        self.failed.append(message)
        # The real one does not return: it aborts the communicator and raises SystemExit. The
        # tests below rely on that, because a handler that carried on afterwards would be wrong.
        raise SystemExit(code)


def _bare_run(monkeypatch, coordinator, *, write_run_state=None):
    """A `ReplicaRun` with only the attributes `_fail_closed` touches.

    Built with `object.__new__` deliberately: constructing a real one needs a protocol, a System
    and a topology, none of which this code path reads, and the indirection would obscure what is
    actually under test.
    """
    from md_tools.remd import driver as driver_module
    from md_tools.remd.driver import ReplicaRun

    run = object.__new__(ReplicaRun)
    run.coordinator = coordinator
    run.files = type("Files", (), {"trajectory": "unused.nc"})()
    if write_run_state is not None:
        monkeypatch.setattr(driver_module.storage, "write_run_state", write_run_state)
    return run


def test_a_failure_while_reporting_the_failure_still_aborts_the_communicator(monkeypatch):
    """THE hardening: the likely second failure must not turn a stopped job into a hung one.

    A full disk or a stalled filesystem is often the reason the FIRST failure happened, and both
    surface again when the handler tries to record it. If that escaped, `fail()` would be skipped
    and every other rank would wait forever on a rank that had already given up.
    """
    def explode(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    coordinator = _RecordingCoordinator(rank=0, size=4)
    run = _bare_run(monkeypatch, coordinator, write_run_state=explode)

    with pytest.raises(SystemExit):
        run._fail_closed({"storage_migrations": []}, {"id": "x"}, RuntimeError("the real failure"))

    assert coordinator.failed, (
        "the reporting failure escaped and the communicator was never taken down")
    assert "the real failure" in coordinator.failed[0], (
        "the abort must name the ORIGINAL failure, not the one from writing the report")


def test_a_non_root_rank_aborts_without_attempting_to_write_the_root_only_record(monkeypatch):
    """Only the root owns the run-state file. A non-root rank writing it would be a race."""
    def explode(*_args, **_kwargs):                     # pragma: no cover - must not be reached
        raise AssertionError("a non-root rank tried to write the run state")

    coordinator = _RecordingCoordinator(rank=2, size=4)
    run = _bare_run(monkeypatch, coordinator, write_run_state=explode)

    with pytest.raises(SystemExit):
        run._fail_closed(None, {"id": "x"}, RuntimeError("device side assert"))
    assert coordinator.failed


def test_a_serial_run_reports_but_does_not_abort(monkeypatch):
    """`size == 1` has no communicator to take down, and calling Abort there would be wrong.

    The failure still propagates -- `run()` re-raises it -- but through the ordinary exception
    path, so a single-process user gets a traceback rather than an MPI abort message.
    """
    written = []
    coordinator = _RecordingCoordinator(rank=0, size=1)
    run = _bare_run(monkeypatch, coordinator,
                    write_run_state=lambda *a, **k: written.append(k.get("reason")))

    run._fail_closed({"storage_migrations": []}, {"id": "x"}, RuntimeError("boom"))

    assert not coordinator.failed, "a serial run must not abort a communicator it does not have"
    assert written and "boom" in written[0], "the serial path still records why it failed"


def test_the_recorded_status_distinguishes_an_interrupt_from_a_failure(monkeypatch):
    """Ctrl-C is resumable and a crash is not; recording both as `failed` loses that."""
    written = {}
    coordinator = _RecordingCoordinator(rank=0, size=1)
    run = _bare_run(monkeypatch, coordinator,
                    write_run_state=lambda _path, status, **k: written.update(status=status))

    run._fail_closed({"storage_migrations": []}, {"id": "x"}, KeyboardInterrupt())
    assert written["status"] == "interrupted"

    run._fail_closed({"storage_migrations": []}, {"id": "x"}, RuntimeError("boom"))
    assert written["status"] == "failed"
