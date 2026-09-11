"""The optional-field migration as a recoverable transaction, proved by hard-killing processes.

NetCDF gives no transaction. `createVariable`, a backfill and an attribute write are three
operations, and a `sync()` after them commits only what it reaches. The previous implementation
called one final `sync()` and described the result as atomic. It was not:

    killed after the backfill call, the file held the variable with HDF5 FILL values
    (-9223372036854775806), no history attribute, and nothing recording that a migration had
    been started at all -- and the reader returned those fill values AS seeds.

So the intent is now written and synced BEFORE the first mutation, and every crash point leaves a
file that says what was in progress. These tests kill real subprocesses with `os._exit`; raising
an exception and closing the file cleanly would prove nothing about crash safety.

PLATFORM_POLICY_EXEMPTION: NetCDF transaction semantics and provenance records. No dynamics are
propagated here; the real continuation evidence is the CPU sequence recorded in
`docs/history/journals/20260831_rest2-migration-transaction.md`.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"

netCDF4 = pytest.importorskip("netCDF4")

from md_tools.remd import storage as storage
from md_tools.remd import validate as validate
from md_tools.remd.engine import Configuration                           # noqa: E402

FIELD = "reservoir_velocity_seed"
EXTRA = "synthetic_second_field"
ZERO = np.zeros((2, 2), dtype=np.int64)

#: A second optional field, so partial MULTI-field recovery is exercised rather than assumed from
#: the one field that happens to exist today.
EXTRA_SPEC = {"dtype": "i8", "dimensions": ("exchange",), "absent_value": -1,
              "added_in": "a test", "meaning": "a synthetic optional field"}


def _make_legacy(path, rows=4, drop=(FIELD,)):
    """A v2 file as an earlier build wrote one: copied from a real file, minus the given fields."""
    source = path.with_name(f"_src_{path.stem}.nc")
    reporter = storage.ReplicaReporter.create(
        source, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2, "tau": [0.0, 0.5]}, metadata={})
    for index in range(rows):
        reporter.write_exchange(
            index, step=(index + 1) * 500, time_ps=(index + 1) * 1.0, state_to_walker=[0, 1],
            proposed=ZERO, accepted=ZERO, u=np.full((2, 2), float(index)),
            u_evaluated=np.ones((2, 2), dtype=np.int8), reservoir=None)
    # A coherent run also has the frame streams, so `--verify-only` has something complete to
    # accept and the verification assertions below cannot pass for an unrelated reason.
    configurations = [Configuration(np.zeros((4, 3)), np.zeros((4, 3)), None) for _ in range(2)]
    for index in range(rows):
        step = (index + 1) * 500
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=index,
                                    configurations=configurations, solute_indices=[0])
    reporter.write_frame(step=rows * 500, time_ps=rows * 500 * 0.002, exchange_index=rows - 1)
    reporter.close()
    with netCDF4.Dataset(str(source), "r") as src, netCDF4.Dataset(str(path), "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, None if dimension.isunlimited() else len(dimension))
        for name, variable in src.variables.items():
            if name in drop:
                continue
            out = dst.createVariable(name, variable.datatype, variable.dimensions)
            out.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            out[:] = variable[:]
    source.unlink()
    return path


_RUNNER = textwrap.dedent('''
    import os, sys
    from md_tools.remd import storage
    if {extra!r}:
        storage.OPTIONAL_EXCHANGE_FIELDS[{extra!r}] = {spec!r}
    reporter = storage.ReplicaReporter(sys.argv[1], mode="a")
    try:
        event = reporter.ensure_optional_exchange_fields()
        print("COMPLETED:" + ",".join(event["fields_added"]))
    finally:
        reporter.close()
''')


def _migrate_in_subprocess(path, *, fault=None, extra=None, tmp_path=None):
    """Run the real transaction in another process, optionally hard-killed at `fault`."""
    script = (tmp_path or path.parent) / "runner.py"
    script.write_text(_RUNNER.format(extra=extra, spec=EXTRA_SPEC))
    environment = dict(os.environ)
    if fault:
        environment[storage._MIGRATION_FAULT_ENV] = fault
    else:
        environment.pop(storage._MIGRATION_FAULT_ENV, None)
    return subprocess.run([sys.executable, str(script), str(path)],
                          capture_output=True, text=True, env=environment)


def _read(path, *, extra=None):
    """Everything a later process can see, read-only and from nothing but the file."""
    if extra:
        storage.OPTIONAL_EXCHANGE_FIELDS[extra] = EXTRA_SPEC
    try:
        reporter = storage.ReplicaReporter(path, mode="r")
        try:
            return {
                "pending": reporter.pending_migration(),
                "history": reporter.migration_history(),
                "rows": reporter.n_exchange_rows(),
                "seeds": [int(s) for s in reporter.reservoir_velocity_seeds()],
                "fields": {name: name in reporter.dataset.variables
                           for name in ((FIELD,) + ((extra,) if extra else ()))},
            }
        finally:
            reporter.close()
    finally:
        if extra:
            storage.OPTIONAL_EXCHANGE_FIELDS.pop(extra, None)


def _payload(path):
    with netCDF4.Dataset(str(path), "r") as d:
        return {"u": np.array(d.variables["u"][:]).tolist(),
                "state_to_walker": np.array(d.variables["state_to_walker"][:]).tolist(),
                "proposed": np.array(d.variables["proposed"][:]).tolist(),
                "exchange_step": np.array(d.variables["exchange_step"][:]).tolist(),
                "last_exchange": int(d.variables["last_exchange"][0])}


# --- the fault seam is inert unless asked -------------------------------------------------------

def test_the_fault_seam_is_inert_by_default(tmp_path):
    assert storage._MIGRATION_FAULT_ENV not in os.environ
    path = _make_legacy(tmp_path / "plain.nc")
    done = _migrate_in_subprocess(path, tmp_path=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "COMPLETED" in done.stdout


def test_the_fault_points_are_the_transaction_steps():
    assert storage.MIGRATION_FAULT_POINTS == (
        "after_intent", "after_create", "after_create_durable", "after_backfill", "after_commit")


# --- hard interruption at each transaction point -----------------------------------------------

@pytest.mark.parametrize("fault", ["after_intent", "after_create", "after_backfill",
                                   "after_commit"])
def test_a_hard_kill_leaves_a_file_that_says_what_was_in_progress(tmp_path, fault):
    path = _make_legacy(tmp_path / f"{fault}.nc", rows=4)
    before = _payload(path)

    killed = _migrate_in_subprocess(path, fault=fault, tmp_path=tmp_path)
    assert killed.returncode == 97, f"the process was not hard-killed: {killed.returncode}"
    assert "COMPLETED" not in killed.stdout

    seen = _read(path)
    assert seen["pending"] is not None, "a crash left no record that a migration had begun"
    assert seen["pending"]["fields"] == [FIELD]
    assert seen["pending"]["rows_at_intent"] == 4
    assert seen["pending"]["backfill"] == {FIELD: -1}
    assert seen["pending"]["transaction_id"]
    assert seen["rows"] == 4
    assert _payload(path) == before, "a crashed migration changed committed data"


@pytest.mark.parametrize("fault", ["after_intent", "after_create", "after_backfill",
                                   "after_commit"])
def test_reconciliation_finishes_every_crash_point_exactly_once(tmp_path, fault):
    path = _make_legacy(tmp_path / f"{fault}.nc", rows=4)
    before = _payload(path)
    _migrate_in_subprocess(path, fault=fault, tmp_path=tmp_path)

    crashed = _read(path)
    identity_before = (crashed["pending"] or {}).get("transaction_id")

    done = _migrate_in_subprocess(path, tmp_path=tmp_path)
    assert done.returncode == 0, done.stderr

    seen = _read(path)
    assert seen["pending"] is None, "the pending marker survived a successful reconciliation"
    assert len(seen["history"]) == 1, seen["history"]
    assert seen["history"][0]["fields_added"] == [FIELD]
    assert seen["history"][0]["transaction_id"] == identity_before, (
        "the event identity did not survive the restart")
    assert seen["seeds"] == [-1, -1, -1, -1]
    assert _payload(path) == before, "reconciliation altered committed data"

    # And doing it again changes nothing.
    again = _migrate_in_subprocess(path, tmp_path=tmp_path)
    assert again.returncode == 0
    assert len(_read(path)["history"]) == 1


def test_a_crash_after_commit_does_not_duplicate_the_event(tmp_path):
    """The history was already durable; only the marker was stale. Reconciliation must clear the
    marker and leave the single event alone."""
    path = _make_legacy(tmp_path / "after_commit.nc", rows=4)
    _migrate_in_subprocess(path, fault="after_commit", tmp_path=tmp_path)

    crashed = _read(path)
    assert crashed["pending"] is not None
    assert len(crashed["history"]) == 1, "the event was not committed before the marker"
    committed_id = crashed["history"][0][storage.MIGRATION_EVENT_ID]

    _migrate_in_subprocess(path, tmp_path=tmp_path)
    seen = _read(path)
    assert seen["pending"] is None
    assert len(seen["history"]) == 1
    assert seen["history"][0][storage.MIGRATION_EVENT_ID] == committed_id


# --- multi-field: a partial migration must be recoverable ---------------------------------------

def test_a_partial_multi_field_migration_recovers_both_fields(tmp_path):
    """Not both fields at once: the crash lands after the first is created, so the restart finds a
    mixture of present and absent targets and must finish both."""
    path = _make_legacy(tmp_path / "multi.nc", rows=4, drop=(FIELD,))
    before = _payload(path)

    killed = _migrate_in_subprocess(path, fault="after_create", extra=EXTRA, tmp_path=tmp_path)
    assert killed.returncode == 97

    crashed = _read(path, extra=EXTRA)
    assert crashed["pending"] is not None
    assert sorted(crashed["pending"]["fields"]) == sorted([EXTRA, FIELD])
    assert crashed["pending"]["backfill"] == {EXTRA: -1, FIELD: -1}

    done = _migrate_in_subprocess(path, extra=EXTRA, tmp_path=tmp_path)
    assert done.returncode == 0, done.stderr

    seen = _read(path, extra=EXTRA)
    assert seen["pending"] is None
    assert seen["fields"] == {FIELD: True, EXTRA: True}
    assert len(seen["history"]) == 1
    assert sorted(seen["history"][0]["fields_added"]) == sorted([EXTRA, FIELD])
    with netCDF4.Dataset(str(path), "r") as d:
        assert [int(x) for x in np.array(d.variables[EXTRA][:])] == [-1, -1, -1, -1]
    assert _payload(path) == before


def test_the_pending_record_decides_which_fields_are_this_transactions(tmp_path):
    """A crash after `createVariable` leaves the field present. The intent, not the file's current
    contents, says it is still ours to finish -- otherwise a restart would see nothing missing and
    record no event."""
    path = _make_legacy(tmp_path / "owned.nc", rows=3)
    _migrate_in_subprocess(path, fault="after_backfill", tmp_path=tmp_path)
    crashed = _read(path)
    assert crashed["fields"][FIELD] is True, "the field is present after this crash point"
    assert crashed["history"] == [], "and no event is recorded yet"

    _migrate_in_subprocess(path, tmp_path=tmp_path)
    seen = _read(path)
    assert len(seen["history"]) == 1, (
        "a restart that trusted the file's contents instead of the intent would have recorded "
        "nothing, because the field was already there")


# --- no fabricated provenance -------------------------------------------------------------------

def test_a_current_file_gains_no_marker_no_event_and_no_history(tmp_path):
    path = tmp_path / "modern.nc"
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2}, metadata={})
    reporter.write_exchange(0, step=500, time_ps=1.0, state_to_walker=[0, 1], proposed=ZERO,
                            accepted=ZERO, u=np.zeros((2, 2)),
                            u_evaluated=ZERO.astype(np.int8), reservoir=None)
    reporter.close()

    done = _migrate_in_subprocess(path, tmp_path=tmp_path)
    assert done.returncode == 0
    assert done.stdout.strip().endswith("COMPLETED:")

    seen = _read(path)
    assert seen["pending"] is None
    assert seen["history"] == []
    with netCDF4.Dataset(str(path), "r") as d:
        assert storage.MIGRATION_PENDING_ATTRIBUTE not in d.ncattrs()
        assert storage.MIGRATION_HISTORY_ATTRIBUTE not in d.ncattrs()


def test_a_pre_fix_crash_state_is_refused_and_never_invented(tmp_path):
    """The state the OLD implementation could leave: field present, fill values, no marker, no
    history. The event is unrecoverable -- so it is refused, not guessed at."""
    path = _make_legacy(tmp_path / "lost.nc", rows=3)
    with netCDF4.Dataset(str(path), "a") as d:
        d.createVariable(FIELD, "i8", ("exchange",))        # created, never backfilled
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        report = reporter.inspect_optional_exchange_fields()
    finally:
        reporter.close()
    problem = report[FIELD]["problem"]
    assert problem and "neither -1" in problem
    assert "cannot be reconstructed" in problem

    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError):
            reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()
    assert _read(path)["history"] == [], "a history was fabricated for an unrecoverable file"


# --- verify-only --------------------------------------------------------------------------------

def test_verify_only_reports_a_pending_migration_and_changes_no_byte(tmp_path):
    path = _make_legacy(tmp_path / "pending.nc", rows=4)
    _migrate_in_subprocess(path, fault="after_intent", tmp_path=tmp_path)

    before, mtime = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    result = validate.validate_replica_output(analysis=path, expect_completed=False)

    assert not result.ok, "an incomplete migration is not a coherent run"
    assert any("incomplete migration transaction" in p for p in result.problems), result.problems
    assert not any("whole-system frame" in p for p in result.problems), (
        "the file must otherwise be coherent, or this test proves nothing")
    assert any("--resume" in p for p in result.problems)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert path.stat().st_mtime_ns == mtime
    seen = _read(path)
    assert seen["pending"] is not None, "verification cleared the marker"
    assert seen["history"] == [], "verification committed an event"


def test_verify_only_accepts_a_reconciled_file(tmp_path):
    path = _make_legacy(tmp_path / "done.nc", rows=4)
    _migrate_in_subprocess(path, fault="after_backfill", tmp_path=tmp_path)
    _migrate_in_subprocess(path, tmp_path=tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = validate.validate_replica_output(analysis=path, expect_completed=False)
    assert result.ok, result.problems
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# --- refusals that leave the file alone ---------------------------------------------------------

def _genuine_pending(path):
    """A real record from `_begin_migration`, so a test can change ONE thing about it."""
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        return reporter._begin_migration([FIELD])
    finally:
        reporter.close()


def _resigned(record, **changes):
    """An edited record whose id is recomputed, so the CONTENT rule is what is being tested."""
    edited = dict(record)
    edited.update(changes)
    edited.pop("transaction_id", None)
    edited["transaction_id"] = storage.migration_transaction_id(edited)
    return edited


def test_a_pending_record_for_another_schema_is_refused(tmp_path):
    path = _make_legacy(tmp_path / "wrong.nc", rows=2)
    record = _resigned(_genuine_pending(path), schema="md-tools-replica-exchange/v9")
    with netCDF4.Dataset(str(path), "a") as d:
        d.setncattr(storage.MIGRATION_PENDING_ATTRIBUTE, json.dumps(record, sort_keys=True))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError, match="not begun against this schema"):
            reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_a_pending_record_naming_an_unknown_field_is_refused(tmp_path):
    path = _make_legacy(tmp_path / "unknown.nc", rows=2)
    record = _resigned(_genuine_pending(path), fields=["not_a_field"],
                       definitions={"not_a_field": {"dtype": "i8", "dimensions": ["exchange"]}},
                       backfill={"not_a_field": -1})
    with netCDF4.Dataset(str(path), "a") as d:
        d.setncattr(storage.MIGRATION_PENDING_ATTRIBUTE, json.dumps(record, sort_keys=True))
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError, match="not defined by this runtime"):
            reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()


def test_a_corrupt_pending_record_is_refused_not_guessed(tmp_path):
    path = _make_legacy(tmp_path / "corrupt.nc", rows=2)
    with netCDF4.Dataset(str(path), "a") as d:
        d.setncattr(storage.MIGRATION_PENDING_ATTRIBUTE, "{not json")
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        with pytest.raises(storage.StorageError, match="not valid JSON"):
            reporter.pending_migration()
    finally:
        reporter.close()


# --- documentation must not claim atomicity -----------------------------------------------------

def test_the_code_does_not_claim_atomicity():
    text = (TEMPLATES / "storage.py").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "is not atomic" in lowered or "no transaction" in lowered
    for claim in ("atomically migrates", "atomic migration", "committed by the same `sync()`"):
        assert claim not in lowered, claim


# --- run-state preservation on every failure path ------------------------------------------------
#
# `write_run_state()` REPLACES the sidecar. A failure path that writes one without the history
# erases the only note that a legacy file was ever migrated -- from the path that runs precisely
# when something has gone wrong. These exercise the driver's real handler rather than its text.

class _Files:
    def __init__(self, trajectory, restart=None):
        self.trajectory = str(trajectory)
        self.restart = str(restart) if restart else None
        self.checkpoint = None
        self.output = None
        self.resume = False
        self.extend = 0


class _Driver:
    """The narrowest realistic boundary: the real methods, bound to a stub with the real files."""

    def __init__(self, files):
        from md_tools.remd import REMDRunner as ReplicaRun
        self.files = files
        self.coordinator = type("C", (), {"is_root": True})()
        self._recover_history_readonly = ReplicaRun._recover_history_readonly.__get__(self)
        self._previous_manifest = ReplicaRun._previous_manifest.__get__(self)


def _migrated(tmp_path, name="run.nc"):
    path = _make_legacy(tmp_path / name, rows=4)
    _migrate_in_subprocess(path, tmp_path=tmp_path)
    return path


def test_history_is_recoverable_read_only_when_begin_never_returned(tmp_path):
    """The exception handler's fallback: `_begin` raised, so there is no in-memory state, and the
    history has to come back out of the file."""
    path = _migrated(tmp_path)
    driver = _Driver(_Files(path))
    history = driver._recover_history_readonly()
    assert history is not None
    assert len(history) == 1
    assert history[0]["fields_added"] == [FIELD]


def test_recovery_returns_none_rather_than_an_empty_claim(tmp_path):
    """An unreadable file must not read as 'no migration'. None means 'could not tell', and the
    caller leaves the existing run state alone."""
    missing = tmp_path / "not-here.nc"
    assert _Driver(_Files(missing))._recover_history_readonly() is None

    broken = tmp_path / "broken.nc"
    broken.write_bytes(b"not a netcdf file at all")
    assert _Driver(_Files(broken))._recover_history_readonly() is None


def test_recovery_merges_the_manifest_when_the_file_predates_the_attribute(tmp_path):
    """A run migrated by the previous build recorded the event only in its manifest."""
    path = _make_legacy(tmp_path / "old.nc", rows=4)
    manifest = tmp_path / "restart.json"
    manifest.write_text(json.dumps({"storage_migration": {
        "fields_added": [FIELD], "rows_initialised": 4, "previous_committed_exchanges": 4}}))
    history = _Driver(_Files(path, restart=manifest))._recover_history_readonly()
    assert len(history) == 1, "a legacy singular manifest was not recovered"


def test_every_run_state_write_in_the_driver_carries_the_history():
    """An audit with teeth: every `write_run_state` call must pass `storage_migrations`."""
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    calls, index = [], 0
    while True:
        index = source.find("storage.write_run_state(", index)
        if index == -1:
            break
        depth, end = 0, index
        for end in range(index, len(source)):
            if source[end] == "(":
                depth += 1
            elif source[end] == ")":
                depth -= 1
                if depth == 0:
                    break
        calls.append(source[index:end + 1])
        index = end
    assert len(calls) >= 6, f"expected every run-state write to be found, got {len(calls)}"
    missing = [c.splitlines()[1].strip() for c in calls if "storage_migrations" not in c]
    assert not missing, f"run-state writes with no migration history: {missing}"


def test_the_failure_handler_skips_the_write_when_provenance_is_unreadable():
    """The failure path must not overwrite the run state with a record that lost the history.

    Reads `_fail_closed`, which is where the driver's failure handling lives -- `run()`'s `except`
    delegates to it so the same handling can be tested behaviourally (see
    `test_driver_fail_closed.py`) rather than only by scanning text as this does.
    """
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    start = source.index("def _fail_closed(self")
    handler = source[start:source.index("\n    def ", start + 1)]
    assert "_recover_history_readonly()" in handler
    assert "if history is not None:" in handler, (
        "an unreadable provenance would still overwrite the run state")


def test_a_failed_run_state_keeps_the_history(tmp_path):
    """End to end through the real writer and reader."""
    path = _migrated(tmp_path)
    driver = _Driver(_Files(path))
    history = driver._recover_history_readonly()
    storage.write_run_state(path, "failed", identity={"n_states": 2},
                            storage_migrations=history, reason="synthetic")
    recorded = storage.read_run_state(path)
    assert recorded["status"] == "failed"
    assert len(storage.migration_history_of(recorded)) == 1


def test_a_fresh_run_records_an_empty_history_truthfully(tmp_path):
    path = tmp_path / "fresh.nc"
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2}, metadata={})
    reporter.close()
    storage.write_run_state(path, "running", identity={"n_states": 2}, storage_migrations=[])
    recorded = storage.read_run_state(path)
    assert recorded["storage_migrations"] == []
    assert storage.migration_history_of(recorded) == []


def test_verification_refuses_a_pending_migration_but_continuation_may_reconcile_it(tmp_path):
    """The two contracts are different, and conflating them deadlocks the file.

    `--verify-only` must refuse: it is read-only and cannot finish the transaction. A CONTINUATION
    must not refuse for the same reason -- it is the only thing that can finish it. An earlier
    draft of this branch failed both ways alike, which made a hard-killed migration unrecoverable
    by the one command able to recover it.
    """
    path = _make_legacy(tmp_path / "pending.nc", rows=4)
    _migrate_in_subprocess(path, fault="after_intent", tmp_path=tmp_path)

    verifying = validate.validate_replica_output(analysis=path, expect_completed=False)
    assert not verifying.ok
    assert any("incomplete migration transaction" in p for p in verifying.problems)

    continuing = validate.validate_replica_output(
        analysis=path, expect_completed=False, reconcilable=True)
    assert continuing.ok, continuing.problems
    assert continuing.facts.get("pending_migration") == "will be reconciled before propagation"

    assert _read(path)["pending"] is not None, "neither call may change the file"


def test_the_driver_marks_its_own_check_reconcilable():
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    block = source[source.index("def _continue"):source.index("def _loop")]
    assert "reconcilable=True" in block, (
        "the continuation would refuse the pending migration it exists to finish")
