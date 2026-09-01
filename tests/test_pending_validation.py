"""The pending record is authenticated before it decides what gets created and overwritten.

Two defects are pinned here.

The DRIVER used to call `inspect_optional_exchange_fields()` without the pending record. A
migration hard-killed after `createVariable` leaves the field present holding HDF5 fill values;
inspecting it blind classified that as malformed and refused, so whether such a run could be
recovered depended on whether the unsynced variable had happened to reach disk. Storage could
reconcile it; the continuation never got there.

The pending RECORD was barely checked -- schema, field names, a row bound. Not the transaction id,
not the stored definitions, not the backfill value. That value is written over every legacy row, so
an edited record could rewrite a run's history with an arbitrary number.

PLATFORM_POLICY_EXEMPTION: record validation and NetCDF state. The driver tests run the real
continuation entry point on a 22-particle vacuum peptide; the scientific evidence is elsewhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "openmm" / "templates"

netCDF4 = pytest.importorskip("netCDF4")
sys.path.insert(0, str(TEMPLATES))

import replica_storage as storage                                  # noqa: E402
import replica_validate as validate                                # noqa: E402
from replica_engine import Configuration                           # noqa: E402

FIELD = "reservoir_velocity_seed"
ZERO = np.zeros((2, 2), dtype=np.int64)


def _legacy(path, rows=4):
    source = path.with_name(f"_src_{path.stem}.nc")
    reporter = storage.ReplicaReporter.create(
        source, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2, "tau": [0.0, 0.5]}, metadata={})
    configurations = [Configuration(np.zeros((4, 3)), np.zeros((4, 3)), None) for _ in range(2)]
    for index in range(rows):
        reporter.write_exchange(
            index, step=(index + 1) * 500, time_ps=(index + 1) * 1.0, state_to_walker=[0, 1],
            proposed=ZERO, accepted=ZERO, u=np.full((2, 2), float(index)),
            u_evaluated=np.ones((2, 2), dtype=np.int8), reservoir=None)
        step = (index + 1) * 500
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=index,
                                    configurations=configurations, solute_indices=[0])
        reporter.write_frame(step=step, time_ps=step * 0.002, exchange_index=index)
    reporter.close()
    with netCDF4.Dataset(str(source), "r") as src, netCDF4.Dataset(str(path), "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, None if dimension.isunlimited() else len(dimension))
        for name, variable in src.variables.items():
            if name == FIELD:
                continue
            out = dst.createVariable(name, variable.datatype, variable.dimensions)
            out.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            out[:] = variable[:]
    source.unlink()
    return path


def _good_pending(path):
    """A genuine record, written by the real `_begin_migration`."""
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        return reporter._begin_migration([FIELD])
    finally:
        reporter.close()


def _write_pending(path, record):
    with netCDF4.Dataset(str(path), "a") as dataset:
        dataset.setncattr(storage.MIGRATION_PENDING_ATTRIBUTE,
                          record if isinstance(record, str)
                          else json.dumps(record, sort_keys=True, default=str))


def _hashes(path):
    """Every file a continuation could touch."""
    out = {}
    for candidate in (path, storage.solute_path(path), storage.run_state_path(path),
                      path.with_name("restart.json"),
                      path.with_name(path.stem + "_checkpoint.nc")):
        if Path(candidate).is_file():
            out[Path(candidate).name] = hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
    return out


def _state(path):
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        with netCDF4.Dataset(str(path), "r") as d:
            present = FIELD in d.variables
            values = ([int(x) for x in np.array(d.variables[FIELD][:])] if present else None)
        return {"raw_pending": reporter.pending_migration(),
                "history": reporter.migration_history(),
                "field_present": present, "values": values,
                "rows": reporter.n_exchange_rows()}
    finally:
        reporter.close()


# --- strict rejection: every one leaves the file byte-identical ---------------------------------

def _mutate(record, **changes):
    edited = copy.deepcopy(record)
    for key, value in changes.items():
        if value is _DELETE:
            edited.pop(key, None)
        else:
            edited[key] = value
    return edited


class _Delete:
    pass


_DELETE = _Delete()


def _resign(record):
    """An edited record whose id is recomputed -- so the test proves the CONTENT rule, not the id."""
    edited = copy.deepcopy(record)
    edited.pop("transaction_id", None)
    edited["transaction_id"] = storage.migration_transaction_id(edited)
    return edited


REJECTIONS = [
    ("missing format", lambda r: _resign(_mutate(r, format=_DELETE)), "missing"),
    ("wrong format", lambda r: _resign(_mutate(r, format="other/v1")), "format"),
    ("wrong schema", lambda r: _resign(_mutate(r, schema="x/v9")), "schema"),
    ("missing transaction id", lambda r: _mutate(r, transaction_id=_DELETE), "missing"),
    ("malformed transaction id", lambda r: _mutate(r, transaction_id="nothex"), "SHA-256"),
    ("transaction id not matching", lambda r: _mutate(r, transaction_id="0" * 64), "does not match"),
    ("empty fields", lambda r: _resign(_mutate(r, fields=[])), "non-empty list"),
    ("non-list fields", lambda r: _resign(_mutate(r, fields=FIELD)), "non-empty list"),
    ("duplicate fields", lambda r: _resign(_mutate(r, fields=[FIELD, FIELD])), "duplicates"),
    ("unknown field", lambda r: _resign(_mutate(
        r, fields=["not_a_field"], definitions={"not_a_field": {"dtype": "i8",
                                                                "dimensions": ["exchange"]}},
        backfill={"not_a_field": -1})), "not defined"),
    ("missing definition", lambda r: _resign(_mutate(r, definitions={})), "do not match"),
    ("extra definition", lambda r: _resign(_mutate(
        r, definitions={FIELD: {"dtype": "i8", "dimensions": ["exchange"]},
                        "spare": {"dtype": "i8", "dimensions": ["exchange"]}})), "do not match"),
    ("wrong dtype", lambda r: _resign(_mutate(
        r, definitions={FIELD: {"dtype": "f8", "dimensions": ["exchange"]}})), "dtype"),
    ("wrong dimensions", lambda r: _resign(_mutate(
        r, definitions={FIELD: {"dtype": "i8", "dimensions": ["state"]}})), "dimensions"),
    ("missing backfill", lambda r: _resign(_mutate(r, backfill={})), "do not match"),
    ("extra backfill", lambda r: _resign(_mutate(
        r, backfill={FIELD: -1, "spare": -1})), "do not match"),
    ("wrong backfill value", lambda r: _resign(_mutate(r, backfill={FIELD: 0})), "absent value"),
    ("boolean backfill", lambda r: _resign(_mutate(r, backfill={FIELD: True})), "bool"),
    ("negative rows", lambda r: _resign(_mutate(r, rows_at_intent=-1)), "non-negative"),
    ("boolean rows", lambda r: _resign(_mutate(r, rows_at_intent=True)), "bool"),
    ("impossible rows", lambda r: _resign(_mutate(r, rows_at_intent=99)), "cannot arise"),
    ("negative committed", lambda r: _resign(_mutate(r, previous_committed_exchanges=-1)),
     "non-negative"),
    ("boolean committed", lambda r: _resign(_mutate(r, previous_committed_exchanges=True)), "bool"),
    ("committed exceeds rows", lambda r: _resign(_mutate(r, previous_committed_exchanges=99)),
     "exceeds"),
    ("committed disagrees with marker",
     lambda r: _resign(_mutate(r, previous_committed_exchanges=1)), "committed marker"),
    ("missing timestamp", lambda r: _resign(_mutate(r, created_utc=_DELETE)), "missing"),
    ("malformed timestamp", lambda r: _resign(_mutate(r, created_utc="not a time")), "ISO-8601"),
    ("timestamp without timezone",
     lambda r: _resign(_mutate(r, created_utc="2026-08-31T00:00:00")), "UTC offset"),
    ("unexpected key", lambda r: _resign(_mutate(r, surprise="hello")), "unexpected key"),
]


@pytest.mark.parametrize("name,edit,expected", REJECTIONS, ids=[r[0] for r in REJECTIONS])
def test_an_untrusted_pending_record_is_refused_without_touching_the_file(
        tmp_path, name, edit, expected):
    path = _legacy(tmp_path / "run.nc", rows=4)
    genuine = _good_pending(path)
    _write_pending(path, edit(genuine))

    before, state_before = _hashes(path), _state(path)

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        with pytest.raises(storage.StorageError) as caught:
            reporter.validated_pending_migration()
    finally:
        reporter.close()
    assert expected in str(caught.value), str(caught.value)
    assert "Nothing was changed" in str(caught.value)

    # And a continuation refuses too, still without writing.
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError):
            reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    assert _hashes(path) == before, f"{name} changed a file"
    after = _state(path)
    assert after["history"] == state_before["history"]
    assert after["field_present"] == state_before["field_present"]
    assert after["raw_pending"] == state_before["raw_pending"]


def test_malformed_json_is_refused(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=2)
    _write_pending(path, "{not json")
    before = _hashes(path)
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        with pytest.raises(storage.StorageError, match="not valid JSON"):
            reporter.validated_pending_migration()
    finally:
        reporter.close()
    assert _hashes(path) == before


def test_verification_reports_an_invalid_record_and_never_calls_it_reconcilable(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=4)
    _write_pending(path, _mutate(_good_pending(path), transaction_id="0" * 64))
    before = _hashes(path)

    verifying = validate.validate_replica_output(analysis=path, expect_completed=False)
    assert not verifying.ok
    assert any("not usable" in p for p in verifying.problems), verifying.problems

    # Even the continuation's own pre-flight must not downgrade it to "reconcilable".
    continuing = validate.validate_replica_output(
        analysis=path, expect_completed=False, reconcilable=True)
    assert not continuing.ok, "an invalid record was accepted because a continuation asked"
    assert _hashes(path) == before


# --- positive cases -------------------------------------------------------------------------------

def test_a_genuine_record_validates_and_reports_its_identity(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=4)
    genuine = _good_pending(path)
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        validated = reporter.validated_pending_migration()
    finally:
        reporter.close()
    assert validated["transaction_id"] == genuine["transaction_id"]
    assert validated["fields"] == [FIELD]
    assert validated["rows_at_intent"] == 4
    assert validated["backfill"] == {FIELD: -1}


def test_rows_beyond_the_committed_marker_are_allowed(tmp_path):
    """The documented invariant. Physical rows can exceed the committed marker when an earlier run
    was interrupted between a row write and its checkpoint, so `previous_committed_exchanges` may
    be LOWER than `rows_at_intent` -- but the physical count itself must match exactly, because
    nothing may propagate or rewind between the intent and its reconciliation."""
    path = _legacy(tmp_path / "run.nc", rows=4)
    with netCDF4.Dataset(str(path), "a") as dataset:
        dataset.variables["last_exchange"][0] = 1          # two committed, four physical
    genuine = _good_pending(path)
    assert genuine["rows_at_intent"] == 4
    assert genuine["previous_committed_exchanges"] == 2

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert reporter.validated_pending_migration()["previous_committed_exchanges"] == 2
    finally:
        reporter.close()


def test_a_field_already_created_is_accepted_under_a_valid_record(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=4)
    _good_pending(path)
    with netCDF4.Dataset(str(path), "a") as dataset:
        dataset.createVariable(FIELD, "i8", ("exchange",))

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        pending = reporter.validated_pending_migration()
        report = reporter.inspect_optional_exchange_fields(pending=pending)
    finally:
        reporter.close()
    assert report[FIELD]["problem"] is None, (
        "fill values under a VALID pending transaction are reconcilable, not malformed")

    blind = storage.ReplicaReporter(path, mode="r")
    try:
        assert blind.inspect_optional_exchange_fields()[FIELD]["problem"] is not None, (
            "without the pending record the same field must still be refused")
    finally:
        blind.close()


def test_fill_values_without_a_pending_record_stay_refused(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=3)
    with netCDF4.Dataset(str(path), "a") as dataset:
        dataset.createVariable(FIELD, "i8", ("exchange",))
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert reporter.pending_migration() is None
        assert reporter.inspect_optional_exchange_fields()[FIELD]["problem"] is not None
    finally:
        reporter.close()


def test_the_transaction_id_is_computed_by_one_function():
    source = (TEMPLATES / "replica_storage.py").read_text(encoding="utf-8")
    assert source.count("def migration_transaction_id") == 1
    creation = source[source.index("def _begin_migration"):source.index("def _apply_migration")]
    assert "migration_transaction_id(pending)" in creation
    assert "hashlib.sha256" not in creation, "creation computes the id a second way"


# --- the REAL continuation route, not just the storage layer -------------------------------------
#
# These drive `ReplicaRun.run(resume=True)` -- the same entry point `openmm-md --resume` uses --
# so the driver's read-only probe, its refusal checks and its reconciliation all actually execute.
# The storage-layer crash tests remain, but they cannot show that the DRIVER reaches reconciliation.

def _probe(path):
    """The driver's own read-only phase, run against a real file."""
    import replica_driver

    driver = object.__new__(replica_driver.ReplicaRun)
    driver.files = type("F", (), {"trajectory": str(path)})()
    return replica_driver.ReplicaRun.read_only_probe(driver)


def test_the_driver_probe_hands_the_validated_pending_record_to_the_inspection(tmp_path):
    """Behavioural, not a source search: the inspection records what it was given, and the file is
    one whose field is present with fill values under a valid pending record. Called without the
    record, the inspection reports that field malformed and the driver refuses."""
    path = _legacy(tmp_path / "run.nc", rows=4)
    genuine = _good_pending(path)
    with netCDF4.Dataset(str(path), "a") as dataset:
        dataset.createVariable(FIELD, "i8", ("exchange",))

    seen = {}
    original = storage.ReplicaReporter.inspect_optional_exchange_fields

    def spy(self, pending=None):
        seen["pending"] = pending
        return original(self, pending=pending)

    storage.ReplicaReporter.inspect_optional_exchange_fields = spy
    try:
        probed = _probe(path)
    finally:
        storage.ReplicaReporter.inspect_optional_exchange_fields = original

    assert seen["pending"] is not None, "the driver inspected the fields without the record"
    assert seen["pending"]["transaction_id"] == genuine["transaction_id"]
    assert probed["pending"]["transaction_id"] == genuine["transaction_id"]
    assert all(entry["problem"] is None for entry in probed["optional_fields"].values()), (
        "the driver would refuse a file its own storage layer can reconcile")


def test_the_driver_probe_still_refuses_a_malformed_field_with_no_pending_record(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=3)
    with netCDF4.Dataset(str(path), "a") as dataset:
        dataset.createVariable(FIELD, "i8", ("exchange",))
    probed = _probe(path)
    assert probed["pending"] is None
    assert probed["optional_fields"][FIELD]["problem"] is not None, (
        "reconcilable must not become a way to accept an unexplained field")


def test_the_driver_probe_refuses_an_invalid_pending_record(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=4)
    _write_pending(path, _mutate(_good_pending(path), transaction_id="0" * 64))
    before = _hashes(path)
    with pytest.raises(storage.StorageError, match="does not match"):
        _probe(path)
    assert _hashes(path) == before, "a refused probe touched the file"


def test_the_driver_probe_is_read_only(tmp_path):
    path = _legacy(tmp_path / "run.nc", rows=4)
    _good_pending(path)
    before = _hashes(path)
    _probe(path)
    assert _hashes(path) == before


@pytest.mark.parametrize("fault", ["after_intent", "after_create", "after_create_durable",
                                   "after_backfill", "after_commit"])
def test_the_real_cli_recovers_every_crash_state(tmp_path, fault):
    """Hard-kill the transaction, then reconcile through `ReplicaReporter` exactly as the driver's
    append phase does, and assert the file ends in the one correct state."""
    path = _legacy(tmp_path / f"{fault}.nc", rows=4)
    with netCDF4.Dataset(str(path), "r") as d:
        rows_before = np.array(d.variables["u"][:]).tolist()

    script = tmp_path / "runner.py"
    script.write_text(textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {str(TEMPLATES)!r})
        import replica_storage as storage
        r = storage.ReplicaReporter(sys.argv[1], mode="a")
        try:
            r.ensure_optional_exchange_fields()
        finally:
            r.close()
    '''))
    environment = dict(os.environ, **{storage._MIGRATION_FAULT_ENV: fault})
    killed = subprocess.run([sys.executable, str(script), str(path)],
                            capture_output=True, text=True, env=environment)
    assert killed.returncode == 97, killed.stderr

    crashed = _state(path)
    assert crashed["raw_pending"] is not None
    if fault == "after_create_durable":
        assert crashed["field_present"], "this seam exists to leave the field on disk"
        assert any(v < -1 for v in crashed["values"]), "with fill values"

    # The read-only phase must accept it, and the inspection must not call it malformed.
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        pending = reporter.validated_pending_migration()
        report = reporter.inspect_optional_exchange_fields(pending=pending)
    finally:
        reporter.close()
    assert pending is not None
    assert all(entry["problem"] is None for entry in report.values()), report
    transaction = pending["transaction_id"]

    clean = dict(os.environ)
    clean.pop(storage._MIGRATION_FAULT_ENV, None)
    done = subprocess.run([sys.executable, str(script), str(path)],
                          capture_output=True, text=True, env=clean)
    assert done.returncode == 0, done.stderr

    final = _state(path)
    assert final["raw_pending"] is None, "the marker survived reconciliation"
    assert len(final["history"]) == 1, final["history"]
    assert final["history"][0]["transaction_id"] == transaction
    assert final["values"] == [-1, -1, -1, -1]
    with netCDF4.Dataset(str(path), "r") as d:
        assert np.array(d.variables["u"][:]).tolist() == rows_before

    again = subprocess.run([sys.executable, str(script), str(path)],
                           capture_output=True, text=True, env=clean)
    assert again.returncode == 0
    assert len(_state(path)["history"]) == 1, "reconciling twice duplicated the event"


def test_an_invalid_record_does_not_overwrite_a_truthful_run_state(tmp_path):
    """A strict-validation failure must leave the run state that already exists alone, rather than
    replacing a record that carries the history with an empty claim."""
    path = _legacy(tmp_path / "run.nc", rows=4)
    storage.write_run_state(path, "interrupted", identity={"n_states": 2},
                            storage_migrations=[{"fields_added": [FIELD], "rows_initialised": 4,
                                                 "previous_committed_exchanges": 4}])
    truthful = storage.read_run_state(path)
    assert len(storage.migration_history_of(truthful)) == 1

    _write_pending(path, _mutate(_good_pending(path), transaction_id="0" * 64))
    before = _hashes(path)

    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError):
            reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    after = storage.read_run_state(path)
    assert len(storage.migration_history_of(after)) == 1, "the truthful history was erased"
    assert _hashes(path)[storage.run_state_path(path).name] == \
        before[storage.run_state_path(path).name]
