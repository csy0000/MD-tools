"""Migration provenance is cumulative, deduplicated, and durable across interruption.

The singular `storage_migration` field held ONE value and was rewritten by every continuation. A
later resume or extension that needed no schema change produced a no-op record and overwrote the
real migration event, so a completed run could end up with no record that its file had ever been a
legacy file at all.

The replacement is an append-only `storage_migrations` history, identified per event, merged from
every authoritative record, and written into the analysis file itself in the same `sync()` as the
mutation it describes -- so a process killed immediately after migrating still leaves a file that
says it was migrated.

PLATFORM_POLICY_EXEMPTION: provenance records and NetCDF attributes. Nothing here propagates
dynamics; the real continuation evidence is the CPU end-to-end sequence recorded in
`docs/history/journals/20260831_rest2-migration-provenance.md`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"

netCDF4 = pytest.importorskip("netCDF4")

from md_tools.remd import storage as storage
from md_tools.remd import validate as validate
from md_tools.remd.engine import Configuration                           # noqa: E402

FIELD = "reservoir_velocity_seed"
ZERO = np.zeros((2, 2), dtype=np.int64)


def _event(fields=("reservoir_velocity_seed",), rows=3, prior=3, when="2026-08-31T00:00:00+00:00"):
    return {"schema": storage.SCHEMA_VERSION, "fields_added": list(fields),
            "fields_already_present": [], "rows_initialised": rows,
            "initialised_to": {name: -1 for name in fields},
            "previous_committed_exchanges": prior, "recorded_utc": when, "note": "synthetic"}


def _noop(prior=9):
    return {"schema": storage.SCHEMA_VERSION, "fields_added": [],
            "fields_already_present": [FIELD], "rows_initialised": 0, "initialised_to": {},
            "previous_committed_exchanges": prior, "recorded_utc": "2026-08-31T01:00:00+00:00",
            "note": "synthetic"}


def _rows(reporter, count, first=0):
    for offset in range(count):
        index = first + offset
        reporter.write_exchange(
            index, step=(index + 1) * 500, time_ps=(index + 1) * 1.0, state_to_walker=[0, 1],
            proposed=ZERO, accepted=ZERO, u=np.full((2, 2), float(index)),
            u_evaluated=np.ones((2, 2), dtype=np.int8))


def _modern(path, rows=3):
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2, "tau": [0.0, 0.5]}, metadata={})
    configurations = [Configuration(np.zeros((4, 3)), np.zeros((4, 3)), None) for _ in range(2)]
    _rows(reporter, rows)
    for index in range(rows):
        step = (index + 1) * 500
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=index,
                                    configurations=configurations, solute_indices=[0])
    reporter.write_frame(step=rows * 500, time_ps=rows * 500 * 0.002, exchange_index=rows - 1)
    reporter.close()
    return path


def _legacy(path, rows=3):
    """A v2 file as an earlier build wrote one: copied from a real file, minus the optional field."""
    source = path.parent / f"_src_{path.stem}.nc"
    _modern(source, rows=rows)
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


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- 1/2. legacy singular records ---------------------------------------------------------------

def test_a_meaningful_legacy_singular_record_becomes_a_one_event_history():
    record = {"storage_migration": _event()}
    history = storage.migration_history_of(record)
    assert len(history) == 1
    assert history[0]["fields_added"] == [FIELD]
    assert history[0][storage.MIGRATION_EVENT_ID]


def test_a_legacy_singular_no_op_is_ignored():
    """A record that added nothing never described a change, and keeping it is what let a later
    no-op look exactly like the real migration before it."""
    assert storage.migration_history_of({"storage_migration": _noop()}) == []
    assert storage.migration_history_of({"storage_migration": None}) == []
    assert storage.migration_history_of({}) == []


def test_a_record_with_no_history_at_all_yields_nothing():
    assert storage.migration_history_of({"storage_migrations": []}) == []
    assert storage.migration_history_of(None) == []


# --- 3/4. deduplication and ordering ------------------------------------------------------------

def test_the_same_event_from_two_sources_collapses_to_one():
    manifest = {"storage_migrations": [_event()]}
    runstate = {"storage_migration": _event()}          # the same event, legacy shape
    merged = storage.merge_migration_histories(
        storage.migration_history_of(manifest), storage.migration_history_of(runstate))
    assert len(merged) == 1


def test_distinct_migrations_stay_separate_and_ordered():
    first = _event(fields=("reservoir_velocity_seed",), rows=3, prior=3)
    second = _event(fields=("some_later_field",), rows=9, prior=9,
                    when="2026-09-01T00:00:00+00:00")
    merged = storage.merge_migration_histories([first], [second])
    assert [e["fields_added"] for e in merged] == [["reservoir_velocity_seed"],
                                                   ["some_later_field"]]
    merged_again = storage.merge_migration_histories([first, second], [first], [second])
    assert len(merged_again) == 2
    assert [e["fields_added"] for e in merged_again] == [e["fields_added"] for e in merged]


def test_event_identity_is_deterministic_and_content_addressed():
    one, two = _event(), _event()
    assert storage.migration_event_id(one) == storage.migration_event_id(two)
    assert storage.migration_event_id(one) != storage.migration_event_id(_event(rows=4))
    stamped = storage.normalise_migration_event(one)
    assert storage.migration_event_id(stamped) == stamped[storage.MIGRATION_EVENT_ID], (
        "the id must not depend on itself")


# --- durability: the file records its own history in the same sync as the mutation -------------

def test_the_migration_event_is_written_into_the_file_itself(tmp_path):
    """The provenance gap this closes: a process killed between mutating the schema and writing
    the run state used to leave storage changed with nothing saying so."""
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        event = reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()
    assert event["fields_added"] == [FIELD]

    # Reopened from scratch: nothing but the file itself.
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        history = reporter.migration_history()
    finally:
        reporter.close()
    assert len(history) == 1
    assert history[0]["fields_added"] == [FIELD]
    assert history[0]["previous_committed_exchanges"] == 3


def test_a_new_file_carries_no_history(tmp_path):
    path = _modern(tmp_path / "modern.nc", rows=2)
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert reporter.migration_history() == []
    finally:
        reporter.close()


# --- 5/6. a no-op continuation cannot erase the real event --------------------------------------

def test_a_no_op_migration_after_a_real_one_preserves_the_real_one(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()          # the real migration
    finally:
        reporter.close()

    for _ in range(3):                                      # three no-op continuations
        reporter = storage.ReplicaReporter(path, mode="a")
        try:
            event = reporter.ensure_optional_exchange_fields()
            assert event["fields_added"] == []
        finally:
            reporter.close()

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        history = reporter.migration_history()
    finally:
        reporter.close()
    assert len(history) == 1, "a no-op continuation replaced or duplicated the real event"
    assert history[0]["fields_added"] == [FIELD]


def test_merging_a_no_op_into_a_history_changes_nothing():
    real = storage.normalise_migration_event(_event())
    assert storage.merge_migration_histories([real], [_noop()]) == [real]
    assert storage.merge_migration_histories([real], [_noop()], [_noop()]) == [real]


# --- 8/9/10. the file's rows, and verify-only ---------------------------------------------------

def test_migration_changes_no_existing_row(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    with netCDF4.Dataset(str(path), "r") as d:
        before = {name: np.array(d.variables[name][:]).tolist()
                  for name in ("u", "state_to_walker", "proposed", "accepted", "exchange_step")}
        before["last_exchange"] = int(d.variables["last_exchange"][0])

    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    with netCDF4.Dataset(str(path), "r") as d:
        after = {name: np.array(d.variables[name][:]).tolist()
                 for name in ("u", "state_to_walker", "proposed", "accepted", "exchange_step")}
        after["last_exchange"] = int(d.variables["last_exchange"][0])
        assert [int(s) for s in np.array(d.variables[FIELD][:])] == [-1, -1, -1]
    assert after == before


def test_verify_only_adds_no_history_and_changes_no_byte(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    before, before_mtime = _sha(path), path.stat().st_mtime_ns

    result = validate.validate_replica_output(analysis=path, expect_completed=False)
    assert result.ok, result.problems

    assert _sha(path) == before
    assert path.stat().st_mtime_ns == before_mtime
    with netCDF4.Dataset(str(path), "r") as d:
        assert storage.MIGRATION_HISTORY_ATTRIBUTE not in d.ncattrs()
        assert FIELD not in d.variables


def test_current_storage_needs_no_migration_and_gains_no_event(tmp_path):
    path = _modern(tmp_path / "modern.nc", rows=2)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        event = reporter.ensure_optional_exchange_fields()
        history = reporter.migration_history()
    finally:
        reporter.close()
    assert event["fields_added"] == []
    assert history == [], "a run that needed no migration was given a fabricated event"


# --- 7/11/12. the driver's contract, and manifest replacement -----------------------------------

def test_the_driver_merges_every_authoritative_source():
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    # `def _continue(self` and not `def _continue` -- the loose prefix also matches
    # `_continue_cv_states`, which sits earlier in the file, and the slice then spanned half the
    # class and compared offsets from the wrong block.
    block = source[source.index("def _continue(self"):source.index("def _loop(self")]
    probe = source[source.index("def read_only_probe"):source.index("def _previous_manifest")]
    assert "merge_migration_histories" in block
    assert "migration_history()" in probe, "the file's own durable history is not read"
    assert "_previous_manifest()" in block, "the previous completion manifest is not read"
    assert "read_run_state" in block, "the run state is not read"
    # and the merge happens before the payload is built
    assert block.index("merge_migration_histories") < block.index('"storage_migrations"')


def test_the_driver_never_writes_the_singular_field():
    """The driver writes only the plural history. The singular key survives in exactly one place
    -- the legacy reader -- because old records still have to be understood."""
    driver = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    for line in driver.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert '"storage_migration"' not in stripped, stripped
        assert "storage_migration=" not in stripped, stripped

    store = (TEMPLATES / "storage.py").read_text(encoding="utf-8")
    reader = store[store.index("def migration_history_of"):store.index("def merge_migration")]
    assert 'record.get("storage_migration")' in reader
    outside = store.replace(reader, "")
    for line in outside.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("#:"):
            continue
        assert '"storage_migration"' not in stripped or "storage_migrations" in stripped, stripped


def test_an_interrupted_run_state_carries_the_history():
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    block = source[source.index("def _record_interruption"):source.index("def _finish")]
    assert "storage_migrations" in block, (
        "an interrupted run does not record the history a resume will read")


def test_completion_and_run_state_both_carry_the_plural_history():
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    assert "storage_migrations=history" in source, "the run state does not carry the history"
    assert '"storage_migrations": list(state.get("storage_migrations") or [])' in source, (
        "the completion manifest does not carry the history")


def test_a_history_survives_being_reloaded_from_a_replaced_manifest(tmp_path):
    """An extension writes a NEW completion manifest. The history has to come back out of the
    sources that survive that replacement -- which is why the file itself carries it."""
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    manifest = tmp_path / "restart.json"
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        manifest.write_text(json.dumps({"storage_migrations": reporter.migration_history()}))
    finally:
        reporter.close()

    # The extension overwrites the manifest with one that knows nothing.
    manifest.write_text(json.dumps({"storage_migrations": []}))

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        recovered = storage.merge_migration_histories(
            reporter.migration_history(),
            storage.migration_history_of(json.loads(manifest.read_text())))
    finally:
        reporter.close()
    assert len(recovered) == 1
    assert recovered[0]["fields_added"] == [FIELD]


def test_a_corrupt_history_attribute_is_refused_not_guessed(tmp_path):
    path = _modern(tmp_path / "modern.nc", rows=1)
    with netCDF4.Dataset(str(path), "a") as d:
        d.setncattr(storage.MIGRATION_HISTORY_ATTRIBUTE, "{not json")
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        with pytest.raises(storage.StorageError):
            reporter.migration_history()
    finally:
        reporter.close()
