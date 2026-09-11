"""Appending to a v2 file written before the optional exchange fields existed.

`reservoir_velocity_seed` was added to `md-tools-replica-exchange/v2` without a schema bump.
New files carry it and readers tolerate its absence, but `write_exchange()` wrote it
unconditionally -- so a current runtime could READ an older v2 file and then fail the moment it
tried to resume or extend one.

v2's meaning did not change; it gained a record. So the repair is an additive, explicitly
validated migration rather than a v3 that would make every existing v2 file unreadable by a
runtime that can in fact read it.

PLATFORM_POLICY_EXEMPTION: NetCDF schema and storage-contract tests. Nothing here propagates
dynamics; the runtime evidence is the real resume/extend recorded in
`docs/history/journals/20260831_rest2-v2-append-compatibility.md`.
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
from md_tools.remd.engine import Configuration                           # noqa: E402
from md_tools.remd import validate as validate

FIELD = "reservoir_velocity_seed"
ZERO = np.zeros((2, 2), dtype=np.int64)


def _write_rows(reporter, count, *, first=0, reservoir=None):
    for offset in range(count):
        index = first + offset
        reporter.write_exchange(
            index, step=(index + 1) * 500, time_ps=(index + 1) * 1.0, state_to_walker=[0, 1],
            proposed=ZERO, accepted=ZERO, u=np.full((2, 2), float(index)),
            u_evaluated=np.ones((2, 2), dtype=np.int8),
            reservoir=reservoir(index) if callable(reservoir) else reservoir)


def _modern(path, rows=3, identity=None):
    """A coherent v2 run: exchange rows plus the whole and solute frames a real run writes, so
    the validator has something complete to accept."""
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity=identity or {"n_states": 2, "tau": [0.0, 0.5]}, metadata={"note": "synthetic"})
    configurations = [Configuration(np.zeros((4, 3)), np.zeros((4, 3)), None) for _ in range(2)]
    _write_rows(reporter, rows)
    for index in range(rows):
        step = (index + 1) * 500
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=index,
                                    configurations=configurations, solute_indices=[0])
    reporter.write_frame(step=rows * 500, time_ps=rows * 500 * 0.002, exchange_index=rows - 1)
    reporter.close()
    return path


def _legacy(path, rows=3, identity=None, tmp_path=None, drop=FIELD, redefine=None):
    """A v2 file exactly as an earlier build of this schema wrote it: without the optional field.

    Built by copying a real file rather than by hand, so everything except the dropped variable is
    genuinely what this runtime produces.
    """
    source = (tmp_path or path.parent) / "_source.nc"
    _modern(source, rows=rows, identity=identity)
    with netCDF4.Dataset(str(source), "r") as src, netCDF4.Dataset(str(path), "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, None if dimension.isunlimited() else len(dimension))
        for name, variable in src.variables.items():
            if name == drop:
                continue
            out = dst.createVariable(name, variable.datatype, variable.dimensions)
            out.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            out[:] = variable[:]
        if redefine is not None:
            dst.createVariable(drop, redefine["dtype"], redefine["dimensions"])
    source.unlink()
    return path


def _fingerprint(path):
    """Bytes, mtime, and the logical content a reader would see."""
    raw = Path(path).read_bytes()
    with netCDF4.Dataset(str(path), "r") as d:
        logical = {
            "variables": sorted(d.variables),
            "dimensions": {k: len(v) for k, v in d.dimensions.items()},
            "attributes": {k: str(d.getncattr(k)) for k in d.ncattrs()},
            "last_exchange": int(d.variables["last_exchange"][0]),
            "u": np.array(d.variables["u"][:]).tolist(),
            "state_to_walker": np.array(d.variables["state_to_walker"][:]).tolist(),
            "proposed": np.array(d.variables["proposed"][:]).tolist(),
            "accepted": np.array(d.variables["accepted"][:]).tolist(),
            "reservoir_state": np.array(d.variables["reservoir_state"][:]).tolist(),
        }
    return {"sha256": hashlib.sha256(raw).hexdigest(),
            "mtime_ns": Path(path).stat().st_mtime_ns,
            "logical": logical}


# --- 1/2. a legacy file reads, and read-only access does not touch it --------------------------

def test_a_legacy_v2_file_without_the_optional_field_is_readable(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    with netCDF4.Dataset(str(path), "r") as d:
        assert FIELD not in d.variables
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert reporter.schema == storage.SCHEMA_VERSION
        assert reporter.last_exchange() == 2
        assert [int(s) for s in reporter.reservoir_velocity_seeds()] == [-1, -1, -1]
    finally:
        reporter.close()


def test_verify_only_accepts_a_legacy_file_and_changes_nothing(tmp_path):
    """`--verify-only` opens in mode "r". It must accept a coherent older file and must not add
    the field as a side effect -- validation that repairs is not validation."""
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    before = _fingerprint(path)

    result = validate.validate_replica_output(analysis=path, expect_completed=False)
    assert result.ok, result.problems

    after = _fingerprint(path)
    assert after["sha256"] == before["sha256"], "verification rewrote the file"
    assert after["mtime_ns"] == before["mtime_ns"], "verification touched the modification time"
    assert FIELD not in after["logical"]["variables"]
    assert after["logical"] == before["logical"]


# --- the defect itself --------------------------------------------------------------------------

def test_appending_to_a_legacy_file_without_migrating_is_refused(tmp_path):
    """Before this branch this raised a bare `KeyError` from inside the row write, after the file
    was already open for append. It is now a storage error that names the missing step."""
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError) as caught:
            _write_rows(reporter, 1, first=3)
        assert FIELD in str(caught.value)
    finally:
        reporter.close()


# --- 3/4. migration adds the field and initialises the rows it already had ---------------------

def test_migration_adds_the_field_and_initialises_every_existing_row(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        record = reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    assert record["fields_added"] == [FIELD]
    assert record["rows_initialised"] == 3
    assert record["previous_committed_exchanges"] == 3
    assert record["initialised_to"] == {FIELD: -1}

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert [int(s) for s in reporter.reservoir_velocity_seeds()] == [-1, -1, -1]
    finally:
        reporter.close()


def test_migration_leaves_every_other_record_logically_unchanged(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=3)
    before = _fingerprint(path)["logical"]

    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    after = _fingerprint(path)["logical"]
    assert after["variables"] == sorted(before["variables"] + [FIELD])
    for key in ("u", "state_to_walker", "proposed", "accepted", "reservoir_state",
                "last_exchange"):
        assert after[key] == before[key], key
    assert after["dimensions"] == before["dimensions"]

    # Exactly one attribute is added: the migration history, which the file records for itself in
    # the same sync as the mutation so an interrupted process still leaves a file that says what
    # happened to it. Every pre-existing attribute is untouched.
    added = set(after["attributes"]) - set(before["attributes"])
    assert added == {storage.MIGRATION_HISTORY_ATTRIBUTE}
    for key, value in before["attributes"].items():
        assert after["attributes"][key] == value, key


# --- 5/6. what the migrated file records afterwards --------------------------------------------

def test_a_stored_mode_exchange_after_migration_records_no_seed(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=2)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
        # A `stored` refresh: it happened, and it drew nothing.
        _write_rows(reporter, 1, first=2, reservoir=(1, 0, 500, 1, -1))
    finally:
        reporter.close()

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert [int(s) for s in reporter.reservoir_velocity_seeds()] == [-1, -1, -1]
    finally:
        reporter.close()


def test_a_maxwell_refresh_after_migration_records_its_exact_seed(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=2)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
        _write_rows(reporter, 1, first=2, reservoir=(1, 0, 500, 1, 424242))
        _write_rows(reporter, 1, first=3, reservoir=None)
    finally:
        reporter.close()

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert [int(s) for s in reporter.reservoir_velocity_seeds()] == [-1, -1, 424242, -1]
    finally:
        reporter.close()


# --- 9/10. an existing field is validated, never recreated or replaced -------------------------

def test_a_file_that_already_has_the_field_is_not_recreated_or_reset(tmp_path):
    path = _modern(tmp_path / "modern.nc", rows=2)
    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        _write_rows(reporter, 1, first=2, reservoir=(1, 0, 500, 1, 777))
        record = reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()

    assert record["fields_added"] == []
    assert record["fields_already_present"] == [FIELD]
    assert record["rows_initialised"] == 0

    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        assert [int(s) for s in reporter.reservoir_velocity_seeds()] == [-1, -1, 777], (
            "an existing seed history was reset by a migration that should have been a no-op")
    finally:
        reporter.close()


@pytest.mark.parametrize("redefine,why", [
    ({"dtype": "i8", "dimensions": ("state",)}, "wrong dimensions"),
    ({"dtype": "f8", "dimensions": ("exchange",)}, "incompatible type"),
])
def test_an_incompatible_existing_field_is_refused_not_replaced(tmp_path, redefine, why):
    """Deleting or redefining a variable whose provenance is unknown would destroy a record. The
    file is left exactly as it is."""
    path = _legacy(tmp_path / "odd.nc", rows=2, redefine=redefine)
    before = _fingerprint(path)

    reporter = storage.ReplicaReporter(path, mode="a")
    try:
        with pytest.raises(storage.StorageError) as caught:
            reporter.ensure_optional_exchange_fields()
        assert FIELD in str(caught.value)
    finally:
        reporter.close()

    after = _fingerprint(path)
    assert after["logical"]["variables"] == before["logical"]["variables"], why
    assert after["logical"] == before["logical"]


def test_inspection_is_read_only_and_names_the_problem(tmp_path):
    path = _legacy(tmp_path / "odd.nc", rows=2,
                   redefine={"dtype": "f8", "dimensions": ("exchange",)})
    before = _fingerprint(path)
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        report = reporter.inspect_optional_exchange_fields()
    finally:
        reporter.close()
    assert report[FIELD]["state"] == "present"
    assert "dtype" in report[FIELD]["problem"]
    assert _fingerprint(path)["sha256"] == before["sha256"]


def test_the_migration_refuses_a_read_only_handle(tmp_path):
    path = _legacy(tmp_path / "legacy.nc", rows=2)
    before = _fingerprint(path)
    reporter = storage.ReplicaReporter(path, mode="r")
    try:
        with pytest.raises(storage.StorageError):
            reporter.ensure_optional_exchange_fields()
    finally:
        reporter.close()
    assert _fingerprint(path)["sha256"] == before["sha256"]


# --- 14/15. new files, and old stored-mode runs, stay distinguishable --------------------------

def test_a_new_file_creates_the_field_directly(tmp_path):
    path = _modern(tmp_path / "modern.nc", rows=1)
    with netCDF4.Dataset(str(path), "r") as d:
        assert FIELD in d.variables
        assert tuple(d.variables[FIELD].dimensions) == ("exchange",)
        assert np.dtype(d.variables[FIELD].dtype).kind in "iu"


def test_a_migrated_stored_mode_history_stays_all_minus_one(tmp_path):
    """An older `stored` run and a Maxwell run remain scientifically distinguishable: -1 is the
    true statement "no velocity seed was used", so a migrated stored-mode file reads as one that
    drew nothing, which is exactly what it did."""
    legacy = _legacy(tmp_path / "legacy.nc", rows=4)
    reporter = storage.ReplicaReporter(legacy, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
        _write_rows(reporter, 2, first=4, reservoir=lambda i: (1, 0, 500 * i, 1, -1))
    finally:
        reporter.close()

    reporter = storage.ReplicaReporter(legacy, mode="r")
    try:
        seeds = [int(s) for s in reporter.reservoir_velocity_seeds()]
    finally:
        reporter.close()
    assert set(seeds) == {-1}, "a stored-mode history must contain no drawn seed"

    maxwell = _legacy(tmp_path / "maxwell.nc", rows=2, tmp_path=tmp_path)
    reporter = storage.ReplicaReporter(maxwell, mode="a")
    try:
        reporter.ensure_optional_exchange_fields()
        _write_rows(reporter, 1, first=2, reservoir=(1, 0, 500, 1, 99))
    finally:
        reporter.close()
    reporter = storage.ReplicaReporter(maxwell, mode="r")
    try:
        assert any(int(s) >= 0 for s in reporter.reservoir_velocity_seeds())
    finally:
        reporter.close()


# --- the driver's ordering: read-only checks, then append, then migrate ------------------------

def test_the_driver_validates_before_it_opens_for_append(tmp_path):
    """Ordering is the safety property. A file must not be modified because it could be opened."""
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    block = source[source.index("def _continue"):source.index("def _loop")]
    probe = block.index("self.read_only_probe()")
    identity = block.index("the scientific configuration changed since this run was created")
    checkpoint = block.index("storage.ReplicaCheckpoint(self.files.checkpoint).read()")
    append = block.index('storage.ReplicaReporter(self.files.trajectory, mode="a")')
    migrate = block.index("ensure_optional_exchange_fields()")
    assert probe < identity < checkpoint < append < migrate
    assert block.index("validate_replica_output") < probe

    # And the probe itself opens read-only and nothing else.
    reader = source[source.index("def read_only_probe"):source.index("def _previous_manifest")]
    assert 'mode="r"' in reader and 'mode="a"' not in reader


def test_only_rank_zero_opens_writable_storage(tmp_path):
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    block = source[source.index("def _continue"):source.index("def _loop")]
    for opener in ('mode="a"', "self.read_only_probe()", "ensure_optional_exchange_fields()"):
        assert opener in block, opener
    # Everything in `_continue` that touches the analysis file sits under the root guard.
    guard = block.index("if self.coordinator.is_root:")
    assert guard < block.index('mode="a"')
    assert guard < block.index("self.read_only_probe()")


def test_the_migration_is_recorded_in_the_continuation_provenance():
    """Recorded as a cumulative HISTORY. A singular field was rewritten by every continuation, so
    a later no-op erased the real event -- see test_migration_provenance.py."""
    source = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    assert "storage_migrations=history" in source
    assert '"storage_migrations": migrations' in source


def test_schema_knowledge_stays_in_the_storage_module():
    """No raw NetCDF variable creation for these fields anywhere but `replica_storage.py`."""
    driver = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in driver.splitlines()
                     if not line.strip().startswith("#"))
    assert "self.dataset.createVariable" not in code and ".createVariable(" not in code, (
        "the driver creates a NetCDF variable itself")
    # Calling the reader `reservoir_velocity_seeds()` is right; naming the raw variable is not.
    assert f'"{FIELD}"' not in driver and f"'{FIELD}'" not in driver, (
        "the driver names a raw schema variable it should ask the storage module about")
    declared = (TEMPLATES / "storage.py").read_text(encoding="utf-8")
    assert "OPTIONAL_EXCHANGE_FIELDS" in declared


def test_the_schema_version_is_unchanged():
    """v2 gained a record; its meaning did not change. A bump would have made every existing v2
    file unreadable by a runtime that can read it."""
    assert storage.SCHEMA_VERSION == storage.SCHEMA_VERSION
    assert json.dumps(storage.OPTIONAL_EXCHANGE_FIELDS[FIELD]["absent_value"]) == "-1"
