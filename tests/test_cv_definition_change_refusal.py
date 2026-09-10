"""A changed collective-variable definition refuses continuation rather than appending to it.

WHY EVERY ONE OF THESE MATTERS

    A CV series is a column of numbers under a name. Change what the name means -- different
    atoms, a different order, a different unit, a different wrapping or sign convention, a
    different cadence -- and the appended rows are a DIFFERENT MEASUREMENT sharing a heading with
    the old ones. Nothing in the file marks where one ended and the other began, and every
    downstream analysis reads the column as one series.

    So each parametrisation below changes exactly one thing and requires the continuation to
    refuse. They are cheap, in-process checks against the validator, so the whole matrix runs on
    every commit rather than only in a slow lane.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from md_tools.cv import parse_cv_definition
from md_tools.remd.cv_states import COLUMNS, PHASE, CVContinuationError, validate_for_continuation

BASE = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
  - name: psi
    type: torsion
    atom_indices: [6, 8, 14, 16]
"""

TAUS = (0.0, 0.25, 0.5)
INTERVAL = 5


def _definition(text=BASE):
    return parse_cv_definition(text, particles=32)


def _sidecar(index, definition, **overrides):
    body = {
        "schema_version": 1,
        "definition_sha256": definition.digest,
        "interval_steps": INTERVAL,
        "state_index": index,
        "tau": TAUS[index],
        "units": "degrees",
        "wrapping": "[-180, 180)",
        "exchange_phase": PHASE,
        "collective_variables": [
            {"name": cv.name, "type": "torsion", "atom_indices": list(cv.indices)}
            for cv in definition.variables],
    }
    body.update(overrides)
    return body


def _write_series(directory: Path, definition, rows=4, **sidecar_overrides):
    directory.mkdir(parents=True, exist_ok=True)
    header = ",".join(list(COLUMNS) + list(definition.names))
    for index in range(len(TAUS)):
        body = [header]
        for row in range(rows):
            body.append(",".join([str(row * INTERVAL), f"{row * 0.01:.6f}", "-1",
                                  str(index), f"{TAUS[index]}", str(index), PHASE, ""]
                                 + ["0.000000"] * len(definition.variables)))
        (directory / f"cv_state{index}.csv").write_text("\n".join(body) + "\n", encoding="utf-8")
        (directory / f"cv_state{index}.json").write_text(
            json.dumps(_sidecar(index, definition, **sidecar_overrides)), encoding="utf-8")


def _validate(directory, definition, *, committed=4, interval=INTERVAL):
    return validate_for_continuation(directory, definition, taus=TAUS,
                                     interval_steps=interval, committed_rows=committed)


def test_an_unchanged_series_continues(tmp_path):
    """The control. Without it every refusal below could come from an unrelated fault."""
    definition = _definition()
    _write_series(tmp_path, definition)
    assert _validate(tmp_path, definition) == 4


def test_a_missing_committed_count_refuses_with_a_compatibility_message(tmp_path):
    """A legacy checkpoint cannot say which rows are durable, and must not guess."""
    definition = _definition()
    _write_series(tmp_path, definition)
    with pytest.raises(CVContinuationError, match="does not record"):
        _validate(tmp_path, definition, committed=None)


def test_a_series_shorter_than_the_committed_count_refuses(tmp_path):
    """Fewer rows than committed is data loss, not a resumable state."""
    definition = _definition()
    _write_series(tmp_path, definition, rows=2)
    with pytest.raises(CVContinuationError, match="lost"):
        _validate(tmp_path, definition, committed=4)


@pytest.mark.parametrize("missing", ["cv_state1.csv", "cv_state2.json"])
def test_a_missing_file_or_sidecar_refuses(tmp_path, missing):
    definition = _definition()
    _write_series(tmp_path, definition)
    (tmp_path / missing).unlink()
    with pytest.raises(CVContinuationError, match="missing"):
        _validate(tmp_path, definition)


CHANGED_INDICES = BASE.replace("[4, 6, 8, 14]", "[5, 6, 8, 14]")
REORDERED = """\
schema_version: 1
collective_variables:
  - name: psi
    type: torsion
    atom_indices: [6, 8, 14, 16]
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""


def test_changed_atom_indices_refuse(tmp_path):
    """The same names now name different atoms."""
    _write_series(tmp_path, _definition())
    with pytest.raises(CVContinuationError, match="resolved atoms|definition digest"):
        _validate(tmp_path, _definition(CHANGED_INDICES))


def test_a_reordered_definition_refuses(tmp_path):
    """Column order is part of the schema: reordering silently transposes two series."""
    _write_series(tmp_path, _definition())
    with pytest.raises(CVContinuationError, match="columns|resolved atoms|definition digest"):
        _validate(tmp_path, _definition(REORDERED))


def test_a_changed_interval_refuses(tmp_path):
    """A different cadence makes the appended rows a different grid under the same header."""
    definition = _definition()
    _write_series(tmp_path, definition)
    with pytest.raises(CVContinuationError, match="reporting interval"):
        _validate(tmp_path, definition, interval=10)


@pytest.mark.parametrize("field, value, fragment", [
    ("units", "radians", "units"),
    ("wrapping", "[0, 360)", "wrapping"),
    ("exchange_phase", "post-exchange", "exchange-boundary"),
    ("state_index", 9, "state index"),
    ("tau", 0.99, "tau"),
])
def test_a_changed_convention_in_the_sidecar_refuses(tmp_path, field, value, fragment):
    """Each of these silently redefines what the numbers in the column mean."""
    definition = _definition()
    _write_series(tmp_path, definition, **{field: value})
    with pytest.raises(CVContinuationError, match=fragment):
        _validate(tmp_path, definition)


def test_a_changed_definition_digest_refuses(tmp_path):
    """The digest catches a changed definition even when everything it resolved to matches."""
    definition = _definition()
    _write_series(tmp_path, definition, definition_sha256="0" * 64)
    with pytest.raises(CVContinuationError, match="definition digest"):
        _validate(tmp_path, definition)


def test_a_malformed_sidecar_refuses(tmp_path):
    definition = _definition()
    _write_series(tmp_path, definition)
    (tmp_path / "cv_state0.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CVContinuationError, match="not readable JSON"):
        _validate(tmp_path, definition)


def test_a_changed_column_set_refuses(tmp_path):
    """A CSV whose header no longer matches is a different schema, not a continuable series."""
    definition = _definition()
    _write_series(tmp_path, definition)
    path = tmp_path / "cv_state0.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0] + ",extra"] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(CVContinuationError, match="columns"):
        _validate(tmp_path, definition)
