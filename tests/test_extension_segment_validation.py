"""An extension segment shorter than the ORIGINAL request must still be extendable.

The defect (reported by hpREST2 on 0.6.4): a run created to make 50 000 exchanges, extended by 50 000
and then by 5 000, could not be extended a third time. `validate_extension_parent` judged the 5 000-row
segment against the original request ("fewer than the 50 000 attempts the run was created to make"),
because it passed no manifest and the validator then fell back to the scientific identity's count.

Passing the manifest is NOT the fix, and the test below says why: an extension's manifest carries
`schedule.number_of_exchanges` as the CHAIN's total (105 000 here), so the validator's manifest rule
("rows == budget") would then fail every extension. The number that describes one segment's file is
`extends.segment.exchanges`.

PLATFORM_POLICY_EXEMPTION: record validation on a tiny synthetic NetCDF; no dynamics is run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from md_tools.remd import storage, validate                                   # noqa: E402
from md_tools.remd.engine import Configuration                                # noqa: E402

ZERO = np.zeros((2, 2), dtype=np.int64)
ORIGINAL_REQUEST = 20


def _storage(path, rows):
    """A two-state ladder file holding `rows` exchange rows, whose identity asks for 20."""
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2, "tau": [0.0, 0.5], "number_of_exchanges": ORIGINAL_REQUEST,
                  "total_steps": ORIGINAL_REQUEST * 500}, metadata={})
    configurations = [Configuration(np.zeros((4, 3)), np.zeros((4, 3)), None) for _ in range(2)]
    for index in range(rows):
        step = (index + 1) * 500
        reporter.write_exchange(
            index, step=step, time_ps=step * 0.002, state_to_walker=[0, 1], proposed=ZERO,
            accepted=ZERO, u=np.full((2, 2), float(index)), u_evaluated=np.ones((2, 2), dtype=np.int8))
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=index,
                                    configurations=configurations, solute_indices=[0])
        reporter.write_frame(step=step, time_ps=step * 0.002, exchange_index=index)
    reporter.close()
    return path


def _budget_problems(result):
    return [p for p in result.problems if "exchange row" in p]


def test_segment_exchanges_reads_the_segment_not_the_chain():
    extension = {"schedule": {"number_of_exchanges": 105_000},
                 "extends": {"segment": {"exchanges": 5_000}, "chain": {"exchanges": 55_000}}}
    assert validate.segment_exchanges(extension) == 5_000
    assert validate.segment_exchanges({"schedule": {"number_of_exchanges": 50_000}}) is None   # an original run
    assert validate.segment_exchanges(None) is None
    assert validate.segment_exchanges({"extends": {"segment": {}}}) is None


def test_a_short_segment_is_refused_against_the_original_request_and_accepted_against_its_own_count(tmp_path):
    """The reported refusal, and its resolution, on the same file."""
    path = _storage(tmp_path / "ext.nc", rows=5)
    old = validate.validate_replica_output(analysis=str(path), expect_completed=True)
    assert any(f"fewer than the {ORIGINAL_REQUEST} attempts the run was created to make" in p
               for p in old.problems), old.problems
    own = validate.validate_replica_output(analysis=str(path), expect_completed=True,
                                           expected_exchanges=5)
    assert _budget_problems(own) == [], own.problems
    assert own.facts["segment_exchanges"] == 5


def test_a_segment_that_lost_rows_is_still_caught(tmp_path):
    """The relaxation is to the segment's OWN count, not away from accounting: a file with fewer rows
    than its manifest recorded is truncated, and says so."""
    path = _storage(tmp_path / "ext.nc", rows=4)
    result = validate.validate_replica_output(analysis=str(path), expect_completed=True,
                                              expected_exchanges=5)
    assert any("holds 4 exchange row(s) but this extension segment recorded 5" in p
               for p in result.problems), result.problems


def test_an_original_run_is_still_judged_against_its_request(tmp_path):
    path = _storage(tmp_path / "orig.nc", rows=5)
    result = validate.validate_replica_output(analysis=str(path), expect_completed=True,
                                              expected_exchanges=None)
    assert any("fewer than the 20 attempts" in p for p in result.problems)
    complete = validate.validate_replica_output(analysis=str(_storage(tmp_path / "done.nc", rows=20)),
                                                expect_completed=True)
    assert _budget_problems(complete) == []


def test_the_extension_parent_check_hands_the_validator_the_segments_own_count(tmp_path, monkeypatch):
    """The wiring: `validate_extension_parent` must pass the parent manifest's segment rows, which is
    what the first test needs to be reachable from `--extend`. A chain of 500 + 50 + 50 ns of a run
    created for 500 ns: the 50 ns parent holds 5 000 rows."""
    from md_tools.remd import driver

    manifest = {"schedule": {"number_of_exchanges": 105_000},
                "extends": {"segment": {"exchanges": 5_000}, "chain": {"exchanges": 55_000}},
                "run_status": "completed", "scientific_identity": {"number_of_exchanges": 50_000}}
    (tmp_path / "REST2.nc").write_bytes(b"")
    (tmp_path / "REST2_checkpoint.nc").write_bytes(b"")
    files = {"analysis": tmp_path / "REST2.nc", "checkpoint": tmp_path / "REST2_checkpoint.nc",
             "manifest": tmp_path / "restart_ext.json"}
    monkeypatch.setattr(driver.ReplicaRun, "parent_files", classmethod(lambda cls, p, name=None: (manifest, files)))
    seen = {}

    class Stop(Exception):
        pass

    def capture(**kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(validate, "validate_replica_output", capture)
    with pytest.raises(Stop):
        driver.ReplicaRun.validate_extension_parent(tmp_path)
    assert seen["expected_exchanges"] == 5_000 and seen["expect_completed"] is True
    assert "manifest" not in seen          # the chain's schedule count must not become this file's budget
    manifest.pop("extends")
    with pytest.raises(Stop):
        driver.ReplicaRun.validate_extension_parent(tmp_path)
    assert seen["expected_exchanges"] is None      # an original run: judged against its request
