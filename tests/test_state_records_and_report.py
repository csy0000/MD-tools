"""The resolved per-state records (section 4) and the completion report (section 8).

Two contracts that are easy to satisfy loosely and easy to get subtly wrong:

  * a state record must carry the index, the file that state writes, the exact tau, and an
    effective temperature that is derived and labelled as reporting-only;
  * the completion report must contain neighbouring pairs and an overall figure, and must NOT
    have grown a convergence diagnostic.

PLATFORM_POLICY_EXEMPTION: pure bookkeeping over recorded numbers. No dynamics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"

from md_tools.remd import statistics as statistics
from md_tools.remd.protocol import REST2Protocol                         # noqa: E402

TAUS = [0.0, 0.1, 0.3, 0.5]


def _protocol(taus=TAUS, temperature_k=300.0):
    return REST2Protocol(tau=list(taus), temperature_k=temperature_k, timestep_fs=2.0,
                         exchange_interval_ps=2.0, number_of_exchanges=4)


# --- section 4: the resolved per-state records ----------------------------------------------------

def test_every_state_records_index_trajectory_tau_and_effective_temperature():
    records = _protocol().state_records()
    assert [r["index"] for r in records] == [0, 1, 2, 3]
    assert [r["trajectory"] for r in records] == ["whole_state0_prod1.nc", "whole_state1_prod1.nc", "whole_state2_prod1.nc", "whole_state3_prod1.nc"]
    assert [r["tau"] for r in records] == TAUS
    assert set(records[0]) == {"index", "trajectory", "tau", "effective_temperature_k"}


def test_the_effective_temperature_is_the_documented_formula():
    """T_eff = T_physical / (1 - tau)^2 -- the solute-solute scale, not the solute-environment one.

    Using (1 - tau) here would quietly report a cooler ladder than the one being run, and the
    number is exactly the kind that gets copied into a methods section unchecked.
    """
    for physical in (275.0, 300.0, 310.0):
        for record in _protocol(temperature_k=physical).state_records():
            assert record["effective_temperature_k"] == pytest.approx(
                physical / (1.0 - record["tau"]) ** 2)


def test_the_coldest_state_is_reported_at_the_physical_temperature():
    assert _protocol().state_records()[0]["effective_temperature_k"] == pytest.approx(300.0)


def test_tau_is_recorded_exactly_and_not_rounded_to_the_printed_form():
    tau = [0.0, 1.0 / 3.0, 0.5]
    records = _protocol(taus=tau).state_records()
    assert [r["tau"] for r in records] == tau


def test_the_trajectory_name_follows_the_index_and_never_the_tau():
    """Two ladders with different tau at the same index name the same file.

    Section 4 forbids inferring tau from the filename; the guarantee that makes that safe is that
    the name carries the index alone.
    """
    a = _protocol(taus=[0.0, 0.2, 0.4, 0.6]).state_records()
    b = _protocol(taus=[0.0, 0.05, 0.1, 0.15]).state_records()
    assert [r["trajectory"] for r in a] == [r["trajectory"] for r in b]
    assert [r["tau"] for r in a] != [r["tau"] for r in b]


def test_the_state_records_stay_out_of_the_scientific_identity():
    """`describe()` feeds the identity a continuation compares key by key.

    `effective_temperature_k` is derived and reporting-only. Putting it in the identity would mean
    every file written before the field existed reports a difference and refuses to continue --
    a reporting change breaking continuations, which is exactly backwards.
    """
    assert "states" not in _protocol().describe()


# --- section 8: the completion report -------------------------------------------------------------

def _statistics(n_states=4, *, accepted=None, proposed=None):
    """Lifetime statistics from an explicit neighbour-only proposal history."""
    n_exchanges = 10
    a = np.zeros((n_exchanges, n_states, n_states), dtype=np.int64)
    p = np.zeros((n_exchanges, n_states, n_states), dtype=np.int64)
    for row in range(n_exchanges):
        for i in range(n_states - 1):
            j = i + 1
            p[row, i, j] = p[row, j, i] = 1
            if (row + i) % 2 == 0:
                a[row, i, j] = a[row, j, i] = 1
    return statistics.lifetime_statistics(
        accepted if accepted is not None else a, proposed if proposed is not None else p,
        tau=TAUS[:n_states])


def test_the_report_lists_every_neighbouring_pair_once():
    report = statistics.completion_report(_statistics())
    assert [p["state_pair"] for p in report["by_neighbouring_pair"]] == [[0, 1], [1, 2], [2, 3]]


def test_no_non_neighbouring_pair_appears():
    report = statistics.completion_report(_statistics())
    for pair in report["by_neighbouring_pair"]:
        i, j = pair["state_pair"]
        assert abs(i - j) == 1


def test_the_overall_figure_is_the_sum_of_the_pairs_shown():
    report = statistics.completion_report(_statistics())
    pairs = report["by_neighbouring_pair"]
    assert report["overall"]["accepted"] == sum(p["accepted"] for p in pairs)
    assert report["overall"]["proposed"] == sum(p["proposed"] for p in pairs)
    assert report["overall"]["acceptance"] == pytest.approx(
        report["overall"]["accepted"] / report["overall"]["proposed"])


def test_the_overall_acceptance_is_none_rather_than_zero_when_nothing_was_proposed():
    """An unproposed ladder has no acceptance rate. Reporting 0.000 would read as a failed run."""
    zeros = np.zeros((4, 3, 3), dtype=np.int64)
    report = statistics.completion_report(
        statistics.lifetime_statistics(zeros, zeros, tau=TAUS[:3]))
    assert report["overall"]["proposed"] == 0
    assert report["overall"]["acceptance"] is None


def test_the_report_carries_no_convergence_diagnostic():
    """Section 8: round trips, transition matrices, first-passage times and convergence
    diagnostics belong downstream and must not be added here."""
    report = statistics.completion_report(_statistics())
    text = repr(report).lower()
    for banned in ("round_trip", "round trip", "transition_matrix", "first_passage",
                   "converged", "convergence"):
        assert banned not in text
    assert set(report) <= {"basis", "by_neighbouring_pair", "overall", "reservoir"}


def test_the_basis_states_the_committed_exchange_range():
    report = statistics.completion_report(_statistics())
    assert "0-9" in report["basis"] and "committed" in report["basis"]


def test_a_run_without_a_reservoir_reports_no_reservoir_block():
    """rREST2's refresh block is archived (0.5.4); no ladder report carries one."""
    assert "reservoir" not in statistics.completion_report(_statistics())
