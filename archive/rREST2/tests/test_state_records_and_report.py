"""rREST2 cases moved out of tests/test_state_records_and_report.py when rREST2 was archived (0.5.4).

Kept as the record of what the method was tested against; not collected, and not expected to
run against current md-tools. See archive/rREST2/README.md and the tag rREST2-final.
"""
import numpy as np

from md_tools.remd import statistics as statistics

TAUS = [0.0, 0.1, 0.3, 0.5]


def _statistics(n_states=4, *, accepted=None, proposed=None, reservoir_events=None):
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
        tau=TAUS[:n_states], reservoir_events=reservoir_events)

def test_a_reservoir_refresh_is_reported_separately_and_never_folded_into_the_pairs():
    """rREST2: a refresh replaces a configuration rather than swapping two of them."""
    events = np.full((10, 4), -1, dtype=np.int64)
    events[3] = [3, 4, 2000, 1]
    events[7] = [3, 9, 6000, 0]
    stats = _statistics(reservoir_events=events)
    plain = statistics.completion_report(_statistics())
    report = statistics.completion_report(stats)
    assert report["reservoir"]["attempts"] == 2 and report["reservoir"]["accepted"] == 1
    assert report["reservoir"]["states_refreshed"] == [3]
    assert report["overall"] == plain["overall"]
    assert report["by_neighbouring_pair"] == plain["by_neighbouring_pair"]
