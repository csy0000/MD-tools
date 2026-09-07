"""Test A: the AIS global segment is what THIS invocation did, not what history says.

THE DEFECT

    The global segment was built by summing the `segment` stored in every completed path's
    manifest. That segment describes the invocation which COMPLETED that path. It is not
    necessarily the invocation assembling the table, and it is not when a campaign resumes after
    some paths finished, when paths complete across several interruptions, when a completed
    campaign is entered again, or when the worker count changes between interruption and resume
    -- which is to say, in exactly the cases resume exists for.

    The error is in the flattering direction: it reports more work than was done. A reader sizing
    an allocation from it, or comparing an interrupted campaign against an uninterrupted one,
    gets a number that grows every time the campaign is re-entered while nothing is computed.

WHY A UNIT TEST FIRST

    The end-to-end cases (paths completed across invocations, no-op re-entry, a two-to-four rank
    CUDA resume) each take minutes and prove the behaviour through a great deal of machinery. This
    one is arithmetic: dictionaries in, one record out. It states the rule exactly, distinguishes
    the historical segment from the current contribution by construction -- every path here has
    values for both, and they differ -- and it fails against the old implementation, which is
    asserted directly in `test_summing_stored_segments_is_what_this_test_refuses`.

Two torsions throughout, so `cv_evaluations == 2 * cv_observations` and a reporter-call counter
cannot satisfy the arithmetic. Every expected number below is written out from the schedule
rather than computed by the code under test.
"""
from __future__ import annotations

import pytest

from md_tools.ais.run import aggregate_cv_cost
from md_tools.cv.cost import CVCost, cost_record, parse_aggregate_record

N_CV = 2
ROWS = 5

#: Three paths, each with a DISTINCT cumulative and a DISTINCT stored historical segment. The
#: historical segments are deliberately non-zero for every path, including the one that did no
#: work now: if the implementation reaches for them, the totals below cannot come out right.
#:
#:   path 0  finished in an EARLIER invocation      history 5 obs   now 0 obs
#:   path 1  resumed here, part done earlier        history 3 obs   now 3 obs
#:   path 2  ran fresh here                         history 5 obs   now 5 obs
HISTORY = {
    0: CVCost(observations=5, evaluations=10, wall_seconds=0.005),
    1: CVCost(observations=3, evaluations=6, wall_seconds=0.003),
    2: CVCost(observations=5, evaluations=10, wall_seconds=0.005),
}
CUMULATIVE = {
    0: CVCost(observations=5, evaluations=10, wall_seconds=0.005),
    1: CVCost(observations=5, evaluations=10, wall_seconds=0.004),
    2: CVCost(observations=5, evaluations=10, wall_seconds=0.005),
}
NOW = {
    0: CVCost(observations=0, evaluations=0, wall_seconds=0.0),
    1: CVCost(observations=3, evaluations=6, wall_seconds=0.003),
    2: CVCost(observations=5, evaluations=10, wall_seconds=0.005),
}
DISPOSITION = {
    0: "already_complete",
    1: "resumed_and_completed",
    2: "fresh_and_completed",
}

#: Written out, not computed. Cumulative is 3 paths x 5 rows; the current segment is 0 + 3 + 5.
EXPECTED_CUMULATIVE_OBSERVATIONS = 15
EXPECTED_CUMULATIVE_EVALUATIONS = 30
EXPECTED_SEGMENT_OBSERVATIONS = 8
EXPECTED_SEGMENT_EVALUATIONS = 16
#: What the OLD implementation would have reported: 5 + 3 + 5 historical observations.
OLD_WRONG_SEGMENT_OBSERVATIONS = 13


def _manifests(order=(0, 1, 2)):
    """What each path's `completed.json` stores: its own cumulative, and its own history."""
    return {index: {**cost_record(HISTORY[index], CUMULATIVE[index], rows=ROWS)}
            for index in order}


def _contributions(order=(0, 1, 2)):
    """What each rank reports about this invocation, gathered through the MPI authority."""
    return {index: {"disposition": DISPOSITION[index],
                    "cost": cost_record(NOW[index], CUMULATIVE[index])}
            for index in order}


def test_the_global_cumulative_is_the_sum_over_completed_paths():
    record = aggregate_cv_cost(_manifests(), contributions=_contributions())
    assert record["cumulative"]["cv_observations"] == EXPECTED_CUMULATIVE_OBSERVATIONS
    assert record["cumulative"]["cv_evaluations"] == EXPECTED_CUMULATIVE_EVALUATIONS


def test_the_global_segment_is_this_invocation_not_the_stored_history():
    """THE assertion. 8, not 13."""
    record = aggregate_cv_cost(_manifests(), contributions=_contributions())
    assert record["segment"]["cv_observations"] == EXPECTED_SEGMENT_OBSERVATIONS
    assert record["segment"]["cv_evaluations"] == EXPECTED_SEGMENT_EVALUATIONS
    assert record["segment"]["cv_observations"] != OLD_WRONG_SEGMENT_OBSERVATIONS, (
        "the aggregate summed the stored historical segments")


def test_a_path_already_complete_contributes_exactly_zero():
    record = aggregate_cv_cost(_manifests(), contributions=_contributions())
    first = next(e for e in record["per_path"] if e["path_index"] == 0)
    assert first["disposition"] == "already_complete"
    assert first["segment"]["cv_observations"] == 0
    assert first["segment"]["cv_evaluations"] == 0
    assert first["segment"]["wall_seconds"] == 0.0
    # And its cumulative is still counted: the campaign contains it.
    assert first["cumulative"]["cv_observations"] == ROWS


def test_every_path_retains_its_disposition_and_current_segment():
    record = aggregate_cv_cost(_manifests(), contributions=_contributions())
    by_index = {entry["path_index"]: entry for entry in record["per_path"]}
    assert set(by_index) == {0, 1, 2}
    for index in (0, 1, 2):
        assert by_index[index]["disposition"] == DISPOSITION[index]
        assert by_index[index]["segment"]["cv_observations"] == NOW[index].observations
        assert by_index[index]["cumulative"]["cv_observations"] == CUMULATIVE[index].observations


def test_input_ordering_changes_neither_output_ordering_nor_totals():
    """The aggregate is a property of the campaign, not of the order ranks happened to report."""
    forward = aggregate_cv_cost(_manifests((0, 1, 2)), contributions=_contributions((0, 1, 2)))
    reverse = aggregate_cv_cost(_manifests((2, 1, 0)), contributions=_contributions((2, 0, 1)))
    assert forward == reverse
    assert [e["path_index"] for e in forward["per_path"]] == [0, 1, 2]


def test_a_missing_contribution_is_refused():
    """Silently reading it as zero would under-report the work this invocation actually did."""
    contributions = _contributions()
    contributions.pop(1)
    with pytest.raises(SystemExit) as refusal:
        aggregate_cv_cost(_manifests(), contributions=contributions)
    assert "path 1" in str(refusal.value)


def test_an_unexpected_contribution_is_refused():
    """Work credited to a path the run cannot show finished."""
    contributions = _contributions()
    contributions[7] = {"disposition": "fresh_and_completed",
                        "cost": cost_record(NOW[2], CUMULATIVE[2])}
    with pytest.raises(SystemExit) as refusal:
        aggregate_cv_cost(_manifests(), contributions=contributions)
    assert "[7]" in str(refusal.value)


def test_an_unknown_disposition_is_refused():
    contributions = _contributions()
    contributions[1]["disposition"] = "probably_fine"
    with pytest.raises(SystemExit) as refusal:
        aggregate_cv_cost(_manifests(), contributions=contributions)
    assert "probably_fine" in str(refusal.value)


def test_a_malformed_contribution_cost_is_refused_before_anything_is_summed():
    contributions = _contributions()
    contributions[2]["cost"]["segment"]["cv_observations"] = 5.0     # a float, not an integer
    with pytest.raises(Exception) as refusal:
        aggregate_cv_cost(_manifests(), contributions=contributions)
    assert "cv_observations" in str(refusal.value)


def test_the_aggregate_validates_against_its_own_per_path_entries():
    """The record it writes must satisfy the strict aggregate parser it will later be read by."""
    record = aggregate_cv_cost(_manifests(), contributions=_contributions(),
                               invocation_id="0123456789abcdef")
    parsed = parse_aggregate_record(record, where="unit", entries_key="per_path",
                                    identity_key="path_index", expected_identities=[0, 1, 2])
    assert parsed.segment.observations == EXPECTED_SEGMENT_OBSERVATIONS
    assert record["aggregation"] == "sum over completed paths"
    assert record["invocation_id"] == "0123456789abcdef"


def test_summing_stored_segments_is_what_this_test_refuses():
    """The old implementation, restored here as arithmetic, and shown to disagree.

    Rather than reverting the source to prove the point, the discarded rule is stated directly:
    sum each manifest's stored segment. It gives 13 where the truth is 8, and it does so with
    every input in this file unchanged -- which is what makes the difference a property of the
    RULE rather than of the fixture.
    """
    old_rule = CVCost()
    for index in (0, 1, 2):
        old_rule = old_rule.plus(HISTORY[index])
    assert old_rule.observations == OLD_WRONG_SEGMENT_OBSERVATIONS

    record = aggregate_cv_cost(_manifests(), contributions=_contributions())
    assert record["segment"]["cv_observations"] != old_rule.observations
    assert record["segment"]["cv_observations"] == EXPECTED_SEGMENT_OBSERVATIONS


def test_without_contributions_no_segment_is_invented():
    """A caller that supplies no coordination result gets zero, never the stored history."""
    record = aggregate_cv_cost(_manifests(), contributions=None)
    assert record["segment"]["cv_observations"] == 0
    assert record["cumulative"]["cv_observations"] == EXPECTED_CUMULATIVE_OBSERVATIONS
    assert all(e["disposition"] == "unknown" for e in record["per_path"])


def test_no_completed_paths_produces_no_record():
    assert aggregate_cv_cost({}, contributions={}) is None
