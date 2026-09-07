"""The one strict parser for stored collective-variable cost records.

WHY STRICT, AND WHY IN ONE PLACE

    A stored counter is external input. The readers used to call `int()` and `float()` on whatever
    they found, which destroys the evidence of a violation and then reports the result as
    authoritative: `int(2.7)` is 2, `int("5")` is 5, and `int(True)` is 1. Three records that
    break the schema became three that satisfy it, before anything could refuse them.

    These records are read in eight places -- cMD checkpoint prefixes and logs, the ladder's
    per-state prefixes and aggregate, its completion and extension validation, AIS checkpoint
    prefixes, per-path completion, global aggregation and completed-path verification, and the
    provenance consumers downstream. Eight readers is eight policies, and the loosest one decides
    what the system actually accepts. Hence one parser, and this file is its contract.

WHAT A MUTATION TEST HAS TO DO

    Every case below starts from ONE valid two-torsion record and changes exactly one thing, so a
    refusal cannot come from an unrelated fault in a hand-built dictionary. The untouched record
    is asserted to pass first, in `test_the_untouched_record_is_accepted` -- without it every
    rejection here would also be satisfied by a parser that refuses everything.

    Each case also asserts the offending FIELD is named. "Invalid cost record" sends a reader to
    diff two files by eye; "cumulative.cv_evaluations must be an integer, got float 13.0" does not.
"""
from __future__ import annotations

import math

import pytest

from md_tools.cv.cost import (COST_SCHEMA_VERSION, CVCost, CVCostError, cost_record,
                              migrate_schema_1, parse_aggregate_record, parse_cost_record)

#: Two torsions, so `cv_evaluations == 2 * cv_observations` and a reporter-call counter cannot
#: satisfy the arithmetic. Derived here, from the schedule, not from anything under test.
N_CV = 2
ROWS = 13
SEGMENT = CVCost(observations=4, evaluations=4 * N_CV, wall_seconds=0.002)
CUMULATIVE = CVCost(observations=ROWS, evaluations=ROWS * N_CV, wall_seconds=0.005)


def _valid() -> dict:
    return cost_record(SEGMENT, CUMULATIVE, rows=ROWS)


def _parse(record):
    return parse_cost_record(record, where="unit", rows=ROWS, n_cv=N_CV)


def test_the_untouched_record_is_accepted():
    """The control. Every rejection below is also satisfied by a parser that refuses everything."""
    parsed = _parse(_valid())
    assert parsed.cumulative.observations == ROWS
    assert parsed.cumulative.evaluations == ROWS * N_CV
    assert parsed.rows == ROWS


def _mutate(**changes) -> dict:
    record = _valid()
    record.update(changes)
    return record


def _scope(name, **changes) -> dict:
    record = _valid()
    scope = dict(record[name])
    scope.update(changes)
    record[name] = scope
    return record


def _drop(mapping, key):
    without = dict(mapping)
    without.pop(key)
    return without


#: (id, record, fragment that must appear in the refusal). One mutation each.
CASES = [
    ("missing-schema-version", _drop(_valid(), "schema_version"), "schema_version"),
    ("schema-version-string", _mutate(schema_version="2"), "schema_version"),
    ("schema-version-float", _mutate(schema_version=2.0), "schema_version"),
    ("schema-version-bool", _mutate(schema_version=True), "schema_version"),
    ("schema-version-unsupported", _mutate(schema_version=7), "version 7"),

    ("missing-segment", _drop(_valid(), "segment"), "segment"),
    ("missing-cumulative", _drop(_valid(), "cumulative"), "cumulative"),
    ("segment-not-a-mapping", _mutate(segment=[1, 2, 3]), "segment"),

    ("segment-missing-observations",
     {**_valid(), "segment": _drop(_valid()["segment"], "cv_observations")}, "cv_observations"),
    ("segment-missing-evaluations",
     {**_valid(), "segment": _drop(_valid()["segment"], "cv_evaluations")}, "cv_evaluations"),
    ("segment-missing-seconds",
     {**_valid(), "segment": _drop(_valid()["segment"], "wall_seconds")}, "wall_seconds"),
    ("cumulative-missing-observations",
     {**_valid(), "cumulative": _drop(_valid()["cumulative"], "cv_observations")},
     "cv_observations"),

    ("scope-carries-an-extra-field", _scope("segment", cv_seconds=1.0), "unexpected"),

    ("observations-bool", _scope("cumulative", cv_observations=True), "cv_observations"),
    ("observations-string", _scope("cumulative", cv_observations="13"), "cv_observations"),
    ("observations-float-whole", _scope("cumulative", cv_observations=13.0), "cv_observations"),
    ("observations-float-fractional", _scope("cumulative", cv_observations=13.7),
     "cv_observations"),
    ("observations-negative", _scope("segment", cv_observations=-1), "cv_observations"),
    ("evaluations-bool", _scope("segment", cv_evaluations=True), "cv_evaluations"),
    ("evaluations-string", _scope("segment", cv_evaluations="8"), "cv_evaluations"),
    ("evaluations-float", _scope("segment", cv_evaluations=8.0), "cv_evaluations"),
    ("evaluations-negative", _scope("cumulative", cv_evaluations=-2), "cv_evaluations"),

    ("seconds-bool", _scope("segment", wall_seconds=True), "wall_seconds"),
    ("seconds-string", _scope("segment", wall_seconds="0.002"), "wall_seconds"),
    ("seconds-nan", _scope("segment", wall_seconds=float("nan")), "finite"),
    ("seconds-positive-infinity", _scope("segment", wall_seconds=float("inf")), "finite"),
    ("seconds-negative-infinity", _scope("segment", wall_seconds=float("-inf")), "finite"),
    ("seconds-negative", _scope("segment", wall_seconds=-0.5), "wall_seconds"),

    ("segment-observations-exceed-cumulative",
     _scope("segment", cv_observations=ROWS + 1), "exceeds cumulative"),
    ("segment-evaluations-exceed-cumulative",
     _scope("segment", cv_evaluations=ROWS * N_CV + 2), "exceeds cumulative"),
    ("segment-seconds-exceed-cumulative",
     _scope("segment", wall_seconds=1.0), "exceeds cumulative"),

    ("rows-bool", _mutate(cv_rows=True), "cv_rows"),
    ("rows-string", _mutate(cv_rows="13"), "cv_rows"),
    ("rows-float", _mutate(cv_rows=13.0), "cv_rows"),
    ("rows-negative", _mutate(cv_rows=-3), "cv_rows"),
    ("rows-inconsistent-with-verified", _mutate(cv_rows=ROWS + 1), "cv_rows"),

    ("cumulative-observations-inconsistent-with-rows",
     _scope("cumulative", cv_observations=ROWS - 1), "verified series holds"),
    ("cumulative-evaluations-not-rows-times-n-cv",
     _scope("cumulative", cv_evaluations=ROWS), "collective variable"),

    ("unexpected-top-level-field", _mutate(cv_seconds=1.0), "unexpected"),
]


@pytest.mark.parametrize("record, fragment",
                         [(record, fragment) for _id, record, fragment in CASES],
                         ids=[case_id for case_id, _r, _f in CASES])
def test_every_mutation_is_refused_with_the_field_named(record, fragment):
    with pytest.raises(CVCostError) as refusal:
        _parse(record)
    assert fragment in str(refusal.value), (
        f"the refusal does not name the offending field: {refusal.value}")


def test_a_fractional_counter_is_not_silently_truncated():
    """THE defect this parser exists for, stated on its own.

    `int(2.7)` is 2. A reader that coerced first turned a record describing 2.7 observations --
    which is not a thing -- into one describing 2, and reported it as authoritative.
    """
    with pytest.raises(CVCostError) as refusal:
        _parse(_scope("segment", cv_observations=2.7))
    assert "2.7" in str(refusal.value), refusal.value
    assert "integer" in str(refusal.value)


def test_a_numeric_string_is_not_a_counter():
    with pytest.raises(CVCostError) as refusal:
        _parse(_scope("segment", cv_observations="4"))
    assert "str" in str(refusal.value)


def test_true_is_not_one():
    """`isinstance(True, int)` is True in Python, so a bool passes a naive check and counts as 1."""
    with pytest.raises(CVCostError):
        _parse(_scope("segment", cv_observations=True))


def test_a_valid_record_passes_without_verified_facts_too():
    """`rows`/`n_cv` are optional: a reader that has not counted the file still validates shape."""
    parsed = parse_cost_record(_valid(), where="unit")
    assert parsed.segment.observations == SEGMENT.observations


def test_rows_are_required_when_the_caller_says_so():
    without = _drop(_valid(), "cv_rows")
    parse_cost_record(without, where="unit")          # fine when not required
    with pytest.raises(CVCostError) as refusal:
        parse_cost_record(without, where="unit", require_rows=True)
    assert "cv_rows" in str(refusal.value)


# --- schema version 1: an explicit, tested, documented policy ---------------------------------

def _legacy() -> dict:
    """What version 1 stored: one counter, which counted reporter CALLS, and a duration."""
    return {"cv_evaluations": ROWS, "cv_rows": ROWS, "cv_seconds": 0.004}


def test_a_schema_1_record_is_refused_with_an_actionable_message():
    """Refused, not guessed at. Its one counter cannot yield a scalar total without N_cv."""
    with pytest.raises(CVCostError) as refusal:
        parse_cost_record(_legacy(), where="an old checkpoint")
    message = str(refusal.value)
    assert "an old checkpoint" in message
    assert "--overwrite" in message, "the refusal must say what to do about it"


def test_an_explicit_schema_1_version_is_refused_the_same_way():
    with pytest.raises(CVCostError) as refusal:
        parse_cost_record({**_valid(), "schema_version": 1}, where="an old checkpoint")
    assert "--overwrite" in str(refusal.value)


def test_migration_is_available_only_where_the_facts_are_supplied():
    """The other half of the policy: convert exactly, and say that it was converted."""
    migrated = migrate_schema_1(_legacy(), rows=ROWS, n_cv=N_CV, where="an old checkpoint")
    parsed = parse_cost_record(migrated, where="migrated", rows=ROWS, n_cv=N_CV)
    assert parsed.cumulative.observations == ROWS
    assert parsed.cumulative.evaluations == ROWS * N_CV
    assert migrated["aggregation"] == "migrated from schema 1", (
        "a reconstructed total must be distinguishable from a measured one")


def test_migration_refuses_facts_it_cannot_trust():
    for rows, n_cv in ((ROWS, "2"), (13.0, N_CV), (-1, N_CV), (ROWS, True)):
        with pytest.raises(CVCostError):
            migrate_schema_1(_legacy(), rows=rows, n_cv=n_cv, where="unit")


# --- aggregates ---------------------------------------------------------------------------------

def _entry(identity_key, identity, segment, cumulative):
    record = cost_record(segment, cumulative)
    record[identity_key] = identity
    return record


def _aggregate(identity_key="path_index", entries_key="per_path", *, segments, cumulatives):
    entries = [_entry(identity_key, index, seg, cum)
               for index, (seg, cum) in enumerate(zip(segments, cumulatives))]
    total_segment = CVCost()
    total_cumulative = CVCost()
    for seg, cum in zip(segments, cumulatives):
        total_segment = total_segment.plus(seg)
        total_cumulative = total_cumulative.plus(cum)
    record = cost_record(total_segment, total_cumulative)
    record["aggregation"] = "sum over completed paths"
    record[entries_key] = entries
    return record


VALID_SEGMENTS = [CVCost(0, 0, 0.0), CVCost(5, 10, 0.002), CVCost(5, 10, 0.003)]
VALID_CUMULATIVES = [CVCost(5, 10, 0.004), CVCost(5, 10, 0.002), CVCost(5, 10, 0.003)]


def test_a_consistent_aggregate_is_accepted():
    record = _aggregate(segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    parsed = parse_aggregate_record(record, where="unit", entries_key="per_path",
                                    identity_key="path_index", expected_identities=[0, 1, 2])
    assert parsed.cumulative.observations == 15
    assert parsed.segment.observations == 10, "the already-complete path contributes nothing"


def test_a_duplicated_identity_is_refused():
    record = _aggregate(segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    record["per_path"][2]["path_index"] = 1
    with pytest.raises(CVCostError) as refusal:
        parse_aggregate_record(record, where="unit", entries_key="per_path",
                               identity_key="path_index")
    assert "more than once" in str(refusal.value)


def test_a_missing_identity_is_refused():
    record = _aggregate(segments=VALID_SEGMENTS[:2], cumulatives=VALID_CUMULATIVES[:2])
    with pytest.raises(CVCostError) as refusal:
        parse_aggregate_record(record, where="unit", entries_key="per_path",
                               identity_key="path_index", expected_identities=[0, 1, 2])
    assert "missing [2]" in str(refusal.value)


def test_an_unexpected_identity_is_refused():
    record = _aggregate(segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    with pytest.raises(CVCostError) as refusal:
        parse_aggregate_record(record, where="unit", entries_key="per_path",
                               identity_key="path_index", expected_identities=[0, 1])
    assert "unexpected [2]" in str(refusal.value)


def test_a_total_that_does_not_match_its_entries_is_refused():
    record = _aggregate(segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    record["cumulative"]["cv_observations"] += 1
    with pytest.raises(CVCostError) as refusal:
        parse_aggregate_record(record, where="unit", entries_key="per_path",
                               identity_key="path_index")
    assert "sum to" in str(refusal.value)


def test_a_segment_total_that_reuses_historical_segments_is_refused():
    """The accounting defect, caught by the parser rather than only by the AIS tests.

    Summing every path's STORED segment gives 10 here only because the already-complete path's
    stored segment happens to be 5; an aggregate claiming that as this invocation's work is
    inconsistent with the per-path contributions beside it.
    """
    record = _aggregate(segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    record["segment"]["cv_observations"] = 15      # as if the historical segments were summed
    with pytest.raises(CVCostError) as refusal:
        parse_aggregate_record(record, where="unit", entries_key="per_path",
                               identity_key="path_index")
    assert "segment.cv_observations" in str(refusal.value)


def test_state_aggregates_use_the_same_parser():
    record = _aggregate(identity_key="state_index", entries_key="per_state",
                        segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    record["aggregation"] = "sum over thermodynamic states"
    parsed = parse_aggregate_record(record, where="unit", entries_key="per_state",
                                    identity_key="state_index", expected_identities=[0, 1, 2])
    assert parsed.cumulative.evaluations == 30


def test_seconds_are_not_compared_more_tightly_than_they_are_stored():
    """Stored durations are rounded to microseconds, so a re-summation may differ in the last
    place. Counters are exact; seconds carry the rounding they were written with."""
    record = _aggregate(segments=VALID_SEGMENTS, cumulatives=VALID_CUMULATIVES)
    record["cumulative"]["wall_seconds"] += 5e-7
    parse_aggregate_record(record, where="unit", entries_key="per_path",
                           identity_key="path_index")
    record["cumulative"]["wall_seconds"] += 1.0
    with pytest.raises(CVCostError):
        parse_aggregate_record(record, where="unit", entries_key="per_path",
                               identity_key="path_index")


def test_the_schema_version_constant_is_what_the_writer_writes():
    assert cost_record(SEGMENT, CUMULATIVE)["schema_version"] == COST_SCHEMA_VERSION
    assert not isinstance(COST_SCHEMA_VERSION, bool) and isinstance(COST_SCHEMA_VERSION, int)


def test_non_finite_seconds_cannot_be_written_either():
    """The writer refuses what the reader refuses; one policy, both directions."""
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(CVCostError):
            CVCost(observations=1, evaluations=2, wall_seconds=bad)
    assert math.isfinite(CVCost(1, 2, 0.5).wall_seconds)
