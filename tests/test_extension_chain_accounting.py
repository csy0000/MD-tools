"""A chain total must count the CHAIN, and a chain is not two segments long.

Every cumulative field in the extension record was built from the parent's own segment --
`exchanges_committed`, and the parent's completion report -- so a chain of any length described
its last two segments. The failure is invisible from inside the record: `steps` and
`production_ps` are derived from the absolute step count and stay correct, so a five-segment
manifest is perfectly self-consistent and simply disagrees with the run it describes. That is
the whole reason the test walks THREE segments. Two is exactly the length at which the broken
arithmetic and the correct arithmetic agree, and a two-segment test passes against both.

The numbers below are the shipped 125 ns ALA ladder's shape -- 10000 exchanges and 25000
proposals per 25 ns segment -- so a reader can compare a failure here against
`data/reference/ALA-explicit-HMR/REST2-run1/remd_records/restart_prod*.json` directly.
"""

from __future__ import annotations

import types

import pytest

from md_tools.remd.driver import ReplicaRun, parent_chain_inheritance

SEGMENT_EXCHANGES = 10_000
SEGMENT_PROPOSALS = 25_000
SEGMENT_ACCEPTED = 12_335
SEGMENT_STEPS = 25_000_000
PS_PER_STEP = 0.004


def _provenance(parent, *, step):
    """One segment's record, built by the runtime's own method on a minimal stand-in.

    `_extension_provenance` reads four things off `self`: the reporter's last exchange number,
    and the format constant. Nothing else about a ladder is involved in the arithmetic under
    test, so a stand-in keeps this in the fast lane without weakening what is asserted.
    """
    run = types.SimpleNamespace(
        reporter=types.SimpleNamespace(last_exchange=lambda: SEGMENT_EXCHANGES - 1),
        EXTENSION_FORMAT=ReplicaRun.EXTENSION_FORMAT,
    )
    state = {
        "extends": parent,
        "step": step,
        "schedule": types.SimpleNamespace(step_to_ps=lambda s: s * PS_PER_STEP),
    }
    report = {"overall": {"accepted": SEGMENT_ACCEPTED, "proposed": SEGMENT_PROPOSALS}}
    return ReplicaRun._extension_provenance(run, state, report)


def _original_parent():
    """The parent fields an ORIGINAL run contributes: no chain, no format."""
    manifest = {
        "exchanges_committed": SEGMENT_EXCHANGES,
        "production_ps_per_replica": SEGMENT_STEPS * PS_PER_STEP,
    }
    return {
        "exchanges_committed": SEGMENT_EXCHANGES,
        "steps_completed": SEGMENT_STEPS,
        "production_ps": SEGMENT_STEPS * PS_PER_STEP,
        "completion_report": {"overall": {"accepted": SEGMENT_ACCEPTED,
                                          "proposed": SEGMENT_PROPOSALS}},
        **parent_chain_inheritance(manifest),
    }


def _parent_from(record, *, step):
    """The next segment's parent, inherited from `record` the way `_extend_from` inherits it.

    The chain and the format come through `parent_chain_inheritance`, the one function the
    runtime uses, rather than through a copy of that mapping written here. A second copy would
    let this test agree with itself while both halves disagreed with a manifest on disk.
    """
    manifest = {"extends": record}
    return {
        "exchanges_committed": record["segment"]["exchanges"],
        "steps_completed": step,
        "production_ps": step * PS_PER_STEP,
        "completion_report": {"overall": dict(record["segment"]["neighbouring_acceptance"])},
        **parent_chain_inheritance(manifest),
    }


def _chain_of(n_segments):
    """Walk a chain of `n_segments` and return the last segment's record."""
    record = _provenance(_original_parent(), step=2 * SEGMENT_STEPS)
    for index in range(2, n_segments):
        parent = _parent_from(record, step=index * SEGMENT_STEPS)
        record = _provenance(parent, step=(index + 1) * SEGMENT_STEPS)
    return record


def test_the_second_segment_totals_two():
    """The one length at which both the broken and the correct arithmetic agree."""
    record = _provenance(_original_parent(), step=2 * SEGMENT_STEPS)
    assert record["chain"]["segments"] == 2
    assert record["chain"]["exchanges"] == 2 * SEGMENT_EXCHANGES
    assert record["segment"]["first_exchange_number"] == SEGMENT_EXCHANGES + 1


@pytest.mark.parametrize("n_segments", [3, 4, 5])
def test_a_chain_counts_its_own_segments(n_segments):
    record = _chain_of(n_segments)
    assert record["chain"]["segments"] == n_segments


@pytest.mark.parametrize("n_segments", [3, 4, 5])
def test_chain_exchanges_accumulate_over_every_segment(n_segments):
    record = _chain_of(n_segments)
    assert record["chain"]["exchanges"] == n_segments * SEGMENT_EXCHANGES


@pytest.mark.parametrize("n_segments", [3, 4, 5])
def test_the_first_exchange_number_is_chain_global(n_segments):
    """Segment n starts after every exchange of segments 1..n-1, not after the parent's own."""
    record = _chain_of(n_segments)
    assert record["segment"]["first_exchange_number"] == (n_segments - 1) * SEGMENT_EXCHANGES + 1


@pytest.mark.parametrize("n_segments", [3, 4, 5])
def test_the_chain_acceptance_is_measured_over_the_whole_chain(n_segments):
    """The field that matters, and the one a reader cannot sanity-check.

    A wrong `segments` is visible to anyone who knows how many times they extended. A chain
    acceptance of 0.49378 over 50000 proposals, when the chain really proposed 125000, reads as
    a whole-ladder statistic and is the mean of the last two segments. Nothing in the record
    says so.
    """
    record = _chain_of(n_segments)
    acceptance = record["chain"]["neighbouring_acceptance"]
    assert acceptance["proposed"] == n_segments * SEGMENT_PROPOSALS
    assert acceptance["accepted"] == n_segments * SEGMENT_ACCEPTED
    assert acceptance["acceptance"] == pytest.approx(SEGMENT_ACCEPTED / SEGMENT_PROPOSALS)


@pytest.mark.parametrize("n_segments", [2, 3, 4, 5])
def test_the_derived_fields_were_right_all_along(n_segments):
    """Why it survived: these two come from the absolute step and never needed accumulating.

    Asserted so a future change cannot "fix" the cumulative fields by switching these to the
    same per-segment source the broken ones used.
    """
    record = _chain_of(n_segments) if n_segments > 2 else _provenance(
        _original_parent(), step=2 * SEGMENT_STEPS)
    assert record["chain"]["steps"] == n_segments * SEGMENT_STEPS
    assert record["chain"]["production_ps"] == pytest.approx(
        n_segments * SEGMENT_STEPS * PS_PER_STEP)
    assert record["segment"]["steps"] == SEGMENT_STEPS
    assert record["segment"]["exchanges"] == SEGMENT_EXCHANGES


def test_a_segment_record_is_v2():
    assert _provenance(_original_parent(), step=2 * SEGMENT_STEPS)["format"] \
        == "md-tools-extension/v2"


def test_a_v1_parent_leaves_the_cumulative_fields_empty():
    """An older parent's totals are the broken ones, so they are not a base to add to.

    The alternative -- trusting them -- would make a v2 segment on a v1 parent report a wrong
    number under a format that promises a right one, which is worse than v1 was. Empty says
    go and count; wrong does not.
    """
    v1 = {
        "exchanges_committed": SEGMENT_EXCHANGES,
        "steps_completed": 4 * SEGMENT_STEPS,
        "production_ps": 4 * SEGMENT_STEPS * PS_PER_STEP,
        "completion_report": {"overall": {"accepted": SEGMENT_ACCEPTED,
                                          "proposed": SEGMENT_PROPOSALS}},
        **parent_chain_inheritance({"extends": {
            "format": "md-tools-extension/v1",
            "chain": {"segments": 2, "exchanges": 2 * SEGMENT_EXCHANGES,
                      "neighbouring_acceptance": {"accepted": 2 * SEGMENT_ACCEPTED,
                                                  "proposed": 2 * SEGMENT_PROPOSALS,
                                                  "complete": True}},
        }}),
    }
    record = _provenance(v1, step=5 * SEGMENT_STEPS)
    assert record["chain"]["segments"] is None
    assert record["chain"]["exchanges"] is None
    assert record["segment"]["first_exchange_number"] is None
    assert record["chain"]["neighbouring_acceptance"]["acceptance"] is None
    assert "v1" in record["chain"]["chain_totals"]

    # The segment's OWN counts are exact regardless of the parent's format: the defect was in
    # accumulation, and nothing about this segment's own propagation is in doubt.
    assert record["segment"]["exchanges"] == SEGMENT_EXCHANGES
    assert record["segment"]["neighbouring_acceptance"]["proposed"] == SEGMENT_PROPOSALS
    assert record["chain"]["steps"] == 5 * SEGMENT_STEPS


def test_the_record_says_where_its_totals_came_from():
    """`chain_totals` is prose, and it is the field that distinguishes 1 from unknown."""
    second = _provenance(_original_parent(), step=2 * SEGMENT_STEPS)
    assert "original run" in second["chain"]["chain_totals"]
    third = _chain_of(3)
    assert "parent's chain" in third["chain"]["chain_totals"]
