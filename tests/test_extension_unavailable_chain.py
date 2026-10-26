"""A v2 extension whose parent is a v2 extension with EMPTY chain totals must still complete.

The defect (reported by hpREST2 on afd704a): a chain 0.6.4 original -> v1 extension -> v1 extension
-> v2 extension -> v2 extension. The first v2 segment honestly writes `None` for the chain's segments
and exchanges (its parent's totals are the broken v1 ones). The NEXT segment ran all its exchanges
and then crashed writing its record: the branch for a v2 parent did `int(parent_chain["segments"])`
on that `None`. The crash is at completion, after the compute, and leaves no manifest.

The rule is the one `test_extension_chain_accounting.py` already pins for a v1 parent: an empty field
says "go and count"; a wrong one does not. An unavailable total stays unavailable down the chain.
"""
from __future__ import annotations

import types

from md_tools.remd.driver import ReplicaRun, parent_chain_inheritance

EXCHANGES = 5_000
PROPOSALS = 13_750
ACCEPTED = 4_513
STEP = 12_500_000
PS = 0.004


def _record(parent, *, step):
    run = types.SimpleNamespace(
        reporter=types.SimpleNamespace(last_exchange=lambda: EXCHANGES - 1),
        EXTENSION_FORMAT=ReplicaRun.EXTENSION_FORMAT)
    state = {"extends": parent, "step": step,
             "schedule": types.SimpleNamespace(step_to_ps=lambda s: s * PS)}
    return ReplicaRun._extension_provenance(
        run, state, {"overall": {"accepted": ACCEPTED, "proposed": PROPOSALS}})


def _as_parent(record, *, step):
    return {"exchanges_committed": record["segment"]["exchanges"], "steps_completed": step,
            "production_ps": step * PS,
            "completion_report": {"overall": dict(record["segment"]["neighbouring_acceptance"])},
            **parent_chain_inheritance({"extends": record})}


def _v1_parent():
    return {"exchanges_committed": EXCHANGES, "steps_completed": 10 * STEP,
            "production_ps": 10 * STEP * PS,
            "completion_report": {"overall": {"accepted": ACCEPTED, "proposed": PROPOSALS}},
            **parent_chain_inheritance({"extends": {
                "format": "md-tools-extension/v1",
                "chain": {"segments": 2, "exchanges": 2 * EXCHANGES,
                          "neighbouring_acceptance": {"accepted": 2 * ACCEPTED,
                                                      "proposed": 2 * PROPOSALS,
                                                      "complete": True}}}})}


def test_a_v2_segment_on_a_v2_segment_with_empty_totals_completes_and_stays_empty():
    first = _record(_v1_parent(), step=11 * STEP)                    # v1 -> v2: totals honestly empty
    assert first["chain"]["segments"] is None and first["chain"]["exchanges"] is None
    second = _record(_as_parent(first, step=11 * STEP), step=12 * STEP)    # v2 -> v2: used to raise TypeError
    assert second["format"] == "md-tools-extension/v2"
    assert second["chain"]["segments"] is None and second["chain"]["exchanges"] is None
    assert second["segment"]["first_exchange_number"] is None
    assert second["chain"]["neighbouring_acceptance"]["acceptance"] is None
    assert second["chain"]["neighbouring_acceptance"]["complete"] is False
    # what is exact stays exact: this segment's own counts, and the quantities derived from the step
    assert second["segment"]["exchanges"] == EXCHANGES
    assert second["segment"]["neighbouring_acceptance"]["proposed"] == PROPOSALS
    assert second["chain"]["steps"] == 12 * STEP
    assert second["chain"]["production_ps"] == 12 * STEP * PS
    assert "unavailable" in second["chain"]["chain_totals"]


def test_the_emptiness_goes_on_down_a_longer_chain():
    record = _record(_v1_parent(), step=11 * STEP)
    for index in range(12, 16):
        record = _record(_as_parent(record, step=(index - 1) * STEP), step=index * STEP)
        assert record["chain"]["segments"] is None and record["chain"]["exchanges"] is None


def test_a_v2_parent_with_totals_but_no_acceptance_is_not_summed_as_zero():
    """`int(None or 0)` turned a missing acceptance count into 0 and added it: a chain acceptance
    over only the last segment, published under a name that says whole chain."""
    parent = _as_parent(_record({"exchanges_committed": EXCHANGES, "steps_completed": 2 * STEP,
                                 "production_ps": 2 * STEP * PS,
                                 "completion_report": {"overall": {"accepted": ACCEPTED,
                                                                    "proposed": PROPOSALS}}},
                                step=3 * STEP), step=3 * STEP)
    parent["chain"] = dict(parent["chain"], neighbouring_acceptance={"accepted": None, "proposed": None})
    record = _record(parent, step=4 * STEP)
    assert record["chain"]["segments"] == 3                          # the counts that exist are still summed
    assert record["chain"]["exchanges"] == 3 * EXCHANGES
    assert record["chain"]["neighbouring_acceptance"]["acceptance"] is None
