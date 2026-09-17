"""rREST2 cases moved out of tests/test_runtime_packaging.py when rREST2 was archived (0.5.4).

Kept as the record of what the method was tested against; not collected, and not expected to
run against current md-tools. See archive/rREST2/README.md and the tag rREST2-final.
"""


def test_the_reservoir_rule_is_exposed_generically():
    """rREST2 composes it; a future reservoir REMD method must be able to reuse it."""
    from md_tools.remd.reservoir import ReservoirRefreshRule

    assert hasattr(ReservoirRefreshRule, "propose")
    assert hasattr(ReservoirRefreshRule, "describe")
