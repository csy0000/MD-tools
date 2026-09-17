"""rREST2 cases moved out of tests/test_exchange_entry_reuse.py when rREST2 was archived (0.5.4).

Kept as the record of what the method was tested against; not collected, and not expected to
run against current md-tools. See archive/rREST2/README.md and the tag rREST2-final.
"""
from md_tools.remd.reservoir import ReservoirRefreshRule
from md_tools.remd.rules import NeighbouringExchangeRule

def test_the_reservoir_rule_declares_the_sweep_it_delegates_to():
    """The refresh consults no reduced potential, so it adds nothing to the declaration."""
    plain = NeighbouringExchangeRule()
    rule = ReservoirRefreshRule()
    for exchange_index in (1, 2, 3):
        arguments = {"n_states": 4, "state_to_walker": [3, 2, 1, 0],
                     "exchange_index": exchange_index}
        assert rule.required_entries(**arguments) == plain.required_entries(**arguments)
