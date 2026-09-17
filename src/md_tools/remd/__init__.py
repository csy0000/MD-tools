"""Replica exchange: the ladder, the rules that move between its rungs, and what it writes.

Generic on purpose. `REMDRunner` coordinates replicas, schedules events, places devices and owns
the storage; the *decision* about which replicas swap is a rule object it is given. REST2 supplies
`NeighborExchangeRule`. A method whose transitions differ is a new rule file passed with
`--exchange-rule`, not a forked driver. (rREST2, which added a reservoir refresh rule, is archived
as of 0.5.4: see `archive/rREST2/`.)

The Hamiltonian scaling is NOT here. It lives in `md_tools.rest2`, because three of the four
protocols that scale never exchange.

A generated ladder script is an entry point:

    #!/usr/bin/env python
    from md_tools.remd import run_generated_remd
    raise SystemExit(run_generated_remd(__file__, protocol="REST2"))
"""
from __future__ import annotations

from .driver import ReplicaRun as REMDRunner
from .facade import REST2Protocol
from .engine import exchange_log_acceptance, reduced_potential
from .generated import run_generated_remd, run_remd
from .rules import ExchangeRuleError, NeighbouringExchangeRule as NeighborExchangeRule
from .rules import alternating_pairs, load_rule

__all__ = [
    "REMDRunner",
    "REST2Protocol",
    "NeighborExchangeRule",
    "run_remd",
    "run_generated_remd",
    # the exchange mathematics, in ONE place: it used to be duplicated in md_tools.rest2
    "reduced_potential",
    "exchange_log_acceptance",
    "alternating_pairs",
    "load_rule",
    "ExchangeRuleError",
]
