#!/usr/bin/env python
"""The rREST2 transition rule: conventional REST2, plus a Boltzmann reservoir refresh at tau_max.

Copied verbatim into a generated rREST2 project and selected with

    the executor, with --exchange-rule rrest2_exchange.py and --reservoir reservoir.yaml

It carries no path and no parameter of its own: the refresh schedule comes from the reservoir
declaration `--reservoir` names, and everything else comes from the protocol. That is what lets
this file be copied verbatim rather than generated with values baked in.

THE ORDER, CHOSEN ONCE AND PERSISTED
    When an ordinary exchange iteration and a reservoir attempt coincide, the NEIGHBOURING SWEEP
    HAPPENS FIRST and the reservoir refresh second.

    The reason is that a refresh installs a configuration that this ladder has not propagated. If
    the refresh went first, that configuration would immediately take part in the sweep and could
    be swapped down the ladder in the very same iteration, having never been propagated in the
    ladder at all. Doing the sweep first means a reservoir configuration must survive at least one
    propagation segment at the top rung before it can descend, which is the conventional reservoir
    REMD ordering.

    The order is recorded in `describe()` and in the run's manifest, so it is never left to
    process scheduling and never has to be inferred from a log.

ACCEPTANCE
    Under the v1 contract -- Boltzmann-weighted reservoir at exactly the top rung's tau,
    temperature, Hamiltonian and fixed-volume ensemble -- a refresh is accepted with probability
    one, because the drawn configuration is a sample from the same distribution as the one it
    replaces. `rrest2_reservoir.py` checks every clause of that contract and refuses anything else;
    this rule does not re-derive it and must not be pointed at a reservoir that does not satisfy it.
"""
# An ABSOLUTE import: this file is loaded BY PATH through `--exchange-rule`, so it is not part of
# any package when it runs and a relative import has no parent to resolve against. A user writing
# their own rule file imports from the installed package the same way.
from md_tools.remd.reservoir import ReservoirRefreshRule

# The rule itself is `md_tools.remd.reservoir.ReservoirRefreshRule`. This file stays as the
# worked example of the `--exchange-rule` plug-in contract: a module that names one rule
# object and nothing else. It carries no copy of the logic.

rule = ReservoirRefreshRule()
