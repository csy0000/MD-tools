#!/usr/bin/env python
"""The rREST2 transition rule: conventional REST2, plus a Boltzmann reservoir refresh at tau_max.

Copied verbatim into a generated rREST2 project and selected with

    openmm-md ... --exchange-rule rREST2/rrest2_exchange.py --reservoir rREST2/reservoir.yaml

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
from exchange_rules import NeighbouringExchangeRule, RULE_INTERFACE_VERSION

#: Where the reservoir refresh sits relative to the ordinary sweep. See the module docstring.
REFRESH_ORDER = "after_neighbouring_sweep"


class ReservoirREST2Rule:
    """Neighbouring REST2 with a periodic Boltzmann reservoir refresh of the hottest state."""

    name = "rrest2-boltzmann"
    version = RULE_INTERFACE_VERSION

    def __init__(self):
        self.neighbouring = NeighbouringExchangeRule()

    def describe(self):
        return {
            "name": self.name,
            "interface_version": self.version,
            "parameters": {"refresh_order": REFRESH_ORDER},
            "schedule": ("strict alternation of odd/even adjacent-pair sweeps, plus a reservoir "
                         "refresh of the top state every refresh_interval_exchanges exchange "
                         "iterations"),
            "refresh_order": REFRESH_ORDER,
            "refresh_acceptance": ("probability one under the Boltzmann/same-state contract "
                                   "checked by rrest2_reservoir.py"),
            "criterion": "log(alpha) = [u_i(x_i)+u_j(x_j)] - [u_i(x_j)+u_j(x_i)]",
            "source": "generated rrest2_exchange.py",
        }

    def propose(self, context):
        # 1. the ordinary sweep, unchanged.
        outcome = self.neighbouring.propose(context)

        reservoir = context.reservoir
        if reservoir is None:
            return outcome

        declaration = getattr(reservoir, "declaration", {}) or {}
        interval = int(declaration.get("refresh_interval_exchanges", 1))
        if interval < 1:
            raise ValueError(
                f"refresh_interval_exchanges must be >= 1; got {interval}")

        state = dict(context.rule_state)
        attempts = int(state.get("refresh_attempts", 0))
        # Counted in EXCHANGE iterations, not segments: an interval of 1 refreshes at every
        # exchange, and the count survives a resume because the rule state is checkpointed.
        exchanges_seen = int(state.get("exchanges_seen", 0)) + 1
        state["exchanges_seen"] = exchanges_seen

        if exchanges_seen % interval != 0:
            outcome.rule_state = state
            return outcome

        # 2. the refresh, second and separately recorded.
        top_state = context.n_states - 1
        frame = int(context.rng.integers(reservoir.n_frames))
        # A DCD carries no velocities, so the installed configuration needs fresh momenta at the
        # one common temperature. The seed is derived from the rule's own stream so a resumed run
        # redraws the same way it would have.
        velocity_seed = int(context.rng.integers(1, 2 ** 31 - 1))
        state["refresh_attempts"] = attempts + 1
        state["last_refresh_frame"] = frame
        state["last_refresh_iteration"] = int(context.iteration)

        outcome.reservoir_refresh = {
            "state": top_state,
            "frame": frame,
            "velocity_seed": velocity_seed,
            "accepted": True,
            "order": REFRESH_ORDER,
        }
        outcome.rule_state = state
        outcome.diagnostics = dict(outcome.diagnostics or {})
        outcome.diagnostics["reservoir_refresh"] = {
            "state": top_state, "frame": frame, "interval_exchanges": interval}
        return outcome


rule = ReservoirREST2Rule()
