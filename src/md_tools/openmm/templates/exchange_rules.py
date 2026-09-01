#!/usr/bin/env python
"""The narrow boundary between "run a ladder" and "decide what moves". Copied into every project.

A transition rule receives a bounded view of the ladder and returns explicit proposals and
decisions. It never propagates, never opens storage, never touches MPI and never parses a command
line. That is the whole point: a later method -- a non-Boltzmann reservoir, a kinetic reservoir --
becomes a new rule file and changes nothing in `openmm-md`.

This is a CONTRACT, not a framework. There is no registry, no plugin discovery, no dependency
injection and no method database. `--exchange-rule FILE.py` loads one file by location and takes
the object it defines. A rule that needs to remember something across iterations returns a small
JSON-serialisable dictionary, which the driver stores in the checkpoint and hands back on resume.
"""
import hashlib
from pathlib import Path

import numpy as np

#: Bumped when the meaning of what a rule receives or returns changes.
RULE_INTERFACE_VERSION = "md-tools-exchange-rule/v1"


class ExchangeRuleError(ValueError):
    """A rule cannot be loaded, or refuses the situation it was given."""


class ExchangeContext:
    """Everything a rule may see. Deliberately small.

    `reduced_potential(state_index, walker)` evaluates the configuration currently held by
    `walker` in the Hamiltonian of `state_index`, independently -- never by scaling another
    energy. It is a callable rather than a precomputed matrix so a rule that needs only four
    numbers pays for four, not for N squared.
    """

    def __init__(self, *, iteration, segment, protocol, state_to_walker, reduced_potential,
                 rng, reservoir=None, rule_state=None, exchange_index=0):
        self.iteration = int(iteration)
        self.segment = int(segment)
        #: How many exchange attempts have happened, this one included and counted from 0. The
        #: odd/even schedule MUST alternate on this and not on `iteration`: exchanges land every
        #: `exchange_stride` iterations, so with an even stride the iteration parity never changes
        #: and one phase is used forever. A three-state ladder exchanging every second segment
        #: proposed (1,2) six times and (0,1) not once.
        self.exchange_index = int(exchange_index)
        self.protocol = protocol
        #: state_to_walker[i] is the walker whose configuration currently sits in state i.
        self.state_to_walker = list(int(w) for w in state_to_walker)
        self._reduced_potential = reduced_potential
        #: A dedicated numpy Generator. A rule must draw from this and from nothing else, so a
        #: run repeats exactly under a recorded seed.
        self.rng = rng
        self.reservoir = reservoir
        self.rule_state = dict(rule_state or {})

    @property
    def n_states(self):
        return self.protocol.n_states

    @property
    def walker_to_state(self):
        inverse = [None] * self.n_states
        for state, walker in enumerate(self.state_to_walker):
            inverse[walker] = state
        return inverse

    def reduced_potential(self, state_index, walker):
        return float(self._reduced_potential(int(state_index), int(walker)))


class ExchangeOutcome:
    """What a rule decided. Swaps are stated as state-index pairs, not as walker moves.

    `swaps` are applied in order by the driver; each exchanges the configurations occupying the two
    named states. `reservoir_refresh` is an optional replacement of one state's configuration from
    a prepared reservoir, which is not a swap and is recorded separately so it can never be
    mistaken for one.
    """

    def __init__(self, *, proposals=(), swaps=(), reservoir_refresh=None, rule_state=None,
                 diagnostics=None):
        #: (state_i, state_j, log_alpha, accepted) for every pair actually considered.
        self.proposals = [tuple(p) for p in proposals]
        self.swaps = [tuple(int(x) for x in pair) for pair in swaps]
        self.reservoir_refresh = reservoir_refresh
        self.rule_state = dict(rule_state or {})
        self.diagnostics = dict(diagnostics or {})


def alternating_pairs(n_states, phase):
    """Adjacent pairs for one phase of the odd/even schedule.

    phase 0 -> (0,1), (2,3), ...      phase 1 -> (1,2), (3,4), ...

    With TWO states the odd phase covers no pair at all. A schedule that still drew phase 1 there
    would propose nothing on half its exchange iterations and quietly exchange at half the
    configured rate, so the driver's phase choice must never produce an empty sweep -- see
    `deterministic_phase`.
    """
    return [(i, i + 1) for i in range(phase % 2, n_states - 1, 2)]


def deterministic_phase(n_states, exchange_index):
    """Which phase this EXCHANGE ATTEMPT uses. Deterministic and recorded, never random.

    Driven by the count of exchange attempts, never by the iteration number: exchanges happen every
    `exchange_stride` iterations, so an even stride makes the iteration parity constant and one
    phase would be used for the whole run.

    Strict alternation covers every adjacent pair over any two consecutive exchange iterations,
    which a random phase only does on average. When only one phase contains a pair -- two states --
    that phase is used every time rather than half the time.
    """
    if n_states < 2:
        return 0
    phases = [p for p in (0, 1) if alternating_pairs(n_states, p)]
    if len(phases) == 1:
        return phases[0]
    return int(exchange_index) % 2


class NeighbouringExchangeRule:
    """Conventional REST2: one sweep of adjacent pairs per exchange iteration.

    All four reduced potentials of a pair are evaluated independently through the context. The
    criterion is

        log(alpha) = [u_i(x_i) + u_j(x_j)] - [u_i(x_j) + u_j(x_i)]

    and the swap is accepted when log(U) < min(0, log alpha) for U drawn from the rule's dedicated
    stream. Because every rung shares one beta, no temperature factor and no velocity rescaling
    appears anywhere in this.
    """

    name = "neighbouring"
    version = RULE_INTERFACE_VERSION

    def __init__(self, **parameters):
        self.parameters = dict(parameters)

    def describe(self):
        return {"name": self.name, "interface_version": self.version,
                "parameters": dict(self.parameters),
                "schedule": "strict alternation of odd/even adjacent-pair sweeps",
                "criterion": "log(alpha) = [u_i(x_i)+u_j(x_j)] - [u_i(x_j)+u_j(x_i)]",
                "source": "built in to exchange_rules.py"}

    def propose(self, context):
        phase = deterministic_phase(context.n_states, context.exchange_index)
        proposals, swaps = [], []
        # A pair is decided against the mapping as it stands, and an accepted swap is applied to a
        # local copy immediately, so a state cannot be swapped twice in one sweep. Adjacent pairs
        # within one phase are disjoint, so this is bookkeeping rather than a constraint.
        occupancy = list(context.state_to_walker)
        for state_i, state_j in alternating_pairs(context.n_states, phase):
            walker_i, walker_j = occupancy[state_i], occupancy[state_j]
            u_ii = context.reduced_potential(state_i, walker_i)
            u_jj = context.reduced_potential(state_j, walker_j)
            u_ij = context.reduced_potential(state_i, walker_j)
            u_ji = context.reduced_potential(state_j, walker_i)
            log_alpha = (u_ii + u_jj) - (u_ij + u_ji)
            accepted = bool(log_alpha >= 0.0 or np.log(context.rng.random()) < log_alpha)
            proposals.append((state_i, state_j, float(log_alpha), accepted))
            if accepted:
                swaps.append((state_i, state_j))
                occupancy[state_i], occupancy[state_j] = walker_j, walker_i
        return ExchangeOutcome(proposals=proposals, swaps=swaps,
                               diagnostics={"phase": phase})


def load_rule(path):
    """Load one rule file by location and take the object it defines.

    The file must define `rule` -- an object with `describe()` and `propose(context)` -- or a
    `make_rule()` returning one. That is the entire loading protocol; there is no registry and no
    entry point.
    """
    import importlib.util

    path = Path(path)
    if not path.is_file():
        raise ExchangeRuleError(f"--exchange-rule {path} does not exist")
    spec = importlib.util.spec_from_file_location("openmm_md_exchange_rule", str(path))
    if spec is None or spec.loader is None:
        raise ExchangeRuleError(f"{path} could not be loaded as a Python file")
    module = importlib.util.module_from_spec(spec)
    # The rule sits beside the runtime modules it needs, so its own directory is importable.
    import sys
    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec.loader.exec_module(module)

    rule = getattr(module, "rule", None)
    if rule is None and hasattr(module, "make_rule"):
        rule = module.make_rule()
    if rule is None:
        raise ExchangeRuleError(
            f"{path} defines neither `rule` nor `make_rule()`. An exchange rule is one object "
            f"with describe() and propose(context); see exchange_rules.py for the contract.")
    for required in ("describe", "propose"):
        if not callable(getattr(rule, required, None)):
            raise ExchangeRuleError(f"the rule in {path} has no callable {required}()")
    return rule, rule_identity(path, rule)


def rule_identity(path, rule):
    """What is recorded about a rule, so a resume can refuse a changed one."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    described = rule.describe()
    return {"name": described.get("name"), "interface_version": described.get("interface_version"),
            "file": path.name, "sha256": digest, "parameters": described.get("parameters"),
            "describe": described}


def builtin_rule_identity(rule):
    described = rule.describe()
    return {"name": described.get("name"), "interface_version": described.get("interface_version"),
            "file": None, "sha256": None, "parameters": described.get("parameters"),
            "describe": described}
