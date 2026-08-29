#!/usr/bin/env python
"""Lifetime mixing statistics and round trips, computed from the authoritative reporter history.

Copied verbatim into every generated REST2 project.

WHY THIS EXISTS
    `sampler._n_proposed_matrix` and `_n_accepted_matrix` are reset at the start of every mixing
    call, so reading them after a run gives the LAST mixing event only. A summary built from them
    described 1000 exchange attempts using the statistics of one, and reported acceptance from a
    single sweep as though it were the run's.

    `MultiStateReporter.write_mixing_statistics` is called on EVERY iteration, including the ones a
    stride skips, which record zeros. `read_mixing_statistics(slice(None))` therefore returns the
    complete per-iteration, non-cumulative history -- including across an interrupted resume and an
    extension, because it is the storage that is authoritative and not the process that wrote it.
    Summing that history is the only honest lifetime statistic, and it is a public reporter method,
    so nothing here parses a raw NetCDF variable.

DIAGONAL ENTRIES ARE NOT EXCHANGES
    `swap-all` draws replica pairs uniformly and may draw i == j. A self-swap has log_p = 0, is
    always "accepted", and is recorded on the DIAGONAL of both matrices. Counting the diagonal as
    acceptance would report a number near 100% that means nothing. Every figure below is computed
    from off-diagonal entries, and the diagonal is reported separately and labelled.
"""
import numpy as np


def lifetime_mixing_statistics(accepted, proposed, *, scheme, exchange_stride=None,
                               first_iteration=0):
    """Aggregate the per-iteration mixing history into lifetime statistics.

    `accepted` and `proposed` are what `MultiStateReporter.read_mixing_statistics(slice(None))`
    returns: arrays of shape (n_iterations, n_states, n_states), NOT cumulative.

    Returns a plain dictionary; nothing here formats or prints.
    """
    accepted = np.asarray(accepted, dtype=np.int64)
    proposed = np.asarray(proposed, dtype=np.int64)
    if accepted.shape != proposed.shape:
        raise ValueError(f"accepted{accepted.shape} and proposed{proposed.shape} disagree in shape")
    if accepted.ndim != 3 or accepted.shape[1] != accepted.shape[2]:
        raise ValueError(
            f"expected (iterations, states, states); got {accepted.shape}")

    n_iterations, n_states, _ = accepted.shape

    # An iteration is a MIXING EVENT if it actually proposed something. A stride-skipped iteration
    # records an all-zero row, so this counts events rather than inferring them from the schedule.
    per_iteration_proposals = proposed.sum(axis=(1, 2))
    mixing_iterations = [int(i) for i in np.nonzero(per_iteration_proposals)[0]]
    events = len(mixing_iterations)

    total_proposed = proposed.sum(axis=0)
    total_accepted = accepted.sum(axis=0)

    off_diagonal = ~np.eye(n_states, dtype=bool)
    pairs = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            # The matrices are symmetric by construction: both (i,j) and (j,i) are incremented.
            n_proposed = int(total_proposed[i, j])
            n_accepted = int(total_accepted[i, j])
            pairs.append({
                "state_pair": [i, j],
                "adjacent": bool(j == i + 1),
                "proposed": n_proposed,
                "accepted": n_accepted,
                "acceptance": (n_accepted / n_proposed) if n_proposed else None,
            })

    schedule = None
    if exchange_stride:
        expected = [i for i in range(first_iteration, n_iterations)
                    if i > 0 and i % int(exchange_stride) == 0]
        unexpected = sorted(set(mixing_iterations) - set(expected))
        missing = sorted(set(expected) - set(mixing_iterations))
        schedule = {
            "exchange_stride": int(exchange_stride),
            "expected_mixing_events": len(expected),
            "observed_mixing_events": events,
            "agrees_with_schedule": not unexpected and not missing,
            "iterations_that_mixed_unexpectedly": unexpected[:16],
            "scheduled_iterations_that_did_not_mix": missing[:16],
        }

    return {
        "scheme": scheme,
        "basis": "lifetime, summed over every stored iteration in the analysis NetCDF",
        "iteration_range": [int(first_iteration), int(n_iterations - 1)],
        "iterations_stored": int(n_iterations),
        "mixing_events": events,
        "proposal_semantics": _proposal_semantics(scheme),
        "total_proposed_off_diagonal": int(total_proposed[off_diagonal].sum() // 2),
        "total_accepted_off_diagonal": int(total_accepted[off_diagonal].sum() // 2),
        "overall_acceptance": (
            float(total_accepted[off_diagonal].sum() / total_proposed[off_diagonal].sum())
            if total_proposed[off_diagonal].sum() else None),
        # Self-swaps: i == j, log_p = 0, always accepted. Reported so the number is visible and
        # cannot be mistaken for exchange acceptance.
        "self_proposals_on_diagonal": int(np.trace(total_proposed)),
        "by_state_pair": pairs,
        "schedule": schedule,
    }


def _proposal_semantics(scheme):
    if scheme == "swap-all":
        return ("swap-all: n_replicas**3 uniformly random replica pairs per mixing event, over ALL "
                "state pairs and not only adjacent ones. Pairs with i == j are self-swaps, always "
                "accepted, and are counted on the diagonal. This is NOT a neighbouring sweep.")
    if scheme == "swap-neighbors":
        return ("swap-neighbors: one sweep of alternating adjacent pairs per mixing event, so only "
                "adjacent state pairs ever receive a proposal.")
    return f"{scheme}: proposal semantics not described by this module"


def adjacent_pairs(statistics):
    """Just the adjacent pairs, for the readable summary of an ordered ladder."""
    return [pair for pair in statistics["by_state_pair"] if pair["adjacent"]]


# --- round trips ---------------------------------------------------------------------------------

def count_round_trips(series, *, cold_state, hot_state):
    """Complete round trips in one walker's state history.

    A complete round trip is:

        cold state visited  ->  hot state visited later  ->  cold state visited later still

    The starting position earns nothing. A walker that BEGINS at the hot state has not completed a
    round trip when it first reaches cold: it has completed half of one. It must then return to hot
    and come back to cold. The previous implementation credited that walker immediately, which
    inflated the count for exactly the walkers that had travelled least.

    Implemented as an explicit three-state machine so the rule is readable and testable:

        seeking_cold  -- has not yet been seen at cold; nothing can be completed
        at_cold       -- armed; reaching hot advances it
        at_hot        -- half-way; reaching cold completes a trip and re-arms
    """
    if cold_state == hot_state:
        raise ValueError(f"cold and hot states must differ; both are {cold_state}")
    phase = "seeking_cold"
    completed = 0
    for value in np.asarray(series).tolist():
        value = int(value)
        if value == cold_state:
            if phase == "at_hot":
                completed += 1
            phase = "at_cold"
        elif value == hot_state:
            if phase == "at_cold":
                phase = "at_hot"
            # From `seeking_cold`, reaching hot proves nothing: the walker may have started there.
    return completed


def round_trips(mapping, *, n_states=None):
    """Round trips for every walker, plus the half-trip diagnostics worth seeing.

    `mapping` is `reporter.read_replica_thermodynamic_states()`: shape (iterations, walkers), where
    entry [i, w] is the thermodynamic-state index walker w occupied at iteration i.
    """
    mapping = np.asarray(mapping)
    if mapping.ndim != 2:
        raise ValueError(f"expected (iterations, walkers); got {mapping.shape}")
    states = int(mapping.max()) + 1 if n_states is None else int(n_states)
    cold, hot = 0, states - 1
    result = []
    for walker in range(mapping.shape[1]):
        series = mapping[:, walker]
        visited = sorted({int(v) for v in series.tolist()})
        result.append({
            "walker": walker,
            "started_at_state": int(series[0]),
            "round_trips": count_round_trips(series, cold_state=cold, hot_state=hot),
            "visited_cold": cold in visited,
            "visited_hot": hot in visited,
            "visited_all_states": visited == list(range(states)),
            "states_visited": visited,
        })
    return result


def mapping_is_permutation_every_iteration(mapping, *, n_states=None):
    """Every iteration must occupy each thermodynamic state exactly once."""
    mapping = np.asarray(mapping)
    states = int(mapping.max()) + 1 if n_states is None else int(n_states)
    expected = list(range(states))
    bad = [int(i) for i, row in enumerate(mapping) if sorted(int(v) for v in row) != expected]
    return (not bad), bad[:16]
