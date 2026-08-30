#!/usr/bin/env python
"""Lifetime statistics and round trips, from the authoritative stored history.

Copied verbatim into every generated replica project.

Every figure here is summed over the WHOLE stored history, so it survives an interrupted resume and
an extension unchanged: the storage is authoritative, not the process that happened to write it.
Nothing reads a live counter, because a live counter describes the last event and not the run.
"""
import numpy as np


def lifetime_statistics(accepted, proposed, *, tau, exchange_stride=None, reservoir_events=None):
    """Aggregate the per-iteration history into lifetime figures.

    `accepted` and `proposed` are (iterations, states, states) and NOT cumulative. A non-exchange
    iteration is an all-zero row, which is what makes exchange attempts countable rather than
    inferred from the schedule.
    """
    accepted = np.asarray(accepted, dtype=np.int64)
    proposed = np.asarray(proposed, dtype=np.int64)
    if accepted.shape != proposed.shape:
        raise ValueError(f"accepted{accepted.shape} and proposed{proposed.shape} disagree")
    if accepted.ndim != 3 or accepted.shape[1] != accepted.shape[2]:
        raise ValueError(f"expected (iterations, states, states); got {accepted.shape}")

    n_iterations, n_states, _ = accepted.shape
    per_iteration = proposed.sum(axis=(1, 2))
    exchange_iterations = [int(i) for i in np.nonzero(per_iteration)[0]]

    total_proposed = proposed.sum(axis=0)
    total_accepted = accepted.sum(axis=0)

    pairs = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            # Both (i,j) and (j,i) are incremented, so either entry is the pair's count.
            n_proposed = int(total_proposed[i, j])
            n_accepted = int(total_accepted[i, j])
            pairs.append({
                "state_pair": [i, j],
                "adjacent": bool(j == i + 1),
                "tau_pair": [float(tau[i]), float(tau[j])],
                "proposed": n_proposed,
                "accepted": n_accepted,
                "acceptance": (n_accepted / n_proposed) if n_proposed else None,
            })

    schedule = None
    if exchange_stride:
        stride = int(exchange_stride)
        # Iteration k is segment k+1, so an exchange happens where (k+1) % stride == 0.
        expected = [k for k in range(n_iterations) if (k + 1) % stride == 0]
        missing = sorted(set(expected) - set(exchange_iterations))
        unexpected = sorted(set(exchange_iterations) - set(expected))
        schedule = {
            "exchange_stride_segments": stride,
            "expected_exchange_iterations": len(expected),
            "observed_exchange_iterations": len(exchange_iterations),
            "agrees_with_schedule": not missing and not unexpected,
            "scheduled_iterations_without_proposals": missing[:16],
            "unscheduled_iterations_with_proposals": unexpected[:16],
        }

    reservoir = None
    if reservoir_events is not None:
        events = np.asarray(reservoir_events, dtype=int)
        if events.size:
            attempted = events[:, 2] >= 0
            accepted_mask = events[:, 2] == 1
            used = events[attempted, 1]
            reservoir = {
                "attempts": int(attempted.sum()),
                "accepted": int(accepted_mask.sum()),
                "acceptance": (float(accepted_mask.sum() / attempted.sum())
                               if attempted.sum() else None),
                "states_refreshed": sorted({int(s) for s in events[attempted, 0]}),
                "distinct_frames_used": int(len(set(int(f) for f in used))) if used.size else 0,
                "frame_usage_counts": {int(f): int((used == f).sum())
                                       for f in sorted(set(int(x) for x in used))} if used.size
                else {},
                "note": ("a reservoir refresh replaces a configuration and is NOT a swap; it is "
                         "never counted in the pair statistics above, and it is not a "
                         "thermodynamic-state round trip"),
            }
        else:
            reservoir = {"attempts": 0, "accepted": 0, "acceptance": None,
                         "states_refreshed": [], "distinct_frames_used": 0,
                         "frame_usage_counts": {}}

    return {
        "basis": "lifetime, summed over every committed iteration in the analysis NetCDF",
        "iteration_range": [0, int(n_iterations - 1)] if n_iterations else [],
        "iterations_committed": int(n_iterations),
        "exchange_iterations": len(exchange_iterations),
        "total_proposed": int(total_proposed[~np.eye(n_states, dtype=bool)].sum() // 2),
        "total_accepted": int(total_accepted[~np.eye(n_states, dtype=bool)].sum() // 2),
        "overall_acceptance": (
            float(total_accepted[~np.eye(n_states, dtype=bool)].sum()
                  / total_proposed[~np.eye(n_states, dtype=bool)].sum())
            if total_proposed[~np.eye(n_states, dtype=bool)].sum() else None),
        "by_state_pair": pairs,
        "schedule": schedule,
        "reservoir": reservoir,
    }


def adjacent_pairs(statistics):
    return [pair for pair in statistics["by_state_pair"] if pair["adjacent"]]


def count_round_trips(series, *, cold_state, hot_state):
    """Complete round trips in one walker's state history.

        cold visited  ->  hot visited later  ->  cold visited later still

    The starting position earns nothing. A walker that BEGINS at the hot state has not completed a
    round trip when it first reaches cold: it has completed half of one, and must then return to
    hot and come back.
    """
    if cold_state == hot_state:
        raise ValueError(f"cold and hot states must differ; both are {cold_state}")
    phase, completed = "seeking_cold", 0
    for value in np.asarray(series).tolist():
        value = int(value)
        if value == cold_state:
            if phase == "at_hot":
                completed += 1
            phase = "at_cold"
        elif value == hot_state:
            if phase == "at_cold":
                phase = "at_hot"
    return completed


def walker_view(mapping):
    """`walker_to_state[iteration][walker]`, the exact inverse of the stored state->walker map."""
    mapping = np.asarray(mapping, dtype=int)
    inverse = np.empty_like(mapping)
    for row_index, row in enumerate(mapping):
        for state_index, walker in enumerate(row):
            inverse[row_index, int(walker)] = state_index
    return inverse


def mapping_is_permutation_every_iteration(mapping, *, n_states=None):
    mapping = np.asarray(mapping, dtype=int)
    if mapping.size == 0:
        return True, []
    states = int(mapping.max()) + 1 if n_states is None else int(n_states)
    expected = list(range(states))
    bad = [int(i) for i, row in enumerate(mapping) if sorted(int(v) for v in row) != expected]
    return (not bad), bad[:16]


def round_trip_report(mapping, *, n_states):
    """Round trips per walker, from the WALKER view of the stored mapping."""
    mapping = np.asarray(mapping, dtype=int)
    ok, offending = mapping_is_permutation_every_iteration(mapping, n_states=n_states)
    cold, hot = 0, int(n_states) - 1
    walkers = []
    if mapping.size:
        inverse = walker_view(mapping)
        for walker in range(int(n_states)):
            series = inverse[:, walker]
            visited = sorted({int(v) for v in series.tolist()})
            walkers.append({
                "walker": walker,
                "started_at_state": int(series[0]),
                "round_trips": count_round_trips(series, cold_state=cold, hot_state=hot),
                "visited_cold": cold in visited,
                "visited_hot": hot in visited,
                "visited_all_states": visited == list(range(int(n_states))),
            })
    return {
        "definition": ("cold state visited, then the hot state later, then the cold state again. "
                       "The starting position earns nothing, and a reservoir refresh is not a "
                       "round trip."),
        "cold_state": cold, "hot_state": hot,
        "mapping_rows": int(mapping.shape[0]) if mapping.size else 0,
        "every_row_is_a_permutation": bool(ok),
        "offending_rows": offending,
        "by_walker": walkers,
    }
