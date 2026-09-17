#!/usr/bin/env python
"""Lifetime statistics and round trips, from the authoritative stored history.

Copied verbatim into every generated replica project.

Every figure here is summed over the WHOLE stored history, so it survives an interrupted resume and
an extension unchanged: the storage is authoritative, not the process that happened to write it.
Nothing reads a live counter, because a live counter describes the last event and not the run.
"""
import numpy as np


def lifetime_statistics(accepted, proposed, *, tau):
    """Aggregate the stored exchange history into lifetime figures.

    `accepted` and `proposed` are (exchanges, states, states) and NOT cumulative. Every stored row
    IS an exchange attempt now -- the schedules are independent, so there are no zero rows standing
    for skipped propagation and nothing has to be inferred from a stride.
    """
    accepted = np.asarray(accepted, dtype=np.int64)
    proposed = np.asarray(proposed, dtype=np.int64)
    if accepted.shape != proposed.shape:
        raise ValueError(f"accepted{accepted.shape} and proposed{proposed.shape} disagree")
    if accepted.ndim != 3 or accepted.shape[1] != accepted.shape[2]:
        raise ValueError(f"expected (iterations, states, states); got {accepted.shape}")

    n_exchanges, n_states, _ = accepted.shape
    per_row = proposed.sum(axis=(1, 2))
    rows_with_proposals = [int(i) for i in np.nonzero(per_row)[0]]

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

    return {
        "basis": "lifetime, summed over every committed exchange row in the analysis NetCDF",
        "exchange_range": [0, int(n_exchanges - 1)] if n_exchanges else [],
        "exchanges_committed": int(n_exchanges),
        "rows_with_proposals": len(rows_with_proposals),
        "total_proposed": int(total_proposed[~np.eye(n_states, dtype=bool)].sum() // 2),
        "total_accepted": int(total_accepted[~np.eye(n_states, dtype=bool)].sum() // 2),
        "overall_acceptance": (
            float(total_accepted[~np.eye(n_states, dtype=bool)].sum()
                  / total_proposed[~np.eye(n_states, dtype=bool)].sum())
            if total_proposed[~np.eye(n_states, dtype=bool)].sum() else None),
        "by_state_pair": pairs,
    }


def adjacent_pairs(statistics):
    return [pair for pair in statistics["by_state_pair"] if pair["adjacent"]]


def completion_report(statistics):
    """The completion report: neighbouring-pair acceptance, an overall figure, and nothing else.

    One builder so the printed report and the persisted one cannot drift apart -- the summary that
    reaches the terminal is rendered from this same structure.

    `overall` is summed over the neighbouring pairs only. That is the whole of what was proposed
    under a neighbour-only rule, so it agrees with `overall_acceptance` there; under any rule that
    proposes a non-neighbouring swap it would not, and this figure is the one that matches the
    table printed above it.
    """
    pairs = []
    accepted = proposed = 0
    for pair in adjacent_pairs(statistics):
        pairs.append({
            "state_pair": list(pair["state_pair"]),
            "tau_pair": list(pair["tau_pair"]),
            "accepted": int(pair["accepted"]),
            "proposed": int(pair["proposed"]),
            "acceptance": pair["acceptance"],
        })
        accepted += int(pair["accepted"])
        proposed += int(pair["proposed"])
    report = {
        "basis": ("cumulative over every committed exchange row in the authoritative NetCDF, "
                  "exchanges {}-{}".format(*statistics["exchange_range"])
                  if statistics["exchange_range"] else "no committed exchange rows"),
        "by_neighbouring_pair": pairs,
        "overall": {"accepted": accepted, "proposed": proposed,
                    "acceptance": (accepted / proposed) if proposed else None},
    }
    return report


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


# Round trips, transition matrices, first-passage times and convergence diagnostics are
# deliberately NOT here. They are downstream analysis: they need a burn-in choice and a window
# choice that this layer has no basis to make, and a single number computed here would end up
# quoted as if the choice had been justified. The committed mapping is preserved in the NetCDF,
# so any of them can be computed later from the authoritative record.
