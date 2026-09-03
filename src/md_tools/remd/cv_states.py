"""Collective-variable series for a REST2/rREST2 ladder: one file per THERMODYNAMIC STATE.

WHY PER STATE AND NOT PER WALKER

    The same reason the trajectories are per state. A ladder's result is a property of a rung --
    "the distribution sampled at tau = 0.3" -- and a walker visits many rungs. A per-walker series
    is a series over a changing Hamiltonian, which is not an ensemble average of anything.

    So `remd2.cv.csv` holds the configurations that OCCUPIED state 2, whichever walker supplied
    each of them, and `walker_index` records which one did. That column is what makes the
    exchange history reconstructible from the CV files alone, and what lets someone check the
    series against a walker-centric analysis if they want one.

THE EXCHANGE-BOUNDARY CONVENTION

    PRE-EXCHANGE, everywhere, stated in every sidecar. A row landing on an exchange boundary
    describes the configuration the walker actually propagated to that step, taken before any
    swap is applied.

    Post-exchange would report, against that step, a configuration that arrived from another rung
    and was never integrated at this one -- so a state's series would contain values from
    trajectories that never visited it at that step. The ordering is enforced structurally rather
    than by comment: `cv` precedes `exchange` in `EVENT_ORDER`, so the driver cannot write the row
    late without changing that tuple.
"""

from __future__ import annotations

from pathlib import Path

from ..cv import CVSeries

#: The leading columns, before the collective variables themselves.
COLUMNS = ("step", "time_ps", "exchange_attempt", "state_index", "tau", "walker_index",
           "exchange_phase", "trajectory_frame_index")

#: The one value `exchange_phase` ever takes in this version. Written as a column rather than
#: assumed, so a reader never has to know the convention to use the file, and so a future
#: post-exchange row would be distinguishable rather than silently mixed in.
PHASE = "pre-exchange"


class StateCVSet:
    """One `CVSeries` per thermodynamic state, opened and closed together."""

    def __init__(self, directory, definition, *, taus, interval_steps, fingerprint=None):
        self.directory = Path(directory)
        self.definition = definition
        self.taus = tuple(float(t) for t in taus)
        self.interval_steps = int(interval_steps)
        self.series = []
        for index, tau in enumerate(self.taus):
            self.series.append(CVSeries(
                self.directory / f"remd{index}.cv.csv", definition,
                extra_columns=COLUMNS,
                sidecar_extra={
                    "state_index": index, "tau": tau,
                    "interval_steps": self.interval_steps,
                    "exchange_phase": PHASE,
                    "exchange_phase_meaning": (
                        "rows on an exchange boundary describe the configuration as propagated, "
                        "before any swap was applied"),
                    "series_follows": "thermodynamic state",
                    "walker_index_meaning": (
                        "which walker supplied the configuration occupying this state at this "
                        "step"),
                    **({"fingerprint": fingerprint} if fingerprint else {})}))

    def open(self, *, committed_rows=0):
        for series in self.series:
            series.open(append_from=int(committed_rows))
        return self

    def observe(self, *, step, time_ps, exchange_attempt, state_to_walker, configurations,
                frame_index=None):
        """One row in every state's file, from the configuration currently occupying it."""
        for state_index, series in enumerate(self.series):
            walker = int(state_to_walker[state_index])
            configuration = configurations[walker]
            values = series.evaluate(configuration.positions, configuration.box)
            series.write((step, time_ps, exchange_attempt, state_index, self.taus[state_index],
                          walker, PHASE, frame_index), values)

    def rows_written(self) -> int:
        """Every state's file holds the same number of rows, so one number describes the set."""
        counts = {series.rows_written for series in self.series}
        if len(counts) > 1:
            raise RuntimeError(
                f"the per-state collective-variable files hold different row counts ({counts}); "
                f"they are written together and must stay in step")
        return counts.pop() if counts else 0

    def cost(self):
        total = {"cv_evaluations": 0, "cv_seconds": 0.0, "cv_rows": 0}
        for series in self.series:
            one = series.cost()
            total["cv_evaluations"] += one["cv_evaluations"]
            total["cv_seconds"] += one["cv_seconds"]
            total["cv_rows"] += one["cv_rows"]
        total["cv_seconds"] = round(total["cv_seconds"], 6)
        return total

    def close(self):
        for series in self.series:
            series.close()
