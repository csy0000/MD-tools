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

import json

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
                frame_index=None, frame_index_for_state=None):
        """One row in every state's file, from the configuration occupying it PRE-EXCHANGE.

        `frame_index_for_state` is a per-state sequence, and it exists because the answer differs
        between states at the same step. The state trajectory frame is written AFTER the exchange
        -- that is the established trajectory convention, and it is not changed here -- while the
        CV row is measured BEFORE it. For a state whose walker the exchange did not move, those
        are the same configuration and the frame may be named. For a state whose walker WAS
        swapped, they are different configurations, and naming the frame would attribute a value
        to coordinates it was not measured on.

        `frame_index` remains for the steps where the answer is the same for every state.
        """
        for state_index, series in enumerate(self.series):
            walker = int(state_to_walker[state_index])
            configuration = configurations[walker]
            values = series.evaluate(configuration.positions, configuration.box)
            named = (frame_index if frame_index_for_state is None
                     else frame_index_for_state[state_index])
            series.write((step, time_ps, exchange_attempt, state_index, self.taus[state_index],
                          walker, PHASE, named), values)

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


class CVContinuationError(RuntimeError):
    """A CV series that cannot be continued. Refused rather than appended to."""


def validate_for_continuation(directory, definition, *, taus, interval_steps, committed_rows):
    """Every reason this ladder's CV series may not be continued. READ-ONLY.

    Called before anything is opened for writing, because a continuation that has already
    truncated a file cannot decide afterwards that it should have refused.

    The checks are the ones that make appending meaningful at all:

      the files EXIST, one per state, with the sidecars that say how to read them. A ladder
      continued into a directory missing half its series would silently produce a set where some
      states cover the whole run and others only its tail;

      the committed count is KNOWN. A checkpoint written by a build that did not record
      `cv_rows` cannot say which rows are durable, and guessing -- from the file length, say --
      is exactly the guess the count exists to avoid: rows written after the last commit are the
      ones a resume must discard. That refuses with a compatibility message rather than a
      traceback;

      every file HOLDS AT LEAST the committed count. Fewer rows than committed means records the
      checkpoint believes exist have been lost, which is data loss, not a resumable state;

      the DEFINITION still resolves to the same atoms, in the same order, under the same units,
      wrapping, periodic and sign conventions, at the same interval. Any of those changing makes
      the appended rows a different measurement sharing a column heading with the old ones, and
      nothing in the file would show where one ended.
    """
    directory = Path(directory)
    problems: list[str] = []

    if committed_rows is None:
        raise CVContinuationError(
            "this run's checkpoint does not record how many collective-variable rows were "
            "committed. It was written by a build that reported CVs without binding their row "
            "count into the checkpoint transaction, so which rows are durable cannot be "
            "established and a continuation would either duplicate or drop observations. Start a "
            "fresh run with --overwrite, or continue it with the build that wrote it and no CV "
            "reporting enabled.")
    committed_rows = int(committed_rows)

    expected_columns = list(COLUMNS) + list(definition.names)
    for index, tau in enumerate(taus):
        csv_path = directory / f"remd{index}.cv.csv"
        sidecar = directory / f"remd{index}.cv.json"
        if not csv_path.is_file():
            problems.append(f"{csv_path.name} is missing")
            continue
        if not sidecar.is_file():
            problems.append(f"{sidecar.name} is missing, so {csv_path.name} cannot be interpreted")
            continue

        lines = csv_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            problems.append(f"{csv_path.name} is empty")
            continue
        header = lines[0].split(",")
        if header != expected_columns:
            problems.append(
                f"{csv_path.name} has columns {header}, and this run writes {expected_columns}. "
                f"The definition, its order or the schema has changed")
            continue
        rows = len(lines) - 1
        if rows < committed_rows:
            problems.append(
                f"{csv_path.name} holds {rows} row(s) and the checkpoint vouches for "
                f"{committed_rows}. Records the checkpoint believes exist have been lost")

        try:
            body = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError as broken:
            problems.append(f"{sidecar.name} is not readable JSON ({broken})")
            continue

        for field, expected, what in (
                ("definition_sha256", definition.digest, "definition digest"),
                ("interval_steps", int(interval_steps), "reporting interval"),
                ("state_index", index, "state index"),
                ("units", "degrees", "units"),
                ("wrapping", "[-180, 180)", "wrapping convention"),
                ("exchange_phase", PHASE, "exchange-boundary convention")):
            if body.get(field) != expected:
                problems.append(
                    f"{sidecar.name} records {what} {body.get(field)!r} and this run resolves "
                    f"{expected!r}. Appending would put a different measurement under the same "
                    f"column heading")
        if abs(float(body.get("tau", float("nan"))) - float(tau)) > 1e-12:
            problems.append(
                f"{sidecar.name} records tau {body.get('tau')} for state {index} and this ladder "
                f"resolves {tau}")
        recorded = [tuple(cv.get("atom_indices") or ()) for cv in
                    (body.get("collective_variables") or [])]
        resolved = [tuple(cv.indices) for cv in definition.variables]
        if recorded != resolved:
            problems.append(
                f"{sidecar.name} resolved atoms {recorded} and this run resolves {resolved}. The "
                f"same names now name different atoms")

    if problems:
        raise CVContinuationError(
            "the collective-variable series cannot be continued:\n  - "
            + "\n  - ".join(problems))
    return committed_rows
