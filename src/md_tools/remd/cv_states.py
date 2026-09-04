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

    def open(self, *, committed_rows=0, committed=None):
        """Open every state's series, restoring rows AND cost together.

        `committed` is the typed per-state prefix; `committed_rows` remains for the fresh-run
        call that has no cost to restore. A caller cannot restore one without the other.
        """
        for index, series in enumerate(self.series):
            if committed is not None:
                series.open(committed=committed[index])
            else:
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

    def segment_cost(self):
        """This invocation's cost, SUMMED OVER STATES. Per-state records are kept below."""
        from ..cv.cost import CVCost

        total = CVCost()
        for series in self.series:
            total = total.plus(series.segment_cost())
        return total

    def cumulative_cost(self):
        from ..cv.cost import CVCost

        total = CVCost()
        for series in self.series:
            total = total.plus(series.cumulative_cost())
        return total

    def cost(self):
        """The ladder's two-scope record.

        The headline figures are the SUM over states, and the per-state records are retained
        beside them so the total is auditable rather than a number a reader has to trust. A
        ladder's cost is genuinely the sum: every state evaluates the same definition on its own
        configuration at every observation step.
        """
        from ..cv.cost import cost_record

        record = cost_record(self.segment_cost(), self.cumulative_cost(),
                             rows=sum(series.rows_written for series in self.series))
        record["aggregation"] = "sum over thermodynamic states"
        record["per_state"] = [
            {"state_index": index, "tau": self.taus[index], **series.cost()}
            for index, series in enumerate(self.series)]
        return record

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


def manifest_entries(directory, definition, *, taus, interval_steps, total_steps, cost=None):
    """What the completion manifest records about every state's CV series. READ FROM DISK.

    Read from the files rather than from the writer's in-memory counters on purpose: the manifest
    has to describe what is actually on disk at the moment completion is claimed, which is the
    only thing a later reader can check it against. A count carried out of the writer would agree
    with itself after a partial flush.

    `None` when this run reported no collective variables -- and that is recorded explicitly
    rather than by omission, so a reader can tell "this run had none" from "this manifest predates
    the field".
    """
    directory = Path(directory)
    entries = []
    for index, tau in enumerate(taus):
        csv_path = directory / f"remd{index}.cv.csv"
        sidecar = directory / f"remd{index}.cv.json"
        lines = csv_path.read_text(encoding="utf-8").splitlines() if csv_path.is_file() else []
        rows = [line.split(",") for line in lines[1:]]
        steps = [int(row[0]) for row in rows] if rows else []
        entries.append({
            "state_index": index,
            "tau": float(tau),
            "csv": csv_path.name,
            "csv_sha256": _digest(csv_path),
            "csv_bytes": csv_path.stat().st_size if csv_path.is_file() else None,
            "rows": len(rows),
            "header": lines[0] if lines else None,
            "first_step": steps[0] if steps else None,
            "final_step": steps[-1] if steps else None,
            "interval_steps": int(interval_steps),
            "expected_steps": list(range(0, int(total_steps) + 1, int(interval_steps))),
            "sidecar": sidecar.name,
            "sidecar_sha256": _digest(sidecar),
            "sidecar_bytes": sidecar.stat().st_size if sidecar.is_file() else None,
            "schema_version": definition.schema_version,
            "definition_sha256": definition.digest,
            "atom_indices": [list(cv.indices) for cv in definition.variables],
            "columns": list(COLUMNS) + list(definition.names),
            "units": "degrees",
            "wrapping": "[-180, 180)",
            "periodic_convention": (
                "triclinic minimum image applied to the three sequential bond vectors"),
            "exchange_phase": PHASE,
        })
    record = {"series": entries, "interval_steps": int(interval_steps),
              "definition_sha256": definition.digest}
    if cost:
        record["cost"] = dict(cost)
    return record


def _digest(path):
    import hashlib

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_entries(directory, record):
    """Every reason a completed ladder's CV series may not be believed. Returns a list of strings.

    Called before completion is committed AND whenever a completed run is verified afterwards, so
    a file edited after the fact is caught by the same rules that let it be written.

    Structure as well as digests: a digest catches any change at all, but says nothing about
    WHAT is wrong, and a reader who has only a mismatch cannot tell a truncation from a mutated
    value. The structural checks below name the fault.
    """
    directory = Path(directory)
    problems: list[str] = []
    for entry in (record or {}).get("series") or []:
        index = entry.get("state_index")
        csv_path = directory / str(entry.get("csv"))
        sidecar = directory / str(entry.get("sidecar"))

        if not csv_path.is_file():
            problems.append(f"state {index}: {csv_path.name} is missing")
            continue
        if not sidecar.is_file():
            problems.append(f"state {index}: {sidecar.name} is missing")
            continue
        if _digest(csv_path) != entry.get("csv_sha256"):
            problems.append(
                f"state {index}: {csv_path.name} has changed since the run finished "
                f"(sha256 does not match the manifest)")
        if _digest(sidecar) != entry.get("sidecar_sha256"):
            problems.append(
                f"state {index}: {sidecar.name} has changed since the run finished")

        lines = csv_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            problems.append(f"state {index}: {csv_path.name} is empty")
            continue
        if entry.get("header") is not None and lines[0] != entry["header"]:
            problems.append(f"state {index}: {csv_path.name} has a different header")
            continue
        rows = [line.split(",") for line in lines[1:]]
        if len(rows) != int(entry.get("rows") or 0):
            problems.append(
                f"state {index}: {csv_path.name} holds {len(rows)} row(s) and the manifest "
                f"records {entry.get('rows')}")
        expected = [int(step) for step in (entry.get("expected_steps") or [])]
        actual = [int(row[0]) for row in rows]
        if expected and actual != expected:
            # Name WHERE they diverge, not the first six of each. Truncating a series leaves two
            # identical prefixes, and printing them side by side showed the reader two matching
            # lists above the words "does not match".
            missing = [step for step in expected if step not in set(actual)]
            extra = [step for step in actual if step not in set(expected)]
            detail = []
            if missing:
                detail.append(f"missing {missing[:6]}")
            if extra:
                detail.append(f"unexpected {extra[:6]}")
            if not detail:
                detail.append(f"out of order: {actual[:6]}...")
            problems.append(
                f"state {index}: the step grid is wrong ({'; '.join(detail)}). It must be "
                f"0, {entry.get('interval_steps')}, ..., {expected[-1]} exactly once")
        columns = entry.get("columns") or []
        for position, row in enumerate(rows):
            if len(row) != len(columns):
                problems.append(
                    f"state {index}: row {position} has {len(row)} field(s), not {len(columns)}")
                break
            if int(row[3]) != int(index):
                problems.append(
                    f"state {index}: row {position} reports state_index {row[3]}")
                break
            if abs(float(row[4]) - float(entry.get("tau"))) > 1e-12:
                problems.append(f"state {index}: row {position} reports tau {row[4]}")
                break
            if row[6] != PHASE:
                problems.append(
                    f"state {index}: row {position} reports exchange phase {row[6]!r}")
                break
            named = row[7].strip()
            if named and (not named.lstrip("-").isdigit() or int(named) < 0):
                problems.append(
                    f"state {index}: row {position} names trajectory frame {named!r}")
                break
            for value in row[len(COLUMNS):]:
                number = float(value)
                if number != number or number in (float("inf"), float("-inf")):
                    problems.append(
                        f"state {index}: row {position} carries a non-finite value {value!r}")
                    break
    return problems


def prefix_records(directory, definition, *, taus, interval_steps, rows, cost=None,
                   per_state_cost=None):
    """What a ladder checkpoint generation stores about its CV series: one prefix per state.

    Per state rather than one aggregate digest, because the states are separate files and a
    single combined hash could not say WHICH of them changed -- and a swapped pair of files
    would leave a combined hash unchanged while every state's series became another's.
    """
    from ..cv import prefix as cv_prefix

    directory = Path(directory)
    self_costs = list(per_state_cost) if per_state_cost else None
    entries = []
    for index, tau in enumerate(taus):
        csv_path = directory / f"remd{index}.cv.csv"
        entry = cv_prefix.record(
            csv_path, rows=int(rows), sidecar=directory / f"remd{index}.cv.json",
            definition=definition)
        entry["state_index"] = index
        entry["tau"] = float(tau)
        # The cost that produced THIS state's committed rows, so a continuation restores rows and
        # counters together rather than one without the other.
        entry["cost"] = self_costs[index] if self_costs else None
        entries.append(entry)
    record = {"rows": int(rows), "interval_steps": int(interval_steps), "states": entries}
    if cost:
        record["cost"] = dict(cost)
    return record


def validate_prefixes(directory, definition, *, taus, interval_steps, block):
    """Validate every state's committed prefix. Returns the committed row count, or raises.

    Raised as `CVContinuationError` so a caller distinguishes "this cannot be continued" from a
    programming fault, and so the message reaches the operator unchanged.
    """
    from ..cv import prefix as cv_prefix

    if not block or block.get("rows") is None or not block.get("states"):
        raise CVContinuationError(
            "this run's checkpoint does not record a committed collective-variable prefix (per "
            "state row count and digest). It was written by a build that reported CVs without "
            "binding them into the checkpoint transaction, so which rows are durable cannot be "
            "established and a continuation would either duplicate observations or silently keep "
            "rows that were edited after the last commit. Start a fresh run with --overwrite.")

    directory = Path(directory)
    columns = list(COLUMNS) + list(definition.names)
    rows = int(block["rows"])
    for entry in block["states"]:
        index = int(entry["state_index"])
        try:
            cv_prefix.validate(
                directory / f"remd{index}.cv.csv", entry,
                sidecar=directory / f"remd{index}.cv.json",
                definition=definition, expect_columns=columns,
                value_columns=list(definition.names),
                identifiers={"state_index": index, "tau": float(entry["tau"]),
                             "exchange_phase": PHASE},
                interval=int(interval_steps), step_column="step")
        except cv_prefix.CVPrefixError as refusal:
            raise CVContinuationError(str(refusal)) from None
    return rows


def truncate_to(directory, *, taus, rows):
    """Cut every state's series back to the committed prefix. Only after validation."""
    from ..cv import prefix as cv_prefix

    directory = Path(directory)
    for index in range(len(taus)):
        cv_prefix.truncate(directory / f"remd{index}.cv.csv", int(rows))


def committed_prefixes(block, n_states):
    """The typed per-state prefix from a checkpoint block: rows AND restored cumulative cost."""
    from ..cv.cost import CommittedPrefix

    entries = {int(e["state_index"]): e for e in (block or {}).get("states", [])}
    rows = int((block or {}).get("rows", 0))
    return [CommittedPrefix.from_record({"rows": rows,
                                         "cost": (entries.get(index) or {}).get("cost")})
            for index in range(int(n_states))]
