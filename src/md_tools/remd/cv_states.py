"""Collective-variable series for a REST2 ladder: one file per THERMODYNAMIC STATE.

WHY PER STATE AND NOT PER WALKER

    The same reason the trajectories are per state. A ladder's result is a property of a rung --
    "the distribution sampled at tau = 0.3" -- and a walker visits many rungs. A per-walker series
    is a series over a changing Hamiltonian, which is not an ensemble average of anything.

    So `cv_state2.csv` holds the configurations that OCCUPIED state 2, whichever walker supplied
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

from dataclasses import dataclass
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
                self.directory / f"cv_state{index}.csv", definition,
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


@dataclass(frozen=True)
class ValidatedLadderPrefix:
    """What a checkpoint's CV block committed, once every field of it has been checked.

    Returned by `validate_prefixes` and consumed by both the truncation and the restoration, so
    the numbers that cut the files are the same numbers that were validated. Passing the raw
    block on instead is how the two came to disagree.
    """

    rows: int
    per_state: tuple


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
        csv_path = directory / f"cv_state{index}.csv"
        sidecar = directory / f"cv_state{index}.json"
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
        csv_path = directory / f"cv_state{index}.csv"
        sidecar = directory / f"cv_state{index}.json"
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
    entries = list((record or {}).get("series") or [])
    n_states = len(entries)
    #: state index -> {observation step: walker}, filled only for series that parsed cleanly.
    #: The permutation check below runs across whatever this collected: a state whose file is
    #: missing or malformed has already been reported by name, and re-reporting it as a broken
    #: permutation would bury the real fault under a derived one.
    occupancy: dict[int, dict[int, int]] = {}
    for entry in entries:
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
        mine: dict[int, int] = {}
        intact = True
        for position, row in enumerate(rows):
            if len(row) != len(columns):
                problems.append(
                    f"state {index}: row {position} has {len(row)} field(s), not {len(columns)}")
                intact = False
                break
            if int(row[3]) != int(index):
                problems.append(
                    f"state {index}: row {position} reports state_index {row[3]}")
                intact = False
                break
            if abs(float(row[4]) - float(entry.get("tau"))) > 1e-12:
                problems.append(f"state {index}: row {position} reports tau {row[4]}")
                intact = False
                break
            if row[6] != PHASE:
                problems.append(
                    f"state {index}: row {position} reports exchange phase {row[6]!r}")
                intact = False
                break
            # THE WALKER. Which walker supplied the configuration occupying this rung is what
            # makes the series joinable to a walker-centric analysis, and nothing checked it:
            # a walker column of -1, of `n_states`, of "2.5", or one naming the same walker in
            # two rungs at once read back perfectly and meant a ladder that never existed.
            walker = row[5].strip()
            if not walker.lstrip("-").isdigit():
                problems.append(
                    f"state {index}: row {position} reports walker {row[5]!r}, which is not an "
                    f"integer. A walker is an identity, not a measurement")
                intact = False
                break
            walker = int(walker)
            if not 0 <= walker < n_states:
                problems.append(
                    f"state {index}: row {position} reports walker {walker}, and this ladder has "
                    f"{n_states} walker(s), numbered 0 to {n_states - 1}")
                intact = False
                break
            mine[int(row[0])] = walker
            named = row[7].strip()
            if named and (not named.lstrip("-").isdigit() or int(named) < 0):
                problems.append(
                    f"state {index}: row {position} names trajectory frame {named!r}")
                intact = False
                break
            for value in row[len(COLUMNS):]:
                number = float(value)
                if number != number or number in (float("inf"), float("-inf")):
                    problems.append(
                        f"state {index}: row {position} carries a non-finite value {value!r}")
                    intact = False
                    break
        if intact and isinstance(index, int):
            occupancy[index] = mine

    problems.extend(_permutation_problems(occupancy, n_states))
    problems.extend(_cost_problems(record, entries))
    return problems


def _cost_problems(record, entries):
    """The stored cost, through the one strict parser rather than a reader of its own.

    A ladder's aggregate claims to be the sum over its states, and the per-state records beside
    it are what make that auditable. Neither was checked when a completed run was verified, so a
    manifest could carry an aggregate that no set of per-state entries adds up to and still pass
    every other check in this file.
    """
    from ..cv.cost import CVCostError, parse_aggregate_record, parse_cost_record

    block = (record or {}).get("cost")
    if block is None:
        # AN ABSENT COST IS ONLY ACCEPTABLE WHEN THERE IS NOTHING IT COULD DESCRIBE.
        #
        # This returned unconditionally, which made the two halves of the same rule disagree: a
        # committed PREFIX with no cost record is refused by name (`cv/prefix.py`, "carries no
        # usable cost record"), and a completion MANIFEST with none passed every check in this
        # file. The strict parsers already refuse `None` -- both answer "no collective-variable
        # cost record" -- so the omission was here, in the early return that never reached them.
        #
        # A run with CV reporting off has no series and no cost, and that is correct rather than
        # missing. A run that recorded series for N states and no cost is a record with a hole in
        # it: the evaluations happened, something wrote the rows, and what they cost is gone.
        if entries:
            return [f"this manifest records collective-variable series for {len(entries)} "
                    f"state(s) and carries no cost record at all. A committed prefix without one "
                    f"is refused; a completed run without one was not, which left the same "
                    f"omission reported in one place and silent in the other. If this run "
                    f"genuinely evaluated no collective variables it should carry no series "
                    f"either."]
        return []
    problems: list[str] = []
    rows_by_state = {}
    for entry in entries:
        index = entry.get("state_index")
        columns = entry.get("columns") or []
        n_cv = max(0, len(columns) - len(COLUMNS)) or None
        rows_by_state[index] = (entry.get("rows"), n_cv)
    for one in (block.get("per_state") or []):
        index = one.get("state_index")
        rows, n_cv = rows_by_state.get(index, (None, None))
        try:
            parse_cost_record(one, where=f"restart.json cost per_state state {index}",
                              rows=rows, n_cv=n_cv)
        except CVCostError as refusal:
            problems.append(str(refusal))
    try:
        parse_aggregate_record(block, where="restart.json collective-variable cost",
                               entries_key="per_state", identity_key="state_index",
                               expected_identities=sorted(rows_by_state))
    except CVCostError as refusal:
        problems.append(str(refusal))
    return problems


def _permutation_problems(occupancy, n_states):
    """Every step at which the ladder's walkers are not a permutation of 0 .. n_states - 1.

    An exchange PERMUTES walkers among rungs; it never creates, destroys or duplicates one. So at
    any step every rung is occupied, and by a different walker. Each series on its own can satisfy
    every other check in this file and still be wrong in a way only the SET reveals: two states
    both claiming walker 1 at one step, or a walker that occupies no rung at all at one step.
    Neither is visible from inside a single file, where every row is internally consistent, on the
    right step, at the right tau, and carries a walker in range.

    (A whole-file swap is a different fault and is already refused upstream -- the swapped rows
    report the other state's `state_index`, and the digests no longer match the manifest. This
    check is for the cases that survive all of that.)

    Reported per step, with the walkers seen, because "not a permutation" without the offending
    assignment sends the reader back to the CSVs to work out which rung was wrong.
    """
    if not occupancy or len(occupancy) != int(n_states):
        # Not every state parsed. Those failures are already reported by name; a permutation
        # complaint derived from a partial set would be noise.
        return []
    problems: list[str] = []
    steps = sorted({step for mine in occupancy.values() for step in mine})
    wanted = set(range(int(n_states)))
    for step in steps:
        seen = {index: mine[step] for index, mine in sorted(occupancy.items()) if step in mine}
        if len(seen) != int(n_states):
            absent = sorted(set(range(int(n_states))) - set(seen))
            problems.append(
                f"step {step}: state(s) {absent} record no observation, and every state observes "
                f"at every step on the reporting interval")
            continue
        if set(seen.values()) != wanted:
            duplicated = sorted({w for w in seen.values()
                                 if list(seen.values()).count(w) > 1})
            missing = sorted(wanted - set(seen.values()))
            detail = []
            if duplicated:
                detail.append(f"walker(s) {duplicated} occupy more than one state")
            if missing:
                detail.append(f"walker(s) {missing} occupy none")
            problems.append(
                f"step {step}: the walkers across the ladder are {seen} "
                f"(state -> walker), which is not a permutation of 0 to {int(n_states) - 1} "
                f"({'; '.join(detail)})")
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
        csv_path = directory / f"cv_state{index}.csv"
        entry = cv_prefix.record(
            csv_path, rows=int(rows), sidecar=directory / f"cv_state{index}.json",
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


def validate_prefixes(directory, definition, *, taus, interval_steps, block,
                      committed_step=None, committed_rows=None):
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

    from ..cv.cost import (CVCostError, CommittedPrefix, parse_aggregate_record,
                           parse_cost_record, require_count)

    directory = Path(directory)
    columns = list(COLUMNS) + list(definition.names)
    where = "the checkpoint's collective-variable block"

    # 1. THE BLOCK'S OWN ROW COUNT, strictly. This went through `int()`, so 1.9 became 1 and the
    #    continuation truncated every state's series to a length nobody had committed -- silently,
    #    because the number it truncated to was a perfectly ordinary integer by then.
    try:
        rows = require_count(block["rows"], where=where, scope="block", field="rows")
    except CVCostError as refusal:
        raise CVContinuationError(str(refusal)) from None

    # 2. THE STATE SET, before any dictionary is built from it. Collapsing the entries into a
    #    mapping keyed by `int(state_index)` made a list holding state 0 twice and omitting
    #    state 1 look complete: the duplicate overwrote, and the missing state fell through to an
    #    empty record that restored zero. Expected identities come from the PROTOCOL -- the taus
    #    this run resolved -- not from the record being checked, which cannot vouch for itself.
    entries = block["states"]
    if not isinstance(entries, list):
        raise CVContinuationError(
            f"{where}: `states` must be a list, got {type(entries).__name__}")
    expected = list(range(len(taus)))
    seen: dict[int, dict] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CVContinuationError(f"{where}: states[{position}] must be a mapping")
        try:
            index = require_count(entry.get("state_index"), where=where,
                                  scope=f"states[{position}]", field="state_index")
        except CVCostError as refusal:
            raise CVContinuationError(str(refusal)) from None
        if index in seen:
            raise CVContinuationError(
                f"{where}: state {index} appears more than once. Every thermodynamic state "
                f"commits exactly one prefix, so a duplicate means one state's rows would be "
                f"restored twice and another's not at all.")
        if index not in expected:
            raise CVContinuationError(
                f"{where}: state {index} is not one of this ladder's states {expected}")
        seen[index] = entry
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise CVContinuationError(
            f"{where}: no committed prefix for state(s) {missing}. This ladder resolves "
            f"{len(expected)} state(s) and every one of them commits a prefix; a state without "
            f"one would silently restore zero rows and zero cost.")

    # 3. EACH STATE, against the tau the protocol resolved for it and against the file and
    #    sidecar that state owns. `cv_prefix.validate` carries the digest, columns, cadence,
    #    finiteness, identifiers and -- now -- the required cost.
    prefixes: list[CommittedPrefix] = []
    for index in expected:
        entry = seen[index]
        tau = float(taus[index])
        recorded_tau = entry.get("tau")
        if not isinstance(recorded_tau, (int, float)) or isinstance(recorded_tau, bool) \
                or abs(float(recorded_tau) - tau) > 1e-12:
            raise CVContinuationError(
                f"{where}: state {index} records tau {recorded_tau!r} and this ladder resolves "
                f"{tau}. The series belongs to a different rung.")
        try:
            entry_rows = require_count(entry.get("rows"), where=where,
                                       scope=f"state {index}", field="rows")
        except CVCostError as refusal:
            raise CVContinuationError(str(refusal)) from None
        # 4. ALL REPRESENTATIONS OF THE COUNT AGREE. The block said one number and each state
        #    said its own, and nothing compared them -- so a block count that had been edited,
        #    or an entry written by a different generation, went unnoticed and decided the
        #    truncation for every state.
        if entry_rows != rows:
            raise CVContinuationError(
                f"{where}: the block commits {rows} row(s) and state {index} commits "
                f"{entry_rows}. One of them decided the truncation and the other did not agree.")
        try:
            cv_prefix.validate(
                directory / f"cv_state{index}.csv", entry,
                sidecar=directory / f"cv_state{index}.json",
                definition=definition, expect_columns=columns,
                value_columns=list(definition.names),
                identifiers={"state_index": index, "tau": tau, "exchange_phase": PHASE},
                interval=int(interval_steps), step_column="step")
        except cv_prefix.CVPrefixError as refusal:
            raise CVContinuationError(str(refusal)) from None
        try:
            parse_cost_record(entry["cost"], where=f"{where} state {index}",
                              rows=entry_rows, n_cv=len(definition.names))
            prefixes.append(CommittedPrefix.from_record(entry,
                                                        where=f"{where} state {index}"))
        except CVCostError as refusal:
            raise CVContinuationError(str(refusal)) from None

    # 4b. THE COUNT AGAINST THE RUN'S OWN PROGRESS.
    #
    #     Everything above compares the record with itself: the block count against each state's
    #     count, each cost against its own rows, each digest against the prefix it describes. All
    #     of that is satisfied by a prefix that is simply TOO SHORT -- shorten it, recompute the
    #     digests and costs to match, and every check passes. What happened next was the reason
    #     this exists: `truncate_to` cut each series to that shorter length, dynamics resumed from
    #     the step the checkpoint recorded, and the committed observations in between were gone.
    #     The series that remained had a hole in the middle of it, and nothing said so until the
    #     completion check at the very END of the run -- after the whole segment had been
    #     recomputed, and only because that run happened to reach its end at all. An interruption
    #     before then leaves a gapped set that no later reader can distinguish from a good one.
    #
    #     So the expectation is taken from somewhere the record cannot influence: the checkpoint's
    #     committed STEP and the configured observation grid. The three cadences are independent
    #     -- checkpoints commit on their own interval, frames on theirs -- and only the CV cadence
    #     is used here. The origin is the first committed row's own step rather than zero, which
    #     is what makes this correct for an out-of-place extension: its steps are absolute and
    #     carry on from the parent's terminal value, so its series begins where the parent stopped
    #     and not at 0.
    #
    #     A ladder writes its CV row for a step before the checkpoint that vouches for it (`cv`
    #     precedes `checkpoint` in EVENT_ORDER), so the row at the committed step is itself
    #     committed -- hence the inclusive count. The equality is exact in both directions: fewer
    #     rows than the grid demands is the data loss above, and more is a prefix vouching for
    #     observations the run never reached, which would restore an uncommitted tail as durable.
    if committed_step is not None:
        step = int(committed_step)
        interval = int(interval_steps)
        if rows < 1:
            raise CVContinuationError(
                f"{where}: the checkpoint reached step {step} and commits {rows} "
                f"collective-variable row(s). A ladder observes its initial configuration before "
                f"it commits any checkpoint, so a checkpoint that vouches for no rows at all "
                f"cannot be reconciled with the run that wrote it.")
        first = _first_committed_step(directory, where=where)
        if step < first:
            raise CVContinuationError(
                f"{where}: the checkpoint reached step {step} and the committed series begins at "
                f"step {first}. The checkpoint and the series describe different runs.")
        expected_rows = (step - first) // interval + 1
        if rows != expected_rows:
            lost = expected_rows - rows
            raise CVContinuationError(
                f"{where}: the checkpoint reached step {step} and the series begins at step "
                f"{first}, so on a {interval}-step observation grid exactly {expected_rows} "
                f"collective-variable row(s) are committed -- but the prefix commits {rows}. "
                + (f"Continuing would truncate {lost} committed observation(s) and then resume "
                   f"dynamics at step {step}, leaving a gap in the middle of every state's "
                   f"series with nothing in the file to show where it is. "
                   if lost > 0 else
                   f"The prefix vouches for {-lost} row(s) beyond the last committed step, which "
                   f"would restore an uncommitted tail as though it were durable. ")
                + f"This checkpoint's progress and its collective-variable record disagree; "
                  f"neither can be trusted to decide the truncation. Start a fresh run with "
                  f"--overwrite, or recover the checkpoint generation that matches the series.")

    # 4c. `extra.cv_rows` AND THE PREFIX BLOCK. Two independent statements of the same number live
    #     in one checkpoint, and each has its own reader: `validate_for_continuation` uses the
    #     first, the truncation uses the second. Nothing compared them, so a checkpoint carrying
    #     both could pass each check separately while disagreeing about how much of the run is
    #     durable.
    if committed_rows is not None:
        try:
            counted = require_count(committed_rows, where=where, scope="checkpoint",
                                    field="cv_rows")
        except CVCostError as refusal:
            raise CVContinuationError(str(refusal)) from None
        if counted != rows:
            raise CVContinuationError(
                f"{where}: the checkpoint records {counted} committed collective-variable row(s) "
                f"and the prefix block commits {rows}. The same checkpoint states the durable "
                f"length twice and the two statements disagree.")

    # 5. THE AGGREGATE, against those verified per-state entries. Only completion manifests were
    #    checked before, which is the wrong half: continuation is the operation that TRUNCATES.
    try:
        parse_aggregate_record(block.get("cost"), where=f"{where} aggregate",
                               entries_key="per_state", identity_key="state_index",
                               expected_identities=expected)
    except CVCostError as refusal:
        raise CVContinuationError(str(refusal)) from None

    # 6. TYPED DATA OUT, for both the truncation and the restoration, so nothing downstream
    #    rereads or re-coerces a field this function has already checked.
    return ValidatedLadderPrefix(rows=rows, per_state=tuple(prefixes))



def _first_committed_step(directory, *, where):
    """The step of the first row of state 0's series, read from the file itself.

    The reconciliation above needs an origin, and for an extension that origin is not zero: its
    steps are absolute and continue from the parent's terminal value. Taking it from the series
    rather than assuming it makes the check correct for both, and the row it reads has already
    been covered by the verified prefix digest by the time this is called.
    """
    from ..cv import prefix as cv_prefix

    series = Path(directory) / "cv_state0.csv"
    lines = cv_prefix.read_lines(series)
    if len(lines) < 2:
        raise CVContinuationError(
            f"{where}: {series.name} holds no data rows, so the step its committed series begins "
            f"at cannot be established.")
    header = lines[0].split(",")
    try:
        column = header.index("step")
        return int(lines[1].split(",")[column])
    except (ValueError, IndexError):
        raise CVContinuationError(
            f"{where}: {series.name} does not begin with a readable `step` value, so the committed "
            f"prefix cannot be reconciled with the checkpoint's progress.") from None


def truncate_to(directory, *, taus, rows):
    """Cut every state's series back to the committed prefix. Only after validation."""
    from ..cv import prefix as cv_prefix

    directory = Path(directory)
    for index in range(len(taus)):
        cv_prefix.truncate(directory / f"cv_state{index}.csv", int(rows))


def committed_prefixes(validated, n_states):
    """The per-state prefixes, taken from what `validate_prefixes` already verified.

    This used to re-read the raw block: it keyed the entries by `int(state_index)`, so a
    duplicate overwrote its twin and a missing state fell through to an empty record restoring
    zero -- the same permissive read the validator had just been strengthened to refuse, running
    again a few lines later and undoing it. A helper that bypasses the parser is a second policy,
    and it is the one that decides what is actually restored.
    """
    if not isinstance(validated, ValidatedLadderPrefix):
        raise CVContinuationError(
            "committed_prefixes must be given the validated prefix returned by "
            "validate_prefixes; re-reading the raw checkpoint block would bypass every check it "
            "just performed")
    if len(validated.per_state) != int(n_states):
        raise CVContinuationError(
            f"the validated prefix covers {len(validated.per_state)} state(s) and this ladder "
            f"resolves {n_states}")
    return list(validated.per_state)
