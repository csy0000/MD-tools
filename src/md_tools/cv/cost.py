"""What collective-variable reporting cost, in terms that mean one thing each.

WHY THE OLD COUNTER WAS A MISNOMER

    `cv_evaluations` was incremented once per reporter call. A call evaluates the WHOLE
    configured set, so for a definition with two torsions it counted one where two scalar values
    had been computed. The name said "evaluations" and the number said "observations", and the
    two coincide only for the single-CV case that happened to be tested.

    Worse, a test asserted that behaviour as correct. A call-count implementation therefore had a
    passing test claiming it measured scalar work, which is how a misnomer becomes a contract.

    The three quantities are now separate and each says one thing:

      `observations`   configurations on which the complete configured set was evaluated;
      `evaluations`    scalar CV values actually computed -- `observations x N_cv`;
      `wall_seconds`   time inside the evaluation itself, excluding CSV serialisation, hashing,
                       validation and everything else the reporting point also does.

TWO SCOPES, NEVER ONE

    `segment` is what THIS invocation did. `cumulative` is what the whole logical simulation has
    done across every invocation. On a fresh run they are equal.

    Both are kept because either alone is misleading. Cumulative alone hides how much a resumed
    run actually did; segment alone makes an interrupted run look cheaper than an identical
    uninterrupted one, which is precisely the comparison the counters exist to support.
"""

from __future__ import annotations

from dataclasses import dataclass


class CVCostError(ValueError):
    """A cost record that cannot be believed."""


@dataclass(frozen=True)
class CVCost:
    """One scope's cost. Frozen: accumulation returns a new value rather than mutating."""

    observations: int = 0
    evaluations: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self):
        for name in ("observations", "evaluations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CVCostError(f"{name} must be a non-negative integer; got {value!r}")
        seconds = self.wall_seconds
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise CVCostError(f"wall_seconds must be a number; got {seconds!r}")
        if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
            raise CVCostError(f"wall_seconds must be finite and non-negative; got {seconds!r}")

    def plus(self, other: "CVCost") -> "CVCost":
        return CVCost(
            observations=self.observations + other.observations,
            evaluations=self.evaluations + other.evaluations,
            wall_seconds=round(self.wall_seconds + other.wall_seconds, 6),
        )

    def as_record(self) -> dict:
        return {"cv_observations": int(self.observations),
                "cv_evaluations": int(self.evaluations),
                "wall_seconds": round(float(self.wall_seconds), 6)}

    @classmethod
    def from_record(cls, record, *, where: str = "cost record", scope: str = "scope") -> "CVCost":
        """Read a stored scope STRICTLY. No coercion: the decoded type is the type.

        This used to call `int()` and `float()` on whatever it found, which is the whole defect.
        `int(2.7)` is 2, `int("5")` is 5, `int(True)` is 1: three records that violate the schema
        became three that satisfy it, and the violation was gone before anything could refuse it.
        A counter read from a file is external input, and coercing external input before
        validating it is how a corrupt record becomes an authoritative one.
        """
        return parse_scope(record, where=where, scope=scope)


#: Bumped when the meaning of a stored counter changes, so a reader can tell which it has.
COST_SCHEMA_VERSION = 2


def cost_record(segment: CVCost, cumulative: CVCost, *, rows: int | None = None) -> dict:
    """The two-scope record written into checkpoints, manifests and logs."""
    if cumulative.observations < segment.observations:
        raise CVCostError(
            f"cumulative observations ({cumulative.observations}) are fewer than this segment's "
            f"({segment.observations}); cumulative covers every invocation including this one")
    record = {
        "schema_version": COST_SCHEMA_VERSION,
        "segment": segment.as_record(),
        "cumulative": cumulative.as_record(),
    }
    if rows is not None:
        # Rows written, kept beside the cost rather than inside a scope: it is a property of the
        # FILE, not of either scope's work, and putting it in one made it read as a cost.
        require_count(rows, where="cost record", scope="record", field="cv_rows")
        record["cv_rows"] = rows
    return record


def read_scope(record, scope: str, *, where: str = "cost record") -> CVCost:
    """One scope out of a stored record. Absent record is zero; a stored one is validated."""
    if not record:
        return CVCost()
    if not isinstance(record, dict):
        raise CVCostError(f"{where}: expected a mapping, got {type(record).__name__}")
    if "segment" not in record and "cumulative" not in record:
        raise legacy_refusal(where)
    return parse_scope(record.get(scope), where=where, scope=scope)


@dataclass(frozen=True)
class CommittedPrefix:
    """What a checkpoint committed about one CV series: its rows AND the cost that produced them.

    One object, because they are one fact. They used to travel as separate integer arguments, and
    a caller that restored `rows` while forgetting the counters got a resumed run whose series was
    correct and whose cost had silently reset -- the failure this type exists to make impossible.
    """

    rows: int = 0
    cumulative: CVCost = CVCost()

    def __post_init__(self):
        if not isinstance(self.rows, int) or isinstance(self.rows, bool) or self.rows < 0:
            raise CVCostError(f"rows must be a non-negative integer; got {self.rows!r}")

    @classmethod
    def from_record(cls, record, *, where: str = "committed prefix") -> "CommittedPrefix":
        if not record:
            return cls()
        if not isinstance(record, dict):
            raise CVCostError(f"{where}: expected a mapping, got {type(record).__name__}")
        rows = record.get("rows", record.get("cv_rows", 0))
        if rows is None:
            rows = 0
        require_count(rows, where=where, scope="prefix", field="rows")
        return cls(rows=rows, cumulative=read_scope(record.get("cost"), "cumulative",
                                                    where=where))


# -- the one strict parser ------------------------------------------------------------------------
#
# WHY ONE
#
#     These records are read in eight places -- cMD checkpoint prefixes and logs, the ladder's
#     per-state prefixes and aggregate, its completion and extension validation, AIS checkpoint
#     prefixes, per-path completion, the global aggregation and completed-path verification, and
#     the provenance consumers downstream of all of them. Eight readers is eight policies, and the
#     one that is loosest decides what the system actually accepts.
#
# WHY IT REFUSES RATHER THAN COERCES
#
#     A stored counter is external input. `int(2.7)` is 2, `int("5")` is 5 and `int(True)` is 1,
#     so coercing before validating destroys the evidence of the violation and then reports the
#     result as authoritative. Every check below therefore tests the DECODED type first and never
#     converts.

#: Exactly the counters a scope carries. Not a minimum: an unexpected key in a scope means the
#: record was written by something with a different idea of what a scope is, and accepting it
#: silently is how two schemas come to share one name.
SCOPE_FIELDS = ("cv_observations", "cv_evaluations", "wall_seconds")

#: What may sit beside the two scopes. Everything here is documented metadata rather than cost.
RECORD_FIELDS = frozenset({
    "schema_version", "segment", "cumulative", "cv_rows",
    "aggregation", "per_state", "per_path", "invocation_id",
    "state_index", "tau", "path_index", "disposition",
})

#: Dispositions a per-path aggregate entry may declare. Named rather than free text, because the
#: whole point of the field is that a reader can tell "did no work now" from "did none ever".
DISPOSITIONS = ("already_complete", "resumed_and_completed", "fresh_and_completed")


def legacy_refusal(where: str) -> "CVCostError":
    """The explicit schema-version-1 policy: refuse, with the reason and the way out.

    Migration is not offered silently. A version-1 record stored ONE counter, `cv_evaluations`,
    which counted reporter calls -- observations. Scalar evaluations cannot be recovered from it
    without `N_cv`, and presenting the call count as a scalar total would be exactly the misnomer
    the version-2 schema exists to remove. `migrate_schema_1` performs the conversion where a
    caller can supply the verified row count and the CV definition, and records that it did.
    """
    return CVCostError(
        f"{where}: this collective-variable cost record predates schema version "
        f"{COST_SCHEMA_VERSION} (it carries no `segment`/`cumulative` scopes). Its single stored "
        f"counter counted reporter calls, not scalar evaluations, so a scalar total cannot be "
        f"recovered from it and will not be invented. Start a fresh run with --overwrite, or "
        f"continue it with the build that wrote it and no CV reporting enabled.")


def require_count(value, *, where: str, scope: str, field: str) -> int:
    """A counter. `type(value) is int` -- not a bool, not a float, not a numeric string.

    `isinstance(True, int)` is True in Python, so a bool passes an `isinstance` check and then
    counts as 1. It is excluded by name.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise CVCostError(
            f"{where}: {scope}.{field} must be an integer, got {type(value).__name__} "
            f"{value!r}. Numeric strings and floats -- including 5.0 -- are not integers here, "
            f"because accepting them means accepting 5.7 as 5.")
    if value < 0:
        raise CVCostError(f"{where}: {scope}.{field} must be non-negative, got {value!r}")
    return value


def require_seconds(value, *, where: str, scope: str, field: str = "wall_seconds") -> float:
    """A measured duration: real, finite, non-negative. A bool is not a duration."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CVCostError(
            f"{where}: {scope}.{field} must be a real number, got "
            f"{type(value).__name__} {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise CVCostError(f"{where}: {scope}.{field} must be finite, got {value!r}")
    if value < 0:
        raise CVCostError(f"{where}: {scope}.{field} must be non-negative, got {value!r}")
    return float(value)


def parse_scope(record, *, where: str, scope: str) -> CVCost:
    """One scope, strictly. Exactly the documented counters, each of the documented type."""
    if record is None:
        raise CVCostError(f"{where}: {scope} is required and is missing")
    if not isinstance(record, dict):
        raise CVCostError(
            f"{where}: {scope} must be a mapping, got {type(record).__name__}")
    missing = [name for name in SCOPE_FIELDS if name not in record]
    if missing:
        raise CVCostError(f"{where}: {scope} is missing {', '.join(missing)}")
    unexpected = sorted(set(record) - set(SCOPE_FIELDS))
    if unexpected:
        raise CVCostError(
            f"{where}: {scope} carries unexpected field(s) {unexpected}; a scope holds exactly "
            f"{list(SCOPE_FIELDS)}")
    return CVCost(
        observations=require_count(record["cv_observations"], where=where, scope=scope,
                                   field="cv_observations"),
        evaluations=require_count(record["cv_evaluations"], where=where, scope=scope,
                                  field="cv_evaluations"),
        wall_seconds=require_seconds(record["wall_seconds"], where=where, scope=scope))


@dataclass(frozen=True)
class ParsedCost:
    """A validated schema-2 record: both scopes, and the row count when the record carries one."""

    segment: CVCost
    cumulative: CVCost
    rows: int | None = None


def parse_cost_record(record, *, where: str, rows: int | None = None,
                      n_cv: int | None = None, require_rows: bool = False) -> ParsedCost:
    """Validate one stored cost record before anything is believed or written.

    `rows` and `n_cv`, when given, are the VERIFIED facts from outside the record -- the row count
    someone counted in the file and the number of CVs the definition resolves to. Checking the
    record against them is the difference between "internally consistent" and "true": a record can
    agree with itself perfectly and still describe a different series.
    """
    if record is None:
        raise CVCostError(f"{where}: no collective-variable cost record")
    if not isinstance(record, dict):
        raise CVCostError(f"{where}: cost record must be a mapping, got "
                          f"{type(record).__name__}")

    if "schema_version" not in record:
        if "segment" not in record and "cumulative" not in record:
            raise legacy_refusal(where)
        raise CVCostError(f"{where}: cost record has no schema_version")
    version = record["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise CVCostError(
            f"{where}: schema_version must be the integer {COST_SCHEMA_VERSION}, got "
            f"{type(version).__name__} {version!r}")
    if version != COST_SCHEMA_VERSION:
        if version == 1:
            raise legacy_refusal(where)
        raise CVCostError(
            f"{where}: cost schema version {version} is not supported; this build writes and "
            f"reads version {COST_SCHEMA_VERSION}")

    unexpected = sorted(set(record) - RECORD_FIELDS)
    if unexpected:
        raise CVCostError(f"{where}: cost record carries unexpected field(s) {unexpected}")

    segment = parse_scope(record.get("segment"), where=where, scope="segment")
    cumulative = parse_scope(record.get("cumulative"), where=where, scope="cumulative")

    # The scopes are nested, not parallel: cumulative covers every invocation INCLUDING this one.
    for field, mine, theirs in (("cv_observations", segment.observations, cumulative.observations),
                                ("cv_evaluations", segment.evaluations, cumulative.evaluations),
                                ("wall_seconds", segment.wall_seconds, cumulative.wall_seconds)):
        if mine > theirs:
            raise CVCostError(
                f"{where}: segment.{field} ({mine}) exceeds cumulative.{field} ({theirs}); "
                f"cumulative covers every invocation including this one")

    stored_rows = None
    if "cv_rows" in record:
        stored_rows = require_count(record["cv_rows"], where=where, scope="record",
                                    field="cv_rows")
    elif require_rows:
        raise CVCostError(f"{where}: cost record is required to carry cv_rows and does not")

    if rows is not None:
        require_count(rows, where=where, scope="verified", field="rows")
        if stored_rows is not None and stored_rows != rows:
            raise CVCostError(
                f"{where}: cost record says cv_rows={stored_rows} and the verified series holds "
                f"{rows} row(s)")
        if cumulative.observations != rows:
            raise CVCostError(
                f"{where}: cumulative.cv_observations is {cumulative.observations} and the "
                f"verified series holds {rows} row(s); every committed row is one observation")
        if n_cv is not None:
            require_count(n_cv, where=where, scope="verified", field="n_cv")
            if cumulative.evaluations != rows * n_cv:
                raise CVCostError(
                    f"{where}: cumulative.cv_evaluations is {cumulative.evaluations} and "
                    f"{rows} row(s) of {n_cv} collective variable(s) is {rows * n_cv}")
    return ParsedCost(segment=segment, cumulative=cumulative, rows=stored_rows)


#: Wall time is summed from values already rounded to microseconds, so an aggregate total may
#: differ from a re-summation in the last place. Counters are compared exactly; seconds carry the
#: rounding they were stored with, one microsecond per contributing entry.
SECONDS_EPSILON = 1e-6


def parse_aggregate_record(record, *, where: str, entries_key: str, identity_key: str,
                           expected_identities=None) -> ParsedCost:
    """Validate an aggregate and prove its totals are the sum of the entries beside it.

    An aggregate that nobody checks against its own parts is a number to be trusted rather than
    audited, which is the opposite of why the per-entry records are kept.
    """
    parsed = parse_cost_record(record, where=where)
    entries = record.get(entries_key)
    if entries is None:
        raise CVCostError(f"{where}: aggregate carries no {entries_key} records, so its totals "
                          f"cannot be audited")
    if not isinstance(entries, list):
        raise CVCostError(f"{where}: {entries_key} must be a list, got "
                          f"{type(entries).__name__}")

    seen: dict[int, CVCost] = {}
    segment_total = CVCost()
    cumulative_total = CVCost()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CVCostError(f"{where}: {entries_key}[{position}] must be a mapping")
        if identity_key not in entry:
            raise CVCostError(f"{where}: {entries_key}[{position}] carries no {identity_key}")
        identity = require_count(entry[identity_key], where=where,
                                 scope=f"{entries_key}[{position}]", field=identity_key)
        if identity in seen:
            raise CVCostError(
                f"{where}: {identity_key} {identity} appears more than once in {entries_key}")
        one = parse_cost_record(entry, where=f"{where} {identity_key}={identity}")
        seen[identity] = one.cumulative
        segment_total = segment_total.plus(one.segment)
        cumulative_total = cumulative_total.plus(one.cumulative)

    if expected_identities is not None:
        wanted = set(expected_identities)
        if set(seen) != wanted:
            missing = sorted(wanted - set(seen))
            extra = sorted(set(seen) - wanted)
            raise CVCostError(
                f"{where}: {entries_key} covers {identity_key} {sorted(seen)} and this run "
                f"expects {sorted(wanted)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else ""))

    for scope, total, stated in (("segment", segment_total, parsed.segment),
                                 ("cumulative", cumulative_total, parsed.cumulative)):
        for field, mine, theirs in (("cv_observations", total.observations, stated.observations),
                                    ("cv_evaluations", total.evaluations, stated.evaluations)):
            if mine != theirs:
                raise CVCostError(
                    f"{where}: {scope}.{field} is {theirs} and the {len(seen)} {entries_key} "
                    f"entries sum to {mine}")
        drift = abs(total.wall_seconds - stated.wall_seconds)
        if drift > SECONDS_EPSILON * max(1, len(seen)):
            raise CVCostError(
                f"{where}: {scope}.wall_seconds is {stated.wall_seconds} and the entries sum to "
                f"{total.wall_seconds}")
    return parsed


def migrate_schema_1(record, *, rows: int, n_cv: int, where: str) -> dict:
    """The other half of the version-1 policy: convert, but only where the facts are supplied.

    A version-1 record cannot say how many scalar values it measured. Given the VERIFIED row count
    and the CV definition it can be restated exactly, and the result records that it was migrated
    rather than written that way -- so a reader can tell a reconstructed total from a measured one.
    """
    if not isinstance(record, dict):
        raise CVCostError(f"{where}: expected a mapping to migrate, got "
                          f"{type(record).__name__}")
    require_count(rows, where=where, scope="verified", field="rows")
    require_count(n_cv, where=where, scope="verified", field="n_cv")
    seconds = record.get("cv_seconds", record.get("wall_seconds", 0.0))
    cost = CVCost(observations=rows, evaluations=rows * n_cv,
                  wall_seconds=require_seconds(seconds, where=where, scope="legacy"))
    migrated = cost_record(cost, cost, rows=rows)
    migrated["aggregation"] = "migrated from schema 1"
    return migrated
