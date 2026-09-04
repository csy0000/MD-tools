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
    def from_record(cls, record) -> "CVCost":
        """Read a stored scope. Absent is zero; a record from the OLD flat schema is migrated.

        The old schema had `cv_evaluations` meaning observations and no observation count at all,
        so a stored value cannot be reinterpreted as scalar work without knowing `N_cv` -- and
        guessing would silently multiply a real number by a number we do not have. Such a record
        is therefore migrated as `observations`, with `evaluations` left at zero rather than
        fabricated, and the caller documents the version behaviour.
        """
        if not record:
            return cls()
        if "cv_observations" in record:
            return cls(observations=int(record.get("cv_observations", 0)),
                       evaluations=int(record.get("cv_evaluations", 0)),
                       wall_seconds=float(record.get("wall_seconds",
                                                     record.get("cv_seconds", 0.0))))
        # LEGACY (schema 1): `cv_evaluations` counted reporter calls, i.e. observations.
        return cls(observations=int(record.get("cv_evaluations", 0)),
                   evaluations=0,
                   wall_seconds=float(record.get("cv_seconds", 0.0)))


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
        record["cv_rows"] = int(rows)
    return record


def read_scope(record, scope: str) -> CVCost:
    """One scope out of a stored record, tolerating the legacy flat shape."""
    if not record:
        return CVCost()
    if "segment" in record or "cumulative" in record:
        return CVCost.from_record(record.get(scope) or {})
    # Legacy flat record: it described the whole run, so it is the cumulative scope and this
    # invocation's segment is unknown -- reported as zero rather than as the whole run's work.
    return CVCost.from_record(record) if scope == "cumulative" else CVCost()


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
    def from_record(cls, record) -> "CommittedPrefix":
        if not record:
            return cls()
        return cls(rows=int(record.get("rows", record.get("cv_rows", 0)) or 0),
                   cumulative=read_scope(record.get("cost"), "cumulative"))
