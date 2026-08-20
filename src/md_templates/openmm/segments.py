"""Segment and exchange arithmetic: durations in, exact integer steps out.

Two ideas are kept apart here, and the separation is the point:

    duration_per_segment    SCIENTIFIC. One segment of production. Lives in the JSON.
    number of segments      EXECUTION. How many times the driver asks for one. Lives in Bash.
    completed segments      RUNTIME STATE. What actually finished. Lives in the run manifest.

The previous contract folded all three into ``n_chunks``/``chunk_ns`` in the scientific input,
which meant a configuration hash changed when a user merely wanted to run longer, and a run that
was extended looked like a different calculation. Segment count is no longer an input to the
science.

Every conversion here refuses to round. A duration that is not a whole number of steps is a
configuration error, not something to nudge: a segment silently shortened by one step drifts the
exchange schedule out of alignment with the committed watermark, and the run still looks healthy.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Relative tolerance for deciding that a float ratio "is" an integer. Durations arrive as
#: doubles built from decimal text ('5 ns', '2 fs'), so the ratio of two exact decimals can land a
#: few ULP away from the integer it means. 1e-9 accepts that and still refuses a genuine mismatch:
#: the smallest real error -- one step in a segment -- is a relative deviation of 1/steps, which
#: for the largest segment considered here (2.5e6 steps) is 4e-7, four hundred times this bound.
_INTEGER_RATIO_RTOL = 1e-9


def _exact_ratio(numerator: float, denominator: float, *,
                 numerator_label: str, denominator_label: str,
                 numerator_source: str, denominator_source: str) -> int:
    """``numerator / denominator`` when that is an integer; a precise error when it is not."""
    if denominator <= 0.0:
        raise ValueError(f"{denominator_label} must be positive; got {denominator_source}")
    if numerator <= 0.0:
        raise ValueError(f"{numerator_label} must be positive; got {numerator_source}")

    ratio = numerator / denominator
    nearest = round(ratio)
    if nearest < 1:
        raise ValueError(
            f"{numerator_label} ({numerator_source}) is shorter than {denominator_label} "
            f"({denominator_source}); it does not contain a single whole unit."
        )
    if abs(ratio - nearest) > _INTEGER_RATIO_RTOL * nearest:
        raise ValueError(
            f"{numerator_label} ({numerator_source}) is not a whole number of "
            f"{denominator_label} ({denominator_source}): {ratio:.9f}. Rounding it would run a "
            f"different length than the configuration declares. Choose a {numerator_label} that "
            f"divides exactly -- the nearest whole values are "
            f"{int(nearest) * denominator:.9g} and {(int(nearest) + 1) * denominator:.9g} "
            f"in the same units."
        )
    return int(nearest)


@dataclass(frozen=True)
class SegmentPlan:
    """The exact integer plan for ONE segment.

    Nothing here says how many segments will be run. That is deliberate.
    """

    steps_per_segment: int
    steps_per_exchange: int | None
    number_of_exchanges_per_segment: int | None

    @property
    def has_exchanges(self) -> bool:
        return self.steps_per_exchange is not None


def steps_for_duration(duration_value: float, timestep_value: float, *,
                       duration_source: str, timestep_source: str,
                       duration_label: str = "duration") -> int:
    """Whole integrator steps in a duration, or a refusal."""
    return _exact_ratio(
        duration_value, timestep_value,
        numerator_label=duration_label, denominator_label="integrator.timestep",
        numerator_source=duration_source, denominator_source=timestep_source,
    )


def plan_segment(duration_per_segment_value: float, timestep_value: float, *,
                 duration_source: str, timestep_source: str,
                 number_of_exchanges_per_segment: int | None = None) -> SegmentPlan:
    """Resolve one segment into exact integer steps, including the exchange cadence.

    At 5 ns, a 2 fs timestep and 100 exchanges this is 2,500,000 steps per segment and 25,000
    steps (50 ps) between exchange rounds.
    """
    steps_per_segment = steps_for_duration(
        duration_per_segment_value, timestep_value,
        duration_source=duration_source, timestep_source=timestep_source,
        duration_label="production.duration_per_segment",
    )

    if number_of_exchanges_per_segment is None:
        return SegmentPlan(steps_per_segment=steps_per_segment,
                           steps_per_exchange=None,
                           number_of_exchanges_per_segment=None)

    if number_of_exchanges_per_segment < 1:
        raise ValueError(
            "exchange.number_of_exchanges_per_segment must be at least 1; got "
            f"{number_of_exchanges_per_segment}"
        )
    if steps_per_segment % number_of_exchanges_per_segment != 0:
        raise ValueError(
            f"production.duration_per_segment ({duration_source}) is {steps_per_segment} steps, "
            f"which is not divisible by exchange.number_of_exchanges_per_segment "
            f"({number_of_exchanges_per_segment}): "
            f"{steps_per_segment / number_of_exchanges_per_segment:.6f} steps per exchange. "
            "An exchange round landing mid-step would drop or duplicate an attempt across a "
            "segment boundary, and the committed watermark would no longer agree with the "
            "exchange history."
        )

    return SegmentPlan(
        steps_per_segment=steps_per_segment,
        steps_per_exchange=steps_per_segment // number_of_exchanges_per_segment,
        number_of_exchanges_per_segment=number_of_exchanges_per_segment,
    )


def plan_segment_from_duration_and_exchanges(
    duration_value: float, number_of_exchanges_per_segment: int, timestep_value: float, *,
    duration_source: str, timestep_source: str) -> SegmentPlan:
    """Resolve a REST2 segment stated as a DURATION and an EXCHANGE COUNT.

    This is the public form. Both divisions happen in integer step space and both must be exact::

        steps_per_segment  = duration_per_segment / timestep
        steps_per_exchange = steps_per_segment / number_of_exchanges_per_segment
        exchange_interval  = steps_per_exchange * timestep

    Deriving in steps rather than in picoseconds is what makes this safe. The earlier form stated a
    count and an interval and multiplied, which is exact by construction; stating a duration makes
    the interval a quotient, and a quotient can fail to divide. It is refused here, never rounded:
    a segment silently shortened by one step drifts the exchange schedule out of alignment with the
    committed watermark while the run still looks healthy.

    5 ns at 4 fs is 1,250,000 steps; over 1000 exchanges that is 1250 steps per round, a 5 ps
    interval.
    """
    if number_of_exchanges_per_segment < 1:
        raise ValueError(
            "exchange.number_of_exchanges_per_segment must be at least 1; got "
            f"{number_of_exchanges_per_segment}"
        )
    steps_per_segment = steps_for_duration(
        duration_value, timestep_value,
        duration_source=duration_source, timestep_source=timestep_source,
        duration_label="production.duration_per_segment",
    )
    if steps_per_segment % number_of_exchanges_per_segment != 0:
        exact = steps_per_segment / number_of_exchanges_per_segment
        raise ValueError(
            f"production.duration_per_segment ({duration_source}) is {steps_per_segment:,} steps at "
            f"{timestep_source}, which does not divide into "
            f"{number_of_exchanges_per_segment:,} exchanges: that would be {exact:.6f} steps per "
            "round.\n"
            "  Refusing rather than rounding: a rounded exchange interval drifts the schedule out "
            "of alignment with the\n"
            "  committed watermark while the run still looks healthy.\n"
            f"  Nearby exchange counts that divide exactly: "
            f"{_nearby_divisors(steps_per_segment, number_of_exchanges_per_segment)}"
        )
    return SegmentPlan(
        steps_per_segment=steps_per_segment,
        steps_per_exchange=steps_per_segment // number_of_exchanges_per_segment,
        number_of_exchanges_per_segment=number_of_exchanges_per_segment,
    )


def _nearby_divisors(total_steps: int, wanted: int, span: int = 5000) -> str:
    """Exchange counts near `wanted` that divide `total_steps` exactly.

    An error that only says "this does not divide" leaves the reader to factorise by hand.
    """
    low = max(1, wanted - span)
    candidates = [n for n in range(low, wanted + span + 1) if total_steps % n == 0]
    if not candidates:
        return "none within +/-%d" % span
    candidates.sort(key=lambda n: (abs(n - wanted), n))
    return ", ".join(str(n) for n in candidates[:5])


def plan_segment_from_exchanges(n_exchange_per_segment: int, exchange_interval_value: float,
                                timestep_value: float, *, interval_source: str,
                                timestep_source: str) -> SegmentPlan:
    """Resolve a REST2 segment stated as a COUNT and an INTERVAL.

    This is the preferred form and the reason is arithmetic: the segment length is a product,

        steps_per_segment = n_exchange_per_segment * steps_per_exchange

    so it is exact whenever the interval itself is a whole number of steps. Stating a duration and
    an exchange count instead makes the interval a quotient, which can fail to divide and then has
    to be refused -- or, worse, rounded, drifting the exchange schedule out of alignment with the
    committed watermark while the run still looks healthy.

    1000 exchanges at 5 ps with a 4 fs timestep is 1250 steps per round and 1,250,000 per segment.
    """
    if n_exchange_per_segment < 1:
        raise ValueError(
            "exchange.n_exchange_per_segment must be at least 1; got "
            f"{n_exchange_per_segment}"
        )
    steps_per_exchange = steps_for_duration(
        exchange_interval_value, timestep_value,
        duration_source=interval_source, timestep_source=timestep_source,
        duration_label="exchange.exchange_interval",
    )
    return SegmentPlan(
        steps_per_segment=steps_per_exchange * n_exchange_per_segment,
        steps_per_exchange=steps_per_exchange,
        number_of_exchanges_per_segment=n_exchange_per_segment,
    )


def reporting_interval_steps(interval_value: float, timestep_value: float, *,
                             interval_source: str, timestep_source: str,
                             label: str) -> int:
    """Whole steps between reporter writes, or a refusal.

    A reporting interval that does not divide the timestep exactly produces frames at drifting
    physical times, which is invisible in the file and wrong in any time-resolved analysis.
    """
    return steps_for_duration(
        interval_value, timestep_value,
        duration_source=interval_source, timestep_source=timestep_source,
        duration_label=label,
    )
