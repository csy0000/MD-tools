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
