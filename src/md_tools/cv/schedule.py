"""When a collective variable is observed, and the exactness that is required of it.

WHY A SCHEDULE IS ITS OWN MODULE

    The CV cadence is INDEPENDENT of the trajectory and state-data cadences, and may be more
    frequent than either. That independence is the point -- a torsion is cheap to evaluate and a
    frame is expensive to store, so the whole reason to have a separate CV series is to sample it
    finely without inflating the trajectory.

    Independence is also what makes the schedule easy to get subtly wrong. Three properties have
    to hold at once, and each fails silently:

      the interval must DIVIDE the enclosing span. An interval that does not divide it produces a
      final partial gap, so the last observation sits at an irregular spacing from its
      predecessor. Every downstream time-series analysis -- autocorrelation, block averaging,
      spectral density -- assumes uniform spacing and none of them can detect the violation;

      step 0 and the final step must each appear EXACTLY ONCE. Step 0 is the initial
      configuration, and the final step is what the run's result is quoted at. A schedule that
      emits either twice puts a duplicate row in the CSV, which double-weights one configuration;
      one that omits the final step reports a run as ending somewhere it did not;

      minimisation produces no series at all. A minimiser's iterations are not dynamics: they have
      no timestep, so `time_ps` would be a fiction, and the intermediate geometries are on no
      physical trajectory. A CV series over them looks exactly like one over dynamics.

    So the schedule is computed once, checked, and then iterated -- rather than each reporter
    deciding for itself whether this step is one it should write.
"""

from __future__ import annotations


class CVScheduleError(ValueError):
    """A cadence that cannot be honoured exactly. Refused rather than approximated."""


def observation_steps(total_steps: int, interval_steps: int, *, where: str = "this stage",
                      divides: str = "the stage step count") -> tuple[int, ...]:
    """Every step at which a CV is observed, from 0 to `total_steps` inclusive.

    The interval must divide `total_steps` exactly. Both endpoints appear exactly once, which
    `range(0, total + 1, interval)` gives precisely when the division is exact -- and the check
    below is what guarantees it is, rather than leaving the property to a caller's arithmetic.
    """
    total = int(total_steps)
    interval = int(interval_steps)
    if interval <= 0:
        raise CVScheduleError(
            f"{where}: a collective-variable interval must be a positive number of steps; "
            f"got {interval}")
    if total < 0:
        raise CVScheduleError(f"{where}: a negative step count ({total}) has no schedule")
    if total == 0:
        # Not an error: a zero-length span is a minimisation or an empty stage. It has no
        # dynamics, so it has no series -- see the module note.
        return ()
    if total % interval:
        raise CVScheduleError(
            f"{where}: collective_variables.interval_steps = {interval} does not divide "
            f"{divides} ({total}). It would leave a final gap of {total % interval} steps, so the "
            f"last observation would sit at an irregular spacing from the one before it -- and "
            f"every time-series analysis downstream assumes uniform spacing and none can detect "
            f"that it was violated. Choose an interval that divides {total} exactly.")
    return tuple(range(0, total + 1, interval))


def check_divides(interval_steps: int, span: int, *, where: str, what: str) -> None:
    """`interval_steps` divides `span`, or an explanation of why that is required.

    The REST2 and AIS cadences are stated against something other than a stage length --
    the exchange interval, the switching length -- but the requirement and the reason are the
    same, so they share the message rather than each inventing one.
    """
    interval = int(interval_steps)
    span = int(span)
    if interval <= 0:
        raise CVScheduleError(
            f"{where}: a collective-variable interval must be a positive number of steps; "
            f"got {interval}")
    if span <= 0:
        raise CVScheduleError(f"{where}: {what} is {span}, which no interval can divide")
    if span % interval:
        raise CVScheduleError(
            f"{where}: collective_variables.interval_steps = {interval} does not divide {what} "
            f"({span}). The collective-variable series must land on an exact grid within every "
            f"{what.split('(')[0].strip()}, so that step 0 and the final step of each appear "
            f"exactly once and the spacing is uniform. Choose an interval that divides {span}.")
