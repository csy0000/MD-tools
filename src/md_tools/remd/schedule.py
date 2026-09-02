#!/usr/bin/env python
"""When things happen, in absolute integration steps. Copied into every generated project.

A replica-exchange run has four independent periodic events and one terminal one:

    exchange        attempt a transition
    whole           store every walker's complete coordinates
    solute          store the solute subset only, typically far more often
    checkpoint      write the restart state
    end             the requested budget is reached

NONE OF THESE IS DERIVED FROM ANOTHER. The previous runtime had a single `segment_ps` propagation
quantum and wrote coordinates only on the whole-output stride, so the "frequent solute stream" it
documented did not exist at all, and every other interval had to be a multiple of a quantum the
user had to supply.

`segment_ps` is gone from the user's vocabulary. A user states physical intervals; each converts to
an EXACT integer number of integration steps or is refused; and propagation advances to the next
scheduled event, whatever it is. The internal quantum is whatever gap the schedule produces -- it
is derived, private, and never presented as a scientific input.

SIMULTANEOUS EVENTS
    Several events land on the same step routinely: with a 2 ps solute interval and a 10 ps
    exchange interval they coincide every fifth solute frame. They execute ONCE each, in this
    order, which is fixed and documented rather than left to dictionary iteration:

        1. exchange      the transition happens first, so everything stored afterwards describes
                         the state the run actually continued from
        2. whole         complete coordinates
        3. solute        the solute subset
        4. checkpoint    written last, so a checkpoint never claims a step whose frames are missing

    A frame is never written twice for one step, and the checkpoint is always the newest thing on
    disk for the step it names.
"""

#: The event kinds, in the order they execute when they coincide. Do not reorder without changing
#: the documented contract and the tests that assert it.
EVENT_ORDER = ("exchange", "whole", "solute", "checkpoint")


class ScheduleError(ValueError):
    """An interval cannot be expressed as a whole number of integration steps."""


def exact_steps(duration_ps, timestep_fs, *, what):
    """`duration_ps` as a whole number of steps, or an error naming the offender.

    Rejected rather than rounded. A rounded interval means the physical time every record claims is
    not the time that was simulated, and nothing downstream can detect it.
    """
    timestep_ps = timestep_fs / 1000.0
    ratio = duration_ps / timestep_ps
    steps = int(round(ratio))
    if abs(ratio - steps) > 1e-9:
        raise ScheduleError(
            f"{what} ({duration_ps} ps) is not a whole number of {timestep_fs} fs steps: it is "
            f"{ratio} steps. Rejected rather than rounded -- a rounded interval makes every "
            f"recorded time a lie. Choose an interval that divides exactly, or change the "
            f"timestep.")
    if steps < 1:
        raise ScheduleError(
            f"{what} ({duration_ps} ps) is shorter than one {timestep_fs} fs step.")
    return steps


class EventSchedule:
    """Absolute-step scheduling for one run. Pure arithmetic; it owns no state of the simulation.

    Every interval is independent. `next_events(step)` answers "what is the next step at which
    anything happens, and what happens there", which is all the driver needs to advance.
    """

    def __init__(self, *, timestep_fs, exchange_interval_ps, number_of_exchanges,
                 whole_output_interval_ps=None, solute_output_interval_ps=None,
                 checkpoint_interval_ps=None, equilibration_ps=0.0):
        self.timestep_fs = float(timestep_fs)
        self.number_of_exchanges = int(number_of_exchanges)
        if self.number_of_exchanges < 1:
            raise ScheduleError(
                f"number_of_exchanges must be >= 1; got {self.number_of_exchanges}")

        self.exchange_interval_ps = float(exchange_interval_ps)
        self.exchange_steps = exact_steps(self.exchange_interval_ps, self.timestep_fs,
                                          what="the exchange interval")

        # An absent output interval means "only when the run ends", which is honest: the run still
        # stores a final frame, and nothing pretends a stream exists that does not.
        self.whole_output_interval_ps = (None if whole_output_interval_ps is None
                                         else float(whole_output_interval_ps))
        self.whole_steps = (None if self.whole_output_interval_ps is None else
                            exact_steps(self.whole_output_interval_ps, self.timestep_fs,
                                        what="the whole-system output interval"))
        self.solute_output_interval_ps = (None if solute_output_interval_ps is None
                                          else float(solute_output_interval_ps))
        self.solute_steps = (None if self.solute_output_interval_ps is None else
                             exact_steps(self.solute_output_interval_ps, self.timestep_fs,
                                         what="the solute output interval"))
        # Checkpoints default to the exchange interval: a restart then lands exactly where a
        # transition did, which is the only boundary at which the mapping and the RNG agree.
        self.checkpoint_interval_ps = (self.exchange_interval_ps
                                       if checkpoint_interval_ps is None
                                       else float(checkpoint_interval_ps))
        self.checkpoint_steps = exact_steps(self.checkpoint_interval_ps, self.timestep_fs,
                                            what="the checkpoint interval")

        self.equilibration_ps = float(equilibration_ps)
        self.equilibration_steps = (0 if self.equilibration_ps <= 0 else
                                    exact_steps(self.equilibration_ps, self.timestep_fs,
                                                what="the per-state equilibration duration"))

        #: The run ends after the last exchange attempt. Production is exactly that many
        #: exchange intervals, and nothing is propagated past it.
        self.total_steps = self.exchange_steps * self.number_of_exchanges

    # -- derived, private ---------------------------------------------------------------------

    @property
    def intervals(self):
        return {"exchange": self.exchange_steps, "whole": self.whole_steps,
                "solute": self.solute_steps, "checkpoint": self.checkpoint_steps}

    def step_to_ps(self, step):
        return step * self.timestep_fs / 1000.0

    def events_at(self, step):
        """Which events fall exactly on `step`, in execution order. Never includes step 0."""
        if step <= 0:
            return []
        happening = []
        for kind in EVENT_ORDER:
            period = self.intervals[kind]
            if period is not None and step % period == 0:
                happening.append(kind)
        # The final step always commits a whole frame and a checkpoint, so a run's last state is
        # always both readable and restartable even when the intervals do not divide the budget.
        if step == self.total_steps:
            for kind in ("whole", "checkpoint"):
                if kind not in happening:
                    happening.append(kind)
            happening = [kind for kind in EVENT_ORDER if kind in happening]
        return happening

    def next_event_step(self, step):
        """The next step after `step` at which anything happens, capped at the budget.

        This is the private propagation quantum: the driver advances by `next - current` steps and
        never asks how long that is in picoseconds.
        """
        if step >= self.total_steps:
            return None
        candidates = [self.total_steps]
        for period in self.intervals.values():
            if period is None:
                continue
            candidates.append(((step // period) + 1) * period)
        return min(candidate for candidate in candidates if candidate > step)

    def event_steps(self, kind):
        """Every step at which `kind` happens, for tests and for records. Bounded by the budget."""
        period = self.intervals[kind]
        steps = ([] if period is None else
                 list(range(period, self.total_steps + 1, period)))
        if kind in ("whole", "checkpoint") and self.total_steps not in steps:
            steps.append(self.total_steps)
        return sorted(set(steps))

    def counts(self):
        return {kind: len(self.event_steps(kind)) for kind in EVENT_ORDER}

    def describe(self):
        """What the records store: the requested physical intervals AND their exact step counts."""
        return {
            "timestep_fs": self.timestep_fs,
            "exchange_interval_ps": self.exchange_interval_ps,
            "exchange_interval_steps": self.exchange_steps,
            "whole_output_interval_ps": self.whole_output_interval_ps,
            "whole_output_interval_steps": self.whole_steps,
            "solute_output_interval_ps": self.solute_output_interval_ps,
            "solute_output_interval_steps": self.solute_steps,
            "checkpoint_interval_ps": self.checkpoint_interval_ps,
            "checkpoint_interval_steps": self.checkpoint_steps,
            "number_of_exchanges": self.number_of_exchanges,
            "total_steps": self.total_steps,
            "total_ps": self.step_to_ps(self.total_steps),
            "equilibration_ps": self.equilibration_ps,
            "equilibration_steps": self.equilibration_steps,
            "event_counts": self.counts(),
            "event_order_when_simultaneous": list(EVENT_ORDER),
            "scheduling": ("absolute integration steps; the propagation span between events is "
                           "derived and private, and there is no user-facing segment"),
        }

    def extended(self, additional_exchanges):
        """A schedule for the same run, lengthened by exactly N more exchange attempts."""
        return EventSchedule(
            timestep_fs=self.timestep_fs,
            exchange_interval_ps=self.exchange_interval_ps,
            number_of_exchanges=self.number_of_exchanges + int(additional_exchanges),
            whole_output_interval_ps=self.whole_output_interval_ps,
            solute_output_interval_ps=self.solute_output_interval_ps,
            checkpoint_interval_ps=self.checkpoint_interval_ps,
            equilibration_ps=self.equilibration_ps)
