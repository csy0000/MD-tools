"""The OpenMM adapter for collective-variable reporting.

WHY THE ADAPTER IS HERE AND NOT IN `md_tools.cv`

    `md_tools.cv` imports no OpenMM at all, and a test asserts it -- through the AST, so the
    modules can explain at length why `CustomTorsionForce` is the wrong tool without the
    explanation tripping the check. That independence is what makes "enabling CVs changes no
    energy, no force group and no coordinate" testable rather than merely intended.

    Something still has to speak OpenMM's reporter protocol, so it lives here, in the package that
    already does. It is a thin shim: get positions, hand them to `CVSeries`, write a row.

POSITIONS ONLY, DELIBERATELY

    `describeNextReport` asks for positions and nothing else -- no velocities, no forces, and
    above all NO ENERGY. Requesting energy would make the Context evaluate the Hamiltonian at
    every CV observation, which is both the expense the design avoids and the thing that would
    make a position-only torsion indistinguishable from an energy evaluation in the run's own
    cost accounting.

STEP 0 IS NOT A REPORTER'S JOB

    OpenMM reporters fire after steps, never before the first one, so step 0 -- the initial
    configuration, which the schedule requires exactly once -- cannot come from this class. The
    caller writes it before `simulation.step()` begins. Putting it here instead would mean either
    a reporter that fires on construction (which OpenMM has no way to express) or a first interval
    of a different length, and the second silently breaks the uniform spacing the schedule exists
    to guarantee.
"""

from __future__ import annotations

import numpy as np


class CVReporter:
    """Writes one CV row every `interval` steps, from positions alone.

    Plugged into `simulation.reporters` exactly like a DCDReporter. Records the ABSOLUTE step, so
    an observation stays traceable to the step of the run that produced it across a restart.
    """

    def __init__(self, series, interval_steps, *, periodic, timestep_fs,
                 frame_index_for_step=None, start_step=0):
        self.series = series
        self.interval = int(interval_steps)
        if self.interval < 1:
            raise ValueError(f"the collective-variable interval must be >= 1 step; "
                             f"got {interval_steps}")
        self.periodic = bool(periodic)
        self.timestep_fs = float(timestep_fs)
        #: The ABSOLUTE step this stage began at. The cadence is measured from here rather than
        #: from step 0 of the run: see `describeNextReport`.
        self.start_step = int(start_step)
        #: Maps an absolute step to the trajectory frame written at it, or None when there is no
        #: frame there. Supplied by the caller, which is what knows the trajectory's cadence.
        self.frame_index_for_step = frame_index_for_step or (lambda _step: None)

    def describeNextReport(self, simulation):          # noqa: N802 - OpenMM's interface
        # THE CADENCE IS THE STAGE'S, measured from the step the stage began at.
        #
        # An OpenMM reporter ordinarily fires on the absolute grid -- `interval - currentStep %
        # interval` -- which is what every other reporter here does, and for a trajectory or a
        # state table that is right: they promise a cadence and nothing about endpoints.
        #
        # A CV series promises more, and `cv/schedule.py` refuses an interval that does not
        # divide the stage's step count so that it can: step 0 and the final step each appear
        # EXACTLY ONCE, with uniform spacing throughout, because every downstream time-series
        # analysis assumes uniform spacing and none can detect a violation. On the absolute grid
        # those promises hold only when the stage happens to begin on a multiple of the interval.
        # A stage beginning at 1500 with an interval of 400 would be observed at 1600, 2000, ...
        # -- a first gap of 100 against a cadence of 400 -- and its final step, 1500 + the stage
        # length, is not a multiple of 400 and would never be observed at all.
        #
        # Measuring from `start_step` makes the grid the one the schedule actually checked. Where
        # the stage does begin on a multiple, which is the ordinary case, this is the same grid
        # as before and no existing series changes.
        steps = self.interval - (simulation.currentStep - self.start_step) % self.interval
        # positions, velocities, forces, energies, wrapped. Only the first is True: see the
        # module note on why asking for energy here would be a category error.
        return (steps, True, False, False, False, self.periodic or None)

    def report(self, simulation, state):
        # `simulation.currentStep` IS the absolute step, and is the only authority for it.
        #
        # There used to be a `step_offset=done` added to it here. OpenMM's `loadCheckpoint`
        # RESTORES the Context's step count -- a checkpoint taken at step 25 comes back with
        # `currentStep == 25` -- so adding the already-completed count a second time labelled the
        # first observation after a resume `2N + interval`. Measured: a resume from step 30 of a
        # 40-step run put its next observation at step 65, past the budget entirely, in a file
        # with the right header, the right column count and plausible monotonic numbers.
        #
        # The parameter is REMOVED rather than defaulted to zero. An inert argument that used to
        # mean something is the next person's bug: two step conventions in one runtime is the
        # defect, not the value that was passed.
        self.observe(state, int(simulation.currentStep))

    # -- shared with the caller's step-0 write -------------------------------------------------

    def observe(self, state, step):
        """One observation at `step`, from an already-retrieved State. Also used for step 0."""
        from openmm import unit

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = None
        if self.periodic:
            box = np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                unit.nanometer), dtype=float)
        values = self.series.evaluate(positions, box)
        self.series.write(
            (step, step * self.timestep_fs / 1000.0, self.frame_index_for_step(step)), values)

    def close(self):
        self.series.close()
