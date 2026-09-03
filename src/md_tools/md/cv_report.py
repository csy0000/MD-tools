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

    def __init__(self, series, interval_steps, *, periodic, timestep_fs, step_offset=0,
                 frame_index_for_step=None):
        self.series = series
        self.interval = int(interval_steps)
        if self.interval < 1:
            raise ValueError(f"the collective-variable interval must be >= 1 step; "
                             f"got {interval_steps}")
        self.periodic = bool(periodic)
        self.timestep_fs = float(timestep_fs)
        self.step_offset = int(step_offset)
        #: Maps an absolute step to the trajectory frame written at it, or None when there is no
        #: frame there. Supplied by the caller, which is what knows the trajectory's cadence.
        self.frame_index_for_step = frame_index_for_step or (lambda _step: None)

    def describeNextReport(self, simulation):          # noqa: N802 - OpenMM's interface
        steps = self.interval - simulation.currentStep % self.interval
        # positions, velocities, forces, energies, wrapped. Only the first is True: see the
        # module note on why asking for energy here would be a category error.
        return (steps, True, False, False, False, self.periodic or None)

    def report(self, simulation, state):
        step = self.step_offset + int(simulation.currentStep)
        self.observe(state, step)

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
