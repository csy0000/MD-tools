"""Annealed importance sampling: the switching schedule.

AIS moves `lambda` from 0 to 1 while the coordinates propagate, mixing two end-state Systems as
`V(lambda) = (1 - lambda) V0 + lambda V1` (`md_tools.ais.two_state`), and records the
nonequilibrium work that switching costs.

What lives here is the arithmetic that both `build-md` and the runtime have to agree on: which
lambdas the path visits, when the parameters change, and which of those points are observed.

`lambda` is the one public, persisted coordinate. It always runs 0 -> 1, the Amber convention, and
the source ensemble is always V0's: a reverse switch is expressed by exchanging the two end-state
files, never by a schedule running downhill, so the Jarzynski average is over the V0 ensemble by
construction. The schedule is linear; non-linear schedules are future work, and the work
definition (a finite potential difference at frozen coordinates) is written so they cannot change
what `work` means.
"""
from __future__ import annotations

from typing import Any

#: Endpoint-inclusive: 21 observations are 20 equal intervals plus the starting configuration.
DEFAULT_OBSERVATIONS = 21
#: The endpoints. Not configurable: a partial window has no use without softcore, and refusing it
#: is cheaper than supporting it wrongly.
LAMBDA_START = 0.0
LAMBDA_END = 1.0
#: The only path type and interpolation this implementation supports. Named rather than assumed so
#: a record says which it was.
PATH_TYPE = "two_state_linear"
INTERPOLATION = "linear"
COORDINATE_SCOPE = "whole_system"
SELECTION = "uniform_random"

#: The discrete nonequilibrium work convention, written into every generated record so a reader
#: never has to infer which convention produced a column of numbers.
WORK_CONVENTION = (
    "delta_W_j = V(lambda_{j+1}, x_j) - V(lambda_j, x_j), with V(lambda) = (1 - lambda) V0 "
    "+ lambda V1: the parameters change first, at frozen coordinates, and the configuration then "
    "propagates under the new Hamiltonian. Physical work is in kJ/mol; reduced work is beta*W "
    "with the single common beta of the run temperature. Fixed volume throughout, so no "
    "pressure-volume term is included."
)


def exact_steps(duration_ps: float, timestep_fs: float, *, field: str) -> int:
    """A whole number of steps, or a refusal that names the field.

    Rounding here would run a path of a different length than the one the record describes, and the
    work is path-length dependent.
    """
    exact = float(duration_ps) * 1000.0 / float(timestep_fs)
    if abs(exact - round(exact)) > 1e-9:
        raise ValueError(
            f"{field} = {duration_ps} ps is not a whole number of {timestep_fs} fs steps "
            f"({exact}). Choose a duration that divides exactly.")
    return int(round(exact))


def switching_schedule(*, switching_steps: int,
                       parameter_update_interval_steps: int, observation_interval_steps: int,
                       timestep_fs: float,
                       trajectory_interval_steps: int | None = None,
                       state_interval_steps: int = 0,
                       checkpoint_interval_steps: int = 0,
                       cv_interval_steps: int = 0) -> dict[str, Any]:
    """Every lambda the path visits, and which of them are observed.

    EVERY LENGTH HERE IS AN INTEGER STEP COUNT. A step count is exact; a duration in picoseconds is
    a number that has to divide by a timestep the file may not have been written against, and work
    is path-length dependent, so a rounded path silently reports the work of a different protocol.
    The log derives ps and ns from the resolved timestep for the reader.

    Three counts have to divide exactly into one another, and each failure is refused with the
    arithmetic that would fix it rather than rounded away:

        switching_steps % parameter_update_interval_steps == 0
            otherwise the final parameter change lands mid-interval;
        switching_steps % observation_interval_steps == 0
            otherwise the last observation is not at lambda = 1;
        observation_interval_steps % parameter_update_interval_steps == 0
            otherwise observations are not on the update grid and "evenly spaced in lambda" would be
            evenly spaced only after rounding.

    FOUR INDEPENDENT CADENCES, because they answer four different questions:

        observation_interval_steps    how often the WORK is measured. This one is the method.
        trajectory_interval_steps     how often a configuration is written to the path's NetCDF.
        state_interval_steps          how often the thermodynamic state is tabulated.
        checkpoint_interval_steps     how often the path becomes resumable.

    They were once tied together -- the trajectory had to match the observations -- and that
    answered one question with another's answer. Somebody who wants work every 10 steps and frames
    every 50 is not asking for something confused; they are asking for a small file. Each divides
    `switching_steps` on its own, so every stream has a record on the final step, and 0 disables
    the state table and the checkpoint (never the observations: a path with no work rows is not a
    measurement).

    `lambdas[j]` is the Hamiltonian in force during the j-th propagation interval, so `lambdas[0]`
    is 0 -- V0, the source Hamiltonian, before any change -- and `lambdas[number_of_updates]` is 1.

    OBSERVATION 0 PRECEDES ALL WORK. It is the source configuration under the source Hamiltonian,
    before any parameter change and before any propagation, and its cumulative work is exactly zero
    by definition. Both endpoints are always included: the first row pairs lambda = 0 with zero work,
    the last pairs lambda = 1 with the total.
    """
    steps = int(switching_steps)
    interval = int(parameter_update_interval_steps)
    observe_every = int(observation_interval_steps)
    for label, value in (("switching_steps", steps),
                         ("parameter_update_interval_steps", interval),
                         ("observation_interval_steps", observe_every)):
        if value < 1:
            raise ValueError(f"ais.{label} must be a positive whole number of steps; got {value}")

    if steps % interval:
        raise ValueError(
            f"ais.switching_steps = {steps} is not a whole number of "
            f"parameter_update_interval_steps = {interval}. The final parameter change would land "
            f"mid-interval. Choose a step count divisible by {interval}; the nearest are "
            f"{steps - steps % interval} and {steps - steps % interval + interval}.")
    if steps % observe_every:
        raise ValueError(
            f"ais.switching_steps = {steps} is not a whole number of "
            f"observation_interval_steps = {observe_every}, so the last observation would not fall "
            f"at lambda = 1. Choose a step count divisible by {observe_every}; the nearest are "
            f"{steps - steps % observe_every} and {steps - steps % observe_every + observe_every}.")
    if observe_every % interval:
        raise ValueError(
            f"ais.observation_interval_steps = {observe_every} is not a whole number of "
            f"parameter_update_interval_steps = {interval}, so observations would not land on the "
            f"parameter-update grid and would be evenly spaced only after rounding.")

    # The trajectory follows the observations unless it is given its own cadence -- that is the
    # behaviour every existing AIS project was generated with, and it stays the default.
    frame_every = int(observe_every if trajectory_interval_steps is None
                      else trajectory_interval_steps)
    state_every = int(state_interval_steps or 0)
    checkpoint_every = int(checkpoint_interval_steps or 0)
    for label, value in (("reporting.crd_printout_solute", frame_every),
                         ("reporting.info_printout", state_every),
                         ("reporting.checkpoint_printout", checkpoint_every)):
        if value < 0:
            raise ValueError(f"{label} cannot be negative; got {value}")
        if value and steps % value:
            raise ValueError(
                f"{label} = {value} does not divide ais.switching_steps = {steps} "
                f"({steps} % {value} = {steps % value}). Every enabled stream must have a record "
                f"on the final step, or the end of one path is not comparable with the end of "
                f"another.")
    if frame_every < 1:
        raise ValueError(
            "reporting.crd_printout_solute cannot be 0 for AIS: a switching path with no "
            "configurations written is a work value with nothing to attribute it to.")

    updates = steps // interval
    per_observation = observe_every // interval
    number_of_observations = steps // observe_every + 1

    frame_steps = list(range(0, steps + 1, frame_every))
    state_steps = list(range(0, steps + 1, state_every)) if state_every else []
    # Not step 0: a checkpoint before anything has happened saves nothing worth resuming from.
    checkpoint_steps = ([s for s in range(checkpoint_every, steps + 1, checkpoint_every)]
                        if checkpoint_every else [])

    lambdas = [LAMBDA_START + (LAMBDA_END - LAMBDA_START) * j / updates
               for j in range(updates + 1)]
    # Written back exactly rather than left to accumulate rounding: the last lambda must BE 1, not
    # a float a few ulp away from it, because the final work row is the work of reaching V1.
    lambdas[0] = LAMBDA_START
    lambdas[-1] = LAMBDA_END

    observations = []
    for index in range(number_of_observations):
        update_index = index * per_observation          # 0 is the source, before any update
        observations.append({
            "observation_index": index,
            "updates_completed": update_index,
            "protocol_step": update_index * interval,
            # Derived from the step count and the resolved timestep, for the reader. The step
            # count above is what runs.
            "switching_time_ps": round(update_index * interval * float(timestep_fs) / 1000.0, 9),
            "lambda": lambdas[update_index],
        })

    # Collective variables: on the parameter-update grid, and dividing the switching length.
    #
    # Both, not either. Dividing `switching_steps` alone would allow an observation between two
    # updates, at a lambda the path never actually held -- lambda is piecewise constant across an
    # update interval, so a row there would report a value against a Hamiltonian that was never
    # in force. Sitting on the update grid alone would allow a final partial gap.
    cv_every = int(cv_interval_steps or 0)
    if cv_every:
        if cv_every % interval:
            raise ValueError(
                f"collective_variables.interval_steps = {cv_every} is not a multiple of the "
                f"parameter update interval ({interval} steps). lambda is piecewise constant "
                f"across an update, so an observation between two updates would report a value against a "
                f"Hamiltonian the path never held.")
        if steps % cv_every:
            raise ValueError(
                f"collective_variables.interval_steps = {cv_every} does not divide "
                f"switching_steps ({steps}). It would leave a final partial gap, so the last "
                f"observation would sit at an irregular spacing from the one before it.")

    return {
        "timestep_fs": float(timestep_fs),
        "switching_steps": steps,
        "cv_interval_steps": cv_every,
        "number_of_cv_rows": (steps // cv_every + 1) if cv_every else 0,
        "parameter_update_interval_steps": interval,
        "observation_interval_steps": observe_every,
        "number_of_updates": updates,
        "number_of_observations": number_of_observations,
        "updates_per_observation": per_observation,
        "steps_per_observation": observe_every,
        "trajectory_interval_steps": frame_every,
        "state_interval_steps": state_every,
        "checkpoint_interval_steps": checkpoint_every,
        "frame_steps": frame_steps,
        "state_steps": state_steps,
        "checkpoint_steps": checkpoint_steps,
        "number_of_frames": len(frame_steps),
        "number_of_state_rows": len(state_steps),
        "number_of_checkpoints": len(checkpoint_steps),
        "lambdas": lambdas,
        "observations": observations,
        # Derived for the reader; the step counts above are what runs.
        "switching_ps": steps * float(timestep_fs) / 1000.0,
        "observation_interval_ps": observe_every * float(timestep_fs) / 1000.0,
        "observation_zero_precedes_all_work": True,
        "note": (f"{updates} parameter changes over {steps} integration steps, observed at "
                 f"{number_of_observations} points including both endpoints. An observation is a "
                 f"coordinate frame, not an integration step."),
    }
