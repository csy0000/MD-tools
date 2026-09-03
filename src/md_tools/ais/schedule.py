"""Annealed importance sampling: the path definition, and the schedule derived from it.

AIS switches the REST2 Hamiltonian along a path in `tau` while the coordinates propagate, and
records the nonequilibrium work that switching costs. It is the same Hamiltonian decomposition
REST2 uses -- `s = (1 - tau)^2` for solute-solute terms, `sqrt(s) = 1 - tau` for solute-environment
terms, torsions about an omega bond left alone -- driven through the parameters of a live Context
instead of built once per rung.

What lives here is the arithmetic that both `build-md` and the generated runtime have to agree on:
which taus the path visits, when the parameters change, and which of those points are observed. The
generated project cannot import this module, so `build-md` writes the resulting schedule into
`AIS/path_definition.yaml` and the runtime reads it. Computing it once and recording it, rather than
recomputing it at both ends, is what stops a project from observing a schedule its own record does
not describe.

`tau` is the SOURCE parameter. `s`, `sqrt(s)` and the effective solute temperature are derived and
are never accepted back as input.

Not implemented here, deliberately: the reverse path, pV work, and any estimator built on the
resulting work values. See configs/md/AIS.config for the path and the work convention.

Mid-path restart WAS on that list and no longer is: it lives in `md_tools.openmm.checkpoint` and
`md_tools.ais.run`, not here, because it is a property of the runtime rather than of the schedule
this module computes.
"""
from __future__ import annotations

from typing import Any

#: The default forward path. tau = 0.5 is the scaled end (s = 0.25), tau = 0 the physical one.
DEFAULT_TAU_START = 0.5
DEFAULT_TAU_END = 0.0
#: Endpoint-inclusive: 21 observations are 20 equal intervals plus the starting configuration.
DEFAULT_OBSERVATIONS = 21
#: The only path type and interpolation this implementation supports. Named rather than assumed so
#: a configuration that asks for something else is refused instead of silently getting this.
PATH_TYPE = "rest2_tau"
INTERPOLATION = "linear"
ENHANCED_REGION = "solute"
COORDINATE_SCOPE = "whole_system"
SELECTION = "uniform_random"

#: The discrete nonequilibrium work convention, written into every generated record so a reader
#: never has to infer which of the two conventions produced a column of numbers.
WORK_CONVENTION = (
    "delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j): the parameters change first, at frozen "
    "coordinates, and the configuration then propagates under the new Hamiltonian. Physical work "
    "is in kJ/mol; reduced work is beta*W with the single common beta of "
    "common.temperature_kelvin. Fixed volume throughout, so no pressure-volume term is included."
)


def scale_factor_for_tau(tau: float) -> float:
    """`s = (1 - tau)^2`. The one definition; the generated runtime uses the copied REST2 module."""
    tau = float(tau)
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be in [0, 1); got {tau}")
    return (1.0 - tau) ** 2


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


def switching_schedule(*, tau_start: float, tau_end: float, switching_steps: int,
                       parameter_update_interval_steps: int, observation_interval_steps: int,
                       timestep_fs: float,
                       trajectory_interval_steps: int | None = None,
                       state_interval_steps: int = 0,
                       checkpoint_interval_steps: int = 0,
                       cv_interval_steps: int = 0) -> dict[str, Any]:
    """Every tau the path visits, and which of them are observed.

    EVERY LENGTH HERE IS AN INTEGER STEP COUNT. A step count is exact; a duration in picoseconds is
    a number that has to divide by a timestep the file may not have been written against, and work
    is path-length dependent, so a rounded path silently reports the work of a different protocol.
    The log derives ps and ns from the resolved timestep for the reader.

    Three counts have to divide exactly into one another, and each failure is refused with the
    arithmetic that would fix it rather than rounded away:

        switching_steps % parameter_update_interval_steps == 0
            otherwise the final parameter change lands mid-interval;
        switching_steps % observation_interval_steps == 0
            otherwise the last observation is not at tau_end;
        observation_interval_steps % parameter_update_interval_steps == 0
            otherwise observations are not on the update grid and "evenly spaced in tau" would be
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

    `taus[j]` is the Hamiltonian in force during the j-th propagation interval, so `taus[0]` is
    tau_start -- the source Hamiltonian, before any change -- and `taus[number_of_updates]` is
    tau_end.

    OBSERVATION 0 PRECEDES ALL WORK. It is the source configuration under the source Hamiltonian,
    before any parameter change and before any propagation, and its cumulative work is exactly zero
    by definition. Both endpoints are always included: the first row pairs tau_start with zero work,
    the last pairs tau_end with the total.
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
            f"at tau_end. Choose a step count divisible by {observe_every}; the nearest are "
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
    for label, value in (("reporting.solute_printout", frame_every),
                         ("reporting.system_printout", state_every),
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
            "reporting.solute_printout cannot be 0 for AIS: a switching path with no "
            "configurations written is a work value with nothing to attribute it to.")

    updates = steps // interval
    per_observation = observe_every // interval
    number_of_observations = steps // observe_every + 1

    frame_steps = list(range(0, steps + 1, frame_every))
    state_steps = list(range(0, steps + 1, state_every)) if state_every else []
    # Not step 0: a checkpoint before anything has happened saves nothing worth resuming from.
    checkpoint_steps = ([s for s in range(checkpoint_every, steps + 1, checkpoint_every)]
                        if checkpoint_every else [])

    span = float(tau_end) - float(tau_start)
    taus = [float(tau_start) + span * j / updates for j in range(updates + 1)]
    # Written back exactly rather than left to accumulate rounding across `updates` additions: the
    # last tau must BE tau_end, not a float a few ulp away from it, because the final work row is
    # reported as the work of reaching tau_end.
    taus[0] = float(tau_start)
    taus[-1] = float(tau_end)

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
            "tau": taus[update_index],
            # Deliberately NOT recording s or sqrt(s). tau is the one public, persisted protocol
            # coordinate; the scale factors are derived from it inside the scaler and are not an
            # alternative coordinate a reader could take as authoritative.
        })

    # Collective variables: on the parameter-update grid, and dividing the switching length.
    #
    # Both, not either. Dividing `switching_steps` alone would allow an observation between two
    # updates, at a tau the path never actually held -- the tau is piecewise constant across an
    # update interval, so a row there would report a value against a Hamiltonian that was never
    # in force. Sitting on the update grid alone would allow a final partial gap.
    cv_every = int(cv_interval_steps or 0)
    if cv_every:
        if cv_every % interval:
            raise ValueError(
                f"collective_variables.interval_steps = {cv_every} is not a multiple of the "
                f"parameter update interval ({interval} steps). tau is piecewise constant across "
                f"an update, so an observation between two updates would report a value against a "
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
        "taus": taus,
        "observations": observations,
        # Derived for the reader; the step counts above are what runs.
        "switching_ps": steps * float(timestep_fs) / 1000.0,
        "observation_interval_ps": observe_every * float(timestep_fs) / 1000.0,
        "observation_zero_precedes_all_work": True,
        "note": (f"{updates} parameter changes over {steps} integration steps, observed at "
                 f"{number_of_observations} points including both endpoints. An observation is a "
                 f"coordinate frame, not an integration step."),
    }


def resolve_source_topology(source: dict[str, Any]) -> str:
    """`inputs/topology.pdb` when the user left `source.topology` null, and the choice is recorded.

    Defaulting is fine here and only here: the prepared topology is the one the System was built
    from, so it is the only topology whose atom order can match. What is not fine is defaulting
    silently, which is why the resolved value is written into the AIS records.
    """
    topology = source.get("topology")
    return str(topology) if topology else "inputs/topology.pdb"
