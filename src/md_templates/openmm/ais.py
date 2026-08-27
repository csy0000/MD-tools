"""Annealed importance sampling: the path definition, and the schedule derived from it.

AIS switches the REST2 Hamiltonian along a path in `tau` while the coordinates propagate, and
records the nonequilibrium work that switching costs. It is the same Hamiltonian decomposition
REST2 uses -- `s = (1 - tau)^2` for solute-solute terms, `sqrt(s) = 1 - tau` for solute-environment
terms, torsions about an omega bond left alone -- driven through the parameters of a live Context
instead of built once per rung.

What lives here is the arithmetic that both `md-gen` and the generated runtime have to agree on:
which taus the path visits, when the parameters change, and which of those points are observed. The
generated project cannot import this module, so `md-gen` writes the resulting schedule into
`AIS/path_definition.yaml` and the runtime reads it. Computing it once and recording it, rather than
recomputing it at both ends, is what stops a project from observing a schedule its own record does
not describe.

`tau` is the SOURCE parameter. `s`, `sqrt(s)` and the effective solute temperature are derived and
are never accepted back as input.

Not implemented here, deliberately: the reverse path, mid-path restart, pV work, and any estimator
built on the resulting work values. See `docs/journal/2026-08-27_ais-method-and-release-gaps.md`.
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


def switching_schedule(*, tau_start: float, tau_end: float, switching_duration_ps: float,
                       parameter_update_interval_steps: int, number_of_observations: int,
                       timestep_fs: float) -> dict[str, Any]:
    """Every tau the path visits, and which of them are observed.

    The three quantities have to divide exactly into one another:

        total_steps            = switching_duration_ps at this timestep
        number_of_updates      = total_steps / parameter_update_interval_steps
        updates_per_observation = number_of_updates / (number_of_observations - 1)

    If the second division is inexact the last parameter change lands mid-interval; if the third is
    inexact the observation points are not evenly spaced in tau and 21 "evenly spaced" observations
    would be 21 rounded ones. Both are refused rather than rounded, which is why the configuration
    asks for a duration that fits rather than for a number of updates.

    `taus[j]` is the Hamiltonian in force during the j-th propagation interval, so `taus[0]` is
    tau_start (the source Hamiltonian, before any change) and `taus[number_of_updates]` is tau_end.
    """
    total_steps = exact_steps(switching_duration_ps, timestep_fs,
                             field="AIS.path.switching_duration_ps")
    interval = int(parameter_update_interval_steps)
    if total_steps % interval:
        raise ValueError(
            f"AIS.path.switching_duration_ps = {switching_duration_ps} ps is {total_steps} steps "
            f"at {timestep_fs} fs, which is not a whole number of "
            f"parameter_update_interval_steps = {interval}. The final parameter change would land "
            f"mid-interval. Choose a duration whose step count is divisible by {interval}.")
    updates = total_steps // interval

    intervals = int(number_of_observations) - 1
    if updates % intervals:
        raise ValueError(
            f"AIS: {updates} switching updates cannot be divided into "
            f"{intervals} equal observation intervals "
            f"(number_of_observations = {number_of_observations}, both endpoints included). "
            f"The {number_of_observations} observations would be rounded onto the update grid "
            f"instead of evenly spaced in tau. Choose a switching_duration_ps whose update count "
            f"is divisible by {intervals} -- at {timestep_fs} fs and an update every "
            f"{interval} step(s) that is a multiple of "
            f"{intervals * interval * timestep_fs / 1000.0:g} ps.")
    per_observation = updates // intervals

    span = float(tau_end) - float(tau_start)
    taus = [float(tau_start) + span * j / updates for j in range(updates + 1)]
    # Written back exactly rather than left to accumulate rounding across `updates` additions: the
    # last tau must BE tau_end, not a float a few ulp away from it, because the final work row is
    # reported as the work of reaching tau_end.
    taus[0] = float(tau_start)
    taus[-1] = float(tau_end)

    observations = []
    for index in range(int(number_of_observations)):
        update_index = index * per_observation          # 0 is the source, before any update
        observations.append({
            "observation_index": index,
            "updates_completed": update_index,
            "protocol_step": update_index * interval,
            "switching_time_ps": round(update_index * interval * float(timestep_fs) / 1000.0, 9),
            "tau": taus[update_index],
            "s": scale_factor_for_tau(taus[update_index]),
            "sqrt_s": 1.0 - taus[update_index],
        })

    return {
        "timestep_fs": float(timestep_fs),
        "total_steps": total_steps,
        "parameter_update_interval_steps": interval,
        "number_of_updates": updates,
        "number_of_observations": int(number_of_observations),
        "updates_per_observation": per_observation,
        "steps_per_observation": per_observation * interval,
        "taus": taus,
        "observations": observations,
        # Stated because "21 observations" and "21 steps" are different numbers and the difference
        # is the whole point of a switching path.
        "note": (f"{updates} parameter changes over {total_steps} integration steps, observed at "
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
