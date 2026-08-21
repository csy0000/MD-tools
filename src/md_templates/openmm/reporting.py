"""Attaching the reporters a stage declares, in one place.

A declared output that is never written is worse than a missing feature. The staged project used to
record `full_system_interval_steps` and `selected_atoms_interval_steps` in every stage
configuration, list a `.log`, and attach no reporters at all -- so the manifest described a
trajectory that did not exist, and nothing downstream noticed until someone went looking for frames.

This module owns the wrapping convention, which is a scientific choice and not a formatting one:

* **All-atom trajectories are wrapped into the box** (`enforcePeriodicBox=True`), the usual
  convention for a solvated trajectory.
* **Selected-atom trajectories are not wrapped.** OpenMM wraps whole *molecules*, so bonds are
  never broken either way. What wrapping does is teleport the solute across the box whenever its
  centre crosses a face, which breaks every analysis that reads the trajectory as continuous --
  RMSD without re-imaging, diffusion, Cartesian TICA -- and makes the structure jump in a viewer.
  Unwrapped, the solute may drift far from the origin, which nothing here cares about.

Both statements were established for the chunked production path; this module exists so the staged
path cannot drift away from them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

__all__ = ["attach_reporters", "exact_steps"]


def exact_steps(interval, *, what: str) -> int:
    """A reporter interval as a whole number of steps, or a refusal.

    Rounding here would silently change the sampling frequency, and a trajectory whose cadence is
    not what the configuration says is unusable as evidence for anything measured from it.
    """
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise ValueError(f"{what} must be a number of steps, got {interval!r}")
    if abs(float(interval) - round(float(interval))) > 1e-9:
        raise ValueError(
            f"{what} = {interval} is not a whole number of steps. Choose an interval divisible by "
            "the timestep: a rounded reporter interval silently changes the sampling frequency."
        )
    steps = int(round(float(interval)))
    if steps <= 0:
        raise ValueError(f"{what} must be at least one step, got {steps}")
    return steps


def attach_reporters(
    sim,
    *,
    all_atom_path: Optional[Path] = None,
    all_atom_interval_steps: Optional[int] = None,
    selected_path: Optional[Path] = None,
    selected_interval_steps: Optional[int] = None,
    selected_atoms: Optional[Sequence[int]] = None,
    state_path: Optional[Path] = None,
    state_interval_steps: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    append: bool = False,
    periodic: bool = True,
) -> dict:
    """Attach the declared reporters and return what was attached.

    Returns a record naming each file and its cadence, so the stage's own results can state what it
    wrote rather than what it intended to write.

    A reporter whose interval exceeds the length of the stage would produce an empty trajectory. That
    is reported rather than silently accepted: an empty file is indistinguishable from a broken one.
    """
    from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter

    attached: dict = {}
    # `periodic` comes from the SYSTEM, not from whether a State happens to return box vectors. A
    # nonperiodic Context still exposes default vectors, so asking a State is not a test of
    # periodicity -- it is a test of whether OpenMM filled in a default, and it always has.
    sim.reporters.clear()

    def _check(steps: int, label: str) -> None:
        if total_steps is not None and steps > total_steps:
            raise ValueError(
                f"{label} interval is {steps} steps but this stage runs {total_steps}, so no frame "
                "would ever be written. Shorten the interval or lengthen the stage."
            )

    if all_atom_path is not None and all_atom_interval_steps:
        steps = exact_steps(all_atom_interval_steps, what="all-atom reporting interval")
        _check(steps, "all-atom trajectory")
        # Wrapping a nonperiodic system is meaningless: there is no box to wrap into, and writing
        # the trajectory as though there were labels it periodic for every downstream reader.
        sim.reporters.append(
            DCDReporter(str(all_atom_path), steps, append=append, enforcePeriodicBox=periodic))
        attached["all_atom"] = {"file": all_atom_path.name, "interval_steps": steps,
                                "wrapped": bool(periodic)}

    if selected_path is not None and selected_interval_steps and selected_atoms:
        steps = exact_steps(selected_interval_steps, what="selected-atom reporting interval")
        _check(steps, "selected-atom trajectory")
        subset = [int(i) for i in selected_atoms]
        sim.reporters.append(
            DCDReporter(str(selected_path), steps, append=append, enforcePeriodicBox=False,
                        atomSubset=subset))
        attached["selected_atoms"] = {"file": selected_path.name, "interval_steps": steps,
                                      "n_atoms": len(subset), "wrapped": False}

    if state_path is not None and state_interval_steps:
        steps = exact_steps(state_interval_steps, what="state-data reporting interval")
        _check(steps, "state log")
        # `append` also decides whether a header is written, so a resumed stage does not repeat it.
        # Volume and density are requested only when there is a box. Without one, OpenMM would
        # still emit numbers -- computed from the default vectors -- and a log column full of
        # fictional volumes is worse than a missing column, because it reads as a measurement.
        sim.reporters.append(
            StateDataReporter(str(state_path), steps, step=True, time=True, potentialEnergy=True,
                              kineticEnergy=True, temperature=True, volume=bool(periodic),
                              density=bool(periodic), speed=True, append=append))
        attached["state_log"] = {"file": state_path.name, "interval_steps": steps,
                                 "volume_and_density": bool(periodic)}

    if checkpoint_path is not None and checkpoint_interval_steps:
        steps = exact_steps(checkpoint_interval_steps, what="checkpoint interval")
        sim.reporters.append(CheckpointReporter(str(checkpoint_path), steps))
        attached["checkpoint"] = {"file": checkpoint_path.name, "interval_steps": steps}

    return attached
