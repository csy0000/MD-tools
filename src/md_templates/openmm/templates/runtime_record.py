#!/usr/bin/env python
"""The companion runtime record a finished run leaves beside its trajectory.

Copied verbatim into every generated project that produces a trajectory another calculation may
consume. Small on purpose: this is bookkeeping, and it lives here so a generated protocol file
stays the size of the science it describes.

WHY THIS FILE EXISTS AT ALL
    AIS and rREST2 both start from configurations drawn out of a finished equilibrium run, and
    both refuse to guess what that run was. `resolved_run.yaml` is where the answer lives:

        tau                 the Hamiltonian the frames were generated at
        temperature_kelvin  the thermostat they were generated at
        ensemble            NVT or NPT
        frame_time_map      the physical time of every frame

    Each is recorded HERE, by the run that decided it. A directory called `cMD_tau0p5` is a
    convenience for a human reader and is never evidence: it can be renamed or copied, and a
    method seeded from the wrong ensemble completes normally while being wrong. A frame index is
    likewise never a time.
"""
import json
from pathlib import Path

#: Bumped when the meaning of the record changes.
RESOLVED_RUN_FORMAT = "md-templates-resolved-run/v1"

#: The file every consumer looks for beside, or above, a trajectory.
RESOLVED_RUN_NAME = "resolved_run.yaml"


def frame_time_map(interval_ps):
    """When each stored frame happened, given a reporter interval.

    `DCDReporter` writes AT step `interval`, not at step 0, so frame 0 is at one interval of
    physical time and not at zero. Getting this wrong shifts every selected configuration by one
    frame, silently.
    """
    return {
        "first_frame_time_ps": float(interval_ps),
        "frame_interval_ps": float(interval_ps),
        "convention": "DCDReporter writes at step interval; frame i is at (i+1)*interval",
    }


def write_resolved_run(directory, *, method, tau, temperature_kelvin, ensemble, timestep_fs,
                       friction_per_ps, steps, duration_ps, trajectory_name, frames,
                       interval_ps, rest2_implementation=None, extra=None):
    """Write the record. Returns it, so a caller may also print or embed it.

    `rest2_implementation` is the scaling identity this run was propagated under. It matters
    whenever `tau != 0`: v1 and v2 differ in the generalized-Born term, so two fixed-tau walkers
    carrying the same tau can be different energy functions. The identity used to reach only
    `resolved_stage.yaml`, which the `setup` route never writes -- so a fixed-tau walker generated
    that way recorded no identity anywhere, and a downstream gate could not tell which Hamiltonian
    produced it. It belongs in the record that always exists.
    """
    record = {
        "format": RESOLVED_RUN_FORMAT,
        "method": method,
        "tau": float(tau),
        "temperature_kelvin": float(temperature_kelvin),
        "ensemble": ensemble,
        "timestep_fs": float(timestep_fs),
        "friction_per_ps": float(friction_per_ps),
        "steps": int(steps),
        "duration_ps": float(duration_ps),
        "trajectories": {
            "whole_system": {
                "file": trajectory_name,
                "frames": int(frames),
                "frame_time_map": frame_time_map(interval_ps),
            },
        },
    }
    if rest2_implementation is not None:
        record["rest2_implementation"] = dict(rest2_implementation)
    if extra:
        record.update(extra)
    # JSON is valid YAML, and writing it this way keeps the generated protocol free of a YAML
    # dependency it would otherwise need only for this.
    Path(directory).joinpath(RESOLVED_RUN_NAME).write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return record
