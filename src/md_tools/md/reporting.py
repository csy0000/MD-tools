"""What a run writes, declared once rather than assembled by hand in every script.

`ReportingConfig` is the public face. It composes specialised writers rather than being one:
ordinary state tables and DCD trajectories, phase-space streams, checkpoints, and -- in
`md_tools.remd` -- NetCDF state trajectories and an Amber `rem.log`. Those are genuinely different
formats with different invariants, and forcing them into a single implementation to reduce the
file count would produce something nobody can change safely.

What is shared is the *contract*: intervals are integer step counts, a marker is written after the
data it describes so it can lag but never lead, and every output is recorded with its digest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ReportingConfig"]


@dataclass(frozen=True)
class ReportingConfig:
    """Intervals, in steps, for everything a stage may write.

    Zero disables a stream. There is no "every N picoseconds" form: a duration that does not divide
    into a whole number of steps has to be rounded, and a rounded reporting interval silently
    changes what a trajectory means.
    """

    trajectory_interval_steps: int = 0
    state_interval_steps: int = 0
    checkpoint_interval_steps: int = 0
    phase_space_interval_steps: int = 0

    @classmethod
    def from_stage(cls, stage: dict[str, Any]) -> "ReportingConfig":
        """Read the intervals a resolved stage declares."""
        return cls(
            trajectory_interval_steps=int(stage.get("trajectory_interval_steps") or 0),
            state_interval_steps=int(stage.get("state_interval_steps") or 0),
            checkpoint_interval_steps=int(stage.get("checkpoint_interval_steps") or 0),
            phase_space_interval_steps=int(stage.get("phase_space_interval_steps") or 0),
        )

    @property
    def writes_trajectory(self) -> bool:
        return self.trajectory_interval_steps > 0

    @property
    def writes_phase_space(self) -> bool:
        return self.phase_space_interval_steps > 0

    def as_record(self) -> dict[str, int]:
        """The intervals as they are recorded, in steps, for the machine record."""
        return {
            "trajectory_interval_steps": self.trajectory_interval_steps,
            "state_interval_steps": self.state_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "phase_space_interval_steps": self.phase_space_interval_steps,
        }
