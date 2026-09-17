"""Annealed importance sampling: non-equilibrium switching paths between two end states.

    V(lambda, x) = (1 - lambda) V0(x) + lambda V1(x),        lambda 0 -> 1

V0 is `-s`/`-p`, the state the source ensemble was sampled from; V1 is `-s2`/`-p2`, the same
particles with different parameters (`md_tools.ais.two_state`). AIS scales nothing itself: a REST2
switch is a pair of files, the scaled System and the physical one.

    dW_j = V(lambda_{j+1}, x_j) - V(lambda_j, x_j)

Parameters move at frozen coordinates, and only then does the configuration propagate. Observation
0 precedes all work and is exactly zero. Switching is at fixed volume and a barostat in either end
state is refused.

A generated AIS script is an entry point:

    #!/usr/bin/env python
    from md_tools.ais import run_generated_ais
    raise SystemExit(run_generated_ais(__file__))
"""
from __future__ import annotations

from .run import ais_main as run_ais
from .run import run_generated_ais
from .schedule import switching_schedule


__all__ = ["run_generated_ais", "run_ais", "switching_schedule",
           "path_trajectory_name", "paths_for_rank", "path_id_width"]


# ---------------------------------------------------------------------------------------------
# Path identity: which global path owns which file, and which rank runs it.
#
# These are pure functions on (path id, total, rank, size) and deliberately know nothing about
# MPI. Scheduling must never change which global path owns which filename -- a restart with a
# different worker count has to land on the same files, or a resumed campaign silently reshuffles
# which trajectory is which.
# ---------------------------------------------------------------------------------------------

def path_id_width(number_of_paths: int) -> int:
    """Zero-padding width for a path id. At least four digits, wider when the count needs it.

    Wide enough that sorting by filename is sorting by path id: with a narrow field, `traj10`
    sorts before `traj9`, and every tool that globs the directory sees the wrong order.
    """
    return max(4, len(str(max(int(number_of_paths) - 1, 0))))


def path_trajectory_name(path_id: int, number_of_paths: int, *, suffix: str = ".nc") -> str:
    """`AIS_traj0000.nc` … `AIS_traj0099.nc` for 100 paths. Zero-based, as the ids are."""
    if not 0 <= int(path_id) < int(number_of_paths):
        raise ValueError(
            f"path id {path_id} is outside 0..{int(number_of_paths) - 1}")
    return f"AIS_traj{int(path_id):0{path_id_width(number_of_paths)}d}{suffix}"


def paths_for_rank(rank: int, size: int, number_of_paths: int) -> list[int]:
    """The global path ids this worker runs.

    Round-robin from the global id, so the assignment is a pure function of `(rank, size, total)`:
    every path is owned by exactly one rank for a given world size, and the union over ranks is
    always the complete set. `number_of_paths` is the GLOBAL total, never a per-rank count.
    """
    rank, size = int(rank), int(size)
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"rank {rank} is not valid in a world of {size}")
    return list(range(rank, int(number_of_paths), size))
