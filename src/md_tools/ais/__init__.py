"""Annealed importance sampling: non-equilibrium switching paths and their work.

AIS scales the Hamiltonian with the SAME `md_tools.rest2.REST2Scaler` that REST2 uses. There is no
second AIS scaler, which is what makes "the same scaling everywhere" true rather than intended:
a correction to the scaling rules cannot reach the ladder and miss the switching paths.

What is specific to AIS is the *schedule* and the *work convention*, and those live here:

    dW_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)

Parameters move at frozen coordinates, and only then does the configuration propagate. Observation
0 precedes all work and is exactly zero. Switching is at fixed volume and a barostat in the
prepared System is refused.

A generated AIS script is an entry point:

    #!/usr/bin/env python
    from md_tools.ais import run_generated_ais
    raise SystemExit(run_generated_ais(__file__))
"""
from __future__ import annotations

from .run import ais_main as run_ais
from .run import run_generated_ais
from .schedule import switching_schedule

__all__ = ["run_generated_ais", "run_ais", "switching_schedule"]
