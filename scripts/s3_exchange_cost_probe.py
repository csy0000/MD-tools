#!/usr/bin/env python
"""What does a cross-lambda energy evaluation cost, against the MD it interrupts?

Development evidence for the 0.7.0 lambda-exchange milestone (A3b), not package code.

A nearest-neighbour exchange attempt between windows j and j+1 needs four reduced potentials:
u_j(x_j), u_j(x_{j+1}), u_{j+1}(x_j), u_{j+1}(x_{j+1}). With lambda a CONTEXT PARAMETER rather than
a baked System, each rank can produce its own two by setting parameters and asking for an energy --
no second System, no second Context, no reinitialisation. This measures what that actually costs,
because the design question ("can we attempt exchanges every N steps?") is a ratio, not a principle.

THE EXPECTATION, written before running: a cross-lambda evaluation costs about one force
evaluation, so ~1 MD step, and the exchange overhead at an interval of N steps is ~2/N of run time.
A measured cost far above one step means something is being REBUILT and not merely re-parameterised
-- the candidate being OpenMM's long-range dispersion correction, which a CustomNonbondedForce
re-integrates by quadrature whenever a parameter it depends on changes. That is why the probe
separates a lambda_sterics move (touches the dispersion carrier) from a lambda_electrostatics-only
move (does not).

Platform CPU by default and NOT CUDA evidence.

WHAT THE FIRST RUN (2026-09-21) SHOWED ABOUT THIS PROBE ITSELF, which anyone reading its numbers
needs: on the 102-particle fixture the plain end-state System A also costs 17.7 ms per step and
19.2 ms for a single energy. A 102-particle system does not need 17 ms of arithmetic, so on this
fixture BOTH the step and the energy are dominated by fixed per-call overhead, and "a cross-lambda
energy costs 1.17 steps" is largely a statement that two overheads are equal. The QUALITATIVE
finding transfers -- a neighbour evaluation costs the same as a same-lambda one, and an
electrostatics-only move costs the same as a full one, so nothing is being rebuilt on a parameter
change -- and the RATIO does not. A cadence must be chosen from a measurement on a card with a
campaign-sized system, where the hybrid's 11 forces against an end state's 4 (measured here as 2.0x
per step) and the dispersion carrier's quadrature will both behave differently.

Usage: python scripts/s3_exchange_cost_probe.py [--platform CPU] [--steps 200] [--repeats 20]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def main():
    import openmm
    from md_tools.alchemy.hamiltonian import build_hamiltonian
    from tests import alchemy_s3_fixture as fx

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--steps", type=int, default=200, help="steps per timed block")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    sa, sb, a, b, x = fx.build(True, dispersion=True)
    h = build_hamiltonian(sa, sb, a, b)
    platform = openmm.Platform.getPlatformByName(args.platform)
    c = openmm.Context(h.system, openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001), platform)
    c.setPositions(x)
    here = dict(zip(NAMES, (0.5, 0.5, 0.5)))
    h.set_state(c, here)
    openmm.LocalEnergyMinimizer.minimize(c, 10.0, 200)

    def timed(fn, repeats):
        fn()                                    # warm up: first call pays for lazy setup
        out = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return _median(out)

    integrator = c.getIntegrator()
    per_step = timed(lambda: integrator.step(args.steps), max(3, args.repeats // 4)) / args.steps
    same = timed(lambda: h.energy(c, here), args.repeats)
    # the neighbour a real ladder would have: a small move in every component
    neighbour = dict(zip(NAMES, (0.55, 0.55, 0.55)))
    def cross():
        h.energy(c, neighbour)
        h.set_state(c, here)
    both = timed(cross, args.repeats)
    # electrostatics only: no lambda_sterics move, so the dispersion carrier is untouched
    elec_only = dict(here, lambda_electrostatics=0.55)
    def cross_elec():
        h.energy(c, elec_only)
        h.set_state(c, here)
    elec = timed(cross_elec, args.repeats)

    print(f"\nplatform {args.platform}, {h.system.getNumParticles()} particles, "
          f"median of {args.repeats}\n")
    print(f"  one MD step                              {per_step*1e3:8.3f} ms")
    print(f"  energy at the CURRENT lambda             {same*1e3:8.3f} ms   "
          f"= {same/per_step:6.2f} steps")
    print(f"  energy at a NEIGHBOUR lambda + restore   {both*1e3:8.3f} ms   "
          f"= {both/per_step:6.2f} steps   <- what an attempt needs, twice")
    print(f"  ... electrostatics only (no sterics)     {elec*1e3:8.3f} ms   "
          f"= {elec/per_step:6.2f} steps")
    attempt = 2 * both
    print(f"\n  one attempt (two cross evaluations)      {attempt*1e3:8.3f} ms "
          f"= {attempt/per_step:6.2f} steps")
    for interval in (250, 500, 1000, 2500):
        print(f"    attempted every {interval:5d} steps -> "
              f"{100*attempt/(interval*per_step):5.2f}% overhead")
    if both > 8 * same:
        print("\n  WARNING: a neighbour evaluation costs far more than an evaluation at the same "
              "lambda.\n  Something is being REBUILT on a parameter change, not re-parameterised; "
              "suspect the\n  dispersion carrier's quadrature. Compare the electrostatics-only row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
