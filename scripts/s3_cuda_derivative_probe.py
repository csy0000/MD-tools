#!/usr/bin/env python
"""S3 diagnostic: is a CUDA-vs-Reference derivative difference platform arithmetic or a defect?

Development evidence for 0.7.0 A2, not package code. On S3's tail fixture (PME, dispersion on),
per force group and per lambda component, it reports:

  * E_CUDA - E_Reference, per group, at the state;
  * the analytic dU/dlambda_k per group on CUDA and on Reference (`derivative_components`);
  * a central / one-sided finite difference of each group's energy ON CUDA and ON REFERENCE,
    h = 1e-3 and 1e-4, and Richardson.

If CUDA's own finite difference agrees with CUDA's analytic derivative while both differ from
Reference, the difference is platform arithmetic (float intermediates, PME). If CUDA's finite
difference disagrees with CUDA's analytic derivative, the derivative is defective on CUDA.

PME settings are shared by construction: the Hamiltonian fixes alpha and the grid explicitly on
both end-state NonbondedForces, and the same System runs on both platforms.

Usage: CUDA_VISIBLE_DEVICES=<card> python scripts/s3_cuda_derivative_probe.py [--precision double]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")


def main():
    import openmm
    from md_tools.alchemy.hamiltonian import FORCE_GROUPS, build_hamiltonian
    from tests import alchemy_s3_fixture as fx

    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="double", choices=("mixed", "double"))
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=True)
    h = build_hamiltonian(sa, sb, a, b)
    ctx = {}
    for name, props in (("CUDA", {"Precision": args.precision}), ("Reference", {})):
        c = openmm.Context(h.system, openmm.VerletIntegrator(0.001),
                           openmm.Platform.getPlatformByName(name), props)
        assert c.getPlatform().getName() == name
        c.setPositions(x)
        ctx[name] = c

    def group_energy(c, t, g):
        h.set_state(c, dict(zip(NAMES, t)))
        return c.getState(getEnergy=True, groups={g}).getPotentialEnergy()._value

    lam = args.lam
    base = (lam, lam, lam)
    out = {"precision": args.precision, "lambda": lam, "groups": {}}
    parts = {p: h.derivative_components(c, dict(zip(NAMES, base))) for p, c in ctx.items()}
    for gname, g in FORCE_GROUPS.items():
        row = {"dE_cuda_minus_ref": group_energy(ctx["CUDA"], base, g)
               - group_energy(ctx["Reference"], base, g)}
        for k, comp in enumerate(NAMES):
            analytic = {p: parts[p][comp].get(gname) for p in ctx}
            if all(v is None for v in analytic.values()):
                continue
            fds = {}
            for p, c in ctx.items():
                vals = []
                for step in (1e-3, 1e-4):
                    def f(v, k=k):
                        t = list(base)
                        t[k] = v
                        return group_energy(c, t, g)
                    if lam + step > 1:
                        d = (3 * f(lam) - 4 * f(lam - step) + f(lam - 2 * step)) / (2 * step)
                    elif lam - step < 0:
                        d = (-3 * f(lam) + 4 * f(lam + step) - f(lam + 2 * step)) / (2 * step)
                    else:
                        d = (f(lam + step) - f(lam - step)) / (2 * step)
                    vals.append(d)
                fds[p] = {"fd_1e-3": vals[0], "fd_1e-4": vals[1],
                          "richardson": (100 * vals[1] - vals[0]) / 99}
            row[comp] = {"analytic": analytic, "fd": fds}
        out["groups"][gname] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
