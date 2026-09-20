#!/usr/bin/env python
"""Where a plan's bonded integrand comes from, with an expectation attached.

Development diagnostic for 0.7.0/0.7.1, not package code. For each topology plan it prints
dU/dlambda_bonded and splits it into the part contributed by terms that touch a unique atom and the
part from genuine core parameter changes, and it WARNS when the first is not ~0.

Why it warns rather than tabulates: on 2026-09-19 a probe of mine printed 660.77 kJ/mol for S2's
ethane->chloroethane plan and it was read as "large but plausible bonded mixing". It was the entire
bonded integrand arriving from junction terms switched between zero and full strength, and it took
S4's M2 gates -- unresolvable end windows, four legs under the overlap floor, repeats scattering
0.80 kcal/mol, a cycle closing at 0.242 against 0.168 -- to make the consequence visible. A
diagnostic without an expectation cannot be surprised, and one that cannot be surprised cannot warn
anybody.

Usage: python scripts/s3_bonded_derivative_probe.py [--lambda 0.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")


def main():
    import openmm
    from md_tools.alchemy.hamiltonian import from_plan
    from md_tools.alchemy.topology import build_topology_plan
    from tests import alchemy_fixtures as af
    from tests import alchemy_s3_fixture as fx

    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--policy", default=None, choices=(None, "retain-all", "separable"),
                    help="the plan's junction_policy, when the plan record does not carry one. It "
                         "decides the EXPECTATION: retain-all requires exactly 0 from junction "
                         "terms, separable requires a non-zero value (its known cost)")
    args = ap.parse_args()

    plans = {"ethane->chloroethane": af.CHLOROETHANE, "ethane->ethanol": af.ETHANOL,
             "ethane->pentane": af.PENTANE}
    worst = 0.0
    for name, target in plans.items():
        a, b = af.package(af.ETHANE), af.package(target)
        plan = build_topology_plan(a, b, af.core_map(a, b), af.water_environment(), mode="hybrid")
        x = plan.positions_nm
        h = from_plan(plan)
        context = openmm.Context(h.system, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName("Reference"))
        context.setPositions(x)
        state = dict(zip(NAMES, (args.lam,) * 3))
        analytic = sum(h.derivative_components(context, state)["lambda_bonded"].values())
        box = plan.system_a.getDefaultPeriodicBoxVectors()[0][0]._value
        policy = plan.record.get("terms", {}).get("junction_policy") if isinstance(
            plan.record.get("terms"), dict) else None
        policy = policy or args.policy
        split = fx.bonded_derivative_split(plan.system_a, plan.system_b,
                                           set(plan.a_only) | set(plan.b_only), x, box=box,
                                           policy=policy)
        print(f"\n{name}: dU/dlambda_bonded = {analytic:.2f} kJ/mol at lambda = {args.lam}"
              f"   [junction_policy = {policy or 'not stated'}]")
        print(f"    from terms touching a unique atom: {split['unique_touching']:.2f}")
        print(f"    from core parameter changes:       {split['core']:.2f}   <- the only part that "
              "should be here")
        if split["warning"]:
            print(f"    WARNING: {split['warning']}")
        worst = max(worst, abs(split["unique_touching"]))
    print(f"\nworst junction contribution across these plans: {worst:.2f} kJ/mol "
          f"(expected 0.00; warning threshold {fx.BONDED_JUNCTION_WARNING_KJ})")
    return 1 if worst > fx.BONDED_JUNCTION_WARNING_KJ else 0


if __name__ == "__main__":
    raise SystemExit(main())
