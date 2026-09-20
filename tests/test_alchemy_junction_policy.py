"""S4 acceptance rows M3.1-M3.4: `retain-all` against `separable`, one construction difference.

The protocol is M3.0 in `docs/development/0.7.0/handoffs/S4-acceptance-matrix.md`, registered
before this ran. The SAME edge (ethane -> chloroethane), the same legs, schedule, window
placement, lengths, seeds and analysis, built twice: once under S2's new default
`junction_policy = retain-all`, once under `separable`, the construction M2 ran. CPU only.

What is gated is the two ΔΔG agreeing. Per-leg ΔG is expected to DIFFER -- `separable` removes a
dummy's junction terms at its own end, so each leg's endpoint is a different physical state, and
the dummy's internal free energy cancels only between legs.

The campaign lives in $MD_TOOLS_S4_M3_ROOT: `<policy>/<leg>/`, so a window can be run by a
separate process (`python -m ... run_one`) pinned to one core, and the analysis reads what they
left.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")
openmm = pytest.importorskip("openmm")
from openmm import app  # noqa: E402

from tests import alchemy_fixtures as af  # noqa: E402
from md_tools.alchemy import estimators as est  # noqa: E402
from md_tools.alchemy.campaign import (analyze_leg, combine_repeats, matched_leg_report,  # noqa: E402
                                       prepare_leg, read_leg, relative_hydration_from_legs,
                                       run_leg)
from md_tools.alchemy.paths import linear_path  # noqa: E402
from md_tools.alchemy.samples import KJ_PER_KCAL  # noqa: E402
from md_tools.alchemy.windows import WindowSettings  # noqa: E402

pytestmark = [pytest.mark.slow]          # CPU: no card, so not a gpu lane

POLICIES = ("retain-all", "separable")
NAMES = ["lambda_bonded", "lambda_electrostatics", "lambda_sterics"]
BASE = [k / 15 for k in range(16)]
#: M2's placement exactly, so the POLICY is the only difference.
S_VALUES = {"solvent_v2": BASE, "vacuum": sorted(BASE + [1 / 60, 1 / 30]),
            "vacuum_ba": sorted(BASE + [1 - 1 / 60, 1 - 1 / 30])}
REPEATS = ("r1", "r2", "r3")
SEEDS = {"r1": 101, "r2": 202, "r3": 303}
ENDPOINTS = {"solvent_v2": ("ethane", "chloroethane"), "vacuum": ("ethane", "chloroethane"),
             "vacuum_ba": ("chloroethane", "ethane")}
ENVIRONMENT = {"solvent_v2": "solvent", "vacuum": "vacuum", "vacuum_ba": "vacuum"}


def root() -> Path:
    value = os.environ.get("MD_TOOLS_S4_M3_ROOT")
    if not value:
        raise SystemExit("MD_TOOLS_S4_M3_ROOT is not set")
    return Path(value)


def settings(leg: str, repeat: str) -> WindowSettings:
    """M3.0: solvent 200 ps production, vacuum 1 ns; 50 ps equilibration; reports every 1 ps."""
    steps = 100_000 if leg == "solvent_v2" else 500_000
    return WindowSettings(steps=steps, report_interval=500, checkpoint_interval=25_000,
                          equilibration_steps=25_000, timestep_fs=2.0, seed=SEEDS[repeat],
                          minimize_iterations=500)


def plan_for(policy: str, leg: str):
    """The plan and Hamiltonian for one policy and leg. `junction_policy` is S2's."""
    from md_tools.alchemy.hamiltonian import from_plan
    from md_tools.alchemy.topology import Environment, build_topology_plan
    from md_tools.ligands.mapping import LigandSelector

    eth, cle = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    if leg == "solvent_v2":
        a, b, env = eth, cle, af.water_environment_v2()
    elif leg == "vacuum":
        v = af.FIXTURE_ROOT.parent / "vacuum-v1"
        a, b = eth, cle
        env = Environment.from_files(v / "built.xml", v / "built.pdb",
                                     LigandSelector(resname="ETA"), record=v / "built.log")
    else:
        a, b, env = cle, eth, af.vacuum_environment(cle, constraints=app.HBonds)
    plan = build_topology_plan(a, b, af.core_map(a, b), env, mode="hybrid",
                               junction_policy=policy)
    recorded = plan.record.get("junction_policy") or \
        plan.record.get("bonded", {}).get("junction_policy")
    if recorded != policy:
        raise SystemExit(f"the plan records junction_policy {recorded!r}, not {policy!r}: the "
                         f"comparison would not be of the two policies")
    return plan, from_plan(plan)


def prepare(policy: str) -> None:
    for leg in S_VALUES:
        d = root() / policy / leg
        plan, h = plan_for(policy, leg)
        if d.exists():
            assert read_leg(d)[0]["plan_sha256"] == plan.sha256, f"{d} holds another plan"
            continue
        a, b = ENDPOINTS[leg]
        prepare_leg(d, plan=plan, hamiltonian=h,
                    path=linear_path(NAMES, endpoint_a=a, endpoint_b=b,
                                     description="Amber18 one-step diagonal"),
                    s_values=S_VALUES[leg], temperature_k=300.0,
                    pressure_bar=1.01325 if leg == "solvent_v2" else None,
                    environment=ENVIRONMENT[leg], endpoint_a=a, endpoint_b=b,
                    scheme=f"amber18-hybrid/{policy}")


def run_one(policy: str, leg: str, repeat: str, window: str) -> None:
    """One window, for a single pinned process."""
    _, h = plan_for(policy, leg)
    run_leg(root() / policy / leg, hamiltonian=h, settings=settings(leg, repeat), repeat=repeat,
            windows=[window], cpu=True)


def _legs(policy: str):
    out = {}
    for leg in S_VALUES:
        out[leg] = {}
        for r in REPEATS:
            try:
                out[leg][r] = analyze_leg(root() / policy / leg, repeat=r)
            except Exception as exc:
                out[leg][r] = ("UNUSABLE", {"reason": f"{type(exc).__name__}: {exc}"})
    return out


def _combined(per, leg):
    usable = [v[0] for v in per[leg].values() if not isinstance(v[0], str)]
    return combine_repeats(usable) if usable else None


@pytest.mark.parametrize("policy", POLICIES)
def test_m3_prepare(policy):
    prepare(policy)
    report = matched_leg_report(root() / policy / "vacuum", root() / policy / "solvent_v2")
    print(f"\n{policy}: matched_legs {report['ligand_hamiltonian_sha256'][:16]}")


def test_m3_analysis():
    out = {"rows": {}, "per_leg": {}}
    per = {p: _legs(p) for p in POLICIES}
    # M3.2 overlap, M3.4 the bonded integrand at the end points, per policy and leg
    m32, m34 = [], []
    for p in POLICIES:
        for leg in S_VALUES:
            for r, v in per[p][leg].items():
                if isinstance(v[0], str):
                    m32.append({"policy": p, "leg": leg, "repeat": r, "unusable": v[1]["reason"]})
                    continue
                an = v[1]
                d = an["estimates"]["MBAR"]["diagnostics"]
                n = min(an["samples_per_state"].values()) / max(d["statistical_inefficiency"]
                                                                .values())
                m32.append({"policy": p, "leg": leg, "repeat": r,
                            "min_overlap": d["min_neighbour_overlap"],
                            "min_decorrelated_samples": n,
                            "clears_floor": d["min_neighbour_overlap"] >= 0.03})
                w = an["estimates"]["TI"]["diagnostics"]["windows"]
                first, last = sorted(w)[0], sorted(w)[-1]
                m34.append({"policy": p, "leg": leg, "repeat": r,
                            "dU_dlambda_bonded_at_s0":
                                w[first]["mean_dU_dlambda_kJ_mol"]["lambda_bonded"],
                            "dU_dlambda_bonded_at_s1":
                                w[last]["mean_dU_dlambda_kJ_mol"]["lambda_bonded"]})
    out["rows"]["M3.2"], out["rows"]["M3.4"] = m32, m34
    # M3.3 the vacuum closure, per policy
    m33 = {}
    for p in POLICIES:
        ab, ba = _combined(per[p], "vacuum"), _combined(per[p], "vacuum_ba")
        if ab is None or ba is None:
            m33[p] = {"verdict": "INCONCLUSIVE", "reason": "a vacuum leg has no usable repeat"}
            continue
        closure = (ab.delta_g_kj_mol + ba.delta_g_kj_mol) / KJ_PER_KCAL
        m33[p] = {"A_to_B_kcal": ab.delta_g_kj_mol / KJ_PER_KCAL,
                  "B_to_A_kcal": ba.delta_g_kj_mol / KJ_PER_KCAL,
                  "sigma_used": [ab.source.get("sigma_used"), ba.source.get("sigma_used")],
                  **est.agreement_gate(closure, ab.sigma_kj_mol / KJ_PER_KCAL, 0.0,
                                       reference_sigma_kcal=ba.sigma_kj_mol / KJ_PER_KCAL)}
    out["rows"]["M3.3"] = m33
    # M3.1 the two ddG against each other -- the gated row
    cycles = {}
    for p in POLICIES:
        usable = sorted({r for r in REPEATS if not isinstance(per[p]["vacuum"][r][0], str)}
                        & {r for r in REPEATS if not isinstance(per[p]["solvent_v2"][r][0], str)})
        cycles[p] = relative_hydration_from_legs(root() / p / "vacuum",
                                                 root() / p / "solvent_v2", repeats=usable)
        cycles[p].pop("analyses", None)
        out["per_leg"][p] = {leg: (None if _combined(per[p], leg) is None
                                   else _combined(per[p], leg).to_record()) for leg in S_VALUES}
    a, b = (cycles[p]["delta_g_kcal_mol"] for p in POLICIES)
    sa, sb = (cycles[p]["sigma_kcal_mol"] for p in POLICIES)
    out["rows"]["M3.1"] = {"retain_all_kcal": a, "separable_kcal": b,
                           "sigma_kcal": [sa, sb],
                           **est.agreement_gate(a, sa, b, reference_sigma_kcal=sb)}
    (root() / "m3_result.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("\nM3.1 ddG:", json.dumps(out["rows"]["M3.1"], default=str))
    print("M3.3 closure:", json.dumps(m33, default=str))
    for row in m32:
        print("M3.2 %-11s %-11s %s %s" % (row["policy"], row["leg"], row["repeat"],
                                          {k: v for k, v in row.items()
                                           if k not in ("policy", "leg", "repeat")}))
    for row in m34:
        print("M3.4 %-11s %-11s %s bonded dU/dl: s0 %.1f s1 %.1f" %
              (row["policy"], row["leg"], row["repeat"],
               row["dU_dlambda_bonded_at_s0"], row["dU_dlambda_bonded_at_s1"]))
    assert out["rows"]["M3.1"]["verdict"] == "PASS", out["rows"]["M3.1"]


if __name__ == "__main__":       # one window, one pinned process
    import sys
    run_one(*sys.argv[1:5])
