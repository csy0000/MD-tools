"""S4 acceptance rows M2.1-M2.5: relative hydration ethane -> chloroethane, the production lane.

The protocol is M2.0 in `docs/development/0.7.0/handoffs/S4-acceptance-matrix.md`, written before
any sampling, with the one pilot change M2.0p allows (vacuum windows at s = 1/60 and 1/30):

    one-step diagonal path, 2 fs, LangevinMiddle 1/ps, 300 K; water NPT 1.01325 bar, vacuum NVT;
    each window minimised at its own state, 20 ps equilibration discarded, 1 ns production,
    1 ps reports; 3 independent repeats of every leg.

Legs: `solvent_v2` (ethane-tip3p v2, 2.7 nm, recorded build; M2.0f replaced v1), `vacuum` (vacuum-v1, recorded build), and
`vacuum_ba` (chloroethane -> ethane in vacuum, an independently built plan, for the M2.4 closure;
its extra windows mirror the A->B ones, s = 1 - 1/60 and 1 - 1/30, because the stiff end is the
one where chlorine is the dummy, which is s = 1 in that orientation).

The campaign lives in $MD_TOOLS_S4_M2_ROOT, a directory that persists across invocations, so the
run tests can be launched as separate processes (one per leg and repeat) and the analysis test
reads what they left. gpu + slow: a CUDA lane, run with --error-on-skip on an allocated card.
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
from md_tools.alchemy.campaign import (analyze_leg, matched_leg_report, prepare_leg,  # noqa: E402
                                       read_leg, relative_hydration_from_legs, run_leg)
from md_tools.alchemy.paths import linear_path  # noqa: E402
from md_tools.alchemy.samples import KJ_PER_KCAL  # noqa: E402
from md_tools.alchemy.windows import WindowSettings  # noqa: E402


#: The campaign root is configuration this lane cannot invent: without it there is no campaign to
#: run or analyse, which is NOT the same as a failing campaign. An unset variable therefore
#: DESELECTS these tests rather than reddening a suite with "not configured" (S3, 2026-09-21).
#:
#: A skipif MARKER, not `pytest.skip(allow_module_level=True)`: a module-level skip happens during
#: COLLECTION and produces no test report, so `--error-on-skip` cannot see it -- checked, it
#: reported "1 skipped". A marker skips at setup, which does produce a report, and the conftest
#: hook then turns it into a failure. That matters because a skip is also how a campaign quietly
#: stops being evidence: skip by default, FAIL where the campaign is supposed to have run.
_NO_ROOT = pytest.mark.skipif(
    not os.environ.get("MD_TOOLS_S4_M2_ROOT"),
    reason="MD_TOOLS_S4_M2_ROOT is not set: no campaign root to run or analyse. Set it to the campaign "
           "directory, or run an evidence lane with --error-on-skip.")

pytestmark = [pytest.mark.gpu, pytest.mark.slow, _NO_ROOT]

NAMES = ["lambda_bonded", "lambda_electrostatics", "lambda_sterics"]
BASE = [k / 15 for k in range(16)]
#: the water leg is `solvent_v2` since M2.0f: ethane-tip3p v2 (2.7 nm). The v1 leg directory,
#: `solvent`, holds the void windows of the stopped run and is never read.
S_VALUES = {"solvent_v2": BASE, "vacuum": sorted(BASE + [1 / 60, 1 / 30]),
            "vacuum_ba": sorted(BASE + [1 - 1 / 60, 1 - 1 / 30])}
REPEATS = ("r1", "r2", "r3")
SEEDS = {"r1": 101, "r2": 202, "r3": 303}
EXPERIMENT_KCAL = {"ethane": 1.83, "chloroethane": -0.63}   # FreeSolv; reported, never gated


def _root() -> Path:
    return Path(os.environ["MD_TOOLS_S4_M2_ROOT"])


def _machine(root: Path) -> Path:
    p = root / "machine.config"
    if not p.exists():
        p.write_text("schema_version: '1.0'\nmachine:\n  openmm:\n    platform: CUDA\n"
                     "    precision: mixed\n    device_policy: local_rank\n")
    return p


def _plan(leg: str):
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
    plan = build_topology_plan(a, b, af.core_map(a, b), env, mode="hybrid")
    return plan, from_plan(plan)


ENDPOINTS = {"solvent_v2": ("ethane", "chloroethane"), "vacuum": ("ethane", "chloroethane"),
             "vacuum_ba": ("chloroethane", "ethane")}
ENVIRONMENT = {"solvent_v2": "solvent", "vacuum": "vacuum", "vacuum_ba": "vacuum"}


def _settings(repeat: str) -> WindowSettings:
    return WindowSettings(steps=500_000, report_interval=500, checkpoint_interval=50_000,
                          equilibration_steps=10_000, timestep_fs=2.0, seed=SEEDS[repeat],
                          minimize_iterations=500)


def test_m2_prepare():
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    for leg in S_VALUES:
        d = root / leg
        plan, h = _plan(leg)
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
                    scheme="amber18-hybrid")
    report = matched_leg_report(root / "vacuum", root / "solvent_v2")
    print(f"\nM2 matched_legs: {report['ligand_hamiltonian_sha256']}")


@pytest.mark.parametrize("leg,repeat", list(itertools.product(S_VALUES, REPEATS)))
def test_m2_run(leg, repeat):
    root = _root()
    _, h = _plan(leg)
    results = run_leg(root / leg, hamiltonian=h, settings=_settings(repeat), repeat=repeat,
                      cpu=False, machine_config=_machine(root))
    assert all(r["disposition"] in ("fresh", "resumed", "verified-complete") for r in results)


def _gate(value, sigma, ref, ref_sigma=0.0, sigma_int=0.0):
    return est.agreement_gate(value, sigma, ref, reference_sigma_kcal=ref_sigma,
                              integration_sigma_kcal=sigma_int)


def test_m2_analysis():
    root = _root()
    out: dict = {"rows": {}}
    per: dict = {}
    unusable: list = []
    for leg in S_VALUES:
        per[leg] = {}
        for r in REPEATS:
            try:
                per[leg][r] = analyze_leg(root / leg, repeat=r)
            except Exception as exc:                     # a leg that cannot even be formed
                unusable.append({"leg": leg, "repeat": r, "verdict": "INCONCLUSIVE",
                                 "reason": f"{type(exc).__name__}: {exc}"})
    out["unusable_legs"] = unusable
    # M2.1: overlap and decorrelated samples, every leg and repeat
    m21 = []
    for leg, reps in per.items():
        for r, (_, an) in reps.items():
            d = an["estimates"]["MBAR"]["diagnostics"]
            n = min(an["samples_per_state"].values()) / max(d["statistical_inefficiency"].values())
            m21.append({"leg": leg, "repeat": r, "min_overlap": d["min_neighbour_overlap"],
                        "min_decorrelated_samples": n,
                        "ok": d["min_neighbour_overlap"] >= 0.03 and n >= 50})
    out["rows"]["M2.1"] = m21
    # M2.2: TI vs MBAR, every leg and repeat
    m22 = []
    for leg, reps in per.items():
        for r, (_, an) in reps.items():
            m, ti = an["estimates"]["MBAR"], an["estimates"]["TI"]
            g = _gate(ti["delta_g_kcal_mol"], ti["sigma_kcal_mol"], m["delta_g_kcal_mol"],
                      ref_sigma=m["sigma_kcal_mol"],
                      sigma_int=ti["diagnostics"]["quadrature_discrepancy_kJ_mol"] / KJ_PER_KCAL)
            m22.append({"leg": leg, "repeat": r, "TI": ti["delta_g_kcal_mol"],
                        "MBAR": m["delta_g_kcal_mol"], **g})
    out["rows"]["M2.2"] = m22
    # M2.3: repeats agree pairwise (only repeats that produced a usable leg)
    m23 = []
    for leg, reps in per.items():
        for x, y in itertools.combinations(sorted(reps), 2):
            lx, ly = reps[x][0], reps[y][0]
            g = _gate(lx.delta_g_kj_mol / KJ_PER_KCAL, lx.sigma_kj_mol / KJ_PER_KCAL,
                      ly.delta_g_kj_mol / KJ_PER_KCAL, ref_sigma=ly.sigma_kj_mol / KJ_PER_KCAL)
            m23.append({"leg": leg, "pair": [x, y], **g})
    out["rows"]["M2.3"] = m23
    # M2.4: vacuum closure A->B + B->A = 0
    from md_tools.alchemy.campaign import combine_repeats
    if not per["vacuum"] or not per["vacuum_ba"]:
        out["rows"]["M2.4"] = {"verdict": "INCONCLUSIVE",
                               "reason": "a vacuum leg produced no usable repeat"}
        out["rows"]["M2.5"] = {"verdict": "INCONCLUSIVE", "reason": "M2.4 has no legs"}
        (root / "m2_result.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                        default=str))
        assert False, out["unusable_legs"] or "no usable vacuum repeats"
    ab = combine_repeats([per["vacuum"][r][0] for r in sorted(per["vacuum"])])
    ba = combine_repeats([per["vacuum_ba"][r][0] for r in sorted(per["vacuum_ba"])])
    closure = (ab.delta_g_kj_mol + ba.delta_g_kj_mol) / KJ_PER_KCAL
    out["rows"]["M2.4"] = {"A_to_B_kcal": ab.delta_g_kj_mol / KJ_PER_KCAL,
                           "B_to_A_kcal": ba.delta_g_kj_mol / KJ_PER_KCAL,
                           **_gate(closure, ab.sigma_kj_mol / KJ_PER_KCAL, 0.0,
                                   ref_sigma=ba.sigma_kj_mol / KJ_PER_KCAL)}
    # M2.5: the cycle, experiment reported
    usable = sorted(set(per["vacuum"]) & set(per["solvent_v2"]))
    if not usable:
        out["rows"]["M2.5"] = {"verdict": "INCONCLUSIVE",
                               "reason": "no repeat produced a usable leg in BOTH environments"}
        (root / "m2_result.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                        default=str))
        assert False, out["unusable_legs"] or "no usable pair of legs"
    cycle = relative_hydration_from_legs(root / "vacuum", root / "solvent_v2", repeats=usable)
    cycle.pop("analyses")
    out["rows"]["M2.5"] = {"repeats_used": usable, "ddG_hyd_kcal": cycle["delta_g_kcal_mol"],
                           "sigma_kcal": cycle["sigma_kcal_mol"],
                           "experiment_kcal": EXPERIMENT_KCAL["chloroethane"]
                           - EXPERIMENT_KCAL["ethane"],
                           "experiment_is_gated": False, "cycle": cycle}
    (root / "m2_result.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("\nM2 result:", json.dumps({k: v for k, v in out["rows"].items() if k != "M2.5"},
                                     default=str)[:3000])
    print("M2.5:", {k: v for k, v in out["rows"]["M2.5"].items() if k != "cycle"})
    failures = unusable + [r for r in m21 if not r["ok"]] + \
               [r for r in m22 + m23 if r["verdict"] != "PASS"] + \
               ([out["rows"]["M2.4"]] if out["rows"]["M2.4"]["verdict"] != "PASS" else [])
    assert not failures, failures
