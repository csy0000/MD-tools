#!/usr/bin/env python
"""Pair-resolved REST2 exchange diagnostics for an acceptance pilot.

A REST2 ladder is a CHAIN: a walker can only reach the hot end by passing every neighbouring pair
in turn, so the ladder is only as good as its weakest link. An overall acceptance fraction averages
over that structure and can look healthy while one pair is closed. This reports every pair
separately, with a Wilson interval so a small attempt count cannot masquerade as precision.

    python scripts/rest2_pilot_acceptance.py --exchange-log pilot_exchange_attempts.csv \\
        --out-prefix reports/.../alanine_pilot

Writes ``<prefix>_pairs.csv`` and ``<prefix>_summary.json``.

The classification thresholds are PREDECLARED for this pilot (see
``claude-instruction/20260814_explicit_solvent_fix_2.md``) and are pilot decisions, not universal
REST2 criteria:

    provisionally acceptable   every neighbouring-pair point estimate >= 0.20
    borderline                 worst pair >= 0.10 but < 0.20
    failed                     any neighbouring pair < 0.10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

Z = 1.959963984540054  # 95 %


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Not the normal approximation: at the acceptance levels and attempt counts of a short pilot the
    normal interval can run below 0 or above 1, and is badly calibrated exactly where the answer
    matters (a pair with few successes).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(worst: float) -> str:
    if worst >= 0.20:
        return "provisionally acceptable"
    if worst >= 0.10:
        return "borderline"
    return "FAILED ladder"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange-log", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--expected-rounds", type=int, default=None,
                    help="production_ps / exchange_interval_ps; validates per-pair attempt counts")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.exchange_log)
    n_rep = sum(1 for c in df.columns if c.startswith("walker_at_replica_"))
    df["pair"] = list(zip(df.i.astype(int), df.j.astype(int)))

    rows, worst = [], 1.0
    for (i, j), g in df.groupby("pair"):
        k, n = int(g.accepted.sum()), int(len(g))
        lo, hi = wilson(k, n)
        p = k / n if n else float("nan")
        worst = min(worst, p)
        rows.append({
            "pair": f"{i}-{j}", "i": i, "j": j,
            "s_i": float(g.scale_factor_i.iloc[0]), "s_j": float(g.scale_factor_j.iloc[0]),
            "T_eff_i_K": float(g.effective_temperature_i_k.iloc[0]),
            "T_eff_j_K": float(g.effective_temperature_j_k.iloc[0]),
            "attempts": n, "accepted": k, "acceptance": round(p, 5),
            "wilson_lo": round(lo, 5), "wilson_hi": round(hi, 5),
            "mean_log_acceptance": round(float(g.log_acceptance.mean()), 4),
            "sd_log_acceptance": round(float(g.log_acceptance.std(ddof=1)), 4)
            if n > 1 else float("nan"),
        })
    pairs = pd.DataFrame(rows).sort_values(["i", "j"])

    # every ADJACENT pair must exist; a missing one is a broken chain, not a small number
    expected_pairs = {(r, r + 1) for r in range(n_rep - 1)}
    seen_pairs = {(int(r.i), int(r.j)) for r in pairs.itertuples()}
    missing = sorted(expected_pairs - seen_pairs)

    # the even/odd schedule gives each pair roughly half the rounds
    attempt_note = None
    if a.expected_rounds is not None:
        got = int(pairs.attempts.sum())
        want = sum(len(range(ph % 2, n_rep - 1, 2)) for ph in range(a.expected_rounds))
        attempt_note = {"expected_total_attempts": want, "observed_total_attempts": got,
                        "matches": want == got}

    # Rung occupancy and round-trip counting came from the originating research project's analysis
    # package, which is not part of this template. The call sat inside a bare `except` that turned
    # any failure into a diagnostic string, so once that package went away it would have reported
    # an ImportError as a "result" indefinitely. Named as unavailable instead of faked.
    occ = None
    round_trips = [{
        "unavailable": "rung occupancy and round-trip counting are not implemented in this "
                       "template; the acceptance statistics below are computed here and are "
                       "unaffected",
    }]

    overall = float(df.accepted.mean())
    summary = {
        "system": a.system,
        "exchange_log": str(a.exchange_log),
        "n_replicas": n_rep,
        "n_rounds_logged": int(df.step.nunique()),
        "overall_acceptance": round(overall, 5),
        "worst_pair_acceptance": round(float(worst), 5),
        "worst_pair": pairs.loc[pairs.acceptance.idxmin(), "pair"],
        "classification": classify(worst),
        "classification_thresholds": {"provisionally_acceptable": ">= 0.20",
                                      "borderline": "0.10 - 0.20", "failed": "< 0.10"},
        "missing_adjacent_pairs": [f"{i}-{j}" for i, j in missing],
        "attempt_count_check": attempt_note,
        "round_trips": round_trips,
        "walker_occupancy_final": occ[-1] if occ else None,
        "note": "A pilot diagnoses the ladder; it does not converge anything.  Absence of a round "
                "trip in 2 ns/replica means 'not observed within the pilot', not 'cannot occur'.",
    }

    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(f"{a.out_prefix}_pairs.csv", index=False)
    Path(f"{a.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"\n{a.system}  {n_rep} replicas  overall {overall:.3f}  "
          f"worst pair {worst:.3f}  -> {summary['classification']}")
    print(pairs[["pair", "s_i", "s_j", "attempts", "accepted", "acceptance",
                 "wilson_lo", "wilson_hi"]].to_string(index=False))
    if missing:
        print(f"  MISSING adjacent pairs: {missing}")
    print(f"  wrote {a.out_prefix}_pairs.csv and _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
