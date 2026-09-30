#!/usr/bin/env python
"""Turn a finished alchemical ladder into a hydration free energy.

    python ethanol_analysis.py md_script

Analysis is deliberately not a `build-md` setting: the engine produces samples, and which
estimator you trust over them is your decision, not the run's. So this is a script rather than a
configuration, exactly as WHAM and MBAR are not configured under `umbrella`.

WHAT IT PRINTS, and why each line is there:

* **dG_hyd**, which is MINUS the decoupling free energy. Decoupling removes the molecule from
  water; hydration puts it in. Getting that sign wrong is the easiest mistake in the whole
  calculation and it produces a plausible-looking number.
* **four estimators over the same samples** -- MBAR, BAR, TI and both EXP directions. They are not
  four measurements. They are four ways of reading one, and their SPREAD is a diagnostic: if EXP
  forward and reverse disagree while MBAR and BAR agree, the ladder is fine and the exponential
  average is being dominated by its tail, which is what it always does. If MBAR and TI disagree,
  the windows are too far apart for one of them.
* **the smallest nearest-neighbour overlap**, which decides whether any of the above means
  anything. Below about 0.03 the neighbouring windows barely share configurations and MBAR is
  extrapolating between them.

It prints kcal/mol because every experimental reference for hydration is in kcal/mol, and converts
once, here. `Leg.delta_g_kj_mol` is kJ/mol; mixing the two silently is the other easy mistake.
"""
from __future__ import annotations

import sys
from pathlib import Path

KCAL_PER_KJ = 1.0 / 4.184


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    run = Path(argv[1])
    leg = run / "leg"
    if not (leg / "leg.json").is_file():
        print(f"{leg} is not a prepared leg; give the directory `build-md` generated",
              file=sys.stderr)
        return 2

    from md_tools.alchemy.campaign import analyze_leg

    record, analysis = analyze_leg(leg, repeat="r1", estimator="MBAR")
    diagnostics = analysis["estimates"]["MBAR"]["diagnostics"]

    # -dG(decouple). The sign is the point; see the module docstring.
    print(f"dG_hyd(MBAR)          {-record.delta_g_kj_mol * KCAL_PER_KJ:+8.3f} "
          f"+/- {record.sigma_kj_mol * KCAL_PER_KJ:.3f} kcal/mol")
    print()
    print("the same samples, four ways:")
    for name in ("MBAR", "BAR", "TI", "EXP_forward", "EXP_reverse"):
        estimate = analysis["estimates"].get(name)
        if estimate is None:
            continue
        print(f"  {name:12s} dG_hyd {-estimate['delta_g_kJ_mol'] * KCAL_PER_KJ:+8.3f} kcal/mol")
    print()
    # The gate, read from the key `analyze_leg` actually writes. An earlier version of this script
    # guessed at three other spellings, got None from all of them, and reported `null` for every
    # leg -- a check that cannot fail, which closes the question instead of asking it.
    print(f"smallest neighbour overlap  {diagnostics['min_neighbour_overlap']:.3f}   "
          f"(a ladder is in trouble below ~0.03)")
    print(f"poor overlap anywhere       {diagnostics['poor_overlap']}")
    print(f"samples per window          {min(diagnostics['samples_per_state'])}"
          f"-{max(diagnostics['samples_per_state'])} after decorrelation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
