#!/usr/bin/env python
"""Close the thermodynamic cycle: ΔΔG_hyd from a vacuum leg and a solvent leg.

    python relative_analysis.py md_vacuum md_solvent [repeat ...]

A relative hydration free energy is a DIFFERENCE OF TWO DIFFERENCES, and the cycle is what makes
that legitimate:

    A(vac) --[ΔG_vac]--> B(vac)
      |                    |
    ΔG_hyd(A)            ΔG_hyd(B)
      v                    v
    A(sol) --[ΔG_sol]--> B(sol)

    ΔΔG_hyd = ΔG_hyd(B) - ΔG_hyd(A) = ΔG_sol - ΔG_vac

Neither leg means anything on its own — an alchemical mutation in vacuum is not a physical process
and its ΔG is not a measurable quantity. Only the difference is, and only because the cycle closes.

WHY THE VACUUM LEG IS NOT ZERO HERE, unlike in the decoupling tutorial beside this one. There, the
ligand's own Hamiltonian is retained unchanged at both ends, so the vacuum leg is λ-independent by
construction. Here the molecule itself changes — ethane becomes chloroethane — so its internal
energy changes with λ and that leg has to be sampled.

`relative_hydration_from_legs` runs `matched_legs` FIRST: two legs that do not carry one ligand
Hamiltonian are refused before a single sample is read. That check is the reason this is a cycle
and not two unrelated numbers — without it, a vacuum leg built from one parameterisation and a
solvent leg from another would subtract cleanly and mean nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

KCAL_PER_KJ = 1.0 / 4.184


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: relative_analysis.py <vacuum_runs> <solvent_runs>\n"
              "  each a COMMA-SEPARATED list of run directories, one per repeat",
              file=sys.stderr)
        return 2
    # A REPEAT IS A RUN DIRECTORY, not a name inside one. `build-md` gives every generated run its
    # own identity and `window_main` always writes `r1` into it, because two repeats sharing one
    # directory would be two experiments over one set of paths. So repeats arrive here as
    # `alchemical-run1,alchemical-run2,...`, each differing from the others in `dynamics.seed`
    # and in nothing else.
    vacuum_dirs = [Path(d) / "leg" for d in argv[1].split(",")]
    solvent_dirs = [Path(d) / "leg" for d in argv[2].split(",")]
    for d in vacuum_dirs + solvent_dirs:
        if not (d / "leg.json").is_file():
            print(f"{d} is not a prepared leg; give the directories `build-md` generated",
                  file=sys.stderr)
            return 2
    repeats = ["r1"]

    from md_tools.alchemy.campaign import (analyze_leg, combine_repeats, matched_leg_report)
    from md_tools.alchemy.cycles import relative_hydration

    # `matched_legs` FIRST, on one pair, before a sample is read. Two legs that do not carry one
    # ligand Hamiltonian are refused here; without that check a vacuum leg from one
    # parameterisation and a solvent leg from another would subtract just as cleanly and mean
    # nothing. (`campaign.relative_hydration_from_legs` does exactly this when every repeat lives
    # in ONE leg directory, which is the shape a Python-driven campaign produces; the CLI produces
    # one directory per repeat, so the same three steps are spelled out here.)
    report = matched_leg_report(vacuum_dirs[0], solvent_dirs[0])

    legs = {}
    for role, dirs in (("vacuum", vacuum_dirs), ("solvent", solvent_dirs)):
        per = [analyze_leg(d, repeat="r1", estimator="MBAR")[0] for d in dirs]
        for leg, d in zip(per, dirs):
            if leg.ligand_hamiltonian_sha256 != report["ligand_hamiltonian_sha256"]:
                print(f"{d} carries a different ligand Hamiltonian from the matched pair",
                      file=sys.stderr)
                return 2
        # Error-barred by the REPEAT SPREAD when that is larger than what the estimator claims.
        # On the M2 campaign the repeats of one leg differed by up to 0.80 kcal/mol while MBAR
        # claimed 0.09: an asymptotic covariance is not a measurement of run-to-run scatter.
        legs[role] = combine_repeats(per)
    cycle = relative_hydration(vacuum=legs["vacuum"], solvent=legs["solvent"])
    cycle["matched_legs"] = report
    repeats = vacuum_dirs

    # The sign convention is the cycle's, not this script's: `relative_hydration` states the
    # quantity it returns, and it is printed rather than re-derived here.
    print(cycle["quantity"])
    print()
    # `terms` carries each leg WITH THE SIGN the cycle gave it, so the arithmetic below is the
    # cycle's own and not a second copy of it. Printing the sign is the point: a reader can check
    # that the solvent leg enters positive and the vacuum leg negative, which is the whole content
    # of "ddG = dG_sol - dG_vac".
    for term in cycle["terms"]:
        print(f"  {term['sign']:+.0f} x {term['environment']:8s} {term['name']:24s} "
              f"dG = {term['delta_g_kJ_mol'] * KCAL_PER_KJ:+8.3f} "
              f"+/- {term['sigma_kJ_mol'] * KCAL_PER_KJ:.3f} kcal/mol")
    print()
    print(f"ddG_hyd = {cycle['delta_g_kcal_mol']:+8.3f} +/- {cycle['sigma_kcal_mol']:.3f} "
          f"kcal/mol   ({len(repeats)} repeat(s), MBAR)")
    print(f"  {cycle['variance_rule']}")
    print()
    # matched_legs is the check that licenses the subtraction; say that it ran and what it found.
    matched = cycle["matched_legs"]
    print(f"matched_legs: one ligand Hamiltonian "
          f"{matched['ligand_hamiltonian_sha256'][:16]}... across both legs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
