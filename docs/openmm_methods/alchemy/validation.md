# What the alchemical runtime has been shown to compute

The topology builder is checked by its own refusals; the **runtime** — softcore mixing, the λ
ladder, the window sampler and the estimators — can only be checked against a free energy somebody
else already knows. This page records that check and nothing else. Every number here came from a
run on this machine, and the command that produced it is named.

## The check, and why it is a real one

Three small molecules are **decoupled from TIP3P water**: every interaction between the ligand and
its environment is switched off across twelve λ windows, and MBAR turns the samples into a free
energy. ΔG_hyd = −ΔG(decouple).

Then the same quantity is computed a **second, unrelated way**. An earlier campaign
([0.7.0 M2](../../development/0.7.0/handoffs/S4-acceptance-matrix.md)) obtained the *relative*
hydration free energy of ethane → chloroethane by **mutating one molecule into the other** — a
different plan, a different Hamiltonian, different particles appearing and disappearing. Two routes
through the same thermodynamic cycle must land on the same difference:

> ΔG_hyd(chloroethane) − ΔG_hyd(ethane), by decoupling each, must reproduce ΔΔG_hyd by mutation.

That is the only check available that does not compare the code to itself. It is worth more than
agreement with experiment, which a force field can miss for reasons that have nothing to do with
whether the estimator is right.

## One leg per ligand, not two

The vacuum leg of this construction is **definitionally zero**. `build_decoupling_plan` keeps the
ligand's own bonded terms, internal exceptions and internal pairs physical at *both* ends, so in
vacuum U(0) ≡ U(1): the Hamiltonian does not depend on λ, ΔG = 0, and there is no variance to
sample. Running it would produce a leg that carries no information, and reporting that leg as
evidence would be worse than skipping it. So ΔG_hyd = −ΔG(decouple in solvent), one leg.

## The ligands

All three are in FreeSolv v0.52, parameterised here with **openff-2.2.1 (Sage)/AM1-BCC** — *not*
the GAFF/AM1-BCC that FreeSolv's own calculated column used. That column is therefore **not our
reference**: a match to it would be reassuring, not validating, and a mismatch would not be a
defect. Experiment is quoted because it is the physical answer; the force field was never fitted to
it.

| ligand | FreeSolv id | experiment, kcal/mol |
|---|---|---|
| ethane | `mobley_2008055` | +1.83 |
| chloroethane | `mobley_2198613` | −0.63 |
| ethanol | `mobley_2310185` | −5.00 |

FreeSolv's experimental uncertainties for all three are the placeholder 0.60 — its own note is
"Experimental uncertainty not presently available" — so they are not error bars and are not used
as one.

Ethanol is in the set for a specific reason. Ethane and chloroethane are weakly polar with small
hydration free energies, so their agreement could be insensitive to how well **electrostatic**
decoupling works. Ethanol hydrates through hydrogen bonding: its ΔG is large and dominated by the
charging leg, which the other two barely exercise.

## Results

Twelve evenly spaced windows, 1 ns of production sampling each after 10 ps of equilibration at the
window's own Hamiltonian, 2 fs, 300 K, 1 atm, CUDA/mixed. MBAR, with BAR, TI and both EXP
directions computed alongside from the same samples.

<!-- RESULTS TABLE: regenerate with scratchpad/abs-hydration/summarise.py -->

| ligand | repeats | ΔG_hyd, kcal/mol | ± (repeat spread) | per-run σ (MBAR) | min. overlap |
|---|---|---|---|---|---|
| ethane | r1, r2, r3 | **+2.478** | 0.033 | 0.078 | 0.141 |
| chloroethane | r1, r2, r3 | **+0.839** | 0.067 | 0.093 | 0.131 |
| ethanol | r1, r2, r3 | **−3.523** | 0.025 | 0.102 | 0.139 |

Per repeat: ethane +2.507 / +2.515 / +2.413, chloroethane +0.747 / +0.801 / +0.969, ethanol
−3.515 / −3.570 / −3.483. Nine legs, 108 windows, 108 ns of sampling.

Against experiment: ethane +2.478 (expt +1.83), chloroethane +0.839 (−0.63), ethanol −3.523
(−5.00). All three are too POSITIVE by 0.65 to 1.48 kcal/mol — the same direction for a
hydrocarbon, an alkyl halide and an alcohol, which is what a systematic force-field and
water-model offset looks like rather than a defect in any one ligand's parameters. It is reported
and never gated on.

**The error bar is the repeat spread, not MBAR's σ**, and the two are different quantities. MBAR's
σ is the within-run statistical error at fixed sampling; it cannot see anything that differs
between independent realisations — the configuration each window inherits, where the barostat
takes the box, which basins a nanosecond visits. Those appear only as scatter *between* repeats.
Here the two are comparable (spread/σ of 0.4 to 1.2), so neither dominates, and quoting either
alone would understate the uncertainty. M2's reference number is error-barred the same way, which
is what makes the comparison below like for like.

**The placement gate passed on every leg.** The smallest nearest-neighbour overlap anywhere is
0.131 (chloroethane r2) against the registered floor of 0.03, and `poor_overlap` is false throughout — twelve even
windows are enough for these three ligands, and no re-placement was needed. The gate is read from
`estimates.MBAR.diagnostics.min_neighbour_overlap`; the first version of the analysis script
guessed three other key spellings and fell through to `null` on all of them, so it reported nothing
and passed every time. A check that cannot fail closes the question instead of asking it.

## The cross-check

| route | ΔΔG_hyd(ethane → chloroethane), kcal/mol |
|---|---|
| decoupling each ligand separately (three repeats each) | **−1.639 ± 0.074** |
| mutating one into the other (0.7.0 M2, three repeats) | **−1.743 ± 0.026** |

They differ by **0.104 ± 0.079**, which is **1.3 σ** — consistent. Two alchemical routes with
different plans, different end states and different particles appearing arrive at the same number
within the scatter of both.

A CAUTION THAT IS PART OF THE RESULT. The first repeat alone gave −1.759 against M2's −1.743, an
apparent agreement to 0.016 — far better than either method's own reproducibility, and it was
written down here as the headline before the other repeats finished. It was a coincidence of one
realisation. Chloroethane's three repeats span 0.747 to 0.969, a spread of 0.22 kcal/mol, so any
single repeat can land 0.1 from the mean by luck in either direction. An agreement much tighter
than the reproducibility of the things being compared is not a stronger result; it is a sign that
not enough of them have been run.

Against experiment (−2.46) both routes are off by about 0.7 kcal/mol in the same direction. That is
force-field and water-model error — openff-2.2.1/AM1-BCC in TIP3P was not fitted to this — and it
is reported, never gated on.

## What this does not show

* **These nine legs were driven from Python** (`md_tools.alchemy.campaign`), because they predate
  `build-md` generating an alchemical directory. The generated route has since been run on the same
  calculation — three repeats, **−3.656 ± 0.075** against the −3.523 ± 0.025 below, agreeing to
  1.7 σ (see the [ethanol hydration tutorial](../../tutorial/ethanol/hydration.md)). The plan the
  CLI builds is bit-identical to the plan these legs used, asserted by
  `tests/test_combine_decoupling.py` rather than assumed, and the `WindowSettings` match field for
  field.
* **The two routes differ only in the seed, and that is not enough to make them reproduce.** A
  fourth ladder run with the campaign's own seed gave **−3.6829** against its **−3.5150**: same
  plan digest, same settings, same seed, **0.168 kcal/mol apart.** `LocalEnergyMinimizer` takes no
  seed and does not reproduce on CUDA, so every fresh start is a new realisation. The practical
  consequence for reading this page: MBAR's per-ladder σ is ~0.10 and the run-to-run difference is
  larger, so **every number below is error-barred by its repeats and none by its estimator.**
* **Nothing about protein–ligand binding.** A decoupling in water is not a decoupling in a binding
  site, where the sampling problem is the whole difficulty.
* **Nothing about a mutation with a charge change**, an atom mapping across a ring, or softcore at
  a junction the map had to cut. Those have their own acceptance rows.

## Related

* [Alchemical topology](README.md) — the builder, its configuration and its refusals
* [0.7.0 acceptance matrix](../../development/0.7.0/handoffs/S4-acceptance-matrix.md) — where the
  mutation route's number comes from, row M2.5
