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

| ligand | repeat | ΔG_hyd, kcal/mol | σ (MBAR) | min. neighbour overlap |
|---|---|---|---|---|
| ethane | r1 | **+2.507** | 0.078 | 0.141 |
| ethane | r3 | **+2.413** | 0.080 | 0.151 |
| chloroethane | r1 | **+0.747** | 0.094 | 0.152 |
| ethanol | r1 | **−3.515** | 0.105 | 0.179 |

The remaining repeats are running; this page is updated from `summarise.py` when they land, and the
cross-check below is restated over repeats when the ladder for each ligand has as many of them as
M2's reference number does.

**The placement gate passed on every leg.** The smallest nearest-neighbour overlap anywhere is
0.141 against the registered floor of 0.03, and `poor_overlap` is false throughout — twelve even
windows are enough for these three ligands, and no re-placement was needed. The gate is read from
`estimates.MBAR.diagnostics.min_neighbour_overlap`; the first version of the analysis script
guessed three other key spellings and fell through to `null` on all of them, so it reported nothing
and passed every time. A check that cannot fail closes the question instead of asking it.

## The cross-check

| route | ΔΔG_hyd(ethane → chloroethane), kcal/mol |
|---|---|
| decoupling each ligand separately (r1) | **−1.759** |
| mutating one into the other (0.7.0 M2, three repeats) | **−1.743 ± 0.026** |

They agree to **0.016 kcal/mol** — well inside M2's own uncertainty. Two alchemical routes with
different plans, different end states and different particles appearing arrive at the same number.

Against experiment (−2.46) both routes are off by about 0.7 kcal/mol in the same direction. That is
force-field and water-model error — openff-2.2.1/AM1-BCC in TIP3P was not fitted to this — and it
is reported, never gated on.

## What this does not show

* **Nothing about a generated run directory.** These campaigns were driven from Python
  (`md_tools.alchemy.campaign`). `build-md` and `md-run` accept `protocol: alchemical` but refuse to
  run one; see [README.md](README.md).
* **Nothing about protein–ligand binding.** A decoupling in water is not a decoupling in a binding
  site, where the sampling problem is the whole difficulty.
* **Nothing about a mutation with a charge change**, an atom mapping across a ring, or softcore at
  a junction the map had to cut. Those have their own acceptance rows.

## Related

* [Alchemical topology](README.md) — the builder, its configuration and its refusals
* [0.7.0 acceptance matrix](../../development/0.7.0/handoffs/S4-acceptance-matrix.md) — where the
  mutation route's number comes from, row M2.5
