# Ethane → chloroethane

**Tested against md-tools `0.6.4`.** Every command and every number on these pages comes from a run
executed as written, at that version, on NVIDIA RTX 3080 GPUs with CUDA and mixed precision.

Two molecules that differ by one atom, and the smallest **relative** free energy calculation there
is. This is the pattern every protein–ligand campaign uses: rather than computing two absolute free
energies and subtracting, mutate one molecule into the other and measure only the difference.

## Why this pair

* **The difference is the whole point.** Relative free energy is cheaper and better-conditioned
  than two absolutes, because everything the two molecules have in common cancels instead of being
  sampled twice. Seven of ethane's eight atoms map onto chloroethane; only the mutating end is
  alchemical.
* **Both are in FreeSolv**, so the answer can be checked against experiment: ethane **+1.83**,
  chloroethane **−0.63** kcal/mol, a difference of **−2.46**.
* **There is a second, independent route to the same number.** Decoupling each molecule from water
  separately gives two *absolute* hydration free energies whose difference is the same quantity —
  computed with a different plan, a different Hamiltonian and a different number of legs. That
  cross-check is in [the alchemical validation page](../../openmm_methods/alchemy/validation.md),
  and it is worth more than agreement with experiment.
* **It is small.** Six ladders — two legs, three repeats each — finish in about two hours
  across four cards.

## Two molecules, two boxes, two legs

This is the structural difference from [absolute hydration](../ethanol/hydration.md), and
everything else follows from it:

```text
   ethane(vac) --[ dG_vac ]--> chloroethane(vac)
        |                            |
   dG_hyd(ethane)            dG_hyd(chloroethane)
        v                            v
   ethane(sol) --[ dG_sol ]--> chloroethane(sol)

   ddG_hyd = dG_hyd(chloroethane) - dG_hyd(ethane) = dG_sol - dG_vac
```

**Neither leg is a physical process and neither ΔG is measurable.** Only the difference is, and
only because the cycle closes. A mutation in vacuum has no meaning on its own — but subtract it
from the same mutation in water and what remains is the difference of two hydration free energies.

**The vacuum leg is NOT zero here.** In the decoupling tutorial the molecule's own Hamiltonian is
retained unchanged at both ends, so the vacuum leg is λ-independent by construction and is not run.
Here the molecule itself changes, so its internal energy changes with λ and that leg has to be
sampled.

So you build **two boxes** — ethane in vacuum and ethane in water — and run **two legs** of one
cycle. Both legs live under ONE `<system>`, and the layout says which is which:

```text
<system>/
  input/RHFE-leg1.in          the solvent leg's input, shared by its repeats
  input/RHFE-leg2.in          the vacuum leg's
  RHFE-leg1-run1/  RHFE-leg1-run2/  RHFE-leg1-run3/
  RHFE-leg2-run1/  RHFE-leg2-run2/  RHFE-leg2-run3/
```

`alchemical.cycle` and `alchemical.leg` name the run, exactly as every other protocol's runs are
`<method>-run<N>`. **The cycle is declared, not derived**, and the reason is worth knowing: an
RBFE's solvent leg and an RHFE's solvent leg are the same calculation on the same box. What
distinguishes them is the OTHER leg. Deriving the cycle from one leg would be guessing from
evidence that does not contain the answer.

## Build them

Two `build-top` runs from the same parameter package, differing only in the solvent block. Both
belong to the same `<system>`; they produce the two boxes the two legs run in:

```yaml
# vacuum/build.config
solute:
  kind: ligand
  residue_name: ETA
  parameters: ./parameter        # a PATH, relative to this file
solvent:
  model: vacuum
constraints:
  type: HBonds
```

```yaml
# solvent/build.config
solute:
  kind: ligand
  residue_name: ETA
  parameters: ./parameter
solvent:
  model: TIP3P
  padding_nm: 1.35
  box_shape: cube
  ionic_strength_molar: 0.0
  cutoff_nm: 0.9
constraints:
  type: HBonds
  rigid_water: true
```

Both boxes hold **ethane** — endpoint A. Chloroethane never has a box of its own: it is appended
into ethane's, through the atom map, when the plan is built. Two independently solvated boxes would
not be atom-matched, and `combine-topology` refuses that rather than trying to reconcile them.

## Where to go next

| method | what it is for | |
|---|---|---|
| [**Relative hydration free energy**](relative.md) | ΔΔG_hyd by mutating one molecule into the other | published |
| **Relative binding free energy** | the same machinery in a binding site | planned, 0.6.5 |

*A row with no link is a page that does not exist yet.*

## Related

* [Ethanol / hydration free energy](../ethanol/hydration.md) — the absolute route, by decoupling
* [Alchemical topology](../../openmm_methods/alchemy/README.md) — `hybrid`, `single` and `dual`
  plans, and what each mode claims about the atom map
