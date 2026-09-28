# Amber `igb=8` cross-engine comparison, 2026-09-28

The measurement that `amber_igb8_parity_claimed` asserted for its whole life and nobody ever made.
The field was removed in `ee364362` because it was unmeasured; this page is the measurement, run
afterwards. **Its result is that the removed claim was also wrong as worded.**

## What was compared

One prmtop, two engines. `tleap` (openmm-env's AmberTools, the one `build-top` calls) writes the
topology; that same file is then read by **`pmemd` 26.0, CPU** and by **`md_tools.openmm.implicit.
build_implicit_system`**. No GPU is involved anywhere.

```text
pmemd:    imin=1, maxcyc=1, ntb=0, igb=8, gbsa=0, cut=9999.0, ntc=1, ntf=1
md-tools: implicit_model=GBn2, nonpolar_sasa=False, constraints=None, Reference platform
```

`nonpolar_sasa=False` is required for the comparison to mean anything: `gbsa=0` has no nonpolar
term, and Amber's `gbsa=1` is **LCPO**, a different model from the ACE term md-tools now defaults
to. There is no Amber setting that corresponds to the 0.6.2 default build, which is the first
reason a parity claim on it could not have been true.

## Result

| system | atoms | radii | EGB Δ (kcal/mol) | EGB relative |
|---|---|---|---|---|
| ACE-ALA-NME | 22 | mbondi3 → mbondi2 | −0.0006 | 3.9e-5 |
| ALA₈, neutral, minimised | 83 | mbondi3 → mbondi2 | −0.0052 | 2.9e-5 |
| ALA-ASP-GLU-ARG-HIE-LYS-SER-ALA, minimised | 124 | **mbondi2** | −0.0121 | 3.2e-5 |
| the same peptide, same coordinates | 124 | **mbondi3** | **−0.1246** | **3.1e-4** |

Bonded terms agree to the printed precision of `mdout` in every case:

```text
ACE-ALA-NME          pmemd        md-tools      delta
  BOND               0.0203        0.0203       0.0000
  ANGLE              0.3669        0.3669       0.0000
  DIHED              9.6432        9.6432       0.0000
  VDW + EEL + 1-4  -23.3564      -23.3576      -0.0012
  EGB              -15.3046      -15.3052      -0.0006
  TOTAL            -28.6310      -28.6324      -0.0014   (4.9e-5 relative)
```

The worst total-energy disagreement measured is **2.6e-4 relative**, on the charged mbondi3
peptide (−523.90 vs −524.0338 kcal/mol).

## What the fourth row means

Rows 3 and 4 are the **same molecule at the same coordinates with the same charges**. Only the
radius set differs, and the discrepancy moves by a factor of ten. So it is not charge and not
size — row 2 rules out size at 83 neutral atoms, row 3 rules out charge at 124 charged ones.

**It tracks mbondi3, which is md-tools' default.** The old field claimed parity most confidently
exactly where the agreement is weakest.

## What was ruled out

* **`rgbmax`.** Amber truncates the Born-radius integral at 25 Å by default and OpenMM does not.
  Setting `rgbmax=999.0` moves EGB by 0.0010 kcal/mol, against a discrepancy of 0.1246. Not this.
* **System size.** 2.9e-5 at 83 atoms, 3.9e-5 at 22.
* **Formal charge.** 3.2e-5 on the charged peptide under mbondi2.
* **A per-atom parameter mismatch.** Once GBn2's 0.195141 Å offset is applied, **0 of 124 atoms**
  disagree in radius, and the per-element screen factors are identical to the GBn2 table
  (H 1.425952, C 1.058554, N 0.733599, O 1.061039). Note that `tleap`'s prmtop SCREEN column is
  *overridden* by `igb=8` in both engines, so reading it raw makes the two look different when
  they are not — that is a decoding trap, not a finding.

What remains is the GBn2 functional form itself — the neck-function implementation differs between
OpenMM and Amber. **Not chased further**, and this page does not claim to have identified it.

## What this licenses, and what it does not

**Licensed:** describing md-tools' implicit GB as *agreeing with Amber `igb=8` to better than
3e-4 relative on the systems below, with bonded terms exact*. That is good agreement and it is now
a measured statement with a number attached.

**Not licensed:** "parity", "bit-for-bit", or any claim about the 0.6.2 **default** build, which
carries an ACE term Amber has no equivalent for. If a released record ever wants to say something
about Amber again, it should quote a tolerance from a table like this one and name the build it
applies to.

**Four systems, one conformation each, one force field, one Amber version.** Nothing here
generalises to a protein, to a ligand with OpenFF parameters, or to any element outside the
GB-Neck2 fit — those are flagged experimental by `gb_parameter_coverage` for separate reasons.

## Reproducing

Working tree: `scratchpad/amber-parity/{,charged,charged_mbondi2,neutral8}/`, holding each
`tleap` input, prmtop, `sp.in`, and `mdout`. `AMBERHOME=/data3/data/chen/software/amber26`;
`tleap` from `openmm-env`. The peptides are built by `sequence { NALA ... CALA }` and minimised
with `pmemd` (`maxcyc=2000, ncyc=1000`) before the single point, so the comparison is not made at
a clashed extended geometry — the unminimised charged chain has VDWAALS = +1172 kcal/mol and
GMAX = 7122, where every difference is amplified and none of it means anything.
