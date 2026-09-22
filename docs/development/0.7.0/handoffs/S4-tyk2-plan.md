# TYK2 campaign plan (A4, A5)

Written 2026-09-21 by S4, after M2 passed (`2f06ceb`). The acceptance rows are already registered
in [S4-acceptance-matrix.md](S4-acceptance-matrix.md) (T1.0–T5, written before any protein
sampling); this page is the SEQUENCE, the revised cost, and how the grants are asked for.

Nothing here has run.

## What is ready, and what is not

| # | prerequisite | owner | state |
|---|---|---|---|
| P1 | TYK2 + three ligands prepared once: complex, solvated and vacuum builds per ligand, with records | S2 | **DONE** (`tests/data/alchemy/tyk2-v1`, ff14SB + TIP3P, PME 0.9 nm, HBonds; complex 53,030 particles at 8.238 nm, ligand-in-water 1,733–1,928, vacuum 32–38) |
| P2 | hybrid plans for the edges in both environments, pairing under `matched_legs` | S2 | **DONE** (automatic maps, `rdkit-fmcs-heavy/1`, 21-atom MCS; endpoint recovery recorded) |
| P3 | a decoupling construction for ABFE | S2/S3 | **DONE** (`topology.build_decoupling_plan`) — A5 is no longer blocked on construction |
| P4 | Boresch anchors: an equilibrated complex trajectory per ligand, ligand heavy-atom names, pocket residues | S2 prepares, S4 selects | **NOT STARTED** — needs GPU equilibration (run 3 below) |
| P5 | measured ns/day for the SOLVATED COMPLEX under the alchemical Hamiltonian | S4 | **DONE 2026-09-22** — run 2. **163.0 ns/day sustained** on one granted RTX 3080; **697 MiB**, so a 10 GB card carries it and the campaign is not forced onto the one 24 GB card |

## The measured cost (P5, 2026-09-22), and what it replaces

**Measured, one granted RTX 3080, one window of the real T1 complex leg** (53,030 particles in
the plain complex; **53,034 under the hybrid**, the difference being the 4 unique atoms of
`ejm_42`), reports every 2 ps over 16 states:

**Every rate below says which rate it is**, because a reader who assumes "whole window" would
call the sizing optimistic and one who assumes "sustained" would call it pessimistic.

| | ns/day |
|---|---|
| whole window, including plan build, Context creation and minimisation | 99.3 |
| first half of sampling | 146.7 |
| **second half — the sustained rate campaigns are sized from** | **163.0** |

**A short window UNDER-reports here, by 39 %, and the direction matters.** S1's probe caveat runs
the other way (a 100 ps probe read 26 % HIGH), so the caveat cannot simply be inherited: in this
window set-up dominates, and sizing from the naive whole-window number would over-book the
campaign by a third. The curve was measured rather than corrected for — one window polled every
second, first half against second half.

Memory: **697 MiB** on the 3080 (681 MiB on the A5000). The campaign fits a 10 GB card with room.

For comparison, one granted **A5000** gave 138.7 ns/day whole-window against the 3080's 146.2 on
the identical window. **That 5 % is thin as a hardware claim** — one window, one system, and the
0.6.0 session measured these two models within 6 % of each other when idle and 9.4× apart when
one was contended. What it supports is the useful conclusion and no more: **card choice between
these models is not a reason to delay or re-plan this campaign**, and the 3080 pool is adequate.
The A5000 figure is recorded because it was measured, and it sizes nothing.

This replaces the earlier table, which assumed ~40,000 particles and 80 ns/day and was then
corrected to "roughly 2.7× more expensive" from S3's 2.0× per-step ratio. **Both were wrong, and
the pessimistic correction was wrong by more than the original optimism**: measurement gives
163 ns/day sustained, so the campaign is about **10.9 GPU-days, not 22**.

## The sequence, and why this order

### Run 1 — M2.6, the independent engine. CPU, **no card**, ~8 h
S0's instruction, and I agree with the reasoning: every check M2 passed is MD-tools against
itself, so the first independent number should not arrive after the GPU-days are spent. AMBER 26
pmemd **CPU** (which S3's gate 4 already used) runs the same ethane → chloroethane edge, same λ
grid, same lengths, TI and MBAR, against the same parameters. This is the matrix's M2.6, it needs
no card, and it can run while the TYK2 preparation proceeds.

### Run 2 — P5 and the complex plumbing. **1 card, ~1 h**
One complex window, short, on the real 53,030-particle system: measures ns/day under the
alchemical Hamiltonian, and proves the complex leg runs end to end (preflight, NPT barostat,
cross-state evaluation, checkpoint, resume) before anything long starts. Produces the number the
pilot and the production request are sized from.

### Run 3 — equilibration for P4, and the pilot. **1 card, hours set by P5**
Equilibrates each complex (≥ 5 ns) for Boresch anchor selection, then a PILOT of T1's complex
leg long enough to ESTIMATE overlap.

**The pilot rule, as rewritten after M2.0 and applying here:** a pilot may change window
placement only if **every window has at least 100 decorrelated samples** (`n / g`, reported per
window); a placement chosen from a pilot is re-checked against production's own overlap, and
production overlap below 0.03 **FAILS the campaign** rather than re-placing windows after the
fact. M2's pilot chose placement from 40 samples and was optimistic; the campaign then failed at
1 ns. A pilot too short to estimate overlap changes nothing.

### The campaign, sized from the measurement

Per leg: 16 windows × 3 repeats × 5.5 ns (0.5 ns equilibration + 5 ns production), plus ~60 s of
per-window set-up counted explicitly since P5 showed it is not negligible.

| campaign | GPU-days (one 3080) |
|---|---|
| one RBFE edge: complex leg 1.65 + solvent leg 0.24 | **1.89** |
| T1, T2, T3 — the three edges | 5.67 |
| T4 ABFE: complex decoupling, restraint attachment, solvent decoupling | 2.75 |
| T4 second anchor set (restraint independence) | 2.45 |
| **total** | **10.86**, about 2.7 days on four cards |

The solvent-leg figure assumes ~1,900 particles run 8× faster than the complex; that factor is
**assumed, not measured**, and it is 12 % of the total, so it is worth measuring inside T1's own
solvent leg rather than in a separate grant.

### Runs 4+ — production, ONE GRANT PER RUN
T1 (`ejm_31` → `ejm_42`), T2 (`ejm_31` → `ejm_43`), T3 (`ejm_42` → `ejm_43`, the closing edge),
then T4/T5 (ABFE with restraints). Each is a separate request with its cards and hours named,
sized from P5 — never a block of GPU-days. The faster card is excluded from anything with exchange: its
speed advantage is wasted behind an exchange barrier and it must not join a homogeneous ladder.

## Registration of tutorial datasets (the sandbox lift, 2026-09-21)

A simulation that SUCCEEDED and is CITED BY A TUTORIAL may be registered; everything else about
the sandbox stands (no other dataset read, retrieved, altered or deleted). Rules, recorded here
because the moment they apply is weeks after the moment they were given:

- **Path is the contract's, not a convenient one**: `$MD_DATA/2026/tutorials/{data_name}/` via
  `md-openmm data-register`. No `dev` segment, no month segment, and never a directory made by
  hand. The v2 contract is `$MD_DATA/{year}/{project_name}/{data_name}/`.
- **`data_name` and the ALIAS LIST go to S0 for the user's approval BEFORE registering.**
  Registration is write-once and `solute.aliases` defaults to empty, so a dataset registered
  without aliases is permanently unfindable by name.
- **The provenance must carry the corroboration distinction**, not only the handoff: M2.6a
  corroborates the ENERGIES against AMBER; the FREE ENERGY has no independent engine behind it
  while M2.6c/d are unrun. A dataset outlives both the handoff and the person who remembers.
- **Nothing from the FAILED M2 campaign is registered.** It is evidence *against* which the fix
  is measured and it belongs in the repository's record; a registered dataset implies a result
  someone may reuse.
- The TYK2 production runs are the ones worth registering, and they have not happened.

## For the TI-FEP tutorial, decided now rather than at writing time

Three things the tutorial must say, all of them learned from M2.6 rather than from the campaign
it precedes:

1. **"Validated against AMBER" is true of the ENERGIES and false of the free energy.** M2.6a
   corroborates the Hamiltonian at fixed coordinates to ~1e-4 kJ/mol against a tolerance declared
   beforehand. M2's ΔΔG has no independent engine behind it while M2.6c/d are unrun. The tutorial
   keeps those two sentences apart; it is the single most temptingly quotable line in the branch.
2. **Only ΔΔG crosses engines.** A reader comparing a per-leg number against an AMBER tutorial's
   will find a difference that is a convention, not an error: in `mode="dual"` with no common
   atoms MD-tools' vacuum leg is exactly 0 (its Hamiltonian does not depend on λ — asserted by
   `test_a_dual_vacuum_leg_with_no_common_atoms_carries_no_information`), while AMBER scales each
   copy's whole potential with λ and reports the molecules' internal free-energy difference. Both
   reach the same ΔΔG.
3. **Why that was visible at all**: the row had already been corrected once, before any number
   existed, from comparing hybrid-against-dual to dual-against-dual. Had it run as first written,
   the convention difference and an engine difference would have arrived together and neither
   could have been attributed. The tutorial says this in its own words, because it is the habit
   being taught, not an anecdote.

## What each production run must carry

Unchanged from the registered rows: experiment reported and never gated; the internal checks
gated (repeats, TI vs MBAR, overlap, `matched_legs`, cycle closure T3c, and for ABFE the
independence of the result from the restraint); every combined result error-barred by **the
larger of the estimator's uncertainty and the repeat spread**; every leg prepared on CPU and its
preconditions verified before a card is taken.
