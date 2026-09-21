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
| P5 | measured ns/day for the SOLVATED COMPLEX under the alchemical Hamiltonian | S4 | **NOT RUN** — run 2 below, and every cost here is provisional until it lands |

## Why the earlier cost table is wrong, and by how much

The matrix's cost table assumed ~40,000 particles and 80 ns/day. Two facts have since landed:
the complex is **53,030** particles, and S3 measures the hybrid Hamiltonian at **2.0×** a plain
end state per step. Taken together that is roughly 2.7× the per-window cost assumed there —
a 3.3-day leg becomes ~9 days. **I am not requesting anything on that arithmetic.** P5 replaces
both numbers with one measurement on the real system before any production grant is asked for.

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

### Runs 4+ — production, ONE GRANT PER RUN
T1 (`ejm_31` → `ejm_42`), T2 (`ejm_31` → `ejm_43`), T3 (`ejm_42` → `ejm_43`, the closing edge),
then T4/T5 (ABFE with restraints). Each is a separate request with its cards and hours named,
sized from P5 — never a block of GPU-days. The faster card is excluded from anything with exchange: its
speed advantage is wasted behind an exchange barrier and it must not join a homogeneous ladder.

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
