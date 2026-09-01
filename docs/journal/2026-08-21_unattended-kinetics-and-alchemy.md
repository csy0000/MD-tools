# Unattended kinetics and alchemy campaign — MD-templates handoff

**2026-08-22.** A two-day unattended campaign runs against this repository at commit **`b3fc8fb`**
(`dev`). This note records only what concerns `MD-templates`: what was generated with it, how it
behaved at production scale, and where the work lives. No data and no analysis code is committed
here.

## Where the work lives

| what | where |
|---|---|
| analysis prototype (local git, `dev`) | `/path/to/MD-analysis-prototype-20260821` |
| simulation data, outside every worktree | `/path/to/MD-analysis-data/20260821_unattended` |
| live queue, GPU map, resume commands | the prototype's `STATUS.md` |
| campaign journal | the prototype's `docs/journal/2026-08-21_unattended-kinetics-and-alchemy.md` |

The prototype is a **local** repository with `main` and `dev`. It was not pushed anywhere and no
GitHub repository, branch or PR was created.

## What was generated with this repository

Six independent 1 microsecond explicit-solvent cMD replicates, all through the accepted generator
and the committed-generation continuation contract:

- **alanine dipeptide x3** — ff19SB + OPC, master seeds 20260821001/002/003;
- **cyclo-(RGDfV) x3** — Sage/openff-2.2.0 + AM1-BCC with **plain TIP3P**, seeds 20260821101/102/103.

Both use 0.15 M NaCl, a dodecahedral box with 1.2 nm padding, a 1.0 nm cutoff, 300 K / 1 bar, HMR to
3.024 amu with a 4 fs timestep, and CUDA mixed precision. Each replicate is 20 committed 50 ns
segments in one run directory: full system every 100 ps (10,000 frames at completion), solute every
10 ps (100,000 frames), state log every 100 ps.

Each replicate was generated and equilibrated independently from its own seed. They are not three
copies of one production checkpoint.

## The water policy did the right thing without being told

The RGDfV systems resolved to `amber19/tip3p.xml` + `tip3p` with `basis: OpenFF Sage (openff-2.x)`
recorded in `water_policy`, and ion parameters sourced from the same file. That is the behaviour the
force-field-family derivation added in `e922073` exists to produce: the macrocycle is a cyclic
pentapeptide, so a label-driven rule would plausibly have called it a peptide and handed it ff19SB's
OPC. `protein_forcefield` is `null` throughout, so an accidental ff19SB load would have been an
error rather than a silent substitution.

## Production-scale behaviour of the cMD contract

The first 50 ns segment committed on all three alanine replicates with watermarks of exactly 500
all-atom frames, 5,000 solute frames and 500 log rows — the contract's values, at a segment length
250x longer than anything the test suite exercises. `cmd_schema_version` is 3, the invocation history
is contiguous from step 0, and the log is monotonic with no NaN or Inf.

Frame counts on disk exceed the watermark while a segment is in flight, which is the designed
behaviour: those are the uncommitted tail, and they are what the recovery path truncates if the run
is interrupted.

## Observations worth recording

**AM1-BCC dominates system preparation for a macrocycle.** OpenFF's ELF10 charge assignment runs
`sqm` over ten conformers; for the 41-heavy-atom RGDfV that took over an hour per molecule with
three running concurrently, and it was the critical path delaying three GPUs. Nothing is wrong with
it, but a campaign that budgets minutes for preparation will be surprised.

**No defect in this repository was exposed.** Nothing in the generator, the continuation contract,
the reporting or the run-state handling misbehaved at production scale, so no regression test or fix
was needed here. If one appears later in the campaign it will be added as a focused change.

## Status at the time of writing

All six replicates are running, one process per GPU, on GPUs 0-5. Alchemical work is prototyped in
the local repository and has not launched production: it will not, unless its full pilot gate
passes. The current queue, PIDs, GPU UUIDs and exact resume commands are in the prototype's
`STATUS.md`.
