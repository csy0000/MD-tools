# 0.7.0 status

Updated 2026-09-18 by the coordinator session (S0). The coordinator owns this file; workers report
in [`handoffs/`](handoffs/) and never edit it.

| | |
|---|---|
| branch | `0.7.0` |
| baseline | `e524e0e` on `dev-0.6.0` (the commit carrying the 20260918 instruction) |
| contract commit | `fabbb4a` on `0.7.0` — [shared contracts](../shared-contracts.md) |
| aims | [AIMS.md](AIMS.md) |
| worker branches | `work/0.7.0-topology` (S2), `work/0.7.0-hamiltonian` (S3), `work/0.7.0-execution-analysis` (S4) |
| release state | **not released, not merged, not tagged.** Released only after 0.6.1 is released and integrated |

## Milestones

| | milestone | owner | state | evidence |
|---|---|---|---|---|
| A0 | interfaces, schemas, fixtures, adapter choice | S0 with S2–S4 | IN PROGRESS | [shared contracts](../shared-contracts.md) |
| A1 | `combine-topology` construction | S2 | NOT STARTED | — |
| A2 | Amber18 softcore, energies and derivatives | S3 | NOT STARTED | — |
| A3 | windows, FEP/BAR/MBAR and TI, hydration | S4 | NOT STARTED | — |
| A4 | relative binding cycle | S4 | NOT STARTED | — |
| A5 | absolute binding cycle with restraints | S4 | NOT STARTED | — |
| A6 | CUDA/CI, exports, registration, wheel validation | S0 | NOT STARTED | — |

State values are NOT STARTED, IN PROGRESS, IMPLEMENTED (code and deterministic tests), VALIDATED
(evidence on the required platform), or BLOCKED (with the blocker named).

## Dependencies between the workers

S3 and S4 do not wait for S2. S3 builds against the agreed miniature topology-plan fixtures; S4
builds against frozen energy and derivative fixtures. Real end-to-end execution waits for A1 and
A2 — individual development does not.

## Integration commits

None yet.

## Blockers

- The OpenFE adapter choice (A0) is not settled: which components are pinned, at which versions,
  under which license, and what is vendored. S0 owns closing this with S2 and S3.
- No AMBER cross-engine reference environment has been identified yet for gate 4
  (fixed-coordinate comparison at lambda 0, 0.25, 0.5, 0.75, 1). Until one exists, that gate is
  BLOCKED, which is not the same as passed.
