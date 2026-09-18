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
| A1 | `combine-topology` construction | S2 | IMPLEMENTED (callable layer; CLI not wired) | integrated at `6df0df6`; [handoffs/S2.md](handoffs/S2.md); 114 passed on Reference, `MD_DATA` at an empty temp root |
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

## Coordinator queue (S0)

| item | why it waits |
|---|---|
| wire `md-run` / `build-md` for alchemical windows (`protocol` value, `.in` keys, `resolved.config`) | the window runner needs S3's Hamiltonian interface to settle first; wiring it against a moving API means doing it twice |
| a data-contract method entry for alchemical datasets | needed before registration can even be tested against a temporary root; real registration stays BLOCKED (sandbox) |
| `md-openmm combine-topology` CLI and its input schema | S2's callable surface is accepted; the input schema is PROPOSED in `topology-plan.md` |

## Integration commits

| commit | what | checks |
|---|---|---|
| `f04397b` | merge S2 at `7d8b7f9` | held: the CUDA inventory found six unclassified S2 sites, two taking a free `platform` string |
| `6bae714` | `ligands.package` comparison functions made public for S2 | stale-entry guard PASS |
| `84a77b0` | `alchemy` extra: `pymbar>=4,<5`, `scipy` | — |
| `6df0df6` | merge S2 at `ea43898`: recovery Contexts pinned to Reference by constant, six sites classified non-CUDA | 114 passed (S2's files plus all ligand tests, slow included, CUDA hidden); both inventory guards PASS run directly; `MD_DATA` temp root empty afterwards |

## Blockers

- Registration (A6): BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every
  session (2026-09-19); registration is tested against temporary roots only.
- The OpenFE adapter choice (A0) is not settled: which components are pinned, at which versions,
  under which license, and what is vendored. S0 owns closing this with S2 and S3.
- No AMBER cross-engine reference environment has been identified yet for gate 4
  (fixed-coordinate comparison at lambda 0, 0.25, 0.5, 0.75, 1). Until one exists, that gate is
  BLOCKED, which is not the same as passed.
