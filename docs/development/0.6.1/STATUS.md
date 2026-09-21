# 0.6.1 status

Updated 2026-09-18 by the coordinator session (S0). The coordinator owns this file; workers report
in [`handoffs/`](handoffs/) and never edit it.

| | |
|---|---|
| branch | `0.6.1` |
| baseline | `e524e0e` on the 0.6.0 development line (released as `v0.6.0` at `3927105`; the `dev-0.6.0` branch has since been deleted) (the commit carrying the 20260918 instruction) |
| contract commit | `a531ce7` on this branch — [shared contracts](../shared-contracts.md) |
| aims | [AIMS.md](AIMS.md) |
| worker branch | `work/0.6.1-selection` (session S1) |
| release state | **not released, not merged, not tagged.** 0.6.0 beneath it is still under the user's testing |

## Milestones

| | milestone | state | evidence |
|---|---|---|---|
| S1-A | mask parser and resolved selection record | IMPLEMENTED | [S1](handoffs/S1.md) rows 1, 10, 17 |
| S1-B | backbone / sidechain membership and torsion ownership | IMPLEMENTED | rows 2, 8 (ff19SB and ff14SB) |
| S1-C | partial-selection Hamiltonian construction | IMPLEMENTED | rows 3–7: tau 0 and no-selector byte-identical to 0.6.0; PME energy `a²E_SS + aE_SE + E_EE` to 1e-9 on Reference |
| S1-D | ligand instances and exclusion files | IMPLEMENTED against a synthetic mapping record; the registered-package row is BLOCKED (sandbox) | rows 2, 19 |
| S1-E | versioned CMAP rule | IMPLEMENTED | rows 2, 9 |
| S1-F | identity, resume and refusal | IMPLEMENTED | rows 11–15: a ladder run by 0.6.0 code extends; selective states refused by name; exports verify; each integration patch reverted fails its tests |
| S1-G | CUDA explicit-solvent ladders | **VALIDATED** 2026-09-20, card 5 (user-granted, in a window hpREST2 freed): `tests/test_selective_rest2_cuda.py` 12 passed, 0 skipped, `--error-on-skip`, 73 s. All four ladders (backbone, sidechain, ligand, combined) completed on CUDA/mixed, `selection_mode: explicit`, fingerprint `md-tools-hamiltonian-identity/v3`, read from `restart.json`. Exchange energies recomputed on CUDA at stored coordinates: worst |u| 0.00582 kT, worst cross 0.00901 kT, against a 0.05 kT tolerance calibrated beforehand — 3–5× the CPU figures, which is what mixed precision costs. Each test also re-checked that the whole-solute Hamiltonian at the same taus is REJECTED, so no pass is vacuous | [S1](handoffs/S1.md) row 18 |

State values are NOT STARTED, IN PROGRESS, IMPLEMENTED (code and deterministic tests), VALIDATED
(evidence on the required platform), or BLOCKED (with the blocker named). IMPLEMENTED is not
VALIDATED, and neither is released.

## Integration commits

| commit | what | checks |
|---|---|---|
| `e36ff56` | merge S1 at `d013ca9`, with its integration patches in `remd/driver.py`, `run/preflight.py`, `remd/executor.py`, `reference/export.py`, `reference/rest2_export.py`, `remd/protocol.py` | code byte-identical to S1's tested tree; S0 reran the non-CUDA selective tests, slow included, CUDA hidden, `MD_DATA` at an empty temp root: 173 passed. Held: `rest2/regions.py::explicit_selection` unclassified in the CUDA matrix |
| (this merge) | merge S1 at `20f4036`: that site classified non-CUDA | inventory, stale-entry and constructor guards PASS, run directly on the in-tree file |
| `62adec3` | merge released **0.6.0** (`dev-0.6.0` = `main` = `v0.6.0` → `3927105`): $MD_DATA test isolation, alias reporting, `--extend-from` refusal writes nothing, OPC 1-4 scale | fast lane: 2270 passed, 3 skipped, 2 known failures (build-top example; `--check` needs a device). `test_selective_rest2_integration.py`: 8 passed, **1 failed as expected** — `test_an_extension_of_the_0_6_0_run_onto_selective_states_is_refused_on_its_hamiltonian` reads `extended/REST2.out`, which a refused `--extend-from` no longer writes; S1 updates it |
| `4c2002e` (+ handoff `4585d14`) | merge S1: caller-supplied ladder rungs (user decision) — `rung_source` declared with a reason, per-rung origins in `restart.json`, the `rungs` identity key with a legacy compatibility branch, run-time re-derivation refused; `solute.yaml` unchanged | fast lane (CUDA hidden, `MD_DATA` temp root, empty afterwards): 2288 passed, 2 environmental failures; ladder/rung/selective/saved-state files: 46 passed; both inventory guards PASS; S1 shows each of 8 guards failing when removed |

## Blockers

- Open for S0: the compact `L01: <path>` form, which needs an instance name in `md-tools-ligand-mapping/1`. The `--extend-from` defect is fixed in released 0.6.0, merged here (`62adec3`), and S1's extension test now asserts the new behaviour.
- Registration: BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every session
  (2026-09-19); the registration half of S1-G is reported BLOCKED, not passed.
- S1-G: DONE, validated on CUDA (see the milestone row). What remains BLOCKED is the registered-package ligand row and dataset registration (the `$MD_DATA` sandbox rule), and the two TYK2 ladders, which need S2's prepared complex and 4 cards.
- Integration: done. Resume and export of a selective ladder work on `0.6.1`.

## TYK2 campaign (S1, cards 1-4)

**Ladder A — ligand only, 32 hot atoms, 4 scaled torsion bonds, tau 0 -> 0.5, 8 rungs — FINISHED
2026-09-21 14:07.** 2,500,000 steps, 2500 exchanges, 5000 ps per state, 40 ns aggregate; 83m22s
wall, 4618 s of it production. Reported by S1; the numbers below are copied from its run, not
restated from a plan.

| gate | result |
|---|---|
| G2-1 ladder completes | PASS |
| G2-2 every rank CUDA/mixed | PASS — MPS devices 0-3 = physical cards 1-4, 2 workers per card |
| G2-3 MPS recorded not inferred | PASS — `execution.acceleration.placement.mps` records `verified: true` with the driver listing the process as an MPS client (M+C), pipe under the session's own cache |
| G2-4 acceptance per pair | 0.590, 0.602, 0.581, 0.550, 0.546, 0.489, 0.483 over the 7 neighbouring pairs; 0.549 overall. No pair at zero |
| G2-5 round trips | 105 over 8 walkers (8, 10, 10, 11, 16, 16, 17, 17); every exchange row a permutation |
| G2-6 exchange energies at stored coordinates | DEFERRED to CUDA, after ladder B, before the cards are released. Correct: at 53,030 atoms a cross-platform recomputation is ~2 kT against a 0.05 kT tolerance, so a CPU check here would be evidence of nothing |
| G2-7 identity | PASS — `selection_mode: explicit`, `md-tools-hamiltonian-identity/v3` |
| G2-8 throughput | **93.5 ns/day per state, 748 ns/day aggregate.** Against 330 ns/day for a single rank alone on one card, two ranks per card give 57% each: a shared card delivers ~1.13x its single-rank throughput, not 2x |

The mid-run figure of 79 ns/day included startup and is superseded; the tutorial quotes 93.5.

**Ladder B — ligand + 19 pocket residues' sidechains, 193 hot atoms, 52 scaled torsion bonds,
tau 0 -> 0.25, 8 rungs — FINISHED 2026-09-21 15:28** (80m54s). 2,500,000 steps, 2500 exchanges,
5000 ps per state. Acceptance 0.408, 0.404, 0.390, 0.388, 0.392, 0.373, 0.358; **0.388 overall
against A's 0.549**, and 45 round trips against A's 105 — roughly half the mixing for six times
the hot region, at half the tau_max. Every exchange row a permutation; explicit selection,
fingerprint v3; MPS verified. Throughput 96.3 ns/day per state, 770.8 aggregate: nominally faster
than A, but A and B did not run against the same background load, so **B is not evidence that a
larger hot region is free.** The per-step cost is the 53,030-atom box either way; that is the
claim the tutorial may make, and only that one.

**G2-6 FAILS for both ladders against the row as written, and the row stays FAILED.** Recomputed
on CUDA/mixed from the float64 checkpoint at exchange 2499, 64 values each: ladder A worst |u|
0.077 kT and worst cross 0.093 kT, ladder B 0.106 and 0.121, against the written 0.05 kT.

The diagnosis is S1's and it is sound: 0.05 kT was calibrated on the 1,760-atom fixture where
|u| ~ 8,000 kT, so 0.0058 kT was 7.3e-7 RELATIVE; TYK2 is 53,030 atoms with |u| ~ 286,000 kT, and
0.09-0.12 kT is 3.1e-7 relative — relatively about twice as good as the fixture. An absolute kT
tolerance does not transfer between system sizes, because mixed-precision error grows with the
number of terms summed. The row is wrong, not the Hamiltonian.

**It is still a FAIL, and it is recorded as one** (S0, 2026-09-21), for the same reason the
clash-tail and pmemd rows are carried: a tolerance is never weakened after a failure, and a bound
fitted to the two measurements that just failed it has no power against the next system. What
replaces it is a NEW row, G2-6b, derived rather than fitted: a relative criterion argued from
mixed-precision accumulation, its PREDICTION written down first, and then checked on a system size
that was not used to set it. Until G2-6b exists and passes that way, **G2-6 on production-sized
systems is UNVALIDATED.**

**G2-8, the scaling probes** (100 ps per state, 50 exchanges, same system and same equilibrated
start; cards 1-4 RELEASED 2026-09-21 after them, MPS daemon shut down and its pipe directory
removed):

| configuration | ranks per card | ns/day per rung | ns/day aggregate |
|---|---|---|---|
| 4 rungs on 4 cards | 1 | 209.9 | 839.5 |
| 8 rungs on 4 cards | 2 | 117.6 | 940.8 |
| 8 rungs on 2 cards | 4 | 69.1 | 553.1 |

Read as a decomposition against 330 ns/day for a bare loop with no ladder: the ladder machinery
(exchange barriers, per-state trajectories, checkpoints) costs **36%**; a second rank on a card
buys **1.12x aggregate** and returns 56% per rung -- the same 1.13x the production ladders gave,
from an independent measurement; a fourth rank **loses throughput outright** (553.1 aggregate,
below the 1-per-card configuration), it does not merely stop helping. Two per card is the setting.

**The probes overestimate sustained throughput by ~26%**: 117.6 ns/day per rung on a 100 ps probe
against the 93.5 the 5 ns production ladder of the same shape sustained. Plan from the PRODUCTION
figure; the probes are evidence for the RELATIVE scaling between configurations and nothing else.
S1 flagged this itself rather than leaving the two numbers to be read as a contradiction.

**G2-6b is registered BEFORE measurement** (S1 `181e8c7`), derived rather than fitted:
`dU ~ K*sqrt(|u|)` from independent single-precision rounding of M summed terms, with K taken from
the 1,760-atom fixture ALONE; TYK2 then lands 1.7x/2.7x above that law, which is what partially
correlated rounding predicts. Bound `3*K*sqrt(|u|)`, the >=100x discrimination check kept in the
row, and a TWO-SIDED prediction for a third, unused system size -- too small refutes K, too large
refutes the sqrt law. It needs one card for ~10 minutes and is requested as its own grant.

Discrimination is not in doubt either way: the whole-solute Hamiltonian at the same taus misses by
15,548 kT (A) and 8,002 kT (B), five orders of magnitude above the discrepancy.
**Ladder B** (ligand + pocket sidechains, tau 0 -> 0.25) started 14:07, expected ~15:30, then G2-6
for both ladders and the three scaling probes, then the cards are released.

## Notes

- No AMBER-mask parser exists anywhere in `src/` today; S1-A is new code, not a wiring job.
- `src/md_tools/rest2/` is identical between `dev` and this baseline, so older notes about those
  files still apply. `src/md_tools/cli/md_openmm.py` and `src/md_tools/openmm/system.py` are
  **not** identical — read them here, not in an older checkout.
