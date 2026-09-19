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
| A1 | `combine-topology` construction | S2 | IMPLEMENTED (callable layer, and `md-openmm combine-topology` over it: `md_tools.build.combine`, format `md-tools-combine-topology/1`) — single/dual/hybrid plans; explicit maps and a validated automatic map (`propose_map`, `rdkit-fmcs-heavy/1`: chemically ambiguous survivors refused, symmetric ties accepted; single topology explicit-only); endpoint recovery with internal-group terms; complex leg (capped ALA + ethane, ff14SB/TIP3P); barostat carried through; equal nonzero charge (acetate→propanoate); charge change refused. CMAP row NOT RUN: ff19SB+OPC ligand builds are refused in 0.6.0 (OPC's rounded 1-4 scale wins OpenMM's merge); the fix is `3927105` on dev-0.6.0, being gated | integrated at `92d29c2`; [handoffs/S2.md](handoffs/S2.md); 198 passed on Reference, `MD_DATA` temp root empty |
| A2 | Amber18 softcore, energies and derivatives | S3 | IN PROGRESS — 82 Reference tests; gate 4 vs **CPU** pmemd (Amber 26) PASS at λ 0–1 (≤4.4e-4 kJ/mol, scaled boundary; disclosed rule). **CUDA (GPU 0, user-granted), run 1:** 8 tests, 1 passed, 7 FAILED — CUDA requires one exclusion set across NonbondedForce/CustomNonbondedForce (Reference and CPU do not); fixed with a new CPU invariant test. **Run 2:** 10 tests, 5 passed, 5 FAILED — no construction failures; NVE drift ×3 and the clash-dominated appearing tail (CUDA vs Reference) exceed bounds fixed before the run; under diagnosis on CPU, bounds unchanged. pmemd.cuda: NOT RUN (exited 255 before the first single point). Not yet integrated | `handoffs/S3.md` on `work/0.7.0-hamiltonian` (not yet on this branch) |
| A3 | windows, FEP/BAR/MBAR and TI, hydration | S4 | IN PROGRESS — estimators and windows integrated. Analytic harmonic model (exact 8.633 kcal/mol), CPU: W7 (5 windows) FAIL (partial), gate not moved; **W8 (17 windows, same gate) PASS for all five** — MBAR 8.579±0.068, BAR 8.595±0.063, EXP-fwd 8.591±0.097, EXP-rev 8.546±0.097, TI 8.646±0.081. N1 (NPT, 1038-atom TIP3P/PME, barostat, one window interrupted and resumed) PASS on MBAR and BAR against exact 2.435. **CUDA (card 4, allocated by the user through MD-tools-0.6.0), G1 lane `tests/test_alchemy_windows_cuda.py --error-on-skip`: 5 passed, 0 skipped**, resolved_platform CUDA; W8 on CUDA passes all five estimators; N1 on CUDA passes MBAR 2.260±0.136 and BAR 2.613±0.095 against the exact 2.435. One tolerance changed after a failure and disclosed (`d65d4f0`): the evaluation self-check, 1e-6 relative for every precision, set but never measured, is now 2e-5 for mixed/single (measured scatter up to 1.8e-6 on card 4) and still 1e-6 for double; a 0.5 kJ/mol disagreement is still refused. No molecular hydration leg yet: needs A2 | [handoffs/S4.md](handoffs/S4.md), [S4 acceptance matrix](handoffs/S4-acceptance-matrix.md) |
| A4 | relative binding cycle | S4 | IN PROGRESS — cycle arithmetic only; needs A1 plus A2 for a real leg | [handoffs/S4.md](handoffs/S4.md) |
| A5 | absolute binding cycle with restraints | S4 | IN PROGRESS — Boresch restraints (minimum image), standard-state release term and cycle arithmetic integrated; no molecular binding run | [handoffs/S4.md](handoffs/S4.md) |
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

## Integration commits

| commit | what | checks |
|---|---|---|
| `f04397b` | merge S2 at `7d8b7f9` | held: the CUDA inventory found six unclassified S2 sites, two taking a free `platform` string |
| `6bae714` | `ligands.package` comparison functions made public for S2 | stale-entry guard PASS |
| `84a77b0` | `alchemy` extra: `pymbar>=4,<5`, `scipy` | — |
| `6df0df6` | merge S2 at `ea43898`: recovery Contexts pinned to Reference by constant, six sites classified non-CUDA | 114 passed (S2's files plus all ligand tests, slow included, CUDA hidden); both inventory guards PASS run directly; `MD_DATA` temp root empty afterwards |
| `3748945` | merge S4 at `9b7d605`: samples, estimators, restraints, cycles, windows, 12 windows.py matrix entries (10 CUDA sites, lane NONE YET — BLOCKED) | 130 passed, 3 slow deselected (S4's own CPU campaigns); both inventory guards PASS run directly; `MD_DATA` temp root empty afterwards |
| `192820a` | merge S4 at `60c923b`: W8, N1, `run_window(prepared=, check=)` | 132 passed (non-slow alchemy), both inventory guards PASS run directly, `MD_DATA` temp root empty. Reviewed the disclosed N1 assertion change (`7d925fe`): it still detects a missing pV (0.28 kT against a ~7e-9 kT tolerance) |
| `6695b2b` | merge S2 at `346e6bf`: unique-group internal nonbonded kept at the dummy end (per connected group), n-pentane fixture `internal-v1` | 180 passed (alchemy + ligand, non-slow, CUDA hidden), both inventory guards PASS, `MD_DATA` temp root empty; the annihilation guard test passes on the real plan |

## Blockers

- Registration (A6): BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every
  session (2026-09-19); registration is tested against temporary roots only.
- CUDA: `alchemy/windows.py`'s CUDA sites are covered by S4's G1 lane (PASS on card 4). A2's CUDA lane has 5 of 10 failing (S3, diagnosing); the S1 ladders stay BLOCKED with no card granted.
- The OpenFE adapter choice (A0) is not settled: which components are pinned, at which versions,
  under which license, and what is vendored. S0 owns closing this with S2 and S3.
- No AMBER cross-engine reference environment has been identified yet for gate 4
  (fixed-coordinate comparison at lambda 0, 0.25, 0.5, 0.75, 1). Until one exists, that gate is
  BLOCKED, which is not the same as passed.
