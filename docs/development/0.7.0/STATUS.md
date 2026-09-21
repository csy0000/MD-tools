# 0.7.0 status

Updated 2026-09-18 by the coordinator session (S0). The coordinator owns this file; workers report
in [`handoffs/`](handoffs/) and never edit it.

| | |
|---|---|
| branch | `0.7.0` |
| baseline | `e524e0e` on the 0.6.0 development line (released as `v0.6.0` at `3927105`; the `dev-0.6.0` branch has since been deleted) (the commit carrying the 20260918 instruction) |
| contract commit | `fabbb4a` on `0.7.0` — [shared contracts](../shared-contracts.md) |
| aims | [AIMS.md](AIMS.md) |
| worker branches | `work/0.7.0-topology` (S2), `work/0.7.0-hamiltonian` (S3), `work/0.7.0-execution-analysis` (S4) |
| release state | **not released, not merged, not tagged.** Released only after 0.6.1 is released and integrated |

## Milestones

| | milestone | owner | state | evidence |
|---|---|---|---|---|
| A0 | interfaces, schemas, fixtures, adapter choice | S0 with S2–S4 | IN PROGRESS | [shared contracts](../shared-contracts.md) |
| A1 | `combine-topology` construction | S2 | IMPLEMENTED (callable layer, and `md-openmm combine-topology` over it: `md_tools.build.combine`, format `md-tools-combine-topology/1`) — single/dual/hybrid plans; explicit maps and a validated automatic map (`propose_map`, `rdkit-fmcs-heavy/1`: chemically ambiguous survivors refused, symmetric ties accepted; single topology explicit-only); endpoint recovery with internal-group terms; complex leg (capped ALA + ethane, ff14SB/TIP3P); barostat carried through; equal nonzero charge (acetate→propanoate); charge change refused. CMAP row **PASS** (`complex-cmap-v1`: ff19SB + OPC with CMAP, 750 particles incl. 180 OPC virtual sites, applied 1-4 scale 0.833333; both endpoints recover against amber19-all + OPC). The applied 1-4 scale is read from the environment's build record (`nonbonded_compatibility`, never inferred); package rows are rescaled to it and compared at the unchanged 1e-9; `combine-topology` requires `environment.record` | integrated at `92d29c2`; [handoffs/S2.md](handoffs/S2.md); 198 passed on Reference, `MD_DATA` temp root empty |
| A2 | Amber18 softcore, energies and derivatives | S3 | **IMPLEMENTED; CUDA NOT VALIDATED for the clash case; pmemd.cuda NOT RUN** — integrated at `f0fbe17` (S3 `1feabee`). 99 Reference tests; gate 4 vs CPU pmemd PASS (≤4.4e-4 kJ/mol scaled). CUDA (GPU 0, user-granted), run 3, 12 tests: **9 passed, 3 FAILED** — FD force/energy consistency PASS in double with an injected step and kink caught on the device; one-atom, pentane-plan, live set_state PASS; the 3 failures are all the clash-tail softcore_b_lj term at λ=1 (0.22 nm overlap, 23744 kJ/mol): tail-mixed energy, tail-double dU/dλ, FD-mixed force. The probe shows each platform's derivative is consistent with its own energy (identical FD truncation on both); the platforms' λ-slopes differ by 2.7e-7 relative in double, 3.4e-6 in mixed — platform arithmetic, not a derivative defect. Bounds unchanged; the policy for clash cases is with the user. NVE: smoke test only, shown to have no power. By the user's decision (2026-09-19) the three clash-tail results stay recorded FAILs, precision-limited; any magnitude-relative bound applies to future clash fixtures only. **pmemd.cuda_DPFP** (GPU 0, user-granted; order 4, grid 144, 3.6 nm box), same pre-declared rule, rows separate from CPU pmemd: scaled PASS, scaled tail PASS, unscaled tail PASS, **unscaled one-atom FAIL** (dU/dλ 1.14e-3, shape 9.5e-4 vs tol 9.1e-4; the same row passed on CPU pmemd by 8.7e-4 vs 9.3e-4). The failing row is the non-default `sc_boundary_14: unscaled` path. **Diagnosed** (S3 `cf203c8`, verified by S0): the two engines' Coulomb constants differ — Amber 18.2223² kcal·Å/(mol·e²) = 138.93065 kJ·nm/(mol·e²) against OpenMM's 138.93546, −3.46e-5 relative. Under `gti_add_sc=0` the boundary 1-4 electrostatics (31.5 kJ/mol) sit in the end states but in neither calibration System, so the constant is not absorbed: predicted −1.09e-3 kJ/mol against −1.00e-3 observed, identical on CPU pmemd and pmemd.cuda. A units convention, not an implementation defect; the row stays FAIL, since the rule was not moved. MD-tools uses OpenMM's constant everywhere, so any "Amber18 equivalence" statement must name this 3.5e-5 relative difference in electrostatics Environment virtual sites are carried exactly (a softcore one is refused by name), and on S2's ff19SB + OPC CMAP complex plan (180 M sites) U(0)/U(1) equal the plan's Systems to 1e-7 on Reference; not on CUDA **CUDA run 4 (card 6, user-granted, released): 14 tests, 10 passed, 4 FAILED.** The three clash-tail rows reproduced to every digit (3.427121222102869e-3, 6.306892351858551e-3, 4.192640415341884) — nothing drifted, and they remain the diagnosed FAILs the user decided to carry. NEW: `test_npt_on_cuda_leaves_nothing_stale[double]` FAILED because the run blew up (1e15 kJ/mol); the `[mixed]` case passed VACUOUSLY at that magnitude and S3 records it as a failure too. Test defect, not a Hamiltonian one: the Reference NPT twin passes and FD-in-double passed on the device. Fix on CPU first, with the stale-box power demonstration required before another card **CUDA run 5 (card 0, 2026-09-21, `5a9620c`): 14 tests, 11 passed, 3 failed.** The rebuilt NPT row **PASSES in both mixed and double**, and it discriminates: every comparison is repeated against a Context left at the ORIGINAL box and must differ by >100× the tolerance, with an assertion that the barostat moved the box — a stale PME grid, dispersion coefficient or kappa could not pass it. The three clash-tail rows reproduced BIT-IDENTICALLY to the 6f6e7f7 run hours earlier (0.003427121222102869, 0.006306892351858551, 4.192640415341884), which settles their character: CUDA's deterministic arithmetic on a 0.22 nm overlap, not noise. They remain the recorded FAILs the user decided to carry | [handoffs/S3.md](handoffs/S3.md) |
| A3 | windows, FEP/BAR/MBAR and TI, hydration | S4 | **M2 RE-RUN PASSES EVERY GATE** 2026-09-21 (`2f06ceb`, 156 windows, card 0; see the M2 entry under Blockers) -- overlap, TI-MBAR, repeat scatter and cycle closure all inside the thresholds the FAILED campaign was judged by. **M2.6 is NOT RUN, so no independent engine stands behind the free energy.** The superseded failure is kept below because it is what the fix is evidence against: **M2 COMPLETED AND FAILED ITS GATES**, 2026-09-20 (cards 2+3, 150 windows, 9 legs, 0 skips, 3 h 17 m; no threshold changed). M2.1 FAIL: 4/9 legs below the 0.03 overlap floor, all vacuum. M2.2 INCONCLUSIVE with one FAIL: TI − MBAR 0.37–1.64 kcal/mol. M2.3 FAIL: repeats scatter up to 0.80 kcal/mol against σ_c ≈ 0.09 — the MBAR uncertainty underestimates run-to-run spread by ~10×, while BAR's σ (0.16–0.28) is close to it. M2.4 FAIL: vacuum closure 0.242 vs 0.168. M2.5 reported: ΔΔG −1.80 ± 0.06 vs experiment −2.46, σ marked not credible. **One cause:** `dU/dλ_bonded` = +611 kJ/mol at s=0 and −238 at s=1 — a bonded endpoint singularity from linearly mixing the junction terms the plan removes at a dummy's end — which 16/18 windows cannot resolve. S4 checked the suspicious part first: vacuum r1 and r2 agreeing to 0.00025 kcal/mol is a coincidence (different streams, seeds and fingerprints; r3 is 0.8 away), not a seeding defect. Under analysis: whether junction bonded terms need be λ-dependent at all (S2 + S3, CPU). The earlier W8/N1 harmonic and NPT results stand; they carry no junction terms | [handoffs/S4.md](handoffs/S4.md), `m2_result.json` |
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
| `f0fbe17` | merge S3 at `1feabee`: the Amber18 softcore Hamiltonian | fast lane on the combined tree (CUDA hidden, `MD_DATA` temp root, empty afterwards): 2329 passed, 3 skipped, 2 failed — both environmental, below; both inventory guards PASS |
| `d4ce201` | merge S2 at `53410f6`: applied 1-4 scale from the build record, CMAP complex leg; `integration:` `9727676` makes `environment.record` required in `combine-topology` | 318 passed (alchemy, CLI, ligand incl. aliases; CUDA hidden), both inventory guards PASS, `MD_DATA` temp root empty; committed fixture build logs carry no machine path (hostname and user redacted) |
| `143ff04` | merge S2 `2c1681c` (`junction_policy`, **retain-all by default**, in `PLAN_IDENTITY_FIELDS`, plan schema /4) and S3 `c0dc708` (the plan tests cover BOTH policies) | fast lane 2439 passed, 3 skipped, the 2 environmental failures; both inventory guards PASS. The four plan tests that asserted a differing bonded term now ask for `separable` deliberately, and a mirror test asserts that under retain-all nothing differs, `dU/dλ_bonded` is exactly 0.0, the junction split is 0.0 with no warning, and both end states stay exact |
| `eeb22cd` | merge S2 `7713aa1` (plan schema /5: internal pairs carried symmetrically at both ends; the carried convention asserted where it is PRODUCED, not only where it is consumed; the inventory entry) and S3 `45cacd6` (its xfail becomes a plain assertion) | fast lane 2458 passed, 3 skipped, **0 failed**; all three inventory guards PASS, checked BEFORE the push. **RECORD SHAPES ARE NOW FROZEN** until the TYK2 campaign has run |
| `131547f` | merge S3 at `a3d7285`: environment virtual sites; the Hamiltonian over the CMAP complex plan | 328 passed (alchemy, CLI, ligand; CUDA hidden), both inventory guards PASS, `MD_DATA` temp root empty |
| `5b16b54` | merge S2 at `a4d020b`: `build-top solvent.model: vacuum` (ligand only; refusals by name), `matched_legs` and a ligand-Hamiltonian digest in every plan (OPC-vs-vacuum refused), the applied 1-4 scale always recorded; `build-md` refuses a vacuum System | fast lane (CUDA hidden, `MD_DATA` temp root, empty afterwards): 2366 passed, 3 skipped, 2 failed — the two environmental ones; slow build-top/ligand/protonation files: 39 passed; both inventory guards PASS. |
| `238ffe4` | merge S2 at `70d0179`: `integration:` `a126baa` — `md-run` refuses a System whose own build record (outputs sha256 = this file) says vacuum, before any output; `preflight_stage(vacuum_leg=True)` lets an alchemical vacuum leg through, labelled `vacuum` | fast lane: 2371 passed, 2 environmental failures; guards PASS; `MD_DATA` empty. **Open:** a λ window's `-s` is the Hamiltonian's own serialised System, which no build record hashes — ruled: the plan carries the environment's solvation and S4 labels windows from it |

## Blockers

- **M3 FINISHED 2026-09-21 04:03** (312 windows, 0 errors, CPU): **retain-all is CONFIRMED and the bonded junction term was the whole of M2's fault.**
  - M3.1, the gated row: retain-all −1.7436 ± 0.0370 against separable −1.7690 ± 0.3471 kcal/mol — **agreeing to 0.0254**, twenty times inside the 0.5 ceiling. Reported INCONCLUSIVE rather than PASS because σ_c = 0.349 exceeds the registered 0.25 band: separable cannot certify agreement at the power M3.0 demanded, and that inability IS the defect. The band was not moved.
  - M3.2: retain-all clears the 0.03 overlap floor in all nine legs (0.039–0.068); separable is below it in 5 of 6 vacuum legs (0.007–0.029) — M2's failure reproduced on CPU and removed by the policy.
  - M3.3: vacuum closure 0.00014 ± 0.00042 under retain-all (PASS); 0.0123 ± 0.2746 under separable (INCONCLUSIVE).
  - M3.4: `dU/dλ_bonded` = 0.00 at both ends in every retain-all leg; separable +619/−225, +566/−244, +256/−503 — S3's 660.77 decomposition and S2's 610.585 angle, sampled.
  - Per-leg ΔG differs by construction and is ungated, as registered; S3's cutoff asymmetry is in both arms and cancels from the gated difference. S4 checked the suspicious number first: retain-all's three vacuum repeats agree to 0.0009 kcal/mol with differing stream digests, so it is genuinely better determined, not a duplicated trajectory.
- **M2 RE-RUN PASSES EVERY GATE**, 2026-09-21 15:28 (S4 `2f06ceb`, code at `c6ca0ab`; 156 windows,
  9 legs, 0 skips, 2 h 56 m, all on card 0, now released). **Same thresholds, same schedule, same
  fixture as the failed campaign -- only the construction differs**, which is what makes it evidence
  rather than a second opinion.
  - M2.1 PASS: overlap 0.038-0.084 in all nine legs, 216-529 decorrelated samples (separable had four legs under the floor).
  - M2.2 PASS: TI - MBAR 0.000-0.059 kcal/mol, sigma_c 0.001-0.039 (separable: 0.37-1.64).
  - M2.3 PASS: repeats agree to 0.000-0.090, the vacuum legs to 0.001 (separable: up to 0.80 against sigma_c 0.09).
  - M2.4 PASS: closure 0.0003 kcal/mol against a 0.0012 tolerance (separable: 0.242, FAIL).
  - M2.5 reported and NOT gated: ΔΔG_hyd = -1.743 +/- 0.026 kcal/mol against experiment -2.46.
    The 0.72 kcal/mol gap is force-field and model error (openff-2.2.1/AM1-BCC in TIP3P), not an
    implementation result, and it must never be quoted as one.
  - **M2.6 NOT RUN: there is no independent engine behind this number.** Every check above is
    MD-tools against itself. Internal consistency at this quality is necessary and is not
    sufficient, and the campaign's ΔΔG stays uncorroborated until M2.6 exists.
  - Retain-all ran ~14% faster per water window, on a different card model, unpredicted: an
    OBSERVATION, not a benchmark, and it is not to be cited as a performance result.
- **TYK2 is unblocked.** Its gate was M2 passing; M2 now passes on the same thresholds. The protein
  campaign proceeds to a PLAN -- prerequisites, schedule, and a grant request naming cards and
  hours -- and no sampling starts before the user grants it.
- A pilot that is too short to ESTIMATE overlap must not be allowed to choose window placement: M2.0's rule let a 40-sample pilot do it, and its overlap estimates were optimistic. The rule is being rewritten before the next campaign.

- **Two fast-lane failures are environmental, not defects.** `test_the_build_top_example_resolves_to_the_model_defaults`
  reads the example through `md_tools.configs.example_root()`, which resolves to the WHEEL's data files -- and the shared
  `openmm-env` still has md-tools **0.5.4** installed, whose `build-top.config` says `parameters: null`, not 0.6.0's
  `search`. `test_md_run_check_creates_nothing_not_even_the_output_directory` needs a CUDA device and is unverified with
  CUDA hidden. Neither is fixed by reinstalling the shared environment unilaterally; an installed-wheel lane (gate 6) is
  where both are settled.
- Registration (A6): BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every
  session (2026-09-19); registration is tested against temporary roots only.
- CUDA: `alchemy/windows.py`'s CUDA sites are covered by S4's G1 lane (PASS on card 4). A2's CUDA lane has 5 of 10 failing (S3, diagnosing); the S1 ladders stay BLOCKED with no card granted.
- The OpenFE adapter choice (A0) is not settled: which components are pinned, at which versions,
  under which license, and what is vendored. S0 owns closing this with S2 and S3.
- No AMBER cross-engine reference environment has been identified yet for gate 4
  (fixed-coordinate comparison at lambda 0, 0.25, 0.5, 0.75, 1). Until one exists, that gate is
  BLOCKED, which is not the same as passed.
