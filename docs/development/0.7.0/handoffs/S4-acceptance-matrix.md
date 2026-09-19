# S4 acceptance matrix — windows, estimators, cycles

Written 2026-09-19, **before** any long campaign. Owned by S4. Thresholds below are fixed now and
are not weakened after a failure; a failing check is fixed in the implementation and rerun.

Verdicts: PASS, FAIL, BLOCKED (named blocker), NOT RUN. A CPU/Reference run is never CUDA
evidence. Every PASS row names the command that produced it; the counts are in
[S4.md](S4.md).

## Fixed thresholds

| name | value | why |
|---|---|---|
| free-energy agreement gate | \|Δ\| < 0.5 kcal/mol **and** \|Δ\| < 3 σ_c, σ_c = √(σ² + σ_ref² + σ_int²) | AIMS.md acceptance criteria; `estimators.agreement_gate` |
| inconclusive | σ_c > 0.25 kcal/mol → INCONCLUSIVE, never PASS | above this a 0.5 kcal/mol error is not distinguishable from noise |
| numerical floor (exact fixtures only) | stated per row, never above 0.5 kcal/mol | 3 σ of an exact computation is round-off |
| TI integration uncertainty σ_int | \|cubic spline − trapezoid\| per the path's segments | the quadrature bias is real (0.56 kJ/mol on the S4 fixture) and must be inside the gate, not beside it |
| neighbour overlap | MBAR overlap O(k, k+1) ≥ 0.03, else `poor_overlap` | Klimovich, Shirts & Mobley 2015 |
| EXP effective samples | Kish ESS ≥ 50, else `poor_overlap` | EXP dominated by a handful of weights is not an estimate |
| cross-state self-check | own-state energy on the evaluation Context = sampling Context to 1e-3 kJ/mol + 1e-6 relative | two Contexts that disagree are two Hamiltonians |
| molecular sampling (A3–A5), before it starts | ≥ 3 independent repeats; ≥ 50 decorrelated samples per window; overlap ≥ 0.03 everywhere; EXP forward/reverse gap reported; gate as above against the matched reference | shared-contracts validation gate 5 |

## Rows

| id | fixture | expected quantity | independent reference | tolerance | platform | command | evidence | verdict |
|---|---|---|---|---|---|---|---|---|
| E1 | S4 harmonic, `harmonic_staged_v1.json` | MBAR ΔG(A→B) | closed form (D kT/2) ln(K1/K0) + C | gate | CPU (numpy) | `pytest tests/test_alchemy_estimators.py` | S4.md §Evidence | PASS |
| E2 | same | BAR, EXP forward, EXP reverse | same | gate | CPU | same | same | PASS |
| E3 | same | TI with σ_int | same | gate incl. σ_int | CPU | same | same | PASS |
| E4 | same, w001/w002 | BAR ΔF and σ | pymbar 4.2.0 `other_estimators.bar` | 1e-8 kT; σ rel 1e-6 | CPU | same | same | PASS |
| E5 | same | statistical inefficiency | pymbar `timeseries.statistical_inefficiency(fast=False)` | rel 1e-10 | CPU | same | same | PASS |
| E6 | deterministic staged path | TI across a knot | exact 8 kJ/mol | 1e-12 | CPU | same | same | PASS |
| E7 | S4 harmonic | estimates under sample shuffling / window reordering | unshuffled | 1e-10 kT | CPU | same | same | PASS |
| E8 | S4 harmonic, NPT variant | u = β(U + pV) | direct arithmetic with OpenMM's R and bar·nm³ | rel 1e-12 | CPU | same | same | PASS |
| R1 | six random particles | Boresch energy and dU/dλ_restraints | independent numpy geometry | 1e-9 abs, 1e-10 rel | Reference | `pytest tests/test_alchemy_cycles.py` | same | PASS |
| R2 | random quadruples | dihedral sign | OpenMM `dihedral()` | 1e-10 | Reference | same | same | PASS |
| R3 | soft restraint | ΔG_release by quadrature | Cartesian grid (anchor) × Haar-measure Monte Carlo (orientation), 2·10⁶ rotations | max(4 SE, 0.01 kJ/mol) | CPU | same | same | PASS |
| R4 | stiff restraint | quadrature vs Boresch eq. 32 | Boresch 2003 closed form | 0.01 kJ/mol | CPU | same | same | PASS |
| C1 | synthetic legs | the signs of the four cycles; standard-state term included; refusals | hand arithmetic | exact | CPU | same | same | PASS |
| W1 | OpenMM harmonic model | cross-state energies and derivatives | analytic expressions | rel 1e-10 | Reference | `pytest tests/test_alchemy_windows.py -m "not slow"` | same | PASS |
| W2 | same | dU/dλ_k | central differences (interior), 3-point one-sided (ends), h = 1e-2, 1e-3, 1e-4 | best h < 1e-5 relative | Reference | same | same | PASS |
| W3 | same | refusals before output (path/Hamiltonian mismatch, foreign `-s`, NPT without a box, rounding schedules) | the output directory does not exist afterwards | exact | CPU `--cpu` | same | same | PASS |
| W4 | same | interrupted then resumed window | uninterrupted run, same seed | byte-identical sample stream | CPU `--cpu` | same | same | PASS |
| W5 | same | edited committed row, changed definition | read-only refusal: every byte in `-odir` unchanged | exact | CPU `--cpu` | same | same | PASS |
| W6 | same | completed window re-run | verified and skipped; no file's mtime changes | exact | CPU `--cpu` | same | same | PASS |
| W7 | OpenMM harmonic model (18 coordinates), 5 windows × 100 000 steps | MBAR, BAR, EXP, TI ΔG | closed form | gate | CPU `--cpu` | `pytest tests/test_alchemy_windows.py -m slow` | S4.md §W7 | **FAIL (partial)**, 2026-09-19: MBAR, BAR and EXP forward PASS. EXP reverse FAILS by 0.81 kcal/mol, flagged `poor_overlap` (Kish ESS 24 < 50). TI is INCONCLUSIVE: σ_int = 0.62 kcal/mol from its quadrature indicator. The cause is a schedule too coarse for 18 coordinates; the gate is unchanged |
| W7b | the W7 data | every estimator that does not PASS is flagged (`poor_overlap`, or INCONCLUSIVE through σ_int); no silent failure | the gate | exact | CPU | same | S4.md §W7 | PASS |
| W8 | same model, **17 windows** (s = k/16; knot at 0.5 included) × 100 000 steps, defined 2026-09-19 after W7 and before sampling | MBAR, BAR, EXP forward, EXP reverse, TI (with σ_int) | closed form | gate, unchanged | CPU `--cpu` | `pytest tests/test_alchemy_windows.py -m slow -k w8` | S4.md §W8 | **PASS**, 2026-09-19, 36 min CPU: MBAR 8.579 ± 0.068, BAR 8.595 ± 0.063, EXP forward 8.591 ± 0.097, EXP reverse 8.546 ± 0.097, TI 8.646 ± 0.081 kcal/mol against the exact 8.633; all five PASS the unchanged gate |
| G1 | any window | real propagation, resume and changed-state refusal on **CUDA** | W4–W6 repeated on CUDA | as W4–W6, except that CUDA resume is not claimed byte-exact until shown | CUDA | the W-tests with `cpu=False` | — | BLOCKED: no GPU allocated to S4 (cards 0–4 held for the 0.6.0 release gate, 5–8 hpREST2's live runs) |
| M1 | absolute hydration, small neutral molecule | ΔG_hyd | matched reference: AMBER TI with the same parameters, or a published value computed with the same force field and protocol | gate; sampling rules above | CUDA | — | — | BLOCKED on A1 (S2 plan) and A2 (S3 molecular Hamiltonian) |
| M2 | relative hydration, one neutral substitution | ΔΔG_hyd, vacuum and solvent legs | as M1 | gate | CUDA | — | — | BLOCKED on A1, A2 |
| M3 | relative binding, simple ligand pair | ΔΔG_bind, solvent and complex legs | as M1 | gate | CUDA | — | — | BLOCKED on A1, A2, and a validated pair (A4) |
| M4 | absolute binding with Boresch restraints | ΔG°_bind incl. restraint attachment and standard-state release | as M1 | gate | CUDA | — | — | BLOCKED on A1, A2 (A5) |
| N1 | two dummies on a λ-dependent harmonic bond in TIP3P/PME water, periodic, MC barostat, 5 windows × 10 000 steps; one window interrupted and resumed | MBAR, BAR ΔG; pV present in u; the box moves | closed form (3/2) kT ln(K1/K0) + C, unaffected by the solvent | gate; pV to rel 1e-12 | CPU `--cpu` | `pytest tests/test_alchemy_windows.py -m slow -k npt` | S4.md §N1 | **PASS**, 2026-09-19, 5 min 48 s CPU: pV in u to rel 1e-12, the box moved, w002 interrupted and resumed; MBAR 2.462 ± 0.163, BAR 2.328 ± 0.079 kcal/mol vs exact 2.435 (asserted). Also reported, not asserted by this row: EXP forward 2.306 ± 0.094, EXP reverse 2.214 ± 0.155, TI 2.431 ± 0.098. A first run failed on the test's own round-off before its gate (see S4.md) |
| X1 | `md-openmm md-run` / `build-md` entry to a window | the same refusals and records through the public command | W3–W6 | exact | CPU, CUDA | — | — | NOT STARTED: CLI and configuration schema are S0's |
| X2 | window export and `$MD_DATA` registration | a registered window campaign re-analysed from the registry | the local analysis | exact | CPU | — | — | NOT RUN: needs a data-contract method entry (S0) |
| X3 | installed wheel, outside the checkout | W-tests against the wheel | the checkout run | same verdicts | CPU, CUDA | — | — | NOT RUN |
