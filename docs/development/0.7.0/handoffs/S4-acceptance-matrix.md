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
| cross-state self-check | own-state energy on the evaluation Context = sampling Context to 1e-3 kJ/mol + r·\|E\|; r = 1e-6 for double/Reference, **2e-5 for mixed/single** | two Contexts that disagree are two Hamiltonians. **Recalibrated after a failure, 2026-09-19**: r was 1e-6 for every precision, set before comparison but not measured. On CUDA mixed, two Contexts at identical positions differ by up to 1.8e-6 relative (median 1.5e-3, max 2.0e-2 kJ/mol over 200 frames, 1038-atom PME), and one Context re-evaluated moves 1.1e-3 kJ/mol; the G1 NPT run failed on that noise. Double agrees to 1e-10 and keeps 1e-6. A 0.5 kJ/mol disagreement is still refused (test) |
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
| G1 | the harmonic and NPT/PME models | fresh window on CUDA recording `resolved_platform: CUDA`, `platform_selection: machine-config`; self-check; interrupt/resume; read-only refusals; W8 and N1 on CUDA | W4–W8, N1; the closed forms | as those rows; CUDA resume byte identity recorded, not asserted | **CUDA, card 4 (RTX 3080), mixed precision** | `CUDA_VISIBLE_DEVICES=4 python -m pytest tests/test_alchemy_windows_cuda.py -s -rs --error-on-skip` | S4.md §G1 | **PASS** at `d65d4f0`, 2026-09-19: 5 passed, 0 skipped, 2 min 42 s. The CUDA resume WAS byte-identical to the uninterrupted run for this model. W8 on CUDA: MBAR 8.596, BAR 8.610, EXP fwd 8.612, EXP rev 8.695, TI 8.667 kcal/mol vs 8.633, all PASS. N1 on CUDA: MBAR 2.260 ± 0.136, BAR 2.613 ± 0.095 vs 2.435, both PASS (asserted); EXP and TI are reported, not asserted. The first G1 run failed twice: once on a wrong key in the test, once on the self-check tolerance, which was then recalibrated with measurements (see thresholds) |
| M1 | absolute hydration, small neutral molecule | ΔG_hyd | matched reference: AMBER TI with the same parameters, or a published value computed with the same force field and protocol | gate; sampling rules above | CUDA | — | — | BLOCKED on A1 (S2 plan) and A2 (S3 molecular Hamiltonian) |
| M2 | relative hydration ethane → chloroethane on v2 under `retain-all`, three repeats per leg | ΔΔG_hyd | see M2.1–M2.6 | see sub-rows | **CUDA, card 0 alone** (RTX A5000, mixed), 2026-09-21, commit c6ca0ab | `pytest tests/test_alchemy_hydration_m2.py --error-on-skip` | [m2_result.json](S4-evidence-m2-result.json), S4.md §M2 retain-all | **PASS**: 156 windows, 0 skips, 2 h 56 m. M2.1–M2.4 all PASS; M2.5 reported. Every window ran on ONE card, so there is no per-card throughput caveat |
| M2.0 | as M2. **Written 2026-09-19 before any sampling.** Schedule: Amber18 one-step diagonal path (λ_elec = λ_ster = λ_bond = s), 16 windows at s = k/15; 2 fs, LangevinMiddle 1/ps, 300 K; NVT for vacuum, NPT at 1.01325 bar (MC barostat every 25 steps) for water; each window minimised at its own state, then 20 ps equilibration discarded; production 1 ns (water) / 1 ns (vacuum) per window, reports every 1 ps; 3 independent repeats (seeds). A short pilot may change ONLY the window placement, adding windows where the neighbour overlap < 0.03; the pilot is not evidence and the change is recorded | — | — | — | — | — | — | the protocol for M2.1–M2.5 |
| M2.0p | the M2.0 pilot, 2026-09-19, CPU `--cpu`, 16 windows, 40 (vacuum) / 20 (water) samples per window — **not evidence** | neighbour overlap | — | < 0.03 → add windows there, nothing else | CPU | scratch pilot | S4.md §M2 pilot | vacuum: only s = 0 → 1/15 is below (0.026). Water: minimum 0.049, none below. **Decision (applied before production):** the vacuum leg adds s = 1/60 and 1/30 (18 windows); the water leg keeps its 16. The pilot's dU/ds shows why: dU/dλ_bonded is +611 kJ/mol at s = 0 and −238 at s = 1 — the linear mixing of junction terms that the plan removes at the dummy end — identical in both legs by `matched_legs`, so it cancels in ΔΔG but makes the endpoint intervals stiff |
| M2.0f | **fixture change, S0's ruling of 2026-09-19** (option a): M2.0's water environment changes from `ethane-tip3p` **v1** (1.9 nm cube) to **v2** (a cube with edge ≥ 2 × cutoff + 0.8 nm, ≥ 2.6 nm at 0.9 nm; same water model, cutoff, constraints and package, built through build-top with a record, by S2). Reason: v1 cannot hold NPT at the 0.9 nm cutoff (the stopped run of M2). **No free energy had been computed.** Every other part of M2.0 is unchanged. The v1 water windows are VOID (fixture v1, box too small), kept, not deleted, and not reused: the water leg reruns all three repeats on v2. The vacuum legs resume as they are, after `matched_legs` is re-checked against the v2 solvent plan | — | — | — | — | — | — | recorded before the rerun |
| M2.1 | each leg, each repeat | every MBAR neighbour overlap ≥ 0.03, every window ≥ 50 decorrelated samples | — | as stated | as M2 | same | same | **PASS** 2026-09-21 (retain-all): overlap 0.038–0.084 in all nine legs, 216–529 decorrelated samples. Under `separable` four legs were below the floor |
| M2.2 | each leg, each repeat | TI (complete dU/ds from S3's derivatives, σ_int included) vs MBAR (cross-state energies) | each other: derivative and energy code paths of one Hamiltonian | gate | as M2 | same | same | **PASS** 2026-09-21: TI − MBAR is 0.000–0.059 kcal/mol across the nine legs (σ_c 0.001–0.039). Under `separable` it was 0.37–1.64 |
| M2.3 | each leg | the three repeats agree pairwise | each other | gate, σ of each repeat | as M2 | same | same | **PASS** 2026-09-21: pairwise repeat differences 0.000–0.090 kcal/mol against σ_c 0.001–0.043; the vacuum legs agree to 0.001. Under `separable` they differed by up to 0.80 against σ_c 0.09 |
| M2.4 | vacuum leg | ΔG_vac(ethane→chloroethane) + ΔG_vac(chloroethane→ethane) = 0, the second from an independently built B→A plan | closure | gate | CPU (vacuum is cheap) | same | same | **PASS** 2026-09-21: A→B −1.6385, B→A +1.6382, closure 0.0003 kcal/mol against a 0.0012 tolerance |
| M2.5 | the cycle | ΔΔG_hyd with σ (MBAR, repeats combined), through `cycles.relative_hydration` | experiment (FreeSolv: ethane 1.83, chloroethane −0.63 → −2.46 kcal/mol) **reported, not gated**: the force field is not matched to experiment | — | as M2 | same | same | **reported, not gated** 2026-09-21: ΔΔG_hyd = **−1.743 ± 0.026** kcal/mol (MBAR, three repeats, error-barred by the repeat spread) against experiment −2.46. The 0.72 kcal/mol difference is force-field and model error, not an implementation result: openff-2.2.1/AM1-BCC in TIP3P was never fitted to it |
| M2.6 | the cycle | ΔΔG_hyd | a matched cross-engine sampling reference (Amber 26 pmemd TI/MBAR on the same parameters, through S3's driver) | gate | CPU pmemd | — | — | NOT RUN: not built; pmemd exists (S3 gate 4) |
| M3 | relative binding, simple ligand pair | ΔΔG_bind, solvent and complex legs | as M1 | gate | CUDA | — | — | BLOCKED on A1, A2, and a validated pair (A4) |
| M4 | absolute binding with Boresch restraints | ΔG°_bind incl. restraint attachment and standard-state release | as M1 | gate | CUDA | — | — | BLOCKED on A1, A2 (A5) |
| N1 | two dummies on a λ-dependent harmonic bond in TIP3P/PME water, periodic, MC barostat, 5 windows × 10 000 steps; one window interrupted and resumed | MBAR, BAR ΔG; pV present in u; the box moves | closed form (3/2) kT ln(K1/K0) + C, unaffected by the solvent | gate; pV to rel 1e-12 | CPU `--cpu` | `pytest tests/test_alchemy_windows.py -m slow -k npt` | S4.md §N1 | **PASS**, 2026-09-19, 5 min 48 s CPU: pV in u to rel 1e-12, the box moved, w002 interrupted and resumed; MBAR 2.462 ± 0.163, BAR 2.328 ± 0.079 kcal/mol vs exact 2.435 (asserted). Also reported, not asserted by this row: EXP forward 2.306 ± 0.094, EXP reverse 2.214 ± 0.155, TI 2.431 ± 0.098. A first run failed on the test's own round-off before its gate (see S4.md) |
| X1 | `md-openmm md-run` / `build-md` entry to a window | the same refusals and records through the public command | W3–W6 | exact | CPU, CUDA | — | — | NOT STARTED: CLI and configuration schema are S0's |
| X2 | window export and `$MD_DATA` registration | a registered window campaign re-analysed from the registry | the local analysis | exact | CPU | — | — | NOT RUN: needs a data-contract method entry (S0) |
| X3 | installed wheel, outside the checkout | W-tests against the wheel | the checkout run | same verdicts | CPU, CUDA | — | — | NOT RUN |

## TYK2: absolute (A5) and relative (A4) binding free energies

Written 2026-09-20, **before any protein sampling**, at S0's instruction, for
[the campaign](../../protein-ligand-campaign.md). Model system: TYK2 with `ejm_31`, `ejm_42`,
`ejm_43`. Nothing below has run. M2 must pass first: these campaigns are GPU-days, and the
machinery they exercise is proved on the cheap hydration fixture, not here.

### Thresholds, fixed now

Every threshold in "Fixed thresholds" above applies unchanged (gate, INCONCLUSIVE band, overlap
≥ 0.03, ≥ 50 decorrelated samples per window, the cross-state self-check, 3 repeats).
Two additions, for these rows only:

| name | value | why |
|---|---|---|
| experimental comparison | **reported, never gated** | the force field was not fitted to these measurements, and ABFE carries systematic errors (protonation, buried water, sampling) that a 0.5 kcal/mol gate would attribute to the implementation. Expectation, recorded as an observation: \|error\| ≤ 1.0 kcal/mol for RBFE edges, ≤ 2.0 kcal/mol for ABFE |
| restraint independence (ABFE) | two independent Boresch anchor sets, and one of them at half the force constants, must give the same ΔG° within the gate | ΔG° must not depend on the restraint; this is the one internal check of the ABFE cycle that does not need another engine |

### Prerequisites — none of this is mine, and none exists yet

| # | what | owner | state |
|---|---|---|---|
| P1 | the prepared TYK2 complex per ligand | S2 | **DONE** `tests/data/alchemy/tyk2-v1`: complex 53,030 particles at 8.238 nm, ligand-in-water 1,733–1,928, vacuum 32–38; ff14SB + TIP3P |
| P2 | hybrid plans for the RBFE edges in BOTH environments, pairing under `matched_legs` | S2 | **DONE** (automatic maps, 21-atom MCS, endpoint recovery recorded) |
| P3 | a decoupling construction for ABFE | S2/S3 | **DONE**: `topology.build_decoupling_plan`. A5 is no longer blocked on construction |
| P4 | Boresch anchor selection inputs: an equilibrated complex trajectory (≥ 5 ns) per ligand, the ligand's heavy-atom names, and the pocket residue list | S2 prepares, S4 selects and records | NOT STARTED |
| P5 | measured ns/day for the solvated complex under the alchemical Hamiltonian | S4 | NOT RUN — **run 2** of [the plan](S4-tyk2-plan.md). Every cost below is provisional until it lands |

### Rows

| id | fixture | expected quantity | independent reference | tolerance | platform | command | evidence | verdict |
|---|---|---|---|---|---|---|---|---|
| T1.0 | **protocol, registered before sampling.** Amber18 one-step diagonal path; 16 windows at s = k/15 for every RBFE leg, 20 for each ABFE decoupling leg, 8 for restraint attachment (λ_restraints 0 → 1, ligand coupled); 2 fs, LangevinMiddle 1/ps, 300 K, NPT 1.01325 bar (MC barostat every 25 steps); each window minimised at its own state, 500 ps equilibration discarded, **5 ns production**, reports every 2 ps; **3 independent repeats**; complex and solvent legs of one edge paired by `matched_legs`. The pilot rule of M2.0 applies unchanged: a short pilot may add windows only where neighbour overlap < 0.03 | — | — | — | — | — | — | the protocol for T1–T5 |
| T1 | RBFE `ejm_31` → `ejm_42` (ΔΔG_exp = −0.24 kcal/mol), solvent and complex legs | ΔΔG_bind | experiment, **reported not gated**; gated: repeats, TI vs MBAR, overlap, `matched_legs` | gate | CUDA | `pytest tests/test_alchemy_tyk2.py -k rbfe_42` | — | NOT RUN (P1, P2) |
| T2 | RBFE `ejm_31` → `ejm_43` (ΔΔG_exp = +1.28) | ΔΔG_bind | as T1 | gate | CUDA | `-k rbfe_43` | — | NOT RUN (P1, P2) |
| T3 | RBFE `ejm_42` → `ejm_43`, **approved by the user 2026-09-20**, solvent and complex legs | ΔΔG_bind | as T1 | gate | CUDA | `-k rbfe_42_43` | — | NOT RUN (P1, P2) |
| T3c | **the closed cycle** 31→42→43→31, from T1, T2 and T3 | ΔΔG(31→42) + ΔΔG(42→43) + ΔΔG(43→31) = 0. The third leg is T2 reversed (`reversed_leg`), so its sign is explicit | the cycle's own closure — the only internal accuracy check without a second engine; a systematic error in one edge is invisible in that edge alone | **threshold fixed 2026-09-20, before sampling**: \|sum\| < 0.5 kcal/mol AND < 3 σ_c, σ_c = √(σ₁² + σ₂² + σ₃²) over the three combined-repeat edges; σ_c > 0.25 kcal/mol → INCONCLUSIVE. Same gate as everywhere else; no closure-specific relaxation | CUDA | `-k rbfe_closure` | — | NOT RUN |
| T4 | ABFE `ejm_31` in TYK2: restraint attachment, complex decoupling (restrained), solvent decoupling, analytic release to 1 M | ΔG°_bind | experiment (−9.54 kcal/mol) reported, not gated; gated: restraint independence (two anchor sets), repeats, TI vs MBAR, overlap | gate | CUDA | `-k abfe` | — | NOT RUN (**P3**, P1, P4) |
| T5 | the Boresch restraint chosen for T4 | the release term by quadrature vs Boresch's closed form; anchors away from collinear | rows R3, R4 above, on the real anchors | as R3, R4 | CPU | `-k abfe_restraint` | — | NOT RUN (P4) |

### Anchor selection for T4/T5, fixed now

From a ≥ 5 ns equilibrated complex trajectory (P4): ligand anchors L1, L2, L3 are three heavy
atoms of the ligand, mutually ≥ 0.25 nm apart and not collinear, with the lowest positional
fluctuation; receptor anchors P1, P2, P3 are protein heavy atoms 0.8–1.2 nm from L1, each angle
at least 30° from collinear over the whole trajectory. Equilibrium values are the trajectory
means; force constants are 4184 kJ/mol/nm² and 41.84 kJ/mol/rad² (10 kcal/mol/Å² and
10 kcal/mol/rad², Boresch's own values). Both the anchors and their fluctuations are recorded.
The second anchor set for the restraint-independence check is chosen the same way from a
disjoint set of candidates.

### What it costs, before anyone approves it

**These numbers are now known to be optimistic and are NOT what anything will be requested on.**
They assumed ~40 000 particles and 80 ns/day. P1 landed at **53 030** particles, and S3 measures
the hybrid Hamiltonian at **2.0×** a plain end state per step — together about 2.7× the assumed
per-window cost, turning a 3.3-day leg into ~9 days. P5 (run 2 of [the plan](S4-tyk2-plan.md))
replaces both with one measurement on the real system before any production grant is asked for.

| campaign | windows × repeats × ns | simulated time | one RTX 3080 |
|---|---|---|---|
| T1 complex leg | 16 × 3 × 5.5 | 264 ns | 3.3 days |
| T1 solvent leg | 16 × 3 × 5.5 | 264 ns | 0.45 day |
| T2 | as T1 | 528 ns | 3.75 days |
| T3 (approved) | as T1 | 528 ns | 3.75 days |
| T4 complex decoupling | 20 × 3 × 5.5 | 330 ns | 4.1 days |
| T4 restraint attachment | 8 × 3 × 2.5 | 60 ns | 0.75 day |
| T4 solvent decoupling | 20 × 3 × 5.5 | 330 ns | 0.55 day |
| T4 second anchor set (restraint independence) | complex decoupling + attachment again | 390 ns | 4.85 days |
| **total, three edges + ABFE** | | ≈ 2 600 ns | **≈ 22 GPU-days**, or about 5.5 days on four cards |
| of which the three RBFE edges alone (T1, T2, T3, the path S0 named) | | ≈ 1 580 ns | ≈ 11.5 GPU-days |

Levers, if that is too much: 2 repeats instead of 3 (−33 %), 3 ns production instead of 5
(−40 %, at the cost of precision on the near-null `ejm_42` edge), or dropping the second anchor
set (−4.85 days, at the cost of the only internal ABFE check). Each is a protocol decision to be
recorded **before** sampling, not after.

## M3: the two-policy comparison (junction `retain-all` vs `separable`)

Registered 2026-09-20, **before it runs**, at S0's instruction after M2 failed. One construction
difference, measured rather than bounded: the SAME edge, legs, schedule, lengths, seeds and
analysis, built twice — once with S2's new default `junction_policy = retain-all`, once with
`separable`, the construction M2 ran under. CPU only, no card.

| name | value | why |
|---|---|---|
| agreement of the two ΔΔG | the usual gate: \|Δ\| < 0.5 kcal/mol AND < 3 σ_c; σ_c > 0.25 → INCONCLUSIVE | if they agree, S2's sensitivity bound (0.02–0.05 kJ/mol against a 2.09 kJ/mol gate) is confirmed and `retain-all` is simply correct. If they disagree, **that difference IS the bias**, measured, and it changes the recommendation |
| per-leg ΔG | expected to DIFFER between policies, and not gated | `separable` removes junction terms at a dummy's end, so each leg's endpoint state is a different physical state. The dummy's internal free energy cancels between legs; ΔΔG is where the comparison belongs |
| uncertainty | every combined result is error-barred by **the larger of the estimator's uncertainty and the repeat spread** (`campaign.combine_repeats`) | M2.3: MBAR claimed 0.09 kcal/mol where repeats scattered by 0.80 |

| id | fixture | expected quantity | independent reference | tolerance | platform | command | evidence | verdict |
|---|---|---|---|---|---|---|---|---|
| M3.0 | **protocol, registered before sampling.** Both policies, identical in everything else: vacuum A→B and B→A at 18 windows (the M2 placement), 1 ns production, 50 ps equilibration, 3 repeats; solvent v2 at 16 windows, 200 ps production, 50 ps equilibration, 3 repeats. 2 fs, reports every 1 ps, minimised per window. CPU, one thread per window, windows in parallel. No pilot, and no window placement change: the placement is M2's so that the policy is the only difference | — | — | — | CPU | — | — | the protocol for M3.1–M3.4 |
| M3.1 | both policies | ΔΔG_hyd(retain-all) vs ΔΔG_hyd(separable) | each other | gate | CPU | `pytest tests/test_alchemy_junction_policy.py -k ddg` | — | **INCONCLUSIVE by the registered rule, and the two agree**: retain-all −1.7436 ± 0.0370, separable −1.7690 ± 0.3471 kcal/mol; they differ by **0.0254**, far inside the 0.5 ceiling. The verdict is INCONCLUSIVE only because σ_c = 0.349 exceeds the 0.25 band — and that width is separable's own noise, the defect retain-all removes. Retain-all's ΔΔG is 9× better determined from identical sampling |
| M3.2 | both policies | minimum neighbour overlap and decorrelated samples per leg | M2's values under `separable` | reported; `retain-all` is expected to clear 0.03 where M2 did not | CPU | `-k overlap` | — | **retain-all clears the floor everywhere, separable does not**: retain-all 0.039–0.068 overlap with 43–551 decorrelated samples; separable 0.007–0.029 in **5 of its 6 vacuum legs** (and 0.044–0.075 in solvent). This is M2's failure reproduced on CPU, and removed by the policy |
| M3.3 | both policies | vacuum closure A→B + B→A = 0, each policy | closure | gate | CPU | `-k closure` | — | retain-all **PASS**: A→B −1.6382, B→A +1.6381, closure 0.00014 ± 0.00042 kcal/mol. separable **INCONCLUSIVE**: 0.0123 ± 0.2746. The three retain-all repeats of one vacuum leg agree to 0.0009 kcal/mol with distinct trajectories (stream digests differ); separable's scatter across 0.61 |
| M3.4 | both policies | dU/dλ_bonded at s = 0 and s = 1 | S3's decomposition (660.77 kJ/mol from junction terms, 0.00 from the core) and S2's single 610.585 kJ/mol angle | reported: `retain-all` must show no λ-dependent bonded slots and no endpoint spike | CPU | `-k integrand` | — | **as predicted**: retain-all 0.00 kJ/mol at both end points in all three legs. separable +619.00/−225.45 (solvent), +566.34/−243.89 (vacuum), +255.99/−502.71 (vacuum_ba) — S3's 660.77 decomposition and S2's single 610.585 kJ/mol angle, sampled |

**M2.0's pilot rule is rewritten, and the old one is void.** It let a 40-sample pilot choose
window placement, and its overlap estimates were optimistic — M2's vacuum legs then failed the
0.03 floor at 1 ns. From now on: a pilot may change window placement only if every window has at
least **100 decorrelated samples** (`n / g`, reported per window), and a placement chosen from a
pilot is re-checked against the production run's own overlap; if production overlap falls below
0.03 the campaign FAILS rather than being re-placed after the fact. A pilot that cannot estimate
overlap may not change anything.

## M2.6: the independent engine (AMBER 26 pmemd, CPU)

Declared 2026-09-21, **before any pmemd number exists**, at S0's instruction. The point of this
run is that it is independent; a tolerance chosen after seeing a disagreement would throw that
away. Every check M2 passed is MD-tools against itself.

**What is compared.** The same ethane → chloroethane edge, same packages, same λ grid (M2.0's
16 windows for solvent, 18 for vacuum), same temperature, cutoff, PME settings and constraints,
sampled by pmemd rather than by MD-tools, analysed by MBAR from pmemd's own `ifmbar` energies.
Quantities, in order of what they can show:

| id | quantity | why it comes first |
|---|---|---|
| M2.6a | **single-point calibration**: pmemd vs OpenMM on the plain end states, at fixed coordinates | if the two engines do not agree on the unsampled Hamiltonian, no sampled comparison means anything. This is S3's gate-4 method reused, not a new one |
| M2.6b | ΔG of the **vacuum** leg, both engines, both in **dual topology** and both **unconstrained** | cheap, no solvent, and the sharpest test — see the amendment below |
| M2.6c | ΔG of the **solvent** leg, both engines, dual topology, unconstrained | the expensive one: a dual-topology prmtop with rigid water needs `noshakemask` over the TI region, and pmemd CPU on ~1 900 atoms is hours per window |
| M2.6d | **ΔΔG_hyd**, both engines | the quantity M2.5 reports |

**Amendment, 2026-09-21, before any M2.6b number exists.** As first written, M2.6b compared
pmemd against M2's own vacuum leg. That is TWO differences at once: AMBER TI is **dual topology**
(both end-state copies present, never seeing each other) while M2's leg is **hybrid** (one mapped
core with dummies), and the two constructions differ in the dummy atoms' own internal free
energy, which cancels between legs but not within one. A disagreement would have been
unattributable — the mistake this matrix exists to prevent. Corrected: **M2.6b builds the
MD-tools leg in `mode="dual"`**, so the only difference between the two numbers is the engine.
M2's hybrid number is not the comparison target; the bridge between constructions is ΔΔG, where
the dummy contributions cancel, which is M2.6d. Nothing about the tolerances changes.

**Declared tolerances.**

| source | size | treatment |
|---|---|---|
| statistical | each engine's own σ, error-barred by repeat spread where repeats exist | the gate is \|Δ\| < 0.5 kcal/mol AND < 3 σ_c, σ_c = √(σ_MDtools² + σ_pmemd²); INCONCLUSIVE above σ_c = 0.25, exactly as everywhere else in this matrix |
| Coulomb constant | the engines differ by **3.5e-5 relative** (S3, gate 4) | expected, not a surprise: on the λ-dependent electrostatic part of these legs (tens of kJ/mol) it is ~1e-3 kJ/mol, three orders below the statistical σ. Declared here so it is accounted rather than discovered, and reported in the result |
| pmemd print resolution | 1e-4 kcal/mol = 4.18e-4 kJ/mol per printed energy | added twice, as in S3's rule, to the single-point row M2.6a only |
| erfc table vs exact | pmemd uses `eedmeth=1` under PME and refuses exact erfc | an intentional difference, reported; it is inside M2.6a's calibration by construction |

**M2.6a's rule is S3's, unchanged**: the λ-dependent part of the engines' shared discrepancy,
\|cal_B − cal_A\|, plus two print resolutions. A constant offset cancels in every free energy and
is reported separately.

**What a disagreement would mean, decided now:** M2.6b failing with M2.6a passing implicates the
sampling or the estimator, not the Hamiltonian; both failing implicates the Hamiltonian or the
matching of settings. Either way the result stands as measured and comes back to S2/S3 — the
tolerance is not revisited.

| id | verdict |
|---|---|
| M2.6a | NOT RUN |
| M2.6b | NOT RUN |
| M2.6c | NOT RUN |
| M2.6d | NOT RUN |
