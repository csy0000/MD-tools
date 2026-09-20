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
| M2 | relative hydration ethane → chloroethane (fixture `alchemy-endpoints/1`), hybrid plans by S2, Amber18 softcore by S3; vacuum leg (ethane alone, HBonds, NoCutoff) and TIP3P leg (`ethane-tip3p`, PME 0.9 nm, dispersion on, HBonds); see M2.1–M2.6 | ΔΔG_hyd = ΔG_solv(A→B) − ΔG_vac(A→B) | see sub-rows | see sub-rows | development CPU; the claimed values on CUDA (a grant through S0) | `pytest tests/test_alchemy_hydration.py` | S4.md §M2 | **STOPPED 2026-09-19 13:59, at 5c15d89 on CUDA cards 0 and 1** (granted via S0): every water repeat died with OpenMM's "periodic box size has decreased to less than twice the nonbonded cutoff". The ethane-tip3p fixture box (1.9 nm cube) equilibrates at a mean volume of 6.33 nm³ (1.850 nm edge), only 2.8 % above 2 × 0.9 nm, and 1 ns NPT windows fluctuate to 1.808 nm. A fixture defect, not code; the protocol was NOT changed. State at the stop: water 3/1/3 of 16 windows (r1/r2/r3); vacuum r1 18/18, r2 1/18; vacuum_ba not started. All windows are checkpointed. M2.1–M2.5 NOT RUN pending S0's and the user's decision (a larger fixture, recommended) |
| M2.0 | as M2. **Written 2026-09-19 before any sampling.** Schedule: Amber18 one-step diagonal path (λ_elec = λ_ster = λ_bond = s), 16 windows at s = k/15; 2 fs, LangevinMiddle 1/ps, 300 K; NVT for vacuum, NPT at 1.01325 bar (MC barostat every 25 steps) for water; each window minimised at its own state, then 20 ps equilibration discarded; production 1 ns (water) / 1 ns (vacuum) per window, reports every 1 ps; 3 independent repeats (seeds). A short pilot may change ONLY the window placement, adding windows where the neighbour overlap < 0.03; the pilot is not evidence and the change is recorded | — | — | — | — | — | — | the protocol for M2.1–M2.5 |
| M2.0p | the M2.0 pilot, 2026-09-19, CPU `--cpu`, 16 windows, 40 (vacuum) / 20 (water) samples per window — **not evidence** | neighbour overlap | — | < 0.03 → add windows there, nothing else | CPU | scratch pilot | S4.md §M2 pilot | vacuum: only s = 0 → 1/15 is below (0.026). Water: minimum 0.049, none below. **Decision (applied before production):** the vacuum leg adds s = 1/60 and 1/30 (18 windows); the water leg keeps its 16. The pilot's dU/ds shows why: dU/dλ_bonded is +611 kJ/mol at s = 0 and −238 at s = 1 — the linear mixing of junction terms that the plan removes at the dummy end — identical in both legs by `matched_legs`, so it cancels in ΔΔG but makes the endpoint intervals stiff |
| M2.0f | **fixture change, S0's ruling of 2026-09-19** (option a): M2.0's water environment changes from `ethane-tip3p` **v1** (1.9 nm cube) to **v2** (a cube with edge ≥ 2 × cutoff + 0.8 nm, ≥ 2.6 nm at 0.9 nm; same water model, cutoff, constraints and package, built through build-top with a record, by S2). Reason: v1 cannot hold NPT at the 0.9 nm cutoff (the stopped run of M2). **No free energy had been computed.** Every other part of M2.0 is unchanged. The v1 water windows are VOID (fixture v1, box too small), kept, not deleted, and not reused: the water leg reruns all three repeats on v2. The vacuum legs resume as they are, after `matched_legs` is re-checked against the v2 solvent plan | — | — | — | — | — | — | recorded before the rerun |
| M2.1 | each leg, each repeat | every MBAR neighbour overlap ≥ 0.03, every window ≥ 50 decorrelated samples | — | as stated | as M2 | same | same | NOT RUN |
| M2.2 | each leg, each repeat | TI (complete dU/ds from S3's derivatives, σ_int included) vs MBAR (cross-state energies) | each other: derivative and energy code paths of one Hamiltonian | gate | as M2 | same | same | NOT RUN |
| M2.3 | each leg | the three repeats agree pairwise | each other | gate, σ of each repeat | as M2 | same | same | NOT RUN |
| M2.4 | vacuum leg | ΔG_vac(ethane→chloroethane) + ΔG_vac(chloroethane→ethane) = 0, the second from an independently built B→A plan | closure | gate | CPU (vacuum is cheap) | same | same | NOT RUN |
| M2.5 | the cycle | ΔΔG_hyd with σ (MBAR, repeats combined), through `cycles.relative_hydration` | experiment (FreeSolv: ethane 1.83, chloroethane −0.63 → −2.46 kcal/mol) **reported, not gated**: the force field is not matched to experiment | — | as M2 | same | same | NOT RUN |
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
| P1 | the prepared TYK2 complex per ligand: `build-top` complex, one parameter package per ligand, with build records, as a versioned fixture | S2 (campaign page: prepared once) | NOT STARTED |
| P2 | hybrid plans for the two RBFE edges in BOTH environments (complex and solvent), pairing under `matched_legs` | S2 | NOT STARTED |
| P3 | **a decoupling construction for ABFE**: an end state where the ligand does not interact with its environment. S2's plans are A→B with both endpoints real, and S3's `build_hamiltonian` applies softcore only to unique particles, so a common-particle decoupling would be linear in λ and its integrand would diverge at the end point. This is new work for S2/S3, not a parameter of mine | S0 to assign | **BLOCKED — A5 cannot start without it** |
| P4 | Boresch anchor selection inputs: an equilibrated complex trajectory (≥ 5 ns) per ligand, the ligand's heavy-atom names, and the pocket residue list | S2 prepares, S4 selects and records | NOT STARTED |
| P5 | a measured throughput number (ns/day) for the solvated complex under the alchemical Hamiltonian, from one short window, before any grant request | S4 | NOT RUN |

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

Assumptions, to be replaced by P5's measurement: solvated complex ≈ 40 000 particles; one RTX
3080 ≈ 80 ns/day under the alchemical Hamiltonian (about 10 % below plain MD, from the
cross-state reporting at 2 ps); ligand in water ≈ 3 000 particles ≈ 600 ns/day.

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
