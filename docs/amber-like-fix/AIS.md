# AIS against Amber's thermodynamic integration

2026-09-14. **Nothing in this document is implemented.** It is an analysis with measurements, and
one concrete proposal that needs your decision because it touches a scientific invariant. No AIS
code was changed.

---

## Amber has three different things, and only one is AIS's counterpart

I expected to find that Amber has no alchemical non-equilibrium switching. That was wrong.

| Amber | coordinate driven | work | our analogue |
|---|---|---|---|
| §27.1 TI | λ **fixed** per window; ⟨∂V/∂λ⟩ averaged, windows combined by MBAR (`ifmbar`, `mbar_states`, `edgembar`) | none | none |
| §27.6 SMD | a **geometric** restraint centre `x₀(t)` | ∫F·dx | `md_tools.umbrella` |
| **§27.8 Jarzynski** | **λ(t), ramped during the run** | Eq. 27.30 | **AIS** |

§27.8 is the real counterpart:

```
exp(−βΔF) = ⟨exp(−βW)⟩                                          (27.28)
W = ∫ (∂H/∂λ)(∂λ/∂t) dt                                          (27.29)
W = Σₙ (∂U/∂λ)|_{r(nΔt)} · [λ(nΔt+Δt) − λ(nΔt)]                  (27.30)
```

λ is incremented by `dynlmb` once every `ntave` steps; `ijarzynski=1` picks `dynlmb` automatically
and adds consistency checks (`ntave > 0`, `mod(nstlim, ntave) == 0`,
`clambda + dynlmb*nstlim/ntave ≤ 1.0`). One simulates the λ_min state, draws N independent samples,
and runs a switching trajectory from each. That is AIS's construction, and it has been in sander
for years.

---

## Where AIS genuinely differs

### 1. The work convention — a real divergence, and ours is better conditioned

Amber accumulates the **thermodynamic gradient** times the λ increment: `(∂U/∂λ)·Δλ`.
We accumulate the **finite potential difference at frozen coordinates**:

```
ΔW_j = U(τ_{j+1}, x_j) − U(τ_j, x_j)
```

These agree only to first order in Δλ. Amber's is exact as Δλ → 0 and carries O(Δλ²) bias at
finite increments; ours is exact at *any* step size, because it is the actual energy change the
switch costs. This is also why `ais.decomposition` can reconstruct `U(τ)` exactly from three
probes — there is no derivative to estimate.

### 2. What is switched

Amber switches an **alchemical** λ: softcore atoms appearing and disappearing, mixed as
`V(λ) = (1 − S_p(λ))V₀ + S_p(λ)V₁`. We switch a **REST2 τ**: the same molecule throughout, solute
charges by `a = 1 − τ`, ε by `a²`, σ untouched.

### 3. Softcore is irrelevant to us — and this can be stated firmly

Softcore (§27.1.6, eqs. 27.5–27.6, `ifsc=1`, `scalpha=0.5`, `scbeta=12`) exists for the
**end-point catastrophe**: linear mixing of `r⁻¹²` diverges as an atom is annihilated. Amber's cure
is a modified distance, `r_ij^VDW(λ;α) = (r_ij^n + λασ_ij^n)^{1/n}`, plus the smoothstep family
`S₀…S₄` whose derivatives vanish at both endpoints so ⟨∂V/∂λ⟩ is numerically integrable.

AIS never annihilates anything:

- `scale_factor_for_tau` **refuses τ ≥ 1**;
- the configs cap τ at **0.95**;
- **σ is never touched** — only charge and ε are scaled.

So no pair distance ever approaches a singular limit. Adopting softcore would import a cure for a
disease our parameterisation cannot contract.

### 4. Two places Amber is ahead

- **`ijarzynski=1` as a refusal gate.** It rejects a λ schedule that overruns the endpoint or does
  not divide the step count — exactly the discipline CLAUDE.md demands elsewhere ("an interval that
  does not divide is refused, never rounded"). Worth auditing `switching_schedule` for the
  equivalent guarantee on τ.
- **Smoothstep scheduling.** Our `INTERPOLATION = "linear"` is the only path we support. Amber's
  `S_p(λ)` exists because a schedule with vanishing endpoint derivatives yields smoother work. A
  real option we do not offer.

---

## Is Amber faster? Yes, architecturally — and here is why

**Amber's λ change is a constant-block copy.** `gpu_ti_dynamic_lambda_` (`cuda/gpu.cpp:1342`):

```c
gpu->sim.AFElambda[0]   = 1.0 - *clambda;
gpu->sim.AFElambda[1]   = *clambda;
gpu->sim.AFElambdaSP[0] = 1.0 - *clambda;
gpu->sim.AFElambdaSP[1] = *clambda;
gpuCopyConstants();
```

Four doubles. Every kernel reads λ via `TISetLambda(CVterm, TIregion, cSim.AFElambdaSP[1])`. Called
from `ti.F90:2973`, once every `ntave` steps.

**Ours rewrites every solute particle, exception and torsion** and calls
`updateParametersInContext` per force. `reparameterise_for_global_switching`'s own recorded
measurement (22-atom alanine, CUDA mixed, ff14SB + GBn2):

> **2.89 ms** against **0.076 ms** for an energy evaluation — **38×** — so a tau change was **95% of
> an AIS `work`-mode update** and the physics was the rest.

And the cadence makes it worse: `parameter_update_interval_steps: 1` by default — τ moves **every
step** — with `work` mode costing two energy evaluations per update. Amber moves λ every `ntave`
steps.

**The critical part: explicit solvent cannot use our fast path.** `global_switching_refusal`
declines global-parameter switching whenever a periodic `NonbondedForce` uses the long-range
dispersion correction, because OpenMM computes that tail from the particles' *stored* epsilons and
does not apply parameter offsets to it. I confirmed this fires on a real TIP3P/PME system:
`switches_by_global_parameter = False`, `globals present: (none)` at both τ = 0 and τ = 0.1. So an
explicit-solvent AIS path pays the 2.89 ms re-upload on every update, and is switch-bound rather
than physics-bound.

---

## Proposed fix — measured, exact, NOT implemented

The refusal is correct as implemented but need not be permanent, because **the tail falls inside the
same quadratic basis the rest of the Hamiltonian obeys.**

### The three-combination test matrix

Measured on systems built through the real `build-top` path, so the force groups and the
resolved force fields are the genuine ones:

| | ff14SB+GBn2 | ff14SB+TIP3P | ff19SB+TIP3P |
|---|---|---|---|
| System route | ParmEd `createSystem` | OpenMM `ForceField` | OpenMM `ForceField` |
| particles | 22 | 1796 | 1796 |
| force groups in `built.xml` | **[0, 1, 2, 11]** | [0] | [0] |
| decomposes as built | **yes** | no — probe engages | no — probe engages |
| global τ switching | **ALLOWED** | REFUSED | REFUSED |
| `switching_note` | "tau set by global parameters" | parameter re-upload | parameter re-upload |
| CMAP present | no | no | **yes** |
| dispersion tail varies over τ 0→0.5 | n/a (no box) | 2.0315 kJ/mol | 2.0315 kJ/mol |
| tail exactly quadratic in `a` | n/a | yes (7.8e-13) | yes (6.3e-13) |

**GBn2 is already on the fast path and already decomposes.** `switches_by_global_parameter` is
True, the decomposition probe does not engage, and `energy_components.csv` has four real columns.
So *both* the performance problem and the decomposition problem are explicit-solvent-only — and
the scaler's 2.89 ms / 0.076 ms benchmark was itself measured on GBn2, the one combination that
does not suffer from either.

**The refusal is force-field-independent.** Both explicit builds are refused for the same
dispersion-correction reason, with tails identical to nine decimal places. ff19SB changes nothing
about it.

### Measurement 1 — the tail is exactly coordinate-independent at fixed box

Reference platform (double), same box, four physically valid configurations of the larger tleap
system (7920 particles, 2630 waters, 104.3 nm³):

| configuration | tail (kJ/mol) |
|---|---|
| as equilibrated | −458.507150201 |
| translated +0.4 nm | −458.507150201 |
| translated −0.9 nm | −458.507150201 |
| jittered 0.02 nm | −458.507150201 |

Spread **0.000e+00**. Two further cases were **excluded as unphysical**: a 0.05 nm jitter reaches
+3.9e8 kJ/mol and a 0.20 nm jitter +1.8e16 kJ/mol — atom overlaps, where the double ULP is itself
~512 and the difference quantises to exactly −512.0. (My first attempt let one of those set the
verdict and wrongly reported coordinate dependence.)

### Measurement 2 — the tail is exactly quadratic in `a = 1 − τ`

**The tail's magnitude is a property of the box, not of the force field.** On the tleap system
above it is ≈ −458.5 kJ/mol; on the `build-top` systems (1796 particles, 590 waters, dodecahedron
19.1 nm³) it is ≈ −100.4 kJ/mol. Both are correct for their own composition and volume. What is
invariant — and what the proposal rests on — is the *functional form*.

`build-top` ff14SB+TIP3P and ff19SB+TIP3P, identical to nine decimals:

| τ | a | tail (kJ/mol) |
|---|---|---|
| 0.0 | 1.0 | −100.377788299 |
| 0.1 | 0.9 | −99.969695153 |
| 0.2 | 0.8 | −99.562493691 |
| 0.3 | 0.7 | −99.156183912 |
| 0.4 | 0.6 | −98.750765817 |
| 0.5 | 0.5 | −98.346239406 |

Fit `−0.044584 a² − 3.996222 a − 96.336983`, **max residual 7.8e-13 kJ/mol** (ff14SB) and
**6.3e-13** (ff19SB). Varies by **2.0315 kJ/mol** over τ = 0 → 0.5.

On the tleap system the fit is `−0.010668 a² − 4.247548 a − 454.248934`, max residual
**3.0e-12 kJ/mol**, varying by **2.13 kJ/mol** — consistent with the 2.61 kJ/mol the existing
refusal docstring reports at τ = 0.5 for a different box again.

In every case the tail is exactly quadratic in `a`, which is the only property the fix needs.

### The proposal

Disable `useDispersionCorrection`, take the global-parameter path, and add the tail back as
`c₀ + c₁a + c₂a²` — three coefficients determined once at setup from three evaluations, exact at
every τ, zero per-step cost.

**Soundness condition:** AIS and REST2 are both **fixed-volume by invariant**, which is exactly
what makes the tail a function of τ alone. This would **not** be valid for an NPT stage, where the
volume moves.

**Projected gain:** the switch drops from ~2.89 ms to ~0.076 ms, so a `work`-mode update goes from
~3.04 ms to ~0.23 ms — **~13× per update**, and the run stops being switch-bound. This is a
projection from the recorded 22-atom GBn2 numbers, **not** a measurement of an explicit-solvent
system.

### Before this becomes work, decide

1. **`CMAPTorsionForce` keeps its re-upload**, but the residue is small. Counted structurally
   (platform-independent, unlike a CPU timing): ff19SB+TIP3P keeps **16 maps / 1 torsion**
   re-uploaded, against **22 particles + 98 exceptions + 29 torsions** converted to offsets.
   ff14SB+TIP3P and ff14SB+GBn2 keep nothing. So the fast path would still capture most of the
   saving under ff19SB — a smaller caveat than it first appeared.
2. **It changes what `system_sha256` describes.** Turning a correction off and restoring it
   arithmetically needs the same "same Hamiltonian, reached a cheaper way" equivalence test
   `test_global_parameter_switching.py` already applies to the scaler.
3. **It edits a scientific invariant's implementation.** I would not do this without a failing test
   first and your explicit agreement.

---

## One thing to check in our own AIS, unrelated to speed

`DEFAULT_TAU_START = 0.5 → DEFAULT_TAU_END = 0.0` runs τ **downhill** (scaled → physical), while
Amber's λ conventionally runs 0 → 1. Direction fixes which ensemble the Jarzynski average is taken
over, and `schedule.py` says the reverse path is "not implemented here, deliberately". Worth
confirming the recorded convention makes the reference state unambiguous before anyone combines our
work values with a Jarzynski or Hummer–Szabo estimator.

---

## Pending benchmark

Not run: GPUs 1–8 are held by the overnight REST2 ladder, and GPU 0 is the peer's A5000.

When free: single-GPU AIS (one path, 50 000 switching steps, `work` mode) against Amber
`ijarzynski=1` TI at matched `nstlim`/`ntave` on the same system, reporting ns/day and the
**switch fraction** of each — across all three combinations, since they behave differently:

| combination | expectation, and what it tests |
|---|---|
| ff14SB+GBn2 | already on the global-parameter path; the *control* — no re-upload to remove |
| ff14SB+TIP3P | re-upload path; the case the fix targets |
| ff19SB+TIP3P | re-upload path **plus** CMAP; measures the residue the fix cannot remove |

Note ff19SB+TIP3P is an off-spec pairing — ff19SB's CMAPs were fit with OPC — and `build-top`
says so, emitting `CROSSED PROTEIN/WATER PAIR ... will build and run; with a water model it was
not fit for` and recording `supported_pair_expects: amber14-all.xml` in the machine record. It is
tested here because it is the only shipped way to get CMAP into an explicit box, not because it
is a combination to run science with.

Everything above is mechanism, not wall-clock measurement, until that runs.
