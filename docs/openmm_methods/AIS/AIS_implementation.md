# How AIS is implemented in this version

This document describes what the AIS runtime actually does, and why the long-range dispersion
correction can be restored analytically rather than recomputed. It complements the README beside
it: the README is how to *use* AIS (inputs, configuration fields, generated files, output
columns); this is how it *works* and what each choice rests on.

Nothing here is aspirational. Every number quoted was measured on this branch, on the system and
device named with it.

---

## 1. What a switching path is

A path takes one configuration from a source ensemble and drives τ from `ais.tau_start` to
`ais.tau_end` over `ais.switching_steps` integration steps, accumulating non-equilibrium work.

**τ is the only public, persisted coordinate.** `s` and `sqrt(s)` are derived inside the scaling
and never written to a configuration, a record or a trajectory. The REST2 convention, expressed
in τ, is:

```text
solute-solute, solute torsions, CMAP     (1 - tau)^2
solute-environment                       (1 - tau)
generalized Born                         (1 - tau)
environment-environment, bonds, angles   1
ordinary amide omega torsions            1        (this repository's convention)
```

With `a = 1 - tau` and `lambda = a^2`, the potential is **exactly quadratic in `a`**:

```text
U(tau, x) = U_non_scaled + a * U_sqrt_scaled + a^2 * U_lin_scaled
```

This is an identity, not an approximation, and it holds term by term — including the PME
reciprocal sum, the Ewald self-energy and the dispersion correction. Section 4 is about the one
term where that identity meets an implementation detail of OpenMM.

## 2. The work convention

```text
dW_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)
```

Parameters move at **frozen coordinates**, and only then does the configuration propagate.
Observation 0 precedes all work and carries exactly zero. `taus[j]` is the Hamiltonian in force
during the j-th propagation interval, so `taus[0]` is the source Hamiltonian before any change.

Two modes, set by `ais.work_measurement` (`md_tools/ais/run.py`):

| mode | evaluations per update | what you get |
|---|---|---|
| `work` (default) | **2** direct evaluations, subtracted | the work integral for the schedule that ran |
| `components` | **3** probe evaluations, one exact quadratic | the same work, plus the three basis terms |

### The two probes, which must never be confused

* the **work-basis probe** runs at the frozen pre-switch `x_j`, where the work convention defines
  work;
* the **observation-potential probe** runs at the coordinate a row *saves*, under that row's τ,
  which is what Hummer–Szabo reweighting consumes.

They are the same arithmetic at different configurations. A row naming a
`coordinate_frame_index` carries potentials recomputed at that frame; a row with no saved
coordinate leaves them **empty** rather than borrowing a neighbour's. Writing the first under
names that read as the second would pair one configuration's work with another's energy, silently.

## 3. The λ-basis probe is not a fit

`md_tools/ais/decomposition.py` measures the three basis components with three potential-energy
evaluations at amplitudes `BASIS_PROBE_AMPLITUDES = (0, 1/2, 1)` — as far apart as the domain
allows — and then solves for the quadratic through them.

`quadratic_through` is an explicit **Lagrange expansion**, written as three divisions by pairwise
differences so the whole computation is visible in the source. Its docstring states the point
plainly:

> Three points determine a quadratic exactly; there is no fitting residual to report and none is
> invented.

This distinction matters. A least-squares fit to sampled data has a residual, and trusting it
afterwards is an act of faith. Three points through a function that is *known* to be a quadratic
is interpolation of an exact form — the coefficients are the function. `GROUPS` names them
`non_scaled`, `sqrt_scaled`, `lin_scaled`.

Every reconstruction is checked against a direct measurement:

```text
U_reconstructed(tau) = non_scaled + a * sqrt_scaled + a^2 * lin_scaled     (Components.total_at)
```

and `verify_every_updates` schedules that comparison. On disagreement beyond the
precision-dependent tolerance the run **refuses** rather than recording components that do not
add up to the potential the path is running under. Both totals are written to the output —
`potential_reconstructed_kj_mol` and `potential_direct_kj_mol` — so the identity is checkable
from the file, by the reader, after the fact.

`ComponentWork.non_scaled` is the literal `0.0`: that group carries `lambda^0` and the
coordinates did not move, so no arithmetic could produce anything else. Computing it as
`non_scaled - non_scaled` to make the column look measured would dress a tautology as evidence.

## 4. The dispersion correction, and why it can be restored analytically

### The problem

OpenMM's `NonbondedForce` with `useDispersionCorrection` adds an analytic long-range tail term
computed from the particles' **stored** epsilon values — and it does **not** apply parameter
offsets when computing it.

That matters because there are two ways to change τ on a live Context:

| path | how τ changes | cost per switch (measured) |
|---|---|---|
| **re-upload** | rewrite solute charges/epsilons, then `updateParametersInContext` | **4.79 ms** |
| **global parameter** | two `context.setParameter` calls on `rest2_a`, `rest2_a2` | **0.012 ms** |

The fast path carries the real parameter values in `addParticleParameterOffset` and leaves the
stored epsilon at zero. So under the offset path OpenMM's tail would stay frozen at its τ = 0
value while the potential around it moved. Measured on 22-atom alanine in 522 TIP3P waters, double
precision:

```text
useDispersionCorrection = True     tau 0.0: -3.8e-05   0.5: -2.61     0.9: -4.63 kJ/mol
useDispersionCorrection = False    tau 0.0: -3.8e-05   0.5: -4.1e-05  0.9: -4.2e-05
```

The error is zero at τ = 0 (where `a = 1` and nothing is scaled), grows monotonically, and
survives double precision — so it is the Hamiltonian differing, not arithmetic. `md_tools.rest2.
scaler.global_switching_refusal` therefore refuses the fast path for any periodic System with the
correction on, and such a System takes the re-upload path: slower, and right.

### Why the tail is exactly quadratic in `a`

The analytic correction is a sum over pairs of a term proportional to `epsilon_ij * sigma_ij^6`,
divided by the volume. Under REST2 scaling each pair falls into exactly one of three classes:

| pair | epsilon scales by |
|---|---|
| environment–environment | `1` |
| solute–environment | `a` |
| solute–solute | `a^2` |

so the tail — a fixed linear combination of those three sums — is a polynomial in `a` of degree
exactly 2. **Crucially, it depends only on the epsilons, the sigmas, the particle count and the
volume. It does not depend on the coordinates.** That is what makes a setup-time determination
valid for every configuration the path ever visits, rather than only for the one it was measured
on.

Measured on three independently built systems, fitting the tail in `a`:

| system | fit | max residual | range over τ 0→0.5 |
|---|---|---|---|
| ff14SB + TIP3P | `-0.044584 a^2 - 3.996222 a - 96.336983` | 7.8e-13 kJ/mol | 2.0315 kJ/mol |
| ff19SB + TIP3P | identical to 9 decimals | 6.3e-13 kJ/mol | 2.0315 kJ/mol |
| tleap box | `-0.010668 a^2 - 4.247548 a - 454.248934` | 3.0e-12 kJ/mol | 2.13 kJ/mol |

Residuals at 1e-12 kJ/mol on a term of order 100 kJ/mol are the arithmetic's noise floor, not a
model error.

### The tail is part of the work, and is recorded today

This is not a reporting detail. The tail varies with τ, so it enters **every** work increment.
Measured on ff14SB + TIP3P, 1796 particles, τ 0.5 → 0.0 in 50 updates at a frozen coordinate,
double precision:

```text
accumulated work, correction ON    -51.980698 kJ/mol      (what is recorded today)
accumulated work, correction OFF   -49.949147 kJ/mol      (tail dropped, not restored)
difference                          -2.031552 kJ/mol  =  -0.81 kT at 300 K
```

Per update the discrepancy is only ~4e-02 kJ/mol — individually invisible, and it **accumulates**
to 0.81 kT, which is a large systematic bias in a free energy. It is recorded correctly today
because `direct_potential()` and `ComponentProbe.measure` both call an *unrestricted*
`context.getState(getEnergy=True)`, and OpenMM folds the tail into that number.

So any implementation that disables the correction **must** restore it at every energy read that
feeds work or observations. The chokepoints are exactly two:

* `md_tools/ais/run.py::direct_potential` — every work value, the verify measurement, and the
  direct observation potential;
* `md_tools/ais/decomposition.py::ComponentProbe.measure` — the three amplitude energies.

Because the restored term is itself an exact quadratic in `a`, adding it back preserves the
`total_at` identity, and `verify_every_updates` continues to check the whole potential against a
direct measurement. The safety net is not bypassed by the optimisation; it still guards it.

### For comparison: how Amber handles the same term

Amber's default is `vdwmeth = 1`, described in the manual as a long-range dispersion correction
"based on an analytical integral assuming an isotropic, uniform bulk particle distribution beyond
the cutoff". There is **no** `gti_*` flag governing it, and the TI chapter's enumeration of what
enters `dV/dλ` never mentions it; every shipped GTI/TI test runs with it on, by default.

Amber nevertheless tracks it. Measured with `pmemd.cuda_DPFP` on
`test/cuda/gti/lambda_remd/multi-window`, single-point energies at matched coordinates:

| λ | tail (kcal/mol) |
|---|---|
| 0.00 | −66.0197 |
| 0.30 | −66.0164 |
| 0.70 | −66.0119 |
| 1.00 | −66.0084 |

The tail moves monotonically with λ — 0.0113 kcal/mol (0.047 kJ/mol) over the full range, in
double precision, so not noise. Amber recomputes it from λ-mixed LJ parameters as part of pushing
λ to the device as a constant (`gpu_ti_dynamic_lambda_`), so it gets both a cheap λ change and a
λ-tracking tail by construction.

The magnitude differs from ours for a structural reason: Amber's alchemical step mutates only the
handful of atoms that differ between the two topologies, whereas REST2 τ-scaling multiplies
**every** solute epsilon by `a^2`. Hence 0.019 kT for that transformation against 0.81 kT for our
τ ramp — the same physics, a much larger lever.

## 5. NVT only, and the refusal that enforces it

**Switching is at fixed volume, and this is a property of the System rather than a runtime flag.**

The tail argument in section 4 depends on it: the correction is a function of epsilons *and
volume*, so it is a function of τ alone only when the volume is constant. Under a barostat the
volume moves and the three coefficients no longer describe the term.

The physics requires it independently of any optimisation: a pressure–volume term in the work
would make the path measure something the Jarzynski and Crooks relations are not written for.

`md_tools/run/preflight.py` refuses a barostatted System before the run directory exists:

```text
the prepared System carries a barostat. AIS switches at FIXED VOLUME: each path keeps the box of
the frame it started from, and no pressure-volume term enters the work. Build the System without
a barostat.
```

This is checked in the **preflight**, so nothing is created before the refusal — no `-odir`, no
`resolved.config`, no `.out`, no `.log`. A `-odir` holding a `resolved.config` is
indistinguishable from a run that happened, so a refusal must leave nothing behind.

A scaled run is fixed-volume *throughout*, equilibration included, and its pressure-coupled
stages are renamed rather than run as NPT with the pressure ignored.

## 6. The schedule

Every length is an **integer step count**. A step count is exact; a duration in picoseconds has
to divide by a timestep the file may not have been written against, and work is path-length
dependent, so a rounded path silently reports the work of a different protocol. Logs derive ps and
ns for the reader.

Three counts must divide exactly, and each failure is refused with the arithmetic that would fix
it rather than rounded away:

```text
switching_steps % parameter_update_interval_steps == 0
observation_interval_steps % parameter_update_interval_steps == 0
switching_steps % observation_interval_steps == 0
```

**Four independent cadences**, because they answer four different questions:

| field | question |
|---|---|
| `observation_interval_steps` | how often the WORK is measured — this one is the method |
| `trajectory_interval_steps` | how often a configuration is written to the path's NetCDF |
| `state_interval_steps` | how often the thermodynamic state is tabulated |
| `checkpoint_interval_steps` | how often the path becomes resumable |

Each divides `switching_steps` on its own, so every stream has a record on the final step. `0`
disables the state table and the checkpoint — never the observations, since a path with no work
rows is not a measurement.

`ais.parameter_update_interval_steps` defaults to **1**: τ changes every step. That is why the
per-switch cost dominates an update, and why section 4's 4.79 ms matters.

## 7. Identity, resume and cost

**An output directory has one identity.** `AIS_run.json` records the source digest, the τ
schedule, the seed policy, the selected frames, the reporting schema, the path count and the
column schema. A second invocation into the same `-odir` with any of those changed is refused by
name: completed paths would be skipped, the rest run under new settings, and the work table
assembled out of two different experiments — readable, and describing neither.

**Path identity does not depend on the worker count.** `paths_for_rank(rank, size, total)` is a
pure function, and global path *n* always writes `AIS_traj000n.nc`. A campaign resumed on a
different number of GPUs lands on the same files.

**An interrupted path resumes mid-path**, from its last committed generation. A checkpoint commit
is a generation transaction: new generation, fsync, digest, sidecar, fsync, then
`current_checkpoint.json` replaced atomically last. Resume follows only that pointer and verifies
the digest — never the newest generation on disk, which is exactly what a crash leaves behind.

A resumed path's prior evaluations are restored as **useful**, not discarded: they produced the
committed work this resume continues from. What is genuinely lost is whatever the dead process
evaluated after the last commit, and a checkpoint cannot know that number, so it is marked
unobservable rather than guessed at or counted as zero.

## 8. Measured cost, for reference

ff14SB + TIP3P, 1796 particles, CUDA, mixed precision, per parameter update. The MD step is
common to every route and excluded.

| route | per update | gain |
|---|---|---|
| re-upload (today) | 5.06 ms | 1× |
| global parameter + analytic tail | 0.31 ms | **16.5×** |

In `components` mode (3 evaluations per update) the same change gives 11.5×. At double precision
it gives 2.8×, because the energy evaluations the switch is divided against become ~9× dearer
while the switch itself does not change.

---

## What this version does not claim

* The 16.5× figure is `work` mode at mixed precision on this system. It is not a promise about
  another system, another mode or another precision, and the three variants above are given so the
  reader can see which applies.
* Under **ff19SB** the CMAP force keeps its re-upload — OpenMM has no `CustomCMAPTorsionForce` —
  so a System with CMAP retains 16 maps / 1 torsion of upload. ff14SB + TIP3P and ff14SB + GBn2
  retain nothing.
* `DEFAULT_TAU_START = 0.5 → DEFAULT_TAU_END = 0.0` runs τ **downhill** (scaled → physical) while
  Amber's λ conventionally runs 0 → 1. Direction fixes which ensemble the Jarzynski average is
  taken over; the reverse path is deliberately not implemented. Confirm the recorded convention
  before combining these work values with a Jarzynski or Hummer–Szabo estimator.
