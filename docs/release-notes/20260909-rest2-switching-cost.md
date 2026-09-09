# REST2 tau switching: global parameters, Context reuse, and where it does not work

From cond-LREX's request of 2026-09-09
(`projects/cond-LREX/docs/requests/20260909_mdtools_rest2_switching.md`). Every number below was
re-measured here rather than adopted; where mine differ from theirs, mine are reported and the
difference explained.

## Task 0 — the environment that hid all of this

Confirmed on MD-tools' own baseline, no cond-LREX code involved. `cuda-nvrtc 13.3.33` against a
driver at CUDA 13.0 gives `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` whenever a kernel must be
compiled. PTX is backward compatible only, and CUDA's minor-version-compatibility rule exempts
the PTX JIT, so the solve was legal and could not compile.

| probe | result |
|---|---|
| warm cache (normal) | OK |
| `OPENMM_CACHE_DIR` empty | **fails** |
| **a novel kernel, normal warm cache** | **fails** |

That third row goes beyond the report and is the one that matters: it is not only a cold-cache
problem. Any force combination never compiled on this machine fails, which is why a completed
campaign sat beside a Context that would not start. It also means Task 1 could not have been
tested at all before this was fixed — it introduces a `CustomTorsionForce` and parameter offsets,
i.e. new kernels.

Fixed by installing `cuda-nvrtc=13.0.*` (27 CUDA packages down; OpenMM, Python, AmberTools,
ParmEd and RDKit untouched). Both failing probes now pass.

**Deviation from the request, deliberate.** They asked for the pin in `environment-ci.yml`. That
file is CPU-only by design and adding CUDA there would pull a GPU stack onto runners with no
driver. `environment-cuda.yml` carries it instead, with the reasoning in its header. The pin is
13.0 not as a local workaround but because PTX is backward compatible: a 13.0 runtime works on
every driver at 13.0 or newer, so the oldest supported minor is the conservative choice.

**Separately:** `environment-ci.yml` claimed `tests/test_install.py` asserted its pins matched
`src/md_tools/install/openmm.py`. Neither file exists — lost and never restored — so that
cross-check had not been running. `tests/test_environment_files.py` now does the part that can be
checked.

## Task 1 — tau as global context parameters

`set_amplitude` rewrote every solute particle, exception and torsion and called
`updateParametersInContext`. Re-measured here on 22-atom alanine, ff14SB + GBn2, CUDA mixed:

| | measured here | cond-LREX |
|---|---|---|
| `set_tau`, re-upload | 2.93 ms | 3.18 ms |
| energy evaluation | 0.076 ms | 0.075 ms |
| ratio | **38x** | 43x |
| switch as a share of a `work` update | **95%** | 86% |

The claim holds. The nonbonded terms and the eligible torsions now carry `rest2_a` and
`rest2_a2`, the way `CustomGBForce` already carried `rest2_scale_gb`: scaling the solute's own
charge by `a` and its own epsilon by `a^2` produces both pair factors without either being
written down. `PeriodicTorsionForce` has no offsets, so eligible torsions move to a
`CustomTorsionForce`; ineligible ones stay put, unscaled.

`scaling_for_amplitude` and the work convention are untouched. This is the same Hamiltonian
reached a cheaper way, and the equivalence test is the proof.

## Task 3 — the four combinations, and two honest losses

Equivalence is measured across 10 tau, the 3 probe amplitudes and perturbed configurations.

| force field | solvent | worst \|dU\| | `set_tau` before → after | speedup |
|---|---|---|---|---|
| ff14SB | GBn2 | 5.1e-05 kJ/mol | 2.93 → **0.0026 ms** | **1064x** (globals) |
| ff19SB | GBn2 | 5.0e-05 | 34.8 → 7.26 ms | 4.8x (restore narrowing) |
| ff14SB | TIP3P | 4.6e-05 | 16.4 → **4.57 ms** | 3.6x (restore narrowing) |
| ff19SB | OPC | 1.5e-04 | 71.0 → **13.5 ms** | 5.3x (restore narrowing) |

ff19SB + OPC is the combination that force field was parameterised for, and it is the worst case:
it hits both blockers at once. It is also where the second finding below pays best.

On a `work` update the implicit gain is **17.8x** and on a `components` update **34.2x** -- not
1064x, because the energy evaluations do not get faster and are what remains. Switching was 95% of
an update, so Amdahl caps it near 20x; after the change `set_tau` is 1.6% of a `work` update and
the rest is the potential being computed, which is where the time should be. The evaluation itself
costs ~7-11% more under offsets, which is the price paid for the 1064x.

### CMAP: the re-upload stays, but it got 3.4x cheaper anyway

OpenMM has no `CustomCMAPTorsionForce`, so there is no global to attach and CMAP keeps the
re-upload — which the request allowed as a legitimate outcome. I did not reimplement it as a
tabulated `CustomCompoundBondForce`: that is reimplementing a force rather than rescaling one,
and it should be judged on its own.

Profiling it first found something better. A CMAP switch cost 34.8 ms, of which **27.8 ms was
Python and 3.1 ms the upload** — because `_restore_cmap` restored *every* map. ff19SB defines one
map per residue type, sixteen for alanine dipeptide, of which the solute's torsions reference one.
Restoring a map that is never scaled writes its own values back over itself. Restoring only the
maps `_scale_cmap` scales took the switch from 34.8 ms to 10.1 ms. That is a pure win, independent
of global parameters, and it applies to explicit solvent too.

### The restore was walking the whole system, and that is most of what explicit solvent cost

The CMAP finding generalises, and it is the answer to "can explicit solvent be made faster at
all". `_restore_nonbonded` walked EVERY particle and EVERY exception before rescaling, but
`_scale_nonbonded` touches only the solute and the exceptions with a solute partner. Restoring an
untouched water writes its own values back over itself.

Profiled on 22-atom alanine in a small TIP3P box (406 particles, 482 exceptions):

| | ms |
|---|---|
| `_restore_nonbonded`, all particles | 6.76 |
| `_scale_nonbonded`, all particles | 2.51 |
| `updateParametersInContext` (the actual GPU upload) | **0.28** |

97% of the switch was Python walking solvent that is never scaled, to prepare a 0.28 ms upload.
Restoring only what gets scaled took ff14SB + TIP3P from 16.4 ms to 4.57 ms and ff19SB + OPC from
71.0 ms to 13.5 ms, with the equivalence unchanged.

So explicit solvent does not get global parameters, but it is no longer paying for a full-system
rewrite per switch. The remaining cost is genuinely per-solute work plus, for ff19SB, CMAP.

### Explicit solvent and global parameters: it loses, on a term nobody had named

The request predicted PME might be the problem. It is not — `addParticleParameterOffset` works on
PME and the reciprocal sum follows the offset charges. The problem is the **long-range dispersion
correction**. OpenMM computes that analytic tail from the particles' stored epsilon and does not
apply parameter offsets to it, so the offset route keeps its tau = 0 correction while the
potential moves:

```
useDispersionCorrection = True    tau 0.0: -3.8e-05   0.5: -2.61     0.9: -4.63 kJ/mol
useDispersionCorrection = False   tau 0.0: -3.8e-05   0.5: -4.1e-05  0.9: -4.2e-05
```

Zero at tau = 0, growing monotonically, surviving double precision: the Hamiltonian differing, not
arithmetic. Turning the correction off would make the two agree and would be changing the physics
to suit the implementation, so a System that uses it keeps the re-upload. `TauSwitcher` declines
the fast path and `switching_note` says why.

This is not as costly as it sounds in practice: MD-tools disables the dispersion correction for
implicit solvent and enables it for explicit, so implicit AIS — which is what these campaigns run
— gets the full benefit and explicit correctly does not.

## Task 2 — the counters reported the wrong quantity

They counted energy evaluations, and a tau change dragged a parameter upload with it costing 38x
an evaluation. An update reporting "2 evaluations" spent 95% of its time on the one number nobody
counted. `evaluation_counters` now carries `parameter_change_seconds` beside `probe_seconds`.

In seconds rather than as a count, deliberately: a count of parameter changes is a cost only if
you know what one costs, and after this change that answer differs by three orders of magnitude
between two Systems reporting the same count.

**This corrects my own release note of earlier today.** `20260909-ais-work-measurement.md` called
`ais.work_measurement` "the dominant cost of an AIS run" and reduced `components` from five
evaluations to three. The evaluations were never the dominant cost; the switch was. The
work-measurement change is still right — three evaluations beat five — but it was the smaller term,
and with switching now at 0.0025 ms a `components` update costs 0.24 ms against the old `work`
mode's 3.08 ms. Recording `U` as a function of tau is no longer a cost decision at all, which was
cond-LREX's actual objective.

## Task 4 — one Context per rank: TRIED, MEASURED, AND REVERTED

| | measured here | cond-LREX |
|---|---|---|
| build a fresh Context (cache warm) | 205.6 ms | 165–330 ms |
| point an existing one at a new path | 0.17 ms | 0.143 ms |

Both claims reproduce, and for 100 paths that is 20.6 s of construction against 0.1 s of switching
— after Task 1, construction is the largest fixed cost an AIS campaign pays. It was implemented,
and then reverted, because it changes the science.

**A path's trajectory would depend on how many paths had already run in the same process.** An
integrator's RNG stream carries over, and `setRandomNumberSeed` does not reset it;
`reinitialize(preserveState=False)` does not either, and at 227 ms costs more than building
afresh.

**I got this wrong first, and the tests corrected me.** I measured the effect on CUDA, found that
two FRESH Contexts with identical seeds already diverge by 2.9e-04 nm in a single step — the CUDA
platform is not bitwise deterministic, because force reductions are atomic and order-dependent —
and concluded that reuse was no worse than the status quo. That conclusion held only on the
platform I measured on. On the CPU and Reference platforms the arithmetic IS exact, and there
reuse is plainly visible: `test_ais_invocation_segment_runs.py` and `test_cv_mpi_cuda_ais.py` both
failed, each comparing a path against an uninterrupted reference row by row.

Those tests defend a real contract: a path must be identical whether its campaign ran in one
invocation or several, and under any rank count. Reuse makes the science depend on scheduling,
which is not a trade available for wall clock. The 20 s is well spent.

To revisit it, the missing piece is a way to restore an integrator's RNG stream to its
freshly-seeded state. Without that, this is closed.

**A finding worth separating from the decision:** an AIS run on CUDA is not bit-reproducible from
its seeds, before or after any of this. The per-path seeds buy statistical independence, which is
what the protocol needs; they have never bought bitwise repeatability on that platform.

## Not evaluated: openmmtools' nonequilibrium integrators

MD-tools drives switching from Python with a plain `LangevinMiddleIntegrator`.
`AlchemicalNonequilibriumLangevinIntegrator` and `PeriodicNonequilibriumIntegrator` are installed
— openmmtools 0.26.0 is already a hard dependency for the REST2 sampler — and are not used
anywhere.

They perform the alchemical update inside the integrator's `CustomIntegrator`, on the GPU,
accumulating protocol work there, so there would be no per-update Python round trip at all. That
attacks exactly the case global parameters cannot help: explicit solvent.

It is not a substitution. They expect an alchemically modified System from `openmmtools.alchemy`,
whose lambda conventions are not MD-tools' REST2 tau convention, and they accumulate their own
work definition rather than the three-group decomposition. Whether MD-tools' work convention can
be expressed as one of their protocols is a design question that has not been asked here. Recorded
as unevaluated rather than rejected.

## What a resume must refuse

Reparameterising changes the System's sha256, so a run started before this cannot be continued
after it. That is correct — the Context would be built from a different object graph — and the
existing fingerprint machinery already refuses it. No new mechanism was needed.

## Scope

`scaling_for_amplitude` and the work convention are unchanged. No exchange rule, Hamiltonian
scaling, tau ladder or written value differs. The equivalence tests across four force-field and
solvent combinations are the argument that this is the same physics.
