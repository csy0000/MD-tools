# Fixed-tau single-walker MD, and two defects found by validating it

Date: 2026-08-25
Branch: `feature/fixed-tau-md`, from `550b8fc`

## What was added

`cMD.tau` puts a single conventional-MD walker on **one fixed rung of the REST2 ladder**. No
replicas, no exchange, no coupling to anything.

```yaml
cMD:
  tau: 0.5              # 0 is ordinary cMD; s = (1 - tau)^2 is derived
  omega_exclusion: true
```

`tau` is the source parameter, matching the ladder. `s` and `sqrt(s)` are recorded in
`resolved_stage.yaml` as labelled derived diagnostics and are never accepted as input.

This is **not** high-temperature MD. The thermostat stays at `common.temperature_kelvin` and beta
is unchanged; only the solute Hamiltonian is scaled. An "effective solute temperature" describes
that scaling, it is not a second thermostat.

Nothing was reimplemented. `cmd_run.py` keeps its own propagation, checkpointing and restart, and
gets its System from `build_scaled_system` — the same function, in the same file, that
`REST2/run.py` and `REST2/equilibrate.py` use. `md-gen` copies `rest2_scaling.py` into `cMD/` as
well, so a walker and the rung it matches cannot drift apart. `build_stage_system` gained a
`scale_system` hook that runs on the bare System, before the restraint and barostat: those are
stage machinery, not terms of the molecular Hamiltonian, and must not be scaled. That is the same
order `REST2/run.py` uses.

### Equivalence, measured on CUDA in double precision

ACE-ALA-NME, GBn2/mbondi3, HMR 3.024 amu, 4 fs. Not tolerances — exact:

| claim | energy | max per-atom force |
|---|---|---|
| fixed-tau at tau = 0 vs ordinary cMD | `0.000e+00` kJ/mol | `0.000e+00` kJ/mol/nm |
| fixed-tau at tau = 0.5 vs the REST2 tau = 0.5 rung | `0.000e+00` kJ/mol | `0.000e+00` kJ/mol/nm |

At tau = 0 `build_scaled_system` returns the System unmodified, so the first result is exact by
construction rather than by luck. A 1 ns run completed 250,000 steps at 4,850 ns/day with no
non-finite energy.

## Defect 1: implicit REST2 was not scaling the generalised-Born energy

`templates/rest2_scaling.py` handled `NonbondedForce`, `PeriodicTorsionForce` and
`CMAPTorsionForce` and **silently ignored `CustomGBForce`**. Under implicit solvent that is most of
the solvation physics. Measured before the fix, ACE-ALA-NME at tau = 0.5:

```
force                    tau=0       tau=0.5    ratio   expected
PeriodicTorsionForce   10.0005       2.5001    0.2500   s = 0.25
CMAPTorsionForce       -1.6941      -0.4235    0.2500   s = 0.25
NonbondedForce        -97.7280     -24.4320    0.2500   s = 0.25
CustomGBForce         -64.0368     -64.0368    1.0000   s = 0.25   <-- unscaled
```

64 kJ/mol — roughly 26 kT at 300 K — left at full strength while every other term scaled. The run
completes, the exchange log looks healthy, and the acceptance ratio absorbs the discrepancy.

The correct implementation already existed in this repository, in `system.py`
(`_scale_customgb_force`, `audit_force_classes`), with **zero callers**: the reduction to the
six-command CLI left it behind and the generated projects took a reimplementation that had lost the
GB branch. It is now ported into the template, along with the force audit, so an energy-bearing
force nobody classified is refused instead of being left at the wrong scale.

After the fix `CustomGBForce` scales as `s` exactly, at every tau tested, and bonds and angles
remain unscaled by convention.

## Defect 2: neither cMD nor REST2 discarded reporter output past the checkpoint

A checkpoint is written every `checkpoint_interval_ps`; the trajectories and the table are written
far more often. An interrupted stage therefore leaves frames **ahead** of the last checkpoint, and
resuming from that checkpoint replayed the interval and appended frames the files already had.

Measured: a 1 ns fixed-tau run killed at 12 s had written 58 solute frames with the checkpoint at
step 125,000 (50 frames). Resuming produced **108 frames instead of 100, with 8 duplicated steps**
and a step column that was no longer monotonic. Every file looked healthy.

`md_stages.trim_to_checkpoint` now cuts every stream back to the checkpoint before anything is
opened for append — DCD frame counts are rewritten in the header, the table keeps its single
header. Both `cMD/run.py` and `REST2/run.py` call it on resume. After the fix the same
kill-and-resume gives exactly 100 solute frames, 10 whole-system frames, 100 table rows, one
header, no duplicates, monotonic steps.

This is the invariant `CLAUDE.md` states as "treat files beyond the committed boundary as an
uncommitted tail"; it had no test, which is why it survived the rewrite.

## Tests

`tests/test_fixed_tau_md.py`, 15 tests: tau = 0 leaves the System untouched (including not
injecting the GB parameter); the whole GB energy scales by `s` at three taus; a partial enhanced
region is refused under GB; bonds are not scaled; an unclassifiable energy-bearing force is
refused; `cMD.tau` outside `[0, 1)` is refused by the config; the default `cMD.tau` is 0; and the
uncommitted tail is discarded.

`test_the_explicit_tree_is_the_documented_one` was updated: `cMD/` now contains `rest2_scaling.py`.

119 tests pass.

## Not done here

The `system.py` REST2 block is still dead code. It is now the *second* copy of a scaling
convention, and the divergence between the two is exactly what caused defect 1. Deleting it, or
making the template import it, is a separate change and is noted in the simplification inventory.
