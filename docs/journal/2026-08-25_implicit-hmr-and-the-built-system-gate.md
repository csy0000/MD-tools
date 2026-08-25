# Implicit HMR, and the gate that compares a built System against its configuration

Date: 2026-08-25
Branch: `dev`
Starting commit: `56214be` (conservative-defaults work in progress) on top of `a8901ba` / `openmm-v0.1.0`

Not a scientific validation. A correctness fix, with operational evidence.

## The defect, measured before anything was changed

`implicit-md-peptide-hmr-v1` declares `hydrogen_mass: 3.024 amu`, `hmr_scope: solute`,
`timestep: 4 fs`. Building a bundle and deserializing `system.xml`:

```
mass histogram   {1.008: 12, 12.01: 6, 14.01: 2, 16.0: 2}
total mass       144.176 amu      n constraints 12
```

and generating a project against it:

```
eq        5,000 steps   (20 ps, restrained)
cMD_1 1,250,000 steps   (5 ns)   ->  5 ns / 1,250,000 = 4 fs
```

So the implicit performance profiles produced **4 fs on unrepartitioned hydrogens with `HBonds`
alone** — the combination the conservative revert exists to remove, still reachable by opting in.
All four implicit `-hmr-v1` profiles were affected. Both halves were reproduced independently
before any code was touched.

Three separate causes:

- `system_prep.py`'s implicit branch overwrote `hydrogen_mass_amu` and `hmr_scope`
  **unconditionally**, after the user's value had been applied;
- `implicit.py:build_implicit_system` called `createSystem` with no `hydrogenMass=` at all;
- the provenance hard-coded `"hmr": {"scope": "none", ...}`, so the record agreed with the broken
  build rather than with the request.

Neither existing guard could see it. `test_4_fs_is_never_paired_with_unrepartitioned_hydrogens_anywhere`
compares a profile's timestep against **that same profile's** hydrogen mass — a profile against
itself — so it passes for any profile that states both. And
`test_the_implicit_system_is_not_hydrogen_mass_repartitioned` asserted hydrogens stayed below
1.5 amu, which was true *of the defect*.

## Requirement 1: how a performance profile reaches the System builder

**Decision: the serialized System stays authoritative, and the profile is carried into the builder
by a new `MD_system_gen.py --profile`.**

The alternative — record "no HMR applied" in the bundle and repartition later, where the profile is
known — was rejected. It would make `system.xml` a lie: the file would no longer describe what gets
integrated, while the bundle hash, the continuity contract and every provenance record key off
exactly that file. A consumer reading `system.xml` would get masses that never ran.

The cost of the chosen route is that the same fact can be written in two places — the profile that
declares the timestep, and `system_config.json`. That cost is paid down by making disagreement
**impossible to ignore** rather than by removing one of the declarations:

- `MD_system_gen.py --profile <id>` takes the hydrogen mass from the profile, so the profile is the
  single source and `system_config.json` need not restate it;
- if both state it and they differ, the build is **refused with both values named**;
- `MD_input_gen.py` compares the resolved profile against the bundle's recorded `hmr` and refuses,
  naming the profile's value, the bundle's value, the timestep that would have been used, and the
  command that fixes it.

Nothing is averaged, and nothing wins by precedence. Either value could be the intended one, and
guessing wrong changes the masses that get integrated.

```
MD_input_gen: the protocol's hydrogen mass repartitioning disagrees with the prepared bundle, and
the masses that would actually be integrated are the bundle's:
    profile 'implicit-md-peptide-hmr-v1' asks for : hydrogen_mass=3.024 amu, hmr_scope='solute'
    the bundle's System carries    : hydrogen_mass=none amu, hmr_scope='none'
    the protocol would integrate at: 4 fs
```

## Requirement 2–3: the repartitioning, and provenance that states what happened

`build_implicit_system` now delegates to ParmEd's own `createSystem(hydrogenMass=...)` rather than
adding a third implementation of arithmetic that already exists twice in `system.py`. Under
implicit solvent there is no solvent, so `"solute"` and `"all"` select the same particles; that is
recorded in the `hmr` block rather than left to look like a setting that was ignored. Total mass is
asserted conserved and a heavy atom driven below 1 amu is refused, both at build time.

```
mass histogram   {3.024: 12, 5.962: 3, 9.994: 1, 11.994: 2, 12.01: 2, 16.0: 2}
total mass       144.176 amu      (conserved exactly)     n constraints 12

"hmr": {"scope": "solute", "target_hydrogen_mass_amu": 3.024, "n_hydrogens": 12,
        "n_hydrogens_repartitioned": 12, "total_mass_amu_before": 144.176,
        "total_mass_amu_after": 144.176, "lightest_heavy_atom_amu": 5.962,
        "scope_note": "implicit solvent has no solvent atoms, so 'solute' and 'all' select the
                       same particles"}
```

The count is *measured* — particle masses compared against the topology's elements — so a record
claiming twelve means twelve were found to have moved.

The superseded note that justified no-HMR by appealing to the pinned GBn2 reference is deleted
rather than edited around. `e187c85` overturned it: mass enters the kinetic term only.

## Requirement 5: the GBn2 identity, measured

Same positions, two Systems from one prmtop, differing only in particle masses, on ACE-ALA-NME:

```
GBn2 potential   plain          -151.838490829 kJ/mol
                 repartitioned  -151.838490829 kJ/mol
                 delta           0.000e+00
max |force delta|                0.000e+00 kJ/mol/nm
```

Exactly zero, not "small": no term in the potential reads a mass. This also protects the pinned
ParmEd-vs-`AmberPrmtopFile` evidence — if repartitioning perturbed the potential, the 16.05 kJ/mol
`CustomGBForce` comparison would stop being a comparison of construction paths.

## Requirement 6: the guard that was missing

`tests/test_built_system_matches_configuration.py` deserializes a built `system.xml` and checks it
against the configuration that produced it: hydrogen masses against `build.hydrogen_mass`,
constraint count against the record (and that every constrained pair involves a hydrogen),
periodicity against the solvation mode, `CustomGBForce` present for implicit and absent for
explicit, and the provenance stating the same thing the System carries.

Coverage is bounded honestly. The guard inspects build-defining fields, and a profile's production
method does not change the System, so the 21 shipped profiles group into **8 distinct builds**, one
bundle each. A separate test asserts the grouping accounts for every profile, so a new profile
falling outside it fails rather than going untested. 10 passed, 0 skipped, 35 s.

## Requirement 7: `prepare --config` on an implicit document

It raised `AttributeError: 'NoneType' object has no attribute 'water_model'`. It now refuses,
naming the supported route and both commands. **This is a refusal by choice.** Supporting implicit
there would add a second builder able to produce an implicit System — a second place for the
profile and the System to disagree, which is the defect this whole change closes.

## Dynamics gate — 1 ns each, on CUDA

Run on GPU, not CPU: measured on this 22-atom implicit system, CUDA does 20,169 steps/s against
222 on CPU — 91×, or 0.4 min against 37.6 min per nanosecond. An earlier attempt at these runs was
started on CPU by mistake and abandoned.

| run | profile | dt | H mass | result |
|---|---|---|---|---|
| implicit cMD | `implicit-md-peptide-v1` | 2 fs | 1.008 amu | committed generation 1, absolute step 500,000, t = 1000.0 ps |
| implicit cMD | `implicit-md-peptide-hmr-v1` | 4 fs | 3.024 amu | committed generation 1, absolute step 250,000, t = 1000.0 ps |
| implicit REST2 | `implicit-rest2-peptide-v1` | 2 fs | 1.008 amu | 8 committed chunk records, acceptance 0.703 |
| implicit REST2 | `implicit-rest2-peptide-hmr-v1` | 4 fs | 3.024 amu | 8 committed chunk records, acceptance 0.710 |

No NaN or Inf: every serialized State was deserialized and its positions checked finite.

## Consequences

- **Applying HMR is build-defining, so implicit bundle hashes change.** An implicit bundle built
  before this lands does not match a `-hmr-v1` profile and is now refused rather than silently run.
- The eight `-hmr-v1` profiles I added earlier made implicit auto-selection **ambiguous**: the
  implicit branch of `select_profile` matches on the identifier prefix, not on `is_default`, so
  every implicit document without an explicit profile started failing. They now carry
  `performance_variant: true` and are excluded from automatic selection. An explicit flag, not a
  name suffix — selection must not depend on a convention a future profile could fail to follow.
- Everything cyclo-(RGDfV) stays pinned to NAGL charges, matching the packaged manifest, because
  those examples reproduce trajectories generated with them. The drift guard records the exemption
  and its reason.
- `software/md-stack/conda/openfftools860` now holds an **editable** install pointing at this
  checkout, so `md-openmm` follows it. The stale non-editable copy that served pre-`e187c85`
  profiles is gone.

## For krREST2

Its configs must name `implicit-md-peptide-hmr-v1` and `implicit-rest2-peptide-hmr-v1` rather than
the base profiles, since that campaign requires HMR + 4 fs and those are now the opt-in variants.
Its bundles must be built with `MD_system_gen.py --profile <the same id>`, or `MD_input_gen.py`
will refuse the pair. `check_system.py` needs no change and passes on its own.
