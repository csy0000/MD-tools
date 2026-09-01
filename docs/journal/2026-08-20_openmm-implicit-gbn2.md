# 2026-08-20 — Implicit GBn2/mbondi3, and correcting the public generators

Instruction: `claudecode-instructions/20260820_openmm-generator-fixes-and-implicit-gbn2.md`
Reference: `csy0000/partitioned-REST2` at `537d5b6c46e9cbd6fa250c0d60b0649f616f4698`

## Phase 0 — branch integration, done first

| ref | SHA |
|---|---|
| `main` (untouched) | `5471471` |
| `dev` before | `a4b4372` |
| `feature/public-system-and-md-generators` = instruction commit | `7d1ffe3` |
| `dev` after fast-forward | `7d1ffe3` |

The feature branch was a strict descendant of `dev`: 5 ahead, 0 behind, matching the instruction's
"four commits ahead" plus the instruction commit itself. 80 focused generator tests passed **before**
the push. PR #3 and `migration/pr3-pr8-reusable-template-platform` were not touched; no new branch or
pull request was created; `dev` was never merged into `main`.

## The pinned reference, and reproducing its central claim before writing code

`csy0000/partitioned-REST2` is private, but the pinned commit exists locally — the repository was
renamed from `Bidirectional-Annealed-Important-Sampling`. Both reference files were read **at that
commit**, not at a branch head.

Before implementing anything, the claim that makes the reference a reference was reproduced here,
on ACE-ALA-NME built by tleap with `set default PBRadii mbondi3`:

```
force                  ParmEd (reference)   AmberPrmtopFile        diff
CMAPTorsionForce                 -1.6941           -1.6941      0.0000
CustomGBForce                   -63.9813          -47.9292     16.0521
HarmonicAngleForce                1.5146            1.5146     -0.0000
HarmonicBondForce                 0.0862            0.0862     -0.0000
NonbondedForce                  -97.7455          -97.7455     -0.0000
PeriodicTorsionForce             10.0010           10.0010     -0.0000
TOTAL                          -151.8192         -135.7670     16.0521
```

**16.0521 kJ/mol**, on `CustomGBForce` alone, with identical per-particle GB parameters and identical
radii. That matches the 16.05 kJ/mol the reference documents. The construction branch is part of the
Hamiltonian, established by measurement rather than accepted on authority.

Both halves of the reference's `changeRadii` claim also reproduce:

| topology | route | max \|dR\| after `changeRadii` | verdict |
|---|---|---|---|
| ACE-ALA-NME | tleap / ff19SB | **0.0000 Å** | no-op; tleap already wrote mbondi3 |
| cyclo-RGDfV | OpenFF / Sage | **1.7000 Å** | load-bearing; the prmtop carries no GB radii |

Applying it unconditionally is therefore safe for the first and required for the second.

## Phase 1 — making the existing public path truthful

Every defect was confirmed to exist before being fixed.

### 1. Declared trajectories that were never written

`stage.py` contained **no reporter code at all**, while every stage configuration recorded
`full_system_interval_steps`, `selected_atoms_interval_steps` and a `.log`. The trajectories were not
even named in the output block, so the manifest described files that could not exist.

New `reporting.py` owns attachment and the wrapping convention, which is a scientific choice: all-atom
trajectories are wrapped into the box; the selected-atom trajectory is **not**, because wrapping
teleports the solute across the box whenever its centre crosses a face and breaks every analysis that
reads the trajectory as continuous. That convention already existed for the chunked production path
and now has one home rather than two copies.

`solute_atom_indices()` was extracted from the restraint code so "solute" has a single definition.
Two definitions would eventually disagree, and the disagreement would appear as a trajectory whose
atom order does not match the restraint's — silently, since both are plausible lists of integers.

Measured on CPU: `cMD_1` over 25,000 steps wrote 10 all-atom frames at interval 2,500 and 100
selected-atom frames of 22 atoms at interval 250, with one log header and 10 rows.

### 2. Velocities re-drawn at every stage

Every dynamic stage called `setVelocitiesToTemperature`, discarding the equilibration the previous
stage had just paid for and hiding it behind a plausible-looking temperature.

Velocities are now created **once**, on entering the first dynamics stage from a state that carries
none, and inherited thereafter. Making minimisation write no velocities is load-bearing, not
tidiness: velocities drawn before minimisation are constraint-projected for the pre-minimisation
geometry, and minimisation then moves every atom. Measured on the alanine OPC box, inheriting them is
an immediate `Particle coordinate is NaN` on the first step, while a fresh draw from the same
positions is stable. The absence of velocities in the minimisation state is the handshake that
triggers the single initialisation; writing zeros would read as inherited velocities at 0 K.

### 3. A hard-coded seed

`20260820` was compiled into the integrator and into the velocity draw. New `seeds.py`:

```
seed(purpose) = 1 + sha256("md-templates/seed/v1|<master>|<purpose>") mod (2**31 - 2)
```

SHA-256 rather than `hash()`, which PEP 456 randomises per process; nonzero, because OpenMM reads 0
as "choose randomly"; 32-bit, because that is what the integrator, barostat and velocity draw accept.
Purposes are hashed in, so neighbouring master seeds no longer produce overlapping sets — under
`master + offset`, master 1000 stage 1 and master 1001 stage 0 were the same number.

**A documented deviation.** The instruction asks for "one shared derivation algorithm".
`RandomnessSpec.resolve()`'s `master_seed + index` rule is frozen by
`tests/goldens/seed_derivation.json` as a compatibility contract whose stated purpose is that
migrated inputs reproduce *existing trajectories*. Replacing it would silently change the trajectory
of every existing configuration. It was kept; both algorithms are now named and versioned in
`seeds.py`, so which regime produced a seed is always recoverable, and the new one is used for every
purpose that previously had no derivation at all. This conflict is recorded rather than resolved
silently in either direction.

### 4. The REST2 exchange contract

```json
"production": {"duration_per_segment": "5 ns"},
"exchange": {"number_of_exchanges_per_segment": 1000}
```

replaces `n_exchange_per_segment` plus `exchange_interval`. The interval is derived in integer step
space; both divisions must be exact and are refused otherwise, with nearby exchange counts that do
divide, so a reader is not left factorising by hand.

The previous code argued *against* this form, on the grounds that a stated duration makes the
interval a quotient that can fail to divide. That objection is answered rather than ignored: refusal
is the documented alternative, and deriving in steps is stricter than the old picosecond product
because it guarantees a segment is a whole number of exchange rounds.

Protocol/profile schema 4 → 5. The worked protocols keep their 5 ns segments: `1000 x 5 ps` became
`"5 ns" / 1000`, the same schedule, not a shortened one.

### 5. A manifest that recorded the user's document

`system_manifest.json` labelled the raw user JSON `resolved_system_config`. For alanine that
"resolution" carried neither constraints, nor the nonbonded method, nor HMR — the input omits nearly
every value that defines the built System.

It now records `stated_system_config` and `resolved_system_config` separately, with `value_sources`
attributing every build-defining value as user input, package default or route-derived. Execution-only
fields are excluded on purpose: including platform or device would make an identical System built on
a different machine look like a different System.

A build-time check requires `forcefield.json`, `system.yaml` and `system_manifest.json` to agree
about the Hamiltonian. It earned its place immediately, catching three defects in this work — see
below.

### 6. Formats that were advertised and not implemented

`.mol`, `.mol2` and `.sdf` were accepted and then handed to the **PDB reader**. They are now refused
with "not implemented yet", as is `system.type: protein-ligand`. `ligand_build` demanded six fields
and consumed one; its values are now validated and cross-checked against the resolved force field,
in the library rather than the entry point, and under `--dry-run` as well.

## Phases 2–5 — implicit solvent

### The configuration contract

`solvation.mode` discriminates; each mode rejects the other's fields by name. Defaults are `GBn2` and
`mbondi3`, canonicalised case-insensitively. Implicit mode has no water, box, ions, salt, PME,
cutoff, pressure or barostat, and an NPT stage or `pressure` field is refused before generation.

Four named profiles were added rather than hidden branching inside the explicit ones. They do **not**
inherit the explicit HMR/timestep: the reference builds its base System without repartitioning and
the GBn2 validation is against an unrepartitioned System, so 3.024 amu hydrogens would be a different
build that still passes every structural check. Without HMR, 4 fs is not stable, so the implicit
profiles use 2 fs — stated in the profile, not inherited by accident.

### A compatibility break caught by the goldens

Adding `implicit` to `BuildSpec` put `"implicit": null` into every explicit build document, which
moved `system_build_sha256` and `prepared_state_sha256` for **every existing explicit
configuration** — every prepared bundle would have looked stale for a field that says nothing.

The canonical build document now omits whichever solvent treatment is absent. Verified: not one
existing build or prepared-state hash moved, and the four new profiles are purely additive.
`BUILD_SCHEMA_VERSION` was deliberately **not** bumped — the addition is backward compatible, every
existing document resolves identically, and bumping would invalidate every prepared bundle for no
semantic reason.

### Construction, verified exactly

| system | atoms | route | max \|diff\| vs the pinned path |
|---|---|---|---|
| ACE-ALA-NME | 22 | tleap / ff19SB | **0.00e+00 kJ/mol** |
| cyclo-RGDfV | 79 | OpenFF Sage 2.2.0 + AM1-BCC | **0.00e+00 kJ/mol** |

Per-force, not only total. The comparison is on the *same* prmtop and the *same* rst7: comparing
across coordinate sources (a PDB rounded to three decimals versus the rst7) shifts every force by
~0.02 kJ/mol and looks like a construction discrepancy when it is not. `initial_state.xml` is
therefore written from the rst7.

RGDfV component energies for the record (kJ/mol): CustomGB −197.252709, Nonbonded −674.966030,
PeriodicTorsion 113.619582, HarmonicAngle 85.826531, HarmonicBond 12.887917, total −659.884709.

### REST2 on a generalised-Born System

`build_rest2_scaled_system` previously **refused** `CustomGBForce`. It now scales it, extending the
one REST2 implementation rather than adding a second.

The entire GB energy is multiplied by `s` through an injected global parameter, because charge
scaling alone is wrong here: GBn2 has three energy terms and one is a non-polar correction with no
charge dependence, which `charge x sqrt(s)` leaves untouched.

| tau | s | GB(scaled) | s × GB(unscaled) | diff |
|---|---|---|---|---|
| 0.00 | 1.0000 | −63.981294 | −63.981294 | 0.00e+00 |
| 0.10 | 0.8100 | −51.824848 | −51.824848 | −3.84e−13 |
| 0.25 | 0.5625 | −35.989478 | −35.989478 | −4.55e−13 |
| 0.50 | 0.2500 | −15.995323 | −15.995323 | 0.00e+00 |

Exactness at *every* tau is what shows all three terms scale. At `tau = 0` the whole System is
bitwise identical to the unscaled one (0.00e+00 kJ/mol on the total).

A partial enhanced region is refused: a Born radius depends on every other atom's position, so a
partial selection needs a validated treatment of the cross terms and there is none here.

### Execution

Implicit alanine, CPU, through the generated launchers: `min → eq_nvt → cMD_1 → REST2_1`, no barostat
in any stage, velocities initialised once at `eq_nvt` and inherited after, every declared trajectory
and log written at its exact cadence. REST2 ran and **continued**: 2 chunks, 50 lifetime exchange
attempts with 25 per invocation, in the same run directory from the committed-generation record.

Implicit cyclo-RGDfV, CPU, the whole chain through `run_all.sh`: prepare (AM1-BCC for the 79-atom
macrocycle) → `min` → `eq_nvt` → `cMD_1` → REST2 over two segments, exit 0.

```
6 replicas, s = [1.0, 0.81, 0.64, 0.49, 0.36, 0.25]
T_eff       = [300, 370, 469, 612, 833, 1200] K
2 chunks, 50 lifetime exchange attempts, 25 per invocation, acceptance 0.26
```

The lifetime/invocation split is what shows the second segment continued rather than restarted.
Again: 50 attempts is a smoke figure and says nothing about convergence or ladder quality.

Ladders are shorter than their explicit counterparts, at the user's direction: **4 replicas for
alanine, 6 for cyclo-RGDfV**, set in the profile defaults so the split follows the route. Measured for
the 4-replica ladder: `s = [1.0, 0.694, 0.444, 0.25]`, `T_eff = [300, 432, 675, 1200] K`, acceptance
0.333 over 15 attempts. **That is a smoke figure and carries no statistical weight** — it is evidence
the ladder executes, not that it is converged or well-spaced.

## Defects this work exposed in itself

The agreement check and the new tests caught five problems in code written during this task:

1. The implicit path claimed `amber19/protein.ff19SB.xml` while tleap had used
   `leaprc.protein.ff19SB`.
2. Fixing that exposed `system.yaml` reading the **user's document** rather than the resolution — the
   same defect as `resolved_system_config`, in a second file.
3. The implicit manifest claimed **PME, a 1.0 nm cutoff, HMR and rigid water** for a System with none
   of them. The agreement check had compared only force fields, which is why it missed this; it now
   compares nonbonded method and HMR scope too.
4. Writing that comparison surfaced that `forcefield.json` records the OpenMM enum **integer** (`4`)
   where everything else records `"PME"`. Comparison is now by name and the file gains `method_name`
   alongside the integer — added, not replaced, so readers of either generation find what they expect.
5. The ligand implicit route recorded a **water force field** because water was cleared only on the
   peptide branch.

Two further instances of the same *shape* appeared afterwards, in different components. The
package defaults name a force field for protein, water **and** small molecule, because an
explicit-water build may need all three; each route parameterises one of them and was carrying the
others as provenance for chemistry that never ran:

* the ligand route recorded a water force field (water was cleared only on the peptide branch);
* the ligand route recorded `amber19/protein.ff19SB.xml`, which `validate_system` correctly refused
  — a smiles route may not load a protein force field.

After the third occurrence the rule was written once instead of patched again: keep what the route
used, null the rest. The lesson is that a per-branch fix to a defect with a shared cause will be
needed once per branch, and the count of branches is not known in advance.

`system.yaml` also omitted `canonical_smiles_sha256` on the SMILES route. It exists to catch an
edited SMILES in an environment that cannot parse chemistry, which is why it is stored rather than
re-derived on read.

Two more, found by running rather than by reasoning:

- **Resolution was not idempotent.** Re-resolving an already-resolved implicit document under the
  default (explicit) profile produced a build stating *both* solvent treatments, then an NPT stage
  the user never wrote. Profile defaults now never contradict what the document states. This is a
  latent failure well beyond implicit solvent: a resolved document that cannot be re-read is broken
  regardless of solvent.
- **Launchers lost to a relative `PYTHONPATH`.** `${PYTHONPATH:=...}` keeps whatever the caller
  exported, and a relative value stops resolving the moment the launcher changes directory — with the
  correct absolute path sitting unused. They now append instead of defaulting.

## A test of mine that asserted nothing

Omega exclusion was checked by torsion **energy**. It does not change at the prepared geometry,
because a tleap-built amide sits at the minimum of its omega term, which contributes ~0. The test
passed and would have passed with the feature deleted.

It now checks force **constants**: 12 torsions about 2 peptide bonds, `k` summing to 188.28 kJ/mol
unscaled, 47.07 (= 0.25 ×) without exclusion, and 188.28 (unchanged) with it, while every other
torsion scales either way.

This is the third time on this branch that a stochastic or geometry-dependent *outcome* was asserted
as a *contract*. The pattern is worth naming: what can be asserted exactly is what was configured and
what conservation requires; everything else needs aggregation, an honest tolerance, or a
parameter-level check.

## Files added

`src/md_templates/openmm/`: `seeds.py`, `reporting.py`, `solvation_mode.py`, `implicit.py`
`tests/`: `test_implicit_gbn2.py`
`test/ala/implicit/`, `test/ala/implicit/extension/`, `test/rgd/implicit/`,
`test/rgd/implicit/extension/`
Four profiles: `implicit-{md,rest2}-{peptide,ligand}-v1.json`

## Persistent-format changes

- protocol/profile schema 4 → 5 (retired exchange fields, with a migration message)
- `system_manifest.json` gains `stated_system_config`, `value_sources`, `implicit`, `amber_files`
- `forcefield.json` gains `method_name` and, for implicit bundles, an `implicit` block
- `system.yaml` and the bundle manifest write their solvation block per mode
- stage configurations gain `seeds` and trajectory output names
- run manifests gain `randomness` (master seed, derivation, version, per-purpose seeds)
- implicit bundles require `system.prmtop` and `system.rst7` as members
- `rest2_summary.json`: `total_ns_per_replica` is now lifetime, with `invocation_ns_per_replica`
  reported separately

## Limitations, stated plainly

- **Implicit REST2 is full-solute only.** A partial enhanced region is refused, not approximated.
- **No salt screening under implicit solvent.** A Debye–Hückel term is a different Hamiltonian from
  the one the energy validation pins, so it is refused rather than accepted and ignored.
- **MOL/MOL2/SDF and protein-ligand complexes are not implemented** and are refused with that word.
- **No CUDA smoke was run for the implicit path.** Everything reported here is CPU. Nothing in this
  entry is labelled CUDA validation.
- **Nothing here is scientific validation.** The energy comparisons are identity checks against a
  pinned construction. The short runs are engineering evidence that the chain executes and continues.
  No statement about convergence, ensemble quality or ladder spacing is supported by any of it.
- The GB models other than GBn2 are accepted by the schema because OpenMM supports them, but only
  GBn2/mbondi3 is exercised end to end.

## Packaging: a gap in the acceptance criteria, found and closed

`MD_system_gen.py` and `MD_input_gen.py` were in **neither the wheel nor the sdist**, and the
examples were in neither. A public entry point that exists only in a source tree is not installable:
a consuming project could import the library and still have no way to run the generators, which is
most of what this repository offers.

Their implementations moved to `md_templates.openmm.cli_system_gen` and `cli_input_gen`, with the
root scripts reduced to shims — a shim rather than a copy, because two implementations of an entry
point drift and the drift is invisible until they disagree. Console scripts `md-system-gen` and
`md-input-gen` are registered, and `MANIFEST.in` carries the entry points and `test/` into the sdist.

Verified with a real wheel installed into a clean target outside the checkout: `md-system-gen`
prepared an implicit alanine bundle, `md-input-gen` generated the project, and `min`, `eq_nvt` and
`cMD_1` executed — with the launcher recording the **installed** package path.

An earlier relocation check had run the *checkout* scripts, which insert `src` at `sys.path[0]`, so
it exercised the source rather than the wheel. It was re-done through the console scripts. Recording
that here because the first result looked like a pass and was not one.

## Deferred

- A real-CUDA smoke for the implicit path
