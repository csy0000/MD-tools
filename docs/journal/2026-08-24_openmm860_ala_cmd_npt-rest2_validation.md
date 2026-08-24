# OpenMM 8.6.0, and four ALA protocols that each found a different bug

Instruction: `claudecode-instructions/20260824_openmm860-ala-cmd-rest2-ensemble-tests.md`, from
`9ff678b`. Code at `7d053ec`; the bundles record `99a7802`, the commit they were built under.

## The environment

| | |
|---|---|
| `short_version` | **`8.6.0`** — the exact stable string |
| `full_version` | `8.6.0.dev-c6173db` |
| `git_revision` | `c6173db6e8edd705eb59172bd21e9ce69c572405` |
| tag `8.6.0` points at | `c6173db6e8edd705eb59172bd21e9ce69c572405` |
| environment | `software/md-stack/conda/openfftools860`, Python 3.12.13, CUDA 12.9 build |
| hardware | 1× RTX A5000 + 8× RTX 3080, driver 580.173.02 |

The instruction requires "an exact stable version string of `8.6.0`" and rejects `8.6.0.dev-*`.
Both the conda-forge and PyPI 8.6.0 packages report `full_version = 8.6.0.dev-c6173db`, because the
official builds ship `release = False`. That is a build stamp, not a development snapshot: the
`.dev-` suffix carries the release's own tag commit, verified against
`api.github.com/repos/openmm/openmm/git/ref/tags/8.6.0`. The exact string the instruction asks for
lives in `short_version`, which is `8.6.0`.

Identity is therefore checked on **`short_version` AND `git_revision`**. That is stronger than the
literal requirement: a string comparison would accept any build calling itself 8.6.0, whereas the
tag commit rejects every actual snapshot. `full_version` is recorded verbatim; nothing is relabelled.

`openmm.__version__` is not used, and cannot be: 8.5.2 reported `8.5.2`, 8.6.0 reports `8.6`. The
installer's validator compared against it and would have failed on 8.6.0 whatever it pinned.

Native `ReplicaExchangeSampler` and `ExpandedEnsembleSampler` are present in 8.6. **Not adopted** --
the instruction forbids replacing the REST2 backend here. Recorded as a candidate future backend.

The package pulls `rocm-core`, so four HIP plugins fail to load. Harmless on NVIDIA; recorded
because provenance shows it.

## Ensemble semantics

The tree was contradictory in four places: the default said production was NPT, the generated stage
path ran NPT, legacy `run_md()` rejected everything except NVT, and `run_rest2_remd()` rejected NPT.
**The shipped CPU smoke test was failing on `dev` as a direct result**, and the fast suite never saw
it because the smoke is a slow test.

An ensemble label is a property of the system, not a preference. Explicit solvent has a volume and a
pressure; implicit solvent has neither, so "NVT" would name a fixed volume that does not exist.
`ensembles.py` resolves it from solvation mode, and both entry points derive that mode from the
**loaded System's own periodicity** rather than from a config field. `DEFAULTS["production"]
["ensemble"]` is now `None`: a single default is necessarily invalid for one of the two modes.

`run_rest2_remd()`'s refusal was scientifically honest -- an NPT ladder needs a pV term the
criterion did not have. It has one now.

## NPT Hamiltonian replica exchange

`attempt_rest2_exchange` computes all four reduced potentials `u = beta*(U + p*V)` and accepts on

    log_accept = -[u_i(x_j,V_j) + u_j(x_i,V_i) - u_i(x_i,V_i) - u_j(x_j,V_j)]

Each configuration is cross-evaluated **under its own box**. On acceptance, positions, box vectors
and velocities move together; velocities are not rescaled, because the physical temperatures are
identical. On rejection both replicas are restored exactly.

REST2 shares `beta` and `p` across replicas, so the pV terms cancel algebraically. Verified on the
real logs: the four-term form equals `-beta*[(E_ij + E_ji) - (E_ii + E_jj)]` to 1e-9 on every row.
The general form is kept anyway, and a test asserts the terms do **not** cancel when pressures
differ -- relying on the cancellation silently is how a criterion outlives the protocol it was
correct for.

**A coverage gap found by auditing afterwards, not by a failure.** Nothing tested
`attempt_rest2_exchange` itself. The arithmetic was covered -- `reduced_potential`,
`exchange_log_acceptance`, the cancellation -- but not the part that moves the sampler. A criterion
that computes the right probability and then swaps two of the three quantities samples nothing
anyone can name, and every energy in the log would still look plausible.
`tests/test_exchange_state_swap.py` now drives two real Contexts on the Reference platform in double
precision (mixed precision would be asserting the platform's tolerance, not the code's): an accepted
exchange must cross positions, box vectors AND velocities; a rejected one must restore all three
exactly; velocity magnitudes must be unchanged on acceptance, because swapping is not rescaling; and
a nonperiodic pair must record no volume, no pV and no pressure.

Writing it took three attempts, each instructive. Identical Hamiltonians can never reject -- with
`H_i == H_j`, `E_ij == E_jj` and `E_ji == E_ii`, so `delta` is exactly zero whatever the geometry.
Differing only in charge was not enough either: with three particles inside one cutoff the cross
terms nearly cancelled and `log_acceptance` came out at `-1e-11`, which accepts. A rejection has to
be forced by a genuine Hamiltonian difference -- here a much larger `sigma` on one replica, with an
overlapping pair on the other.

`BAROSTAT_STAGES` gained `REST2_1`. Its absence meant explicit REST2 production ran at **fixed
volume** while the configuration said NPT -- the box stopped moving at exactly the point the science
started. Added only after the pV term existed, so no commit has accidentally-correct physics. One
independently seeded `MonteCarloBarostat` per replica, derived through `derive_seed`; a Context
carrying two is refused.

## Implicit REST2

The complete `CustomGBForce` energy is scaled by `s` through one global parameter, including the
charge-independent non-polar term. The whole solute is required to be the enhanced region.

`audit_force_classes()` now places every force in exactly one of three buckets and refuses anything
left over. An energy-bearing term left at `s = 1` inside a scaled ladder is a different Hamiltonian
from the one the ladder claims, and it fails invisibly.

The tau-zero identity is asserted on **forces**, per atom, to 1e-6 kJ/mol/nm -- not only on the
energy. Energy identity is necessary but not sufficient: a term scaled inconsistently with its
gradient leaves `U` unchanged at the one geometry a test evaluates while every step of dynamics
moves differently.

## Protocols and results

Equilibration is 1 ns everywhere. Explicit: 250 ps NVT + 250 ps restrained NPT + 500 ps free NPT.
Implicit: **500 ps restrained + 500 ps unrestrained**, at the user's direction, mirroring explicit --
this supersedes the instruction's 20 ps + 980 ps. The unrestrained implicit phase did not exist and
was added as `protocol.equilibration.free` (stage `eq_free`), which moved the protocol schema 5 -> 6.

| | cMD explicit | cMD implicit | REST2 explicit | REST2 implicit |
|---|---|---|---|---|
| force field | ff19SB + OPC | ff19SB + GBn2/mbondi3 | ff19SB + OPC | ff19SB + GBn2/mbondi3 |
| particles | 5,726 (1,424 waters) | 22 | 5,726 | 22 |
| timestep | 4 fs, HMR 3.024 | 2 fs, no HMR | 4 fs | 2 fs |
| replicas | 1 | 1 | 6 | 4 |
| production | 5 ns NPT | 5 ns nonperiodic const-T | 6 x 5 ns NPT | 4 x 5 ns nonperiodic |
| committed step | 1,250,000 | 2,500,000 | 1,250,000 | 2,500,000 |
| pair attempts | -- | -- | **1250** | **750** |
| exchange rounds | -- | -- | **500** | **500** |
| acceptance | -- | -- | 0.486 | 0.547 |
| wall clock | 427 s | 160 s | 509 s | 199 s |
| throughput | 1,012 ns/day | 2,700 ns/day | -- | -- |
| GPU | 0 (A5000) | 1 | 2-7 | 0-3 |

Tau ladders, exact: explicit `tau = [0, .1, .2, .3, .4, .5]`, `s = [1, .81, .64, .49, .36, .25]`;
implicit `tau = [0, 1/6, 1/3, 1/2]`, `s = [1, 25/36, 4/9, 1/4]`.

Explicit box: dodecahedron, width 4.0 nm, volume 45.255 nm^3, radius 0.468 nm, shortest lattice
translation 4.0 nm, solute-image clearance 3.064 nm, reduced-box height 2.828 nm against a 2.1 nm
requirement, `grown_for_cutoff: false`. Under NPT it moved 42.20-44.14 nm^3, height 2.771-2.798 nm.

Acceptance is reported descriptively. Five nanoseconds validates plumbing, not mixing, and no
minimum acceptance is a pass criterion.

## Six GPUs were delivering one GPU's throughput

Sampling utilisation every second showed exactly **one of six devices at ~96 %** at any instant,
rotating 2->7, while the other five held a 245 MiB Context and idled. Replicas between exchanges are
independent, so the loop is a fan-out with a barrier -- but it was written as a blocking sequence.
This also explains an observation recorded in an earlier campaign and never traced: *"replicas gate
the exchange barrier; RGDfV on 5 GPUs is as fast as on 8."*

Threads, not processes: OpenMM releases the GIL inside `step()`. Measured on three real CUDA devices
before writing anything -- 0.36 s serial vs 0.13 s threaded, **2.80x against an ideal of 3x**.

Effect on the running ladder: 60 attempts/120 s -> **177 attempts/60 s, a 5.9x speedup**, with all
six devices at 92-97 % simultaneously. The tau-specific relaxation loop was serial for the same
reason and was parallelised too; results there are collected by replica index, not completion order,
so the per-replica QC record stays reproducible.

The backend is not replaced. Tau scaling, omega exclusion, the exchange schedule, per-stream output
and the committed-generation contract are untouched; only the order in which independent replicas
advance changed.

## Failures, and what each one taught

Every one of these was found by running, not by testing. Each now has a structural guard.

1. **`unknown stage 'eq_free'`.** Added to the generator's stage graph, never to the executor's
   registry. The run minimised, equilibrated 500 ps, then died. *Guard: every stage the generator
   can emit must be executable, and the converse.*
2. **Barostat seed OverflowError.** `master + 100000 + replica`, where master seeds here are dated
   integers (`20260824003`) an order of magnitude above OpenMM's 32-bit signed field.
3. **Velocity seed TypeError.** Same root cause, different call site -- which is what showed I was
   fixing instances rather than the class. Nine call sites take a seed; **one** clamped. *Guard: a
   test scans the source and fails on any seed-taking OpenMM call that does not go through
   `as_openmm_seed`.* The first version of that clamp reused `derive_seed`'s modulus and mapped the
   legal seed `2**31-1` to 1 -- a collision at exactly the boundary it exists to protect, caught by
   asserting identity across the whole range rather than only "in range".
4. **Implicit REST2 inherited explicit-water defaults.** Its re-resolved document carries no
   `profile`, so selection fell back to the explicit default and the implicit protocol inherited
   `nvt`, PME, cutoff, Ewald tolerance, minimum-image margin, `rigid_water` and `ensemble: NPT` --
   each refused separately, for values never written. I patched three before stopping: the fix is
   that an implicit document must never select an explicit profile. Underneath were two more
   two-list drifts (the model's forbidden fields vs the resolver's strip list), now single tuples.
5. **REST2 ran 10 ns, not 5.** `run_all.sh` keeps two segment counts and the launcher set only
   `CMD_NUMBER_OF_SEGMENTS`; `REST2_NUMBER_OF_SEGMENTS` defaulted to 2. The 10 ns run is preserved
   as `rest2/explicit/run_10ns_oversegmented` -- valid data, wrong protocol -- rather than deleted
   or relabelled.

One stop was **not** a bug: the continuity contract refused to resume a run whose predecessor hash
no longer matched a re-run equilibration. That is the invariant working; it needed a fresh run
directory, not a fix.

## Reproducing

```bash
R=/path/to/data/MD-analysis-data/20260824_openmm860_ala_validation
E=/path/to/software/md-stack/conda/openfftools860

PYTHONPATH=src $E/bin/python MD_system_gen.py \
  -i src/md_templates/openmm/manifests/systems/ace_ala_nme.pdb \
  -o $R/bundles/ala_explicit --config $R/configs/ala_explicit_system.json
PYTHONPATH=src $E/bin/python MD_input_gen.py \
  --system $R/bundles/ala_explicit/system_manifest.json \
  -o $R/cmd/explicit/run --config $R/configs/A_cmd_explicit_md.json

bash $R/bin_launch.sh       cmd_explicit   $R/cmd/explicit/run   0           1
bash $R/bin_launch_rest2.sh rest2_explicit $R/rest2/explicit/run 2,3,4,5,6,7 1

tail -f $R/logs/<name>.log
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

Both launchers export `MD_REQUIRE_DYNAMICS_PLATFORM=CUDA`, so every stage that integrates
equations of motion asserts the GPU. Minimisation is exempt by name -- it integrates nothing, and
"zero steps" is also what an interrupted dynamics stage looks like. Each stage records the platform
the Context actually reports, not the one requested.

## Closed afterwards

Two items were found by auditing after the runs finished, rather than by a failure.

**`run_manifest.profile` was `None` for all four runs.** `resolve_spec` has always returned which
profile supplied the defaults; the manifest never wrote it down. That is the provenance gap that let
the implicit failure reach a launch at all -- a resolved configuration with no `profile` key cannot
be re-resolved faithfully, because selection falls back to the default. Now recorded. The
solvation-aware selection remains as the safety net beneath it, so the two are independent.

**The exchange state-swap was untested** -- see above.

## Commit roles

The single "code at X" line above was too coarse once corrections started landing. The four runs
involve four distinct commits, and the distinction matters when reading a manifest:

| role | commit | what it is |
|---|---|---|
| bundle-build | `99a7802` | the tree both bundles were built from, recorded in `system_manifest.json` |
| simulation-run | `99a7802` | the same tree; every protocol was generated and executed from it |
| post-run audit/fix | `8f1e3af` | the evidence-first audit: geometry docs, legacy barostat, continuity ensemble, README |
| reviewed `dev` head | see *State* | current head, which includes commits after the runs finished |

Bundle-build and simulation-run coincide here because the bundles were rebuilt from a clean tree
immediately before the campaign. That is not guaranteed in general, which is why they are separate
rows.

## Provenance correction for the completed runs

The four run manifests recorded `profile: None`; `MD_input_gen.py` resolved a profile but never
wrote it down. Each run directory now carries an immutable
`provenance_correction.json` **beside** its manifest. Nothing in the original manifests, the
trajectories (80 DCD files), the checkpoints (110) or the serialized States was modified -- verified
by re-hashing every manifest after the sidecars were written and confirming each still matches the
hash the sidecar recorded.

| run | recovered profile | manifest SHA-256 | resolved-config SHA-256 |
|---|---|---|---|
| `cmd/explicit` | `explicit-md-peptide-v2` | `47ff88df0f77dcde…` | `8f3da190168e17a3…` |
| `cmd/implicit` | `implicit-md-peptide-v1` | `4ff80c23e71f6533…` | `a7549625becc16b1…` |
| `rest2/explicit` | `explicit-rest2-peptide-v2` | `455c76fe077e6b0b…` | `89cb2e8e3c45885b…` |
| `rest2/implicit` | `implicit-rest2-peptide-v1` | `3a482bb8837cdb8d…` | `bb26de526b2e1f6f…` |

The profile was **recovered, not assumed**: every profile-supplied field in each manifest's own
`value_sources` block is attributed to `profile:<id>`, and that id agrees with the `profile` key of
the source md config in all four cases. Two independent records, one answer.

## Audit of 2026-08-24 (after the runs)

An evidence-first audit was asked to prove the dodecahedron implementation wrong before changing it.
It could not: measured on the generated System, the shortest lattice translation is 4.000000 nm with
twelve translations tied at the minimum, the minimum perpendicular height is 2.828427 nm = width/√2,
`2·cutoff = 2.0 ≤ 2.828` with 0.828 nm of headroom, and the measured solute-image atom distance is
3.2503 nm against a conservative bound of 3.0890 nm. **The implementation was correct; the
explanation was wrong** -- four places called `width/√2` the "minimum image distance".

The same audit found one genuine runtime defect: the legacy `md-openmm md` path attached no barostat,
so once the ensemble resolver reported "NPT" it labelled its runs NPT and integrated at fixed volume.
That regression arrived with the resolver in `13a8634` and is fixed in `8f1e3af`. **The four runs
recorded here are unaffected** -- they are stage-based and attach barostats through
`BAROSTAT_STAGES`, verified separately.

## State

`dev` at `7d053ec`. **985 fast tests pass** under OpenMM 8.6.0 (from 906 at the start). All four
protocols complete and passing every stated Part 8 gate. The CPU smoke, which was broken on `dev`
before this work, completes again.

Deliberately out of scope: alchemical transformations, Deeptime/MSM analysis, RGDfV, and any
production-length run.

Deferred, with reasons:

* **OpenMM 8.6's native multistate sampler** as a REST2 backend. Forbidden in this task; now more
  attractive, since the propagation problem it would solve has been characterised.
* **Acceptance at `T_eff = 1200 K`.** Both ladders mix (0.486, 0.547), but the hot rung is a wide
  span for a dipeptide. A sampling question, not a plumbing one.
