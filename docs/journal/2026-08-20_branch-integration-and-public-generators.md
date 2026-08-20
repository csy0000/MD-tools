# 2026-08-20 — Branch integration, and the two public generators

Instruction: `claudecode-instructions/20260820_branch-merge-and-public-generators.md`

## Status summary

| item | state |
|---|---|
| Phase 1 branch migration and merge | **complete and verified** |
| `MD_system_gen.py` | implemented, tested |
| `MD_input_gen.py` | implemented, tested |
| staged execution: min, eq_nvt, eq_npt, cMD_1 | **implemented and executed on CPU** |
| staged execution: REST2 | **delegated** to the runner; not executable from a stage JSON |
| worked examples using the generators | updated |
| alanine + RGDfV production runs | **executed on GPUs**, 10 ns protocol + 5 ns extension each |

## Phase 1 — branch migration

The previous journal recorded this as blocked. It is now done.

The blocker was real but narrower than it looked: a token existed in `~/.config/gh/hosts.yml` and
authenticated fine, but it is a **fine-grained PAT without the Administration permission**. Repo
`admin: true` is the *user's* right, not a scope granted to the token.

```
POST /branches/openmm/rename   -> 403 Resource not accessible by personal access token
PATCH /repos default_branch    -> 403
POST /git/refs                 -> 403
GET  (every read endpoint)     -> 200
```

Pushes had always worked because the remote is SSH, so they never consulted the token at all.

The user performed the rename in the GitHub UI. It was verified to be a **real rename**, not a
delete-and-recreate:

| check | result |
|---|---|
| `default_branch` | `openmm` -> **`main`** |
| `GET /branches/openmm` | **301 redirect** (a recreate would 404) |
| `main` tip | `547147133ba5e42adc8d2d4ce041bce6abd1f09d`, unchanged |
| PR #3 base | retargeted `openmm` -> `main` automatically |

Commit IDs:

```
before   origin/openmm                          547147133ba5e42adc8d2d4ce041bce6abd1f09d
         origin/feature/openmm-peptide-rest2…   31989fd24bd4fb01aa6dbcd64454f9f186d94e5e
after    main                                   547147133ba5   (deliberately NOT advanced)
         dev                                    a4b437204a7c
merge commit                                    84f0181a65b772032b887375d9bfa9f0eb7742ff
```

`dev` was created from `main` and the feature branch merged with `--no-ff`, so the integration point
is a named commit. `main` was not advanced with unfinished development work, per the instruction.
`feature/openmm-peptide-rest2-examples` was **not** deleted.

### Branch cleanup

Four obsolete branches were removed after verifying each was redundant, with one ordering subtlety
worth recording: the safety check compares **SHAs**, and a cherry-pick creates new ones.
`docs/cuda-version-selection` held two journal commits whose content-identical copies existed only
on a then-unpushed local `dev`. Deleting it first would have left that content on one machine. It
was held until `dev` was pushed, then deleted.

```
deleted   agent/install-md-stack, migration/pr1-template-catalog,
          migration/pr2-packaged-catalog-provenance, docs/cuda-version-selection
kept      main, dev, feature/openmm-peptide-rest2-examples,
          migration/pr3-pr8-reusable-template-platform (PR #3 open)
```

## The responsibility split, corrected

The instruction states that the previous journal **reversed** the two generators' responsibilities.
It did. The authoritative split, implemented here:

| | `MD_system_gen.py` | `MD_input_gen.py` |
|---|---|---|
| answers | what the molecule **is** | what is **done** to it |
| produces | an immutable system bundle | a staged project |
| never | runs any dynamics | reparameterises, resolvates, or rebuilds the system |

Both reject the other's configuration keys rather than ignoring them.

## What was implemented

```
new  MD_system_gen.py                              root entry point, molecular preparation
new  MD_input_gen.py                               root entry point, protocol generation
new  src/md_templates/openmm/system_prep.py        bundle construction, stops before dynamics
new  src/md_templates/openmm/input_gen.py          staged-project projection
new  src/md_templates/openmm/stage.py              the thin stage command launchers call
new  tests/test_public_generators.py               50 tests
new  test/{ala,rgd}/REST2/system_config.json, md_config.json
new  test/rgd/REST2/cyclo_rgdfv.smi                the vetted SMILES
edit test/{ala,rgd}/REST2/README.md                setup now begins with the two commands
edit README.md, docs/configuration.md
```

`build_simbox` was reused unchanged -- molecular construction and equilibration were **already**
separate functions, so no refactor was needed to stop before dynamics.

### Stage layout

```
min/  eq_nvt/  eq_npt/  cMD_1/  REST2_1/   + run_all.sh, run_manifest.json, run.log, inputs/
```

Conventional MD is `cMD_N` and replica exchange `REST2_N`, at the user's direction. `REST2_1` is one
**run** containing its segments, not one directory per segment -- per-segment directories would
contradict the same-directory resume contract.

Each stage JSON names the topology and input **State** it consumes and which stage produced it. A
State rather than a PDB, because positions alone would discard velocities and box vectors at every
boundary.

## Execution — and a claim I had to withdraw

I first reported that wiring stage execution was blocked by CLAUDE.md's rule against a second
restart authority. The user challenged that. **The claim was wrong.** The rule is:

```
CLAUDE.md:107   The atomic committed-generation record is the sole authority for the
                restart boundary.
```

That constrains the **restart boundary**. `min`, `eq_nvt`, `eq_npt` and `cMD_1` are single-shot and
have no restart boundary at all, so nothing forbade executing them. Only REST2 falls under the rule.

Execution is now implemented for the four single-shot stages by composing existing primitives
(`_make_simulation`, `_apply_coords`, `_add_positional_restraints`, barostat control) -- no physics
is reimplemented. Executed on CPU:

```
min      U -32662.3 -> -33345.4 kJ/mol   V 19.70 nm^3   22 restrained atoms
eq_nvt   U -33346.3 -> -32496.7 kJ/mol   V 19.70 nm^3   22 restrained atoms
eq_npt   U -32496.7 -> -32329.7 kJ/mol   V 19.78 nm^3   22 restrained atoms
cMD_1    U -32355.4 -> -32329.2 kJ/mol   V 19.78 nm^3    0 restrained atoms
```

22 restrained atoms is exactly the alanine solute, on the three equilibration stages and **zero** on
production -- the instruction's 1 kcal/mol/A^2 restrained protocol, which the GPU production runs
below did NOT have. Volume is constant through NVT and changes under NPT. `eq_nvt` starts where
`min` ended, so the hand-off carries positions and velocities.

**REST2 execution remains delegated** and this one genuinely is the rule: it has segments, a
committed-generation record and a durable exchange history, all owned by the runner. `REST2_1.sh`
fails with an explicit message pointing at `md-openmm rest2`.

### A wrong assumption in my own test, found by running it

I asserted that minimisation always lowers the reported potential energy. It does not:

```
  50 iterations:  -36074 -> -33719  (UP)     max|F| = 5535
 500 iterations:  -36074 -> -36945  (down)   max|F| = 3201
5000 iterations:  -36074 -> -37674  (down)   max|F| = 3148
```

OpenMM's `minimizeEnergy` optimises a surrogate objective in which constraints are replaced by stiff
harmonic terms, then restores the constraints. With `HBonds` constrained and few iterations the
reported energy can rise while the minimisation is working correctly. The test now asserts a
finiteness invariant that always holds, plus a trend check at a sane iteration count, with these
measurements in the docstring.

## Production runs — executed on GPUs

Both worked protocols ran to completion: 2 x 5 ns segments (the 10 ns protocol) plus a third 5 ns
segment demonstrating the extension contract.

| | replicas | GPUs | wall | per segment |
|---|---|---|---|---|
| alanine | 6 | 1,2,3 | 45 min | 15 min |
| cyclo-RGDfV | 10 | 4,5,6,7,8 | 2 h 29 min | ~50 min |

Run concurrently on disjoint device sets; 2 h 30 min wall for both.

### Exchange acceptance — the first statistically meaningful figures

| | attempts | overall | per-pair range |
|---|---|---|---|
| alanine (6 rungs) | 7,500 | **0.478** | 0.459 – 0.493 |
| cyclo-RGDfV (10 rungs) | 13,500 | **0.292** | 0.265 – 0.315 |

1,500 attempts per neighbour pair in both. Both ladders are **uniform** -- no bottleneck rung, so a
walker can traverse the full range without a slow step.

RGDfV's 0.292 sits squarely in the conventional 20-40 % band; this is the ladder the project already
validated, and the measurement now supports it independently. It shows a gentle monotonic decline
toward the hot end (~0.31 cold, ~0.27 hot) whose intervals overlap, but the direction is physically
sensible: potential-energy fluctuations grow as the enhanced region softens.

Alanine's 0.478 is **above** the band, uniformly -- its six rungs are closer together than necessary.
That is a ladder-design observation, not a fault.

Earlier figures of 0.400 and 0.389 came from benchmark runs with 10 and 18 attempts and carried no
information; with 2 attempts per pair the only achievable values were 0.00, 0.50 and 1.00.

### Continuity across segments

```
alanine   attempts 0..7,499 strictly increasing, 0 duplicates
          steps 1,250..3,750,000 monotonic; time 5 ps..15,000 ps continuous
          one CSV header; committed generations gen_0001, gen_0002
boundary @2500:  5000.0 ps ->  5005.0 ps   step 1250000 -> 1251250   phase 1 -> 0
boundary @5000: 10000.0 ps -> 10005.0 ps   step 2500000 -> 2501250   phase 1 -> 0
```

Each boundary advances by exactly one 5 ps exchange interval with the phase alternation preserved.

### A benchmark correction

Earlier reported figures of 398 and 143 ns/day per replica **understate** throughput and should not
be quoted. They timed a 1 ns segment including per-invocation setup, which dominates at that length.
Production shows alanine at ~480 ns/day per replica on **three** GPUs -- faster than the benchmark
claimed on six, which is impossible and shows the benchmark was wrong. Benchmarks must use
production segment lengths.

## Honest classification

**Implemented and tested.** Both generators; the staged layout; stage validation; execution of min,
eq_nvt, eq_npt and cMD_1; checksum verification; transactional generation; destination refusal;
`--inherit` lineage; the responsibility boundary in both directions.

**Generated but not executed.** `REST2_1` stages are generated and validated but cannot be executed
from a stage configuration. The generated RGDfV project has not been run through the staged path.

**Executed smoke tests.** The four single-shot stages, on CPU, in a tiny project, as part of the
test suite.

**Existing production data reused, not regenerated.** The GPU runs above were produced by the
`md-openmm` CLI, not by the staged generators. They are **not** relabelled as satisfying the new
restrained protocol: they were prepared with **unrestrained** equilibration, and that remains true.
The staged generator now produces the restrained protocol the instruction specifies, but no GPU run
has used it.

**Deferred.** Amber and GROMACS adapters -- the manifest's `adapter_status` says plainly they are
not implemented. REST2 execution from a stage configuration. Growing the canonical model a
multi-stage production block so `conventional_md.duration` and `minimization.restraint` stop being
generator-level keys.

**Remaining branch management.** None required. `feature/public-system-and-md-generators` is pushed
and left for review; it is **not** merged into `dev`, per the instruction.
