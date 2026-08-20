# 2026-08-20 — Branch integration, and the two public generators

Instruction: `claudecode-instructions/20260820_branch-merge-and-public-generators.md`

## Status summary

| item | state |
|---|---|
| Phase 1 branch migration and merge | **complete and verified** |
| `MD_system_gen.py` | implemented, tested |
| `MD_input_gen.py` | implemented, tested |
| staged execution: min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1 | **implemented and executed on CPU** |
| staged execution: REST2 | **delegated** to the runner, and wired: the stage assembles a v2 bundle and calls `launch_rest2`; verified continuing across two invocations |
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
min/  eq_nvt/  eq_npt_1/  eq_npt_2/  cMD_1/  REST2_1/
                          + run_all.sh, run_manifest.json, run.log, inputs/
```

Conventional MD is `cMD_N` and replica exchange `REST2_N`, at the user's direction. The two NPT
stages are also at the user's direction, and they are two stages rather than one repeated because
they are different protocols: `eq_npt_1` holds the solute under the 1 kcal/mol/A^2 positional
restraint while the box relaxes, `eq_npt_2` releases it. Collapsing them would have made the
restraint schedule invisible in the layout. `REST2_1` is one
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

### REST2 execution: delegated, but wired

I first left `REST2_1.sh` failing with a message pointing at `md-openmm rest2`, on the grounds that
REST2 owns a committed-generation record and CLAUDE.md forbids a second restart authority. That
reasoning confused *delegating to* the authority with *becoming* one. The rule forbids the second
authority, not the hand-off.

So `_execute_rest2` now assembles the version-2 bundle the runner already understands and calls
`runner.launch_rest2`. It decides nothing about restarts: it passes the previous run directory if one
exists and lets the runner read its own committed-generation record to find the restart point.

The one substantive choice in the bundle is that `equilibrated_state.xml` is **cMD_1's endpoint**,
not the prepared system's initial state -- that is what makes the hand-off real rather than
decorative. Assembling it surfaced six separate contract requirements the stage projection did not
carry (`system.yaml`, `simbox.json` under its contract name, `experiment.prepare.yaml`, the
`relaxation_ps`/`equilibration_ps` naming split, and a `config_hash` that must be *recomputed* from
the bundle's own manifests rather than copied from the run manifest -- copying it makes
`validate_bundle` report the bundle as edited after preparation).

Verified by running it twice on the alanine smoke project:

```
invocation 1: relax 6 replicas, chunk 1/1, acceptance 0.444 (lifetime,  250 attempts)
invocation 2: no relaxation,    chunk 2/2, acceptance 0.456 (lifetime,  500 attempts)
```

The second invocation continues rather than restarts: no relaxation pass, and the lifetime attempt
count accumulates while the per-invocation count stays at 250.

One pre-existing reporting inconsistency is visible in that record and I did **not** change it:
`rest2_summary.json` reports `n_chunks: 2` (lifetime) beside `total_ns_per_replica: 0.1` (this
invocation's plan). The `lifetime_*` fields next to them are correct; the mixed semantics are in
`rest2.py:573`, which predates this branch.

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

---

## Addendum — three checklist gaps, and a bug the dry-run path could not reach

Written after an audit against the instruction's Phase 8/9 checklists, prompted by the user asking
whether everything was finished. It was not. Three items were missing.

| gap | resolution |
|---|---|
| bundle **relocation** not tested (Phase 8) | two tests: a bundle verifies from a new path, and generates a project from there |
| CPU-only dry generation for **RGDfV** missing (Phase 8) | added, and the real RGDfV system was prepared through the public generator |
| `forcefield.json` contents not documented (Phase 9) | full field table added to `docs/configuration.md` |

### The SMILES route had never actually been executed

Preparing RGDfV through `MD_system_gen.py` for the first time failed:

```
File "src/md_templates/openmm/system.py", line 51, in initial_structure
    params.randomSeed = int(ecfg["seed"])
TypeError: int() argument must be ... not 'NoneType'
```

The package `DEFAULTS` leave `structure.etkdg.seed` as `None` because the canonical pipeline fills
it from the randomness block. `MD_system_gen.py` has no randomness block, so nothing set it. The
PDB route was unaffected because it never calls `initial_structure`.

Every existing SMILES test passed, because they all covered routing, `ligand_build` validation and
refusals -- all of which stop **before** anything is built. **A dry-run path cannot exercise
anything past the point where it stops.**

This is the fourth defect this session found only by executing code that had passed
validation-level tests, after the OPC water model, the unreachable device mapping, and the two
broken example scripts. The pattern is consistent enough to state as a rule: generation and
validation tests are necessary and are not evidence that a path runs.

Fixed by setting the conformer seed explicitly in the front end, overridable through
`randomness.structure_seed` so a specific conformer can be reproduced. Two regression tests added.

### RGDfV prepared and generated through the public path

```
MD_system_gen.py  -i cyclo_rgdfv.smi   -> 79 solute atoms, 3124 particles
                                          net charge 1.67e-15 vs declared 0
                                          openff-2.2.0 / am1bcc, protein_forcefield None
MD_input_gen.py   -> min, eq_nvt, eq_npt, cMD_1, REST2_1
                     REST2_1: 10 replicas, 1000 x 5 ps = 1,250,000 steps per segment
```

The 79 atoms and the near-zero net charge independently reproduce the vetted manifest, from a
different code path than the earlier `md-openmm prepare` bundle.

### Behaviour discovered while writing the tests

**Profiles are route-bound.** Pinning `explicit-rest2-ligand-v1` against a `pdb`-route bundle is
refused: `profile 'explicit-rest2-ligand-v1' is for route 'smiles' but this document declares
'pdb'`. That is correct -- a ligand profile carries small-molecule defaults that do not describe a
peptide. The first draft of the RGDfV protocol test worked around it; it now asserts the refusal as
intended behaviour instead.

### Still true after the addendum

No GPU work was repeated. The REST2 production results (alanine 0.478 over 7,500 attempts, RGDfV
0.292 over 13,500) stand as reported and were not regenerated. The RGDfV work here is preparation
and generation only -- CPU, no dynamics.

## Addendum 2 — the refinements after the first report

Five changes came from the user after the initial report, and one bug came from running what I had
already called done.

**Stage naming and the NPT split.** `cMD` and `REST2` capitalised, and `eq_npt` split into the
restrained and free stages above.

**`--inherit` selects an endpoint, not just an ancestor.** It now takes
`<run_manifest.json>[:<stage>]` and continues from that stage's endpoint, so a project that varies
only the production protocol does not re-run equilibration another project already did. The stages
it skips -- including the inherited one, since what is inherited is its *endpoint* -- are recorded in
`lineage.skipped_stages`. A stage that was never run in the source project is refused with the reason,
rather than inheriting nothing.

**Test sizing.** REST2 verification runs are 2 ns per segment as `200 x 10 ps`, 2 segments plus 1
extension. The post-install alanine unit test is `100 x 1 ps`, in `test/ala/REST2/md_config_smoke.json`;
production profiles default to `200 x 10 ps`. I had earlier inferred `400 x 5 ps` from the worked
examples, which was wrong.

**Generated data lives in `/path/to/MD-test`,** as a real project would. `md-stack` holds
packages, not run output.

### `run_all.sh` did not run — twice

```
.../ambertools26/bin/python: No module named 'md_templates'
```

The launchers called a bare `python`. `activate-md-stack.sh` puts AmberTools' interpreter first on
PATH, and that one has neither `openmm` nor `md_templates`. Every launcher now records the
interpreter it was **generated** with as an overridable default, exports the matching `PYTHONPATH`,
and preflights `import md_templates, openmm` before doing anything -- so the failure mode is an
instruction rather than a traceback.

The second report of the same error was a *stale* project generated before the fix. Generated
projects are snapshots, so fixing the generator does not fix projects already written; I regenerated
the MD-test projects. Worth remembering: the fix and the artifacts are separate things.

This was the fifth defect on this branch found only by running code that passed its validation-level
tests, after the OPC water box, the unreachable device mapping, the broken example scripts and the
never-executed SMILES route.

### `NUMBER_OF_SEGMENTS` was exported and consumed by nothing

Found while reviewing the wiring, not by a test. `run_all.sh` set and exported `NUMBER_OF_SEGMENTS`,
documented it as controlling how many REST2 segments run, and then called `run_stage REST2_1` exactly
once. Asking for four segments ran one, silently.

It now loops: a REST2 stage is invoked once per segment, and re-invoking the same launcher is what
continues the chain, because the runner reads its own committed-generation record to find where the
last segment stopped. The count stays in the shell and out of the stage JSON -- asking for more
sampling must not move the configuration hash.

This is the same class of defect as the four before it: a declared control that no execution path
read. The generated project *looked* correct and the value was even recorded in the manifest.

### Per-stage `--inherit` worked in the library and not at the CLI

`MD_input_gen.py` resolved the whole `--inherit` argument as a path and checked it existed, before
anything parsed the `:<stage>` suffix. So `--inherit ../parent/run_manifest.json:eq_npt_2` failed
with `--inherit manifest not found`, naming a path that included the stage. The library accepted the
form; only the entry point rejected it.

Caught by the test written for the feature, which is the point of writing the test against the CLI
rather than against `resolve_inheritance`. The CLI now splits the suffix -- and tries the whole
string as a path first, so a manifest whose directory legitimately contains a colon still resolves.
Stage-name validation stays in `resolve_inheritance`, the only place that knows the protocol's stage
order.

### One more wrong assertion of my own

The first version of the bundle test asserted `manifest["schema_version"] == 2`. That field is the
manifest *document's* version and is 1; the bundle contract version is `bundle_schema_version`. The
assertion was checking the wrong number, and a passing version of it would have proved nothing. It
now asserts `bundle_schema_version` and that every required role is present.

## Addendum 3 — refusing to overwrite, precisely

Requested: check whether the targeted files are already there and stop unless `--overwrite`.

Both generators already refused a non-empty destination, but that check was **coarse, late and
CLI-only**:

* *coarse* -- any non-empty directory was refused, so a destination holding unrelated files blocked
  a run for no reason, and the error could not say what would have been clobbered;
* *late* -- the library's own guard sat at the publish step, so `prepare_system` would parameterise
  a ligand for half an hour and only then discover the destination was occupied. Its message,
  `appeared during preparation`, was wrong whenever the directory had been there all along;
* *CLI-only* -- the precise part existed nowhere, and anything calling the library directly got the
  late guard. That is the same shape as the `--inherit` bug earlier today.

Now in `md_templates/openmm/destination.py`, one implementation used by both generators:

**`check_destination`** runs before any work and refuses when a file *being written* already exists,
naming those files. A destination holding only unrelated files is not blocked.

**`publish`** moves the staged output into place, and had to change to match: it moves entries in
individually when the destination exists without collisions, because replacing the directory
wholesale would delete the very files `check_destination` had just declined to complain about. A
check that promises something the publish step then violates is worse than no check.

**The `--overwrite` warning** reports what would be destroyed beyond the generator's own outputs.
This needed a second pass: the first version scanned only the top level, which on an executed
project reported `run.log` and nothing else -- `min/` is a directory the generator owns, so
`min/min_final_state.xml` was invisible to it. Those results are exactly what someone needs warning
about. It now walks the tree and groups by directory, because an alphabetical sample of 228 files
says less than the shape does:

```
  --overwrite replaces the WHOLE destination directory, which would also delete 228 file(s) it did
  not write:
    REST2_1/  (207 files)
    cMD_1/  (4 files)
    ...
    run.log
```

The target lists live in `destination.py` rather than being imported from `system_prep`, so the
entry point can check a destination without importing the scientific stack -- that import is
deferred precisely because it is slow. The duplication is deliberate and a test asserts that
`REQUIRED_BUNDLE_FILES` stays a subset, so the two cannot drift apart silently.

**A deliberate loosening, worth stating plainly.** Writing into a destination that holds unrelated
files now succeeds where it previously failed. The old test asserting `not empty` was replaced,
not deleted: one test now asserts the precise refusal, another asserts that unrelated files survive
the write.
