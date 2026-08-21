# Claude Code instruction: close every cMD review finding and make OPC the explicit default

Date: 2026-08-21  
Repository: `csy0000/MD-templates`  
Branch: `dev`  
Expected starting commit: `76848541915338579b4346712387c67ee8194c71`, or a direct descendant containing this instruction

## Outcome and non-negotiable completion rule

Close every finding from the review of the standalone-cMD and 1 ns validation work. The active
explicit-solvent default must become ff19SB/OPC, not TIP3P-FB.

This is a strict correction pass. Maintain a seven-item acceptance matrix while working. You may
write `PASS` in the final report only when all seven findings below are implemented, tested, and
verified. Do not call a requirement "deferred", "follow-up", "covered elsewhere", or "good enough"
and still return `PASS`.

If a genuinely external blocker remains after reasonable attempts, return `BLOCKED`, name the
exact unmet gate and preserve all successful evidence. Never relabel an unrun or inaccessible check
as passed.

Work directly on `dev`. Do not create another branch or pull request, do not merge `dev` into
`main`, and do not revive PR #3 or the PR3-PR8 migration. Use focused commits and push `dev`.

This remains OpenMM-only. Preserve the user's later decisions:

- implicit equilibration is `min -> eq -> cMD_1`, where `eq` is 20 ps of restrained
  constant-temperature dynamics with no NVT/NPT label and no barostat;
- implicit REST2 replica counts remain 4 for alanine/peptide and 6 for RGD/ligand;
- explicit REST2 replica counts remain 6 for alanine/peptide and 10 for RGD/ligand;
- explicit production uses HMR/4 fs; implicit uses no HMR/2 fs;
- public implicit solvent remains GBn2/mbondi3 only.

## Read and establish the baseline before editing

Read completely:

- `CLAUDE.md`, `README.md`, `docs/configuration.md`, and `pyproject.toml`;
- `claudecode-instructions/20260821_cmd-corrections-and-1ns-validation.md`;
- `docs/journal/2026-08-21_cmd-corrections-and-1ns-validation.md`;
- `src/md_templates/openmm/{input_gen,stage,reporting,cmd_segments,runstate,equilibration}.py`;
- system/bundle preparation, hashing, persistence, profile resolution, and migration code;
- all explicit/implicit MD and REST2 profiles and worked examples;
- `tests/test_cmd_segments.py`, `tests/test_crash_recovery.py`,
  `tests/test_public_generators.py`, `tests/test_implicit_gbn2.py`, and CI scripts;
- OpenMM 8.5.2's `DCDReporter` and `DCDFile` implementation. In particular, account for
  periodic unit-cell records and the fact that an atom-subset DCD can still carry a periodic
  topology even when coordinates are not wrapped.

Before changing code:

1. fetch `dev`, verify its exact remote SHA, branch, and working-tree status;
2. inspect the full diff from `main` and preserve unrelated user changes;
3. run the existing focused cMD tests and record the baseline result;
4. inspect the current named-profile defaults and every place that supplies a system-generation
   default;
5. inspect GPU occupancy and OpenMM platforms, but do not start the final GPU runs yet.

Do not commit trajectories, checkpoints, serialized States, environments, caches, or large logs.

---

## Finding 1 — implicit positional restraints must be genuinely nonperiodic

### Defect

`_add_positional_restraints()` currently uses `periodicdistance(...)` for every solvent mode.
That is correct for an explicit periodic box but wrong for an implicit nonperiodic System. It makes
the restraint depend on periodic images and can make the restraint Force itself report periodic
boundary use even though the implicit protocol claims to have no box.

### Required correction

Make the restraint expression mode-aware from the actual unmodified System:

- explicit periodic System: retain minimum-image `periodicdistance`;
- implicit nonperiodic System: use ordinary Cartesian squared displacement,
  `(x-x0)^2 + (y-y0)^2 + (z-z0)^2`, with no periodic function;
- preserve the current AMBER-style restraint-weight convention and units;
- decide periodicity before adding the restraint Force; do not let the newly added Force alter the
  test used to choose its own expression;
- record the restraint distance convention in the stage configuration/results/provenance;
- leave implicit minimization and the 20 ps restrained `eq` stage nonperiodic;
- ensure cMD remains unrestrained unless explicitly requested by a future supported model.

### Blocking tests

Add tests that fail on `7684854` and prove:

1. an implicit restraint Force does not use periodic boundary conditions;
2. the implicit System remains nonperiodic after the restraint is added;
3. changing arbitrary Context box vectors does not change the implicit restraint energy;
4. an explicit restraint still uses minimum-image distance and behaves correctly across a box face;
5. the implicit `min -> eq -> cMD_1` graph still executes and reports no pressure, volume,
   density, wrapping, or periodic trajectory metadata.

Do not satisfy this by merely reporting `periodic=false` from a Boolean captured before adding the
Force. Test the actual Force/System used by the Context.

---

## Finding 2 — cMD continuation must bind exact System, topology, force field, predecessor, and selection

### Defect

The cMD continuity record currently carries particle/constraint counts, but those do not identify a
System. It is constructed with `selected_atoms_fingerprint=None`. A changed System or topology
with the same counts, or a changed ordered selection with the same length, can therefore be
continued into an existing trajectory.

### Required correction

Resolve and validate the reporting selection before calling `prepare_continuation()`. Build one
versioned cMD continuity contract that includes, at minimum:

- SHA-256 of the exact `system.xml` bytes;
- SHA-256 of the exact topology file bytes;
- prepared-bundle/system-manifest identity and configuration hash;
- force-field provenance/hash, including protein, ligand, water, charge method, implicit model and
  radii where applicable;
- particle and constraint counts and periodicity;
- ordered selected-atom fingerprint;
- atom identity behind that order: index, chain/residue identifiers, residue name, atom name, and
  element, so equal-length selections cannot alias;
- first-segment predecessor State identity/hash, recorded immutably as provenance;
- integrator, ensemble, temperature, pressure/barostat, timestep, restraint state/convention,
  segment length, reporting intervals and output atom order;
- a deterministic continuity hash over the canonical contract.

The first-segment predecessor is provenance for how the cMD chain began. Once a committed
generation exists, continuation restores that generation and must not silently switch back to a
newly overwritten equilibration endpoint.

Validate the contract and committed restart before opening or modifying any output. A change must
produce an actionable field-by-field refusal. Bump/version the persistent cMD schema. Because the
current cMD layout has not passed its acceptance gate, it is acceptable to refuse continuation of
the old layout with precise regeneration guidance; do not silently reinterpret it.

### Blocking tests

Prove refusal before output mutation for independent changes to:

- one byte of `system.xml`;
- topology atom order while preserving atom/constraint counts;
- force-field or water-model provenance;
- selected atom order with the same selected atom count;
- predecessor identity in a fresh run;
- integrator temperature/timestep, ensemble/barostat/pressure, restraint convention, and reporting
  cadence.

Also prove that only the execution segment count remains extension-only and does not change the
continuity hash.

---

## Finding 3 — DCD tail recovery must understand real OpenMM DCD records

### Defect

The current truncation code assumes a frame contains only X/Y/Z float records. OpenMM periodic DCD
frames also contain a 48-byte unit-cell payload plus Fortran record markers (56 bytes total).
OpenMM's subset reporter can retain the topology's periodic vectors even when coordinates are
unwrapped. Hard-coded coordinate-only frame arithmetic can therefore truncate explicit all-atom or
selected-atom DCDs at the wrong byte offset and corrupt them.

### Required correction

Replace hard-coded frame-size truncation with a defensible DCD record scanner/parser or an existing
already-supported library that preserves this repository's packaging/offline requirements.

The implementation must:

- detect and validate DCD endianness/header record structure;
- read the declared atom count and periodic/unit-cell flag;
- walk actual Fortran records to locate every complete frame boundary;
- support periodic triclinic/dodecahedral frames and nonperiodic frames;
- support both full-system and atom-subset DCDs;
- reject incomplete or malformed records;
- update both frame-count and last-step header fields consistently;
- preserve every committed frame byte-for-byte;
- remove only frames beyond the committed watermark;
- write recovery through a same-filesystem temporary file, flush/fsync, and atomically replace, or
  provide an equally crash-safe design;
- record what was recovered without deleting the only diagnostic copy of an uncommitted tail when
  the existing recovery policy calls for quarantine.

Do not add a large dependency solely to avoid understanding the format. If using an existing
dependency, document why it is already part of supported environments and verify it is present in
the installed-wheel test.

### Blocking tests

Construct real OpenMM DCDs and prove recovery followed by append for:

1. explicit periodic all-atom DCD;
2. explicit periodic atom-subset DCD with unwrapped coordinates;
3. implicit nonperiodic all-atom DCD;
4. implicit nonperiodic subset DCD;
5. one and multiple uncommitted tail frames;
6. a deliberately incomplete final record.

After recovery and append, open the DCD with an independent reader and verify exact frame count,
atom count, box metadata, monotonic time/step interpretation, committed-frame coordinates, and no
duplicate boundary frame.

---

## Finding 4 — outputs shorter than the committed watermark are corruption, not a tail

### Defect

The current recovery helper treats an absent or shorter DCD/log as "absent" or "kept", even when
the atomic commit record says more frames/rows were durably committed. Continuing from that state
creates a trajectory with missing history while the run still claims completeness.

### Required correction

Implement a two-phase output-integrity check:

1. inspect every committed stream without mutation;
2. if any stream is absent, shorter than its watermark, malformed, has the wrong atom count/order,
   or has an invalid header, refuse the continuation before changing any output;
3. only after every stream passes the committed-prefix check may uncommitted tails be recovered;
4. then open reporters for append.

For a watermark of zero, an absent stream is valid. For a positive watermark, absence or shortage
is corruption. Apply the same rule to all-atom DCD, selected-atom DCD, and the state-data log.
Validate that the log has exactly one valid header and at least the committed number of monotonic
rows.

### Blocking tests

Cover absent, truncated, malformed and wrong-atom-count files independently for every stream.
Assert byte-for-byte that no other stream was mutated when one stream causes refusal. Also test a
valid longer-than-watermark set where all streams recover together and append successfully.

---

## Finding 5 — invocation history and the committed generation must have one atomic authority

### Defect

The cMD stage currently appends to `run_state.json` invocation history before atomically committing
the restart generation. A crash between those writes leaves a phantom invocation; retrying can
create duplicate segment/invocation indices even though `committed.json` still points to the older
physical state.

### Required correction

Redesign cMD invocation accounting so the atomic committed generation remains the sole authority:

- do not publish an invocation as completed before its generation commit;
- include the completed invocation index/record, segment number, start/end absolute step and time,
  restart source, watermarks, and continuity hash in the atomic commit;
- retain enough committed lifetime history to reconstruct monotonic invocation accounting after a
  crash; a cache in `run_state.json` may be reconciled from the commit but must not overrule it;
- ensure restart generation, reporters, and committed invocation refer to the same boundary;
- close/flush reporters successfully before saving restart members; do not swallow close failures
  and then commit;
- after loading either checkpoint or State, verify its step and time against the committed record
  before opening output;
- checkpoint remains preferred; State fallback must restore step/time, be announced and be recorded
  as non-bitwise stochastic continuation;
- persist and validate the new schema version and reject unsafe old state precisely.

### Blocking tests

Use fault injection and a real subprocess where practical to interrupt:

1. before reporters close;
2. after reporters close but before restart save;
3. after one restart member;
4. after both restart members but before commit;
5. immediately after commit but before any non-authoritative cache/status update.

Every recovery must resume from the last committed physics, with no phantom/duplicate invocation,
segment, step, time, or frame index.

Add the previously missing cMD-specific forced-State-fallback test: corrupt the committed cMD
checkpoint while keeping its State valid, continue in the same directory, verify the fallback
announcement/provenance, exact starting step/time, monotonic output append, and the next committed
generation. Existing REST2 fallback tests do not substitute for this cMD test.

---

## Finding 6 — complete every validation gate and verify remote CI

The previous journal explicitly deferred cMD State fallback and process interruption, did not report
a completed full slow-suite result, and could not verify the remote `dev` Actions result. Those are
open requirements, not documentation details.

### Required validation after Findings 1–5 and 7 are implemented

Run and record exact commands, versions, counts, durations and outcomes for:

1. focused unit/regression tests for all seven findings;
2. all cMD, persistence, crash-recovery, implicit, profile/default and public-generator tests;
3. the complete non-slow suite;
4. the complete suite including slow tests — give pass/fail/skip/deselect counts, not "recorded
   below";
5. `scripts/ci/fast_checks.sh`;
6. `scripts/ci/integration_cpu.sh`, extended so the installed-wheel staged cMD path exercises
   normal resume, forced cMD State fallback, and crash/tail recovery;
7. wheel and sdist builds with package-resource inspection;
8. installed-wheel commands from outside the checkout with no source-tree `PYTHONPATH`;
9. bundle relocation/offline validation;
10. existing explicit and implicit REST2 regression/integration tests and committed goldens;
11. a source scan showing no generated trajectory/checkpoint/State or large validation artifact is
    committed.

### Final CUDA validation

Rerun both final alanine validations from freshly generated bundles/projects under the corrected
code:

- explicit ff19SB/OPC, 0.15 M NaCl, dodecahedral/truncated-octahedral geometry contract, at least
  12 Å padding, 10 Å cutoff, HMR/4 fs, NPT, CUDA mixed precision;
- implicit ff19SB/GBn2/mbondi3, nonperiodic mode-aware restraints, no HMR, 2 fs, no barostat,
  CUDA mixed precision.

Each production is two committed 500 ps segments in one cMD run directory:

- explicit: 250,000 production steps, 10 all-atom frames, 100 selected frames;
- implicit: 500,000 production steps, 10 all-atom frames, 100 selected frames;
- both: exactly 1000 ps production time, two committed generations, one log header, monotonic
  steps/time/frames/invocations, no NaN/Inf, checkpoint preferred on segment 2.

Inspect `nvidia-smi`, active compute processes and memory first. Set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, select idle GPUs explicitly, record model/UUID/index/driver/OpenMM
8.5.2/precision, and never place two jobs on one occupied GPU. Recompute concise output hashes under
the final code. Do not reuse the previous implicit hashes: Finding 1 changes its equilibration
Hamiltonian.

### Remote CI is a blocking gate

Ensure the fast and integration CPU workflows trigger on `dev`. Push the final code, locate the
workflow runs for the final remote SHA, and wait for terminal conclusions. If either fails, inspect
logs, fix the cause, push, and repeat. Record final SHA, workflow names, run URLs/IDs and green
conclusions in the journal.

If authentication or permissions truly prevent reading Actions after reasonable attempts, report
`BLOCKED: remote CI unverified`; do not return `PASS`.

---

## Finding 7 — OPC is the active explicit-solvent default

### Decision

The user explicitly confirms: **OPC is the default explicit water model.** TIP3P-FB must not remain
the active/default explicit profile.

### Required correction

Change every active default source consistently:

- protein route: ff19SB + OPC;
- ligand route: Sage/OpenFF 2.2 + OPC;
- OpenMM force-field resource: `amber19/opc.xml`;
- modeller water model: `opc`;
- 0.15 M NaCl, 10 Å real-space cutoff, current 12 Å padding and dodecahedral geometry contract;
- explicit conventional MD and explicit REST2;
- system-generator defaults, named profiles, default-profile aliases, examples, configuration docs,
  README tables, JSON schemas/examples, adapters, generated resolved documents, and goldens.

Search the repository for `tip3p`, `tip3pfb`, `TIP3P`, and every default-water declaration.
Do not rewrite historical journals that truthfully describe an old run or fixtures that deliberately
test a nondefault water model. Label such occurrences historical or explicit opt-ins where needed.

This is a scientific default change and must be versioned rather than silently changing the meaning
of a named profile:

- add `explicit-md-peptide-v2`, `explicit-md-ligand-v2`,
  `explicit-rest2-peptide-v2`, and `explicit-rest2-ligand-v2` as the active OPC defaults;
- keep the existing v1 IDs name-resolvable only for compatibility, mark them nondefault, and
  document that they represent the older TIP3P-FB choice;
- keep profile-format/schema version separate from scientific profile ID/version;
- ensure the `default` alias selects exactly one v2 profile for each explicit route/method;
- update active worked examples to name v2 where they name a profile;
- confirm resolved force-field provenance records OPC and not merely a profile label.

### Blocking tests

Prove:

1. every explicit `default` alias resolves to the correct v2 OPC profile;
2. all four v2 profiles resolve to `amber19/opc.xml` and modeller model `opc`;
3. system generation without an explicit water override uses OPC;
4. resolved configuration/provenance and bundle hashes identify OPC;
5. active explicit alanine cMD and REST2 examples use OPC;
6. old v1 profiles are never selected by `default` but remain explicitly name-resolvable;
7. implicit profiles contain no water model.

---

## Documentation corrections and journal

Correct stale user-facing text while touching these paths:

- the implicit graph is `min -> eq -> cMD_1 (-> REST2_1)`, not
  `min -> cMD_1 (-> REST2_1)`;
- implicit `eq` is real restrained dynamics, not something minimization already accomplished;
- `cMD_1` execution status must not describe every mode as NPT;
- distinguish nonperiodic restraint distance from explicit minimum-image restraint distance;
- document the new cMD persistence schema, continuity identity and corruption refusal;
- document OPC as the active explicit default and v1 TIP3P-FB profiles as named compatibility
  profiles only;
- document recovery/fallback behavior in the cMD extension examples.

Add:

`docs/journal/2026-08-21_cmd-review-fixes-and-opc-default.md`

The new journal must include:

- starting and final SHAs;
- a seven-row finding → fix → test → evidence matrix;
- persistent/profile compatibility and migration/refusal behavior;
- exact test commands and complete results;
- both fresh 1 ns CUDA commands/results/hashes;
- remote CI run URLs and conclusions;
- any limitations that are outside these seven gates.

Cross-link the earlier 2026-08-21 journal as superseded for final validation hashes where the
implicit Hamiltonian changed. Preserve it as historical evidence; do not rewrite history to make the
old hash look current.

## Final acceptance checklist

Before returning, inspect the final diff and answer each item explicitly:

- [ ] 1. Implicit restraint uses nonperiodic Cartesian distance; explicit remains minimum-image.
- [ ] 2. cMD continuity binds exact System/topology/force field/predecessor/ordered selection.
- [ ] 3. Periodic and nonperiodic DCD tail recovery is record-aware and independently readable.
- [ ] 4. Missing/short/malformed committed outputs refuse before any mutation.
- [ ] 5. Invocation history, restart and watermarks share one atomic committed authority; cMD
      checkpoint and forced-State paths pass interruption/recovery tests.
- [ ] 6. Focused, complete slow/non-slow, CI scripts, package, installed-wheel, relocation, REST2,
      both fresh 1 ns CUDA validations, and final remote CI all pass.
- [ ] 7. Active explicit defaults are versioned v2 OPC profiles across system generation,
      conventional MD and REST2; v1 TIP3P-FB is opt-in compatibility only.
- [ ] Documentation, journal, goldens and persistent-format guidance match the code and evidence.
- [ ] No unrelated work, generated simulation data, secrets, environments or large logs are
      committed.
- [ ] Final remote `dev` SHA is verified after push.

The final report must begin with exactly one of:

- `PASS — all seven cMD/OPC acceptance gates are complete.`
- `BLOCKED — PASS is not claimed.`

Use `PASS` only if every checkbox is satisfied with executed evidence. A failing, skipped,
deferred, inaccessible or merely inferred gate requires `BLOCKED`.
