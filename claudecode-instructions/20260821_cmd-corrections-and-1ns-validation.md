# Claude Code instruction: finish standalone cMD and validate a 1 ns alanine run

Date: 2026-08-21  
Repository: `csy0000/MD-templates`  
Branch: `dev`  
Expected starting commit: `8c0c85c33ae7097091b468e15da79ca48bc2a35b` or a direct descendant containing this instruction

## Goal

Correct every open finding from the review of the GBn2/public-generator work, make conventional MD
a real standalone public method, and execute a reproducible 1 ns alanine-dipeptide cMD validation
in both supported solvent modes:

1. explicit ff19SB/OPC, NPT production;
2. implicit ff19SB/GBn2/mbondi3, NVT production.

Each 1 ns production run must be two committed 500 ps segments in the same run directory. This is
deliberate: a single uninterrupted 1 ns process would show that integration works but would not test
the segment boundary, checkpoint restart, State fallback contract, trajectory append behavior, or
monotonic runtime indices.

This remains OpenMM-only. Do not touch PR #3, do not resume the PR3-PR8 migration, do not create a
new branch or pull request, and do not merge `dev` into `main`. Work directly on `dev` and push it.

## Read before editing

Read completely:

- `CLAUDE.md`, `README.md`, `pyproject.toml`, and `docs/configuration.md`;
- `claudecode-instructions/20260820_openmm-generator-fixes-and-implicit-gbn2.md`;
- `docs/journal/2026-08-20_branch-integration-and-public-generators.md`;
- `docs/journal/2026-08-20_openmm-implicit-gbn2.md`;
- `MD_system_gen.py`, `MD_input_gen.py`, and their package entry-point implementations;
- `src/md_templates/openmm/input_gen.py`, `stage.py`, `reporting.py`, `segments.py`, `seeds.py`;
- `src/md_templates/openmm/runstate.py`, persistence/restart code, `md.py`, `runner.py`, and
  `rest2.py`;
- `src/md_templates/openmm/spec/` models, profiles, resolution, canonicalization, migration, and
  goldens;
- all public-generator, cMD, implicit, persistence, interruption, packaging, and integration tests.

Inspect the current `dev` diff from `main`, current workflow triggers, remote checks, and available
hardware before changing code. Preserve unrelated user changes.

## Scope guard

- Fix and extend the existing architecture; do not build a second cMD runner or configuration
  engine.
- `MD_system_gen.py` still ends before minimization or dynamics.
- `MD_input_gen.py` generates protocol/run inputs from a prepared bundle and never reparameterizes.
- Scientific JSON defines one segment duration, never a segment count.
- Bash controls how many segments are requested; committed runtime state records how many completed.
- Continuation occurs in the same run directory.
- Do not commit generated trajectories, checkpoints, serialized run States, environments, caches,
  licensed Amber content, or large logs. Commit only concise validation summaries and hashes.
- A 1 ns engineering test is not convergence or scientific validation.

## Phase 1 — standalone conventional MD must be a supported public method

The repository ships `implicit-md-*` profiles, but `MD_input_gen.py` currently refuses
`protocol.production.method = "md"` with “Conventional-MD-only projects are not yet generated.”
Fix the public path, not only the resolver.

### Required stage graphs

For `production.method = "md"`:

```text
explicit: min -> eq_nvt -> eq_npt_1 -> eq_npt_2 -> cMD_1
implicit: min -> eq_nvt -> cMD_1
```

For `production.method = "rest2"`, preserve the currently supported worked chain:

```text
explicit: min -> eq_nvt -> eq_npt_1 -> eq_npt_2 -> cMD_1 -> REST2_1
implicit: min -> eq_nvt -> cMD_1 -> REST2_1
```

Do not force a REST2 stage into an MD-only project. Do not instantiate REST2 machinery, tau ladders,
exchange state, or omega policy for conventional MD.

For MD-only projects, canonical `protocol.production.duration_per_segment` controls each cMD segment.
Remove the dependence on the untyped generator-only `conventional_md.duration` field for that path.
For the REST2 worked chain, retain backward compatibility for its pre-REST2 cMD duration, but resolve,
validate, persist, and source-attribute it explicitly; do not silently let two fields compete for the
same stage.

Add public CLI tests that prepare/generate/execute MD-only projects from both a PDB peptide bundle
and a SMILES ligand bundle where dependencies are available. A profile filename existing in a wheel
is not evidence that the method works.

## Phase 2 — give cMD the same durable segment contract as production deserves

The current `cMD_1` is single-shot: reporters open without append, rerunning overwrites outputs, and
there is no committed-generation boundary. Replace that behavior with a conventional-MD segment
runner using the repository's existing atomic persistence primitives.

At each committed cMD segment boundary retain:

- binary OpenMM checkpoint, preferred for same-environment continuation;
- serialized OpenMM State as explicit portable fallback;
- atomic versioned commit record identifying the complete generation;
- completed segment count, absolute step, physical time, frame counts, and invocation history;
- configuration/continuity hash and the input/predecessor identity;
- reporting watermarks for all-atom, selected-atom, and state-data outputs.

Required behavior:

- The first segment starts from the final equilibration State.
- Later segments load the latest complete committed generation, preferring the checkpoint.
- State fallback is announced and recorded, not hidden.
- Validate the restart and continuity-defining configuration before opening any output for append.
- Append DCD and state-data outputs without duplicate headers, overwritten frames, or duplicated
  boundary frames.
- Preserve monotonic absolute step, time, segment, frame, and invocation indices.
- Treat files beyond the committed watermark as an uncommitted tail; recover according to the
  existing repository policy rather than counting files.
- Reject changed topology, System, integrator, ensemble, temperature, pressure, barostat, timestep,
  restraint state, selection, or reporting atom order before mutation.
- Allow only execution/extension changes that the established compatibility rules classify as safe.
- Reinvoking the cMD segment launcher continues the same run; it never starts a sibling run while
  calling it continuation.

Generated Bash may expose `CMD_NUMBER_OF_SEGMENTS`; it must not enter scientific JSON or the
continuity hash. Keep REST2 repetition separate as `REST2_NUMBER_OF_SEGMENTS` when both stages exist.
For a fresh `run_all.sh`, equilibration runs once and then the driver invokes cMD the requested number
of times. A continuation command must not rerun minimization/equilibration.

Add a same-directory `extension/` example for cMD. Use tiny segments in automated tests, including
process interruption around the atomic commit, binary-checkpoint continuation, forced State fallback,
uncommitted-tail recovery, and incompatible-continuation refusal before append.

## Phase 3 — correct the canonical defaults and schema contract

Apply these defaults consistently to the relevant named profiles, public examples, resolved
documents, documentation, and goldens:

### Shared defaults

- minimization maximum iterations: `1000`, not `0`/unlimited;
- minimization solute positional restraint: `1 kcal mol-1 A-2` where the worked protocol requests it;
- NVT equilibration: `10 ps` at 300 K;
- exact integer step conversion; never round;
- default cMD production segment: `5 ns`;
- all-atom trajectory: every `100 ps`;
- solute/custom selection trajectory: every `10 ps`;
- `LangevinMiddleIntegrator`, friction `1 /ps`;
- deterministic persisted master/derived seeds.

### Explicit solvent

- ff19SB/OPC alanine route retained;
- 10 A real-space cutoff, 0.15 M NaCl, existing box/padding contract retained;
- restrained NVT `10 ps`;
- restrained NPT `10 ps`, 300 K, 1 bar;
- free NPT `10 ps` if the current two-NPT stage graph is retained;
- cMD is NPT;
- current explicit HMR/4 fs decision remains unchanged unless a failing stability test proves it
  unsafe; any change would require user approval.

### Implicit solvent

- only GBn2 + mbondi3 is publicly supported for now;
- NoCutoff, HBonds, no HMR, 2 fs;
- NVT `10 ps`, no NPT or barostat;
- cMD is NVT;
- no water, ions, salt, cutoff, periodic box, pressure, volume, or density.

### REST2 defaults and worked ladders

- `duration_per_segment = "5 ns"`;
- `number_of_exchanges_per_segment = 1000`;
- tau range 0.0 to 0.5, linear;
- peptide/alanine: 6 replicas;
- ligand/RGDfV: 10 replicas;
- exchange interval is derived exactly, never accepted independently.

The current implicit 4/6 replica tests and comments saying this was “by instruction” are incorrect
for this repository instruction history. Replace them with 6/10 unless the user supplies a new
explicit scientific decision.

### Schema version enforcement

`PROTOCOL_SCHEMA_VERSION` is 5, but several shipped profiles still embed protocol schema 4 while
using version-5 exchange fields, and the Pydantic models accept arbitrary integers. Fix this.

- New/current profiles and resolved documents must persist protocol schema 5.
- Validate supported schema versions rather than treating `schema_version` as an unconstrained label.
- A version-4 document with retired `n_exchange_per_segment`/`exchange_interval` fields must receive
  the precise migration message or pass through an explicit tested migration.
- A version-4 label carrying version-5 semantics must not be emitted.
- Preserve readable historical artifacts only where their meaning is unambiguous; never silently
  reinterpret them.

Update every affected profile and committed golden through the normal capture/check workflow. Do not
hand-edit hashes without showing which canonical scientific input moved and why.

## Phase 4 — implicit reporting must not invent a box

The staged runner currently requests volume/density in every `StateDataReporter` and always computes
`box_volume_nm3` from `State.getPeriodicBoxVectors()`. A nonperiodic OpenMM Context can expose default
box vectors even though no force uses periodic boundaries. Recording their determinant creates a
fictional implicit-solvent volume.

Make reporting mode-aware:

- Determine periodicity from the resolved System/solvation mode, not merely from whether a State
  object returns vectors.
- Explicit periodic runs record box vectors, volume, and density.
- Implicit nonperiodic runs do not request volume/density columns and persist those quantities as
  inapplicable/null with a clear reason; they must never record a numerical box volume.
- Do not use `enforcePeriodicBox=True` for an implicit all-atom trajectory.
- Final implicit structures/States must not acquire a meaningful periodic box through reporting.
- Tests must fail if an implicit results file, DCD metadata, manifest, or State is described as
  periodic or contains a claimed thermodynamic volume/density.

## Phase 5 — restrict unvalidated implicit combinations

OpenMM exposing HCT/OBC/GBn models and ParmEd accepting several radii sets does not validate every
model/radius pair as a repository template. The only pinned and end-to-end validated public contract
is:

```json
{"solvation": {"mode": "implicit", "implicit_model": "GBn2", "radii": "mbondi3"}}
```

Restrict the public schema/resolver to that pair for now. Other models/radii must fail before
preparation with an actionable `not implemented/validated` message. Do not advertise them in CLI
help or README as supported. The internal builder may remain extensible, but public acceptance is an
evidence boundary.

## Phase 6 — the two 1 ns cMD validation runs

Add small committed user examples:

```text
test/ala/cMD/explicit/
test/ala/cMD/explicit/extension/
test/ala/cMD/implicit/
test/ala/cMD/implicit/extension/
```

Reuse the vetted neutral ACE-ALA-NME molecular input already in the repository; do not generate a
different chemistry under the same name.

For both validation runs, set:

```json
"production": {
  "method": "md",
  "duration_per_segment": "500 ps"
}
```

and request two segments from Bash, giving exactly 1,000 ps total production. This is a worked-test
override; the named profile default remains 5 ns per segment.

### Explicit run

```text
prepare ff19SB/OPC -> restrained min (max 1000) -> restrained NVT 10 ps
-> restrained NPT 10 ps -> free NPT 10 ps -> two x 500 ps NPT cMD
```

Use CUDA mixed precision, the current explicit HMR/4 fs configuration, 300 K and 1 bar. Preserve
the existing truncated-octahedral/dodecahedral OpenMM geometry decision and at least 12 A padding;
do not change the validated box route merely for this test.

Expected production totals:

- 250,000 steps at 4 fs;
- 10 all-atom frames at 100 ps cadence;
- 100 solute frames at 10 ps cadence;
- final physical time exactly 1,000 ps beyond the production start;
- two committed cMD generations with a continuous boundary at 500 ps.

### Implicit run

```text
prepare ff19SB/GBn2/mbondi3 -> restrained min (max 1000)
-> restrained NVT 10 ps -> two x 500 ps NVT cMD
```

Use CUDA mixed precision, 2 fs, 300 K, no HMR, no barostat and no volume/density.

Expected production totals:

- 500,000 steps at 2 fs;
- 10 all-atom frames at 100 ps cadence;
- 100 solute frames at 10 ps cadence;
- final physical time exactly 1,000 ps beyond the production start;
- two committed cMD generations with a continuous boundary at 500 ps.

### Hardware procedure

Before running, inspect `nvidia-smi`, active compute processes, GPU memory, and the installed OpenMM
platforms. Prefer CUDA; do not silently fall back to OpenCL or CPU. Select an idle GPU explicitly and
record its UUID/model, driver, OpenMM version, platform, device index, and precision. Do not place a
second job on a GPU already running another task. The two runs may execute concurrently only on
different idle GPUs; otherwise run sequentially.

If CUDA is unavailable or occupied, complete tiny CPU engineering tests but report the requested
1 ns CUDA validation as blocked/unrun. Do not relabel CPU as GPU validation.

### Validation of each completed run

- Every stage exits zero and has a machine-readable completed status.
- No NaN/Inf appears in coordinates, energies, temperature, box data, or logs.
- The requested ensemble is present: explicit NPT has a barostat; implicit NVT does not.
- Temperature, energy, and explicit box volume remain finite. Record descriptive ranges; do not add
  flaky assertions that a stochastic observable must change by a fixed amount.
- The second segment starts from the first committed boundary, not from equilibration.
- Checkpoint is preferred and the provenance says so.
- Absolute step/time are monotonic across the boundary.
- DCD frame counts and atom counts match the exact reporting plan; no duplicate first/boundary frame.
- State-data output has one header and monotonic rows.
- Full and selected trajectories append; no first-segment data is overwritten.
- Final checkpoint and serialized State reload successfully in a fresh Context.
- Bundle and run manifests validate after relocation.
- Record concise hashes and validation summaries, not trajectory data.

Also execute a tiny forced-State-fallback continuation in automated tests; do not damage the actual
1 ns validation run merely to demonstrate fallback.

## Phase 7 — tests, packaging, CI, and documentation

Add regression tests that fail against starting commit `8c0c85c`, covering at minimum:

1. MD-only public generation for explicit and implicit bundles;
2. no REST2 objects/fields in an MD-only project;
3. correct explicit and implicit stage graphs;
4. 1000 minimization-iteration and 10 ps equilibration defaults;
5. 5 ns cMD default and 500 ps worked override;
6. 5 ns/1000 REST2 defaults and 6/10 ladders;
7. protocol schema 5 emission and old-schema migration/refusal;
8. only GBn2/mbondi3 accepted publicly;
9. no implicit numerical volume/density or periodic DCD metadata;
10. two cMD segments with checkpoint continuation and monotonic committed state;
11. reporter append without duplicate headers/frames;
12. State fallback and uncommitted-tail recovery;
13. incompatible continuation refused before output append;
14. wheel/sdist public console scripts generating MD-only projects outside checkout;
15. explicit/implicit bundle relocation;
16. previous explicit REST2 and implicit REST2 behavior still passing.

Run focused tests first, then the complete supported non-slow suite, slow cMD/persistence tests,
CPU integration, committed goldens, wheel/sdist builds, and an installed-wheel run outside checkout.
Report exact commands and pass/fail/skip counts.

Inspect GitHub Actions. If workflows exist but do not run on pushes to `dev`, add a minimal safe
`dev` trigger so the integration branch receives the same relevant fast/CPU checks. Do not weaken
tests to make CI pass. Verify the actual remote run; if permissions prevent reading it, record the
run URL and limitation rather than claiming success.

Update README, configuration documentation, CLI help, profile docs, and examples. Add:

`docs/journal/2026-08-21_cmd-corrections-and-1ns-validation.md`

The journal must record starting/final SHAs, every review finding and resolution, schema/default
changes, cMD persistence format, exact test results, hardware selection, commands, wall time,
throughput, frame/step/time checks, restart evidence, output hashes, limitations, and the distinction
between engineering execution and scientific validation.

## Acceptance criteria

Complete only when:

- standalone `production.method = "md"` works through both public generators;
- explicit and implicit cMD have correct method-specific stage graphs;
- cMD segments use committed checkpoint/State generations and append outputs safely;
- profile defaults are 1000-step minimization, 10 ps NVT, 5 ns cMD, and requested reporting cadence;
- REST2 defaults are 5 ns/1000 and worked ladders are 6/10;
- protocol schema 5 is emitted and enforced;
- public implicit solvent accepts only GBn2/mbondi3;
- implicit runs never claim a numerical thermodynamic volume/density or periodic box;
- explicit and implicit alanine each complete two 500 ps CUDA cMD segments, totaling 1 ns, or an
  external CUDA blocker is reported honestly;
- both runs prove monotonic continuation and exact frame/step/time totals;
- focused, full, packaging, relocation, and installed-wheel checks pass with exact recorded results;
- the remote `dev` CI result is reported accurately;
- previous REST2 and explicit-solvent behavior remains passing;
- no generated trajectories/checkpoints/States or unrelated files are committed;
- documentation and the new journal match executed evidence.

## Final delivery

Use focused commits directly on `dev`, inspect each diff, push `dev`, and verify the final remote SHA.
Do not merge to `main`, create a PR, delete branches, or touch PR #3.

The final Claude report must lead with whether both 1 ns runs completed, then give:

- commit list and final `dev` SHA;
- each review finding and its fix;
- cMD public API and persistence design;
- exact explicit/implicit commands and resolved protocols;
- GPU/platform/device evidence;
- exact segment, step, time, frame, and restart results;
- test/CI/package/relocation results;
- compatibility or migration consequences;
- journal link;
- limitations and the next recommended milestone.

If a requested invariant cannot be preserved, stop before publishing a misleading result and report
the smallest reproducible conflict with exact files, configuration, commands, versions, and output.
