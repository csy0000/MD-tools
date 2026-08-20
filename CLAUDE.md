# CLAUDE.md

## Repository purpose

This repository is a project-independent source of reusable OpenMM workflows for explicit-water molecular simulation.

The supported public methods are:

- conventional molecular dynamics;
- REST2 replica exchange, with omega exclusion enabled by default.

The repository must remain usable from unrelated consuming projects. Do not introduce assumptions, paths, terminology, datasets, or scientific conclusions belonging to one research project.

Reliability, reproducibility, and explicit failure are more important than convenience.

## Instruction precedence

For every task:

1. follow the user's current request;
2. read this file;
3. read the task-specific file named by the user under `claudecode-instructions/`;
4. read the relevant dated journals under `docs/journal/`;
5. inspect the implementation and tests before editing.

A dated instruction file defines the scope of one task. This file defines durable repository rules.

If an instruction conflicts with the current implementation, investigate and document the conflict. Do not silently reinterpret the instruction to match the code.

## Before making changes

- Inspect `git status` and preserve unrelated user changes.
- Read `README.md`, `pyproject.toml`, the relevant schemas/configuration code, and the affected tests.
- Trace public CLI behavior through parsing, validation, configuration resolution, runner dispatch, persistence, and output.
- Search for existing helpers before adding parallel implementations.
- Identify whether a proposed setting is:
  - system/build defining;
  - run-continuity defining;
  - execution/output only;
  - extension only.
- State any assumption that changes scientific behavior.

Do not report a feature as implemented merely because a helper, schema field, or unit test exists. Verify that the public execution path uses it.

## Scientific safety rules

- Never silently guess a molecular route, force field, charge model, water model, protonation state, ion definition, ensemble, or enhanced-sampling method.
- Reject unknown configuration keys rather than ignoring them.
- Reject incompatible combinations with the complete dotted field paths and requested values.
- Keep counterions required for neutrality separate from ion pairs added for salt concentration.
- Distinguish topology atom count from OpenMM particle count; virtual sites can make them different.
- Preserve periodic box vectors in structures and restart states.
- Use explicit physical units and validate dimensionality.
- Do not reconstruct integer simulation plans by rounding floating-point totals.
- Scientific configuration declares the length of ONE segment (`duration_per_segment`), never a
  segment count. How many segments to run is an execution choice made by the driver script;
  how many committed is runtime state in the run manifest.
- Never reintroduce a segment/chunk count into the scientific input. It made a longer run look
  like a different calculation, because it moved the configuration hash.
- The REST2 ladder is parameterised by `tau`. `s = (1 - tau)^2` and the solute-environment
  coupling is `sqrt(s) = 1 - tau`, derived by one shared function. `tau` is persisted as the
  source parameter; `s`, `sqrt(s)` and effective temperatures are labelled derived diagnostics
  and are never accepted back as input.
- Durations and reporting intervals must convert to exact integer steps. Reject rather than
  round: a segment silently shortened by one step drifts the exchange schedule out of
  alignment with the committed watermark while still looking healthy.
- Do not change scientific defaults without an explicit task requirement, a versioned profile/schema change, documentation, and tests.
- Do not describe a smoke test as scientific validation, convergence, or proof of production suitability.
- Keep operational verification, statistical validation, and scientific validation clearly separated.

## Configuration architecture

All supported input syntaxes must resolve into one canonical typed configuration before building a System or opening a run directory.

- File formats are front ends, not separate semantics.
- YAML and JSON representations of the same input must produce identical canonical data and hashes.
- Defaults come from named, versioned profiles.
- Resolution precedence must be explicit and documented.
- Persist the fully resolved configuration and the source of every value.
- Derived values must be labelled and must not be accepted as competing inputs.
- Keep system, build, protocol, and execution schema versions independent where possible.
- Schema changes require a migration message or migration tool. Never silently reinterpret an old schema.
- Method-specific models must reject settings belonging only to another method.
- A future Amber-style input reader must adapt into the canonical model and reject unsupported keys. It must never create a second configuration engine.

## Build and bundle portability

A prepared bundle is a self-contained scientific artifact.

- Do not store required absolute paths.
- Copy required molecular inputs into the bundle and use bundle-relative references.
- Record canonical system/build configuration, hashes, package version, OpenMM version, force-field provenance, input hashes, topology/particle metadata, box vectors, composition, and file checksums.
- Validate a bundle after relocation and without the source checkout.
- Validation should not require network access.
- Verify file content, not only filenames or manifest labels.
- Treat a changed build-defining field as requiring a new bundle.
- Treat a changed continuity-defining field as requiring a new run.
- Machine choices may be execution-only, but record them in provenance.
- The installed wheel must contain every schema, profile, manifest, example, and package resource required by its public commands.

## Run-directory and continuation invariants

- A user-supplied run name is exact.
- A timestamp is added only when no run name is supplied.
- A fresh run never reuses an existing non-empty directory.
- A continuation always occurs in the same run directory.
- The atomic committed-generation record is the sole authority for the restart boundary.
- Never derive the physical resume boundary solely from `done.json`, directory counts, glob counts, log length, or status text.
- Treat files beyond the committed boundary as an uncommitted tail.
- Load and validate the committed restart before opening output for append.
- Preserve monotonically increasing physical time, step, chunk, frame, and exchange-attempt indices.
- Append history without truncating previous invocations or duplicating headers.
- Preserve the original run manifest and maintain explicit invocation history.
- Compare continuity-defining fields before writing to a resumed run.
- Allow extension-only changes such as additional chunk count.
- Reject incompatible continuation without an unsafe override.

At every committed chunk boundary, keep:

- an OpenMM binary checkpoint for best same-environment continuation;
- a serialized OpenMM State for portable fallback;
- a versioned atomic commit record identifying the complete generation.

The checkpoint is preferred. State fallback must be announced and recorded because it is physically valid but generally not bitwise identical for stochastic dynamics.

For REST2:

- persist global exchange-attempt indices;
- persist and restore exchange phase, replica/walker assignment, and exchange RNG state;
- report invocation and lifetime statistics separately;
- count attempts only through the committed watermark;
- reject duplicate, non-monotonic, missing, or inconsistent durable history.

## Completion and status semantics

- `completed` means the requested work for that invocation reached its committed boundary.
- Interrupted, failed, partial, smoke, and validation-only runs must not be labelled completed production.
- Status records must be versioned and machine-readable.
- A successful process exit alone is not proof that a simulation or ladder is healthy.
- Errors must fail nonzero with concise, actionable messages.

## Code quality

- Prefer small shared abstractions for configuration, run management, restart persistence, and provenance.
- Keep conventional MD and REST2 algorithms separate and readable.
- Avoid duplicated scientific formulas and compatibility lists.
- Use atomic filesystem primitives in Python; do not use shell operations for scientific persistence.
- Use temporary files in the destination filesystem, flush, fsync, close, then replace atomically.
- Validate persistent schemas on every read.
- Preserve backwards compatibility when safe. When it is unsafe, fail clearly and provide migration guidance.
- Do not hide broad exceptions. Catch them only where status/provenance must be recorded, then report the original type and message.
- Do not add a dependency without documenting why it is needed and ensuring it is present in supported environments.
- Keep public functions and persistent formats documented.

## Testing requirements

Every behavior change needs a test that would fail under the previous behavior.

Run the relevant focused tests during development, then the full supported suite before completion.

The repository should maintain tests for:

- strict configuration validation;
- schema migration;
- canonical serialization and hashing;
- default-profile resolution;
- YAML/JSON equivalence;
- bundle hashing and relocation;
- commands from outside the checkout;
- built-wheel package completeness;
- exact run naming;
- normal continuation;
- incompatible-continuation rejection;
- checkpoint preference and State fallback;
- process interruption at persistence boundaries;
- uncommitted-tail recovery;
- continuous time/steps/chunks/frames;
- lifetime REST2 statistics and exchange RNG/phase restoration;
- default omega exclusion and its explicit disable option;
- conventional MD not constructing REST2 machinery.

Use CPU or mocks for default tests. Mark genuinely expensive scientific/integration tests clearly, but do not replace every end-to-end test with mocks.

For floating-point scientific checks:

- use justified tolerances;
- avoid exact equality across platforms;
- test invariants and independently calculated quantities where possible.

## CI expectations

CI should:

- build and install the wheel;
- run from outside the repository checkout;
- validate packaged resources;
- run the unit suite;
- exercise a tiny CPU prepare/MD/REST2 workflow;
- test normal resume and crash recovery;
- relocate and revalidate a bundle;
- test State fallback;
- report supported Python and OpenMM versions.

A local test result is useful evidence but is not a substitute for configured CI.

## Documentation and journals

Update public documentation whenever behavior, defaults, schemas, CLI syntax, persistent formats, or portability guarantees change.

For substantial work, add a dated journal entry under `docs/journal/` containing:

- request and scope;
- files changed;
- design decisions;
- scientific or compatibility consequences;
- persistent-format changes;
- exact validation commands and results;
- remaining limitations;
- deferred work.

Do not conceal incomplete requirements. Separate completed, partial, unverified, and deferred items.

Keep documentation generic and reusable. Historical task instructions may retain necessary provenance, but active package code, examples, and user-facing documentation must remain project-independent.

## Git discipline

- Do not modify unrelated files.
- Do not discard user changes.
- Inspect the complete diff before finishing.
- Do not commit generated run data, large trajectories, checkpoints, environments, caches, or secrets.
- Do not rewrite history or use destructive Git commands.
- Do not push, open a pull request, or publish a release unless the user explicitly requests it.
- Use focused commits with messages describing the behavioral change.

## Required completion report

When finished, report:

- what changed;
- why the design is scientifically safe;
- configuration or persistent-format compatibility;
- tests and exact results;
- smoke/integration results;
- documentation/journal added;
- known limitations and recommended next step.

Do not claim success for checks that were not actually run.
