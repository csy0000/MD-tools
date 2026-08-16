# Claude Code instruction: make explicit-water simulations portable across projects

Work on the `openmm` branch of `csy0000/MD-templates`.

The repository is intended to be a reliable, project-independent source for explicit-water OpenMM simulations. It must not encode assumptions from any consuming research project. A user should be able to install the package, prepare a self-contained bundle, move that bundle to another directory or compatible machine, and launch or continue conventional MD or REST2 without the source checkout.

This task has three ordered milestones:

1. close the remaining run-continuity defects;
2. introduce a canonical, format-neutral simulation configuration with versioned defaults;
3. formalize and test the transferable bundle contract.

Complete them in that order. Do not build a new input parser on top of incorrect restart behavior.

## Required reading

Before editing, inspect:

- `docs/journal/2026-08-16_first-refactor.md`
- `docs/journal/2026-08-16_run-continuity-and-md-cli.md`
- `docs/journal/2026-08-16_rest2-verification-and-instruction-provenance.md`
- `src/md_templates/openmm/runstate.py`
- `src/md_templates/openmm/runner.py`
- `src/md_templates/openmm/schemas.py`
- `src/md_templates/openmm/config.py`
- `src/md_templates/openmm/bundle.py`
- `src/md_templates/openmm/cli.py`
- `src/md_templates/openmm/md.py`
- `src/md_templates/openmm/rest2.py`
- all shipped manifests, documentation, and tests

Preserve the already-correct behavior:

- `n_chunks` and `chunk_ns` are explicit inputs; totals are derived;
- `n_chunks` means additional work for the current invocation;
- named runs are exact and resume occurs in the same directory;
- omega exclusion defaults to enabled and is disabled only explicitly;
- counterions and added salt remain distinct;
- equilibrated structures use the selected NPT box;
- MD and REST2 remain separate public methods;
- lifetime REST2 statistics, exchange phase, replica assignment, and exchange RNG state survive a normal resume.

Do not change scientific defaults during this refactor. Capture the current values in versioned profiles first.

# Milestone 1: make committed restart state authoritative

The current code describes `restart/committed.json` as the only authority for completed state, but MD and REST2 still derive the resume chunk from `chunk_*/done.json`. This leaves a crash window: a chunk can have `done.json` while its restart generation is not committed. Resuming from the older state while advancing the chunk/step counter can skip physical propagation.

## Required behavior

- For an existing run, derive the resume boundary from the committed generation, not from the count of `done.json` files.
- If generation `N` is committed, the next chunk is `N + 1`.
- For a fresh run with no committed generation, the next chunk is zero.
- Treat chunk directories, trajectories, logs, and exchange rows beyond the committed boundary as an uncommitted tail.
- Before opening new reporters, either remove that tail safely or move it under a clearly labelled recovery/quarantine directory. Never append to or silently adopt it.
- Validate that every chunk at or below the committed boundary has the required method-specific outputs. A missing committed artifact is corruption and must stop the run.
- REST2 must validate that all replicas share the committed boundary.
- REST2 exchange history must be truncated only to the `attempts_committed` watermark in the committed record. Detect a log shorter than the watermark as corruption.
- The checkpoint and State files for the committed generation must be loaded before any new output is opened.
- Never set step/time counters to a value inconsistent with the loaded restart. Validate the loaded step/time against the commit record, allowing only a documented numerical tolerance for time.
- Make binary checkpoint writes durable before committing: close them and fsync the file. Keep serialized State writes atomic and durable.
- Retain at least the current and previous committed restart generations.

## Invocation provenance and logs

- Resuming must not truncate `stdout.log` or `stderr.log`.
- Either append with a visible invocation separator or create per-invocation log files plus stable top-level references.
- Keep the original run-level manifest stable. Do not overwrite its original start time or initial invocation.
- Store every later invocation in versioned, append-only invocation history.
- Add `production.md.seed` and `production.remd.seed` to the continuity contract. If the master or derived production seed changes, reject continuation with the differing field named.
- Continue to allow changes to extension-only fields such as `n_chunks`.

## Required recovery tests

Add deterministic tests for both MD and REST2 covering interruption:

1. before any generation files exist;
2. after only some restart members exist;
3. after all restart members exist but before the commit record changes;
4. after chunk outputs and `done.json` exist but before commit;
5. after exchange rows are durable but before commit;
6. immediately after the commit record changes;
7. with a corrupt binary checkpoint and valid State fallback;
8. with a missing/corrupt State and unusable checkpoint;
9. with an exchange log shorter than its committed watermark;
10. with an uncommitted chunk/output tail.

At least one test must terminate a real short CPU subprocess rather than only calling helper functions. Demonstrate that recovery resumes from the last committed physical state without skipping or double-running a committed chunk.

Do not call restart persistence crash-safe until these tests pass.

# Milestone 2: canonical simulation configuration

File syntax must not define simulation semantics. YAML, JSON, and future input formats must compile into one canonical, strictly validated model before any bundle or run directory is created.

## Configuration model

Introduce strict typed configuration models, using Pydantic v2 or an equivalently capable typed/schema library. The model must:

- reject unknown fields at every nesting level;
- distinguish required fields, optional fields, and derived values;
- validate ranges and cross-field relationships;
- expose machine-readable JSON Schema;
- serialize deterministically to canonical JSON;
- produce actionable errors that include the complete dotted field path;
- use discriminated method-specific models so MD cannot silently accept REST2-only fields and vice versa;
- keep system, build, protocol, and execution concerns conceptually separate;
- keep their schema versions independent.

Use these conceptual sections:

- `SystemSpec`: molecular identity and the declared PDB or SMILES route;
- `BuildSpec`: parameterization, force fields, water model, box, ions, nonbonded method, constraints, and HMR;
- `ProtocolSpec`: equilibration and method-specific production controls;
- `ExecutionSpec`: platform/device, reporting, output location, and other machine/runtime choices.

They may live in one document, separate documents, or both, but they must have independent canonical projections and hashes:

- system/build hash: anything requiring a new prepared bundle;
- protocol/continuity hash: anything changing the physical run;
- execution projection: choices that may change performance or outputs without changing the Hamiltonian.

Do not use filenames, project names, or experiment IDs as scientific compatibility checks.

## Physical units

Accept explicit quantities such as:

- `2 fs`
- `300 K`
- `1 /ps`
- `1.0 nm`
- `1 bar`
- `10 ps`

Use a restricted, safe unit parser. Do not use Python `eval`. Normalize each quantity into one documented canonical unit before hashing. The canonical output must show both the normalized numerical value and its unit convention.

For compatibility with existing manifests, support their current numeric fields through a documented migration path. Do not silently reinterpret an old unitless field under a different unit.

## Input formats

Support:

- `.yaml` and `.yml`;
- `.json`.

Both loaders must feed the same typed model. The same configuration expressed in YAML and JSON must produce byte-identical canonical JSON and the same configuration hashes.

Do not create format-specific defaults or validation.

Design a small loader registry so a future Amber-style namelist front end can be added without changing the canonical model or runners. Do not claim Amber compatibility and do not implement the full Amber syntax in this pass. Document the future adapter boundary and require all unsupported namelist keys to fail rather than be ignored.

## Versioned default profiles

Replace hidden mutable defaults with packaged, named, versioned profiles. Preserve the current scientific defaults as the first profile versions. Suggested profile roles include:

- standard explicit-water conventional MD for the peptide/PDB route;
- standard explicit-water conventional MD for the ligand/SMILES route;
- standard explicit-water REST2 for each supported route;
- CPU smoke testing.

Names may differ, but each profile must include:

- a stable profile ID;
- a profile schema version;
- a human-readable description;
- its complete scientific defaults;
- a canonical hash.

A user may provide only the fields they want to override. Resolution precedence is exactly:

1. versioned profile;
2. user input document;
3. explicit CLI overrides.

The resolved output must record every final value and the source layer that supplied it.

If `profile` is omitted, an explicit `default` alias may select a profile from the declared input route and method, but resolution must print and persist the exact versioned profile selected. Never infer the molecular route from file contents; the route remains declared.

A profile update that changes a scientific value requires a new profile ID/version. Do not mutate an existing version in place.

## CLI configuration interface

Add or complete these public commands:

```text
md-openmm config list-profiles
md-openmm config init --method md|rest2 --route pdb|smiles [--profile PROFILE] --output FILE
md-openmm config validate INPUT
md-openmm config resolve INPUT [--format json|yaml] [--output FILE]
md-openmm config diff INPUT_A INPUT_B
md-openmm config explain INPUT DOTTED.FIELD
md-openmm config migrate INPUT [--output FILE]
```

Requirements:

- `init` produces a documented, runnable template.
- `validate` performs complete schema and cross-field validation without running OpenMM.
- `resolve` prints the fully expanded configuration, profile version, source of every value, derived values, and hashes.
- `diff` classifies differences as bundle-defining, run-continuity-defining, execution-only, or extension-only.
- `explain` reports type, unit, default/profile source, allowed values, and scientific effect.
- `migrate` never edits the source file in place unless an explicit overwrite option is given. It reports every semantic change.

For occasional command-line overrides, provide a typed `--set dotted.path=value` mechanism rather than adding dozens of top-level CLI flags. Validate overrides through the same model.

The existing `prepare`, `md`, and `rest2` commands must consume the canonical resolved model. Do not maintain a second legacy execution configuration path indefinitely. Provide a clear migration and deprecation boundary.

# Milestone 3: transferable bundle contract

A prepared bundle must remain meaningful after it is copied away from both the source project and the original machine.

## Required bundle contents

At minimum include:

```text
bundle/
  bundle_manifest.json
  canonical_system_build.json
  canonical_protocol_at_prepare.json
  original_inputs/
  system.xml
  topology.pdb
  topology.cif
  equilibrated_state.xml
  forcefield_provenance.json
  environment.json
  checksums.json
```

Use the repository's actual stable filenames where compatibility requires them, but expose a versioned manifest mapping logical roles to filenames.

The bundle manifest must record:

- bundle schema version;
- package version and source commit when available;
- canonical system/build configuration and hashes;
- selected profile ID/version/hash;
- OpenMM version;
- plugin/toolkit versions used during parameterization;
- force-field identifiers and checksums of the exact parameter resources used;
- original molecular-input hashes;
- topology atom count and OpenMM particle count separately;
- constraint and virtual-site counts;
- periodic box vectors;
- composition and ion accounting;
- every bundle-file checksum;
- known portability limitations.

## Portability rules

- No scientific or required file reference may use an absolute path.
- Copy every required input into `original_inputs/` and address it by a bundle-relative path.
- Validation must work from any current working directory.
- Validation must work after the original source inputs and repository checkout are removed.
- Validation should not require network access.
- The bundle must contain enough provenance to rebuild the System, even when exact rebuilding requires the recorded external software versions.
- Record the distinction between an exact same-environment continuation and a compatible portable continuation.
- Continue to store both OpenMM binary checkpoints and serialized States in run directories. Never describe binary checkpoints as generally portable.
- `validate-bundle` must verify schema versions, canonical hashes, individual file hashes, topology/particle metadata, required files, and the absence of required absolute paths.

Provide a `md-openmm bundle inspect BUNDLE` command that summarizes, without constructing an OpenMM Context:

- system identity and route;
- force fields and water model;
- atom/particle/constraint counts;
- box and composition;
- build/profile hashes;
- compatible methods;
- software versions;
- validation status and warnings.

# CI and release acceptance gates

Add GitHub Actions for the supported environment. At minimum:

- install the package from a built wheel, not only editable source;
- run from a working directory outside the checkout;
- run unit tests;
- validate all shipped profiles and example inputs;
- confirm equivalent YAML and JSON inputs have identical canonical outputs/hashes;
- prepare and validate a tiny CPU bundle;
- copy the bundle elsewhere and validate it again;
- run short conventional MD;
- resume conventional MD in the same directory;
- run short REST2;
- resume REST2 with lifetime statistics and RNG phase preserved;
- exercise checkpoint-to-State fallback;
- exercise the subprocess crash-recovery gate from Milestone 1;
- inspect the built wheel and confirm schemas, profiles, examples, and required package resources are present.

Do not use exact floating-point energy equality across platforms. Use scientifically justified tolerances and invariant checks.

Define and document supported Python and OpenMM versions. Add a changelog and migration notes for schema/profile changes. Do not publish a release as part of this task unless explicitly requested.

# Tests required for configuration portability

At minimum test:

1. YAML and JSON equivalence;
2. deterministic canonical serialization and hashing;
3. unknown-key rejection at nested levels;
4. invalid and dimensionally wrong units;
5. cross-field errors such as non-divisible exchange/chunk intervals;
6. MD rejecting REST2-only settings and REST2 rejecting MD-only settings;
7. default-profile resolution;
8. explicit profile pinning;
9. user overrides winning over profiles;
10. CLI overrides winning over user files;
11. source attribution for every resolved field;
12. profile scientific changes requiring a new version;
13. semantic diff classification;
14. migration from the currently shipped system and experiment schemas;
15. bundle validation after relocation;
16. absence of required absolute paths;
17. tampered input, force-field resource, System, State, or manifest detection;
18. separate topology atom and OpenMM particle counts;
19. wheel/package-data completeness;
20. commands working outside the repository checkout.

# Documentation and examples

Update the README and focused configuration documentation with:

- the separation between system/build/protocol/execution;
- minimal input using defaults;
- fully explicit input;
- equivalent YAML and JSON examples;
- how defaults and overrides are resolved;
- how to validate, resolve, explain, diff, and migrate;
- what makes a bundle portable;
- what is and is not guaranteed across machines;
- the future Amber-style adapter boundary;
- conventional MD and REST2 examples;
- restart and crash-recovery guarantees.

Keep examples generic and project-independent.

# Deliverables and completion report

Before reporting completion:

1. run the full non-slow test suite;
2. run all new configuration CLI commands;
3. build the wheel and inspect its contents;
4. run the CPU portability and crash-recovery integration gates;
5. audit the active package, examples, documentation, and tests for project-specific identity or paths;
6. inspect `git diff` for unrelated changes;
7. add a dated journal entry under `docs/journal/`.

The journal must record:

- architecture decisions;
- schema and profile versions;
- canonicalization and hash rules;
- supported file formats;
- defaults and override precedence;
- bundle format changes;
- migration behavior;
- tests and exact commands/results;
- current portability limitations;
- deferred work for the Amber-style adapter.

Do not report this task complete based only on unit tests. Demonstrate a built wheel, a relocated bundle, a normal resume, and recovery from an interrupted subprocess.

Do not implement an Amber-style parser until the canonical model, profile system, and format-equivalence tests are complete. The future parser must be an adapter into this model, never a parallel set of simulation semantics.
