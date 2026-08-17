# Claude Code instruction: finish the transferable bundle contract and add CI

Work on the `openmm` branch of `csy0000/MD-templates`.

Read and follow `CLAUDE.md` first.

Then read:

- `claudecode-instructions/20260816_portable-simulation-config.md`
- `docs/journal/2026-08-16_portable-simulation-config.md`
- `docs/journal/2026-08-16_run-continuity-and-md-cli.md`
- the current canonical configuration implementation under `src/md_templates/openmm/spec/`
- `src/md_templates/openmm/bundle.py`
- `src/md_templates/openmm/runner.py`
- `src/md_templates/openmm/cli.py`
- all bundle, configuration, packaging, and crash-recovery tests

This is the continuation of the previous instruction, not a request to redo it.

## Accepted current status

Treat the following as completed unless inspection finds a regression:

- committed-generation restart recovery;
- uncommitted-tail quarantine;
- checkpoint fsync and State fallback;
- append-only invocation logs and stable original run manifest;
- production seeds in the continuity contract;
- real subprocess crash-recovery coverage;
- canonical Pydantic configuration;
- strict unknown-key rejection;
- YAML/JSON canonical equivalence;
- explicit unit parsing without `eval`;
- MD/REST2 discriminated models;
- versioned profiles with explicit `is_default`;
- profile -> document -> `--set` precedence;
- per-leaf source attribution;
- configuration inspection commands;
- canonical SMILES preparation through the existing builder.

Do not retune scientific defaults. Do not remove `is_default` or return to selection by filename/sort order. A smoke profile must never be selected automatically for a production document.

This task must complete:

1. the remaining canonical-model integration;
2. the transferable bundle contract;
3. relocation and offline verification;
4. enforceable GitHub Actions CI.

Do not implement the Amber-style namelist adapter in this task.

# Decision 1: model the master seed explicitly

Add an explicit canonical randomness block. Suggested shape:

```yaml
randomness:
  schema_version: 1
  master_seed: 20260814
  stage_seeds:
    structure: null
    equilibration: null
    md: null
    rest2: null
```

Exact names may follow the existing model style, but the semantics are required:

- `master_seed` is part of the canonical resolved configuration.
- Preserve the legacy deterministic stage-seed derivation exactly so migrated inputs reproduce existing trajectories.
- An explicit stage seed overrides only its derived stage seed.
- Persist the master seed and every resolved stage seed.
- Attribute each resolved seed to profile, document, CLI override, or derivation.
- Compare the resolved production seed in the run continuity contract.
- Include the structure-generation and equilibration seeds in the bundle-defining projection because they change the prepared artifacts.
- Include the resolved MD or REST2 production seed in the run-continuity projection.
- Do not hash a label merely because it is named `master_seed`; hash the resolved stage seeds according to their actual scientific consequence.
- A changed master seed normally changes derived seeds and therefore the appropriate hashes. If every affected stage seed is explicitly pinned to the old value, the physical hashes may remain unchanged; explain this in `config diff`.

Migrate the legacy experiment `master_seed` into this block. Add tests proving exact preservation of the existing offset/derivation rule.

# Decision 2: finish canonical configuration integration

## PDB preparation route

Make `md-openmm prepare --config DOCUMENT` support both declared routes:

- `smiles`;
- `pdb`.

For PDB:

- resolve the PDB path relative to the configuration document, never the current working directory;
- validate its hash and declared chemical metadata;
- copy it into the bundle's `original_inputs/`;
- ensure preparation works after the original PDB is moved or deleted;
- do not route a PDB canonical document back through a project manifest as its source of scientific truth.

Remove the temporary SMILES-only rejection after the PDB route is tested end to end.

## MD and REST2 run commands

Add canonical configuration support to both:

```text
md-openmm md    --bundle BUNDLE [--config DOCUMENT] [--set dotted.path=value ...]
md-openmm rest2 --bundle BUNDLE [--config DOCUMENT] [--set dotted.path=value ...]
```

Semantics:

- A version-2 canonical bundle contains the canonical configuration needed to run. When `--config` is omitted, resolve from the bundle's pinned canonical configuration/profile snapshot.
- `--config` is an optional run override, not a second source of build truth.
- Resolve `--config` and `--set` through the same typed model, profile resolution, units, canonicalization, source attribution, and hash code used by `config resolve` and `prepare`.
- Verify the command matches the method discriminator. `md` rejects a REST2 production model and `rest2` rejects an MD model.
- Recompute the bundle-defining projection and require it to match the prepared bundle before creating a run directory.
- Permit a different production plan only when it is compatible with the prepared artifacts.
- On resume, compare the resolved run-continuity projection against `run_state.json` before appending anything.
- Allow extension-only changes such as `n_chunks`.
- Keep platform/device and output paths as execution choices and record them.
- Do not allow a CLI flag and canonical field to disagree silently. Define and test precedence.
- Preserve a documented legacy path for version-1 bundles/manifests, but new version-2 bundles and runs must use the canonical path. Do not maintain two independent scientific resolvers.

Audit the projection classification. Any field affecting the exact prepared starting state belongs in the bundle identity even if its model field is located under `protocol`. This includes the equilibration protocol, equilibration duration, temperature schedule, restraints, barostat settings, box-selection settings, integrator settings used during equilibration, and equilibration seed.

# Milestone 3: versioned transferable bundle contract

Introduce a bundle schema version independent of the system, experiment, canonical-config, and run-state schema versions.

Write new bundles using `BUNDLE_SCHEMA_VERSION = 2` or the next available independent value. Do not reuse `SCHEMA_VERSION` from the legacy system manifest.

Continue to validate existing version-1 bundles without silently treating them as version 2. Clearly label their missing guarantees. A version-1 bundle may remain runnable through the documented compatibility path, but it must not be reported as satisfying the version-2 portability contract.

## Required version-2 layout

Use stable logical roles in the manifest. Filenames may follow existing conventions, but a version-2 bundle must contain the equivalent of:

```text
bundle/
  bundle_manifest.json
  checksums.json
  canonical_configuration.json
  canonical_system_build.json
  canonical_protocol_at_prepare.json
  resolved_runtime_config.json
  system.xml
  topology.pdb
  topology.cif
  simbox.json
  equilibrated_state.xml
  forcefield_provenance.json
  environment.json
  original_inputs/
    simulation.yaml-or-json
    input.pdb | input.smi
    legacy_system.yaml          # only when the legacy front end was used
    legacy_experiment.yaml      # only when the legacy front end was used
```

Do not duplicate data merely to match these example names. Prefer one authoritative copy plus a manifest mapping logical roles to relative paths.

## Bundle manifest requirements

Record at minimum:

- independent bundle schema version;
- bundle ID;
- creation timestamp;
- package version;
- source commit when available;
- canonical configuration schema versions;
- selected profile ID, profile version, profile hash, and exact profile snapshot;
- canonical system/build projection and hash;
- canonical prepared-state projection and hash;
- canonical protocol-at-prepare projection and hash;
- resolved master and stage seeds;
- molecular route and molecular identity hash;
- topology atom count;
- OpenMM particle count;
- virtual-site count;
- massless-particle count;
- constraint count;
- degrees of freedom and the formula used;
- periodic box vectors and volume;
- solute, solvent, counterion, and added-salt accounting;
- force-field identities and resource provenance;
- OpenMM, Python, toolkit, plugin, and parameterization-library versions;
- the logical role and bundle-relative path of every required artifact;
- known exact-reproduction and portable-reproduction limitations.

Do not call OpenMM particle count `n_atoms`. Calculate topology atoms from `topology.atoms()` and particles from `System.getNumParticles()` separately. Add a test containing or mocking virtual sites so equality is not accidentally assumed.

## Original inputs

- Copy the exact user configuration document into `original_inputs/`.
- Copy the exact PDB input when using the PDB route.
- For inline SMILES, write a small canonical input record containing the declared and canonical isomeric SMILES, formal charge, and molecular hash.
- When the legacy manifest front end is used, preserve the exact supplied manifests as original inputs, but also include the canonical migrated configuration.
- Store operational references as bundle-relative paths.
- Do not require the source checkout or original input path after preparation.

## Force-field provenance

Create `forcefield_provenance.json` containing:

- force-field names and versions;
- the package/distribution that supplied each resource;
- package version;
- resolved resource name;
- SHA-256 of the exact XML/offxml/parameter resource where it can be read;
- charge method and toolkit versions;
- combination/mixing assumptions;
- any extra user-supplied force-field files and their hashes.

Copy user-supplied parameter resources into the bundle when licensing and size permit. Otherwise record enough identity and hashing information to detect that a target environment supplies a different resource.

Do not claim that recording a package name alone identifies the force field.

## Checksums

Create one versioned `checksums.json` mapping normalized relative POSIX paths to SHA-256 values.

Define its domain explicitly to avoid self-reference:

- include every required scientific artifact and original input;
- exclude `checksums.json` itself;
- state whether `bundle_manifest.json` is included or excluded and why;
- reject duplicate, absolute, parent-traversal, platform-dependent, or non-normalized paths;
- hash bytes, not parsed/normalized text.

Validation must fail on a changed, missing, extra-required, or mismatched file. Internal hashes provide integrity checks, not authenticity; do not describe them as signatures.

## Environment provenance

Write a machine-readable `environment.json` containing the versions relevant to interpreting or rebuilding the bundle. Do not make the bundle operationally depend on absolute environment paths.

Classify environment fields as:

- required for exact rebuild;
- required for supported execution;
- informational only.

A bundle transfer should use the stored `system.xml` and State. Rebuilding from inputs is a separate operation and is not expected to be bitwise identical unless the exact environment is reproduced.

## Topology formats

Write both:

- PDB for broad interoperability;
- mmCIF/PDBx for topology and box representation without PDB format limits.

Validate that both describe the same atom/residue ordering and periodic box within documented tolerances.

# Bundle CLI

Add a coherent bundle command group while retaining compatibility aliases as needed:

```text
md-openmm bundle validate BUNDLE [--deep]
md-openmm bundle inspect BUNDLE [--format text|json|yaml]
md-openmm bundle relocate-check BUNDLE
```

Requirements:

- `validate` verifies schema, required logical roles, normalized relative paths, checksums, canonical hashes, profile snapshot/hash, topology/particle metadata, force-field provenance, and absence of required absolute references.
- Default validation must not construct an OpenMM Context.
- `--deep` may deserialize the System/State and cross-check counts, box, constraints, parameters, topology ordering, and canonical projections, but still must not propagate dynamics.
- `inspect` summarizes identity, route, method compatibility, profile, force fields, water, atom/particle/virtual-site/constraint counts, box, composition, hashes, environment, and portability warnings.
- `relocate-check` copies the bundle to a temporary unrelated directory, validates it there, and reports any reference that escaped the bundle. It must not modify the source bundle.
- Keep `validate-bundle --bundle B` temporarily as a documented compatibility alias if users already rely on it.

All commands must work from outside the checkout after wheel installation.

# Relocation, offline, and execution gates

Add end-to-end CPU tests that:

1. create an isolated source directory containing a canonical SMILES input;
2. prepare a version-2 bundle;
3. copy it to an unrelated directory;
4. rename or remove the original source directory;
5. validate and inspect the copied bundle;
6. run a short conventional MD job from the copied bundle;
7. run a short REST2 job from the copied bundle;
8. resume each in the same run directory;
9. confirm canonical hashes and prepared artifact hashes do not change during relocation;
10. confirm the commands do not access the original paths.

Repeat the preparation/relocation test for a PDB canonical input.

Make an offline gate by preventing network access in the test process or monkeypatching common network entry points. Validation and execution from an already prepared bundle must not attempt downloads.

Do not delete source data outside a dedicated temporary test directory.

# GitHub Actions CI

Add GitHub Actions workflows that turn the local evidence into repository gates.

## Fast workflow

Run on pull requests and pushes to maintained branches:

- checkout;
- create the documented supported environment;
- build the wheel;
- inspect wheel contents;
- install the wheel into a clean environment;
- run commands from a directory outside the checkout;
- run formatting/static checks already adopted by the repository;
- run all non-slow tests;
- validate every shipped profile, schema, manifest, and example;
- verify YAML/JSON canonical hash equivalence;
- run bundle unit/relocation tests that fit the normal time budget.

## Portable CPU integration workflow

Run on pull requests if practical; otherwise on pushes plus manual dispatch and a scheduled cadence:

- prepare a tiny canonical SMILES bundle;
- prepare a tiny canonical PDB bundle;
- relocate and validate both;
- conventional MD fresh + resume;
- REST2 fresh + resume;
- corrupt-checkpoint State fallback;
- the real subprocess crash-recovery test;
- offline validation/execution;
- built-wheel execution outside the checkout.

Use explicit timeouts. Cache environments safely without caching generated scientific results. Upload concise logs and manifests on failure, but do not upload large trajectories or checkpoints by default.

The workflows must use only CPU and must not assume CUDA.

## Support matrix

Determine supported versions from actual successful CI, not aspiration.

At minimum gate the current documented Python version and one explicitly pinned OpenMM environment. Add another Python/OpenMM version only if the workflow actually passes.

Document:

- supported Python versions;
- supported OpenMM versions;
- tested operating systems;
- what portability means within and outside the matrix;
- binary checkpoint limitations;
- serialized State limitations.

Add a changelog entry for bundle schema version 2 and canonical-run integration. Do not publish a package release in this task.

# Tests required

At minimum add tests for:

1. explicit master seed and legacy stage-seed derivation;
2. stage-seed overrides and source attribution;
3. seed projection classification;
4. PDB canonical preparation;
5. PDB path resolution relative to the document;
6. MD and REST2 consuming bundle-pinned canonical configuration;
7. MD and REST2 `--config` and `--set` precedence;
8. method mismatch rejection;
9. bundle-defining mismatch rejection before run-directory creation;
10. extension-only override acceptance;
11. independent bundle schema versioning;
12. legacy bundle validation with downgraded guarantees;
13. exact original-input preservation;
14. normalized relative logical paths;
15. absolute and parent-traversal path rejection;
16. checksum tampering and missing-file detection;
17. profile snapshot/hash verification;
18. topology atom versus OpenMM particle counts;
19. virtual-site and massless-particle accounting;
20. topology PDB/mmCIF ordering consistency;
21. force-field resource provenance and hashing;
22. environment provenance classification;
23. relocation after source removal;
24. offline validation and execution;
25. bundle inspect structured output;
26. compatibility alias behavior;
27. wheel inclusion of schemas, profiles, examples, and required resources;
28. CI command scripts running outside the checkout.

Keep the existing crash-recovery regression tests and scientific invariants passing.

# Historical instruction audit decision

Do not use `.gitignore` to hide tracked historical instruction files. Adding a tracked file to `.gitignore` does not remove it and would make the audit look cleaner without changing repository contents.

Keep the instruction files tracked as implementation provenance. Scope the active project-independence audit to package code, current user documentation, examples, manifests, tests, packaging metadata, and workflows, while documenting the historical instruction-directory exception.

Do not edit or delete historical instructions solely to change the match count in this task.

# Completion gates

Before reporting completion:

1. run the full non-slow suite;
2. run the real slow crash-recovery test;
3. build and inspect the wheel;
4. install the wheel cleanly;
5. run from outside the checkout;
6. prepare, relocate, validate, inspect, run, and resume both canonical routes on CPU;
7. run the offline gate;
8. execute the exact local commands used by both CI workflows;
9. confirm the workflows are present and syntactically valid;
10. inspect the complete diff for unrelated changes;
11. add a dated journal under `docs/journal/`.

The journal must clearly separate:

- completed requirements;
- local evidence;
- CI evidence actually observed;
- checks configured but not yet observed on GitHub;
- limitations;
- deferred Amber-style adapter work.

Do not report CI as passing merely because workflow files were added. If GitHub has not run them yet, report them as configured and locally reproduced, awaiting or requiring remote observation.

Do not claim cross-machine bitwise reproducibility. State precisely:

- transferred prepared artifacts are byte-identical when checksums match;
- binary checkpoints are environment-specific;
- serialized State provides a physically valid portable fallback but not stochastic bitwise continuation;
- rebuilding from original inputs may be scientifically consistent without being bitwise identical.
