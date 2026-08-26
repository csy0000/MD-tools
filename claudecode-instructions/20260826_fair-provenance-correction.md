# Focused FAIR provenance correction

Date: 2026-08-26  
Repository: `csy0000/MD-templates`  
Working branch: `dev`  
Reviewed implementation: `d39b66508f2e2d3f51cc9434d1b0457bc09fc57f`

## Purpose

Correct the remaining FAIR-provenance defects found after the first `0.4.0.dev0`
implementation. This is a metadata, lineage, and retrofit correction. It is not a new workflow,
not a scientific-method change, and not another long molecular-dynamics validation campaign.

Keep the simplified six-command architecture:

- `md-template init`
- `md-template install`
- `md-openmm show-default`
- `md-openmm sys-config`
- `md-openmm sys-gen`
- `md-openmm md-gen`

Keep:

- `generate_system()` in `src/md_templates/openmm/sysgen.py`;
- `generate_md()` in `src/md_templates/openmm/mdgen.py`;
- the existing staged directory tree;
- standalone generated OpenMM scripts;
- the current cMD and REST2 algorithms, defaults, scaling, exchange, GPU scheduling, restart
  behavior, trajectories, restraints, barostats, seeds, and box-vector handling.

Do not restore the retired framework or the root-level `MD_system_gen.py` and
`MD_input_gen.py`. Do not add a seventh command.

## Priority 0: branch hygiene

Before editing:

1. fetch the current remote state;
2. switch to `dev`;
3. inspect `git status` and preserve unrelated work;
4. merge `origin/main` into `dev` with an ordinary non-destructive merge so that development is
   not left behind the release branch;
5. do not rebase, rewrite history, delete branches, or merge `dev` into `main`.

Resolve only real conflicts. The current difference is expected to be release-history/merge
ancestry rather than a reason to redesign code.

## 1. Record the force field that was actually used

`inputs/forcefield.json` must be written from the builder's actual report, not merely copied from
the requested configuration.

Correct at least these cases:

### Explicit peptide with OPC

- protein resource: the exact resource actually loaded, normally `amber19-all.xml`;
- water/ion resource: the qualified resource actually loaded, normally
  `amber19/opc.xml`, not the shorter user-facing label `opc.xml`;
- builder route: `openmm.app.ForceField.createSystem`.

### Explicit ligand-only route

- protein force field/resource must be `null`, because this route intentionally does not load
  ff19SB;
- record the exact SMIRNOFF/OpenFF force-field resource selected by the template generator;
- retain the requested human-facing Sage label separately if useful, but do not present it as the
  exact loaded resource;
- record the actual charge scheme. For AM1-BCC, retain the selected scheme; for NAGL, retain the
  model name/path-independent identifier and model checksum already reported by the builder;
- record the retained charged ligand representation and its checksum.

### Implicit peptide with GBn2/mbondi3

- do not claim that `amber19-all.xml` constructed this System;
- record the actual tleap resource, normally `leaprc.protein.ff19SB`;
- set an OpenMM XML field to `null` when no OpenMM protein XML was loaded;
- record the ParmEd `Structure.createSystem` route, GBn2, mbondi3, NoCutoff, and the actual
  HMR result.

### Implicit ligand-only route

- protein force field/resource must be `null`;
- record the actual OpenFF/Sage and charge provenance used before conversion through ParmEd;
- record the retained SDF/MOL2 and Amber construction artifacts with checksums;
- retain the ParmEd GBn2/mbondi3 construction route.

Propagate the builder report through `sysgen.py` rather than recreating the decision in
`forcefield_record.py`. Fields that do not apply must be explicit `null`.

Do not flatten two different retained preparation artifacts onto the same basename. Preserve a
stable staging-relative subpath under `inputs/preparation/`, or reject a content collision.
A rerun must include already-present identical retained artifacts in the recorded artifact list.

## 2. Correct the v0.3.x retrofit

Keep `scripts/retrofit_fair_v030.py` Python-plus-PyYAML-only, offline, and non-destructive.

### Relative, valid paths

- remove the absolute `common_root` from `file-inventory.yaml`;
- do not record an absolute `--environment` source path;
- use `Path.is_relative_to()` or equivalent path-aware logic, never string-prefix containment;
- require `--output` to be inside the common project root if that is necessary for one directly
  verifiable relative `SHA256SUMS`;
- compute the output directory's real relative name dynamically. Never hardcode
  `fair-registration/`;
- make every entry in `SHA256SUMS` resolve from one documented root and verify it in a test with
  at least two output names, such as `fair-registration` and `out`.

### Retain supplied evidence

When `--original-input` passes the recorded-hash check:

- copy it byte-for-byte to `<output>/original_inputs/<basename>`;
- record that relative path and checksum;
- include it in the sidecar checksum manifest;
- refuse a conflicting destination.

If an environment record is supplied, embed its parsed content as now and either retain its
original bytes under an evidence subdirectory or omit the source path. Do not leave a required
absolute reference.

### Honest validation

Do not write `source_modified: false` as an unsupported literal. Take source snapshots before and
after sidecar creation and compare path, size, mtime, and SHA-256. Record:

- the validation method;
- before/after file counts;
- whether they match;
- machine-readable changed/added/removed lists.

The synthetic test must independently confirm the same property. Large-file reads must remain
streamed and may print progress.

### Correct A/B/C grading

Grade A requires all of the following evidence:

- prepared `system.xml`, `topology.pdb`, and `initial_state.xml`;
- resolved system configuration and MD configuration;
- the verified original molecular input retained in the registration candidate;
- an exact implementation identity: a non-null recorded commit or installed fingerprint. A package
  version by itself is not exact identity;
- the required build environment;
- required runtime records for the configured common stages and every configured cMD/REST2 method.

If the configured protocol cannot be determined from recorded content, do not infer it from folder
names and do not award A. Grade B applies when the prepared system and protocol are reusable but
original-input, exact-code, environment, or runtime closure is incomplete. Grade C applies when the
prepared simulation itself is incomplete.

Every reason lowering the grade must remain machine-readable. Update the synthetic fixtures:
the existing fixture with `git_commit: null` and no cMD runtime record must not expect A. Add one
genuinely complete A fixture containing exact identity and all records required by its declared
protocol.

Correct the retrofit module documentation: v0.3.x did record an original-input hash in
`inputs/provenance.yaml` when available; it generally did not retain the original bytes.

## 3. Complete runtime input lineage

Do not change propagation.

### cMD

Before loading or overwriting the starting artifact, record exactly what this invocation consumed:

- role: parent final state or own production checkpoint;
- path relative to `MD/cMD/`;
- byte size;
- SHA-256.

Persist it in both `cMD/resolved_run.yaml` and the appended invocation entry. On resume, hash the
checkpoint before it is overwritten at successful completion.

### REST2

For every replica, before loading or overwriting its starting artifact, record:

- role: its tau-specific equilibration final state or its own production checkpoint;
- path relative to `MD/REST2/`;
- byte size;
- SHA-256.

Persist this for every replica in `REST2/resolved_run.yaml` and
`REST2/invocations.jsonl`. Keep exchange-phase and RNG provenance as recorded facts; do not alter
the exchange algorithm in this task.

### Common stages

A common stage already records its parent final-state hash. Fix the output-record ordering:

1. finish and atomically write the scientific outputs;
2. write the final `stage.log`;
3. inspect the finished files for the runtime record;
4. atomically write `resolved_stage.yaml`.

The `stage.log` entry in `outputs` must never describe a missing, stale, or pre-final log.

## 4. Tests: focused and short

Do not rerun the 1 ns/5 ns production protocols. Do not launch a broad multi-GPU campaign.

Add or correct focused tests that would fail under `d39b665`:

1. exact explicit-peptide OPC resources;
2. ligand-only protein resource is null and exact ligand/charge provenance is retained;
3. implicit peptide records tleap ff19SB plus ParmEd GBn2/mbondi3 rather than an OpenMM protein XML;
4. retained-artifact collision safety;
5. retrofit records contain no required absolute paths;
6. checksum paths work for a non-default output-directory name;
7. a verified original input is copied and checksummed;
8. the script's before/after validation is based on real snapshots;
9. incomplete identity or missing required runtime records cannot receive Grade A;
10. one genuinely complete fixture receives Grade A;
11. cMD fresh start and resume record the input artifact and its pre-overwrite hash;
12. every REST2 replica records its starting artifact and hash;
13. common-stage `stage.log` metadata matches the final file.

Run:

- the focused synthetic/unit tests;
- the full non-MD suite;
- at most one deliberately tiny CUDA runtime smoke project, using minimal steps and reusing one
  generated system, only to exercise the newly recorded cMD/REST2 runtime lineage.

Do not run scientific-duration validation. Existing scientific algorithms are outside this patch.
If the tiny CUDA smoke is impossible for an environmental reason, report that limitation; do not
replace it with a long CPU simulation and do not claim CUDA validation.

## 5. Documentation and versioning

- Keep the version at `0.4.0.dev0`.
- Update `docs/FAIR_HANDOFF.md` and README only where field semantics or retrofit usage changed.
- Add `docs/journal/2026-08-26_fair-provenance-correction.md` containing the exact defects fixed,
  files changed, validation commands/results, and remaining limitations.
- Do not create or move a release tag.
- Do not merge `dev` into `main`.

## Completion conditions

Do not return PASS solely because the old tests pass. Inspect representative generated records and
verify their values against the actual builder/runtime path.

Before PASS, report:

- the commit SHA pushed to `dev`;
- branch synchronization performed;
- exact changed files;
- representative force-field records for all four routes;
- retrofit A/B/C evidence;
- proof that all manifest paths verify for a non-default output name;
- source before/after snapshot evidence;
- cMD and REST2 starting-artifact hashes;
- proof that the recorded `stage.log` size/hash matches the finished log;
- exact focused, non-MD, and tiny-CUDA commands and results;
- any remaining limitation.

Continue fixing applicable failures without stopping for routine questions. Commit and push the
completed correction to `dev`.
