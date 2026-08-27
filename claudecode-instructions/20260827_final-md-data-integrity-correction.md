# Final MD-data integrity and execution-preflight correction

## Purpose

Close the remaining correctness gaps found after execution of
`20260827_md-data-contract-preflight-and-iterload.md`. The current implementation has the right
overall architecture. This is a focused integrity correction, not a redesign.

Work on the current `dev` branch of `csy0000/MD-templates`. At the time this instruction was
written, the inspected MD-templates head was
`a5f4a67d1376d7095dacc8a1a527363a9caae8f2`, and the authoritative MD-data `dev` contract head was
`48628f9a5d3ace6c6398a63bc3905cd58d542de3`. Pull the current `dev` branch first and inspect any
newer compatible changes rather than resetting either repository.

Read before editing:

- `CLAUDE.md`
- `claudecode-instructions/20260827_md-data-contract-preflight-and-iterload.md`
- `docs/journal/2026-08-27_md-data-contract-preflight-and-iterload.md`
- `src/md_templates/openmm/md_data_contract.py`
- `src/md_templates/openmm/templates/preflight.py`
- `src/md_templates/openmm/templates/ais_run.py`
- `src/md_templates/openmm/templates/stage_run.py`
- `tests/test_md_data_contract.py`
- the MD-data contract and public validator at the exact compatible commit used by acceptance

Keep the existing six-command interface and the existing canonical generators:

- `src/md_templates/openmm/sysgen.py::generate_system()`
- `src/md_templates/openmm/mdgen.py::generate_md()`

Do not restore deleted framework code, add a registry or database, add a new public command, scan
`$MD_DATA`, implement archival/lifecycle management, merge into `main`, or create/move a tag.

Write the implementation report to:

    docs/journal/2026-08-27_final-md-data-integrity-correction.md

Commit and push the completed work to `dev`.

## Required corrections

### 1. Never hash a production source trajectory at runtime

Remove the full-file SHA-256 calculation currently performed for the AIS source trajectory while
writing `AIS/inputs/sources.yaml`.

It is acceptable and required to stream the specifically selected source trajectory with
`mdtraj.iterload` in bounded chunks to survey frames and select starting configurations. That is
not the same as computing a full-file digest. Do not call `sha256_file`, `hashlib`, or an equivalent
whole-file hashing loop on the production source DCD during generation, preflight, preparation, or
execution.

Record bounded, non-authoritative source observations instead:

- configured and resolved relative source path;
- file size;
- frame count established through bounded `iterload`;
- first-frame time and frame interval;
- eligible and selected frame indices;
- selected-frame source times;
- chunk size and number of chunks read;
- source tau and the evidence from which it was established.

The prepared, selected `AIS/inputs/sources.dcd` is a small generated input. It may have a digest if
the implementation uses it to protect prepared-input identity, but the original production
trajectory must not be hashed. Clearly distinguish those two files in names and metadata.

Add a regression test that makes the production-source hashing helper fail if invoked, while AIS
preparation still succeeds through `mdtraj.iterload`.

### 2. Detect real DCD truncation, not only an incorrect header count

The current `dcd_frame_count()` trusts only the DCD `NSET` header. That cannot establish physical
completeness: an interrupted file can retain the expected header while coordinate bytes are
missing.

For small generated AIS artifacts:

- validate `AIS/inputs/sources.dcd` by actually reading every selected frame with bounded
  `mdtraj.iterload`;
- validate every finished `AIS/trajectory_<n>/observations.dcd` by actually reading the expected
  frames with bounded `mdtraj.iterload`;
- require exactly the expected frame count;
- require the final expected frame to be readable;
- reject read errors, malformed records, extra frames, missing frames, invalid coordinates, and
  missing/invalid periodic boxes for explicit solvent;
- do not use `mdtraj.load`;
- retain the safe replace-not-append behavior for incomplete AIS paths.

These AIS files contain only selected sources or the configured printouts, so bounded full
validation is appropriate. Do not apply this validation by walking or hashing unrelated production
datasets.

Replace or supplement the existing truncation tests with genuine physical truncation:

1. write a valid DCD whose header claims the correct number of frames;
2. remove bytes from the final coordinate record without changing the header;
3. prove preflight/completion refuses it;
4. prove a normal run replaces the incomplete path safely rather than appending;
5. perform the same physical-truncation test for `AIS/inputs/sources.dcd`.

A test that merely rewrites `NSET` to a smaller number is not sufficient.

### 3. Make force-field preflight route-aware and exact

`common/forcefield.json` records what was built. `resolved_sys.config.yaml` records what was
requested. Preflight must compare the scientifically consequential identities exactly and must not
allow an absent expected field to pass.

For the current peptide explicit route, require:

- resolved solvation is explicit;
- built protein resource equals the resolved ff14SB resource;
- built water resource equals the resolved TIP3P resource;
- no ligand force field is claimed to have been loaded.

For the current ligand explicit route, require:

- resolved solvation is explicit;
- built OpenFF resource equals the resolved Sage 2.2.1 resource;
- built ligand charge method equals the resolved ligand charge method;
- built water resource equals the resolved TIP3P resource;
- no protein force field is claimed to have been loaded.

For implicit routes, compare the applicable protein or ligand resource and charge method, GBn2
model, mbondi3 radii, and disabled SASA/nonpolar term. Require that no water model or barostat is
claimed for an implicit system.

Continue supporting ff19SB + OPC as an explicit optional selection, but do not mix it into the
default acceptance fixture. The acceptance matrix must collectively test the intended
method-development combination:

- peptide test: ff14SB + TIP3P;
- ligand test: Sage 2.2.1 with its configured charge method + TIP3P.

The current generator does not yet build a combined protein-ligand complex route, so do not falsely
claim that a single test system loads ff14SB, Sage, and TIP3P together.

Test missing fields as well as mismatched values. Every mismatch must fail during preflight before
an OpenMM `Context` exists.

### 4. Make `--check` verify current and parent stage identity

A stored `stage_config_sha256` being nonempty is not proof that it matches the current stage
request.

Create one canonical stage-fingerprint implementation in the generated standalone project and use
the same implementation for execution, completion, and preflight. Do not maintain two subtly
different hash algorithms.

For the current stage, preflight must:

- recompute the consequential configuration fingerprint from the current `stage.yaml`;
- compare it with `resolved_stage.yaml.stage_config_sha256`;
- verify the completion status and required final-state artifact;
- verify the recorded final-state identity if the existing small-file checksum contract includes
  it;
- refuse a stale completion record rather than skip or overwrite it.

For a parent stage, preflight must:

- load the parent `stage.yaml` and `resolved_stage.yaml`;
- recompute and compare the parent stage fingerprint;
- require completed status;
- require the expected parent final state;
- compare the small parent final-state checksum with the recorded handoff identity when available;
- refuse a changed or inconsistent parent before the downstream stage creates a `Context`.

`./run.sh --check` and `./run_all.sh --check` must execute these comparisons. They must not report
PASS merely because a hash string exists.

Add regression tests that complete a tiny stage, then alter one consequential current-stage value
and one consequential parent-stage value. Both `--check` calls must fail before `Context`
construction. Also test an unchanged completed chain passes.

### 5. Cross-check the MD-templates commit against generator provenance

A syntactically valid arbitrary 40-hex value is not sufficient template identity.

For contract-managed generation:

- obtain the generator commit from the existing generation-provenance mechanism;
- accept an exact Git checkout commit, installed VCS `direct_url.json` commit, or existing packaged
  build-provenance source;
- require `dataset.templates.commit` to equal that established generator commit;
- propagate the same value into resolved system/MD provenance and `dataset.yaml`;
- preflight must compare these records and refuse disagreement;
- if the installed source cannot establish an exact commit, fail contract-managed generation with
  a direct explanation instead of accepting or inventing one.

Unregistered local template inspection may continue without a contract commit if it is clearly
labelled unregistered. Do not fabricate a commit from the version, branch name, current date, or
origin project commit.

Add tests for matching identity, mismatching 40-hex identity, and unavailable generator identity.

### 6. Strengthen AIS atom identity/order validation

Atom count plus atom name is not sufficient because many residues contain repeated atom names.

Compare a per-index identity tuple between the AIS companion topology and
`common/topology.pdb`. Include, where available:

- chain identifier or chain index;
- residue index, residue identifier, and residue name;
- atom name;
- element symbol;
- bond connectivity expressed through atom indices.

Reject any mismatch before frame selection or worker spawning. The error must report the first
differing atom or bond clearly. Do not require coordinates to match, because the source ensemble is
expected to contain different coordinates.

Add a test where atom counts and atom-name sequences are unchanged but residue/chain assignment or
bond order is changed. It must fail.

### 7. Make the MD-data validator installation reproducible

The advertised contract-managed workflow must not instruct users to install from an unpinned SSH
URL or a moving branch.

Use one public HTTPS source pinned to an exact compatible MD-data release or 40-hex commit. Prefer a
published package release when one exists; otherwise use the exact public Git commit through HTTPS.
Keep that compatibility identity in one maintained location rather than duplicating it across
scripts.

Make the existing `md-template install` OpenMM path install and validate the dependency needed by
the advertised MD-data contract workflow. Do not add a seventh public command. The installation
report must record:

- MD-data package version;
- exact compatible contract/repository commit;
- import success;
- public validator availability.

A base Python import may remain lightweight, but a user who successfully runs the documented
installation must not discover only at `sys-gen` that MD-data is missing. Replace documentation
that recommends `git+ssh://.../dev` with the pinned public installation.

Test the generated/install command without downloading a moving branch: inspect the constructed
dependency specification and mock the package operation where appropriate.

## Acceptance and test budget

This task is integrity work, not a new scientific benchmark. Keep validation focused:

1. run the targeted unit and generation tests for the seven corrections;
2. run the repository's normal non-production test suite once;
3. run only the smallest existing CUDA smoke needed to prove generated scripts still execute;
4. do not run nanosecond-scale or multi-replica production simulations;
5. do not run the old multi-hour validation campaign again.

Every MD integration performed for this task must use the OpenMM CUDA platform. Never silently use
CPU or OpenCL. If CUDA is unavailable, run no MD integration, report that limitation, and do not
claim the GPU runtime test passed. Pure Python, manifest, parser, generated-source, and mocked
installer tests are not MD and may run without a GPU.

All runtime scientific fixtures must use the intended default routes:

- peptide: ff14SB + TIP3P;
- ligand: Sage 2.2.1 + configured charge method + TIP3P.

Do not use ff19SB + TIP3P, ff14SB + OPC, Sage + OPC, or TIP3P-FB as the claimed default acceptance
combination.

## Required report

Do not return PASS based on documentation, generated source inspection, or header counts alone.
The journal and final response must report:

- commit SHA pushed to `dev`;
- exact targeted and full test commands with pass/fail/skip counts;
- exact CUDA platform/device used by any MD smoke;
- evidence that the original production source DCD was not hashed;
- evidence from a physically byte-truncated DCD with an unchanged header;
- exact ff14SB/TIP3P and Sage/TIP3P identities tested;
- current-stage and parent-stage mutation results under `--check`;
- matching and mismatching template-commit results;
- atom-identity/order mismatch result;
- installed MD-data version and pinned contract commit;
- any remaining limitation.

Before reporting completion, inspect the final diff for scope creep. Do not redesign the CLI,
restore deleted machinery, merge to `main`, or create/move a release tag.
