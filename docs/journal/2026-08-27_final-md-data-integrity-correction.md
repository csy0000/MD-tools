# 2026-08-27 — final MD-data integrity correction

Instruction: `claudecode-instructions/20260827_final-md-data-integrity-correction.md`.
Branch `dev`. No merge to `main`, no tag. Six commands unchanged, no new machinery.

Seven corrections, and they share one shape: **a record that looks complete**. A DCD header that
claims the right frame count over bytes that are not there. A hash string that is present but was
never recompared. A force-field field that is absent rather than wrong. A 40-hex commit that is
syntactically perfect and names nothing. Every one of these passed the previous round's checks.

## 1. The production source is never hashed

`write_prepared_sources` computed a full-file SHA-256 of the AIS source trajectory. That put back
exactly the cost the bounded `iterload` survey exists to avoid: a source may be hundreds of
gigabytes, and hashing it at generation time makes the cost scale with the length of the
trajectory again.

Removed. `sources.yaml` now records bounded observations — configured and resolved relative path,
byte size, frame count from the survey, first-frame time and interval, eligible and selected frame
indices, selected-frame times, chunk size, chunks read in each pass, and the source tau with its
evidence. `trajectory_sha256` is explicitly `null` beside a `trajectory_not_hashed` note, because
a silently absent field reads as an oversight.

`AIS/inputs/sources.dcd` is a different file: small, generated, and legitimately digestible. The
two are named and recorded separately so nothing can confuse them.

## 2. Physical truncation, not a header count

`dcd_frame_count()` read `NSET` out of the 100-byte header. That cannot establish completeness:
an interrupted write leaves `NSET` at the value the writer *intended* and the final coordinate
record short or missing. A file can claim 21 frames and hold 15.

The header read survives as `dcd_header_frames()`, documented as a first signal only. Completeness
is now established by `validate_generated_dcd()`, which READS every frame with bounded
`iterload` and requires: the exact expected count, a readable final frame, finite coordinates
throughout, and — under explicit solvent — periodic box vectors that are present, finite and
non-degenerate. Read errors, missing frames, extra frames and malformed records all fail.

This is affordable *because* these files are small by construction: one frame per path, or the
configured observations. It is never pointed at a production trajectory, and the AIS source is
still surveyed by bounded `iterload` rather than validated this way.

Replace-not-append is unchanged: an incomplete path directory is replaced, never added to.

## 3. Route-aware, exact force-field preflight

The old comparison only fired when *both* sides carried a value, so a `forcefield.json` missing the
field entirely passed — which is the case where what was built is least knowable. `_exact()` now
fails on an absent expected value from either side, and `_absent()` states which force fields must
*not* have been loaded.

By route:

| route | required |
|---|---|
| peptide, explicit | built protein == resolved ff14SB resource; built water == resolved TIP3P resource; **no ligand force field claimed** |
| ligand, explicit | built OpenFF resource == resolved Sage resource; built charge method == resolved charge method; built water == resolved TIP3P; **no protein force field claimed** |
| implicit | applicable protein/ligand resource and charge method; GBn2 model; mbondi3 radii; SASA term off; **no water model and no barostat claimed** |

The route itself is cross-checked: a record describing a ligand build under a configuration asking
for a peptide fails before any field comparison.

Two supporting changes made this comparable without duplicating knowledge. `_openff_name` moved out
of `sysgen` into `config.openff_resource`, so the **resolved configuration records the resource
that will actually be loaded** (`openff-2.2.1`) rather than the label (`sage-2.2.1`) — preflight
compares exactly, and the generated project carries no second copy of the mapping. And the ligand
route now records `forcefield.protein: null`, because `builder.openmm_xml_loaded` for a ligand
build is `['amber14/tip3p.xml']` — no protein XML is loaded, and naming one was a false claim. Same
rule the implicit route already applied to water.

ff19SB + OPC stays available and is exercised by tests that build no System. No test claims one
system loaded ff14SB, Sage and TIP3P together: the generator has no combined protein-ligand route.

## 4. `--check` recomputes stage identity

`check_own_completion` accepted any nonempty `stage_config_sha256`. Present is not equal.

There is now **one** fingerprint implementation. `md_stages.stage_config_sha256` is the canonical
one and preflight imports it; a test asserts preflight carries no canonicalisation of its own. Two
subtly different hashes over "the stage request" is the failure being avoided — the run writes one
and the check compares another, so either every stage looks stale or every stale stage looks fine.

`--check` now, for the current stage: recomputes from the current `stage.yaml`, compares against
the record, requires status `completed` and the final-state artifact, and compares
`final_state.xml` against its recorded digest. For the parent: loads the parent's `stage.yaml` and
`resolved_stage.yaml`, recomputes and compares the parent fingerprint, requires completion and the
expected final state, and compares the handoff's digest against the bytes on disk.

## 5. The template commit is evidence

`_pinned_repository` only checked `^[0-9a-f]{40}$`. `generator_commit()` now establishes what the
running MD-templates actually is, from a Git checkout (`git rev-parse HEAD`) or PEP 610
`direct_url.json`, and `check_templates_commit()` requires `dataset.templates.commit` to equal it.
Both `sys-gen` and `md-gen` call it — `sys-gen` before the System is built, so a bad provenance
costs nothing.

Nothing is derived from a version, a branch or a date. When no exact commit can be established, the
generation is **refused**, because a pin that points at nothing is worse than none.

A dirty checkout is the one judgement call. Its HEAD *is* an exact commit, and refusing outright
would make the feature unusable for anyone developing — including this repository's own test suite.
So it pins HEAD, prints a `WARNING` naming the deviation, and `provenance.yaml` records
`git_dirty: true`. The pin is recorded, the deviation is recorded, nothing is invented.

Preflight compares the records that name a generator: `dataset.yaml`'s `templates.commit`, each
`stage.yaml`'s `template_commit`, and `provenance.yaml`'s `implementation.git_commit`. A dataset
whose system was built by one checkout and whose scripts were written by another has a single
recorded provenance that is true of only half of it.

## 6. Atom identity and order

`validate_topologies` compared atom names pairwise. A protein is full of repeated names — every
residue has an N, a CA, a C and an O — so a topology whose residues were reassigned or whose chains
were split differently compared equal while describing a different molecule. Parameters are
assigned per index, so that mismatch scales the wrong atoms with every count still agreeing.

`atom_identity()` now builds a per-index tuple of chain id, chain index, residue index, residue id,
residue name, atom name and element symbol, plus the bond set as sorted index pairs. The first
differing atom is reported with the specific fields that differ, and bond differences are reported
with counts and the first differing pair. Coordinates are deliberately not compared — the source
ensemble is *expected* to hold different configurations.

## 7. A reproducible validator install

`require_md_data` told users to `pip install git+ssh://git@github.com/csy0000/MD-data.git` — an
unpinned branch over SSH. Two people running that on the same day can validate against different
contracts.

The compatibility identity now lives in one place, `md_data_contract.MD_DATA_REPOSITORY` /
`MD_DATA_COMMIT` / `MD_DATA_CONTRACT_VERSION`, as HTTPS and an exact 40-hex commit. Everything that
installs, documents or reports the dependency reads it from there. `md-template install` attempts
it as part of creating the OpenMM environment and records the package version, the pinned commit,
import success and validator availability — a failure is a recorded warning, not a failed
installation, because `dataset.enabled` is off by default.

## Evidence

Environment: `/path/to/software/md-stack/envs/openmm-8.6.0` — conda-forge
`openmm 8.6.0=py312hdfcc665_0` (the release; its Python string reads `8.6.0.dev-c6173db`),
`cuda-version 13.0`, `mdtraj 1.11.1`, `md-data 0.2.0`, `openff-toolkit 0.19.0`, Python 3.12.14.

**CUDA devices used by every MD test: 0 (NVIDIA RTX A5000, 24564 MiB) and 1 (NVIDIA GeForce RTX
3080, 10240 MiB), driver 580.173.02.** No test used CPU or OpenCL; `conftest.pytest_collection_modifyitems`
fails GPU-marked tests rather than skipping them where no CUDA platform exists.

### Test commands and counts

```text
pytest tests/test_integrity_corrections.py -q -p no:randomly     32 passed
pytest tests/ -q -p no:randomly -k "not gpu"                    241 passed, 65 deselected
pytest -q -p no:randomly                                        338 passed, 0 failed, 0 skipped
                                                                (207.36s, 71 gpu-marked)
```

### 1. The source was not hashed

```text
trajectory: cMD/whole_system.dcd
trajectory_sha256: None
trajectory_not_hashed: the production source is never hashed at runtime;
                       MD-data hashes it once at archival
trajectory_bytes: 287796      n_frames: 120
loader: mdtraj.iterload       chunk_frames: 50
chunks_read_survey: 3         chunks_read_selection: 2
tau: 0.5                      tau_route: companion record
```

A GPU test additionally replaces `hashlib.sha256` with a guard that raises once more than 4 MB has
been fed to it, and asserts preparation still completes.

### 2. Physical truncation, header untouched

`observations.dcd`, 25% of the bytes removed from the end:

```text
bytes 50592 -> 37944; header NSET 21 -> 21 (UNCHANGED)
[AIS 0000] replacing an incomplete directory (observations.dcd holds 15 readable frame(s),
           21 expected (header claims 21) -- truncated, not short)
[AIS 0000] complete: 21 observations, 40 integration steps
[AIS 0001] already complete: ... Nothing was run.
```

`AIS/inputs/sources.dcd`, a third of the bytes removed:

```text
bytes 5068 -> 3378; header NSET 2 -> 2 (UNCHANGED)
[FAIL] AIS inputs  inputs/sources.dcd holds 1 readable frame(s), 2 expected (header claims 2)
                   -- truncated, not short
preflight failed: 1 check(s). Nothing was run, no Context was created, and no output was written.
```

The rerun restored 21 frames at the original byte length — replaced, not appended — and the intact
path was left alone.

### 3. Force-field identities

```text
peptide acceptance   BUILT protein amber14-all.xml
                     BUILT ligand  None (charges None)
                     BUILT water   amber14/tip3p.xml
                     WANTED        {protein: amber14-all.xml, water: amber14/tip3p.xml}
                     "amber19" anywhere in the record: False

ligand acceptance    BUILT protein None
                     BUILT ligand  openff-2.2.1 (charges am1bcc)
                     BUILT water   amber14/tip3p.xml
                     WANTED        {protein: null, water: amber14/tip3p.xml,
                                    ligand: openff-2.2.1, ligand_charge_method: am1bcc}
                     builder.openmm_xml_loaded: ['amber14/tip3p.xml']
                     "amber19" anywhere in the record: False
```

Refusals covered by test: a wrong protein resource, a wrong water resource, a ligand force field
claimed by a peptide system, a wrong OpenFF resource, a wrong charge method, a protein force field
claimed by a ligand system, a water model claimed by an implicit system, a SASA term that disagrees,
an absent expected field, and a route disagreement.

### 4. Stage mutation under `--check`

```text
unchanged chain
  [PASS] parent stage       nvt_1kcal completed (10 steps), request unchanged, handoff verified
  [PASS] completion record  complete and unchanged (07f49acc0587, final state verified)

parent stage.yaml temperature_kelvin 300 -> 310
  [FAIL] parent stage  nvt_1kcal/stage.yaml has changed since it ran: it now hashes to
                       91c027debaef but its completion record was written for 07f49acc0587

parent final_state.xml edited
  [FAIL] parent stage  nvt_1kcal's final_state.xml has changed since it was recorded
                       (1b541bc7d507 on disk, ffe862c92c5e recorded)

current stage.yaml duration_ps 0.02 -> 0.04
  [FAIL] completion record  stage.yaml has changed since this stage ran: it now hashes to
                            25d98ac74fd3 but the completion record was written for 07f49acc0587

restored
  7/7 ok
```

Every one ended `Nothing was run, no Context was created, and no output was written.`

### 5. Template provenance

Matching identity passes. A well-formed but different 40-hex value:

```text
dataset.templates.commit is 'bbbb...', but the MD-templates actually generating this dataset
is at 'aaaa...' (git checkout: ...). A 40-hex string that is not the generating commit is
worse than none: it records a provenance that can be checked out and will not reproduce this run.
```

An install with no establishable commit:

```text
this MD-templates cannot establish its own exact commit, so the claim cannot be verified.
Contract-managed generation refuses rather than record an unverified pin.
```

Preflight, on records that disagree: `[FAIL] template provenance  dataset.yaml records
templates.commit aaaa… but this stage's stage.yaml says bbbb…. The dataset and the scripts in it
were not generated by the same MD-templates`.

### 6. Topology mismatch

One residue reassigned with **every atom name and the atom order unchanged**: the name sequences
compare equal, the identity tuples do not, and the differing field is the residue index. Renaming a
residue changes OpenMM's inferred bonds and the bond sets differ. End to end, with
`AIS.source.topology` pointed at the reassigned PDB, `run.sh --check` fails with `disagree at atom
index …` before any frame selection or worker, and nothing is prepared.

### 7. Pinned validator

```text
requirement       md-data @ git+https://github.com/csy0000/MD-data.git@48628f9a5d3ace6c6398a63bc3905cd58d542de3
repository        https://github.com/csy0000/MD-data
commit            48628f9a5d3ace6c6398a63bc3905cd58d542de3
contract_version  1.0
installed         md-data 0.2.0, import_ok true, validator_available true
```

Tested without downloading: the constructed specification is inspected, and the failure path is
exercised with a mocked `subprocess.run`.

## Limitations

* **MD-data is not publicly installable.** `csy0000/MD-data` returns 404 unauthenticated (a public
  control repository returns 200) and `md-data` is not on PyPI. The pin is HTTPS and exact, which
  is the correct *form*, but the documented command cannot be run by someone without access to the
  repository. Making it public, or publishing a release, is the user's decision and not something
  this repository can do. Until then the documented install works only for authorised users.
* **A dirty checkout still pins its HEAD.** Judged the lesser evil against refusing all
  contract-managed generation from a working tree, but the recorded commit does not fully describe
  what ran. The warning and `provenance.yaml`'s `git_dirty` are the mitigation, not a fix.
* **`validate_generated_dcd` reads coordinates, not physics.** It catches truncation, unreadable
  records, wrong counts, non-finite values and degenerate boxes. It cannot tell a complete DCD of
  wrong coordinates from a correct one.
* **Preflight compares generator records; it cannot re-establish them.** A generated project
  deliberately cannot import `md_templates`, so if every record was written by the same wrong
  commit they agree and preflight passes. Establishing the commit is `sys-gen`/`md-gen`'s job.
* **No combined protein-ligand route exists**, so the acceptance matrix covers ff14SB + TIP3P and
  Sage 2.2.1 + am1bcc + TIP3P as separate systems. Nothing claims otherwise.
* **AIS work is not bit-reproducible on CUDA** (carried forward from the previous round): two
  identical reruns differ in the 8th significant figure through mixed-precision accumulation order.
* **The CUDA smoke is picoseconds.** It proves the generated scripts execute and the integrity
  checks fire. It is not a scientific validation of anything.
