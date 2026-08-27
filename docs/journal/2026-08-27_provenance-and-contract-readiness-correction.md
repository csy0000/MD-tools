# 2026-08-27 — provenance consistency and contract-readiness correction

Instruction: `claudecode-instructions/20260827_provenance-and-contract-readiness-correction.md`.
Branch `dev`. No merge to `main`, no tag. Six commands unchanged, scientific runtime untouched.

Two themes. **One identity, everywhere** — the generating commit was being re-derived by each
writer, and in a VCS-installed package half of them got null. And **honest status** — the previous
round recorded the MD-data pin it *intended* and called that verification.

## 1. One canonical template identity

`provenance_min.template_identity()` is now the single resolution of "which MD-templates is this".
It returns repository, exact 40-hex commit, evidence route, version, installed fingerprint, dirty
status, and — added here — `reproducible_from_commit` with a plain-language `reproducibility`
statement.

Two evidence routes, in order of directness: a Git checkout (`git rev-parse HEAD`, with `dirty`
saying whether the tree matches it), then PEP 610 `direct_url.json` (which pip writes for a VCS
install, where there is no checkout at all). A wheel built from a tarball has neither and `commit`
is `None` rather than guessed.

`sysgen`, `mdgen`, `_template_commit()`, stage generation, method generation and
`md_data_contract.generator_commit()` all read it. A test scans `sysgen.py` and `mdgen.py` for any
remaining `["git_commit"]` access outside a comment.

The same established commit is now written into every record that names a generator:

| record | field |
|---|---|
| `dataset.yaml` | `templates.commit` |
| `provenance.yaml` (project root) | `template.commit` |
| `common/provenance.yaml` | `template.commit` |
| `resolved_sys.config.yaml` | `provenance.template.commit` |
| `md.config.yaml` | `provenance.template_commit` |
| every `stage.yaml` | `template_commit` |
| `path_definition.yaml` and the other method records | `template_commit` |

`resolved_sys.config.yaml` did not carry it at all before; `md-gen` reads that file back, so the
prepared system and the scripts written from it can now be shown to come from the same generator.

## 2. Contract-managed generation refuses a dirty checkout

Previously it warned and continued, while the function's own docstring said it refused. The
docstring was right and the code was wrong.

A working tree with uncommitted changes is not reproducible from its HEAD: someone who checks that
commit out gets different code. For a dataset that will be registered, archived and cited, writing
that commit as `templates.commit` is a **false** provenance rather than an imprecise one — so
contract-managed generation now stops, before the System is built, and says to commit, stash, or
use unregistered generation.

Unregistered development from a dirty tree is unchanged and deliberately easy. It records
`git_dirty: true`, `reproducible_from_commit: false` and the sentence *"the working tree has
uncommitted changes, so checking out … does NOT reproduce this generation; the installed
fingerprint identifies what ran"*, and `sys-gen` prints a `NOT REPRODUCIBLE` line while it works.

### A correction to the previous journal

`docs/journal/2026-08-27_final-md-data-integrity-correction.md` describes accepting a dirty
checkout as "the lesser evil against refusing all contract-managed generation from a working tree",
and says so partly because the repository's own test suite could not otherwise build a
contract-managed fixture. That reasoning was wrong in its conclusion. The fixture problem is a
*test* problem and had a test solution: generate in-process and present the raw identity
observation as clean. It should not have been allowed to weaken a provenance rule. That earlier
entry stands as written — journals record what was done — and this paragraph is the correction.

## 3. Preflight requires every applicable record

It previously compared only fields that happened to be present, so deleting one was enough to pass.
For a contract-managed project every record in the table above is now required: a missing file, an
unreadable one, or an absent field each fail, naming which. A record carrying both `git_commit` and
`direct_url.vcs_info.commit_id` must not disagree with itself.

Unregistered projects are skipped, and say they are skipped — there is no contract identity to hold
the other records to.

Generated projects stay standalone: this compares records carried inside the project and imports
nothing from `md_templates`.

## 4. Installation status is verified, and three-state

`verify_md_data()` reads the **installed distribution's** `direct_url.json` inside the target
environment and compares the actual source commit against the pin. Contract support is called ready
only when the requirement installed, `md_data` imports, both validator entry points exist, the
contract version matches, and the installed source commit is proved to be the pinned one.

The install now reports three states rather than one: OpenMM runtime ready, MD-data contract
support ready, MD-data contract support unavailable. A researcher doing unregistered local
simulation has a working installation regardless; a failure prints a prominent warning, records
`contract_support_ready: false` with the exact reason, and does not fail the OpenMM install.
`dataset.enabled: true` remains the hard gate and still fails before a system is built.

## 5. The source-hashing test was replaced

The previous guard raised only after 4 MB had been fed to SHA-256, while the fixture's source
trajectory is ~288 kB. It would have passed with the source hashed on every run — worse than no
test, because it reported a guarantee it never checked.

The replacement rejects the **exact resolved source path** regardless of size, allows the named
small prepared-system files to hash normally, and writes a sentinel so a guard that failed to
install is distinguishable from one that was never triggered. A companion unit test proves the
guard fires when the forbidden path is passed.

## 6. Index and stale documentation

The duplicate index entry had already been removed in `97024e4`; each instruction now appears
exactly once, and two tests keep it that way. `path_is_complete()`'s docstring still said only the
DCD's 100-byte header was read — untrue since the previous round — and now describes the bounded
`iterload` frame reading it actually does.

## Evidence

Environment: `/path/to/software/md-stack/envs/openmm-8.6.0` — conda-forge
`openmm 8.6.0=py312hdfcc665_0`, `cuda-version 13.0`, `mdtraj 1.11.1`, `md-data 0.2.0`,
Python 3.12.14.

### Tests

```text
pytest tests/test_template_provenance.py -q -p no:randomly        21 passed
pytest tests/test_integrity_corrections.py -q -p no:randomly -m "not gpu"
                                                                  31 passed, 6 deselected
pytest tests/ -q -p no:randomly -m "not gpu"                     293 passed, 71 deselected
pytest tests/test_integrity_corrections.py -q -p no:randomly -m gpu
                                                                   6 passed, 31 deselected
```

The GPU run is the smallest CUDA smoke: the picosecond AIS fixture and the preflight/truncation
tests that need a real run. **Device 0, NVIDIA RTX A5000, driver 580.173.02**; the generated records
confirm `platform: CUDA` for both minimisation and cMD. No nanosecond runs, no REST2 campaign, no
repeat of the previous full GPU campaign.

### Provenance routes

| route | result |
|---|---|
| clean Git checkout | `commit = HEAD`, `evidence = git checkout`, `dirty = false`, `reproducible_from_commit = true`; the same commit appears in all 8 records of a generated contract project |
| `direct_url.json`, no checkout | `commit` from `vcs_info.commit_id`, `evidence = direct_url.json`, `dirty = false`; the same commit appears in all 8 records — the case that previously wrote nulls into every `stage.yaml` |
| dirty Git checkout, contract | **refused**, `UNCOMMITTED CHANGES`, before construction: no `system.xml`, no `topology.pdb`, no `dataset.yaml` on disk; refusal took 1.6 s |
| dirty Git checkout, unregistered | permitted; `system.xml` built; every record carries `dirty: true`, `reproducible_from_commit: false` and the "does NOT reproduce" statement; `contract_managed: false` |
| no establishable commit | `commit = None`, `evidence = unavailable`; contract generation refused; version and fingerprint still recorded and never promoted to a commit |

### Missing and mismatching records

Missing (each fails with *"contract-managed, so every generated record must name the generator"*):
`provenance.yaml → template.commit`, `md.config.yaml → provenance.template_commit`,
`minimization/stage.yaml → template_commit`, and a deleted `common/provenance.yaml` (*"does not
exist"*).

Mismatching (each fails with *"not generated by the same MD-templates"*): `dataset.yaml`,
`provenance.yaml`, `minimization/stage.yaml`. A record whose own `git_commit` and
`direct_url.vcs_info.commit_id` differ fails with *"disagree"*.

### Installed MD-data

```text
installed version   0.2.0
contract version    1.0        (matches the targeted 1.0)
import_ok           true
validator_available true
installed source    file:///…/scratchpad/MD-data
source_kind         local directory
installed_commit    null
commit_verified     FALSE
contract_support_ready  FALSE
reason  the installed md-data records no source commit (installed from local directory: …),
        so it cannot be shown to be the pinned 48628f9a5d3a
```

The validator works; what cannot be shown is that it is the pinned contract. That is the honest
result and it is reported as unavailable rather than ready.

### Anonymous accessibility of the pin

```text
git ls-remote https://github.com/csy0000/MD-data.git   fatal: could not read Username (no credentials)
GET /repos/csy0000/MD-data                             404
GET /repos/csy0000/MD-data/commits/48628f9a…           404
GET /repos/openmm/openmm            (control)          200
GET pypi.org/pypi/md-data/json                         404
```

**Public contract support is NOT ready.** The visibility of `csy0000/MD-data` was not changed.

### No-hashing evidence

Path-specific: a guard rejecting the exact resolved `cMD/whole_system.dcd` regardless of size. The
sentinel confirms it installed and reads `installed` (not `TRIGGERED`) after a full AIS preparation
run; the prepared inputs still hold 2 configurations, `loader: mdtraj.iterload`,
`trajectory_sha256: null`. The companion unit test confirms the guard raises when the forbidden
path is passed and lets a named small file hash.

### Index

Each `2026*.md` instruction appears at most once in `claudecode-instructions/README.md`;
`20260827_final-md-data-integrity-correction.md` and this instruction appear exactly once. Two
tests enforce it.

## Remaining external blocker

**`csy0000/MD-data` is private and `md-data` is not published.** Every repository-local correction
is complete: the pin is HTTPS and exact, the installer verifies the installed distribution rather
than the intention, and unavailability is reported honestly in three states. What no change here
can fix is that an unauthenticated user cannot install the validator. The required owner action is
to make the repository public or publish a package release. Until then, contract-managed generation
is available only to users with access, and this repository reports that rather than claiming
public readiness.
