# Provenance consistency and public contract-readiness correction

## Purpose

Close the small remaining gaps found after
`20260827_final-md-data-integrity-correction.md`. The scientific runtime and MD-data architecture
are already in place. This task is limited to provenance consistency, honest installation status,
one ineffective regression test, and documentation cleanup.

Work on the current `dev` branch of `csy0000/MD-templates`. The inspected starting head for this
instruction was `25382471b952ec3588a162ae6eb105079c4b443a`.

Read before editing:

- `CLAUDE.md`
- `claudecode-instructions/20260827_final-md-data-integrity-correction.md`
- `docs/journal/2026-08-27_final-md-data-integrity-correction.md`
- `src/md_templates/openmm/provenance_min.py`
- `src/md_templates/openmm/md_data_contract.py`
- `src/md_templates/openmm/sysgen.py`
- `src/md_templates/openmm/mdgen.py`
- `src/md_templates/openmm/templates/preflight.py`
- `src/md_templates/install/openmm.py`
- `tests/test_integrity_corrections.py`
- `claudecode-instructions/README.md`

Keep the current six-command interface, `generate_system()`, `generate_md()`, and the standalone
generated-project design. Do not redesign the repository, restore deleted machinery, add a new
public command, merge to `main`, or create/move a release tag.

Write the report to:

    docs/journal/2026-08-27_provenance-and-contract-readiness-correction.md

Commit and push the completed correction to `dev`.

## 1. Establish one canonical MD-templates implementation identity

The current code can establish the generating commit through `generator_commit()` using either:

- a clean Git checkout; or
- PEP 610 `direct_url.json` from a VCS installation.

However, other writers still use `implementation_identity()["git_commit"]` or
`package_provenance()["md_templates"]["git_commit"]`. In an installed VCS package those values can
be null even though `generator_commit()` correctly established the exact commit from
`direct_url.json`.

Create one canonical resolved template-identity object and use it everywhere contract-managed
generation records the implementation. It must contain at least:

- repository;
- exact 40-hex commit;
- evidence route (`git checkout` or `direct_url.json`);
- package version;
- installed fingerprint;
- dirty status where applicable.

Do not let `sysgen.py`, `mdgen.py`, `_template_commit()`, stage generation, method generation,
`provenance.yaml`, `resolved_sys.config.yaml`, `md.config.yaml`, or `dataset.yaml` independently
derive the commit.

For contract-managed generation, propagate the exact same established commit into every record
that names the generator, including:

- `dataset.yaml: templates.commit`;
- root and `common/` provenance records;
- `resolved_sys.config.yaml`;
- `md.config.yaml`;
- every generated `stage.yaml`;
- every cMD, REST2, and AIS configuration/provenance record that carries template identity.

For unregistered local generation, keep the existing installed fingerprint and version evidence.
An exact commit may remain unavailable there, but it must not be presented as contract-verified.

Do not infer a commit from a package version, branch name, date, user input alone, or an installed
fingerprint.

## 2. Refuse dirty checkouts for contract-managed generation

A dirty checkout is not reproducible from its `HEAD` commit. The code currently accepts it and emits
a warning, while the function documentation says it is refused.

For `dataset.enabled: true`:

- require a clean checkout when the Git route is used;
- if the checkout is dirty, fail before building the System or generating MD files;
- name the changed-state problem clearly and tell the user to commit/stash the changes or use
  unregistered local generation;
- never write `HEAD` as though it exactly described modified code.

For `dataset.enabled: false`, development from a dirty checkout may continue. Record:

- `git_commit`;
- `git_dirty: true`;
- installed fingerprint;
- a clear statement that the commit alone does not reproduce this unregistered generation.

Make the implementation, docstrings, README, and tests agree on this policy.

## 3. Make preflight require every applicable provenance record

The current preflight compares only provenance fields that happen to be present. A missing
`stage.yaml.template_commit` or missing `provenance.yaml.implementation.git_commit` can therefore
be silently ignored.

For a contract-managed project, preflight must fail when any applicable generated record omits its
template identity. It must compare the exact established commit across:

- `dataset.yaml`;
- project-level `provenance.yaml`;
- `common/provenance.yaml`;
- `resolved_sys.config.yaml`;
- `md.config.yaml`;
- the current equilibration/minimization `stage.yaml`;
- the current cMD, REST2, or AIS resolved method configuration.

Use the canonical recorded identity regardless of whether it originally came from Git or
`direct_url.json`. If a provenance record contains both `git_commit` and
`direct_url.vcs_info.commit_id`, they must not disagree.

Keep generated scripts standalone: they compare the records carried inside the generated project
and do not import `md_templates` or inspect the original checkout at runtime.

Add end-to-end generated-project tests for:

1. clean Git identity propagated everywhere;
2. VCS `direct_url.json` identity propagated everywhere when `git_commit` is unavailable;
3. one missing identity field fails preflight;
4. one mismatching identity field fails preflight;
5. dirty contract-managed generation fails before expensive system construction;
6. dirty unregistered generation remains permitted but explicitly records the limitation.

Do not satisfy these tests only by monkeypatching the final comparison function. Verify the files
actually generated.

## 4. Make MD-data installation status honest and enforceable

MD-data is currently pinned correctly by HTTPS and exact commit, but unauthenticated access may
still fail if the repository remains private. Repository visibility is not something this task is
authorized to change.

Do not make the basic OpenMM installation unusable for researchers who only want unregistered local
simulation. Instead, distinguish these outcomes explicitly:

- OpenMM runtime ready;
- MD-data contract support ready;
- MD-data contract support unavailable.

The install record and console summary must never call contract support ready unless all are true:

- installation of the exact pinned requirement succeeded;
- `md_data` imports;
- the public validator functions exist;
- contract version matches the supported version;
- installed VCS provenance or package metadata proves the expected pinned release/commit.

Do not merely echo the intended commit as though it described the installed package. Inspect
`direct_url.json` or equivalent installed-distribution metadata and compare the actual installed
source commit.

If validator installation or verification fails:

- the base OpenMM installation may finish;
- print a prominent, actionable warning that contract-managed generation is unavailable;
- record `contract_support_ready: false` and the exact reason;
- preserve the existing early hard failure when `dataset.enabled: true`, before system building;
- do not state that the installation delivered all contract functionality.

If an existing flag or minimal option can require contract support without changing the command
architecture, make that mode fail nonzero when MD-data is unavailable. If adding such a flag would
meaningfully complicate the simple CLI, do not add it; keep the capability state explicit and let
`dataset.enabled: true` remain the hard gate.

Check unauthenticated availability of the exact pinned HTTPS source during acceptance. If it is
still inaccessible, do not attempt to change `csy0000/MD-data` visibility and do not report public
contract readiness as PASS. Report this as the single external blocker and explain that making
MD-data public or publishing a package release is the required owner action.

## 5. Replace the ineffective source-hashing regression test

The current test raises only after more than 4 MB is fed to SHA-256, while its AIS source
trajectory is much smaller. It would therefore still pass if the source were accidentally hashed.

Replace it with a path-specific runtime guard:

- make any attempt to pass the original production source trajectory to the hashing helper fail
  immediately, regardless of file size;
- allow hashing of named small prepared-system/provenance files;
- run AIS preparation through bounded `mdtraj.iterload`;
- prove the production-source hashing guard was installed and never triggered;
- prove the selected source frames were still prepared correctly.

Prefer importing the generated AIS module and monkeypatching its `sha256_file` function with a guard
that rejects the exact resolved source path. If subprocess isolation makes that impractical, use an
equally path-specific instrumented helper. Do not rely on a byte threshold or source-code string
inspection alone.

## 6. Clean the instruction index and stale documentation

In `claudecode-instructions/README.md`, the previous instruction currently appears twice—once
pending and once executed. Keep exactly one executed entry for:

    20260827_final-md-data-integrity-correction.md

Add this instruction exactly once as pending while it is being executed, then mark that same entry
executed rather than appending another copy.

Correct stale documentation in `AIS/path_is_complete()`: it now validates every small generated
DCD frame with bounded `mdtraj.iterload`; it no longer reads only the header.

Review README claims about MD-data installation. They must distinguish a pinned specification from
a verified installed package and must disclose whether unauthenticated external installation is
currently possible.

Do not rewrite historical journals. Add a correction in the new journal where an earlier claim
requires qualification.

## Acceptance and test budget

This task does not alter MD equations, force-field construction, integration schedules, REST2
exchange, or AIS work accumulation. Do not rerun nanosecond simulations, replica-exchange
campaigns, or the previous full GPU campaign.

Run:

1. targeted provenance, installer, AIS hashing-guard, and generated-preflight tests;
2. the normal non-GPU test suite once;
3. no MD integration unless a changed code path cannot be verified otherwise.

If any MD integration is genuinely necessary, use the smallest existing CUDA smoke only. Every MD
step must use OpenMM CUDA, never CPU or OpenCL. Report the exact device. Pure generation,
provenance, installer, parser, and `--check` tests do not count as MD integration.

The targeted acceptance must include:

- clean Git contract mode: pass;
- dirty Git contract mode: fail before system construction;
- dirty unregistered mode: pass with explicit non-reproducibility evidence;
- VCS-installed/direct-URL contract identity: identical commit in every generated record;
- missing and mismatching provenance record: preflight failure;
- actual installed MD-data commit/release verification;
- inaccessible MD-data: contract readiness false and contract generation hard failure;
- path-specific proof that the production AIS source was never hashed;
- one and only one entry per current instruction in the index;
- corrected DCD validation documentation.

## Required report

Do not return overall PASS merely because the unit tests pass.

Report:

- commit SHA pushed to `dev`;
- exact tests and pass/fail/skip counts;
- the canonical template-identity fields and where they are written;
- clean, dirty, and direct-URL provenance results;
- missing/mismatching preflight results;
- installed MD-data version and actual verified source commit, if available;
- unauthenticated accessibility result for the exact MD-data pin;
- whether public contract support is ready;
- path-specific no-hashing evidence;
- documentation/index cleanup;
- remaining external blocker, if MD-data is still inaccessible.

Continue through routine implementation choices without stopping for questions. If MD-data
visibility is the only unresolved issue, finish and push every repository-local correction, report
the blocker, and do not mislabel the result as full public readiness.
