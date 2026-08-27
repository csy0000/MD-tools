# Finalize dry-run reporting and MD-data validation record shape

## Purpose

Close two small acceptance gaps remaining after
`20260827_installer-capability-and-hash-guard-correction.md`:

1. the real `md-template install --dry-run` path returns before constructing or displaying the
   MD-data dry-run record, so the implemented “not evaluated” output is unreachable;
2. `validate_existing()` stores the direct `verify_md_data()` result, whose key set differs from
   the canonical record stored after installation.

This is a small installer data-shape and CLI correction. Do not modify scientific runtime code,
OpenMM system generation, force fields, MD stages, cMD, REST2, AIS, provenance contracts, or the
six-command architecture.

Work on the current `dev` branch of `csy0000/MD-templates`. The inspected starting head was
`a9e381b522f79c984ac5248a5cd158b95395a539`.

Read before editing:

- `CLAUDE.md`
- `claudecode-instructions/20260827_installer-capability-and-hash-guard-correction.md`
- `docs/journal/2026-08-27_installer-capability-and-hash-guard-correction.md`
- `src/md_templates/install/openmm.py`
- `src/md_templates/cli/md_template.py`
- `tests/test_installer_capability.py`

Write the report to:

    docs/journal/2026-08-27_dry-run-and-validation-shape-finalization.md

Commit and push the correction to `dev`. Do not merge into `main` and do not create or move a
release tag.

## 1. Make the actual install dry-run report MD-data as not evaluated

At present, `install_md_data(dry_run=True)` can construct the correct record, and
`_report_md_data()` can print “not evaluated”, but the real command does neither:

- `install_openmm(dry_run=True)` returns before calling `install_md_data()`;
- `cmd_install()` returns before calling any MD-data reporter.

Fix the production control flow so:

    md-template install -e openmm -ev 8.6.0 --dry-run

prints an explicit line equivalent to:

    md-data contract: not evaluated (dry run)

Requirements:

- call `install_md_data(prefix, dry_run=True)` or an equivalent single source of truth;
- return the structured dry-run `md_data` record and capability summary from
  `install_openmm()`;
- display it from the real `cmd_install()` dry-run path;
- do not claim `ready` or `UNAVAILABLE`;
- do not invoke pip, create the environment, validate OpenMM, import MD-data, or write an installed
  environment record to `machine.yaml`;
- retaining the existing command log is fine.

Add a direct command-path test. Mock only the package-manager lookup or installer boundary needed
to prevent execution; exercise `cmd_install()` or `main()`, capture stdout, and prove it contains
“not evaluated (dry run)” and contains neither “ready” nor “UNAVAILABLE”.

Also test that the dry-run result returned by `install_openmm()` contains the structured
`md_data` and `capabilities` fields.

## 2. Give validation exactly the canonical MD-data record shape

The canonical result is defined by `_md_data_record()`. After this correction, the `md_data`
dictionary persisted by installation and by `validate_existing()` must have identical key sets.

Do not create a second list of record keys in production.

Preferred fix:

- make `verify_md_data()` return a result constructed through `_md_data_record()`;
- for validation, use a neutral installation-attempt value such as `attempted: null`, not
  `attempted: false`, because false is reserved by the CLI for an unevaluated dry run;
- set `installed` from observable evidence such as successful distribution metadata or import;
- keep `command` and `returncode` null during validation because validation did not run pip;
- retain all existing pin, observed source, commit verification, contract readiness, reasons,
  probe and error information;
- when `install_md_data()` calls the verifier after a successful pip command, overlay only the
  operation-specific fields: `attempted: true`, the pip command, return code and installation
  result.

The exact implementation may differ, but it must preserve these semantics:

| Situation | attempted | readiness | CLI meaning |
|---|---:|---:|---|
| install dry-run | false | null | not evaluated |
| real install | true | true/false | ready/unavailable |
| validate existing | null | true/false | ready/unavailable |

Do not make validation appear to be a dry run.

Strengthen the tests so they compare the complete key sets, not selected fields:

1. obtain or construct one canonical successful-install record;
2. obtain a validation record;
3. assert `set(install_record) == set(validation_record)`;
4. round-trip the validation record through `machine.yaml`;
5. assert `attempted is None`, `command is None`, and the actual readiness and reasons survive;
6. call the CLI reporter on that validation result and prove it reports ready or unavailable, never
   “not evaluated”.

## 3. Documentation and index

Append a concise correction section to the new journal explaining:

- why the dry-run reporting branch was previously unreachable;
- the canonical MD-data record semantics for dry-run, installation and validation;
- exact focused tests and results;
- confirmation that no scientific runtime or GPU execution occurred.

Do not rewrite historical journals.

Add this instruction exactly once to `claudecode-instructions/README.md`, initially as pending.
When the work is complete, replace that same entry with one executed entry. Do not create a
pending/executed duplicate.

## Acceptance and test budget

Run only:

1. focused installer dry-run tests;
2. focused CLI-output tests;
3. focused record-key-set and `machine.yaml` round-trip tests;
4. the normal non-GPU unit suite once if the focused tests pass.

Do not run GPU tests, AIS runtime tests, MD simulations, REST2, environment creation, pip
installation, or external repository probes. This correction contains no scientific behavior.

## Required report

Do not return PASS until the report includes:

- pushed commit SHA;
- exact files changed;
- exact test commands and pass counts;
- captured real command-path dry-run output;
- dry-run structured `md_data` and capability fields;
- proof that installation and validation records have identical key sets;
- persisted validation fields from `machine.yaml`;
- confirmation that validation is not reported as a dry run;
- confirmation that no GPU, MD, installation, or external-network test was run;
- confirmation that no scientific runtime file changed.

Continue through routine implementation choices without stopping for questions. Inspect the final
diff for scope creep, commit, and push to `dev`.
