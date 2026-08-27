# Installer capability reporting and AIS hash-guard correction

## Purpose

Fix the final two repository-local gaps found after
`20260827_provenance-and-contract-readiness-correction.md`:

1. `md-template install` computes MD-data contract readiness but neither displays it to the user nor
   persists it in `machine.yaml`;
2. the path-specific AIS no-hashing test patches `builtins.open`, while the generated
   `sha256_file()` uses `pathlib.Path.open`, so the test guard does not intercept the operation it
   claims to test.

This is a small reporting and test-correctness task. Do not modify MD equations, force fields,
system generation, integration, REST2, AIS work accumulation, the six-command architecture, or the
MD-data contract design.

Work on the current `dev` branch of `csy0000/MD-templates`. The inspected starting head was
`ae03edd6934539360013697b1d6f8a69e8c80ea7`.

Read before editing:

- `CLAUDE.md`
- `claudecode-instructions/20260827_provenance-and-contract-readiness-correction.md`
- `docs/journal/2026-08-27_provenance-and-contract-readiness-correction.md`
- `src/md_templates/install/openmm.py`
- `src/md_templates/cli/md_template.py`
- `src/md_templates/openmm/templates/ais_run.py`
- `tests/test_integrity_corrections.py`
- relevant installer/CLI tests

Write the report to:

    docs/journal/2026-08-27_installer-capability-and-hash-guard-correction.md

Commit and push the completed correction to `dev`. Do not merge into `main` and do not create or
move a release tag.

## 1. Give every MD-data installation outcome one complete readiness record

`install_md_data()` must always return a complete, predictable capability result, including when:

- the pinned pip installation succeeds and verifies;
- pip installation fails;
- the package imports but lacks validator functions;
- its contract version disagrees;
- its installed source commit is absent;
- its installed source commit differs from the pin;
- dry-run mode is used.

At minimum every non-dry-run result must contain:

- `installed`;
- `contract_support_ready`;
- `reasons`;
- pinned repository, commit, contract version and requirement;
- attempted command and return code where applicable;
- actual installed version, source kind, source URL and source commit when observable;
- `commit_verified`;
- raw installation error when pip fails.

On pip failure, set:

- `installed: false`;
- `contract_support_ready: false`;
- a nonempty `reasons` list containing the actionable installation failure;
- `commit_verified: false`.

Do not leave failure reasons only in an unrelated `error` field while the user-facing capability
record contains an empty reason list.

For dry-run, state that readiness is not evaluated. Do not report either success or failure as
though installation occurred.

Keep the current policy:

- a failed MD-data installation does not invalidate an otherwise functional OpenMM environment for
  unregistered local simulations;
- `dataset.enabled: true` remains the hard gate and fails before expensive system construction;
- private MD-data remains an external accessibility blocker and this task must not change repository
  visibility.

## 2. Display MD-data readiness prominently in the CLI

Extend the existing `_report_environment()` output without adding a new command.

After the OpenMM platform/runtime lines, print one explicit capability line:

    md-data contract: ready

or:

    md-data contract: UNAVAILABLE

When unavailable, print every reason immediately below it. Include the pinned commit and, when
available, the actual installed version and source commit. The console must make it impossible to
mistake “OpenMM installed successfully” for “contract-managed MD-data generation is ready”.

Do not bury this information only in an installation log.

Test the CLI output directly by passing representative ready and unavailable result dictionaries to
`_report_environment()` and capturing stdout. Require the words `UNAVAILABLE` and the exact reason
in the failure case.

## 3. Persist capability state in `machine.yaml`

`record_environment()` currently selects individual OpenMM fields and drops the computed
`md_data` and `capabilities` dictionaries.

Persist under `installed.openmm`:

- `md_data`: the complete structured MD-data installation/verification result;
- `capabilities.openmm_runtime_ready`;
- `capabilities.md_data_contract_support_ready`;
- `capabilities.md_data_unavailable_reasons`.

Ensure all values are YAML-serializable. Do not store subprocess objects or exception instances.

Add a round-trip test:

1. write a temporary initialized stack;
2. call `record_environment()` with an unavailable MD-data result;
3. reload `machine.yaml`;
4. prove the unavailable state, pinned commit and exact reason survived;
5. repeat or parameterize for a ready result.

Also ensure `validate_existing()` records and reports the same capability structure when it
evaluates an existing environment. Do not let installation and later validation produce
incompatible `machine.yaml` shapes.

## 4. Correct the path-specific AIS hashing guard

The current subprocess guard replaces `builtins.open`, but generated
`ais_run.py::sha256_file()` uses:

    Path(path).open("rb")

`pathlib.Path.open()` delegates through `io.open`, so replacing only `builtins.open` does not prove
that the production source would be rejected if passed to the real helper.

Change the test guard to intercept the actual path used by the implementation. Preferred fix:

- wrap `pathlib.Path.open` for the exact resolved production source path;
- allow the same source to be opened normally by `mdtraj.iterload`;
- reject it only when the call originates from `sha256_file`;
- continue allowing named small system/provenance files to be hashed;
- restore or isolate the patch so it cannot leak to other tests.

An equivalent `io.open` wrapper is acceptable if it is equally path-specific and demonstrably
intercepts `Path.open`.

The same injected subprocess guard must prove both sides:

1. a deliberately invoked `sha256_file(forbidden_source)` triggers the guard and writes
   `TRIGGERED`;
2. after resetting the sentinel, a real AIS preparation succeeds through `mdtraj.iterload`, leaves
   the sentinel at `installed`, produces the selected `sources.dcd`, and records
   `trajectory_sha256: null`.

Do not substitute a source-code string test, byte threshold, unrelated in-process wrapper, or a
separate guard implementation for this evidence. The test must demonstrate that the exact guard
used during the real subprocess would catch the exact helper used by the generated runtime.

If importing the generated AIS runtime is necessary, use the existing fixture/stub mechanism. Do
not add production-only test hooks to the generated script.

## 5. Documentation and instruction index

Add a short correction to the new journal explaining:

- why `builtins.open` did not intercept `Path.open`;
- what the replacement guard intercepts;
- the exact CLI and `machine.yaml` readiness evidence;
- that MD-data remains inaccessible anonymously unless the repository is made public or a package
  is published.

Do not rewrite historical journals.

Add this instruction exactly once to `claudecode-instructions/README.md`, and mark that same entry
executed when complete. Do not append a duplicate pending/executed pair.

## Acceptance and test budget

This task requires no scientific MD validation.

Run only:

1. targeted installer result-shape tests;
2. targeted CLI-output tests;
3. targeted `machine.yaml` round-trip tests;
4. the path-specific hashing-guard trigger test;
5. the existing tiny AIS preparation test only if needed to demonstrate the guard during a real
   generated runtime;
6. the normal non-GPU suite once.

Do not run nanosecond simulations, REST2, the previous full GPU suite, or unrelated installation
campaigns. If the tiny AIS runtime test performs MD, it must use OpenMM CUDA, never CPU or OpenCL,
and the exact device must be reported.

## Required report

Do not return PASS until the report includes:

- pushed commit SHA;
- exact test commands and counts;
- ready and unavailable CLI output;
- the corresponding persisted `machine.yaml` fields;
- pip-failure result showing nonempty reasons;
- proof that the exact subprocess guard triggers on the generated `sha256_file`;
- proof that real AIS source preparation does not trigger it;
- confirmation that no scientific runtime behavior was changed;
- the remaining external MD-data visibility blocker.

Continue through routine implementation choices without stopping for questions. Fix applicable
failures, inspect the final diff for scope creep, commit, and push to `dev`.
