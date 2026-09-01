# 2026-08-27 — dry-run reporting and MD-data record-shape finalisation

Instruction: `claudecode-instructions/20260827_dry-run-and-validation-shape-finalization.md`.
Branch `dev`. No merge to `main`, no tag. **No scientific runtime file changed. No GPU test, MD
run, environment creation, pip install or network probe was performed.**

Two gaps, both left by the previous round, both of the same kind: the code computed the right
answer and the user could not get at it.

## 1. The dry-run branch was unreachable

Last round added a `not evaluated (dry run)` record to `install_md_data()` and a printer for it in
`_report_md_data()`. Neither was reachable from the real command:

* `install_openmm(dry_run=True)` wrote its log and returned `{"prefix", "log", "dry_run": True}`
  **before** `install_md_data()` was ever called;
* `cmd_install()` printed `dry run: nothing was installed` and returned **before** any MD-data
  reporter ran.

So two pieces of correct code existed and the command joining them did not. This is the same shape
of defect as the `builtins.open` guard from the round before: something that looked verified because
its parts were tested individually. The test added here therefore drives `main()` — "the code can
print this" and "the command prints this" are different claims, and only the second matters.

`install_openmm()` now builds the record from the one source of truth on the dry-run path and
returns it with a capability summary; `cmd_install()` prints it. The dry run still runs no pip,
creates no environment, imports no MD-data and writes no installed-environment record. The test
makes `subprocess.run` raise, so any execution at all fails rather than passing quietly.

## 2. Validation and installation produced different shapes

`validate_existing()` stored the raw `verify_md_data()` result. Measured against the canonical
record:

```text
canonical-only: ['attempted', 'command', 'error', 'installed', 'note', 'returncode']
verify-only   : ['probe']
identical     : False
```

Six keys present after installation and absent after validation, and one the other way. A
`machine.yaml` reader would have had to know which operation produced the record.

`verify_md_data()` now builds through `_md_data_record()` — the single production definition — and
`probe` joined the canonical shape, since validation always has one. `install_md_data()` overlays
only what the pip *operation* establishes: `attempted`, `command`, `returncode`, `installed`.

The three semantics, which are deliberately distinct:

| situation | `attempted` | readiness | what the CLI says |
|---|---|---|---|
| install dry-run | `False` | `None` | not evaluated (dry run) |
| real installation | `True` | `True`/`False` | ready / UNAVAILABLE |
| validate existing | `None` | `True`/`False` | ready / UNAVAILABLE |

`attempted: null` rather than `false` for validation is the point: `false` is what the CLI reads as
"not evaluated", and validation *did* evaluate — it simply did not run pip, which is why `command`
and `returncode` stay null. Validation must not look like a dry run, and a test asserts the
reporter never says "not evaluated" for a validation record.

## Evidence

### The real command path

```text
$ md-template install -e openmm -ev 8.6.0 --dry-run --target-dir <stack>
  environment   : <stack>/envs/openmm-8.6.0
  log           : <stack>/logs/install-openmm-20260827T185830Z.log
  dry run: nothing was installed
  md-data contract: not evaluated (dry run)
```

Exit 0. `"ready"` absent, `"UNAVAILABLE"` absent, `installed.openmm` **not** written to
`machine.yaml`, `envs/openmm-8.6.0` **not** created, and `subprocess.run` patched to raise was
never called.

Returned structure:

```text
result["dry_run"]                                True
result["md_data"]["attempted"]                   False
result["md_data"]["contract_support_ready"]      None
result["md_data"]["reasons"][0]                  "dry run: ... readiness was not evaluated"
result["capabilities"]["md_data_contract_support_ready"]   False
set(result["md_data"]) == canonical              True
```

### One key set

```text
install keys == validation keys : True
both == canonical               : True

attempted, command, commit, commit_verified, contract_support_ready, contract_version,
error, installed, installed_commit, installed_source, installed_version, note, probe,
reasons, repository, requirement, returncode, source_kind
```

```text
semantics    attempted  readiness
  dry-run    False      None
  install    True       False
  validate   None       False
```

### Persisted validation record

```yaml
installed:
  openmm:
    md_data:
      attempted: null            # not false: this was evaluated, it just did not run pip
      command: null
      returncode: null
      installed: true            # observable: it imported
      commit: 48628f9a5d3ace6c6398a63bc3905cd58d542de3
      installed_version: 0.2.0
      source_kind: local directory
      commit_verified: false
      contract_support_ready: false
      reasons: ['the installed md-data records no source commit …']
    capabilities:
      md_data_contract_support_ready: false
      md_data_unavailable_reasons: ['the installed md-data records no source commit …']
```

Speaking that persisted record back through the reporter produces `md-data contract: UNAVAILABLE`,
never `not evaluated`.

### Tests

```text
pytest tests/test_installer_capability.py -q -p no:randomly            24 passed
pytest tests/test_installer_capability.py tests/test_install.py \
       tests/test_cli.py -q -p no:randomly                             50 passed
pytest tests/ -q -p no:randomly -m "not gpu"                          318 passed, 71 deselected
```

No GPU test, no AIS execution, no MD integration, no REST2, no environment creation, no pip
installation, no external repository probe. Every subprocess in the focused tests is either
patched to raise or replaced with a stub result.

One note on a transient: run with `CUDA_VISIBLE_DEVICES=""`, two pre-existing `test_install.py`
tests fail — they validate the *running* environment, and a CUDA platform with zero visible devices
is a failure by design (a rule added several rounds ago). With a device visible they pass. That is
the environment probe behaving correctly, not a regression from this change.

### Scope

Changed: `src/md_templates/install/openmm.py`, `src/md_templates/cli/md_template.py`,
`tests/test_installer_capability.py`, `claudecode-instructions/README.md`, and this journal. A grep
of the diff for `ais_run`, `rest2`, `sysgen`, `mdgen`, `stage_run`, `md_stages`, `system.py`,
`solvation`, `forcefield_record`, `implicit`, `preflight`, `config.py` and `defaults.py` returns
nothing.

## Remaining external limitation

Unchanged: `csy0000/MD-data` is private and `md-data` is not published, so anonymous installation of
the pinned validator remains impossible. This environment's own `md-data` came from a local
directory, which is why it reports `contract_support_ready: false` — the honest answer, and now one
with the same record shape whether installation or validation produced it. No network probe was run
for this task; the accessibility result stands from the previous journal.
