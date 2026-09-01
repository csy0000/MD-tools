# 2026-08-27 — installer capability reporting and the AIS hash-guard correction

Instruction: `claudecode-instructions/20260827_installer-capability-and-hash-guard-correction.md`.
Branch `dev`. No merge to `main`, no tag. **No scientific runtime file was touched.**

Two reporting-and-test gaps left by the previous round. Neither changes what the code computes;
both change whether anyone can see it, and one of them is a test that proved nothing.

## 1. `builtins.open` never intercepted `Path.open`

The path-specific hashing guard I added last round patched `builtins.open`. The generated
`sha256_file()` does:

```python
with Path(path).open("rb") as handle:
```

`pathlib.Path.open()` delegates to `io.open`. `io.open` and `builtins.open` start as the same
function object, but rebinding `builtins.open` only changes the *builtins namespace* — the `io`
module still holds its own reference. Measured directly:

```text
Path.open intercepted by builtins.open patch: False
io.open is builtins.open after patch: False
```

So the guard was never in the path it claimed to guard. The test would have passed with the
production trajectory hashed on every run — which makes it worse than having no test, because it
reported a guarantee it never checked. That is the second time in this sequence a guard has been
weaker than it looked; the lesson I am taking from it is that a guard is only evidence once it has
been shown to fire.

The replacement wraps `pathlib.Path.open` itself, and is both path-specific and caller-specific:
the forbidden trajectory may be opened freely by `mdtraj.iterload`, and is rejected only when
`sha256_file` appears in the call stack. Named small files still hash normally.

**Both sides are proved with the same injected guard**, because a guard that never fires is
indistinguishable from one that does not work:

```text
half 1: the real generated sha256_file applied to the source
  GUARD: the production source was passed to sha256_file
  sentinel = TRIGGERED

half 2: sentinel reset, real AIS preparation
  [PASS] AIS inputs  will be prepared into inputs/sources.dcd before any path runs
  sentinel = installed
  sources.dcd exists: True
  trajectory_sha256: None
  loader: mdtraj.iterload
```

Half 1 reaches the helper through `runpy.run_path("run.py")`, so it is the exact function the
generated runtime uses, not a copy. A companion test keeps the old mistake from returning by
asserting that a `builtins.open` patch does *not* see `Path.open`.

## 2. A failed installation reported no reason

`install_md_data()` returned different shapes on different paths. On a pip failure it returned
`{"installed": False, "error": ...}` — with no `reasons` and no `contract_support_ready`. But
`warnings_for()` and the capability record both read `reasons`, so the failure was captured in a
field nothing displayed and shown to the user as an empty list.

Every outcome now returns one complete record through `_md_data_record()`: pinned repository,
commit, contract version and requirement; `attempted`, `command`, `returncode`; `installed`,
`installed_version`, `installed_commit`, `installed_source`, `source_kind`; `commit_verified`,
`contract_support_ready`, `reasons`, `error`. A pip failure sets `installed: false`,
`contract_support_ready: false`, `commit_verified: false` and a nonempty, actionable `reasons`.

A dry run reports `contract_support_ready: None` rather than `false`. Nothing was installed, so
readiness is not a question with an answer yet, and `false` would read as a failure.

`commit_verified` is deliberately narrower than readiness: a package can be from exactly the pinned
commit and still be unready because its validator functions are missing or its contract version is
one this repository does not target.

## 3. The console says which capability you have

`_report_environment()` now prints one unmissable line after the runtime lines:

```text
  md-data contract: ready
    pinned commit : 48628f9a5d3ace6c6398a63bc3905cd58d542de3
    installed     : md-data 0.2.0 from vcs
    source commit : 48628f9a5d3ace6c6398a63bc3905cd58d542de3
```

```text
  md-data contract: UNAVAILABLE
    pinned commit : 48628f9a5d3ace6c6398a63bc3905cd58d542de3
    installed     : md-data 0.2.0 from local directory
    reason        : the installed md-data records no source commit (installed from local
                    directory: …), so it cannot be shown to be the pinned 48628f9a5d3a
    consequence   : unregistered local simulation is unaffected; `dataset.enabled: true` will
                    fail before building a system.
```

"OpenMM installed successfully" and "contract-managed generation is ready" are different claims,
and a user who reads the first and assumes the second finds out at `sys-gen`, after preparing a
system. The banner `warnings_for()` produces is suppressed from the loose notes so it is not
printed twice.

## 4. It survives into `machine.yaml`

`record_environment()` selected individual OpenMM fields and dropped the computed `md_data` and
`capabilities` entirely — so the one durable record of what an environment can do said nothing
about contract support. Both are now persisted under `installed.openmm`, through `_yaml_safe()`,
which reduces anything that is not a plain container, string, number, bool or `None` to its
`str()`. A `machine.yaml` is read months later by something that is not this code, and a
serialised subprocess object or exception is neither parseable nor stable.

`validate_existing()` now runs the same verification and records the same shape, so an environment
validated later does not produce a `machine.yaml` a reader has to special-case.

## Evidence

Environment: `/path/to/software/md-stack/envs/openmm-8.6.0` — openmm 8.6.0,
cuda-version 13.0, mdtraj 1.11.1, md-data 0.2.0, Python 3.12.14.

### Tests

```text
pytest tests/test_installer_capability.py tests/test_install.py -q -p no:randomly
                                                         39 passed
pytest tests/test_integrity_corrections.py -q -p no:randomly -k "hashing or builtins_open"
                                                          3 passed, 35 deselected
pytest tests/ -q -p no:randomly -m "not gpu"            311 passed, 71 deselected
```

The hash-guard test performs MD only through the existing tiny AIS fixture — picoseconds, two
paths — on **CUDA, device 0, NVIDIA RTX A5000, driver 580.173.02**. No CPU or OpenCL. No nanosecond
runs, no REST2, no full GPU suite, no installation campaign.

### A pip failure, with reasons

```text
installed              False
contract_support_ready False
commit_verified        False
returncode             1
reasons[0]             installing the pinned validator failed (pip exit 1):
                       fatal: could not read Username for 'https://github.com'
reasons[1]             install it manually with: pip install 'md-data @ git+https://…@48628f9a…'
error                  fatal: could not read Username for 'https://github.com'
```

### Persisted `machine.yaml`

```yaml
installed:
  openmm:
    capabilities:
      openmm_runtime_ready: true
      md_data_contract_support_ready: false
      md_data_unavailable_reasons:
      - 'the installed md-data records no source commit (installed from local directory: …),
         so it cannot be shown to be the pinned 48628f9a5d3a'
    md_data:
      commit: 48628f9a5d3ace6c6398a63bc3905cd58d542de3
      contract_version: '1.0'
      installed_version: 0.2.0
      installed_commit: null
      source_kind: local directory
      commit_verified: false
      contract_support_ready: false
```

A round-trip test covers both the ready and unavailable cases, and a separate test asserts
`!!python` never appears in the written file.

### Scientific runtime unchanged

The diff touches three files: `src/md_tools/install/openmm.py`,
`src/md_tools/cli/md_template.py`, `tests/test_integrity_corrections.py`, plus the new
`tests/test_installer_capability.py`. A grep of the changed paths for `ais_run`, `rest2`, `sysgen`,
`mdgen`, `stage_run`, `md_stages`, `system.py`, `solvation`, `forcefield_record`, `implicit`,
`preflight`, `config.py` and `defaults.py` returns nothing. No MD equation, force field,
integration schedule, REST2 exchange or AIS work accumulation was modified.

## Remaining external limitation

Unchanged and still the only blocker: **`csy0000/MD-data` is private and `md-data` is not
published.** An unauthenticated `git ls-remote` fails, the repository and commit APIs return 404
while a public control returns 200, and PyPI has no `md-data`. Anonymous installation of the pinned
validator is therefore impossible, and this environment's own `md-data` was installed from a local
directory — which is exactly why it reports `contract_support_ready: false` rather than claiming a
readiness it cannot demonstrate. The required owner action is to make the repository public or
publish a package release. Repository visibility was not changed.
