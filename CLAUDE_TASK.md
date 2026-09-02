# Corrective task: make preflight, MPI, machine configuration, and AIS restart contracts real

## Baseline and purpose

Work on branch `dev`, starting from:

```text
1e2e5b9047aebedbc988eb29acad784066238b4c
```

Read `CLAUDE.md`, the implementation, tests, and `docs/release-notes/v0.5.0.md` before editing. Preserve the completed scientific behavior for cMD, REST2, rREST2, and AIS. This task fixes review defects in the execution and recovery contracts.

The baseline claims that every acceptance criterion passed, but its current GitHub Actions run is red:

```text
https://github.com/csy0000/MD-tools/actions/runs/33634080764
```

The failed step is `md-run refuses -ng outside an MPI launch`. It still invokes the retired flag mapping: it omits mandatory `-s` and passes an XML path through `-x`. The command therefore fails in argparse for the wrong reason and never tests the MPI refusal. Fix this first and do not describe the branch as passing until the current-head CI run is green.

Add regression tests that fail on the baseline before changing runtime code.

---

## 1. Restore green CI with a semantically correct test

Update `.github/workflows/ci.yml`:

- use all four current command names consistently in job names and comments;
- test `-ng` outside MPI with the current flag contract, for example:

```bash
md-openmm md-run -ng 4 \
    -i ladder.in \
    -p missing.pdb \
    -s missing.xml \
    -odir out
```

- do not pass a System through `-x`;
- assert the refusal specifically says the process was not started by an MPI launcher;
- assert the output directory was not created;
- make the test fail if argparse rejected a missing mandatory flag instead;
- keep the command runnable from outside the checkout against the installed wheel.

After pushing the final implementation, wait for the workflow on that exact commit. Record the run URL and result. A previous run or a local pass is not evidence that the final GitHub head is green.

---

## 2. One fail-closed MPI authority

`src/md_tools/remd/mpi.py` must be the only implementation of MPI discovery, validation, coordination, barriers, and aborts.

Remove or replace the permissive duplicates in:

- `src/md_tools/remd/executor.py`;
- `src/md_tools/remd/driver.py`;
- `src/md_tools/remd/generated.py`;
- any other runtime module.

In particular, these baseline behaviors must disappear:

```python
try:
    from mpi4py import MPI
except ImportError:
    return
```

and any coordinator that silently represents a multi-rank launcher as a serial world.

Requirements:

1. World size 1 works without importing `mpi4py`.
2. Launcher world size greater than 1 requires a usable `mpi4py.MPI`.
3. Require MPI initialized and not finalized.
4. Compare launcher rank/size with `MPI.COMM_WORLD`.
5. Compare communicator size with `-ng`.
6. REST2/rREST2 also compare communicator size with the configured replica count.
7. AIS treats `-ng` as worker count and compares it with the communicator.
8. A rank-local fatal startup or runtime error terminates or coordinates the whole communicator; no other rank may continue integrating or hang in a collective.
9. Do not create `-odir`, generated coordination files, logs, resolved configs, Contexts, or scientific outputs before the MPI launch has been validated.
10. Generated `REST2.py`, `rREST2.py`, and `AIS.py` entry points receive the same fail-closed behavior as `md-openmm md-run`.

Do not merely add another guard to `md-run`. The shared lower-level runtime must itself be safe because generated wrappers and the public Python API call it directly.

Regression tests:

- direct generated REST2 wrapper under a fake multi-rank environment with unavailable `mpi4py`: fatal, no output directory;
- direct generated AIS wrapper under the same condition: fatal, no output directory;
- direct `replica_main` and `ais_main` paths cannot bypass the guard;
- no permissive MPI implementation remains outside `remd/mpi.py`;
- real two-rank CUDA REST2 and multi-rank CUDA AIS still pass;
- injected rank-local failure terminates the coordinated launch without a surviving worker or hang.

---

## 3. Strict machine configuration: absent is not invalid

The simulation platform loader currently catches every `RegistrationError` and silently returns built-in CUDA defaults. That conflates:

- no user configuration exists, which may legitimately use built-in defaults;
- a configuration exists but is malformed or invalid, which must be fatal.

Refactor configuration discovery/loading so:

1. A genuinely absent config file resolves to built-in `CUDA / mixed / local_rank`.
2. An existing invalid config is always rejected.
3. Invalid YAML, a non-mapping document, unsupported schema version, invalid `machine.openmm` value, unknown field, or invalid required user field must not be converted to defaults.
4. Duplicate YAML keys or sections are fatal and name the path/key.
5. Use the repository's strict duplicate-rejecting loader, or create one shared strict YAML loader. Do not use plain `yaml.safe_load` for user-editable or authoritative configurations.
6. Apply the same duplicate-key rule when rereading `resolved.config`.
7. `MD_TOOLS_CONFIG` pointing to a missing file is an explicit broken reference and must fail; do not treat it as though the variable were unset.
8. Preserve the discovery order and record the selected path/origin.

Add tests for:

- no config file;
- valid legacy config without `machine.openmm`;
- valid CUDA and CPU machine blocks;
- malformed YAML;
- duplicate `machine`, duplicate `openmm`, and duplicate `platform`;
- invalid platform/precision/device policy;
- explicit environment path that does not exist;
- all invalid cases fail before any run output exists.

---

## 4. Platform and file preflight must precede authoritative output

At baseline:

- `md_run_main` creates `-odir` and writes `resolved.config` before delegated platform resolution;
- `stage_main` opens `.out` and `.log` before loading/validating the machine settings and proving CUDA works;
- `ais_main` creates the output directory and opens reports before resolving the platform;
- an AIS source named only in the input is checked after output creation;
- `replica_main` creates its output directory before group/MPI validation.

Create one shared execution-preflight model or equivalent reusable functions. Before authoritative output creation, validate:

1. CLI path roles and resolved path collisions;
2. input configuration and duplicate keys;
3. topology/System/source existence and basic type;
4. AIS source suffix/content agreement whether supplied by CLI or input;
5. MPI availability and all count/rank agreements;
6. machine configuration;
7. `--cpu` / `--device` compatibility;
8. device policy and resolved device;
9. requested OpenMM platform availability;
10. CUDA Context initialization on the selected device;
11. protocol-specific trajectory suffix and output-path collisions.

Only after this succeeds may code create `-odir`, `resolved.config`, `.out`, `.log`, group/protocol helper files, trajectories, checkpoints, or restarts.

Use resolved paths for collision checks. These must be recognized as the same file:

```text
-o ./run.out -log run.out
-o sub/../run.out -log ./run.out
```

Keep `--check` useful, but do not allow it to bypass validation that the real run relies on.

Tests must assert the filesystem remains unchanged after every preflight failure, including invalid machine configuration, unusable CUDA, invalid input-declared AIS source, MPI mismatch, output collision, and invalid DCD suffix.

---

## 5. Honor machine platform policy and record it truthfully

The accepted machine fields must all have real behavior.

### Platform origin

Distinguish:

- built-in default;
- machine configuration;
- explicit `--cpu` override.

A machine-wide CPU setting is not an explicit CLI CPU request. Do not set `explicit_cpu=True` or `requested_policy=explicit-cpu` for `machine.openmm.platform: CPU`. Introduce an unambiguous selection field if needed, for example:

```yaml
platform_selection: machine-config
requested_platform: CPU
cli_cpu_override: false
resolved_platform: CPU
```

For `--cpu`:

```yaml
platform_selection: cli-override
requested_platform: CPU
cli_cpu_override: true
resolved_platform: CPU
```

### Device policy

The baseline accepts both `local_rank` and `openmm`, but AIS chooses a local-rank device before reading the machine setting.

Choose one coherent contract:

- preferably implement both advertised values:
  - `local_rank`: assign the CUDA device from local rank and record it;
  - `openmm`: do not set `DeviceIndex`; let OpenMM select and record that policy;
- otherwise remove the unsupported value from validation, examples, and documentation.

The machine policy must be loaded before device selection in stage, REST2/rREST2, and AIS. An explicit `--device` overrides placement only, not platform, and is rejected on CPU.

Add tests covering every supported policy in serial and MPI code paths.

---

## 6. Make AIS checkpoint commits crash-atomic

The baseline overwrites the binary OpenMM checkpoint and only afterwards replaces the bookkeeping sidecar. A crash between those writes leaves a new Context checkpoint paired with an old accumulated-work state. Atomic sidecar replacement alone does not make the pair atomic.

Implement a generation-based checkpoint transaction. One acceptable design is:

```text
path_0000/
  checkpoints/
    generation_000012.chk
    generation_000012.json
    generation_000013.chk
    generation_000013.json
  current_checkpoint.json
```

Commit protocol:

1. Never overwrite the currently committed checkpoint/sidecar pair.
2. Write a new generation checkpoint.
3. Flush/sync it and calculate its SHA-256.
4. Write a new generation sidecar containing:
   - generation number;
   - checkpoint filename and SHA-256;
   - path and source identity;
   - protocol step/update and tau;
   - accumulated and since-observation work;
   - emitted work/frame/state counts;
   - integrator and velocity seeds;
   - System/topology/source/resolved-config/schedule fingerprints;
   - output identities.
5. Flush/sync the sidecar.
6. Atomically replace the small `current_checkpoint.json` pointer last.
7. Only then may obsolete generations be removed. Retain at least the previous committed generation until the new pointer is durable.

Resume protocol:

- read only the committed pointer;
- verify pointer, sidecar, checkpoint digest, fingerprints, and stream counts before loading;
- refuse an incomplete, missing, mismatched, or corrupt committed pair;
- ignore or clean uncommitted newer generations without selecting them;
- truncate/rebuild append-only streams to the committed counts;
- never silently restart a partially checkpointed path while claiming resume.

Add deterministic fault-injection tests at every boundary:

- after checkpoint write;
- after checkpoint sync/hash;
- after sidecar write;
- before pointer replacement;
- after pointer replacement;
- during trajectory/CSV writes.

For every injected crash, `--resume` must either resume the last fully committed generation exactly or fail clearly without modifying completed output. Compare row/frame counts, step identities, accumulated work bookkeeping, path/source/seeds, and final completion structure against uninterrupted execution. Do not require bitwise-identical floating-point work across different GPU hardware or worker layouts.

---

## 7. Parser and generated-wrapper parity

All public and generated execution parsers must set `allow_abbrev=False`:

- `md_run_parser`;
- stage parser;
- REST2/rREST2 parser;
- AIS parser;
- any executor parser exposed through generated scripts.

The following must fail everywhere rather than being expanded:

```text
--traj
--plat
--dev
--res
```

Keep the short meanings identical:

| flag | meaning |
|---|---|
| `-i` | MD input |
| `-o` | human-readable output |
| `-p` | topology/reference PDB |
| `-c` | input restart |
| `-r` | output restart |
| `-x` | trajectory |
| `-s` | serialized OpenMM System |
| `-log` | provenance log |
| `-chk` | resumable checkpoint |
| `-odir` | output directory |
| `-ng` | MPI groups/workers |

Run parser-contract tests against both `md-openmm md-run` and every generated wrapper from an installed wheel outside the checkout.

---

## 8. Documentation and evidence corrections

Audit and fix:

- the stale AIS `production.nc` source example in `src/md_tools/run/main.py`;
- stale claims that `--cpu` is the only possible CPU selection now that a machine-wide CPU default exists;
- any statement that the old permissive executor barrier is gone while its code remains;
- CI job names/comments that still say three commands;
- any release-note claim that all acceptance criteria passed on a commit whose CI is red.

Update `docs/release-notes/v0.5.0.md` with a corrective subsection. Preserve prior evidence as historical evidence, but label the failed run accurately. Do not rewrite a failed run as green.

---

## 9. Required verification

Run:

```bash
python -m pytest tests -m "not slow and not gpu"
python -m pytest tests -m "gpu or slow"
python -m build --wheel
```

Additionally run:

- installed-wheel commands from outside the checkout with `PYTHONPATH` scrubbed;
- direct generated stage, REST2, rREST2, and AIS wrapper parser tests;
- real two-rank or greater CUDA REST2;
- real multi-rank CUDA AIS;
- multi-rank missing-`mpi4py` refusal through both `md-run` and generated wrappers;
- rank-local failure/termination test with a timeout that proves no worker hangs;
- AIS checkpoint fault-injection/resume tests;
- malformed/duplicate machine configuration tests with no output creation.

The GitHub Actions run for the final commit must be green. Record its URL, exact pass counts, wall times, CUDA platform/device evidence, MPI world sizes, and wheel path/import origin.

Do not skip, xfail, weaken, or delete scientific tests to obtain a pass. Replace only tests that encode a superseded contract, and explain the supersession in their docstrings.

## Non-goals

- Do not change REST2 scaling, omega exclusions, exchange probabilities, rREST2 reservoir mathematics, AIS work definition, force-field defaults, or MD-data schemas.
- Do not add a second executable.
- Do not add ordinary-MD NetCDF writing.
- Do not merge `dev` into `main`, publish a package, or create a release/tag.

## Completion

Implement in coherent commits on `dev`. Keep this file until:

1. all required local lanes pass;
2. wheel tests pass outside the checkout;
3. real CUDA/MPI checks pass;
4. the GitHub Actions run for the final implementation commit is green.

Then remove `CLAUDE_TASK.md` in a final cleanup commit and wait for the workflow on that cleanup commit too. Report the final SHA and the evidence for that exact head.
