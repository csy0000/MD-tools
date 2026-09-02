# Implementation task: correct trajectory, platform, AIS reporting, Amber flags, and MPI contracts

## Baseline

Implement this task on branch `dev`, starting from implementation commit:

```text
da24bded3bbbe2da66e4c06b159c0cdd3422ce54
```

The commit that adds this instruction is expected to be a child of that baseline. Do not reset or discard the completed `md-openmm md-run` work.

Read `CLAUDE.md`, the current implementation, tests, and release notes before editing. Add failing regression tests first. This task corrects the remaining mismatches found by review; it does not redesign the validated REST2, rREST2, AIS, or MD-data mathematics.

## Decisions

1. Ordinary cMD and equilibration trajectories remain DCD. OpenMM has a native `DCDReporter` but no native NetCDF reporter. Do not introduce another dependency merely to change the suffix.
2. AIS may consume a genuine DCD or genuine NetCDF source trajectory. AIS path outputs remain genuine Amber NetCDF files named `AIS_trajNNNN.nc`.
3. Hardware defaults belong in the machine section of the installed user configuration, not in a scientific protocol.
4. `-o` and `-log` are distinct artifacts.
5. The short execution flags must follow Amber semantics. In particular, `-x` means trajectory; `-s` is the MD-tools-specific serialized OpenMM System.
6. A multi-process launch without working MPI coordination is a fatal preflight error. No barrier or collective may silently become a no-op.
7. Keep exactly one installed executable, `md-openmm`, and the four current subcommands. Do not add a separate `md-run` executable.

---

## 1. Genuine DCD for ordinary MD and genuine NetCDF for AIS/REMD

### Conventional stages

For minimization/equilibration/cMD, `-x` names the trajectory and the supported trajectory format for now is DCD:

```bash
md-openmm md-run \
    -i cMD.in \
    -p built.pdb \
    -s built.xml \
    -c eq_npt_free.xml \
    -o cMD.out \
    -x cMD.dcd \
    -r cMD.xml \
    -log cMD.log
```

Requirements:

- use OpenMM's native `DCDReporter`;
- default to `<stage>.dcd`;
- require a `.dcd` suffix for a conventional-stage trajectory;
- refuse `-x output.nc` with a clear message explaining that the ordinary writer is DCD and that renaming a DCD file does not make it NetCDF;
- verify the written DCD magic/header in tests rather than checking only the filename;
- preserve append/restart behavior and verify that resumed output remains a readable DCD.

Do not write DCD bytes under a `.nc` filename.

### AIS source

The documented source for AIS should normally be a fixed-`tau` cMD DCD:

```bash
-source-traj ../cMD_tau0p5/tau_0p5.dcd
```

AIS must accept and correctly detect both genuine DCD and genuine NetCDF coordinate trajectories. Detection must use file contents and/or a format-aware reader, not trust the suffix alone. A file whose suffix and contents disagree must be refused with both the declared suffix and detected format named.

Keep the current source atom-count checks, hashing, deterministic frame selection, `tau_start` assertion, and Maxwell–Boltzmann velocity resampling.

### AIS and REMD outputs

AIS outputs remain genuine Amber NetCDF:

```text
AIS_traj0000.nc
AIS_traj0001.nc
...
```

REMD/rREST2 fixed-state trajectories remain genuine NetCDF:

```text
remd0.nc
remd1.nc
...
```

Tests must open them with at least one independent compatible reader already present in the environment and verify frame/atom counts. Do not infer correctness from the extension.

Update every README, method page, example, generated comment, `run.sh`, CLI help string, and release-note command accordingly.

---

## 2. Machine-level OpenMM platform defaults

The OpenMM platform is a machine property. Extend the existing installed user configuration example:

```yaml
schema_version: "1.0"

user:
  person_id: "your-name-here"
  name: "Your Name"
  orcid: null
  affiliation: null

machine:
  md_data: "/absolute/path/to/MD_DATA"

  openmm:
    platform: CUDA
    precision: mixed
    device_policy: local_rank
```

Use the existing configuration discovery order:

1. an explicit user/machine configuration path if the command exposes one;
2. `$MD_TOOLS_CONFIG`;
3. `${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config`;
4. built-in machine defaults: CUDA, mixed precision, local-rank device placement.

Do not create a second configuration file that can disagree with the existing user configuration. The `machine:` block is the machine configuration.

Supported machine platform values are:

- `CUDA`: normal default;
- `CPU`: an intentional machine-wide CPU default.

There is never an automatic fallback. If the resolved machine default is CUDA and CUDA cannot initialize, fail before dynamics.

Command-line behavior:

- `--cpu` explicitly overrides the machine default for that invocation and must be recorded as an explicit CLI override;
- remove the public `--platform` option from `md-openmm md-run`;
- remove `dynamics.platform` from cMD/REST2/rREST2/AIS scientific configs and generated `.in` files;
- reject the retired `dynamics.platform` key with a migration message pointing to `machine.openmm.platform`;
- keep a long device override only if it is genuinely needed, document that it is an execution placement option, and reject it for CPU;
- do not introduce short hardware flags that collide with Amber.

The run record must distinguish:

- built-in CUDA default;
- machine-config CUDA or CPU default;
- explicit `--cpu` override;
- resolved platform, precision, device, visible devices, MPI rank/local rank, OpenMM version, and available CUDA runtime/driver information.

Update `md-openmm data-register --init` so the created configuration contains the documented machine OpenMM defaults. Existing user configurations without `machine.openmm` remain valid and resolve to CUDA/mixed/local-rank.

---

## 3. Make all AIS reporting controls real

The current implementation validates `reporting.system_printout` and `reporting.checkpoint_printout` for AIS but does not give them runtime behavior. A strict input language must not accept consequential-looking no-op settings.

Pass the resolved `reporting` block into the AIS runtime and implement independent cadences:

### `ais.observation_interval_steps`

Controls work observations:

- tau and protocol step;
- incremental and cumulative work;
- reduced work;
- the work rows used to build `AIS_work.csv`.

### `reporting.solute_printout`

Controls frames in each `AIS_trajNNNN.nc`.

- step 0 and the final switching step must be present;
- every written frame must carry its protocol step/time association;
- it may differ from the work-observation interval;
- remove the current rule that forces `solute_printout == observation_interval_steps`;
- both intervals must still divide `switching_steps`.

### `reporting.system_printout`

Controls a per-path state table, for example:

```text
path_0000/system.csv
```

At minimum record protocol step, tau, potential energy, kinetic energy, total energy, temperature, volume/density when defined, and elapsed switching time. It may differ from both observation and trajectory cadence but must divide `switching_steps`.

### `reporting.checkpoint_printout`

Implement a real, exact, mid-path restart checkpoint. Each in-progress path needs:

- an OpenMM binary checkpoint;
- a machine-readable sidecar containing the global path ID, source frame, current protocol step/update index, current tau, accumulated work, work accumulated since the last observation, emitted work-row count, emitted trajectory-frame count, integrator and velocity seeds, configuration/System/topology/source hashes, and output identities.

On `--resume`:

- validate every fingerprint before loading;
- restore the OpenMM Context and external AIS state together;
- append without duplicating work rows, state rows, or trajectory frames;
- preserve the same global path ID, source frame, seeds, and filename;
- finish with the same number of scheduled observations and frames as an uninterrupted run;
- never mark a path complete until its final files are flushed, validated, and the completion marker is written;
- retain the current behavior that a completed path is skipped and never overwritten.

If the existing NetCDF writer cannot safely append, implement a recoverable staging strategy rather than silently restarting the path while claiming checkpoint continuation.

### Divisibility

Every enabled cadence must divide `switching_steps` exactly:

```python
switching_steps % interval == 0
```

This applies independently to observation, solute trajectory, system state, and checkpoint intervals. Zero may disable a stream only where explicitly documented.

For example, `switching_steps=250` with either `system_printout=100` or `checkpoint_printout=100` must fail before MPI, OpenMM Context creation, or output creation. Values 10, 25, 50, 125, and 250 are valid divisors.

---

## 4. Keep `-o` and `-log` separate

Do not alias or reconcile `-o` and `-log`. They are different outputs:

- `-o`: Amber-like human-readable simulation output;
- `-log`: MD-data-contract provenance/machine log.

For a conventional stage, AIS, REST2, and rREST2:

- allow different paths;
- default to `<name>.out` and `<name>.log`;
- refuse only if the two resolve to the same path;
- include reciprocal references between them;
- make their roles explicit in help and documentation.

The `.out` should contain information useful while reading a simulation: stage/protocol, progress, steps, derived time, energies/temperature where appropriate, restart/checkpoint status, and completion/failure summary.

The `.log` remains the authoritative provenance record: resolved configuration identity, input/output hashes where available, software/hardware, warnings, source ancestry, status, and MD-data-contract fields. Do not require a human to parse the machine record merely to see ordinary run progress.

Remove the current code and documentation claiming that `-o` and `-log` are two names for one artifact.

---

## 5. Enforce Amber short-flag semantics

This decision supersedes the prior public mapping that used `-x` for `built.xml`.

The canonical conventional command is:

```bash
md-openmm md-run \
    -i min.in \
    -p built.pdb \
    -s built.xml \
    -c previous.xml \
    -o min.out \
    -x min.dcd \
    -r min.xml \
    -log min.log
```

Short-flag contract:

| Flag | Meaning | Amber relationship |
|---|---|---|
| `-i` | MD input | mdin |
| `-o` | human-readable MD output | mdout |
| `-p` | topology/reference PDB | prmtop role |
| `-c` | input coordinates/restart | inpcrd/restrt |
| `-r` | output final restart/state | restrt |
| `-x` | output trajectory | mdcrd |
| `-s` | serialized OpenMM System XML | MD-tools extension because PDB does not carry parameters |
| `-log` | provenance/machine log | MD-tools extension |
| `-chk` | resumable checkpoint | MD-tools extension |
| `-odir` | output directory | MD-tools extension |
| `-ng` | replica/worker groups | Amber group semantics |
| `-groupfile` | heterogeneous group file | Amber group semantics |
| `-source-traj` | AIS source ensemble | MD-tools extension |

Requirements:

- remove the public interpretation `-x = built.xml`;
- make `-s built.xml` mandatory;
- make `-x` optional with a protocol-appropriate default trajectory path;
- reject `-x built.xml` with “`-x` is the trajectory; use `-s` for the serialized System”;
- reject a trajectory passed to `-s`;
- ensure argparse abbreviation cannot reinterpret misspelled flags;
- keep useful unambiguous long options;
- do not repurpose another Amber short flag for an MD-tools-specific concept;
- generated Python compatibility wrappers and `md-openmm md-run` must use the same meanings.

Canonical REST2 example:

```bash
mpirun -n 8 md-openmm md-run \
    -ng 8 \
    -i REST2.in \
    -p built.pdb \
    -s built.xml \
    -c eq_npt_free.xml \
    -o REST2.out \
    -x REST2.nc \
    -r restart.json \
    -log REST2.log
```

Canonical AIS example:

```bash
mpirun -n 8 md-openmm md-run \
    -ng 8 \
    -i AIS.in \
    -p built.pdb \
    -s built.xml \
    -source-traj ../cMD_tau0p5/tau_0p5.dcd \
    -o AIS.out \
    -x AIS_traj \
    -log AIS.log \
    -odir ./AIS
```

For AIS, document whether `-x AIS_traj` is a filename prefix or omit `-x` and use the fixed `AIS_trajNNNN.nc` convention. Do not pretend one path can name all N files. Prefer omitting `-x` for AIS unless a prefix option has a precise, tested meaning.

Audit and fix all stale `-s`/`-x` examples in:

- root CLI epilog and `md-run -h`;
- module docstrings;
- README and `CLAUDE.md`;
- `docs/md-run.md`;
- method READMEs;
- example configs;
- generated `.in` comments;
- generated Python docstrings;
- generated `run.sh`;
- release notes and tests.

Every documented command must be executed by a test from outside the checkout against the installed wheel. A help test that merely checks the flag names appear is insufficient.

---

## 6. Fail closed when MPI coordination is unavailable

Serial execution with MPI world size 1 must continue to work without `mpi4py`.

For any launch whose detected world size is greater than 1:

1. import `mpi4py.MPI` before creating output directories, resolved configs, logs, OpenMM Contexts, or generated coordination files;
2. require MPI to be initialized and not finalized;
3. compare `MPI.COMM_WORLD.Get_rank()` and `Get_size()` with launcher environment values when those values exist;
4. compare communicator size with `-ng`;
5. for REST2/rREST2, also compare with the configured replica count;
6. abort before simulation if any value disagrees.

Missing/broken `mpi4py` in a multi-process launch is a fatal error explaining that the command was launched with multiple ranks but MD-tools cannot coordinate them. Do not continue independently and do not turn `barrier`, broadcast, gather, or abort into no-ops.

Replace the current permissive barrier behavior:

- size 1: a no-op is correct;
- size >1: absence or failure of `mpi4py` raises a coordinated preflight error.

After coordinated startup, a fatal error on one rank must stop the whole communicator so the remaining ranks cannot continue integrating or hang indefinitely. Use a controlled coordinated error path where possible and `MPI.COMM_WORLD.Abort(nonzero)` for unrecoverable rank-local failures.

Tests must cover:

- serial execution without `mpi4py`;
- simulated multi-rank launcher environment without `mpi4py` fails before any output;
- launcher-reported size differing from communicator size fails;
- `-ng` differing from communicator size fails;
- configured REST2 replica count differing from `-ng` fails;
- real two-rank MPI REST2 on CUDA;
- real multi-rank AIS on CUDA;
- injected rank-local preflight failure terminates the coordinated launch without leaving another rank running;
- no partial authoritative output is created on preflight failure.

Declare and document the MPI dependency in the conda environment and an appropriate optional package extra if suitable. Do not make `mpi4py` mandatory for ordinary serial users.

---

## 7. Configuration authority and preflight ordering

While making these corrections, preserve one scientific configuration authority.

- Amber-like `.in` files project onto the existing strict MD schema.
- Machine hardware settings are loaded separately and cannot change scientific protocol fields.
- Duplicate keys and sections remain fatal.
- Validate CLI semantics, run input, machine config, file types, MPI availability/counts, and platform request before creating authoritative outputs.
- Do not use plain `yaml.safe_load` on a user-editable or authoritative config path if it permits duplicate keys.
- A failed preflight must not leave a new `resolved.config`, `.out`, `.log`, trajectory, restart, or checkpoint.
- Clarify in documentation whether the `.in` file or `resolved.config` drives each invocation. Runtime behavior and prose must agree. Do not say “the input is never read again” in a command that reparses it every invocation.
- Python wrappers and the `md-run` route must resolve to the same model and call the same runtime implementation.

---

## 8. Acceptance tests

Add focused regression tests, then run the complete existing suite.

### Trajectory formats

- default cMD writes readable DCD;
- `-x custom.dcd` writes readable DCD;
- `-x fake.nc` is refused before output;
- resumed DCD remains readable and has the expected frames;
- AIS reads a genuine DCD source;
- AIS reads a genuine NetCDF source;
- mismatched extension/content is refused;
- every AIS/REMD `.nc` output is genuine readable NetCDF.

### Machine platform

- absent machine settings resolve to CUDA/mixed/local-rank;
- machine CUDA and CPU defaults are honored and recorded distinctly;
- `--cpu` overrides machine CUDA and is recorded as a CLI override;
- retired `--platform` is rejected;
- retired `dynamics.platform` is rejected with a migration message;
- CUDA failure never falls back;
- device selection follows local rank under MPI.

### AIS

- all four cadences can differ while independently dividing `switching_steps`;
- each cadence controls its actual output;
- invalid divisibility fails with no outputs;
- state tables contain the required physical fields;
- interruption after at least one checkpoint followed by `--resume` produces complete, nonduplicated outputs;
- resumed path identity and accumulated work state are preserved;
- completed paths remain untouched;
- the 100-path naming and deterministic MPI distribution tests remain green.

### Output/log separation

- distinct `-o` and `-log` paths both exist and contain their respective required content;
- using the same resolved path for both is refused;
- default names are separate;
- cMD, AIS, REST2, and rREST2 all obey the same distinction.

### CLI and wheel

- exactly one console script and four subcommands;
- every canonical command in documentation runs with the declared flag meanings;
- `-x` is always trajectory and `-s` is always System on the execution surface;
- invalid/ambiguous flags fail rather than being abbreviated;
- build/install the wheel cleanly and execute tests outside the checkout.

### MPI

- all fail-closed requirements in section 6;
- no MPI test may pass through serial substitution;
- CUDA tests assert the actual Context platform is CUDA.

Run and record:

```bash
python -m pytest tests -m "not slow and not gpu"
python -m pytest tests -m "gpu or slow"
python -m build --wheel
```

Also run the real MPI commands documented above with at least two ranks and real CUDA devices. Record exact commands, pass counts, hardware/platform evidence, and wall times in `docs/release-notes/v0.5.0.md`.

Do not skip, xfail, weaken, or delete a scientific test to obtain a green result. Tests that encode the superseded public `-x = System`, `-o = -log`, protocol-level platform, or no-op MPI behavior must be replaced with tests of the corrected contract and explain the supersession in their docstrings.

## Non-goals

- Do not add ordinary-MD NetCDF writing in this pass.
- Do not change REST2 scaling, omega exclusions, exchange acceptance, rREST2 reservoir rules, AIS work definition, or the MD-data schemas.
- Do not add a second console executable.
- Do not restore copied runtime templates.
- Do not merge `dev` into `main`, publish to PyPI, or create a release/tag.

## Completion procedure

Implement in coherent commits on `dev`. Keep this file during the work. When every acceptance criterion is satisfied:

1. update documentation and release notes with actual evidence;
2. remove `CLAUDE_TASK.md` in the final implementation commit;
3. leave no temporary trajectories, logs, environments, caches, or build artifacts in the repository;
4. report the final commit SHA, test commands/counts, CUDA/MPI evidence, and any remaining limitation.
