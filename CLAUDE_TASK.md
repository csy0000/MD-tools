# Implementation task: warning-only crossed pairs and GPU-first `md-openmm md-run`

## Baseline and scope

Implement this task on branch `dev`, starting from implementation commit:

```text
0b68f496fc572986c857acf40a6f247046d92a74
```

The commit that adds this instruction may be the child of that baseline. Do not reset or discard the baseline work.

This task corrects one scientific-policy bug found in review and adds an Amber-like execution surface without moving the validated simulation physics back into generated files.

The package must continue to install exactly one executable:

```text
md-openmm
```

After this task it has exactly four public subcommands:

```text
md-openmm build-top
md-openmm build-md
md-openmm md-run
md-openmm data-register
```

Do **not** install a separate `md-run`, `openmm-md`, or `openmm-run` executable. “md-run” below always means the `md-openmm md-run` subcommand. Keep the existing name `data-register`; “build-register” in discussion was not a requested rename.

## Design constraints

1. All simulation behavior remains in the installed `md_tools` package. Generated files are declarations or small entry points, never copies of restraints, reporters, REST2 scaling, exchange rules, AIS switching, restart logic, or platform selection.
2. The public runner is Amber-like: `-i` reads a small, human-readable run input and the command-line supplies the topology, serialized System, outputs, and optional continuation.
3. The current compact Python wrappers may remain as compatibility entry points, but they and `md-openmm md-run` must call the same authoritative runtime APIs. Do not create two execution engines.
4. CUDA is the default and is mandatory unless the user explicitly writes `--cpu`. There is no automatic CPU, OpenCL, or Reference fallback.
5. Preserve the existing MD-data contract and transactional behavior of `data-register`.
6. Preserve all validated REST2/rREST2/AIS Hamiltonian and exchange mathematics. This task changes invocation, configuration, platform policy, output organization, and validation—not the equations.

---

## 1. Correct the explicit protein/water pairing policy

The current `md_tools.build.top._check_pairings` still rejects ff19SB + TIP3P before the later warning machinery can run. That contradicts the intended policy.

For explicit solvent, all four combinations below must be buildable:

| Protein | Water | Result |
|---|---|---|
| ff14SB | TIP3P | build silently; supported matched pair |
| ff19SB | OPC | build silently; supported matched pair |
| ff14SB | OPC | build, emit and record a warning |
| ff19SB | TIP3P | build, emit and record a warning |

For both crossed pairs:

- emit a conspicuous warning on stderr;
- write the human-readable warning into `built.log`;
- store a structured warning in the machine record;
- name the selected combination and both supported matched pairs;
- state that the build is permitted but the combination was not jointly validated.

Sage and GAFF are orthogonal ligand choices and must not cause this protein/water warning.

Keep genuinely incoherent combinations as hard errors. In particular, ff19SB + GBn2 remains refused because the implicit-solvent parameterization is incompatible; explicit box/salt settings under GBn2 remain refused.

Remove the early ff19SB + TIP3P rejection from the public `build-top` resolver. Test both crossed pairs through the public `md-openmm build-top` path, not only by calling `pairing_warnings()` directly. Update tests that currently assert ff19SB + TIP3P is rejected.

---

## 2. Add `md-openmm md-run`

Add an `md-run` parser beneath the existing `md-openmm` executable.

A conventional stage must support at least:

```bash
md-openmm md-run \
    -i min.in \
    -p built.pdb \
    -x built.xml \
    -o min.out \
    -r min.xml \
    -log min.log
```

Required meanings:

- `-i`: Amber-like MD input file.
- `-p`: topology/coordinates PDB matching the serialized System.
- `-x`: serialized OpenMM System XML. Use `-x`, not `-s`, on this new public surface.
- `-c`: optional continuation/restart input.
- `-o`: human-readable MD output.
- `-r`: final restart/state output.
- `-log`: MD-data-contract log/machine record.
- `-odir`: optional output directory when a protocol owns multiple files.
- `--cpu`: explicit CPU execution; see the platform policy below.
- `-h`: usable without initializing an OpenMM Context.

Choose unambiguous long aliases as well, but preserve these Amber-like spellings. Do not rely on argparse prefix abbreviation.

### Input parser

Implement one small strict parser for the generated Amber-like input files. It may use namelist-like sections such as `&cntrl`, `&remd`, and `&AIS`, but it does not need to implement arbitrary Amber syntax.

Requirements:

- comments and trailing commas are accepted;
- keys are case-normalized deliberately and documented;
- unknown keys are hard errors with a useful suggestion;
- duplicate keys and duplicate sections are hard errors—never silently overwritten;
- type errors name the key and received value;
- booleans have one documented spelling set;
- all stage lengths, exchange intervals, switching lengths, reporting intervals, and checkpoint intervals are integer step counts;
- real physical constants retain their units in their key names;
- parsing and semantic validation finish before an OpenMM Context or output file is created.

Keep one resolved internal model. Do not let the new parser and the existing YAML resolver independently redefine scientific defaults.

### Relationship to `build-md`

`md-openmm build-md` should generate the appropriate small `*.in` declarations and `run.sh` commands using `md-openmm md-run`.

The current compact Python wrappers may remain for compatibility and for users who prefer:

```bash
python min.py ...
```

However:

- wrappers must remain one import plus one call;
- wrappers must delegate to the same runtime used by `md-openmm md-run`;
- there must be one authoritative resolved configuration;
- a correction in the installed package must affect both invocation paths;
- do not reintroduce `openmm/templates/` or copy runtime modules into generated directories.

Document which file is authoritative if both `resolved.config` and `*.in` are emitted. There must not be two independently editable declarations that can disagree silently. Prefer making the `*.in` file a complete, human-readable projection consumed by the runner and retaining `resolved.config` as immutable generation provenance, with a recorded digest/link between them. If either is edited, mismatch detection must be explicit rather than silently choosing one.

---

## 3. CUDA-first platform policy

The intended analogy is:

```text
md-openmm md-run ≈ pmemd.cuda
```

For every operation that creates an OpenMM Context to minimize, evaluate production energies, or integrate:

- select CUDA by default;
- if CUDA cannot be initialized, fail before dynamics with a clear error;
- do not silently fall back to CPU, OpenCL, or Reference;
- state that `--cpu` is required for intentional CPU execution;
- `--cpu` selects the OpenMM CPU platform and is the only public CPU opt-in;
- reject contradictory requests instead of selecting one by precedence.

Apply exactly the same policy to:

- cMD and every equilibration/minimization stage;
- fixed-`tau` cMD;
- REST2;
- rREST2;
- AIS;
- split and all-in-one workflows;
- direct Python compatibility wrappers.

Remove the currently documented difference in which a replica ladder can fall back to CPU while an ordinary stage refuses. Centralize the policy so stages, REMD, and AIS cannot drift again.

Do not claim that OpenFF/AmberTools parameter assignment itself is GPU accelerated. The policy applies to OpenMM Context work; CPU-only preprocessing in `build-top` is still legitimate.

Every run record must state:

- requested acceleration policy: default CUDA or explicit `--cpu`;
- resolved OpenMM platform;
- CUDA device index when applicable;
- CUDA precision;
- visible GPU model;
- OpenMM version;
- CUDA runtime/driver information when available;
- MPI world rank/local rank for parallel runs;
- whether CPU was explicitly requested.

A CPU result must never look like an unnoticed CUDA fallback.

---

## 4. Amber-like MPI execution for REST2 and rREST2

Support the intended command shape:

```bash
mpirun -n 8 md-openmm md-run \
    -ng 8 \
    -i REST2.in \
    -p built.pdb \
    -x built.xml \
    -odir ./REST2 \
    -o REST2.out \
    -log REST2.log
```

The same surface applies to rREST2.

For this implementation, use one MPI rank per replica/state:

- `-ng` is the number of replica groups/states;
- MPI world size must equal `-ng`;
-ng`;
- the input's replica count must equal `-ng`;
- a mismatch is a preflight error naming all three values;
- `-ng > 1` outside an MPI launch is an error;
- each rank owns one OpenMM Context and one state;
- exchange coordination uses MPI communication and the existing authoritative exchange rules;
- ranks must not concurrently append unsafely to the same file.

CUDA placement must be deterministic. Derive a node-local rank and map it to the CUDA devices visible to that process. Respect `CUDA_VISIBLE_DEVICES`. Record the mapping. If a requested rank cannot initialize its assigned CUDA device, abort the coordinated run; never continue with only that rank on CPU.

Keep the existing state-trajectory convention:

```text
remd0.nc
remd1.nc
...
remd7.nc
```

These are fixed thermodynamic-state trajectories, not walker-named or tau-named trajectories. Preserve the Amber-compatible `rem.log`, neighboring-pair acceptance report, restart identity, and rREST2 reservoir rules.

If a group file remains useful, support an Amber-like `-groupfile` alias, but do not require users to restate identical `-p` and `-x` paths eight times when one shared input describes an ordinary homogeneous REST2 ladder.

---

## 5. AIS execution and source trajectory

AIS uses the same runner. A serial example is:

```bash
md-openmm md-run \
    -i AIS.in \
    -p ../built.pdb \
    -x ../built.xml \
    -source-traj ../cMD_tau0p5/tau_0p5.nc \
    -odir ./AIS \
    -o AIS.out \
    -log AIS.log
```

A parallel example is:

```bash
mpirun -n 8 md-openmm md-run \
    -ng 8 \
    -i AIS.in \
    -p ../built.pdb \
    -x ../built.xml \
    -source-traj ../cMD_tau0p5/tau_0p5.nc \
    -odir ./AIS \
    -o AIS.out \
    -log AIS.log
```

For AIS, `-ng` is the number of worker groups. Workers run independent paths and do not exchange. `number_of_paths` is the global total, not the number per rank. Assign global path IDs deterministically across ranks.

`-source-traj` is required for AIS:

- it names a trajectory file; a path such as `tau_0p5.nc/` with a trailing slash must fail as “not a file”;
- validate atom count and compatibility with `-p` and `-x`;
- the input declares `tau_start`, and the log records that the source is asserted/verified to represent that ensemble;
- hash and record the source trajectory;
- choose source frames deterministically from the global seed;
- record the source frame for every path;
- for coordinate-only NetCDF input, resample Maxwell–Boltzmann velocities at the configured physical temperature and record that policy;
- do not silently pretend an ordinary coordinate trajectory contains stored phase space.

For `number_of_paths = 100`, write exactly 100 path trajectories:

```text
AIS_traj0000.nc
AIS_traj0001.nc
...
AIS_traj0099.nc
```

Use zero-based path IDs and at least four digits of zero padding. Increase the width if needed for a larger path count. MPI scheduling and restart must never change which global path owns which filename. A completed path must not be overwritten; an incomplete path must be safely resumable or explicitly restarted according to the existing restart contract.

Each path trajectory contains its source configuration at switching step 0 and the requested intermediate configurations through the final `tau_end` configuration. For example, `switching_steps=250` and `solute_printout=10` yields 26 frames at steps 0, 10, ..., 250.

Also produce a deterministic global work table, at minimum:

```text
path_id,source_frame,switch_step,tau_before,tau_after,delta_work_kj_mol,total_work_kj_mol
```

Merge rank-local results safely and sort by path ID and switching step. Preserve the existing AIS work convention and REST2 scaler; do not implement a second scaler.

### AIS divisibility validation

Every enabled step-based interval must divide the complete switching path exactly:

```python
switching_steps % interval == 0
```

Apply this to at least:

- `observation_interval_steps`;
- `solute_printout`;
- `system_printout`;
- `checkpoint_printout`.

A value of zero may mean disabled only where the schema explicitly permits it. Validate before loading OpenMM or writing outputs.

Therefore this input is invalid:

```text
switching_steps = 250
system_printout = 100
checkpoint_printout = 100
```

The error must name both values and suggest valid divisors. A valid example is:

```text
&AIS
  tau_start                  = 0.5,
  tau_end                    = 0.0,
  number_of_paths            = 100,
  switching_steps            = 250,
  observation_interval_steps = 10,
  source_frame_start         = 0,
  source_frame_stride        = 10,
  source_frame_selection     = random,
  timestep_fs                = auto,
  temperature_K              = 300.0,
  friction_per_ps            = 1.0,
  solute_printout            = 10,
  system_printout            = 50,
  checkpoint_printout        = 50,
  random_seed                = 20260902,
/
```

---

## 6. Fix review-discovered configuration and documentation defects

Correct these in the same pass:

1. `configs/md/cMD.config` contains an accidental uncommented scalar:
   `dynamics: minimisation, the equilibration chain, then production.`
   Comment or rewrite it so there is only one `dynamics` key.
2. Reject duplicate YAML keys generally instead of accepting PyYAML's silent last-value-wins behavior. Cover both system and protocol configuration loaders.
3. `docs/scientific-defaults.md` refers to nonexistent `docs/md-defaults-references.bib`. Point it to the actual `docs/scientific-defaults.bib`, or rename the bibliography once and update every reference consistently.
4. `configs/sys/build-top.config` currently says ff19SB + TIP3P is refused. Update it to the warning-only explicit-pair policy.
5. The same config shows `--config md_build.config`, which does not exist. Use the shipped `configs/sys/build-top.config` path/name.
6. CLI help currently describes the default workflow as fixed at 2 fs. Describe `timestep_fs: auto`: 2 fs with ordinary serialized hydrogen masses and 4 fs when the serialized System proves HMR.
7. Update the release notes to remove the accepted REMD CPU-fallback difference; after this task every runtime follows the centralized CUDA-first policy.

Keep the root README concise. Put the detailed runner, MPI, REST2/rREST2, and AIS explanations in `docs/`, with short links from the root.

---

## 7. Tests and acceptance gates

Add focused regression tests before changing implementation. At minimum cover:

### CLI and packaging

- a built wheel installs only the `md-openmm` executable;
- `md-openmm -h` lists exactly the four public subcommands;
- `md-openmm md-run -h` works without a CUDA device or OpenMM Context;
- no `md-run` console-script entry exists.

### Pair warnings

- both matched explicit pairs build without pairing warnings;
- ff14SB + OPC builds and records the structured warning;
- ff19SB + TIP3P builds and records the structured warning;
- ligand Sage/GAFF choice does not affect the warning;
- ff19SB + GBn2 remains a hard error.

### Parser

- representative cMD, REST2, rREST2, and AIS inputs parse;
- unknown, duplicate, mistyped, and unit-confused keys fail clearly;
- duplicate YAML keys fail in shipped/user YAML config paths;
- invalid input fails before output creation.

### CUDA/CPU policy

- every runtime family uses the same centralized resolver;
- default execution refuses when CUDA is unavailable;
- no default path selects CPU, OpenCL, or Reference;
- `--cpu` explicitly selects CPU and the record says so;
- real GPU smoke tests assert the actual OpenMM platform name is `CUDA`, not merely that a run completed.

### MPI REMD

- a short two-rank REST2 run uses two states and produces both state trajectories;
- rank/world/`-ng`/replica mismatches fail before integration;
- deterministic device placement is recorded;
- exchange history, state trajectories, restart, and neighboring acceptance remain correct.

### AIS

- 100 paths produce exactly `AIS_traj0000.nc` through `AIS_traj0099.nc`;
- filenames and selected source frames are invariant to MPI worker count;
- each trajectory contains the expected initial and final frames;
- source trajectory must be a compatible file;
- work rows are complete, uniquely keyed, and deterministically ordered;
- `switching_steps=250` with a 100-step enabled interval is refused;
- 10- and 50-step intervals are accepted;
- restart neither overwrites completed paths nor changes path identity;
- GPU tests confirm AIS actually runs on CUDA by default.

### Existing invariants

Run the complete fast and GPU/slow suites. Do not weaken, delete, skip, or xfail an existing scientific test merely to make this task pass. If an old test encodes the explicitly superseded ff19SB + TIP3P rejection or REMD CPU fallback, replace it with the new behavior and explain the change in its docstring.

Build a wheel, install it into a clean environment, and exercise commands from outside the checkout. Verify generated projects contain no source-checkout path and no copied runtime implementation.

Record exact commands, platform/device evidence, pass counts, and any genuinely untestable condition in `docs/release-notes/v0.5.0.md`.

## Non-goals

- Do not change REST2 scaling equations, omega exclusions, exchange acceptance, rREST2 reservoir mathematics, or AIS work convention.
- Do not add a second executable.
- Do not resurrect `openmm/templates/`.
- Do not copy common functions into generated projects.
- Do not add automatic CPU fallback.
- Do not rename `data-register`.
- Do not redesign the MD-data schema.
- Do not merge `dev` into `main` or create a release/tag.

## Completion procedure

Implement in coherent commits on `dev`. Keep this file while working. When every acceptance gate is satisfied:

1. update the release notes with evidence and remaining limitations;
2. remove `CLAUDE_TASK.md` in the final implementation commit;
3. leave the branch with no temporary journals, generated trajectories, environments, caches, or build artifacts;
4. report the final commit SHA and exact test commands/results.
