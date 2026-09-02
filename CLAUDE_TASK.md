# Final acceptance correction: runtime preflight, restart identity, AIS work decomposition, and exhaustive CUDA verification

## Authority and baseline

Work on `csy0000/MD-tools`, branch `dev`.

This instruction is based on implementation commit:

`b57ea19f98b609e9fd0bf837ef65fc75ea1b2c7a`

Read `CLAUDE.md`, the current implementation, tests, documentation, and this file completely before editing. This file is the acceptance authority for this correction. Implement the requirements; do not replace this file with another plan.

Preserve the already validated scientific behavior: REST2/rREST2 scaling and exchange mathematics, rREST2 probability-one reservoir transfer, AIS update/work sign and ordering, tau as the sole persisted Hamiltonian coordinate, the prohibition on barostats during AIS and fixed-tau production, deterministic path/rank assignment, and the MD-data contract.

Add regression tests before or alongside each fix. Do not weaken, delete, skip, or xfail scientific tests merely to obtain a passing result.

## 1. One complete preflight for every entry point

Every independently callable execution path must use the same mode-aware preflight before the first filesystem mutation:

- `md-openmm md-run`;
- direct cMD stage and all-in-one runners;
- generated cMD, REST2, rREST2, and AIS Python wrappers;
- replica and AIS executors callable from Python.

Preflight must finish all deterministic and read-only checks before creating or modifying directories, output/log/config files, helper scripts, trajectories, state tables, checkpoints, or restarts. It must return a typed prepared result that the runtime consumes unchanged. Do not validate at the outer CLI and then reload/re-resolve the same authority later.

The prepared result must include, as applicable:

- validated complete machine/user configuration;
- one resolved platform request and verified OpenMM platform/device properties;
- MPI coordination, rank, world size, and group/device assignment;
- parsed input and validated mode/flag contract;
- loaded topology, serialized System, coordinates/restart, and source trajectory metadata;
- resolved timestep and HMR compatibility;
- solute selection, force classification, omega exclusions, and scaling plan;
- AIS schedule, reporting cadences, source-frame selection/window, atom-count compatibility, seeds, and prepared scaled Systems;
- rREST2 reservoir identity, box, frames, velocities, and refresh constraints;
- complete output inventory and collision policy;
- an immutable configuration/runtime fingerprint.

Move all currently late predictable refusals into this phase, including:

- 4 fs without valid HMR and all other timestep/mass incompatibilities;
- implicit-solvent/NPT and fixed-tau/NPT incompatibilities;
- AIS barostat refusal;
- AIS switching/reporting divisibility and schedule errors;
- source-frame range and source/topology atom-count mismatch;
- REST2 force classification, omega handling, and unsupported-force errors;
- rREST2 reservoir incompatibilities;
- scaled-System construction failures that can be detected without running dynamics.

A rejected run must leave no output artifact, including an empty output directory.

### Missing continuation semantics

An explicitly supplied missing `-c` is an error during any real execution and must fail in preflight. Do not add it to input checks only when it already exists.

The only allowed “pending parent” case is a downstream stage inspected as part of a whole-chain all-in-one `--check`, where the parent would be created by an earlier stage in that same validated chain. Represent that case explicitly; do not infer it from nonexistence.

### Platform propagation

Resolve the OpenMM platform exactly once per execution and pass that validated result through:

`md-run → runtime → executor/driver → Context creation`.

REST2/rREST2 must consume the platform/device result from `LadderPreflight`; the driver must not reload machine configuration or call a second platform resolver after files exist. Remove or refactor stale alternate platform authorities, including automatic CPU fallback paths and obsolete environment-variable selectors. There must be no automatic CPU fallback.

A machine-configured CPU has:

- `explicit_cpu = false`;
- `cli_cpu_override = false`;
- selection provenance `machine-config`.

Only an invocation that actually supplies `--cpu` may set `explicit_cpu = true`.

### Complete existing configuration validation

If the user configuration file is absent, documented built-in defaults may apply. If it exists, validate it completely and fail closed. It must include:

- `schema_version: 1.0`;
- a `user` mapping;
- non-empty `user.person_id`;
- non-empty `user.name`;
- valid machine/OpenMM fields and supported values;
- duplicate-key rejection at every YAML read.

Registration and simulation must use the same validation authority. Do not silently treat a malformed existing file as absent. Do not reread authoritative configuration with `yaml.safe_load`.

## 2. Strict, protocol-specific CLI contract

Audit the full protocol × flag matrix. Every accepted option must have implemented behavior; otherwise reject it before output creation with a specific error.

At minimum correct and test:

- reject `-ng` for cMD;
- reject group files for cMD and AIS;
- reject `-source-traj` outside AIS;
- reject AIS `-x` in preflight if AIS owns per-path trajectory names;
- reject or explicitly implement AIS `-c`, `-r`, and `-chk`; no silent ignores;
- implement REST2/rREST2 `-chk` as a real output contract or reject it;
- accept and propagate REST2/rREST2 `--device`, or remove/reject it consistently at every layer;
- make REST2/rREST2 `--check` genuinely read-only;
- define one consistent cMD `--resume` contract and forward it through generated wrappers;
- give `--overwrite` explicit, safe semantics for the complete output inventory, not only `resolved.config`.

Use `allow_abbrev=False` for every parser, including executor and generated-wrapper parsers. Enforce the Amber mapping everywhere:

- `-i`: MD input;
- `-o`: human-readable MD output;
- `-p`: topology/reference PDB;
- `-c`: input restart/coordinates;
- `-r`: output restart;
- `-x`: output trajectory;
- `-s`: serialized OpenMM System;
- `-log`: separate structured provenance log.

Parameterize tests over the whole matrix so a safe outer wrapper cannot conceal an unsafe lower-level entry point.

## 3. MPI must fail collectively

`md_tools.remd.mpi` remains the only MPI authority.

When launcher state indicates multiple ranks, missing/broken/inconsistent `mpi4py`, world-size mismatch, `-ng` mismatch, or replica-count mismatch must fail before output.

A failure on one rank during CUDA/platform preflight or during REST2/rREST2/AIS execution must not leave other ranks blocked at a barrier or continuing alone. Implement collective error agreement where possible and communicator abort for unrecoverable rank-local failures. Tests must prove the launcher returns nonzero promptly, with no surviving/hanging rank and no partial authoritative output.

## 4. Output inventory, helper identity, and MPI writes

Preflight must derive the complete actual output inventory, including:

- human and structured logs;
- resolved configuration;
- trajectories, state/thermodynamic tables, and phase-space streams;
- restarts, checkpoints, sidecars, manifests, and generation pointers;
- REST2/rREST2 group files, solute files, generated protocol helpers, rank-specific outputs;
- AIS global tables, path trajectories, path observations, completion records, and checkpoint generations.

Check collisions and existing-output policy for every item before mutation.

Under MPI, only rank 0 may atomically write shared files such as `resolved.config`, helper files, and global manifests. All ranks must then synchronize and verify the same digest. Never let every rank overwrite the same path.

For generated ladder execution:

- resolve relative paths relative to the group file’s parent;
- validate all group lines before writing helpers;
- because the current ladder is homogeneous REST2/rREST2, require identical `-i`, `-p`, `-s`, `-c`, and solute identity across lines except the state/group index, unless real per-state inputs are fully implemented;
- fingerprint helper content and refuse stale/incompatible reuse;
- do not overwrite helper files before validation succeeds.

## 5. Crash-safe restart and immutable run identity

### cMD

Replace the non-atomic binary-checkpoint/sidecar sequence with a generation transaction and atomically committed pointer, analogous to the strengthened AIS primitive. Each committed generation must bind:

- OpenMM checkpoint digest;
- protocol/configuration fingerprint;
- simulation step and stage identity;
- trajectory, state-table, and phase-space committed row/frame counts;
- relevant seeds and platform/scientific identity.

On resume, validate the committed pair and truncate appendable streams to committed counts before continuing. Never combine a newer Context checkpoint with older bookkeeping. Inject failures at every transaction and stream boundary.

A “completed” log must not cause an early return before preflight and identity verification. Validate the current fingerprint and required output hashes/counts before treating a run as complete.

### AIS

Create a run-level immutable identity that prevents an output directory from mixing a different source trajectory, source hash, topology/System, tau schedule, seed policy, selected frames, reporting schema, or path count.

Do not overwrite selected-frame mappings against existing completed paths. Global tables and manifests must be atomic and restart-safe.

`completed.json` for each path must be an atomic completion manifest containing at least:

- run/configuration fingerprint;
- source trajectory hash and source frame;
- velocity/source seed identity;
- schedule and schema version;
- hashes and row/frame counts for trajectory and observation outputs;
- final total and component work values.

A completed path may be skipped only after those fields and its outputs verify. Otherwise fail clearly or recover from the last committed generation.

Integrated fault-injection tests must interrupt a real AIS path at checkpoint, work-table, observation, trajectory, and completion-manifest boundaries. Resume must choose only the last fully committed generation, truncate/repair append streams, and produce the same final scientific result as an uninterrupted reference without duplicate or missing records.

Old or incompatible checkpoint/completion schemas must be refused explicitly.

## 6. AIS energy and work decomposition for later Hummer–Szabo reweighting

The current total-potential evaluations are insufficient to recover the requested three groups from a mixed `NonbondedForce`. Implement and report the exact REST2 Hamiltonian basis.

Let:

`a = 1 - tau`

and define:

`U(tau, x) = U_unscaled(x) + a U_linear(x) + a^2 U_quadratic(x)`.

The three groups are:

1. **Unscaled**
   - bonds and angles;
   - excluded omega torsions;
   - environment–environment interactions;
   - any other term proven independent of tau.

2. **Linear / sqrt-scaled basis**
   - solute–environment nonbonded interactions;
   - generalized Born terms for supported implicit-solvent REST2.

3. **Quadratic / linearly scaled basis**
   - solute–solute nonbonded and 1–4 interactions;
   - eligible solute torsions;
   - solute CMAP terms.

Persist tau as the sole scaling coordinate. Do not persist separate `s` or `sqrt(s)` state variables.

Add stable, versioned columns to every per-path observation/work record and the global AIS work table:

- `potential_unscaled_kj_mol`;
- `potential_linear_basis_kj_mol`;
- `potential_quadratic_basis_kj_mol`;
- `potential_linear_contribution_kj_mol`;
- `potential_quadratic_contribution_kj_mol`;
- `potential_total_reconstructed_kj_mol`;
- `delta_work_unscaled_kj_mol`;
- `delta_work_linear_kj_mol`;
- `delta_work_quadratic_kj_mol`;
- `total_work_unscaled_kj_mol`;
- `total_work_linear_kj_mol`;
- `total_work_quadratic_kj_mol`.

Keep the existing total potential/work fields for compatibility. Add final component totals and schema identity to `AIS_paths.csv`, checkpoint bookkeeping, completion manifests, and provenance.

Enforce and test, within documented numerical tolerances:

`potential_total = U_unscaled + a*U_linear + a^2*U_quadratic`

and

`delta_work_total = delta_work_unscaled + delta_work_linear + delta_work_quadratic`

with the same equality for cumulative work.

For a parameter update at fixed coordinates, the unscaled incremental work should be zero within tolerance; retain the column explicitly. Test the frozen-coordinate telescoping identity and forward/reverse sign convention.

Prefer a scientifically transparent decomposition based on energy evaluations at controlled parameter values rather than an invasive refactor of the validated force construction. If the selected exact method requires three potential-energy evaluations per switching update, record in provenance:

- `potential_energy_evaluations_per_update: 3`;
- the total number of switching energy evaluations.

This overhead is acceptable but must be measured and documented. It adds energy evaluations, not integration steps.

Scientific tests must cover:

- explicit solvent and implicit GB;
- all supported scaled force classes, including 1–4, eligible torsions, CMAP, and omega exclusions;
- GB appearing in the linear basis;
- component sum/reconstruction at endpoint and interior tau values;
- observation interval greater than update interval;
- uninterrupted versus resumed component accumulators;
- source from genuine DCD and NetCDF;
- old-schema refusal.

## 7. Exhaustive real-CUDA verification

“CUDA tested” means every CUDA-dependent runtime function and supported scientific branch is inventoried and exercised on real NVIDIA hardware. A serial/CPU/OpenCL/Reference substitute does not count.

Before final acceptance, create a maintained coverage matrix in the release evidence that maps:

`source function or CUDA branch → protocol feature → test name/command → GPU(s) → precision → result`.

Inventory every function/branch that:

- creates an OpenMM `Context` or `Simulation`;
- selects or configures CUDA;
- evaluates GPU energies or forces;
- integrates or minimizes;
- changes REST2/AIS Context parameters;
- writes trajectories/state derived from CUDA execution;
- performs exchange, reservoir refresh, checkpoint/restart, or AIS switching.

Exercise at least the following on real CUDA:

### cMD

- direct stage, generated wrapper, `md-run`, and all-in-one routes;
- minimization, NVT, NPT, restraints, barostat, ordinary dynamics, and fixed-tau dynamics;
- explicit and implicit solvent where supported;
- valid 2 fs/no-HMR and 4 fs/HMR systems;
- DCD, state/thermodynamic, phase-space, restart, checkpoint, interruption/resume, extension, and read-only `--check`.

### REST2

- direct/generated/`md-run` routes;
- explicit and implicit solvent;
- serial where supported and real multi-rank MPI;
- representative 2-, 4-, and 6-state ladders where hardware permits;
- static and dynamic scaling paths;
- nonbonded, 1–4, torsion, CMAP, GB, and omega-exclusion behavior;
- exchange energies, acceptance, swaps, state trajectories, `rem.log`, checkpoint/resume/extension;
- `--check`, `--device`, local-rank placement, and multi-GPU assignment.

### rREST2

- real MPI/CUDA execution;
- reservoir identity and box validation;
- frames and velocities;
- probability-one transfer, refresh, continuation, checkpoint/resume, and deterministic identity.

### AIS

- explicit and implicit solvent;
- DCD and NetCDF source trajectories;
- single-rank and real multi-rank path distribution;
- deterministic source selection and velocity seeds;
- switching schedules and all independent reporting/checkpoint cadences;
- per-path NetCDF trajectories, system tables, global/path work tables, and the new three-component decomposition;
- checkpoint fault injection, partial-path resume, completion verification, and path distribution/naming for a 100-path case;
- barostat refusal before output.

### CUDA policy and precision

- CUDA built-in default and machine-configured CUDA;
- supported `single`, `mixed`, and `double` precision modes on hardware that supports them, with precision-appropriate tolerances;
- device index, CUDA-visible-device handling, local-rank and OpenMM device policies;
- no automatic CPU fallback;
- explicit `--cpu` and machine-configured CPU only in policy tests, never as evidence for a required CUDA lane;
- a rank-local CUDA initialization/runtime failure that terminates the complete MPI job promptly.

Record GPU model(s), count, driver, CUDA runtime/toolkit information, OpenMM version, precision, device assignment by rank, MPI implementation/version, package versions, test command, pass count, skips, failures, and wall time. If hardware cannot execute a required supported mode, report it as an unfulfilled acceptance criterion; do not silently skip it or claim completion.

## 8. Complete verification lanes

Run from a clean worktree/fresh clone without relying on untracked fixtures:

1. Full fast/non-GPU test suite with zero unexpected skips.
2. Full slow/GPU suite on real CUDA with zero skipped CUDA tests.
3. All exhaustive CUDA cases in section 7.
4. Real multi-rank REST2, rREST2, and AIS tests.
5. MPI missing/broken/inconsistent and rank-local-failure tests.
6. Direct-entry refusal tests proving no filesystem mutation.
7. Integrated cMD and AIS crash/restart tests.
8. Wheel build/install in a clean environment, then CLI and representative CUDA/MPI execution from outside the checkout; verify import origin is the installed wheel.
9. GitHub Actions on the exact final SHA. Clearly label CI as non-GPU if it is non-GPU; it does not replace the required local CUDA/MPI evidence.

Report exact commands, counts, skips, hardware, versions, timings, and output artifact hashes where relevant.

## 9. Documentation and completion discipline

After behavior and tests are correct, audit and update:

- root CLI and all command help;
- generated scripts and input comments;
- `README.md`;
- `CLAUDE.md`;
- cMD, REST2, rREST2, AIS, runner, configuration, and data-contract documentation;
- examples and release notes.

Correct stale statements, including:

- AIS “rerun interrupted paths” versus real mid-path resume;
- claims that AIS lacks mid-path restart;
- source atom-count comparison against `-p` and `-s`, not trajectory `-x`;
- the four-command CLI count;
- obsolete platform fallback authorities;
- any acceptance or CUDA claims not supported by exact evidence.

Document the AIS component definitions, column schema, equations, numerical tolerances, Hummer–Szabo use, and added energy-evaluation cost.

Commit coherent implementation and test changes directly to `dev`. Keep `CLAUDE_TASK.md` present during implementation and testing. Remove it only in a final cleanup commit after every required acceptance criterion—including real CUDA/MPI and installed-wheel execution—passes. Then require GitHub Actions to be green on that exact cleanup SHA.

If any required item is not executed or fails, keep this file and report the limitation honestly.

The final report must include:

- implementation commit SHA(s);
- cleanup/final SHA;
- exact-head GitHub Actions URL;
- concise architecture/change summary;
- full test commands, pass counts, skips, failures, and wall times;
- the CUDA function/branch coverage matrix;
- GPU/driver/CUDA/OpenMM/precision evidence;
- MPI rank/device and fail-closed evidence;
- cMD and AIS interruption/recovery evidence;
- AIS component-work invariant and evaluation-count evidence;
- wheel installation and import-origin evidence;
- remaining limitations.
