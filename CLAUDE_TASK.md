# Final execution-safety, recovery, and evidence correction

Work on `csy0000/MD-tools`, branch `dev`.

This instruction is anchored to implementation baseline:

```text
557198236d649b3e3973a4f4807d3480e1c0bee1
```

Read `CLAUDE.md` and this file completely before editing. Implement this task; do not write another review or plan. Preserve all validated REST2, rREST2, fixed-tau cMD, AIS work, scaling, exchange, source-frame, and trajectory-format mathematics.

The baseline substantially improved the package, but its cleanup commit claims completion while important lower-level correctness and recovery defects remain. This is the final consolidation pass. Correct the architecture instead of adding another safe outer wrapper around unsafe runtime functions.

## Non-negotiable scientific invariants

Do not change these conventions:

- Let `a = 1 - tau` and `lambda = a**2`.
- The Hamiltonian identity is
  `U(tau,x) = U_non_scaled(x) + sqrt(lambda) U_sqrt_scaled(x) + lambda U_lin_scaled(x)`.
- AIS work is
  `delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)`: switch at frozen coordinates, then propagate.
- Work-basis components are evaluated at the pre-switch coordinate `x_j`.
- Hummer-Szabo observation potentials are evaluated at the exact saved coordinate named by the row and under that row's tau.
- A row with no coordinate frame must not borrow potential values from another coordinate.
- Explicit solvent includes the complete PME reciprocal contribution, self-energy, and dispersion correction.
- REST2 and rREST2 run at one physical temperature, never rescale velocities on exchange, and write trajectories by thermodynamic state.
- Existing force-field, GB, CMAP, omega-exclusion, scaling, exchange, and reservoir rules remain unchanged.

The existing public AIS column names remain stable:

```text
potential_non_scaled_kj_mol
potential_sqrt_scaled_kj_mol
potential_lin_scaled_kj_mol
potential_reconstructed_kj_mol
potential_direct_kj_mol

delta_work_non_scaled_kj_mol
delta_work_sqrt_scaled_kj_mol
delta_work_lin_scaled_kj_mol
total_work_non_scaled_kj_mol
total_work_sqrt_scaled_kj_mol
total_work_lin_scaled_kj_mol
```

`AIS_hs.csv` remains a frame-aligned table suitable for later Hummer-Szabo reweighting. If convenient without compromising independent cadences, emit a row for every saved trajectory frame. Otherwise retain the validated frame-aligned subset and document that only cadence intersections appear.

## 1. Introduce prepared, immutable execution plans

The typed preflight result must be the object the runtime executes. A runtime may not call preflight and then independently reload or recompute the platform, MPI world, System, topology, selections, scaling, timestep, reservoir, schedule, or output policy.

Create mode-specific immutable plans, or extend the existing types, so they contain everything needed for execution.

### Stage plan

It must include at least:

- validated resolved configuration and fingerprint;
- parsed topology;
- deserialized and particle-count-validated System;
- solute selection and omega exclusions;
- fully prepared fixed-tau scaled System when applicable;
- resolved timestep and HMR evidence;
- ensemble, thermostat, barostat, restraint, and integrator settings;
- prepared platform/device request and resolved acceleration;
- complete expected-output inventory;
- run identity and existing-run disposition;
- MPI/serial coordination decision.

### Ladder plan

It must include at least:

- everything shared with the stage plan;
- validated tau ladder, state count, exchange schedule, and trajectory/checkpoint schedules;
- prepared scaling selection/System authority;
- validated continuation input;
- validated reservoir declaration and source identity for rREST2;
- group-file/helper contents and their expected digests;
- complete shared and per-rank output inventory;
- prepared MPI coordination and platform result.

### AIS plan

It must include at least:

- validated topology and deserialized System;
- prepared scaler/switcher;
- validated source trajectory format, atom count, frame count, digest, and selected frames;
- complete switching/reporting schedule;
- temperature, timestep/HMR, beta, and velocity policy;
- prepared MPI coordination and platform result;
- complete path ownership;
- complete shared, per-rank, per-path, checkpoint, and final-output inventory;
- complete run identity and existing-run disposition.

All predictable validation must finish before any output directory is created or modified. Execution must consume the plan without re-resolving those properties.

## 2. Validate AIS identity before every filesystem mutation

The baseline creates the AIS directory and opens rank reports before calculating the complete source identity and enforcing `AIS_run.json`.

Correct the order:

1. Read and validate all inputs.
2. Hash the source trajectory and other authoritative inputs.
3. select the deterministic source frames.
4. Build the complete fingerprint and expected inventory.
5. Inspect an existing output directory read-only.
6. Decide fresh run, compatible resume, verified completion, refusal, or overwrite.
7. Agree collectively under MPI.
8. Only then create or modify output.

An incompatible source, System, topology, tau schedule, seed policy, selected frames, reporting schema, path count, velocity policy, or relevant package/configuration identity must refuse without changing a single byte.

Represent actual rank-suffixed `.out` and `.log` files in the inventory. An overwrite after an MPI-size change must not retain stale rank reports.

Never remove arbitrary directories merely because their name starts with `path_`. Delete only manifest-owned paths or names matching the exact numeric path schema and verified as belonging to this run.

## 3. Correct AIS useful-cost and discarded-cost accounting

The baseline adds counters restored from a committed checkpoint to `discarded_energy_evaluations`. Those evaluations are not discarded: they produced the committed work, observations, frames, or state rows being resumed.

Use explicit, non-overlapping counters:

- direct work energy evaluations;
- work-basis probe evaluations;
- observation-potential evaluations;
- other scientifically justified energy evaluations;
- useful total;
- known discarded evaluations;
- paid total;
- parameter updates;
- probe wall time and, where measurable, total wall time.

A resumed path restores its committed useful counters and continues from them. It must produce the same final useful counters as an uninterrupted identical path.

Count discarded work only when durable attempt-level evidence proves it occurred after the last committed generation. A committed checkpoint cannot know how much later work was performed before a crash. If that cost cannot be recovered reliably, report it as unknown/unobservable rather than assigning committed useful work to the discarded category.

Document the exact relationships, including:

```text
useful_total = direct_work + work_basis_probe + observation_potential + other_useful
paid_total = useful_total + known_discarded
```

Add unit and integrated recovery tests for these identities.

## 4. Make AIS final publication crash-atomic

The baseline can publish `AIS_trajNNNN.nc` before committing the completion marker. A crash in that interval leaves a published trajectory, no completion record, and no staged trajectory from which the current resume code can continue.

Implement generation-based finalization:

1. Finish and fsync all staged path outputs.
2. Calculate and record their digests, schemas, row/frame counts, final protocol step, source-frame identity, fingerprint, and counters.
3. Validate the full staged generation.
4. Atomically publish a pointer/manifest selecting that completed generation.
5. Expose stable public output names without creating a state that cannot be recovered.
6. Remove checkpoint generations only after completion is durably committed.

A crash at any boundary must resume from the last committed checkpoint or recognize the fully validated completed generation. It may never fail merely because the stable trajectory was already renamed.

Completion verification must require the exact mandatory output-key set. It is invalid if a manifest omits the trajectory, observations, HS table, final state, or any state table required by the configuration. Unknown keys are also refused.

Inject failures:

- before and after each final output fsync;
- before and after each rename/link to a stable name;
- before and after completion-manifest write;
- before and after completion-pointer replacement;
- before and after checkpoint cleanup.

After each failure, a new invocation must either finish identically to an uninterrupted path or clearly refuse a corrupt state. No duplicated row/frame or mixed generation is allowed.

## 5. Complete cMD scientific preflight

Move every read-only or predictable cMD preparation step before output mutation:

- topology/System loading and particle-count comparison;
- solute and omega-exclusion resolution;
- fixed-tau scaling construction and force classification;
- timestep/HMR resolution from actual masses;
- restraint construction and selections;
- ensemble/barostat compatibility;
- reporter/trajectory format and cadence validation;
- complete identity/inventory/collision validation;
- platform Context smoke test.

`stage_main` must execute the prepared stage plan. It must not re-read the machine configuration or independently reconstruct the scientific policy after reports are opened.

Add regression tests for failures in fixed-tau scaling, omega exclusions, HMR/timestep, restraints, barostat/ensemble, topology/System agreement, CUDA Context creation, and output collisions. Every case must prove an absent target remains absent and an existing target is unchanged.

## 6. Strengthen cMD completion, resume, and overwrite

Completion verification must recompute the fingerprint from current authoritative inputs and validate every claimed output.

The manifest must cover, when applicable:

- final restart/State XML;
- DCD trajectory;
- state CSV;
- phase-space NetCDF;
- checkpoint pointer and committed generation while interrupted;
- resolved configuration;
- output/provenance records;
- exact expected frame and row counts.

Verify sha256, schema, counts, final step, stage identity, seeds, platform policy, and fingerprint. Changing the serialized System, topology, resolved configuration, selection, or input restart must prevent a false completed skip.

Keep the documented automatic-resume contract:

- a compatible interrupted stage with a valid committed checkpoint resumes automatically;
- a fully verified completed stage is skipped;
- anything else refuses until `--overwrite`.

Remove the redundant `--resume` option if it has no distinct semantics. Do not allow a flag to bypass collision checks without a valid checkpoint.

A real `--overwrite` removes or transactionally replaces every manifest-owned artifact, including outputs no longer requested by the new configuration. It must never remove unrelated neighboring files and must never load a generation it was asked to replace.

## 7. Preflight the entire all-in-one cMD chain

Before stage 1 starts, build and validate the prepared plans for every stage.

Use an explicit pending-parent representation for a restart that will be produced by an earlier stage in the same validated chain. A pending parent is allowed only during this all-in-one planning pass; it is not a general exemption for a missing `-c`.

Validate all stage-specific:

- lengths, reporting cadences, timestep/HMR, ensemble, tau, restraints, scaling, formats;
- parent/child restart connectivity;
- output inventories and cross-stage collisions;
- machine/platform policy.

Then execute the already-prepared plans in order.

Add a test with a deliberately invalid last stage and prove that stage 1 creates no directory, report, trajectory, checkpoint, or resolved configuration.

## 8. Consolidate generated and direct REST2/rREST2 execution

`replica_main`, generated wrappers, direct grouped executor entry, and any retained programmatic entry must use the same ladder plan.

Before output mutation, validate:

- protocol and all schedules;
- System/topology/selection/scaling plan;
- continuation input;
- tau ladder, replica count, `-ng`, launcher, communicator, and device placement;
- reservoir declaration, source files, velocity policy, and identity;
- all output extensions and path collisions;
- existing-run identity and overwrite/extension policy;
- content and digest of every generated helper.

Inventory must include:

- `reservoir.yaml`;
- `solute.yaml`;
- protocol helper;
- group file;
- shared logs and summaries;
- every rank report;
- every state trajectory;
- checkpoints and restart records;
- completion and provenance manifests.

`--verify-only` must remain entirely read-only even when the target is absent or invalid.

Generated `--overwrite` must reach the executor as the single overwrite policy. Do not update helpers and then allow the executor to refuse the outputs under a different force flag.

Remove the legacy ungrouped route if it is not a supported public/API workflow. If retained, give it the identical preflight and execution-plan contract; it may not call an arbitrary protocol after output creation.

Add direct-entry tests rather than relying on `md-run` to guard these paths.

## 9. Make every post-preflight MPI failure collective

Preflight agreement alone is insufficient. A rank-local failure while creating directories, opening a report, publishing helpers, constructing reporters, loading checkpoints, initializing a Simulation, or finalizing outputs must not leave other ranks blocked.

Use `md_tools.remd.mpi` as the only MPI authority.

At every initialization/finalization phase:

- catch local failure;
- communicate failure state/message to all ranks when the communicator remains usable;
- cleanly stop or abort the communicator;
- never let surviving ranks enter a collective that the failed rank cannot reach.

Test injected failures on rank 0 and a nonzero rank for REST2, rREST2, and AIS. Use real `mpirun`, enforce a timeout, require nonzero exit status, and prove there are no surviving processes or authoritative completion outputs.

## 10. Strengthen source-to-CUDA coverage

The current AST inventory is useful but narrower than the documentation claims. Expand it to identify all source sites that:

- create OpenMM `Context` or `Simulation`;
- integrate or minimize;
- get energy/coordinates/velocities/box state;
- set coordinates, velocities, box vectors, parameters, or State;
- push force parameters into a Context or reinitialize;
- save/load checkpoints;
- create DCD, NetCDF, phase-space, state, or thermodynamic reporters whose data comes from CUDA execution;
- create or aggregate CUDA-derived outputs.

Map every discovered site or branch to at least one real CUDA lane. Fail collection when a new source site is not classified. Do not hide unclassified sites by manually listing only high-level entry points.

Reconcile and define the counts of source sites, pytest cases, parametrized cases, and documented lanes.

## 11. Run the complete CUDA/MPI and installed-wheel matrix

All GPU evidence must use real CUDA devices. Do not substitute CPU/Reference/OpenCL, mocks, monkeypatches, or serial execution for a CUDA/MPI acceptance lane.

At minimum run:

### Core construction and stage execution

- implicit ff14SB/GBn2 and explicit ff14SB/TIP3P;
- explicit ff19SB/OPC;
- ligand Sage and GAFF2 paths that are supported;
- minimization, NVT, NPT, restraints, barostat, HMR 4 fs;
- fixed-tau implicit and explicit stages;
- DCD, state CSV, phase-space NetCDF, checkpoint/resume, overwrite, and completion verification;
- generated wrapper and `md-openmm md-run` parity.

### REST2/rREST2

- real `mpirun` REST2 with 2, 4, and 6 states;
- implicit and explicit systems;
- rREST2 with real reservoir coordinates and stored velocities;
- restart/extension and state-trajectory continuity;
- rank-to-device mapping and all supported device policies;
- single, mixed, and double precision where supported;
- injected rank-local failure and missing/broken `mpi4py` refusal.

### AIS

- implicit and explicit switching on CUDA;
- genuine DCD and NetCDF source reading;
- fresh Maxwell-Boltzmann and stored-velocity policies where supported;
- one path and 100 paths under real `mpirun -n 4`;
- independent observation, trajectory, state, and checkpoint cadences;
- frame-aligned HS recomputation in a new Context for all three basis groups and direct total;
- work-component sum and Hamiltonian reconstruction identities;
- interruption/resume at every checkpoint, stream, and final-publication boundary;
- uninterrupted versus resumed outputs and useful-cost counters;
- incompatible-run and overwrite behavior.

### Platform failures

- no CUDA available;
- invalid CUDA device;
- machine-configured CPU provenance;
- explicit `--cpu`;
- missing, malformed, duplicate-key, and invalid machine configuration;
- launcher/communicator/`-ng` disagreement.

Run at least:

```bash
python -m pytest tests -m "not slow and not gpu"
python -m pytest tests -m "gpu or slow"
python -m build
```

Build and install the wheel in a clean environment outside the checkout. Prove the import origin and exercise all four installed commands. Exercise installed-wheel CUDA and MPI paths, not only help output.

## 12. Produce reproducible evidence and correct the documentation

Write both a human-readable matrix and a machine-readable evidence artifact.

Record:

- exact commit SHA;
- UTC timestamp;
- exact commands and pytest selection;
- passed, failed, skipped, deselected, and xfailed counts;
- wall time per lane and total;
- Python, package, OpenMM, CUDA runtime/driver, and MPI implementation versions;
- GPU model, UUID, memory, and rank-to-device mapping;
- source-site inventory and mapped tests;
- installed-wheel path and import origin;
- energy-evaluation and parameter-update counters;
- known discarded work and the limits of what can be observed after a process crash;
- maximum AIS reconstruction and frame-roundtrip errors;
- skip reasons and remaining limitations.

Update `docs/release-notes/cuda-coverage-matrix.md`, `docs/release-notes/v0.5.0.md`, README, method documentation, and `CLAUDE.md` only after the behavior and evidence are real. Remove stale test counts, commit SHAs, CI links, lane counts, and unsupported “all criteria pass” claims.

## Required regression tests

Tests must reach the intended validation or crash boundary. An argparse failure does not prove a runtime preflight rule.

Add tests for at least:

- AIS incompatible identity without any filesystem change;
- AIS overwrite after changed MPI size with no stale rank reports;
- exact manifest-owned cleanup without deleting an unrelated `path_notes` directory;
- restored useful-cost counters across resume;
- honest handling of unobservable post-checkpoint cost;
- crash after stable trajectory publication but before completion commit;
- missing mandatory completion-manifest keys;
- modified System/topology/output after cMD completion;
- damaged or truncated DCD, state CSV, phase-space NetCDF, and restart;
- automatic resume only with a compatible valid checkpoint;
- complete cMD overwrite inventory;
- invalid late stage in all-in-one mode with no first-stage output;
- generated and direct REST2/rREST2 refusal paths;
- missing/invalid reservoir before output;
- `--verify-only` on absent and corrupt targets without modification;
- generated overwrite propagation;
- rank-zero and nonzero-rank initialization/finalization failures;
- AST/source inventory failure for a deliberately unclassified CUDA-derived reporter site.

## Implementation discipline

- Write failing regression tests before or alongside each correction.
- Prefer deleting duplicate authorities over wrapping them.
- Do not add a fifth command or a second executable.
- Do not duplicate scientific runtime code into generated scripts.
- Do not weaken, skip, xfail, or delete scientific tests to make the suite green.
- Do not manufacture CUDA/MPI evidence with mocks.
- Preserve unrelated user changes.
- Commit coherent implementation slices to `dev`.
- Keep this task file until all local, CUDA/MPI, wheel, and documentation acceptance criteria pass.
- Remove this file only in a separate final cleanup commit after all evidence is committed.
- After cleanup, require GitHub Actions to pass on that exact final SHA.

## Completion report

Report:

- implementation commit SHA(s);
- cleanup commit SHA;
- exact-head GitHub Actions URL;
- exact commands and test counts;
- CUDA device and MPI evidence;
- installed-wheel origin and smoke-test evidence;
- AIS decomposition/reconstruction tolerances;
- uninterrupted-versus-resumed scientific and cost-accounting evidence;
- crash-finalization evidence;
- source-to-CUDA coverage counts;
- skipped tests and explicit reasons;
- every remaining limitation.

Do not report “all criteria pass” unless each item above has direct code and test evidence.
