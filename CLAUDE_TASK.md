# Final corrective task: close the MD runtime and AIS acceptance blockers

Repository: `csy0000/MD-tools`

Target branch: `dev`

Implementation baseline: `600dbfa288eb2c23178a8457424d06ecc57297fd`

This is an implementation task, not another review. Read `CLAUDE.md` completely before editing.
Restore this file as `CLAUDE_TASK.md` on `dev`, keep it present throughout implementation and
verification, and remove it only in a final cleanup commit after every acceptance criterion below
passes on the exact implementation SHA.

Do not change the validated REST2 Hamiltonian, exchange rule, AIS work convention, or data
contract merely to simplify the fixes. Add regression tests that reach the intended behavior
before or alongside each correction. Do not delete, weaken, skip, xfail, or mask a scientific test.

## 1. Make CI truthful before interpreting any result

The workflow run for baseline `600dbfa` is displayed as successful even though pytest reported:

```text
14 failed, 1108 passed, 157 deselected, 1 warning
```

The pytest command is piped through `tee` without preserving pytest's exit status. Fix every shell
pipeline that can mask a failure using `set -o pipefail` or an explicit checked `PIPESTATUS`.
Add a small workflow regression check proving that a failing command upstream of `tee` makes the
step fail. Correct the stale workflow text that calls the package a three-command interface.

After the workflow is truthful, fix all 14 failures. Tests whose purpose is non-platform protocol
validation must explicitly select CPU in the non-GPU lane so missing CUDA cannot mask the intended
particle-count, input-format, HMR, collision, overwrite, or `--check` assertion. Conversely, tests
of CUDA policy must use the real CUDA resolver and must not substitute CPU.

## 2. Correct the AIS Hummer--Szabo state identity

Keep the work convention:

```text
delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)
```

The parameter switch is evaluated at frozen pre-switch coordinate `x_j`, followed by propagation
to observation coordinate `x_{j+1}`. The baseline writes a trajectory frame for `x_{j+1}` but
associates it with potential-basis values measured earlier at `x_j`. This is scientifically invalid
for Hummer--Szabo reweighting.

Separate the two concepts in code, schema, and documentation:

1. **Work-basis probe:** evaluated at the frozen pre-switch `x_j`; it decomposes incremental and
   accumulated work.
2. **Observation-potential probe:** evaluated at the reported/saved coordinate `x_t` under the
   reported `tau_t`; it supplies the energy basis used with `W_t` for Hummer--Szabo reweighting.

Never label a pre-switch basis as belonging to a post-propagation coordinate. A row that names a
`coordinate_frame_index` must contain potential components recomputed at that exact frame. If a
work observation has no saved coordinate because the trajectory cadence is different, the frame
index and observation-potential fields must be explicitly null/absent by schema rather than point
to a different frame. Provide an unambiguous frame-aligned subset or table for downstream HS use.

### Required three-group output

Let

```text
a = 1 - tau
lambda = a^2
```

Print these three observation-potential columns to the per-path and aggregate AIS CSV output:

```text
potential_non_scaled_kj_mol
potential_sqrt_scaled_kj_mol
potential_lin_scaled_kj_mol
```

Their exact identity is:

```text
U(tau, x) = U_non_scaled(x)
          + sqrt(lambda) * U_sqrt_scaled(x)
          + lambda       * U_lin_scaled(x)

          = U_non_scaled(x)
          + a   * U_sqrt_scaled(x)
          + a^2 * U_lin_scaled(x)
```

`sqrt_scaled` is the solute--environment coefficient group; `lin_scaled` is the fully scaled
solute--solute coefficient group. These names describe scaling with respect to `lambda`, avoiding
the ambiguity of calling them merely linear and quadratic in `a`.

Retain directly measured total potential energy and include a reconstructed total. Require the
direct and reconstructed totals to agree within the existing justified numerical tolerance for
implicit and explicit PME systems. If work-component columns are retained, name them distinctly,
for example `delta_work_*` and `total_work_*`; do not reuse the observation-potential names.

The aggregate HS-ready rows must contain at least:

```text
path_id, source_frame, observation_index, switch_step,
coordinate_frame_index, tau, total_work_kj_mol,
potential_non_scaled_kj_mol, potential_sqrt_scaled_kj_mol,
potential_lin_scaled_kj_mol, potential_reconstructed_kj_mol,
potential_direct_kj_mol
```

Document whether `delta_work_kj_mol` represents one switch or the sum since the previous emitted
work observation when work and switching cadences differ.

### Required AIS numerical tests

For both implicit and explicit PME systems:

- read coordinates back from genuine `AIS_trajNNNN.nc` files;
- reconstruct an OpenMM Context at the CSV row's `tau`;
- independently recompute the three basis components and direct potential;
- compare them with the CSV row associated with that exact frame;
- verify the work identity independently at every switch;
- use at least one schedule where switching, work-observation, trajectory, system-table, and
  checkpoint cadences are different divisors of `switching_steps`;
- test observation zero, an interior frame, the final frame, interruption, and resume;
- prove resume creates no duplicate or mismatched HS rows.

## 3. Make AIS evaluation cost and provenance complete

The baseline field named `switching_energy_evaluations` counts basis probes but omits direct energy
evaluations. Replace misleading provenance with separate and summed counters:

```text
basis_probe_energy_evaluations
direct_work_energy_evaluations
observation_energy_evaluations
total_potential_energy_evaluations
```

Define whether a counter counts `Context.getState(getEnergy=True)` calls, parameter pushes, or
both. Record both where they differ. Count discarded work from failed/interrupted generations.
Benchmark implicit and explicit cases after adding the frame-aligned observation probe, reporting
total useful evaluations, parameter updates, wall time, hardware, and path length. Do not turn a
machine-specific timing into a brittle pass threshold.

## 4. Complete preflight, run identity, resume, and overwrite transactions

Nothing may be created or modified until all predictable read-only validation succeeds. This rule
applies independently to `md-openmm md-run`, generated Python wrappers, `stage_main`,
`replica_main`, `ais_main`, and the directly callable REMD executor.

### AIS

- Compute the source digest and complete normalized run fingerprint during preflight.
- Check all output collisions and any existing `AIS_run.json` before `mkdir`, log creation, source
  frame-table creation, or checkpoint access.
- Bind source, System, topology, resolved configuration, tau schedule, seeds, selected frames,
  reporting schema, path count, column schema, and platform policy into the run identity.
- Include `final_state.xml` and every claimed artifact in the completed-path manifest and hash it.
- Implement `--overwrite` as a complete fresh-run transaction; removing only `AIS_run.json` is not
  overwrite. Never combine old path manifests, trajectories, checkpoints, or aggregate tables
  with a new identity.
- Compatible resume must follow only the last atomically committed checkpoint generation.

### cMD stages

- Include the DCD, state CSV, and phase-space NetCDF in collision inventory, checkpoint row/frame
  counts, truncation, completion records, and output hashes.
- A checkpoint transaction commits every appendable stream count. Resume truncates every stream
  to that generation before appending; it never infers progress from the longest file.
- Bind the System digest, topology digest, normalized resolved configuration, stage settings,
  seeds, platform, and output schema into the completion fingerprint.
- Recompute and verify the fingerprint, output hashes, and expected counts before declaring a stage
  already complete.
- Choose one clear resume contract: require `--resume`, or make interrupted resume automatic and
  remove the redundant option. Test and document the chosen contract.
- `--overwrite` must start cleanly and must not load an old checkpoint generation.
- Move fixed-tau scaling construction, force audit, solute classification, timestep/HMR checks,
  and all other predictable System-dependent validation into the prepared preflight result.
  Runtime code must consume that exact result rather than reload and re-resolve it after logs open.

### REST2 and rREST2

- Check output collisions before writing the output directory or helper files.
- A supplied `--groupfile` is an input; do not create an unused default group file.
- Generated `--verify-only` must be genuinely read-only.
- Validate extension-parent identity and completeness in preflight.
- Prepare and validate the protocol, System, topology, solute selection, state count, timestep,
  scaling plan, force audit, platform, device, and MPI world once, then pass the typed result into
  the executor unchanged.
- Implement `--overwrite` consistently through generated and direct executor paths.
- Inventory and fingerprint every helper and shared file.

Add refusal tests for each entry point proving malformed input, invalid System/timestep, collision,
incompatible identity, missing parent, unavailable platform, and invalid trajectory create no new
file and do not modify a pre-existing output directory.

## 5. Close the remaining MPI races and fail collectively

`md_tools.remd.mpi` remains the only module that imports or controls `mpi4py`.

- Reject an ordinary cMD stage launched with `COMM_WORLD.size > 1`; multiple ranks must never run
  the same serial stage against identical outputs.
- Under REST2/rREST2, rank zero writes shared helpers atomically and every rank verifies their
  digests before proceeding.
- Validate the rREST2 reservoir source in preflight. Rank zero alone writes `reservoir.yaml`
  atomically; every rank verifies the committed digest.
- A rank-zero preparation failure must be broadcast and fail collectively instead of leaving other
  ranks blocked at a barrier.
- Include post-preflight shared preparation and AIS output initialization in collective failure
  handling. A rank-local filesystem, CUDA, or reporter failure must abort/fail the complete world.
- Exercise these failures using real `mpirun`, with timeouts proving no surviving or hanging rank.

## 6. Build a truthful source-to-CUDA coverage contract

The current AST check mainly detects `Simulation` and `Context` constructors; it does not prove
coverage of every CUDA operation. Replace the broad claim with a real source inventory or narrow
the claim honestly.

The inventory must locate and classify at least:

- `openmm.Context` and `app.Simulation` construction;
- `step`, `minimizeEnergy`, and integrator stepping;
- `getState` calls requesting energies or forces;
- `setParameter`, `updateParametersInContext`, and context reinitialization;
- DCD/NetCDF/state reporters and checkpoint save/load;
- REST2 cross-Hamiltonian evaluation and exchange;
- rREST2 reservoir refresh and continuation;
- AIS direct-work evaluation, basis probes, parameter switching, observation probes, and resume.

Every discovered production site must map to at least one real-CUDA test lane. Exemptions must name
the source site and justify why CUDA execution is impossible or irrelevant. The source inventory
must fail when a new unclassified site is added.

### Required real-CUDA lanes

Run against the built wheel, outside the checkout, without importing repository sources:

1. cMD implicit and explicit, tau 0 and fixed tau > 0.
2. HMR/4 fs success and non-HMR/4 fs refusal.
3. CUDA single, mixed, and double precision where supported.
4. automatic local-rank devices and explicit `--device` selection.
5. explicit-solvent NPT at tau 0 and fixed-volume scaled stages.
6. cMD interruption/resume/overwrite/extension, validating DCD, state CSV, phase-space NetCDF,
   checkpoint counts, hashes, and completion identity.
7. REST2 implicit and explicit with real `mpirun`, including 2-, 4-, and 6-state ladders,
   exchanges, NetCDF state trajectories, checkpoint resume, and extension.
8. rREST2 under real MPI, including top-rung reservoir production, deterministic refresh,
   interruption/resume, and extension.
9. AIS implicit and explicit PME with genuine DCD and NetCDF sources.
10. Multi-rank AIS with deterministic path ownership and a 100-path run producing exactly
    `AIS_traj0000.nc` through `AIS_traj0099.nc`, with no duplicates or missing IDs.
11. AIS interruption/resume at checkpoint, trajectory, work-table, system-table, aggregate, and
    manifest transaction boundaries.
12. Independent implicit and explicit frame-aligned HS recomputation tests for the three required
    potential groups.
13. Genuine no-CUDA refusal, missing/broken `mpi4py`, launcher/communicator mismatch, and a
    rank-local CUDA failure, all creating no authoritative output and leaving no hung rank.
14. Direct generated-wrapper and direct executor parity for stage, REST2, rREST2, and AIS.

Short runs are acceptable, but each lane must exercise the named branch rather than merely construct
an object. Record exact command, installed package version and import origin, commit SHA, OpenMM
version, Python version, NVIDIA driver, CUDA runtime/toolkit, GPU model/UUID mapping, MPI
implementation/version, ranks, precision, wall time, and result. Record total work as energy/force
evaluations as well as elapsed time. Do not commit a static success matrix detached from the script
and raw machine-readable results that generated it.

## 7. Verification and completion discipline

Run, at minimum:

```bash
python -m pytest tests -m "not slow and not gpu"
python -m pytest tests -m "gpu or slow"
python -m build
```

Then install the wheel into a clean environment, change to a directory outside the checkout, prove
the import origin is the installed wheel, exercise all four CLI help routes, and run the complete
real-CUDA/MPI matrix above. Re-run exact-head GitHub Actions after the final implementation commit.

Acceptance requires:

- zero failing tests in every required lane;
- every skip/deselection listed with a reason and no required CUDA/MPI lane skipped;
- CI actually propagating pytest's exit status;
- no output from any failed preflight or `--check` invocation;
- HS potential components numerically matching the exact saved coordinates they name;
- direct and reconstructed potentials and work components independently agreeing;
- complete transactional resume/overwrite behavior for every stream;
- source-site-to-CUDA-lane coverage with no unclassified production site;
- documentation, `CLAUDE.md`, schemas, examples, help, release notes, and generated scripts agreeing
  with the implemented interface and output columns.

Commit coherent implementation and test changes to `dev`. Keep `CLAUDE_TASK.md` while testing.
After all local, installed-wheel, real CUDA/MPI, and exact-SHA CI checks pass, remove the task file in
a separate cleanup commit. Wait for CI on that cleanup SHA and require it to pass truthfully.

The final report must provide implementation and cleanup SHAs, the exact-head Actions URL, exact
commands and pass counts, skips with reasons, raw CUDA/MPI evidence, HS numerical tolerances and
maximum errors, interruption boundaries tested, installed-wheel import origin, timing/cost results,
and every remaining limitation. Do not claim completion if any required lane was unavailable.
