# Final CV lifecycle and restart-accuracy closure

## Baseline and purpose

Implement this task on `dev`, starting from implementation baseline:

`67b88d8d3ed1bbc439038418caebe9203549afd6`

The current implementation adds useful torsion reporting and AIS REST2-basis decomposition, but its completion claim is premature. This task closes the remaining CV lifecycle, restart, ownership, and coordinate-alignment defects in one pass.

Do not change the validated REST2/rREST2/AIS Hamiltonian mathematics, exchange rules, AIS work convention, trajectory formats, MPI authority, CUDA-first policy, or public four-command CLI. Do not add another executable or a fifth `md-openmm` command.

The goal is not merely to make the dedicated CV tests green. A CV-enabled run must remain correct across fresh execution, interruption, continuation, completion validation, overwrite, installed-wheel execution, CUDA, and MPI.

## 1. Use one absolute-step convention after OpenMM checkpoint restore

OpenMM checkpoints restore the Context step count. A reporter must not add the already completed step count a second time.

At the baseline:

- cMD constructs `CVReporter(..., step_offset=done)`;
- `CVReporter.report()` computes `step_offset + simulation.currentStep`;
- `PhaseSpaceReporter` uses the same pattern.

After a resume at step N, this can label the next observation as approximately `2N + interval` even though the Context already reports the absolute step.

Correct this throughout the stage runtime:

- make `simulation.currentStep` the single absolute-step authority after loading a checkpoint;
- remove or redefine inert/dangerous offset arguments rather than leaving two step conventions;
- keep reporter cadence aligned to the absolute grid;
- ensure `time_ps = absolute_step * timestep_fs / 1000`;
- ensure trajectory-frame indices remain global indices into the complete continued trajectory;
- ensure checkpoint `steps_done`, CV steps, phase-space steps, state rows, and final completion counts all describe the same absolute timeline.

Add integrated interruption/resume tests for both ordinary cMD CV reporting and fixed-tau phase-space reporting. Interrupt at a nonzero committed generation, resume, and require the exact expected steps, times, frame indices, row counts, no duplicates, no gaps, and no values beyond the requested final step.

## 2. REST2/rREST2 CV series must include step 0 and the final step

The accepted contract is universal:

- every enabled CV series includes step 0 exactly once;
- every enabled CV series includes the final production step exactly once;
- all intermediate rows lie on the declared uniform cadence;
- minimization produces no CV series.

The baseline ladder opens `remdN.cv.csv` but first observes only after propagation. Its test explicitly expects `5, 10, ..., 40`, which codifies the defect.

For a fresh REST2/rREST2 run:

- after any declared pre-production equilibration and immediately before production step 0, gather the configurations occupying every thermodynamic state;
- write one pre-exchange CV row per state at step 0;
- use `exchange_attempt=-1`;
- use the identity state-to-walker mapping in a fresh run;
- leave `trajectory_frame_index` empty unless an actual state trajectory frame is saved at step 0;
- retain the documented pre-exchange convention at later exchange boundaries.

A 40-step run with CV interval 5 must contain `0, 5, ..., 40`, exactly once in every state file.

## 3. Implement real REST2/rREST2 CV continuation

At the baseline, `_continue()` does not reopen `cv_states`, and the ladder checkpoint carries no committed CV-row count. A resumed ladder therefore cannot preserve the CV series correctly.

Treat the per-state CV files as coordinated appendable scientific streams:

- every committed checkpoint generation records the number of committed CV rows per state, or one count only after proving all state files have the same count;
- checkpoint commit ordering must ensure the count never leads durable rows;
- continuation validates all CV files and sidecars before modifying anything;
- continuation rejects a missing, shorter, malformed, differently mapped, differently signed, differently unit-labelled, or scientifically incompatible CV series;
- truncate rows written after the selected committed checkpoint generation;
- reopen all state series at the committed count;
- append the repeated dynamics without duplicate or missing steps;
- a changed CV definition, atom resolution, topology, interval, periodic convention, units, or output schema refuses continuation;
- legacy runs without the required CV continuation metadata must fail with a clear compatibility message rather than guessing.

Exercise REST2 and rREST2 interruption at boundaries on both sides of CV write and checkpoint commit. Resume under real MPI and compare the final CV CSVs byte-for-byte, or numerically plus canonical sidecar equality, with an uninterrupted deterministic reference where appropriate.

## 4. Correct REMD trajectory-frame alignment

A CV row that names `trajectory_frame_index=k` must have been evaluated on exactly `remdN.nc[k]`.

At the baseline, CV observation occurs before the coincident state trajectory frame is written and receives the previous `state["frame_index"]`. On the first coincident event this may be `-1`; later it points one frame behind.

Make the write transaction unambiguous:

- preserve the pre-exchange CV convention;
- when CV and state trajectory cadences coincide, associate the CV with the frame written from the same gathered configuration at that same step;
- pass the forthcoming index or reorder the state-frame/CV writes without moving CV evaluation after exchange;
- leave the field empty when no state trajectory frame exists at that step;
- never write `-1` as a stand-in for no frame.

Add a test that loads each referenced NetCDF frame and independently recomputes every reported torsion. Test a cadence finer than the trajectory, coincident CV/frame/exchange events, accepted exchanges, and changing state-to-walker mappings.

## 5. Make CV outputs part of the authoritative output inventories

Use the typed preflight inventories as the single authority for collision detection and transactional `--overwrite`.

### cMD

The actual `CVSeries` sidecar for `<stage>.cv.csv` is `<stage>.cv.json`. The baseline stage inventory incorrectly names `<stage>.cv.yaml`.

Correct the inventory to name every file actually produced. Do not conflate:

- the input `cv.yaml`;
- a content-addressed/resolved input copy, if one is intentionally produced;
- the output interpretation sidecar `<stage>.cv.json`.

Every declared file must exist when expected and must be governed by collision and overwrite policy.

### REST2/rREST2

For every state, add both:

- `remdN.cv.csv`;
- `remdN.cv.json`.

They must participate in preflight collisions, overwrite replacement, checkpoint/continuation validation, completion, provenance, and registration.

### AIS

Add the global `AIS_cv.csv` to the root inventory. Per-path `cv.csv` and `cv.json` remain owned by their path directory but must also be explicit in the path completion manifest.

A CV-disabled overwrite or fresh run must not leave a stale `AIS_cv.csv`, `remdN.cv.csv`, or CV sidecar from an earlier CV-enabled calculation.

## 6. Bind CV outputs into completion and provenance

A run may not be considered completed if any enabled CV file is absent, truncated, malformed, mutated, unreadable, inconsistent with its sidecar, or has the wrong number of rows.

### cMD

Retain and verify:

- CSV and sidecar digests and sizes;
- exact header and expected rows;
- exact step grid;
- resolved atom mapping and definition digest;
- checkpoint committed count for a resumed stage.

Ensure the corrected JSON sidecar path is used everywhere.

### REST2/rREST2

Add to the authoritative completion manifest and human/machine provenance:

- every state CV CSV and sidecar;
- hashes and sizes;
- exact row count and step grid;
- state index, tau, walker/exchange convention, interval, units, atom mapping, and definition digest;
- CV evaluation count and wall time kept separate from energy-evaluation accounting.

Completed-output validation and extension/continuation parent validation must verify these records.

### AIS

The baseline path `completed.json` records `cv_rows` but omits `cv.csv` and `cv.json` from `outputs`. Final fsync and validation also omit them.

Correct finalization so that:

- both files are flushed/fsynced before the completion commit;
- both are hashed into `completed.json`;
- exact header, row count, step grid, endpoint inclusion, source/path identity, indices, units, atom mapping, and definition digest are validated;
- the completion manifest is committed only after those checks;
- mutation or deletion prevents the path from being skipped as completed;
- `AIS_cv.csv` is assembled only from fully verified path manifests;
- aggregate reconstruction is deterministic across MPI worker counts and resume.

Do not silently regenerate a scientifically mutated per-path CV file from a trajectory while still calling the original path completed.

## 7. Preserve and verify the AIS three-group work decomposition

Keep the existing REST2 lambda-basis decomposition and its terminology:

- `non_scaled`;
- `sqrt_scaled`;
- `lin_scaled`.

Continue writing incremental and cumulative component work columns to the per-path observations and `AIS_work.csv`, final component totals to `AIS_paths.csv`, and frame-aligned group potentials to `AIS_hs.csv`.

Retain the two-coordinate distinction:

- work components are evaluated at the frozen pre-switch coordinate;
- Hummer-Szabo observation potentials are evaluated at the coordinate actually saved and named by `coordinate_frame_index`.

Require:

`delta_work_total = delta_work_non_scaled + delta_work_sqrt_scaled + delta_work_lin_scaled`

and the analogous cumulative identity within the documented precision-dependent tolerance.

Do not derive the directly measured total from the components. Keep the independent direct-versus-reconstructed potential check at saved HS frames.

Add resume and completed-file mutation tests with the decomposition enabled, proving schema/version, component accumulators, evaluation counters, and HS alignment survive interruption exactly.

## 8. Required regression tests

Tests must reach the intended runtime behavior rather than pass through an earlier argparse refusal or a mock that bypasses the failing path.

At minimum add:

1. cMD CV interruption/resume from a nonzero step with exact absolute steps and times.
2. Phase-space interruption/resume with exact absolute step labels.
3. REST2 and rREST2 step-0/final inclusion.
4. REST2 and rREST2 interruption/resume without duplicate or missing CV rows.
5. Crash/fault injection before and after ladder CV rows and checkpoint commits.
6. Independent torsion recomputation from every REMD row carrying a trajectory-frame index.
7. AIS CV interruption/resume at stream/checkpoint/finalization boundaries.
8. AIS decomposition accumulator and HS-frame alignment after resume.
9. Completion refusal after truncating, deleting, replacing, or mutating every type of CV CSV and sidecar.
10. Definition-change refusal for indices, selectors, order, sign convention, units, periodic convention, interval, and schema version.
11. Collision and `--overwrite` tests for cMD, ladder, AIS path, and global aggregate CV outputs.
12. CV-enabled to CV-disabled overwrite proving no stale CV artefact remains.
13. Direct generated-script and `md-openmm md-run` parity.
14. Installed-wheel tests outside the checkout for fresh and resumed cMD, REST2, rREST2, and AIS with CV enabled.

Update the existing REMD cadence test so it no longer asserts that step 0 is absent.

Do not weaken, skip, xfail, or delete scientific tests to obtain a green result.

## 9. CUDA and MPI verification

Run all existing lanes plus the new tests:

1. Complete fast/non-GPU suite.
2. Complete slow/GPU suite on real CUDA devices.
3. Real multi-rank CUDA REST2 and rREST2 with CV enabled, including accepted exchanges and resume.
4. Real multi-rank CUDA AIS with CV and three-group decomposition enabled, including interruption/resume.
5. Inject root and non-root failures in CV evaluation, CV writing, trajectory alignment, checkpointing, and completion finalization. Require fail-closed communicator termination with no surviving or hanging rank.
6. Re-run the machine-readable CUDA source-function inventory. Every source function that creates a CUDA Context, integrates, evaluates energy, changes Context parameters, saves/loads checkpoints, or produces scientific output from CUDA state must remain classified and exercised or explicitly proven not to reach a device.
7. Build a wheel, install it into a clean environment, run outside the checkout, verify import origins, and exercise all four public commands.

The GitHub workflow may remain the non-GPU portability lane, but the final evidence must distinguish exact-head CI from separately executed real CUDA/MPI evidence.

## 10. Documentation and completion discipline

Audit README, method documentation, CLI help, generated comments, examples, `CLAUDE.md`, and release notes for the corrected behavior:

- step 0 and final step in every CV series;
- pre-exchange REMD state convention;
- exact meaning of trajectory-frame indices;
- restart truncation/append contract;
- actual JSON sidecar names;
- AIS three-group work and frame-aligned HS potential columns;
- separate CV and energy-evaluation cost accounting.

Do not preserve the current statement that every acceptance criterion passed. Add a new evidence record tied to the final implementation SHA with exact commands, pass/fail/skip counts, wall times, GPU models, CUDA/OpenMM versions, MPI launcher/version, rank/device mapping, installed-wheel import origin, and any limitation.

Implement and test in coherent commits on `dev`. Keep `CLAUDE_TASK.md` during implementation. Remove it only in a separate final cleanup commit after:

- every requirement above passes;
- complete fast and CUDA/slow suites pass;
- real CUDA/MPI resume and failure-injection tests pass;
- the installed wheel passes outside the checkout;
- CI is green on the implementation commit.

After removing the task, wait for CI on the exact cleanup SHA and require it to be green.

The final report must provide:

- implementation commit SHA;
- cleanup commit SHA;
- exact-head CI URL;
- exact commands, counts, skips, and wall times for every lane;
- GPU/CUDA/OpenMM and MPI evidence;
- CV numerical, alignment, restart, overwrite, completion-mutation, and installed-wheel evidence;
- AIS work-decomposition and HS-alignment evidence;
- any remaining limitation.
