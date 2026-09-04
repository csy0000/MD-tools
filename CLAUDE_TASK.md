# Final scientific closure: rREST2 CV semantics, completion, cost, and CUDA evidence

## Baseline and scope

Implement this task on `dev`, starting from:

`941b205156910b3929b24b05d9e51db3a22c6369`

This is a narrow corrective pass over the CV implementation already present. Do not redesign the CLI, trajectory formats, REST2/rREST2 Hamiltonian, exchange rules, reservoir probability-one rule, AIS work convention, MPI authority, CUDA-first policy, or output-directory architecture.

Four acceptance gaps remain:

1. rREST2 can evaluate a purported pre-exchange CV from a post-refresh reservoir configuration.
2. REST2/rREST2 completion and provenance do not own or verify their CV files.
3. CV evaluation counts and wall times are computed in memory but never durably recorded.
4. The committed CUDA matrix attributes CV coverage to tests that explicitly use `--cpu` or do not enable CV reporting.

Correct all four together and add tests that exercise the actual scientific paths.

## 1. Preserve the true pre-exchange configuration in rREST2

The ladder CV convention remains pre-exchange: a row at an exchange boundary describes the configuration that was propagated at that state up to that step, before a swap or reservoir replacement.

At the baseline, the driver snapshots only `state_to_walker`. It then calls `_exchange()`. For rREST2, `_apply_reservoir()` replaces an entry in `state["configurations"]`. The later CV observation uses that modified list while labelling the row pre-exchange.

Fix the event transaction:

- before calling `_exchange()`, retain the complete walker-indexed pre-exchange configurations required for CV evaluation;
- do not retain merely the mapping;
- ensure later exchange or reservoir operations cannot mutate the saved position/box values;
- calculate every pre-exchange CV from the saved pre-exchange configuration and saved mapping;
- continue writing the post-exchange/post-refresh configurations to the established state trajectories;
- when a state trajectory frame does not contain the configuration used by the CV row, leave `trajectory_frame_index` empty;
- in particular, a reservoir-refreshed state must not name the post-refresh frame even when its walker index did not change;
- never use `-1` as “no frame.”

Do not change the accepted-exchange convention simply to make every CV row reference a frame. Empty is correct when no saved state frame holds that coordinate.

Add an rREST2 regression test with a deterministic reservoir refresh. It must independently calculate the expected torsion from the propagated pre-refresh configuration, demonstrate that the reservoir sample has a different torsion, and prove that the emitted row contains the pre-refresh value. It must also verify the frame-index field is empty for the refreshed state and remains correct for unaffected states.

Exercise stored-velocity and Maxwell-velocity policies where they share the configuration-replacement path.

## 2. Make ladder CV files authoritative completed outputs

The typed output inventory now names `remdN.cv.csv` and `remdN.cv.json`, but the authoritative REST2/rREST2 completion manifest and validators still do not.

For every enabled ladder CV series, record in the authoritative completion manifest:

- state index and tau;
- CSV filename, SHA-256, byte size, exact row count, header, first step, final step and interval;
- JSON sidecar filename, SHA-256 and byte size;
- CV schema version and definition digest;
- resolved atom indices and column order;
- units, wrapping/sign convention, periodic convention and exchange phase;
- expected step grid;
- CV evaluation count and wall time.

Completion must not be committed unless every state file:

- exists and is readable;
- has the exact expected header;
- contains `0, interval, ..., total_steps` exactly once;
- reports its own state index and fixed tau;
- has a valid walker index at every row;
- uses the declared pre-exchange phase;
- has an empty or valid in-range trajectory-frame index;
- matches its sidecar and resolved definition.

Update every completion consumer:

- `restart.json`;
- the human/machine ladder log;
- `validate_replica_output`;
- completed-run verification;
- `--resume` and `--extend` parent validation;
- MD-data registration/provenance discovery.

Deleting, truncating, editing, replacing or swapping any ladder CV CSV or sidecar after completion must make validation fail. An extension must not proceed from such a parent.

A CV-disabled run must record that no CV series belongs to it and must not inherit stale CV files.

## 3. Protect the committed CV prefix during continuation

A committed row count alone detects a short file but not a modified committed prefix.

For cMD, REST2/rREST2 and AIS checkpoints, bind the committed CV prefix into the same generation transaction as the Context/configuration state:

- committed row count;
- digest of the CSV header plus exactly the committed rows;
- sidecar digest;
- schema/definition identity;
- accumulated CV cost counters.

A file may legitimately be longer than the selected checkpoint after a crash. Validate the committed prefix against its recorded digest, then truncate the uncommitted tail. Do not hash the uncommitted tail as part of the selected generation.

Before appending, parse and validate the committed prefix:

- exact column set;
- numeric finite CV values;
- exact monotonic step grid without duplicates or gaps;
- correct state/path/source identifiers;
- valid optional observation/frame indices;
- sidecar agreement.

Reject a mutated committed value, step, walker, tau, path identity, header or sidecar before changing any byte. Legacy checkpoints without sufficient CV prefix metadata must fail with a clear compatibility message rather than guessing.

Add fault tests at:

- before CV row;
- after CV row;
- before checkpoint commit;
- after checkpoint data generation;
- after checkpoint pointer commit;
- during final completion publication.

Compare resumed CV values and metadata against an uninterrupted reference, not only the list of steps.

## 4. Persist CV cost accounting

`CVSeries.cost()` and `StateCVSet.cost()` currently expose `cv_evaluations`, `cv_seconds` and `cv_rows`, but the runtimes do not persist them.

For cMD, fixed-tau cMD, REST2, rREST2 and AIS:

- record CV evaluations, elapsed seconds and rows separately from Hamiltonian energy/force evaluation counters;
- checkpoint the accumulated counters and restore them on resume;
- distinguish segment-local cost from cumulative cost when extending a completed ladder;
- aggregate AIS cost across completed paths deterministically;
- ensure a resumed run does not reset, omit or double-count earlier CV cost;
- never count a position-only torsion evaluation as an energy evaluation.

Record the counters in the appropriate human-readable output, machine/provenance log, checkpoint metadata and completion manifest.

Tests must verify:

- `cv_evaluations` equals the number of torsions actually evaluated, accounting for multiple named torsions and per-state/per-path rows;
- `cv_rows` equals written rows and is not confused with evaluations;
- `cv_seconds` is finite and non-negative;
- cumulative counters survive resume;
- energy-evaluation counters are unchanged by the bookkeeping;
- enabled and disabled CV runs are distinguishable.

Do not compare exact wall times across runs.

## 5. Complete AIS CV verification

AIS now hashes and fsyncs `cv.csv` and `cv.json`, but completion verification must also validate their scientific structure.

For a CV-enabled path, require:

`expected_cv_rows = switching_steps / cv_interval_steps + 1`

Validate before committing completion and whenever a completed path is skipped:

- manifest `cv_rows` equals the expected count;
- actual CSV data-row count equals `cv_rows`;
- steps are exactly `0, interval, ..., switching_steps`;
- step 0 and the final step occur exactly once;
- path index, source-frame index, tau, time and optional observation/frame indices are correct;
- the sidecar matches the resolved definition and path fingerprint.

Add AIS damage tests for truncation, row mutation, step mutation, deletion and sidecar replacement. These must exercise AIS path completion rather than only cMD completion.

Extend the AIS interruption tests to compare the resumed `cv.csv`, sidecar identity and CV cost counters with an uninterrupted reference at every relevant transaction boundary.

Preserve the existing `non_scaled`, `sqrt_scaled` and `lin_scaled` work decomposition and the Hummer–Szabo coordinate-alignment convention.

## 6. Truthful CUDA and MPI coverage

Do not cite a CPU test as CUDA evidence.

At the baseline:

- `test_ais_cv_output.py` explicitly passes `--cpu`;
- the cited AIS cases in `test_md_run_mpi_gpu.py` do not enable collective variables;
- the new cMD and REST2 CV restart tests explicitly pass `--cpu`;
- the MPI command reported in the evidence runs general MPI tests, not the new CV lifecycle tests.

Create actual CUDA tests, without `--cpu`, that assert the recorded platform is CUDA:

1. cMD CV-enabled fresh run and interruption/resume.
2. Fixed-tau phase-space plus CV interruption/resume.
3. REST2 CV-enabled fresh run and interruption/resume.
4. rREST2 CV-enabled deterministic reservoir refresh, including the pre-refresh semantic test.
5. AIS CV-enabled fresh run and interruption/resume with the three-group decomposition.
6. Explicit-solvent AIS CV and HS alignment where feasible.

Create real multi-rank CUDA tests for:

- REST2 CV continuation;
- rREST2 CV continuation plus reservoir refresh;
- AIS CV continuation and deterministic aggregation;
- root and non-root failures during CV evaluation, CV writing, checkpoint commit and completion validation.

Use timeouts and require fail-closed communicator termination without a surviving or hanging rank.

The CUDA coverage inventory must connect a source function to a test that actually executes that function on CUDA. Test-name strings are not evidence. Where practical, record runtime execution evidence or assert output provenance proving the CV-enabled invocation used CUDA.

Regenerate the CUDA matrix only after these tests pass. Remove false mappings, particularly any claim that `test_ais_cv_output.py` is a CUDA lane while it invokes `--cpu`.

## 7. Required focused regression tests

At minimum, the final suite must include:

- rREST2 pre-refresh versus post-refresh torsion discrimination;
- no frame reference for the refreshed state's pre-exchange CV;
- REST2 and rREST2 completed-manifest mutation matrix;
- extension refusal from a damaged CV parent;
- committed-prefix mutation refusal before truncation;
- resumed ladder values matching an uninterrupted reference;
- AIS exact CV row/grid validation;
- AIS completion mutation matrix;
- persisted and resumed CV cost counters in every protocol;
- real CUDA execution of each CV runtime;
- real MPI+CUDA REST2, rREST2 and AIS CV continuation;
- installed-wheel fresh and resumed CV workflows outside the checkout.

Tests must reach the intended path. Do not accept an argparse failure, a preflight failure unrelated to the assertion, a serial substitute for MPI, or a CPU invocation as CUDA evidence.

Do not weaken, skip, xfail or delete scientific tests to obtain a green result.

## 8. Documentation and evidence

Correct the statement that CV definitions and outputs are already bound into every completion manifest. That statement becomes true only after the ladder manifest and validator changes above.

Update:

- `docs/collective_variables/README.md`;
- CLI/generated help where relevant;
- `CLAUDE.md`;
- release notes;
- CUDA coverage JSON and Markdown.

The evidence record must distinguish:

- exact-head GitHub CI, which has no CUDA runner;
- local CPU/fast tests;
- local real-CUDA tests;
- real MPI+CUDA tests;
- installed-wheel tests.

For each lane, provide the exact command, commit SHA, pass/fail/skip/deselection counts, wall time, GPU model/device mapping, OpenMM/CUDA/MPI versions and whether CV reporting was enabled.

Explicitly report the rREST2 reservoir event exercised, the CV row/frame semantics observed, the completion-mutation cases, resume boundaries and cost-counter checks.

## 9. Completion discipline

Implement coherent runtime and regression-test commits directly on `dev`.

Keep `CLAUDE_TASK.md` during implementation. Remove it only in a separate cleanup commit after:

- every requirement above passes;
- complete fast and slow suites pass;
- the dedicated real-CUDA CV suite passes;
- real MPI+CUDA CV continuation and injected-failure tests pass;
- installed-wheel fresh/resume tests pass outside the checkout;
- CI is green on the implementation SHA.

Then wait for CI on the exact cleanup SHA and require it to be green.

The final report must provide:

- implementation and cleanup SHAs;
- exact-head CI URL;
- exact commands, counts, skips, deselections and wall times;
- CUDA devices and platform provenance;
- MPI ranks and device mapping;
- rREST2 pre-refresh CV evidence;
- ladder and AIS completion-mutation evidence;
- committed-prefix and resume evidence;
- CV cost accounting evidence;
- installed-wheel import origin and workflow evidence;
- every remaining limitation.
