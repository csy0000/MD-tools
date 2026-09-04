# Final CV accounting, provenance, and CUDA/MPI closure

## Authority and scope

Work on `csy0000/MD-tools`, branch `dev`.

The exact implementation baseline for this task is:

```text
1e83c66b4b6eacd15dda69051fb5d5339599874b
```

This is a focused corrective task following the scientific-closure audit. Preserve all validated cMD, REST2, rREST2, AIS, REST2 scaling, exchange, reservoir-refresh, checkpoint-transaction, CV-prefix, trajectory, and work-decomposition behavior. Do not redesign the scientific methods.

Implement every requirement below. Do not treat green existing tests or the current release-note claims as proof that the requirements are already satisfied.

## 1. Define and correctly persist CV cost

The current implementation increments `cv_evaluations` once per reporter call even when that call evaluates multiple named torsions. That name is scientifically misleading.

Use the following unambiguous definitions everywhere:

- `cv_observations`: number of configurations on which the CV reporter evaluated the complete configured CV set.
- `cv_evaluations`: number of scalar CV values actually evaluated. For a fixed definition containing `N_cv` torsions, one observation increments this by `N_cv`.
- `wall_seconds`: measured wall time spent evaluating the configured CV set, excluding CSV serialization, hashing, validation, and unrelated simulation work.

Persist two scopes in authoritative checkpoints, completion manifests, human-readable output where appropriate, and machine/provenance logs:

```yaml
collective_variable_cost:
  segment:
    cv_observations: ...
    cv_evaluations: ...
    wall_seconds: ...
  cumulative:
    cv_observations: ...
    cv_evaluations: ...
    wall_seconds: ...
```

Semantics:

- `segment` covers only work performed by the current invocation.
- `cumulative` covers the complete logical simulation/path across every invocation.
- On a fresh run, segment and cumulative values are equal.
- On continuation, restore the last committed cumulative counters before any new CV evaluation. Reset only the segment counters.
- Every newly committed checkpoint generation must contain cumulative counters including all prior generations.
- A second or later interruption/resume must not lose or double-count earlier cost.
- Completed-path skip/verification must not count validation or CSV reads as new CV evaluations.
- All counters must be non-negative and finite; integer counters must be integers.
- Do not silently retain an old flat field with a conflicting meaning. If backward compatibility is needed, migrate it explicitly and document the schema/version behavior.

Apply this consistently to:

- conventional/fixed-tau MD;
- REST2;
- rREST2;
- every AIS path and the deterministic global AIS aggregation.

For multi-state or multi-path aggregation, document whether the reported total is the sum over states/paths and retain per-state/per-path records so the total is auditable.

## 2. Restore cost across repeated continuation

The resumed CV writers must consume the committed cost state, not only the committed row count.

Correct the shared CV series and state-set APIs so callers cannot accidentally restore rows while discarding counters. Prefer a typed committed-prefix/cost object over loosely related integer arguments.

Add tests with at least two consecutive interruptions followed by final completion for:

- cMD or a fixed-tau stage;
- REST2;
- rREST2;
- AIS.

For each protocol, compare the resumed result to an uninterrupted reference and prove:

- identical CV CSV headers, row identifiers, step grids, and CV values;
- no duplicated or missing rows;
- exact cumulative `cv_observations`;
- exact cumulative scalar `cv_evaluations`;
- segment counters for the final invocation only;
- cumulative counters survive every checkpoint generation;
- completed re-entry performs no new CV evaluation.

Use a CV definition with at least two named torsions so a call-count implementation cannot pass.

## 3. Complete real MPI + CUDA CV coverage

The existing `test_cv_mpi_cuda_lanes.py` exercises REST2 only. Its description and the evidence report must not imply that rREST2 and distributed AIS were tested.

Add genuine multi-process, real-CUDA tests, with `--cpu` absent and fail-on-no-CUDA behavior, for:

### rREST2

- At least two MPI ranks and one CUDA device assignment per rank according to machine policy.
- CV reporting enabled with at least two torsions.
- A deterministic reservoir refresh forced during the test.
- Prove the refreshed state records the pre-refresh configuration and has no falsely associated post-refresh trajectory frame.
- Prove an unaffected state references the correct post-exchange frame.
- Interrupt and resume through MPI.
- Compare the complete resumed CV series and cumulative cost with an uninterrupted reference.
- Cover the supported reservoir velocity policies through parametrization where feasible; at minimum cover the default stored-velocity policy and one resampling policy.

### AIS

- At least two MPI ranks with multiple globally numbered paths distributed across ranks.
- CV reporting and the AIS work decomposition enabled.
- Interrupt and resume with a changed rank-to-path scheduling opportunity if supported, while preserving deterministic path identities and aggregation.
- Compare every per-path CV CSV, work table, completion manifest, and final aggregate with an uninterrupted reference.
- Verify the work CSV retains the required group columns for Hummer–Szabo analysis: non-scaled, square-root-scaled, and linearly-scaled contributions, with units and accumulated totals defined by the existing validated convention.
- Prove CV segment and cumulative costs are correct per path and in the global aggregate.

Serial CUDA tests do not substitute for these MPI+CUDA tests. Mocked MPI or mocked CUDA tests may supplement, but cannot replace, the real lanes.

## 4. Complete ladder CV provenance

For REST2 and rREST2, every authoritative output inventory must include the CV outputs.

At minimum, include each per-state CV CSV and sidecar in:

- the completion/restart manifest;
- the outer machine-readable `-log` output inventory;
- any data-registration provenance emitted for the run.

For every file record:

- store the path relative to the run root where practical;
- store a cryptographic digest and byte size;
- store the state index and applicable tau;
- cross-reference the CV definition digest;
- distinguish CSV data from its sidecar.

Do not discover these files with an over-broad glob that can include stale or foreign files. Build the inventory from the validated state manifest.

Add tests proving that a fresh run and a continued run contain complete, identical logical inventories and that registration refuses missing, swapped, truncated, or digest-mismatched CV artifacts.

## 5. Validate AIS CV output before completion commit

AIS currently validates completed CV output when a later invocation skips an already completed path. The same full scientific validation must occur before the completion marker is atomically committed.

Before committing a path as complete, validate:

- exact header and CV column order;
- exact row count;
- exact switching-step grid;
- path ID and source-frame identity;
- tau values and schedule consistency;
- finite CV values;
- frame references and their permitted empty/non-empty semantics;
- CSV digest, sidecar digest, CV-definition digest, and committed-prefix metadata;
- cumulative CV cost consistency with the number of rows and configured scalar CVs.

Only after the complete output set passes validation may the completion marker become authoritative.

Use fault injection to corrupt or truncate the CV stream after its last write but before finalization. Prove that:

- no valid completion marker is committed;
- a later resume selects only the last committed checkpoint generation;
- it repairs/recreates the uncommitted suffix without duplicate records;
- the repaired result matches an uninterrupted reference.

Continue to run the same validation whenever an already completed path is skipped.

## 6. Validate ladder walker identity and permutations

Ladder CV completion validation must also validate walker identity.

For every CV row:

- walker must be an integer;
- `0 <= walker < number_of_states`;
- state index, tau, phase, and observation step must match the owning series and schedule;
- at each common observation step, walkers across all state files must form an exact permutation of `0 .. number_of_states - 1`.

This validation must run before completion/restart is committed and whenever a completed ladder is reused or extended.

Add negative tests for:

- negative walker;
- walker equal to `number_of_states`;
- non-integer walker;
- duplicated walker across states at one step;
- missing walker at one step;
- a syntactically valid but scientifically incorrect state-file swap.

No invalid ladder may receive an authoritative completion marker, and an extension must refuse an invalid parent before creating outputs.

## 7. Regression protection for existing scientific behavior

Retain and exercise the already implemented behavior:

- rREST2 CV values are evaluated from a deep snapshot of pre-refresh configurations;
- refreshed states never claim a post-refresh trajectory frame for a pre-refresh CV;
- unaffected states reference the exact corresponding trajectory frame;
- CV committed-prefix digests detect mutation, truncation, deletion, and swapping;
- AIS work output exposes non-scaled, square-root-scaled, and linearly-scaled energy/work contributions for later Hummer–Szabo reweighting;
- all restart operations are append-safe and deterministic;
- CUDA remains mandatory unless `--cpu` is explicitly supplied;
- MPI execution remains fail-closed.

Do not weaken tolerances or change the validated REST2/rREST2/AIS mathematics to make tests pass.

## 8. Required tests and evidence

Add regression tests before or alongside implementation changes. Tests must reach the intended runtime condition rather than fail earlier in argument parsing.

Run all of the following on the final implementation commit:

1. Complete fast/non-GPU suite.
2. Complete slow/GPU suite on real CUDA hardware.
3. Dedicated real-CUDA CV suite covering fresh and multiply resumed cMD, REST2, rREST2, and AIS.
4. Real MPI+CUDA REST2, rREST2, and AIS CV suites, including continuation and injected failure.
5. AIS pre-completion corruption/finalization tests.
6. Ladder walker/permutation corruption tests.
7. Wheel build and clean installation in a new environment outside the checkout.
8. Installed-wheel fresh and multiply resumed CUDA runs for at least cMD and AIS.
9. GitHub Actions on the exact final implementation SHA.

Record exact commands, test counts, skips, wall times, OpenMM version, CUDA platform/device names, MPI implementation/version, rank count, wheel filename/hash, and installed import origin.

Tests that require real CUDA or MPI must skip clearly when unavailable in ordinary developer environments, but the final completion evidence must come from a machine where they actually ran. A serial or CPU substitute is not acceptance evidence.

## 9. Documentation and release evidence

Audit and update:

- public configuration/schema documentation;
- CV reporting documentation;
- REST2/rREST2 and AIS restart documentation;
- AIS work-column documentation for Hummer–Szabo analysis;
- provenance/data-registration documentation;
- release notes and CUDA evidence matrix.

Correct the current evidence report where it overstates MPI+CUDA method coverage or cumulative cost persistence. Do not describe a lane as rREST2/AIS coverage unless that method actually ran in that lane.

Document the cost schema and clearly distinguish reporter observations from scalar torsion evaluations, and segment from cumulative cost.

## 10. Commit and completion policy

Implement coherently on `dev`. Preserve `CLAUDE_TASK.md` while work or evidence remains incomplete.

Suggested commit separation:

1. regression tests demonstrating the accounting, resume, provenance, finalization, and walker failures;
2. shared cost/accounting and resume implementation;
3. AIS/ladders validation and provenance;
4. real CUDA/MPI tests and documentation/evidence;
5. task-file removal only after all acceptance criteria pass.

Do not delete, skip, xfail, or weaken scientific tests merely to obtain green results. Do not claim GPU/MPI execution that did not occur.

Remove `CLAUDE_TASK.md` only in a separate final cleanup commit after:

- all required local suites pass;
- the installed wheel is verified outside the checkout;
- real CUDA and real MPI+CUDA evidence exists for REST2, rREST2, and AIS;
- GitHub Actions is green on the exact implementation commit.

After cleanup, require GitHub Actions to be green on the exact cleanup SHA as well.

The final report must provide:

- implementation and cleanup commit SHAs;
- exact-head GitHub Actions URLs;
- concise change summary;
- exact test commands, counts, skips, and wall times;
- real CUDA and MPI evidence by protocol;
- multiply resumed cost-accounting evidence;
- AIS pre-completion fault-injection evidence;
- ladder walker/permutation validation evidence;
- wheel hash and installed import origin;
- corrected evidence-document location;
- every remaining limitation.
