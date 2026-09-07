# Final AIS invocation accounting and strict CV-cost validation

## Baseline and authority

Work on `csy0000/MD-tools`, branch `dev`.

The exact implementation baseline is:

```text
b7ad4b7d989faa98796fb7bb961b62072ce52d4a
```

This is a narrow corrective task. Preserve the validated cMD, REST2, rREST2, AIS, CV, restart, CUDA/MPI, reservoir-refresh, and three-group AIS work mathematics. Do not redesign the protocols.

Implement both runtime corrections, their concrete tests, and the evidence correction below. A test failure means diagnose the cause, fix the implementation, and rerun the same test. Do not weaken an assertion, tolerance, fault boundary, or scientific invariant to make a lane green.

## 1. Correct AIS global segment accounting

### Defect

`write_work_table()` currently builds the global segment by summing the `segment` stored in every completed path manifest. A path manifest's segment describes the invocation that completed that path. It does not necessarily describe the AIS invocation currently assembling the global output.

This makes the current global segment wrong when:

- a path completed in an earlier invocation and is merely verified/skipped now;
- a campaign resumes after some paths already completed;
- paths complete across multiple interrupted invocations;
- a completed campaign is entered again;
- the MPI worker count changes between interruption and resume.

The cumulative sum remains a sum over completed paths. The segment must instead mean only CV work performed by the current invocation.

### Required semantics

Give every AIS invocation a stable, recorded invocation identifier. The identifier must be common across all ranks in that launch and must not become part of scientific path identity, seeding, or reproducibility.

For each selected path, distinguish:

- committed cumulative cost read from its verified completion/checkpoint state;
- CV cost performed by the current invocation;
- whether the path was already complete at invocation start;
- whether it resumed from a committed prefix and completed now;
- whether it ran fresh and completed now.

A path that was complete at invocation start contributes exactly zero observations, zero scalar evaluations, and zero wall time to the current global segment. Verification, hashing, CSV reading, aggregation, and completed-path skipping are not CV evaluation work.

The global record must retain:

```yaml
collective_variable_cost:
  schema_version: 2
  aggregation: sum over completed paths
  invocation_id: ...
  segment:
    cv_observations: <sum performed in this invocation only>
    cv_evaluations: <scalar CVs performed in this invocation only>
    wall_seconds: <evaluation time in this invocation only>
  cumulative:
    cv_observations: <sum of verified completed-path cumulative observations>
    cv_evaluations: <sum of verified completed-path cumulative scalar evaluations>
    wall_seconds: <sum of verified completed-path cumulative evaluation time>
  per_path:
    - path_index: ...
      segment: <this invocation's contribution for this path>
      cumulative: <verified path cumulative cost>
      disposition: already_complete | resumed_and_completed | fresh_and_completed
```

The authoritative per-path `completed.json` may retain the segment of the invocation that completed that path, because that manifest describes its completion. The current run's aggregate must not reuse that historical segment as though it occurred now.

Collect current-invocation contributions from all MPI ranks through the shared fail-closed coordination authority. Do not infer them from rank-local filenames or an over-broad glob. Rank 0 may write the aggregate only after it has exactly one contribution/disposition for every selected path expected in a complete campaign.

Do not include `invocation_id` in the AIS scientific fingerprint, path seeds, selected source frames, work values, CV values, or output filenames.

## 2. Strictly validate stored CV-cost records

### Defect

Current readers coerce persisted values before validation, for example with `int(value)`. This silently accepts values that violate the schema:

- `2.7` becomes `2`;
- `"5"` becomes `5`;
- `true` becomes `1`.

AIS completion validation also treats a missing/empty cumulative block as optional and does not fully validate the segment scope.

### One strict shared parser

Create one strict parser/validator for schema-version-2 CV cost records and use it everywhere authoritative stored costs are read:

- cMD checkpoint prefixes and completion/log records;
- REST2/rREST2 per-state prefix costs, aggregate costs, completion and extension validation;
- AIS checkpoint prefixes, per-path completion, global aggregation, completed-path verification;
- data-registration/provenance validation where these records are consumed.

For schema version 2:

- `schema_version` is required and exactly integer `2`;
- `segment` and `cumulative` are required mappings;
- each scope requires exactly the documented counters, allowing only explicitly documented metadata;
- `cv_observations` and `cv_evaluations` must satisfy `type(value) is int`; booleans are invalid;
- neither numeric strings nor floats, including `5.0`, are accepted as integer counters;
- integer counters are non-negative;
- `wall_seconds` must be a real numeric value, not boolean, finite, and non-negative;
- every segment counter is less than or equal to its cumulative counterpart;
- `cv_rows`, when required by that record, is a strict non-negative integer;
- cumulative observations equal the verified row count;
- cumulative scalar evaluations equal `verified_rows * number_of_configured_CVs`;
- aggregation totals equal the exact sum of their verified per-state or per-path entries;
- duplicate/missing per-state or per-path identities are rejected.

Do not call `int()` or `float()` on unvalidated external values. Validate the original decoded type first.

Define an explicit policy for schema-version-1 records. Either:

1. migrate them only through a dedicated function supplied with the verified row count and CV definition, recording that migration; or
2. refuse CV-enabled continuation with a clear actionable message.

Do not silently present an unknowable legacy scalar count as a complete schema-version-2 cumulative total.

Malformed authoritative records must fail before output mutation. The error must identify the record, scope, and field.

## 3. Concrete tests: AIS invocation accounting

Use a two-torsion `cv.yaml`, so `cv_evaluations = 2 * cv_observations`. Derive expected counts from the fixed schedule before running; do not calculate expectations using the implementation under test.

### Test A — synthetic aggregation unit test

Construct three verified path costs with distinct cumulative and historical segment values and supply explicit current-invocation contributions.

Pass only if:

- global cumulative equals the sum of all cumulative path costs;
- global segment equals the explicit current contributions, not the stored historical segments;
- each path disposition and current segment is retained;
- input ordering does not change output ordering or totals;
- a missing, duplicated, or unexpected path contribution is rejected.

Restore the old implementation that sums stored segments and prove this test fails.

### Test B — paths completed across invocations

Run an AIS campaign with at least three paths.

1. Invocation 1 completes a strict subset of paths.
2. Invocation 2 resumes and completes the remaining paths.
3. Assemble the full output.

Pass only if:

- paths already complete at invocation 2 contribute zero to its segment;
- newly evaluated observations in invocation 2 give the exact segment count;
- global cumulative equals all paths and is larger than the invocation-2 segment;
- every path's CV/work output matches an uninterrupted reference;
- no path is evaluated twice;
- the per-path dispositions are correct.

### Test C — completed no-op re-entry

Re-enter a completed campaign with `--resume`.

Pass only if:

- segment observations = 0;
- segment scalar evaluations = 0;
- segment wall time = 0;
- cumulative counters remain byte-for-byte/numerically unchanged;
- CV, work, HS, trajectory, and completion artifacts are not rewritten;
- validation/hashing does not increment CV cost.

### Test D — interruption and changed MPI world size on CUDA

On real CUDA, launch at least four paths under two MPI ranks, interrupt after at least one path is fully complete and at least one path has a committed partial prefix, then resume under four MPI ranks.

Pass only if:

- every path ID retains its source frame, seeds, filenames, CV values, and work values;
- all path and aggregate scientific tables match an uninterrupted reference;
- only explicitly non-scientific execution metadata such as `mpi_rank` and timing may differ;
- the resumed invocation's global segment counts only CVs evaluated during that four-rank invocation;
- global cumulative counts every completed path exactly once;
- all ranks agree on one invocation ID;
- no rank hangs or survives a rank-local failure;
- recorded platform is CUDA and `--cpu` is absent.

Use deterministic fault injection and a subprocess timeout.

## 4. Concrete tests: strict cost schema

Create a parameterized unit matrix against the shared strict parser. Starting from a valid two-torsion record, mutate each case independently:

- missing `schema_version`;
- schema version as `"2"`, `2.0`, `true`, or unsupported integer;
- missing `segment` or `cumulative`;
- missing each required counter;
- counter as `true`, `"5"`, `5.0`, `5.7`, negative integer;
- wall time as `true`, string, NaN, positive infinity, negative infinity, or negative number;
- segment observation/evaluation/time greater than cumulative;
- `cv_rows` as boolean, string, float, negative, or inconsistent integer;
- cumulative observations inconsistent with rows;
- cumulative scalar evaluations inconsistent with `rows * N_cv`;
- duplicate/missing path or state identities;
- aggregate totals inconsistent with per-path/per-state records.

Every mutation must be rejected with the offending field named. The untouched record must pass.

Add end-to-end refusal tests through:

- cMD continuation;
- REST2/rREST2 continuation or extension;
- AIS partial-path continuation;
- AIS completed-path skip;
- AIS global aggregation.

For every refusal, snapshot the output tree before the command and prove it remains byte-identical afterward, apart from an explicitly separate ephemeral stderr capture outside the run directory.

## 5. Preserve scientific outputs

Regression tests must prove the correction does not change:

- cMD, REST2, rREST2, or AIS coordinates under matched deterministic continuation;
- REST2/rREST2 exchange decisions or reservoir draws;
- AIS source-frame selection, seeds, tau schedule, total work, or reduced work;
- AIS non-scaled, square-root-scaled, and linearly-scaled work/potential columns;
- Hummer–Szabo frame alignment;
- CV values, cadence, column order, or frame-reference semantics;
- CUDA-by-default and MPI fail-closed behavior.

Do not add the invocation identifier to any scientific equality comparison or random seed derivation.

## 6. Correct evidence

Rerun and correct the individual MPI+CUDA lane counts. The current evidence says the combined lane has 42 tests while the listed REST2/rREST2/AIS rows total 29. Report the current individual results so their sum equals the combined result, or explain any additional collected tests precisely.

Distinguish:

- results run on the exact implementation SHA;
- results run only on an earlier SHA;
- GitHub-hosted non-CUDA CI;
- local real-CUDA/MPI evidence.

Do not copy old counts forward.

Create or update one authoritative final evidence document; cross-link older evidence instead of allowing several documents to make conflicting current-head claims.

## 7. Required execution and retry policy

Run in this order:

1. strict cost-parser unit matrix;
2. synthetic AIS aggregation test;
3. serial AIS subset/resume and no-op re-entry tests;
4. end-to-end cMD, ladder, and AIS malformed-record refusal tests;
5. real CUDA AIS continuation test;
6. real MPI+CUDA changed-world-size AIS test;
7. all dedicated CUDA CV tests;
8. all REST2/rREST2/AIS MPI+CUDA tests;
9. complete fast/non-slow suite;
10. complete slow/GPU suite;
11. wheel build and installation into a fresh environment outside the checkout;
12. installed-wheel CUDA AIS subset/resume and no-op re-entry;
13. GitHub Actions on the exact implementation SHA.

For every failed lane:

- preserve the failing command and diagnostic;
- decide whether it is an implementation defect, test defect, infrastructure failure, or resource contention;
- fix implementation defects in production code;
- fix a test only when its stated expectation is demonstrably inconsistent with this instruction;
- for transient GPU/MPI contention, verify and clear only task-owned stale processes/resources, then rerun;
- rerun the focused failing test until it passes;
- rerun all earlier focused lanes affected by the change;
- after all focused lanes pass, rerun both complete suites.

Do not leave a known failure and call it a limitation. Stop and report a genuine external blocker if CUDA/MPI infrastructure cannot run; do not substitute CPU or serial evidence.

## 8. Passing criteria

This task passes only when all statements below are true:

- old stored path segments cannot contribute to the current AIS global segment;
- a no-op completed AIS resume reports an exactly zero segment;
- cumulative AIS cost remains the exact sum over verified completed paths;
- a two-to-four-rank CUDA resume has exact scientific equality to its uninterrupted reference;
- every schema mutation listed above is rejected before filesystem mutation;
- valid schema-version-2 records pass through all protocols;
- schema-version-1 behavior is explicit, tested, and documented;
- the three-group AIS decomposition is unchanged and reconstruction tests pass;
- focused, complete, CUDA, MPI+CUDA, and installed-wheel lanes all pass;
- exact implementation-head CI is green;
- evidence counts are internally consistent and tied to SHAs;
- no scientific test is skipped, xfailed, deleted, weakened, or converted to CPU/serial to obtain success.

Keep `CLAUDE_TASK.md` until every criterion passes. Remove it only in a separate cleanup commit after the implementation SHA has green CI and complete local CUDA/MPI evidence. Then require green CI on the cleanup SHA.

The final report must give implementation and cleanup SHAs, exact commands and results, retry history for any failure, CUDA devices, MPI version/ranks, wheel hash/import origin, the final evidence path, and every remaining limitation.
