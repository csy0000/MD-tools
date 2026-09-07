# Close CV validation gaps: reproduce, fix, and revalidate

## Authority and scope

Repository: csy0000/MD-tools. Branch: dev.
Reviewed baseline: c9d2295d50d29b3643b08c4dde0d0d98801b6c24.

Implement this task, run its tests, diagnose failures, fix them, and repeat until the gates below pass. Read CLAUDE.md and the current implementation. This task is the current acceptance specification for these corrections. It supersedes the diagnostic-file exemptions introduced in tests/test_cv_cost_refusal_end_to_end.py and resolves how normal runtime logging interacts with a rejected continuation.

Preserve the existing scientific protocols, public commands, CV conventions, AIS invocation accounting, and three-group work mathematics. No redesign of scaling, exchange, reservoir draws, schedules, or seeding. Use the shared runtime and validation authorities; generated entry points must remain thin.

Inspect the actual checkout SHA and working tree first. Preserve unrelated work. If dev has advanced, assess the intervening changes and reproduce against that head rather than resetting to the reviewed baseline. Record the actual starting SHA.

## 1. Remaining defects to reproduce

### A. Missing costs bypass validation

In cv/prefix.py, validate() only calls the strict parser when entry.get("cost") is not None. In cv/cost.py, CommittedPrefix.from_record()/read_scope() can restore absent costs as zero. In remd/cv_states.py, _cost_problems() returns no problems when the cost block is missing.

A two-row, two-torsion prefix with a valid cost restores 2 observations and 4 evaluations. Removing its cost or replacing it with null currently passes prefix validation and restores 0/0. An absent cost in a CV-enabled persisted record is corruption or unsupported legacy data, not a fresh run.

### B. Ladder prefix structure is not fully validated

remd/cv_states.py::validate_prefixes() converts block["rows"] with int(), does not reconcile that count with every per-state entry, and does not enforce unique, complete expected state coverage. committed_prefixes() also coerces fields and collapses identities into a dictionary.

With two valid committed rows per state, both block rows=1.9 and block rows=1 currently return 1. A state list containing state 0 twice and omitting state 1 also passes. driver.py::_continue_cv_states() uses the returned count to truncate files.

The checkpoint's aggregate cost must also be validated against the same verified state entries; validating only completion manifests leaves continuation unprotected.

### C. Refusal tests exclude authoritative files

The current suffix-based exemptions allow .out, .log, and *runstate.json to change. They do not prove that a rejected continuation preserved the prior run. Machine-readable provenance and completion status are part of that prior run.

### D. Changed-rank AIS expectations are circular

test_d_two_rank_interruption_resumed_under_four_ranks_on_cuda derives expected_segment by summing the resumed output's per-path segment counts. Internal consistency does not establish how many observations the resumed invocation should have evaluated.

## 2. One validation contract

### Required versus absent costs

For every persisted CV-enabled prefix, completion record, and aggregate consumed by continuation, completed-run verification, extension, or registration/provenance validation:

- Require a schema-2 cost mapping with both segment and cumulative scopes.
- Refuse missing, null, empty, non-mapping, legacy, and malformed costs before output mutation.
- Keep the existing explicit refusal policy for unsupported legacy CV records; do not invent or silently migrate history.
- Require original decoded integer types for counters and row/state/path identities: type(value) is int, excluding booleans, strings and floats.
- Require finite, non-negative numeric durations; segment must not exceed cumulative.
- Verify observations against committed rows and scalar evaluations against rows times configured CV count.
- Require cv_rows wherever that record's writer/schema requires it, and verify it when present.
- Check aggregate counters against the verified per-entry records, including expected identities. Retain the documented microsecond rounding tolerance for wall-time sums only.

Fresh-run initialization may create a zero CommittedPrefix internally. A truly CV-disabled record may have its documented null/absent CV field. These cases must be distinguished using authoritative run configuration/schema, not the truthiness of a stored cost. Do not make the shared parser reject legitimate CV-disabled runs, or accept a damaged CV-enabled run as disabled.

Use one shared parser. Audit every authoritative reader and enumerate its call site and test in the evidence. Do not leave a permissive restoration helper that bypasses the parser. Registration should call the same authoritative output verifier where applicable, not introduce a competing parser.

### Ladder checkpoint consistency

Validate the entire ladder checkpoint CV block before any truncation or reporter opens:

1. Block rows, each entry's rows, state identities, and any duplicate persisted CV row counters have strict decoded types.
2. The state set is exactly range(n_states), with each state appearing once. Reject duplicate, missing, negative, unexpected, boolean, string and fractional identities before dictionary construction.
3. Each state index maps to the configured tau and correct file/sidecar, definition, columns, cadence, phase and digest. Derive expected states from the current verified protocol, not the record being checked.
4. All representations of committed row count agree. Use checkpoint progress and the configured CV grid as an additional independent check where their semantics define the count. Do not assume CV cadence equals trajectory or checkpoint cadence.
5. Per-state costs and aggregate costs agree with those verified facts. A consistently reduced count is not sufficient if it disagrees with checkpoint progress.
6. Return validated typed data and use it for both truncation and restoration. Do not reread/coerce unchecked fields afterward.

Preserve crash recovery: files may contain an uncommitted tail beyond the committed prefix. Validate the committed prefix, then truncate tails only after acceptance. Never require the entire current file length/digest to equal the committed prefix.

## 3. Read-only rejection, with usable diagnostics

Use two phases for an invocation continuing/verifying existing data:

- Preflight reads and validates all authoritative records needed for the requested operation. It may initialize resources in memory but must not create, rewrite, truncate or replace files in the protected output tree.
- Only after preflight succeeds may ordinary runtime setup, logs/status records, resolved.config, reporter opening, truncation, propagation and publication occur.

For AIS, validate all selected existing completed/partial paths before beginning work on any selected path. For ladders, validate all states and any extension parent before mutating either parent or destination. Otherwise an invalid later path/state can be discovered after an earlier one has already changed.

Under MPI, use the existing coordination authority to agree that all ranks passed this read-only phase before any rank writes. A failed rank must produce collective termination within the subprocess timeout, with no surviving task-owned child processes. Do not add a second MPI implementation.

A preflight refusal exits nonzero and writes an actionable error to stderr, captured by the test outside the protected tree. Do not overwrite the existing .out, .log or runstate to describe an invocation that never started. This is compatible with completion being read from machine records: the rejected attempt does not replace the prior run's authoritative state. Once preflight succeeds and execution begins, normal runtime failure logging remains required.

Apply the boundary to public md-run and generated entry points. If existing guidance says resolved.config/logs are written on every invocation, qualify it to every accepted invocation. Update relevant documentation with this precise rule.

For every refusal test, snapshot all files, directory entries and symlink targets under the protected tree; compare bytes/digests and path sets afterward. Include .out, .log, runstate, resolved.config, checkpoints/pointers, completion records, sidecars, trajectories and aggregates. No suffix exemptions, ignore patterns, delete-after-write tricks, restoring snapshots after refusal, or --overwrite. Ignore only filesystem access timestamps; do not turn read-induced atime changes into a failure. Where the contract forbids rewriting identical bytes, additionally compare inode/mtime_ns for those artifacts.

## 4. Independent AIS accounting oracle

Use at least four paths and two torsions. Retain real two-rank CUDA interruption followed by a four-rank CUDA resume, deterministic fault injection, and timeouts. Ensure at least one path is fully complete and another has a valid committed partial prefix.

Before resume:

- Save the source-frame selection and immutable identities.
- Read the selected committed checkpoint generations, using their progress and the explicitly configured observation grid to determine the committed CV observations per partial path.
- Count/verify the committed CSV prefix as a cross-check. Do not use any cost counters as the oracle.
- Record each path's starting disposition and expected newly evaluated CV grid points.

For a fresh, non-extended AIS path with S switching steps, CV interval d dividing S, and step 0 included, the complete count is R=1+S/d. A completed path contributes 0 new observations. A fresh path contributes R. A partial path contributes R minus the CV grid points already committed; do not count an uncommitted tail as retained work. For other supported scheduling cases derive the corresponding grid explicitly in test code.

Assert each resumed per-path segment and the global segment against these precomputed expectations; evaluations equal twice observations. Cumulative equals R per completed path and R times path count globally. Assert zero wall time for skipped paths; measured wall time for evaluated paths must be finite and non-negative, without a hardware-specific exact target.

Check common invocation ID using independent per-rank launch evidence, or an instrumented existing coordination boundary. A single ID in the aggregate alone does not prove rank agreement. Instrumentation must not affect scientific seeds, fingerprints, coordinates or filenames.

Compare source frames, seeds, tau schedule, per-path/aggregate CV and work tables, all three work/potential components, HS frame alignment, and trajectory coordinates with the uninterrupted reference using the established deterministic settings. Allow only named execution metadata differences (rank, invocation ID, timing, resume provenance). Do not drop entire output/scientific blocks to avoid comparing them.

Keep the serial subset/resume test and completed no-op re-entry test. No-op segment is exactly zero in all three fields; cumulative is unchanged; scientific/completion artifacts are not rewritten. An accepted no-op may record its new invocation in the normal invocation logs.

## 5. Tests that expose the defects

Add focused cases before fixing the implementation and save the baseline failures.

- Prefix/cost readers: remove the entire cost, set null/empty/non-mapping, remove each required scope/field, and exercise strict-type and bounds mutations. Test valid zero-cost fresh initialization and CV-disabled records as positive controls.
- Ladder prefixes: malformed aggregate rows, a valid integer disagreeing with per-state rows, state/aggregate/checkpoint-progress disagreement, duplicate/missing/unexpected identities, swapped identity/tau, missing per-state cost, missing aggregate cost, and aggregate sums inconsistent with verified entries. Include a valid interrupted prefix with an uncommitted tail.
- End-to-end refusal: cMD automatic continuation and completed verification; REST2 and rREST2 continuation and extension-parent verification; AIS partial resume, completed skip and aggregation; registration wherever it consumes these records.
- Put a malformed record in a later state/path as well as the first, proving whole-operation validation precedes any earlier output change.
- Exercise both public md-run and generated wrappers at the shared boundary. Add a real MPI+CUDA malformed-record case on a nonzero rank, asserting unchanged shared output, collective failure and no hang.
- Strengthen Test D with the independent oracle above.

Mutation checks in an isolated disposable checkout must show that the relevant tests fail when restoring: (a) missing-cost-as-zero, (b) permissive ladder int()/duplicate handling, (c) preflight log/status mutation, and (d) historical-segment aggregation or overstated partial-path contribution. Record the test and assertion that failed, not merely an import error or incompatible function signature. Restore clean code afterward. Never run mutation experiments in a user's simulation tree.

## 6. Execute, diagnose, fix, repeat

Start with the focused reproduction matrix, then implement the corrections and rerun the same tests. For each failure record command, SHA/tree state, error, classification, root cause, correction and rerun result.

- An implementation defect requires a production fix.
- Change a test only when its expectation is demonstrably inconsistent with the contract above. Do not weaken scientific tolerances, compare fewer scientific fields, alter the fault boundary to avoid partial resumes, or exclude files to obtain a pass.
- Infrastructure/resource failures require a concrete diagnosis and retry after resolving only task-owned resources. Do not kill unrelated jobs or substitute CPU/serial for CUDA/MPI evidence.
- After any code/test change, rerun affected focused lanes. After the final correction, run the complete final matrix below against one fixed implementation commit.
- If a final lane reveals another defect, fix it, create a new implementation commit, and restart the final matrix on that commit. Prior failures remain in the evidence.
- If required hardware/access is genuinely unavailable, report the exact blocker and leave the task incomplete. Do not label unexecuted lanes as passing.

Pure validation/arithmetic and serial lifecycle tests may explicitly use CPU. This exemption is limited to those tests and is not CUDA evidence.

## 7. Final validation matrix

Record concrete executable commands for every lane; no "generated tree tested" placeholder.

1. Strict parser/prefix/ladder structural unit tests and mutation checks.
2. Serial cMD/ladder/AIS refusal and valid continuation controls, plus AIS subset/resume/no-op tests.
3. Dedicated real-CUDA CV tests.
4. Two-to-four-rank AIS CUDA test with the independent oracle.
5. Nonzero-rank malformed-record refusal on real MPI+CUDA.
6. All REST2/rREST2/AIS MPI+CUDA files; report individual and combined collected/passed counts from fresh runs.
7. Complete fast suite: python -m pytest tests -m "not slow and not gpu".
8. Complete complementary suite: python -m pytest tests -m "slow or gpu".
9. Build wheel and install into a fresh environment outside the checkout. Clear checkout import paths; record wheel SHA256 and actual imported module origin. Run installed-wheel public-command and generated-entry-point checks covering valid resume, malformed-record refusal, and CUDA AIS subset/resume/no-op. Verify scientific outputs and accounting, not just exit codes.
10. GitHub Actions on the exact implementation commit.

Retain regression coverage for cMD coordinates, ladder exchanges/reservoir draws, AIS three-group decomposition/reconstruction, HS alignment, CV cadence/conventions, and CUDA-default/MPI fail-closed behavior. The complete suite includes CPU-exempt tests; report their actual platforms rather than calling every slow test a GPU test.

Required task/scientific lanes must have zero failures, skips and xfails. Use --error-on-skip for dedicated required lanes. For complete suites, report every pre-existing optional skip by node ID and reason; no skip may cover a required scientific invariant, and no new skip/xfail may obtain success. Reconcile pass/skip/deselection totals with actual collection.

## 8. Evidence and completion without a moving-SHA loop

Create one current evidence document at docs/release-notes/20260907-cv-validation-final-evidence.md. Cross-link earlier evidence as historical. Include a requirement-to-test/result table, all commands, failing and passing results, mutation checks, environment/platform/MPI details, independent expected counts, wheel hash/import origin, skip inventory and remaining blockers.

Use these commit roles explicitly:

- I: final implementation AND tests committed, before the final validation matrix.
- E: evidence/documentation-only commit recording results measured on I.
- C: optional separate cleanup commit removing CLAUDE_TASK.md, only after I's required local lanes and CI pass and E's CI is green.

Results on I remain valid for unchanged source/tests/package configuration in E/C; prove that with a diff. Do not claim GPU runs occurred on E/C unless they did. If executable content, tests, dependencies or packaging changes, establish a new I and revalidate. Require green CI on the final dev head, including C if created. This avoids embedding a commit's own SHA in its contents or rerunning GPUs solely to commit documentation.

Keep this task file until every required gate passes. If any gate is blocked, leave it in place with the blocker accurately reported. The final report must provide I/E/C as applicable, final head, CI links, measured test totals, evidence link, independent AIS counts, retry history and any remaining limitations. Do not declare closure while a known defect or required unexecuted lane remains.
