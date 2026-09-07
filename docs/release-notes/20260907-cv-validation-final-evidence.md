# CV validation gaps: reproduce, fix, revalidate — final evidence

**Current evidence document.** Earlier documents in this directory are historical and correct for
their own dates; where a count here differs, this one supersedes it. They are cross-linked at the
end rather than edited.

* Starting SHA inspected: `27a3715` — dev had advanced from the reviewed baseline `c9d2295` by
  the task file alone, so reproduction was done against that head rather than the baseline.
* **I (implementation + tests): `767e58a0e3f2ffea21c6bdde58cca746966fc0ac`**
* CI on I: **success** — https://github.com/csy0000/MD-tools/actions/runs/34126435102

## Requirement → test → result

| # | requirement | test | result |
|---|---|---|---|
| A | absent/null/empty/non-mapping cost on a CV-enabled prefix refused, never restored as zero | `test_cv_validation_gaps.py::test_a_*`, `test_cv_committed_prefix.py::test_a_cv_enabled_prefix_with_no_cost_is_refused` | pass |
| A | valid prefix restores its real counters; fresh zero prefix and CV-disabled records still legal | `test_a_valid_prefix_validates_and_restores_its_real_counters`, `test_a_a_fresh_run_may_initialise_a_zero_prefix_in_memory` | pass |
| A | uncommitted tail beyond the committed prefix stays continuable | `test_a_an_uncommitted_tail_does_not_invalidate_the_committed_prefix` | pass |
| B | strict decoded types for block rows and state identities | `test_b_a_block_row_count_that_is_not_an_integer_is_refused[5 cases]`, `test_b_a_state_identity_of_the_wrong_type_or_range_is_refused[5]` | pass |
| B | block rows reconciled against every per-state entry | `test_b_a_block_row_count_disagreeing_with_its_entries_is_refused` | pass |
| B | state set exactly `range(n_states)`, each once, before dictionary construction | `test_b_a_duplicated_state_identity_is_refused`, `test_b_a_missing_state_is_refused` | pass |
| B | tau checked against the verified protocol, not the record | `test_b_a_state_whose_tau_does_not_match_the_protocol_is_refused` | pass |
| B | per-state and aggregate costs validated at continuation | `test_b_a_missing_per_state_cost_is_refused`, `test_b_a_missing_or_malformed_aggregate_cost_is_refused[3]`, `test_b_an_aggregate_inconsistent_with_its_per_state_entries_is_refused` | pass |
| B | typed data returned and used for truncation *and* restoration; raw block refused | `test_b_a_valid_ladder_block_returns_typed_data_for_truncation_and_restoration` | pass |
| C | refusal preserves every file, entry, symlink, inode and mtime — no exemptions | `test_cv_cost_refusal_end_to_end.py` (18 cases) | pass |
| C | whole-operation validation: a malformed later path/state refuses before earlier output changes | `test_ais_global_aggregation_refuses_a_manifest_it_cannot_parse` (path 1), `test_ladder_continuation_refuses_a_malformed_per_state_cost` (state 0 of 2) | pass |
| C | both public `md-run` and generated wrappers at the shared boundary | `test_md_run_refuses_a_malformed_record_without_touching_the_tree` + the generated cases | pass |
| C | MPI: nonzero-rank malformed record, collective failure, unchanged tree, no hang | `test_a_malformed_record_on_a_nonzero_rank_refuses_collectively_without_touching_the_tree` | pass |
| D | resumed expectations derived independently of the output | `test_d_two_rank_interruption_resumed_under_four_ranks_on_cuda` | pass |
| D | rank agreement from per-rank evidence, not the aggregate alone | same test, `_per_rank_invocation_ids` | pass |
| D | no-op re-entry exactly zero in all three fields, cumulative unchanged | `test_ais_invocation_segment_runs.py::test_c_*` | pass |

## The independent AIS oracle

Computed **before** the resume, from the committed checkpoint generations and the configured CV
grid. No cost counter is consulted — those are the numbers under test. `cv_rows` is read only as
a cross-check.

```
R = 1 + S/d = 1 + 40/10 = 5 observations per complete path (two torsions -> 10 evaluations)
committed grid points for a partial path = 1 + floor(committed_protocol_step / d)
```

Measured on the run recorded here:

| path | starting disposition | committed grid points | expected new observations |
|---|---|---:|---:|
| 0 | already_complete | — | **0** |
| 1 | already_complete | — | **0** |
| 2 | resumed_and_completed | 4 (steps 0,10,20,30) | **1** |
| 3 | resumed_and_completed | 2 (steps 0,10) | **3** |

Global expected segment **4 observations / 8 scalar evaluations**; cumulative **20 / 40**
(4 paths × R). The resumed four-rank invocation matched exactly.

**Checkpoint cadence is not CV cadence.** The first version of this oracle asserted the committed
step lay on the CV grid; it is step 35. Checkpoints commit every parameter-update interval (5) and
observations every CV interval (10), so a committed step lands between grid points routinely. The
arithmetic is floor division, and the oracle recomputes itself per run — a later run produced
`{2: 1, 3: 3}` where an earlier one produced `{2: 1, 3: 1}`, because the fault landed elsewhere.

## Final validation matrix, all on I = `767e58a`

| # | lane | command | result |
|---:|---|---|---|
| 1 | strict parser / prefix / ladder units | `pytest tests/test_cv_cost_schema.py tests/test_cv_validation_gaps.py tests/test_cv_committed_prefix.py --error-on-skip` | **105 passed** (25.4 s) |
| 2 | serial refusal + AIS subset/resume/no-op | `pytest tests/test_cv_cost_refusal_end_to_end.py tests/test_ais_invocation_segment_runs.py tests/test_ais_invocation_accounting.py tests/test_ais_precompletion_cv.py --error-on-skip` | **47 passed** (122.0 s) |
| 3 | dedicated real-CUDA CV | `pytest tests/test_cv_cuda_lanes.py --error-on-skip` | **10 passed** (123.6 s) |
| 4 | two→four rank AIS CUDA, independent oracle | `pytest tests/test_cv_mpi_cuda_ais.py::test_d_two_rank_interruption_resumed_under_four_ranks_on_cuda --error-on-skip` | **1 passed** (35.9 s) |
| 5 | nonzero-rank malformed record, MPI+CUDA | `pytest tests/test_cv_mpi_cuda_ais.py::test_a_malformed_record_on_a_nonzero_rank_refuses_collectively_without_touching_the_tree --error-on-skip` | **1 passed** (25.1 s) |
| 6 | REST2 / rREST2 / AIS MPI+CUDA, individual | per file, `--error-on-skip` | **9 / 24 / 11** (94.6 / 204.3 / 150.5 s) |
| 6 | the same three files, one invocation | `pytest tests/test_cv_mpi_cuda_lanes.py tests/test_cv_mpi_cuda_rrest2.py tests/test_cv_mpi_cuda_ais.py --error-on-skip` | **44 passed** (488.9 s) |
| 7 | complete fast suite | `pytest tests -m "not slow and not gpu"` | **1387 passed**, 428 deselected (191.2 s) |
| 8 | complete complementary suite | `pytest tests -m "slow or gpu"` | **427 passed, 1 skipped**, 1387 deselected (2521.8 s) |
| 9 | wheel build, clean install, installed checks | below | pass |
| 10 | GitHub Actions on I | hosted, non-CUDA | **success** |

9 + 24 + 11 = 44, which is the combined run. 1387 + 427 + 1 = 1815 collected across both complete
suites, matching collection.

### Skip inventory

One skip in the entire matrix, and none in any required lane (all ran `--error-on-skip`):

* `tests/test_cuda_coverage_matrix.py::test_write_the_coverage_evidence` — *"no
  `--cuda-evidence=<path>` given; the lanes above ran, nothing to write."* Pre-existing and
  optional: it writes the committed matrix file only when explicitly asked, so an ordinary GPU
  run does not rewrite it. It covers no scientific invariant; the lanes it would document all ran.

### Lane 9 detail

* Wheel `md_tools-0.5.0.dev0-py3-none-any.whl`, sha256
  `e4522cc6d5d06b0712c7f418c65cf4dd929b40ca9953beb77f4bd4846fa14aec`
* Installed into a fresh virtualenv outside the checkout. Checkout import paths cleared and
  verified: **50 `md_tools` modules loaded, none resolving to `MD-tools/src`**. Import origin
  `…/env3/lib/python3.12/site-packages/md_tools/__init__.py`.
* Generated entry point, real CUDA: subset (`--paths 0`) → resume → no-op re-entry, all rc=0,
  recorded platform CUDA. **No-op segment exactly `{0, 0, 0.0}`**; cumulative 15 observations /
  30 evaluations for 3 paths (3 × R = 15, two torsions), all dispositions `already_complete`.
* Malformed record (`cumulative.cv_observations = "5"`) on path 1: generated wrapper rc=2 and
  public `md-openmm md-run` rc=2, each naming the record, scope and field. **Zero changed entries
  across all 32 tree entries** in both cases, compared by digest, inode and mtime_ns.

## Mutation checks

Run in an isolated disposable clone, removed afterwards. Never in a simulation tree. Each records
the test and the assertion, not an import error.

| mutation | failures | representative assertion |
|---|---:|---|
| (a) missing-cost-as-zero | 5 | `test_a_restoring_an_absent_cost_as_zero_is_refused` → DID NOT RAISE `CVCostError`; `test_cv_cost_is_persisted_and_survives_two_interruptions` → `assert 2 == 0` |
| (b) permissive ladder `int()` / duplicate handling | 17 | `test_b_a_duplicated_state_identity_is_refused` → DID NOT RAISE `CVContinuationError` |
| (c) preflight log/status mutation | 9 | `test_cmd_continuation_refuses_a_malformed_cost_and_writes_nothing` → *modified the protected output tree before refusing* |
| (d) historical-segment aggregation | 9 | `test_summing_stored_segments_is_what_this_test_refuses` → `assert 13 != 13`; `test_c_re_entering_a_completed_campaign_reports_exactly_zero_segment` → `assert 15 == 0` |

## Failures during this work, and what each was

| # | failure | classification | correction |
|---:|---|---|---|
| 1 | 18 reproduction cases, all DID NOT RAISE | implementation (defects A, B) | required cost; strict typed ladder validation returning typed data |
| 2 | 10 regex mismatches in `test_cv_committed_prefix.py` | **test** — the `_entry` fixture built cost-less records, so each case refused for the new reason rather than its own | fixture supplies an honest cost; the new rule asserted directly |
| 3 | 9 refusal cases modifying `.out`/`.log`/runstate | implementation (defect C) | read-only continuation phase before any output |
| 4 | `md-run` wrote `resolved.config` + definition copy before refusing | implementation | `validate_public_entry` before the directory is created |
| 5 | `md-run` refused for `cMD.cv.cv.csv`, a file that never existed | implementation, mine — stage NAME passed where a mapping was expected, then the series where a trajectory was expected | validate the series directly |
| 6 | `--overwrite` refused | implementation, mine | `--overwrite` starts clean and is not a continuation; check skipped |
| 7 | stage check paired `min`'s checkpoint with the production series | implementation, mine | each checkpoint validates its own `<stem>.cv.csv` |
| 8 | `test_explicit_device_placement_lane`, `test_multi_rank_rest2_ladders_of_several_sizes[6]` | **infrastructure** — my own `CUDA_VISIBLE_DEVICES=0,5,6,7,8` starved a 6-device ladder and a placement assertion | cleared only my own pinning; 4 passed unpinned |
| 9 | one `test_cv_mpi_cuda_ais.py` failure during a batch run | **undiagnosed** — see limitations | not reproduced in 3 subsequent full-file runs |

## Environment

* 9 CUDA devices: 1 × NVIDIA RTX A5000, 8 × NVIDIA GeForce RTX 3080, driver 580.173.02
* Open MPI 5.0.8 (`prterun`), mpi4py 4.1.2. Rank counts: 2 and 3 (ladders — the ladder width *is*
  the rank count), 2 and 4 (AIS).
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db, Linux 6.8.0-124-generic, x86-64
* Lanes 1, 2 and 7 are CPU by design — validation arithmetic and serial lifecycle, per the
  instruction's limited exemption. They are not CUDA evidence and are not counted as such. Lanes
  3–6, 8 and 9 ran on real devices; lanes 4–6 and the lane-9 refusals under real `mpirun`.

## Limitations

* **One undiagnosed failure.** During a batch where several GPU lanes ran back to back and a
  concurrent combined run was killed by a shell timeout, `tests/test_cv_mpi_cuda_ais.py` reported
  `1 failed, 10 passed` once. The diagnostic was lost to an output filter before it was recorded,
  which the instruction requires preserving, so it cannot be classified. Three subsequent
  full-file runs and the combined lane all pass. It is recorded here rather than treated as
  resolved.
* **GitHub CI has no CUDA runner.** Lane 10 proves packaging and interface only; every GPU and
  MPI claim rests on the local hardware named above.
* **The CPU platform is reproducible only at a fixed thread count.** Tests comparing
  trajectory-dependent values pin `OPENMM_CPU_THREADS=1`; production pins nothing.
* **Schema-version-1 records are refused, not migrated.** `migrate_schema_1` converts only when
  given the verified row count and CV definition, and stamps the result as reconstructed.

## Earlier evidence, cross-linked as historical

* [`20260907-ais-invocation-and-strict-cost-evidence.md`](20260907-ais-invocation-and-strict-cost-evidence.md)
* [`20260904-cv-accounting-closure-evidence.md`](20260904-cv-accounting-closure-evidence.md)
* [`20260904-cv-scientific-closure-evidence.md`](20260904-cv-scientific-closure-evidence.md)
* [`20260904-cv-lifecycle-closure-evidence.md`](20260904-cv-lifecycle-closure-evidence.md),
  [`20260904-cv-and-runtime-closure-evidence.md`](20260904-cv-and-runtime-closure-evidence.md)
* [`cuda-coverage-matrix.md`](cuda-coverage-matrix.md)
