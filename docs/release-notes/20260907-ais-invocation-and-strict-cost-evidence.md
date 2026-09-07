# AIS invocation accounting and strict CV-cost validation — evidence

**This is the authoritative current-head evidence document.** The earlier documents in this
directory describe the work of their own dates and are correct as history; where any of them
states a count for a lane that has since changed, the number here supersedes it. They are
cross-linked at the end rather than edited, so a reader can see what was true when.

Baseline `b7ad4b7`; instruction `2328f7a`.

## The two corrections

### 1. The AIS global segment described the wrong invocation

`write_work_table()` built the global `segment` by summing the `segment` stored in every
completed path's manifest. That stored segment describes the invocation which **completed that
path**. It is not the invocation assembling the table whenever a campaign resumes after some
paths finished, completes across several interruptions, is re-entered, or changes worker count
between interruption and resume — which is to say, in exactly the cases resume exists for.

The error is in the flattering direction: it reports more work than was done, and re-entering a
finished campaign reports the whole campaign again while nothing is computed.

The segment now comes from **explicit per-path contributions**. `run_one_path` records what this
invocation did for each path as an out-parameter — deliberately not part of `completed.json`,
because two runs of the same finished path must produce the same manifest and different
contributions. Contributions are gathered across ranks through the shared coordination authority,
which refuses a duplicate outright: path ownership is exclusive, and two ranks claiming one path
is a world divided twice. A path already complete at invocation start contributes exactly zero
observations, zero scalar evaluations and zero wall time; verification, hashing, CSV reading and
skipping are not CV evaluation. Cumulative remains the sum over verified completed paths.

One `invocation_id` is drawn on rank 0 and broadcast, so every rank labels the launch the same
way. It appears in no scientific fingerprint, seed, source-frame selection, filename, CV value or
work value, and a test asserts that two launches of one completed campaign differ in it and in
nothing else.

Each `per_path` entry keeps its `disposition`: `already_complete`, `resumed_and_completed`, or
`fresh_and_completed`.

### 2. Stored cost records were coerced before they were validated

The readers called `int()` and `float()` on whatever they found. `int(2.7)` is 2, `int("5")` is 5
and `int(True)` is 1 — three records that break the schema became three that satisfy it, and the
violation was destroyed before anything could refuse it. AIS completion validation additionally
treated a missing cumulative block as optional.

There is now **one strict parser**, used wherever an authoritative stored cost is read: the
checkpoint-prefix validator that cMD, the ladder and AIS all pass through; the ladder's
completion and extension verification; AIS per-path completion, completed-path verification and
global aggregation. It tests the decoded type and never converts:

* `schema_version` present and exactly integer `2`;
* `segment` and `cumulative` required mappings holding exactly the documented counters;
* counters `type(v) is int`, booleans excluded by name, non-negative;
* `wall_seconds` real, not boolean, finite, non-negative;
* every segment counter ≤ its cumulative counterpart;
* `cv_rows` a strict non-negative integer, checked against the **verified** row count;
* cumulative observations = verified rows; cumulative evaluations = rows × N_cv;
* aggregate totals equal the exact sum of their per-state/per-path entries, with duplicate,
  missing and unexpected identities rejected.

**Schema version 1 is refused**, with an actionable message: its single counter counted reporter
calls, so a scalar total cannot be recovered from it and is not invented. `migrate_schema_1`
converts only where the verified row count and CV definition are supplied, and stamps the result
`aggregation: migrated from schema 1` so a reconstructed total is distinguishable from a measured
one.

### A defect found by the refusal lane

AIS validated a committed CV prefix roughly two hundred lines **after** truncating
`observations.csv`, the state table and the staged trajectory to their committed counts — beneath
a comment saying that a continuation which has already truncated cannot decide afterwards that it
should have refused. A malformed record was therefore refused only once the resume had destroyed
part of the run it was complaining about. Validation now precedes the first truncation.

## Test lanes, in the required order

All run on `IMPLEMENTATION_SHA`, on the hardware named below. Wall times are from these runs.

| # | lane | command | result | wall |
|---:|---|---|---|---:|
| 1 | strict cost-parser matrix | `pytest tests/test_cv_cost_schema.py` | **59 passed** | 0.2 s |
| 2 | synthetic AIS aggregation (Test A) | `pytest tests/test_ais_invocation_accounting.py` | **13 passed** | 0.1 s |
| 3 | AIS subset/resume and no-op re-entry (Tests B, C) | `pytest tests/test_ais_invocation_segment_runs.py` | **4 passed** | 28.5 s |
| 4 | end-to-end malformed-record refusals | `pytest tests/test_cv_cost_refusal_end_to_end.py` | **17 passed** | 69.5 s |
| 5 | real CUDA AIS continuation | `pytest tests/test_cv_cuda_lanes.py` | **10 passed** | 113.7 s |
| 6 | real MPI+CUDA changed world size (Test D) | `pytest tests/test_cv_mpi_cuda_ais.py -k test_d_two_rank` | **1 passed** | 33.0 s |
| 7 | all dedicated CUDA CV tests | `pytest tests/test_cv_cuda_lanes.py` | **10 passed** | 113.7 s |
| 8 | all REST2/rREST2/AIS MPI+CUDA tests | see the breakdown below | **43 passed** | 397.2 s |
| 9 | complete fast/non-slow suite | `pytest -q -m "not slow"` | **1355 passed**, 426 deselected | 170.2 s |
| 10 | complete slow/GPU suite | `pytest -q -m "slow"` | SLOW_RESULT | SLOW_WALL |
| 11 | wheel build and clean install outside the checkout | see below | WHEEL_RESULT | — |
| 12 | installed-wheel CUDA AIS subset/resume and no-op re-entry | see below | WHEEL_AIS_RESULT | — |
| 13 | GitHub Actions on the exact implementation SHA | non-CUDA hosted CI | CI_RESULT | — |

### Lane 8, itemised — the count correction §6 asked for

The previous document reported a combined MPI+CUDA lane of 42 while its itemised REST2/rREST2/AIS
rows totalled 29. Both numbers were stale in different ways: the combined figure predated the
rREST2 fail-closed cases and the AIS rescheduling case, and the itemised rows had never been
re-run after those were added. Current, all on `IMPLEMENTATION_SHA`:

| file | method | tests | wall |
|---|---|---:|---:|
| `tests/test_cv_mpi_cuda_lanes.py` | REST2 | **9** | 90.5 s |
| `tests/test_cv_mpi_cuda_rrest2.py` | rREST2 (× 2 velocity policies) | **24** | 190.4 s |
| `tests/test_cv_mpi_cuda_ais.py` | AIS | **10** | 115.2 s |
| combined, one invocation | — | **43** | 397.2 s |

9 + 24 + 10 = 43, which is the combined run. No count is carried forward from an earlier
document.

## What ran where

* **Local real CUDA and real MPI+CUDA** — lanes 5–8, 10, 12. Devices: 1 × NVIDIA RTX A5000,
  8 × NVIDIA GeForce RTX 3080, driver 580.173.02. Open MPI 5.0.8 (`prterun`), mpi4py 4.1.2.
  Rank counts: 2 and 3 for the ladders (the ladder width *is* the rank count), 2 and 4 for AIS.
  No `--cpu` appears in any of them; each fails rather than substituting a CPU platform.
* **Local CPU** — lanes 1–4, 9. These test accounting across processes and refusal before
  filesystem mutation, neither of which a platform changes. Each such file states the exemption.
* **GitHub-hosted CI** — lane 13, packaging and interface only. **There is no CUDA runner**, so
  no GPU or MPI claim rests on it.

## Environment

* Python 3.12.14, OpenMM 8.6.0.dev-c6173db, Linux 6.8.0-124-generic, x86-64
* Wheel: `WHEEL_NAME`, sha256 `WHEEL_SHA`
* Installed import origin: `IMPORT_ORIGIN`

## Earlier evidence, cross-linked

* [`20260904-cv-accounting-closure-evidence.md`](20260904-cv-accounting-closure-evidence.md) —
  the two-scope cost schema, the ladder's bit-for-bit resume, walker validation, CV provenance,
  and the first MPI+CUDA lanes. Its lane counts predate this work; the table above supersedes
  them.
* [`20260904-cv-scientific-closure-evidence.md`](20260904-cv-scientific-closure-evidence.md) —
  the rREST2 pre-refresh semantics and the correction of an over-broad MPI+CUDA coverage claim.
* [`20260904-cv-lifecycle-closure-evidence.md`](20260904-cv-lifecycle-closure-evidence.md) and
  [`20260904-cv-and-runtime-closure-evidence.md`](20260904-cv-and-runtime-closure-evidence.md) —
  CV lifecycle and runtime closure.
* [`cuda-coverage-matrix.md`](cuda-coverage-matrix.md) — generated, one row per function that
  runs on a device.

## Limitations

Recorded here rather than left implicit; none is a known test failure.

LIMITATIONS_BLOCK
