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

All run on `c22b757`, on the hardware named below. Wall times are from these runs. The
commit that adds this document is `c22b757` plus these results and nothing else; `git diff`
between the two touches no code and no test, and CI is green on both.

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
| 10 | complete slow/GPU suite | `pytest -q -m "slow"` | **425 passed**, 1 skipped | 2472.0 s (41:12) |
| 11 | wheel build and clean install outside the checkout | `python -m build --wheel`, then `pip install` into a fresh venv | built and imported | — |
| 12 | installed-wheel CUDA AIS subset/resume and no-op re-entry | generated tree run from the installed wheel | subset, resume and re-entry all rc=0; **segment exactly 0** on re-entry | — |
| 13 | GitHub Actions on the exact implementation SHA | hosted, non-CUDA | **success** — [run 34094228043](https://github.com/csy0000/MD-tools/actions/runs/34094228043) | — |

### Lane 8, itemised — the count correction §6 asked for

The previous document reported a combined MPI+CUDA lane of 42 while its itemised REST2/rREST2/AIS
rows totalled 29. Both numbers were stale in different ways: the combined figure predated the
rREST2 fail-closed cases and the AIS rescheduling case, and the itemised rows had never been
re-run after those were added. Current, all on `c22b757`:

| file | method | tests | contended | dedicated |
|---|---|---:|---:|---:|
| `tests/test_cv_mpi_cuda_lanes.py` | REST2, 2 ranks | **9** | 90.5 s | 61.8 s |
| `tests/test_cv_mpi_cuda_rrest2.py` | rREST2, 3 ranks (× 2 velocity policies) | **24** | 190.4 s | 122.0 s |
| `tests/test_cv_mpi_cuda_ais.py` | AIS, 2 and 4 ranks | **10** | 115.2 s | 77.1 s |
| combined, one invocation | — | **43** | 397.2 s | 259.8 s |

9 + 24 + 10 = 43, which is the combined run. No count is carried forward from an earlier
document.

**Contended** is the original run, sharing devices with unrelated jobs on the same host.
**Dedicated** is a rerun with `CUDA_VISIBLE_DEVICES=0,5,6,7,8`, one rank per device for every
lane including the four-rank case. The counts are identical; only the wall times move.

### The counts do not depend on how many devices are visible

The same 43 tests were run at three device counts, deliberately:

| visible devices | placement | result | combined wall |
|---|---|---|---:|
| 1 (`CUDA_VISIBLE_DEVICES=0`) | round-robin, up to 4 ranks per device | **43 passed** | 259.5 s |
| 2 (`7,8`) | round-robin for the 3- and 4-rank lanes | **43 passed** | 263.3 s (sum of three separate invocations) |
| 5 (`0,5,6,7,8`) | one rank per device throughout | **43 passed** | 259.8 s |

The 1- and 5-device figures are single combined invocations; the 2-device figure is the sum
of the three files run separately, and is marked as such rather than presented as one run.

That invariance is the design working rather than a coincidence: `select_device_for_rank`
falls back to a documented round-robin — "N ranks share M device(s)" — instead of failing or
silently placing every rank on the default device, and AIS path identity does not depend on the
worker count. The three wall times are within 5% of each other, which says these systems are far
too small to be GPU-bound; the 35% gap against the contended run is the unrelated jobs, not the
device count. No throughput claim should be read from any of these numbers.

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
* Wheel: `md_tools-0.5.0.dev0-py3-none-any.whl`, sha256 `1b4bc1c2a9baed07e42d7c4080cbe15d612f5bc757a93cb1f163a332613e07c9`
* Installed import origin: `…/env2/lib/python3.12/site-packages/md_tools/__init__.py` (outside the checkout)

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

* **GitHub-hosted CI has no CUDA runner.** Lane 13 proves packaging and interface only. Every
  GPU and MPI claim in this document rests on the local hardware named above, and is labelled as
  such rather than folded into a single "CI is green".
* **The CPU platform is reproducible only at a fixed thread count.** Tests that compare
  trajectory-dependent values pin `OPENMM_CPU_THREADS=1`; production pins nothing, deliberately.
  Lane 10 caught a file that had not pinned it — see below. Neighbouring CPU files that compare
  only grids and structure remain unpinned, correctly; a future value comparison added to one of
  them would need the same pin.
* **Schema-version-1 records are refused, not auto-migrated.** `migrate_schema_1` exists and is
  tested, but must be called explicitly with the verified row count and CV definition, and stamps
  its output as reconstructed. Nothing converts a legacy record silently.
* **A single-path invocation writes no global table.** `--paths N` selects a subset, so no
  campaign aggregate — and therefore no aggregate segment — is recorded for it. That is by
  design: a subset is not a campaign. The per-path manifest still records its own cost.
* **Aggregate wall-time comparison carries a microsecond epsilon per entry.** Stored durations
  are rounded to six decimal places, so a re-summation can differ in the last place. Integer
  counters are compared exactly; only seconds carry the tolerance, and the constant is named
  `SECONDS_EPSILON` rather than being inlined.

## One lane failed and was diagnosed rather than adjusted

The first run of lane 10 failed `test_rrest2_pre_refresh_cv.py::
test_the_row_holds_the_propagated_configuration_not_the_reservoir_sample[stored]` with a
separation of 0.589 degrees against a threshold of 1.0 — and the same test passed in isolation.

That difference was the diagnosis. The file's subprocesses did not pin `OPENMM_CPU_THREADS`, so
its trajectory depended on machine load; inside a loaded full-suite run the walker drifted 0.589
degrees, and run deterministically the same four events separate by 159.9, 168.3, 122.8 and 97.7
degrees. The threshold was never the problem and was not touched: the pool is pinned, and the
margins were measured afterwards to confirm the fix is real rather than lucky.

This was latent flakiness that the new tests exposed by making the suite busier, not a regression
from this task. Retry history for every other lane: none failed.
