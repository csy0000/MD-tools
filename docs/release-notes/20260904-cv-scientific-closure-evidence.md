# rREST2 CV semantics, completion, cost and CUDA evidence

Baseline `941b205`; instruction `bbb837b`. Every number here was measured on this machine, at
a named commit, with real CUDA devices, a real MPI launcher and a real installed wheel. Where
a claim has no measurement behind it, it is not made.

This record supersedes the completion claims of the two earlier CV passes. Both reported green
lanes and were right about them. Neither had noticed that a generated rREST2 run never touched
its reservoir.

## Implementation commits

| commit | what |
|---|---|
| `657cdcc` | rREST2 evaluates the pre-exchange CV on the propagated configuration; a declared reservoir selects the refresh rule |
| `de3bf5f` | ladder CV series become authoritative completed outputs |
| `c749921` | committed CV prefix protected by digest; CV cost persisted (cMD) |
| `c4b06de` | the same for the ladder and AIS; AIS CV verified structurally |
| `d077322` | genuine CUDA CV lanes; the false CUDA attribution removed |
| `47494ff` | real multi-rank CUDA CV continuation and fail-closed lanes |
| `e23466f` | documentation corrected for the guarantees that only now hold |

## The two scientific findings

### A generated rREST2 run never refreshed from its reservoir

Found while trying to force a deterministic refresh for the CV test. `_load_rule` returned the
plain neighbouring rule unless `--exchange-rule` was given, and the generated rREST2 project
passes `--reservoir` **without** it. Only `ReservoirRefreshRule` produces a `reservoir_refresh`.

So the reservoir was opened, validated against the top rung's Hamiltonian, reported in the run
header as `6 phase-space sample(s), velocity_policy=stored` — and never drawn from once.
Measured directly: `reservoir_events()` returned four rows of `[-1 -1 -1 -1]`.

**Every generated rREST2 run was a REST2 run wearing rREST2's output names**, and nothing in the
output said so, because "0 refreshes" is also what a legitimately rejecting reservoir prints.

A declared reservoir now selects the refresh rule, because that is what rREST2 *is*. An explicit
`--exchange-rule` still wins. The probability-one rule itself is untouched.

*Anyone holding rREST2 results produced before `657cdcc` should treat them as REST2 results.*

### The pre-exchange CV read a post-refresh configuration

The driver snapshotted only `state_to_walker` before `_exchange`. That undoes a permutation, but
`_apply_reservoir` **replaces** an entry in `state["configurations"]` outright. The CV evaluated
from that list afterwards produced a row labelled *pre-exchange* whose value was measured on a
configuration the run never propagated at that state — it came out of the reservoir file.

Measured with the defect restored: state 2 step 10 reported **20.0**, exactly the reservoir
sample's torsion. Six of the eight regression tests fail against it; all eight pass with the fix.

The complete walker-indexed configurations are now deep-copied before the exchange. A refreshed
state also names no trajectory frame, even though its walker index did not change.

## Test lanes

| lane | command | result | wall |
|---|---|---|---:|
| local CPU / fast | `pytest tests -m "not slow and not gpu"` | **1279 passed**, 0 failed, 333 deselected | 171.7 s |
| local CUDA + slow | `pytest tests -m "gpu or slow"` | **332 passed**, 0 failed, 1 skipped, 1279 deselected | 1944.0 s |
| local real CUDA, CV enabled | `pytest tests/test_cv_cuda_lanes.py` | **6 passed**, 0 failed | 53.8 s |
| real MPI + CUDA, CV enabled | `pytest tests/test_cv_mpi_cuda_lanes.py` | **9 passed**, 0 failed | 92.9 s |
| installed wheel, outside the checkout | build + install + fresh and resumed CUDA CV runs | green | — |
| exact-head GitHub CI | packaging and interface; **no CUDA runner** | see the final report | — |

Nothing was xfailed, skipped or deselected by hand. The one skip in the slow lane is
`test_write_the_coverage_evidence`, which writes the matrix only when `--cuda-evidence` is given
so an ordinary GPU run does not rewrite a committed file.

## Hardware, MPI and versions

* 9 CUDA devices: 1 × NVIDIA RTX A5000 (24564 MiB), 8 × NVIDIA GeForce RTX 3080 (10240 MiB each);
  driver 580.173.02
* Open MPI 5.0.8 (`prterun`), mpi4py 4.1.2, 2 ranks in the CV continuation and injection lanes
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db
* Linux 6.8.0-124-generic, x86-64

## CUDA evidence, and what was wrong with the previous claim

The matrix cited `test_ais_cv_output.py` as CUDA evidence for the AIS CV path. **That file invokes
`--cpu`.** The cMD and REST2 CV restart tests do the same, and the AIS cases in the MPI lane do
not enable collective variables at all. The CV runtimes therefore had no CUDA evidence whatsoever
while the published matrix said they did — which is worse than a gap: a gap invites work, a false
entry closes the question.

`tests/test_cv_cuda_lanes.py` runs six lanes **without `--cpu`**, each asserting the recorded
platform is CUDA:

1. cMD CV, fresh
2. cMD CV, interrupted and resumed — absolute-step grid and accumulated cost on a device
3. fixed-tau phase space **plus** CV, across a resume
4. REST2 CV, fresh and resumed
5. rREST2 with a real accepted refresh — the pre-refresh semantics, on a device
6. AIS CV with the three-group decomposition, fresh and resumed

A test that cannot get a device **fails**. It does not fall back and it is not xfailed.

The matrix entry now names the CUDA test by node id. `test_ais_cv_output.py` remains valuable as
the exhaustive CPU bookkeeping matrix; it is simply not CUDA evidence.

### A note on the MPI injection table

The first version was the full product of (rank, boundary), and three cases **passed by failing to
fail**: arming a CV or checkpoint boundary on a non-root rank injects nothing, because those seams
sit inside root-only blocks. The rank never reaches them, the run completes, and the test would
have sat in the suite proving the opposite of what it claimed.

The table now records which pairs are reachable and why — a non-root rank's work during a CV step
is propagation, so that is where its failure is injected — with a companion test pinning that CV
observation is root-only, so the constraint cannot silently drift.

## Committed-prefix and resume evidence

A checkpoint records a **digest of exactly the header plus the committed rows**, in the same
generation transaction as the Context state. Not the whole file: after a crash the file is
legitimately longer, and hashing the uncommitted tail would make every ordinary crash look like
corruption.

Twelve in-process cases plus an end-to-end interrupted stage: mutated value, renumbered step,
short file, missing record, wrong identifier, non-finite value, changed columns, grid gap, changed
sidecar, and an uncommitted tail that is allowed and then truncated. The end-to-end case asserts
the refused continuation left the file **byte-identical** — a refusal must not modify what it is
refusing.

Ladder prefixes are per state, not one combined hash: a combined hash cannot say which file
changed, and two states' files being swapped would leave it unchanged while every series became
another's.

## Completion-mutation evidence

**Ladder** (`restart.json`), seven damage modes, each leaving a file that still parses: delete
CSV, delete sidecar, truncate, mutate a value, mutate a step, replace the sidecar, and **swap two
states' files**. The swap is the sharpest — both files are intact, valid CV series, just each
other's — and only the recorded per-state identity distinguishes them.

**AIS** (`completed.json`), six damage modes: truncate, mutate a value, mutate a step, mutate the
path index, delete the CSV, replace the sidecar. Plus: a refused rerun does not rewrite
`AIS_cv.csv` from the path it has just rejected.

An extension is refused from a damaged CV parent, read-only, before anything local exists.

## CV cost accounting

From the resumed installed-wheel run on CUDA:

```text
{"cv_evaluations": 9, "cv_rows": 9, "cv_seconds": 0.002382,
 "segment_cv_evaluations": 2, "segment_cv_seconds": 0.000573}
```

The segment contributed 2 evaluations and the committed prefix carried 7 — so the resume neither
reset nor double-counted. Both halves stay visible: a single number replacing the other would make
an interrupted run look cheaper than an identical uninterrupted one. No key contains "energy": a
position-only torsion is not an energy evaluation.

## Installed wheel

```text
wheel  : md_tools-0.5.0.dev0-py3-none-any.whl
origin : …/lanes3/wheelbuild/venv/lib/python3.12/site-packages/md_tools/__init__.py
```

The origin check is an assertion, not a print: it fails if `md_tools` resolves into the source
checkout. All four commands answer. Fresh **and** interrupted-then-resumed cMD runs with CV
enabled, on CUDA, outside the checkout, both producing:

```text
step,time_ps,trajectory_frame_index,phi
0,0.000000,,-175.638981
5,0.010000,,-174.937434
10,0.020000,,-172.470879
15,0.030000,,-169.722892
20,0.040000,0,-167.776820
25,0.050000,,-165.766593
30,0.060000,,-162.804455
35,0.070000,,-158.973716
40,0.080000,1,-157.044882
```

## Limitations

* **v1 is torsions only.** No distances, angles, RMSD or biasing.
* **These are mechanism tests.** 20–40 steps, 4 exchanges, 2 AIS paths, 2 ranks. They verify
  bookkeeping and semantics; they support **no** claim about convergence or any physical result.
* **At an accepted exchange or a reservoir refresh, a ladder CV row names no trajectory frame.**
  Deliberate and correct — no frame in that file holds what the row measured — but in a ladder
  whose exchange and trajectory cadences coincide, the fraction of frame-aligned rows falls as
  acceptance rises. A run needing every row aligned should offset those cadences.
* **CV rows are written by root only.** A non-root rank's CV-boundary failure cannot be injected
  because the seam is not on its path; its propagation failure is injected instead.
* **The MPI lanes run at 2 ranks.** Larger worlds are exercised by the other CUDA lanes without
  injected failures.
* **Explicit-solvent AIS CV/HS alignment is not covered.** The AIS CV lanes are implicit-solvent;
  the explicit-solvent case was not run and is not claimed.
* **CI has no CUDA runner.** It validates packaging and interface only. Every CUDA and MPI number
  above is local evidence, deliberately kept distinct from the exact-head CI result.
