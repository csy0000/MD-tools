# CV lifecycle and restart accuracy: what was corrected, and the evidence

Baseline `67b88d8`; instruction `540f7d4`. Every number here was measured on this machine,
at a named commit, with real CUDA devices, a real MPI launcher and a real installed wheel.
Where a claim has no measurement behind it, it is not made.

This record **replaces the completion claim** of
[`20260904-cv-and-runtime-closure-evidence.md`](20260904-cv-and-runtime-closure-evidence.md).
That pass reported green lanes and was right about them; it was wrong that the work was
finished. Ten defects survived it, and every one produced output that looks entirely
ordinary — right header, right column count, plausible monotonic numbers — which is why the
lanes were green.

## Implementation commits

| commit | what |
|---|---|
| `35b9036` | one absolute-step convention after an OpenMM checkpoint restore |
| `281d5ad` | REMD step 0, real CV continuation, honest trajectory-frame alignment |
| `202d4f5` | CV outputs join the typed inventories, the overwrite transaction and completion |
| `dacbe6c` | the AIS three-group decomposition proven to survive interruption exactly |
| `0797608` | definition-change refusal matrix, and `md-run` parity |
| `fbfe706` | documentation corrected for the behaviour above |

## Test lanes

| lane | command | result | wall |
|---|---|---|---:|
| fast | `pytest tests -m "not slow and not gpu"` | **1268 passed**, 0 failed, 287 deselected | 186.8 s |
| CUDA + slow | `pytest tests -m "gpu or slow"` | **286 passed**, 0 failed, 1 skipped, 1268 deselected | 1431.4 s |
| MPI + multi-rank CUDA | `pytest tests/test_mpi_fail_closed.py tests/test_driver_fail_closed.py tests/test_md_run_mpi_gpu.py` | **49 passed**, 0 failed | 704.6 s |
| wheel | `python -m build --wheel`, `pip install --no-deps` into a fresh venv | built, installed, origin-asserted, fresh **and resumed** CV runs verified | — |

The single skip is `test_write_the_coverage_evidence`, which writes the matrix only when
`--cuda-evidence` is given so an ordinary GPU run does not rewrite a committed file. It ran
and wrote in the evidence lane. Nothing was xfailed, skipped or deselected by hand.

## Hardware, MPI and versions

* 9 CUDA devices: 1 × NVIDIA RTX A5000 (24564 MiB), 8 × NVIDIA GeForce RTX 3080
  (10240 MiB each); driver 580.173.02
* Open MPI 5.0.8 (`prterun`), mpi4py 4.1.2
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db
* Linux 6.8.0-124-generic, x86-64

## The ten defects, and how each was proven

### 1. Doubled absolute steps after every checkpoint restore

`loadCheckpoint` **restores** the Context's step count — verified directly: a checkpoint
taken at step 25 comes back with `currentStep == 25`. Both `CVReporter` and
`PhaseSpaceReporter` nevertheless added the already-completed count again.

Measured with the defect deliberately restored: a resume from step 30 of a 40-step run
wrote its next CV observation at **step 65**, and the series read `[0,5,…,30,65,…]` — past
the requested budget entirely.

`simulation.currentStep` is now the single authority. The offset parameters are **removed**,
not defaulted to zero, including a `_CheckpointWithFingerprint.offset` that was stored and
never read.

Evidence: `tests/test_cmd_cv_resume_grid.py` compares an interrupted-then-resumed run
against an uninterrupted one and requires the grids to be **equal**, parametrised over
interruption at steps 10, 20 and 30. All three were confirmed to fail against the restored
defect. It also covers the derived time axis, global frame indices across a continued
trajectory, and the same bug in the phase-space stream a reservoir is built from.

### 2. A stage interrupted before its first trajectory frame could not be continued at all

Found while testing the above. The DCD reporter creates its file on construction, not on
first write, so an early interruption leaves a headerless file and
`DCDReporter(append=True)` fails with *"Cannot append to file with invalid DCD header"*.
The append decision now asks whether the trajectory holds a readable frame.

### 3. REST2/rREST2 CV series omitted step 0

The ladder observed only at schedule events, and `events_at` never returns anything at step
0. A 40-step run at interval 5 wrote `5…40` and silently omitted the initial configuration —
the one every later row is a displacement from. The existing test asserted `[5,…,40]`,
codifying the defect as a contract.

Step 0 is now written before a single step is propagated, with `exchange_attempt = -1` and
the identity mapping, guarded on the row count so it is idempotent under continuation.

### 4. A resumed ladder wrote no CV rows at all

`_continue` never reopened `cv_states`. The run completed, every other stream continued
correctly, and the CV files simply stopped at the crash.

The checkpoint now vouches for the committed row count; continuation validates read-only
**before** anything is opened for writing, then truncates to that count and appends. A
checkpoint with no recorded count refuses with a compatibility message rather than guessing
from file length — which rows are durable is exactly what the count exists to say.

Evidence: resume tested at three boundaries (`after-cv-row`, `before-checkpoint`,
`after-checkpoint`) against an uninterrupted reference, plus a legacy-checkpoint refusal.

### 5. `trajectory_frame_index` named a frame holding a different configuration

This one needed a decision, not an off-by-one fix. Diagnosed directly: at step 30 of a
3-state run, state 0's CV value matched `remd1.nc[2]`, **not** `remd0.nc[2]`.

The state trajectory frame is written from the **post-exchange** occupant; the CV row
describes the **pre-exchange** one. At an accepted swap those are different configurations,
so *no frame in that file holds what the row measured*.

Naming is therefore **per state**: named when the exchange did not move that state's walker,
empty when it did, never `-1`. The trajectory and pre-exchange conventions are unchanged;
only the write moves after the exchange, evaluating from a frozen pre-exchange snapshot.

Evidence: every named NetCDF frame is loaded and the torsion recomputed independently. The
per-state rule is asserted from the files themselves, and the test **fails if no accepted
exchange landed on a frame step**, since the rule would then never have been exercised.

### 6. The cMD inventory named a sidecar that never existed

`<stage>.cv.yaml` against a writer producing `<stage>.cv.json`. The real sidecar was in no
inventory: not collision-checked, not removed by `--overwrite`, free to survive a definition
change and describe the new CSV with the old atom mapping. There is now one definition of
that name, shared by writer, inventory, completion record and AIS manifest.

### 7. Ladder and aggregate CV outputs were in no inventory

`cv_stateN.csv`, `cv_stateN.json` and `AIS_cv.csv` were unowned, so a CV-disabled rerun left
them in place permanently, describing a calculation that no longer exists.

Evidence: a CV-enabled run followed by a CV-disabled `--overwrite` is asserted to leave **no**
CV artefact at all.

### 8. AIS completion recorded the row count but not the files

`cv_rows` was in `completed.json`; `cv.csv` and `cv.json` were not. A path could be skipped
as complete while its CV output had been truncated, mutated or deleted — the count agreed
with itself and nothing looked at the bytes. Both files are now fsynced before the
completion commit, hashed into the manifest, and required on both the write and read side,
from the **schedule** rather than from whether the files happen to exist.

Evidence: completion refused after truncating, mutating, deleting the CSV and after
replacing the sidecar — four damage modes, each leaving a file that still parses.

### 9. `md-run` could not run any CV-enabled input

`md-run` writes its own `resolved.config` into `-odir`, and the definition is resolved
relative to the directory holding it — but the content-addressed copy had never been put
there. A configuration that ran perfectly as a generated script failed with *"no such
collective-variable definition file"*. Fixed by carrying the definition into `-odir`, so the
resolution rule stays single and `-odir` holds everything needed to read its own output.

### 10. A test anchor matched the wrong method

`source.index("def _continue")` also matched the new `_continue_cv_states` and sliced half
the class. Anchors are now precise.

## AIS three-group work and Hummer–Szabo alignment

No production change: the decomposition, its terminology and the two-coordinate distinction
were already correct. What was missing was evidence that they stay correct across a resume,
which is where accumulated quantities go wrong invisibly.

* `delta_work_total = non_scaled + sqrt_scaled + lin_scaled` asserted on **every** row,
  incremental and cumulative. The measured total is never derived from the components; that
  independence is what makes their agreement test anything.
* Interruption at all four stream boundaries (`before-frame`, `after-frame`,
  `before-work-row`, `after-work-row`), with every cumulative group required to match an
  uninterrupted reference — not merely to be self-consistent.
* `AIS_hs.csv` asserted to be the frame-aligned subset: every row carries a saved coordinate,
  all three group potentials, and a direct-versus-reconstructed check.
* A bare `scaled` column asserted absent in all three families. Three components carry a
  scaling, so the name would not say which, and the ambiguity would land in exactly the files
  a reweighting is built from.

## Installed wheel

```text
wheel  : md_tools-0.5.0.dev0-py3-none-any.whl
origin : …/lanes2/wheelbuild/venv/lib/python3.12/site-packages/md_tools/__init__.py
```

The origin check is an assertion, not a print: it fails if `md_tools` resolves into the
source checkout. All four commands answer. Both a **fresh** and an **interrupted-then-resumed**
cMD run were executed outside the checkout with CV enabled, and both produced the identical
grid:

```text
step,time_ps,trajectory_frame_index,phi
0,0.000000,,-175.645070
5,0.010000,,-175.124781
10,0.020000,,-173.115677
15,0.030000,,-170.485457
20,0.040000,0,-166.888169
25,0.050000,,-162.872481
30,0.060000,,-159.834645
35,0.070000,,-157.065244
40,0.080000,1,-153.830993
```

Step 0 and the final step exactly once, frame indices only where a frame exists, and the
resumed values matching the fresh ones to ~1e-6. That is defect 1's evidence from an
installed wheel.

## Limitations, still true

* **v1 is torsions only.** No distances, angles, RMSD or biasing. An unsupported `type` is
  refused by name with the schema version.
* **The runs here are mechanism tests.** Twenty to forty steps, two to four exchanges, two
  AIS paths. They verify bookkeeping; they support **no** claim about convergence, sampling
  quality or any physical result.
* **CPU-platform runs are not bit-reproducible** (~9 × 10⁻⁹ nm on this system), so any
  coordinate comparison after dynamics is a tolerance comparison against a measured floor.
* **The MPI failure-injection lane runs at 2 ranks.** Larger worlds are exercised by the
  other CUDA lanes but not with injected rank-local failures.
* **At an accepted exchange, a ladder CV row names no trajectory frame.** This is deliberate
  and is the honest answer, but it does mean that in a ladder whose exchange and trajectory
  cadences coincide, the fraction of CV rows carrying a frame index falls as acceptance
  rises. A run needing every CV row frame-aligned should set a trajectory cadence that does
  not coincide with the exchange interval.
* **CI has no CUDA runner.** It validates packaging and interface only. Every CUDA and MPI
  number above is local evidence, deliberately kept distinct from the exact-head CI result.
