# CV accounting, ladder resume, provenance and MPI+CUDA closure — evidence

Implementation baseline `1e83c66`; task instruction `b453ddf`. This document records what was
found, what was changed, and what was actually executed. Where an earlier document overstated
coverage, the correction is stated here and in that document rather than edited away.

## What the work found

Five defects, four of them present in code that already had passing tests.

### 1. `cv_evaluations` counted reporter calls, not scalar evaluations

`CVSeries.evaluate` incremented a counter named `cv_evaluations` once per call, regardless of how
many torsions the definition held. A two-torsion run reported half the scalar work it had done,
under a name that says otherwise.

The two counters are numerically identical for a single-CV definition, which is how this survived
a full suite — and a test had been written asserting the behaviour was correct, with a comment
explaining that the counter counts calls. That test was wrong and has been replaced.

`cv_observations` (reporter calls) and `cv_evaluations` (scalar values, `observations × N_cv`) are
now separate fields with stated meanings. Every test that asserts on them uses **at least two
torsions**, where a call counter and a scalar counter cannot agree.

### 2. The ladder's resume was not a resume

This is the significant one, and it was found only because a test was strengthened to compare a
resumed CV series against an uninterrupted reference **by value** rather than by step grid.

Ladder checkpoints stored positions, velocities and box — the complete *physical* state of a
Langevin walker, and deliberately portable across devices. That is not enough to continue a
trajectory: the integrator's pseudo-random stream has a position within it that coordinates do
not carry. Every continued REST2/rREST2 run since the beginning drew fresh noise from the resume
point onward. The continuation was correct and statistically exact, and it was a **different
trajectory** — on a step grid that lined up perfectly, row for row, so every check that compared
step numbers passed.

cMD and AIS never had this: both already resume through `loadCheckpoint`. Only the ladder did
not, and `ReplicaEngine.integrator_state` / `load_integrator_state` sat unused, written for
exactly this purpose and never wired up.

Per-rung OpenMM context checkpoints are now stored **alongside** the coordinates, with the
platform and precision that produced them. They are an optimisation and never a requirement: a
continuation that finds them, written by the platform it is running on, restores them and
reproduces the reference exactly; one on a different device ignores them and falls back to
coordinates exactly as before. The checkpoint stays usable on another machine, and which path was
taken is announced on the run's output and recorded.

**A related fact, established while proving this.** Two replicate ladder runs with the same pinned
seed and identical inputs diverge by the first observation. The cause is OpenMM's **CPU platform**,
which sums force reductions in thread-completion order and is only reproducible at a fixed thread
count; with `OPENMM_CPU_THREADS=1` two replicates agree to the last digit. That is a property of
the platform, not of the ladder. Tests that compare series value-by-value pin the pool and say
why; nothing pins a thread count in production.

### 3. Walker identity was never validated

`walker_index` is what makes a state-centric CV series joinable to a walker-centric analysis.
Every other field was checked — step grid, state index, tau, exchange phase, frame reference,
finiteness of every value, the digest of the whole file — and this one was not read at all. A
walker of `-1`, of `n_states`, of `"2.5"`, or the same walker occupying two rungs at one step
produced a completed run, an authoritative completion marker, and files that read back perfectly.

Now checked per row (integer, `0 <= walker < n_states`) and across the set: an exchange permutes
walkers among rungs and never creates, destroys or duplicates one, so at every observation step
the walkers are exactly `0 .. n_states - 1`, each once. Two rungs claiming one walker, or a walker
occupying none, is invisible from inside any single file.

### 4. AIS validated its CV output one invocation too late

The full scientific validation of a path's CV series existed and already refused every corruption
tested here. It ran in exactly one place: when a *later* invocation skipped an already completed
path. By then `completed.json` is committed and authoritative and the trajectory is published, so
a series that was already wrong when written had been recorded as finished.

The same function now runs before the completion marker is committed, where a refusal costs a
rerun instead of a wrong answer. Four checks were added while moving it: protocol column order,
the tau schedule in the values themselves, the empty-or-non-negative-integer semantics of the two
frame references, and the cumulative cost against the row count and `N_cv`.

### 5. Provenance gaps

The ladder's per-state CV CSV and sidecar were absent from the outer `-log` output inventory —
the block `registry.discovery.check_lineage` indexes by digest to connect one stage's outputs to
the next stage's inputs. They were on disk, in `restart.json`, and in the registry's `SHA256SUMS`,
and invisible to every lineage check. The inventory is now built **from the validated completion
manifest**, never from a glob, so a file left behind by an earlier run into the same directory
cannot be recorded as this run's provenance.

Separately, AIS computed its global CV cost — the sum over completed paths, with the per-path
records that make it auditable — and returned it to a caller that dropped it. The aggregate
existed only inside the call. The aggregate CV table was likewise missing from the run's output
inventory. Both are now recorded.

## Cost schema

```yaml
collective_variable_cost:
  schema_version: 2
  segment:      {cv_observations: N, cv_evaluations: N*N_cv, wall_seconds: t}
  cumulative:   {cv_observations: M, cv_evaluations: M*N_cv, wall_seconds: T}
  aggregation:  "sum over thermodynamic states" | "sum over completed paths"
  per_state / per_path: [...]
```

`segment` covers this invocation; `cumulative` covers the complete logical simulation across every
invocation. Equal on a fresh run. On continuation the cumulative counters are restored from the
committed prefix *before any new evaluation* and only the segment resets, so a second or later
interruption neither loses nor double-counts. Rows and cost are restored as one typed object
(`CommittedPrefix`) so a caller cannot restore the rows while discarding the counters — which one
did. Verification and completed-path skips count as neither.

## Correction to the previous evidence document

`20260904-cv-scientific-closure-evidence.md` listed a lane as "real MPI + CUDA, CV enabled"
without qualification. `test_cv_mpi_cuda_lanes.py` runs **REST2 and only REST2**. rREST2's
reservoir refresh and distributed AIS had not been executed on a device under MPI with collective
variables enabled at all — the AIS cases in the older MPI lane enable no CVs. That row is now
qualified in place, and the two missing methods have lanes of their own.

## Test lanes

Re-audited against the instruction three further times after the first completion; each pass is
recorded under "What later audits found" below, and the counts here are from the current head.

| lane | command | result | wall |
|---|---|---|---:|
| fast / non-GPU | `pytest -q -m "not slow"` | **1283 passed**, 0 failed | 169.7 s |
| slow / GPU, complete | `pytest -q -m "slow"` | **403 passed**, 0 failed, 1 skipped | 2322.6 s (38:42) |
| dedicated real-CUDA CV suite | `pytest tests/test_cv_cuda_lanes.py` | **10 passed** — fresh and **twice resumed** cMD, REST2, rREST2, AIS | 105.1 s |
| all three MPI + CUDA lanes | `pytest tests/test_cv_mpi_cuda_{lanes,rrest2,ais}.py` | **42 passed** — continuation *and* injected failure per method | 385.8 s |
| CUDA coverage matrix | `pytest tests/test_cuda_coverage_matrix.py --cuda-evidence=...` | **27 passed** on real devices | 423.5 s |
| ladder CV cost and two-interruption resume | `pytest tests/test_ladder_cv_cost_resume.py` | **6 passed** (REST2 and rREST2) | 45.9 s |
| ladder walker identity and permutation | `pytest tests/test_ladder_walker_validation.py` | **8 passed** | 8.0 s |
| ladder CV provenance | `pytest tests/test_ladder_cv_provenance.py` | **5 passed** | 21.5 s |
| AIS pre-completion validation | `pytest tests/test_ais_precompletion_cv.py` | **12 passed** | 12.7 s |
| AIS decomposition and CV resume | `pytest tests/test_ais_decomposition_resume.py` | **11 passed** | 43.0 s |
| cMD committed prefix and two-interruption resume | `pytest tests/test_cv_committed_prefix.py` | **13 passed** | 21.0 s |
| real MPI + CUDA, **REST2** | `pytest tests/test_cv_mpi_cuda_lanes.py` | **9 passed** | 91.0 s |
| real MPI + CUDA, **rREST2** | `pytest tests/test_cv_mpi_cuda_rrest2.py` | **14 passed** (3 ranks × both velocity policies) | 92.1 s |
| real MPI + CUDA, **AIS** | `pytest tests/test_cv_mpi_cuda_ais.py` | **6 passed** (2 ranks, 4 globally numbered paths) | 41.4 s |

The single skip is `test_write_the_coverage_evidence`, which writes the matrix only when
`--cuda-evidence` is given, so an ordinary GPU run does not rewrite a committed file; it is
exercised in the matrix row above. No `--cpu` appears in any of the three MPI+CUDA lanes; each fails rather than substituting a CPU
platform when no device is available. Nothing was xfailed, skipped or deselected by hand.

### Restoring the defects

Each change was checked by putting the defect back and confirming the tests fail:

* ladder context blobs disabled → both two-interruption tests fail, REST2 on diverging `phi`;
* walker validation removed → all six negative cases fail, every one because the old code
  returned an empty problem list;
* AIS pre-completion validation reverted → the four new rules and the ordering assertion fail,
  and the seven older rules still pass, which is the point: they were never wrong, only late;
* ladder CV provenance reverted → the inventory and foreign-file tests fail.

## Installed wheel, outside the checkout

`md_tools-0.5.0.dev0-py3-none-any.whl`, sha256
`99ac8fe0f1ff0788df615646ff1d589865b6e60a620ccd78e7206b69fe6a05f2`, installed into a fresh
virtualenv and run from a directory outside the checkout. Import origin:
`…/wheelenv2/lib/python3.12/site-packages/md_tools/__init__.py`.

All runs used a two-torsion definition and **no `--cpu`**.

| run | invocations | result |
|---|---|---|
| cMD fresh | 1 | completed on CUDA; segment == cumulative == 13 observations / **26** scalar evaluations |
| cMD multiply resumed | crash (5 rows) → crash (9 rows) → complete (13 rows) | series **byte-identical** to the uninterrupted reference; segment 4 obs / 8 evals, cumulative 13 obs / 26 evals |
| AIS fresh | 1 | completed on CUDA; 5 rows, 10 scalar evaluations per path |
| AIS multiply resumed | crash `after-work-row` → crash `after-frame` → complete | per-path `cv.csv` and the global `AIS_cv.csv` **byte-identical** to the uninterrupted reference; segment 3 obs / 6 evals, cumulative 5 obs / 10 evals; aggregate "sum over completed paths", 10 obs / 20 evals over 2 paths |

| REST2 fresh, **3 MPI ranks** on CUDA | `mpirun -n 3 … -ng 3` | completed; `local_rank` device policy, 27 observations / **54** scalar evaluations over 3 rungs |
| REST2 **3 ranks, twice interrupted** | crash `after-cv-row` → crash `after-checkpoint` → complete | all three per-state series **byte-identical** to the uninterrupted reference |

The cumulative counter is exactly `rows × 2` in every case, which a reporter-call counter cannot
produce — that is what the two-torsion definition is for. The multi-rank rows also demonstrate
bitwise ladder resume under MPI on real devices from the installed wheel, not only from the
checkout.

## Complete MPI and multi-GPU sweep

With every device free, all MPI and multi-GPU lanes were run together:

```
pytest tests/test_cv_mpi_cuda_lanes.py tests/test_cv_mpi_cuda_rrest2.py \
       tests/test_cv_mpi_cuda_ais.py tests/test_md_run_mpi_gpu.py \
       tests/test_mpi_fail_closed.py tests/test_driver_fail_closed.py
```

**78 passed**, 0 failed, 10:29. That covers REST2, rREST2 and AIS CV lanes under `mpirun` on
real devices, the hundred-path AIS campaign distributed across four ranks, and both fail-closed
suites, in which the subprocess timeout is the assertion: a rank that raises alone must stop the
whole communicator rather than leave the others blocked in a collective.

## What later audits found

Re-reading the instruction clause by clause, three times after first reporting the work
complete, found four things. They are listed because the pattern is the point: each pass found
something, so the confidence attached to any single pass should be moderate.

1. **A real defect, section 1.** cMD's *final* committed checkpoint generation carried no CV
   prefix. A stage commits generations periodically through its checkpoint reporter and once
   more at the end; the periodic path supplied the prefix and the final one did not, beneath a
   comment saying the two went through the same transaction. So the generation a later reader
   actually consults vouched for a CV row count while carrying no digest of those rows and no
   cumulative counters. Found by reading a committed generation's JSON on disk, not from the
   code. The ladder and AIS were unaffected — both write the prefix through a single path.
2. **Section 3's rescheduling clause was untested.** A resume under a changed worker count is
   supported deliberately: a global path id always owns the same file name. Every resume in the
   suite used the rank count it crashed with — the one case where a reshuffle cannot appear. Now
   tested, and `mpi_rank` is asserted to be the ONLY work-table column that changes.
3. **Section 8.3 was not satisfied.** The dedicated real-CUDA CV suite interrupted each protocol
   once, and rREST2 not at all. The clause asks for *multiply* resumed on a device — which
   matters here specifically, because the restore path being exercised loads an OpenMM context
   checkpoint whose contents are platform-specific, so a CPU run proves nothing about it.
4. **Section 8.4's injected-failure half was missing for rREST2 and AIS.** Both had continuation
   coverage under a real launcher and no test that a rank dying ALONE stops the job rather than
   leaving the others blocked in a collective.

Section 9's documentation list was also only half done, and section 7's regression list was
verified by inspection against the existing tests rather than re-derived.

## Limitations, and which of them were closed

* **Bitwise ladder resume is platform-bound.** OpenMM context checkpoints are device-specific by
  construction and there is no public API to read or restore an integrator's pseudo-random
  stream position portably, so a cross-device continuation falls back to coordinates. The
  fallback is correct, statistically exact, announced on the run's output and recorded — it is
  simply not the same trajectory. Not fixable from here.
* **Checkpoint size.** Measured at **3.53×** the coordinate payload on a two-rung ladder.
  Compression was tested and rejected on evidence: zlib reaches 98.9% of raw, because the blobs
  are binary doubles. The cost is bounded — the checkpoint is rewritten whole rather than
  accumulated — and the coordinates remain authoritative.
* **CPU-platform reproducibility.** Not fixable here; OpenMM's CPU platform sums force
  reductions in thread-completion order. The record now carries `cpu_threads` so a reader can
  tell whether two runs were comparable at all, which is what was actually missing.
* **GitHub CI has no CUDA runner.** All GPU and MPI evidence is local, on the hardware named
  above. Closing this needs a self-hosted runner, which is an infrastructure decision.
* **Closed since the first draft:** the installed-wheel evidence was single-device; it is now
  also a three-rank MPI run on real CUDA, twice interrupted, reproducing its reference exactly.

## Hardware, MPI and versions

* 9 CUDA devices: 1 × NVIDIA RTX A5000, 8 × NVIDIA GeForce RTX 3080; driver 580.173.02
* Open MPI 5.0.8 (`prterun`), mpi4py 4.1.2
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db
* Linux 6.8.0-124-generic, x86-64
