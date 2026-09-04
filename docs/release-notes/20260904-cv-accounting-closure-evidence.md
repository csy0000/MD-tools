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

Filled from the runs recorded below; see the final report for the complete command list.

## Hardware, MPI and versions

* 9 CUDA devices: 1 × NVIDIA RTX A5000, 8 × NVIDIA GeForce RTX 3080; driver 580.173.02
* Open MPI 5.0.8 (`prterun`), mpi4py 4.1.2
* Python 3.12.14, OpenMM 8.6.0.dev-c6173db
* Linux 6.8.0-124-generic, x86-64
