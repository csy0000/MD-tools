# REST2 production correctness

| | |
|---|---|
| Date | 2026-08-29 / 2026-08-30 |
| MD-templates | `feat/openmmtools-rest2`, `f2c3c48b06f3f8492f7ce0776a05b75c4fc15757` → `068ef2f0538a20260e2d40b566ca862e50c6b2c8` |
| MD-project | `dev4-openmmtools-rest2`, `3625b7825ae3014d42d75f9187d465118815452a` → `58905c8246507d1859d88a32b667f0b70a1d6216` |
| Instruction | [`../claudecode-instructions/20260829_rest2-production-correctness.md`](../claudecode-instructions/20260829_rest2-production-correctness.md) |

Both remote heads were fetched and verified before any edit. The instruction's stated starting
point for MD-project was `ca03f478`; the branch had advanced by one commit — `3625b782`, which adds
the instruction itself — so work continued from the verified remote head.

Every one of the four claims the instruction identifies was correct, and all four were written in
the previous pass.

---

## Environment

```text
python 3.12.14    openmm 8.6.0 (8.6.0.dev-c6173db)   openmmtools 0.26.0
netCDF4 1.7.4     numpy 2.4.6                        pymbar 4.2.0
numba 0.67.0      mpi4py 4.1.2 + OpenMPI             md-data 0.2.0 @ 48628f9a
```

Environment: `$DATA_ROOT/software/md-stack/envs/openmm-rest2`. Hardware: 9 CUDA devices
(1 × RTX A5000, 8 × RTX 3080), driver 580.173.02.

## Phase 1 — the real OpenMMTools contract, and the exchange-ownership decision

All three mixing alternatives were run on the same system before choosing:

| alternative | result |
|---|---|
| stock `swap-all` | **works.** With numba present the Metropolis runs entirely inside `ReplicaExchangeSampler._mix_all_replicas_numba` |
| stock `swap-neighbors` | fails: `TypeError: only 0-dimensional arrays can be converted to Python scalars` |
| locally patched neighbour | works, but the Metropolis call is project code |

`_mix_all_replicas` passes plain integers to `_attempt_swap`, which is why `swap-all` is unaffected
by the NumPy incompatibility that breaks the neighbour path.

**`swap-all` is now the default.** OpenMMTools genuinely owns every accept/reject decision, so the
documentation can say so. What this repository still owns about mixing is *when* it is attempted —
the stride gate — and that is documented as scheduling, separately from the decision.
`swap-neighbors` remains selectable and records `exchange_decision_owner: md-templates`; the
generated protocol names the scheme explicitly, and an unknown scheme is refused at generation.

Its different proposal semantics are documented rather than glossed: `n_replicas**3` uniformly
random replica pairs per mixing event over **all** state pairs, and pairs may be drawn with
`i == j`. Those self-swaps have `log_p = 0`, are always accepted, and land on the **diagonal** —
counted as acceptance they would have shown a meaningless figure near 100%, so every reported
statistic is off-diagonal and the diagonal is reported separately.

Public APIs used, all of them documented: `create(metadata=...)`, `read_dict('metadata')`,
`read_mixing_statistics`, `read_last_iteration`, `read_checkpoint_iterations`,
`read_replica_thermodynamic_states`, `read_sampler_states`, `read_energies`, `from_storage`,
`extend`. Private methods touched: `_mix_replicas` (scheduling, both schemes) and `equilibrate`;
plus `_mix_neighboring_replicas` and `_attempt_swap` **only** under `swap-neighbors`.

## Phase 2 — interrupted resume

Identity is now built before the sampler exists and stored in reporter metadata via
`create(metadata=...)`, inside the authoritative storage. An atomic `rest2.runstate.json` sidecar
beside the NetCDF tracks `initialized → running → completed/interrupted/failed`, rank 0 only, and
never claims completion. SIGINT and SIGTERM become a recorded interruption.

**`--resume` no longer requires `restart.json`.** Requiring it was the defect: an interrupted run
never writes one. The launcher's precondition was removed and the runtime reads identity from
reporter metadata, with the sidecar as an independent fallback; a continuation that can establish
neither is refused.

Runtime evidence, on CPU, 2269-particle solvated ALA, 3 replicas:

```text
run                  SIGTERM at iteration 49 of 120; restart.json ABSENT
sidecar              status: interrupted, reason: KeyboardInterrupt: signal 15
storage              last committed 49, last checkpoint 48, identity present in metadata
--resume             identity verified against reporter metadata -> completed 120/120
--extend 5           120 -> 130 iterations, 60 -> 65 mixing events
```

## Phase 3 — authoritative validation

One validator, `rest2_validate.py`, copied into every generated project and reached through
`openmm-rest2 --verify-only`. It opens the storage through `MultiStateReporter` and reads it.

Verified against real storage:

| case | result |
|---|---|
| valid completed run, with manifest | VALID, exit 0 |
| valid **extended** run, no manifest | VALID, exit 0 |
| truncated analysis NetCDF | INVALID |
| truncated checkpoint NetCDF | INVALID |
| manifest copied from a different run | INVALID (4 problems) |
| manifest naming a path outside its directory | INVALID |
| missing checkpoint file | INVALID |
| storage short of its manifest's budget | INVALID |

Two diagnosis defects were found and fixed while testing. The reporter opens the analysis file
**and** its checkpoint companion, so an open failure cannot be blamed on the analysis file — the
message now says either may be at fault. And the metadata plan is written once and never rewritten
(OpenMMTools stores it with a fixed dimension), so it records the *original* request: demanding
equality against it reported a correctly **extended** run as incomplete. The manifest and sidecar
are exact sources; metadata is a floor.

## Phase 4 — lifetime statistics

`sampler._n_proposed_matrix` is reset at the start of every mixing call, so the old summary
described 1000 exchange attempts using the statistics of one.
`MultiStateReporter.write_mixing_statistics` runs on every iteration, so
`read_mixing_statistics(slice(None))` is the complete non-cumulative history — across resumes and
extensions alike. Mixing events are counted from that history and cross-checked against the stride,
never inferred as `iteration // stride`.

**A real defect surfaced here.** The stride gate returned early without zeroing the proposal
matrices, which upstream resets at the top of its own `_mix_replicas`. Every skipped iteration
therefore re-wrote the previous event's counts:

```text
before   iterations 24, mixing events 23   (expected 12)
after    iterations 24, mixing events 12   (expected 12)
```

Every lifetime figure built from that history had been inflated by roughly the stride.

Split versus uninterrupted, same budget:

```text
split   last_iteration=120  mixing_events=60  expected=60  schedule_ok=True  range=[0,120]
whole   last_iteration=120  mixing_events=60  expected=60  schedule_ok=True  range=[0,120]
aggregation additive across a split: proposed 1061 == 1061, accepted 89 == 89, events 60 == 60
```

## Phase 5 — round trips

A complete trip is cold → hot → cold; the starting position earns nothing. Implemented as a
three-state machine and tested for every starting state:

```text
[0,1,2,3,2,1,0] -> 1     starts cold
[3,2,1,0]       -> 0     STARTS HOT: reaching cold once is half a trip   (previously 1)
[3,2,0,1,3,2,0] -> 1     starts hot, then a genuine cold->hot->cold
[2,0,1,3,0]     -> 1     starts intermediate
[0,3,0,3,0]     -> 2     two complete trips
```

## Phase 6 — documentation

Ownership statements now match the implementation; the "last attempt" acceptance wording is gone;
the three records (reporter metadata, run-state sidecar, checkpoint, completion manifest) are
distinguished; interrupted resume is documented separately from completed extension; `.verified`
is described as requiring successful NetCDF parsing. The walker-versus-state explanation and the
warning that smoke runs validate neither ladder quality nor convergence are retained.

## `MD-project`

The pin moves to `068ef2f0538a20260e2d40b566ca862e50c6b2c8`, with all 14 system configurations.
`workflow/rest2.smk`'s `verify` rule now **delegates** to `openmm-rest2 --verify-only` and contains
no NetCDF parsing, no openmmtools import and no completion rule of its own. Historical dataset
provenance is untouched: `records/datasets/` is empty, so these are forward-looking generation pins.

Workflow verification, tested against a real finished run:

| case | `.verified` |
|---|---|
| valid run | created |
| truncated analysis NetCDF | refused |
| manifest promising 999 iterations over storage holding 100 | refused |

An earlier attempt at these three appeared to pass for the wrong reason — Snakemake ran from a venv
without openmmtools, so the validator died with `ModuleNotFoundError` and the refusals were
accidental. Re-run with the scientific environment on `PATH`, all three hold for the right reason,
and the workflow now documents that requirement.

## Tests and results

| suite | result |
|---|---|
| MD-templates non-GPU | **543 passed**, 0 failed |
| MD-templates GPU (CUDA) | **90 passed**, 0 failed |
| MD-project | **348 passed**, 0 failed, 0 skipped |
| CPU acceptance: clean / interrupt / resume / extend | all passed |
| Validator matrix (8 cases) | all correct |
| CUDA ALA smoke (6 replicas, 200 ps/replica) | completed, 41–58% lifetime acceptance per adjacent pair |
| MPI smoke, 6 ranks | completed; ranks 0–5 bound to devices 0–5, one rank per device |

57 new correctness tests plus 12 workflow-delegation tests. Most build a REST2 storage directly
through the public reporter writers rather than by running dynamics, so each test creates exactly
the corruption it checks; none needs a GPU.

Two older tests asserted the behaviour this pass corrects — one required `restart.json` to
continue, which *was* the bug — and now assert the corrected behaviour.

Three defects were found by the six-rank MPI smoke and the CPU acceptance run rather than by
reading: the stride/reset accounting error above, the launcher's manifest precondition, and
OpenMMTools opening the storage on rank 0 alone, which made five of six ranks fail after
propagating correctly.

## Commits

```text
MD-templates  feat/openmmtools-rest2
  29420d6  Make OpenMMTools own the exchange decision, and record identity before propagating
  068ef2f  Lifetime statistics, corrected round trips, and documentation that matches

MD-project    dev4-openmmtools-rest2
  2ccee52  Pin MD-templates 068ef2f0: OpenMMTools owns exchange, resume needs no manifest
  c2bc9e7  Delegate REST2 verification to the authoritative validator
  58905c8  Document what the implementation actually does, and journal the corrections
  <this>   Record the final SHAs, which a commit cannot contain for itself
```

Neither branch was merged, force-pushed, or rewritten.

## Not run, and why

- **Long production.** The instruction explicitly excludes it.
- **`ruff`** is not installed in this environment; `compileall` was used instead and every module
  compiles. Recorded as unavailable, not as a pass.
- **Multi-node MPI.** One host only; rank-to-device binding was exercised across 6 local GPUs.

## Remaining limitations

- `swap-neighbors` still needs a version-pinned private-method extension and is refused on any
  OpenMMTools other than 0.26.0. `swap-all`, the default, needs no such pin for its decision path.
- The stride gate remains project-owned scheduling; OpenMMTools offers no stock way to attempt
  exchange every *n*th iteration.
- `md-gen --method REST2` still generates the legacy exchange loop. Out of scope for this pass by
  the instruction's own non-goals, and retiring it needs a migration for contract-managed datasets
  and for AIS.
- Smoke runs are picoseconds and validate neither ladder quality nor convergence.
- Implicit-solvent **ligand** REST2 remains scientifically unvalidated.
