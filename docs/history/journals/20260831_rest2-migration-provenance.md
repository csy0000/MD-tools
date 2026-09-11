# Migration provenance: from one overwritten value to a cumulative history

**Date** 2026-08-31
**Branches** `MD-templates: fix/openmm-md-rest2-migration-provenance` · `MD-project: dev9-openmm-md-rest2-migration-provenance`

## SHAs

| repository | starting SHA | final SHA |
|---|---|---|
| MD-templates | `14faa9213bbd926785f39a26d2522bec144a92ed` | `22433ca01cce387bf5b275a0cea2ffbc0d4ed048` |
| MD-project | `fecaa18b25ad5bf5ed2ed683f3d1b09b4564ce5e` | the final commit of this branch |

Both starting heads were fetched and verified before editing. Nothing was rebased, no history was
rewritten, and the source branches were not altered.

---

## Root cause

The v2 append-compatibility work migrated missing optional exchange fields correctly, but recorded
what it had done in a **single** value:

```python
payload["storage_migration"] = migration        # this continuation's record, and only this one
```

Every continuation overwrote it. A resume or extension that needed no schema change produced a
no-op record — `fields_added: []` — and that no-op replaced the real migration event in both the
run state and the completion manifest. A twice-extended legacy run therefore ended up with a
manifest that no longer said its file had ever been a legacy file.

This was visible in the previous branch's own recorded evidence, which shows
`fields_added: []` in the manifest after a second extension.

There was a second, narrower gap. The event was written only **after** the mutation, in the run
state. A process killed between `ensure_optional_exchange_fields()` and that write left storage
changed with nothing anywhere saying so.

---

## The implemented model

### Append-only history

`storage_migration` (singular) is replaced by `storage_migrations` (ordered list). A continuation
**appends**; it never replaces. A continuation with `fields_added: []` contributes nothing, so it
cannot displace the event before it, and a run that never needed a migration carries an empty
history rather than a fabricated one.

### Content-addressed identity

`event_id` is a SHA-256 over the canonical JSON of the event minus the id itself. The same event
copied into a manifest and into a run state collapses to one entry; two genuinely different
migrations stay separate. Order is **first-seen** — the order the migrations happened in — not
sorted, because no timestamp in these records is guaranteed comparable across machines.

### Durability, and where the authority lives

The event is written into the analysis NetCDF as `storage_migrations_json`, **in the same `sync()`
as the mutation it describes**. A process killed immediately after migrating leaves a file that
already says it was migrated, so the next continuation reads the truth rather than inferring it.

The two-phase design is unchanged: everything is validated read-only, and only then is the file
opened for append and changed.

### Every source merged, none trusted alone

Phase 1 collects, read-only:

```
the file's own storage_migrations_json   the durable authority for its own schema
the previous completion manifest         a run migrated by an earlier build recorded it only here
the run state                            an interrupted run recorded it only here
```

and merges them. The complete history is then persisted in the run state while a run is **running,
interrupted and completed**, and in the completion manifest when it finishes.

### Backward compatibility

`migration_history_of()` accepts the legacy singular field. A meaningful singular record becomes a
one-event history; a singular **no-op** is dropped, because it never described a change. Existing
manifests remain resumable, and a legacy singular record and a plural one describing the same run
merge to one event rather than two.

### `--verify-only`

Strictly read-only. It adds no variable, no attribute, no manifest and no run-state update, and
changes no byte of any of them.

---

## Environment and hardware

| | |
|---|---|
| conda env | `$DATA_ROOT/software/md-stack/envs/openmm-rest2` |
| python / openmm | 3.12.14 / 8.6.0 |
| netCDF4 / numpy | 1.7.4 / 2.4.6 |
| mpi4py | 4.1.2 (OpenMPI) |
| OpenMM platforms present | CPU, CUDA, OpenCL, Reference |
| GPUs | **9 available** (device 0 RTX A5000, 1–8 RTX 3080), driver 580.173.02 |
| snakemake | 9.26.1 (separate venv) |

CUDA and MPI were both available and both were exercised. The end-to-end continuation sequence was
run deliberately on **CPU**, because it is the deterministic platform and this change is about
records rather than dynamics.

---

## Tests

| suite | result |
|---|---|
| MD-templates `tests/` | **691 passed**, 0 failed, 0 skipped (6m32s) |
| MD-project `tests/` | **365 passed**, 0 failed, 0 skipped |

`tests/test_migration_provenance.py` is new: **19 tests, 18 of which fail on `14faa921`**, the head
this branch starts from. They cover legacy singular normalisation, no-op rejection, deduplication
across sources, ordering of distinct migrations, deterministic identity, durability in the file,
append-only behaviour under repeated no-ops, unchanged rows, `--verify-only`, honest empty history
for current storage, the driver's merge wiring, interrupted run state, and recovery of a history
after a completion manifest is replaced.

Two tests in `test_v2_append_compatibility.py` were updated where they pinned the previous
contract: migration now adds exactly one attribute (the history), and provenance is plural.

The MD-project suite skipped 2 before the MD-templates branch was pushed: `test_component_pin.py`
verifies that the remote carries the requested ref, which it could not until the push. They were
re-run afterwards and pass; the 365 above is that final run.

MD-project's `tests/test_v2_append_capability.py` gains five manifest-compatibility tests. Nothing
in MD-project read the migration field before this change, so no schema, example or workflow
needed updating; the tests assert the contract the project depends on when it reads a finished run.

---

## End-to-end continuation, CPU

`PROV_CPU`: ALA implicit REST2, generated **and run** with MD-templates checked out at
`6cb693f748a2b63e5a0a49a9500ae295433fc2e0` — the revision before the optional field existed — via a
temporary git worktree, since removed.

```
setup + equilibrate (legacy build)
run production            → interrupted at step 5000 of 30000
                            file: seed field False, history attr False, 5 rows
--resume (new runtime)    → "storage migrated: added ['reservoir_velocity_seed'] ...
                              5 existing row(s) ... 5 exchange(s) committed"
                          → interrupted again at step 9000
                            file history 1 · interrupted run state 1
--resume                  → completed, 30 exchanges
                            file 1 · run state 1 · manifest 1
--extend 4                → file 1 · run state 1 · manifest 1
--extend 3                → file 1 · run state 1 · manifest 1
mpiexec -n 6 --extend 2   → 39 rows; file 1 · manifest 1; same event_id
--verify-only             → NetCDF, manifest, run state and checkpoint ALL byte-identical
```

Final state:

```
exchange rows                 : 37 (39 after the MPI extension)
backfilled seeds              : all -1  (a stored-mode run drew nothing)
migration history             : 1 event
  fields_added                : ['reservoir_velocity_seed']
  rows_initialised            : 5
  previous_committed_exchanges: 5
manifest history              : 1 event, same event_id
no no-op event anywhere       : True
```

The original real migration appears **exactly once**, survived an interruption that happened after
the mutation, survived completion, survived two extensions and an MPI extension, and was never
replaced by a no-op. On the starting head the second extension's no-op would have overwritten it.

**MPI:** the six-rank extension preserved the history and the same `event_id`; the migration line
appears in rank 0's report and in none of `rank01`–`rank05`. Only rank 0 opens writable storage.

---

## Lint

The repository still defines no lint or format configuration, and ruff, flake8 and black are not
installed in the project environment. `ruff 0.16.5` was used from a throwaway venv as an external
check; no reformatting was applied. On the changed files: **23 findings before, 24 after**. The one
addition is `UP017` on `datetime.datetime.now(datetime.timezone.utc)` — a seventh instance of a
construct this file already uses six times. Changing only the new line would make it inconsistent
with the other six, so it was left as house style. **0** added lines exceed 99 characters.

---

## Workflow and pin

```
generate 8   equilibrate 8   run_source_stage 3   run_stage 8   verify 8   replica_all 1   total 36
```

Acyclic and dry-run capable; the workflow passes no continuation flag anywhere.

`components.yaml` requests `fix/openmm-md-rest2-migration-provenance`; `components.lock.yaml` pins
`22433ca01cce387bf5b275a0cea2ffbc0d4ed048`; every `configs/systems/*.sys.config.yaml` carries the
same full 40-character SHA; `tests/devtest/test_component_pin.py` holds it as an independent second
copy.

---

## Limitations

- **Provenance recovered from records, not reconstructed.** A file migrated by the *previous*
  build carries no `storage_migrations_json` attribute. Its event survives only if the manifest or
  run state that recorded it still exists; if both were lost, the migration is unrecoverable and
  the history will read empty. Nothing can infer it after the fact.
- **A pre-fix run that already lost its event stays lost.** This fix prevents the loss going
  forward; it cannot restore a manifest that a no-op already overwrote.
- **`event_id` covers the event's content, including its timestamp.** Two migrations that added
  the same fields to the same file at different times are correctly distinct — but a record
  hand-edited in any field becomes a different event and would not deduplicate against its
  original.
- **Only one optional field exists**, so the history has been exercised with a single real event
  type. Ordering of two genuinely different migrations is covered by unit tests, not by a real run.
- **This is a records change.** It alters no dynamics, no acceptance rule, no reservoir semantics
  and no default, and none of the evidence above is evidence about sampling.
- **No linter is configured or installed** in the project environment; the ruff figure is an
  external check, not a project standard being met.
