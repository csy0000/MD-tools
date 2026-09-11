# The migration transaction: what one `sync()` did not buy

**Date** 2026-08-31
**Branches** `MD-templates: fix/openmm-md-rest2-migration-transaction` · `MD-project: dev10-openmm-md-rest2-migration-transaction`

## SHAs

| repository | starting SHA | final SHA |
|---|---|---|
| MD-templates | `22433ca01cce387bf5b275a0cea2ffbc0d4ed048` | `15395bdb21a0afa4c9b32e88a3e6e7a2b0b70564` |
| MD-project | `b6a0cd1294753f8d6fbcddbb578ec4d654f3f9ba` | the final commit of this branch |

Both starting heads were fetched and verified before editing. Nothing was rebased, no history was
rewritten, and the source branches were not altered.

---

## The crash window, measured

The previous implementation performed `createVariable` → backfill → attribute write → one
`sync()`, and the documentation called that atomic. It is not, and the claim is withdrawn.

Hard-killing a subprocess with `os._exit` at each step, on this build, against a real legacy file:

| killed at | file readable | variable | history attr | values on disk |
|---|---|---|---|---|
| before anything | yes | absent | absent | — |
| after `createVariable` | yes | absent | absent | — |
| **after the backfill call** | yes | **present** | **absent** | **-9223372036854775806 × 4** |
| after the attribute write | yes | present | absent | -9223372036854775806 × 4 |
| after `sync()` | yes | present | present | -1 × 4 |

The third row is the defect, and it is worse than losing provenance. The variable is on disk
holding **HDF5 fill values**, nothing in the file records that a migration was begun, and the
runtime's own reader returned those fill values *as velocity seeds*:

```
reservoir_velocity_seeds() -> [-9223372036854775806, -9223372036854775806, ...]
migration_history()        -> []
inspect()                  -> {'reservoir_velocity_seed': {'state': 'present', 'problem': None}}
```

Silently wrong data, undetected by the validation of the day.

---

## The transaction state machine

NetCDF offers no transaction, and this does not pretend otherwise. What it is, is **restartable and
idempotent**:

```
read-only validation
  → write intent, sync            from here a crash is DETECTABLE
    → create missing fields       idempotent; the intent says which are ours
      → backfill recorded rows, sync
        → append event by its stable id, sync
          → clear intent, sync
            → propagation allowed
```

`storage_migration_pending_json` carries the schema, a stable `transaction_id`, the fields, their
definitions, the row count at intent time, the backfill value per field, the previous committed
exchange count and a timestamp.

### Reconciliation, per crash point

| killed | on disk | restart does |
|---|---|---|
| before the intent | nothing | begins a fresh migration |
| `after_intent` | marker only | creates, backfills, commits, clears |
| `after_create` | marker; field may exist | creates what is missing, backfills, commits, clears |
| `after_backfill` | marker; values correct | commits, clears |
| `after_commit` | marker; event recorded | clears **only**; does not append twice |

The event id is fixed **in the intent**, so a restart commits the event this attempt would have
committed. That is what makes `after_commit` idempotent.

The intent decides which fields belong to the transaction, not the file's current contents. A crash
after `createVariable` leaves the field present; a restart that trusted the file would see nothing
missing and record no event.

### Multi-field

Exercised with a synthetic second optional field: killed after the first is created, the restart
finds a mixture of present and absent targets and finishes both, backfilling each to its own
recorded value.

---

## Verify-only, and the deadlock it nearly caused

`--verify-only` reports an incomplete transaction and returns a failed validation naming
`--resume`/`--extend`. It creates, backfills, commits and clears nothing.

An earlier draft of this branch made a **continuation** refuse for the same reason. That deadlocked
the file: only a resume can reconcile a pending migration, and the check blocked the resume. The
real end-to-end run below caught it. `reconcilable=True` now marks the continuation's own
pre-flight, where a pending marker is a state to reconcile rather than a reason to stop.

---

## Run-state preservation

`write_run_state()` **replaces** the sidecar, so a record written without the history erases the
only note that a legacy file was migrated — from the paths that run when something went wrong.

| path | before | now |
|---|---|---|
| broad exception handler | no history at all | in-memory state, else read-only recovery |
| `_finish`, `completed < expected` | no history | carries it |
| `_record_interruption` | carried it | unchanged |
| fresh `initialized` / `running` | key absent | explicit `[]` |
| `completed` | carried it | unchanged |

Recovery cannot mask the original exception: it swallows its own failures and returns `None`, which
means "could not tell", not "there is none". On `None` the handler leaves the existing run state
untouched rather than replacing a truthful record with an empty claim.

---

## Environment and hardware

| | |
|---|---|
| conda env | `$DATA_ROOT/software/md-stack/envs/openmm-rest2` |
| python / openmm | 3.12.14 / 8.6.0 |
| netCDF4 / numpy | 1.7.4 / 2.4.6 |
| mpi4py | 4.1.2 (OpenMPI) |
| GPUs | **9 available** (device 0 RTX A5000, 1–8 RTX 3080), driver 580.173.02 |
| snakemake | 9.26.1 (separate venv) |

CUDA and MPI were both available and both exercised. The transaction sequence was run on **CPU**,
the deterministic platform, because this change concerns records rather than dynamics.

---

## Tests

| suite | result |
|---|---|
| MD-templates `tests/` | **721 passed**, 0 failed, 0 skipped (6m44s) |
| MD-project `tests/` | **369 passed**, 0 failed, 0 skipped |

The MD-project suite skipped 2 before the MD-templates branch was pushed: `test_component_pin.py`
verifies the remote carries the requested ref, which it could not until the push. They were re-run
afterwards and pass; the 369 above is that final run.

`tests/test_migration_transaction.py` is new: **30 tests, 27 of which fail on `22433ca0`**. The
fault-injection tests hard-kill real subprocesses with `os._exit` at all four points; each asserts
the file stays readable, the marker is present, committed rows are unchanged, and reconciliation
yields exactly one event with the transaction id preserved.

The fault seam is an environment variable read only inside the transaction, inert unless set, not a
CLI option and not documented for users. A test asserts it is unset in normal execution.

---

## Real continuation, CPU

`TXN_CPU`: ALA implicit REST2, generated **and run** with MD-templates at
`6cb693f748a2b63e5a0a49a9500ae295433fc2e0` — before the optional field existed — via a temporary
worktree, since removed.

```
legacy production            → interrupted at step 5000 of 30000; 5 rows, no field, no marker
--resume, HARD-KILLED        → exit 97; marker present, 0 events, 5 rows, committed data intact
--verify-only                → exit 1, "incomplete migration transaction ... run --resume"
                               NetCDF + solute + checkpoint + run state ALL BYTE-IDENTICAL
--resume                     → "storage migrated: added ['reservoir_velocity_seed'] ... 5 rows"
                               reconciled BEFORE propagation; interrupted again at step 10000
                               marker cleared, 1 event, interrupted run state carries 1 event
--resume                     → completed, 30 rows; run state 1, manifest 1
--extend 4, --extend 3       → 37 rows; still exactly 1 event everywhere
mpiexec -n 6 --extend 2      → 39 rows; 1 event; same id; migration line in rank 0 only
--verify-only                → exit 0; ALL FILES BYTE-IDENTICAL
```

Final: the event appears **once** in the NetCDF, the run state and the manifest, all with
`event_id ced2c1135b3e…`, `rows_initialised: 5` — the rows the legacy file actually held.

**CUDA:** a fresh implicit REST2 completed on GPU and correctly reported no pending marker and no
migration event; a six-rank CUDA extension also completed.

---

## Lint

No lint or format configuration exists in the repository and ruff/flake8/black are not installed in
the project environment; `ruff 0.16.5` was used externally, with no reformatting applied. On the
changed files: **29 findings before, 30 after**. The one addition is `UP017` on
`datetime.datetime.now(datetime.timezone.utc)` — another instance of a construct this file already
uses repeatedly. **0** added lines exceed 99 characters.

---

## Pin

`components.yaml` requests `fix/openmm-md-rest2-migration-transaction`; `components.lock.yaml` pins
`15395bdb21a0afa4c9b32e88a3e6e7a2b0b70564`; every `configs/systems/*.sys.config.yaml` carries the
same full 40-character SHA; `tests/devtest/test_component_pin.py` holds it as an independent copy.
Workflow dry run: acyclic, 36 jobs, no continuation flags anywhere.

---

## Limitations

- **A file left by the pre-transaction implementation cannot be repaired.** Field present, fill
  values, no marker, no history: the event is unrecoverable and nothing here invents one. Such a
  file is refused with that explanation, and its fill values are refused rather than read as seeds.
  Any run continued through the previous build in that window is affected.
- **The protocol is restartable, not atomic.** A torn write inside HDF5 itself, or a filesystem
  that reorders or loses a synced write, is outside what this can guarantee. `sync()` is trusted to
  make what it commits durable; that trust is an assumption about the storage stack, not a proof.
- **The intent is a NetCDF attribute in the file being migrated.** If that file becomes unreadable,
  the intent is lost with it. No external journal is kept.
- **Reconciliation rewrites the recorded row range unconditionally.** That is what repairs fill
  values, and it means a hand-edited value inside that range would be overwritten with the backfill
  value. The range is fixed at intent time and no propagation happens before reconciliation, so no
  legitimately written value can be inside it.
- **Two optional fields were exercised, one of them synthetic.** Ordering and partial recovery
  across more than two fields is untested.
- **This is a storage and records change.** It alters no dynamics, no acceptance rule, no reservoir
  semantics and no default. None of the evidence here is evidence about sampling.
- **No linter is configured or installed**; the ruff figure is an external check, not a project
  standard being met.
