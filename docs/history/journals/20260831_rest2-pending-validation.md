# The pending record: one validator, and authenticating it before it decides anything

**Date** 2026-08-31
**Branches** `MD-templates: fix/openmm-md-rest2-pending-validation` · `MD-project: dev11-openmm-md-rest2-pending-validation`

## SHAs

| repository | starting SHA | final SHA |
|---|---|---|
| MD-templates | `15395bdb21a0afa4c9b32e88a3e6e7a2b0b70564` | `1b0d866c83762117ce437da15b589c0691170253` |
| MD-project | `79e216b6eefd95575b2151b48937448306623c46` | the final commit of this branch |

Both starting heads were fetched and verified before editing. Nothing was rebased, no history was
rewritten, and the source branches were not altered.

---

## Defect 1 — the driver and storage disagreed about the same question

Storage inspected optional fields **with** the pending record:

```python
pending = reporter.pending_migration()
inspection = reporter.inspect_optional_exchange_fields(pending=pending)
```

The driver's continuation probe called the same method **without** it. A migration hard-killed
after `createVariable` leaves the field physically present holding HDF5 fill values; inspected
without the record that reads as malformed, so the driver refused a file its own storage layer
could reconcile. Recovery came down to whether the unsynced variable had happened to reach disk.

Demonstrated through the real CLI **before** the fix, on a genuine legacy run with a **valid**
pending marker and the field present:

```
./REST2/rest2.sh --resume   → exit 1
replica_storage.StorageError: # this run cannot be continued
```

After the fix, the same fixture:

```
# pending migration : transaction e3e6ca5f0964 for ['reservoir_velocity_seed'] is unfinished
                      and will be reconciled before any step is propagated
# storage migrated  : added ['reservoir_velocity_seed'] ... set 5 existing row(s) to -1
run completed; pending cleared; 1 event; transaction id preserved
```

The probe is now `read_only_probe()` — extracted so the ordering it enforces can be exercised
directly. A test that only searched source text would not notice that ordering moving.

`reconcilable=True` does not weaken anything. It says only that a **valid** transaction may be
finished by a continuation; an invalid record is refused for verification and continuation alike,
and a malformed field with no pending record stays refused in both.

---

## Defect 2 — the record was trusted before it was authenticated

`_check_pending_applies()` checked schema, field names and a row bound. Not the transaction id, not
the stored definitions, not the backfill value — and that value is written over **every** legacy
row. An edited record could therefore rewrite a run's history with an arbitrary number.

`validated_pending_migration()` is read-only and returns a validated record or raises. It is the
**one** validator used by `--verify-only`, the driver's probe and append-time reconciliation.

### Schema and rules

```yaml
format:                       md-templates-migration-transaction/v1   # exact
schema:                       md-templates-replica-exchange/v2        # must match the file
fields:                       [..]   non-empty, unique, canonically sorted, all known
definitions:                  {field: {dtype, dimensions}}
backfill:                     {field: int}
rows_at_intent:               int
previous_committed_exchanges: int
created_utc:                  ISO-8601 with an explicit UTC offset
transaction_id:               sha256 over the record without this key
```

- **Keys are exact**, not a minimum: an unexpected key means the record came from something this
  build cannot read.
- **Definitions** compared against `OPTIONAL_EXCHANGE_FIELDS` — dtype after an explicit
  normalisation (`numpy.dtype(x).str`, so `i8` and `int64` agree), dimensions in exact order. A
  conflicting stored definition is refused, never replaced with the current one.
- **Backfill** must be a plain integer (`True` is an `int` in Python and is not one here) equal to
  the registry's `absent_value`. Refused rather than "repaired" — repairing an untrusted record is
  just trusting it.
- **Counts.** Nothing may propagate or rewind between intent and reconciliation, so
  `rows_at_intent` must **equal** the current physical row count. `previous_committed_exchanges`
  must equal the current marker and may be **lower** than the row count: rows can exist past the
  marker when an earlier run was interrupted between a row write and its checkpoint. Tested both
  ways.
- **Identity** recomputed by `migration_transaction_id()` — the one function creation also uses —
  and required to match exactly. Reconciliation never mints a new id.
- **Fields already on disk** checked for dtype and dimensions; their values are treated as
  incomplete only because a valid record authorises a deterministic backfill.

`pending_migration()` is now a pure parser. It previously duplicated a field check, which is
exactly the second contract that drifts.

---

## Tests

| suite | result |
|---|---|
| MD-templates `tests/` | **767 passed**, 0 failed, 0 skipped (6m40s) |
| MD-project `tests/` | **372 passed**, 0 failed, 0 skipped |

The MD-project suite skipped 2 before the MD-templates branch was pushed: `test_component_pin.py`
verifies the remote carries the requested ref, which it could not until the push. They were re-run
afterwards and pass; the 372 above is that final run.

`tests/test_pending_validation.py` is new: **46 tests, 45 of which fail on `15395bdb`**. Thirty are
parameterised rejections covering every rule listed above; each edits a **genuine** record (and
re-signs it where the content rule is what is under test), then asserts a clear error, byte-identical
files, and that no field, event or pending attribute was rewritten.

Recovery is exercised for `after_intent`, `after_create` (field absent), `after_create_durable`
(field present with fill values), `after_backfill` and `after_commit`. `after_create_durable` exists
because the natural `after_create` crash is **not** deterministic — an unsynced variable usually
never reaches disk — so the state that actually blocked recovery must be forced. The seam is an
environment variable read only inside the transaction, inert unless set, not a CLI option; a test
asserts it is unset in normal runs.

Seven existing contract tests were updated where this refactor genuinely moved things (the probe
into its own method, a new fault point, stricter refusal messages) rather than loosened. Two of them
hand-built incomplete pending records that now fail on missing keys first; they build genuine
records and change one thing.

---

## Real end-to-end, CPU

`PV_E2E`: ALA implicit REST2 generated **and run** with MD-templates at
`6cb693f748a2b63e5a0a49a9500ae295433fc2e0`, before the optional field existed, via a temporary
worktree (since removed).

```
legacy production            → interrupted at step 5000 of 20000; 5 rows, no field
--resume, HARD-KILLED at after_create_durable
                             → exit 97; pending present, 0 events, field present with fill values
--verify-only                → exit 1; NetCDF + solute + checkpoint + manifest + run state
                               ALL BYTE-IDENTICAL
--resume                     → "pending migration: transaction 38bf672072bc ... will be reconciled
                               before any step is propagated"; reconciled; interrupted at step 9000
                               pending cleared, 1 event, interrupted run state carries 1
--resume                     → completed, 20 rows
--extend 4, --extend 3       → 27 rows, still exactly 1 event everywhere
--verify-only                → exit 0; ALL FILES BYTE-IDENTICAL
mpiexec -n 6 --extend 2      → 29 rows, 1 event
```

Identity stable across all three records: `txn 38bf672072bc`, `event 309d13c9f946`, counts `1 1 1`,
`rows_initialised: 5` — the rows the legacy file actually held.

**CUDA:** a fresh implicit REST2 completed on GPU and correctly reported no pending marker and no
migration event; a six-rank CUDA extension also completed. 9 GPUs available (device 0 RTX A5000,
1–8 RTX 3080), driver 580.173.02. MPI available (mpi4py 4.1.2, OpenMPI).

---

## Local validation versus CI

Everything above is **local**. No GitHub Actions workflow or status check ran on these commits, and
none is claimed. The repositories define no CI workflow that this session triggered; the results
here are from local pytest runs, local CLI runs and local ruff.

---

## Lint

No lint or format configuration exists in either repository, and ruff/flake8/black are not installed
in the project environment. `ruff 0.16.5` was used from a throwaway venv as an external check; no
reformatting was applied. On the changed files: **44 findings before, 44 after — none added**.
**0** added lines exceed 99 characters.

---

## Pin

`components.yaml` requests `fix/openmm-md-rest2-pending-validation`; `components.lock.yaml` pins
`1b0d866c83762117ce437da15b589c0691170253`; every `configs/systems/*.sys.config.yaml` carries the
same full 40-character SHA; `tests/devtest/test_component_pin.py` holds it as an independent copy.
Workflow dry run: acyclic, 36 jobs, no continuation flags.

---

## Limitations

- **Files left by the pre-transaction implementation still cannot be repaired.** Field present,
  fill values, no marker, no history: the event is unrecoverable, and it is refused rather than
  invented. Unchanged by this branch.
- **The count invariant assumes the current continuation ordering.** `rows_at_intent` must equal
  the physical row count because nothing propagates or rewinds before reconciliation. If that order
  ever changed, this validation would have to change with it — it is a contract, not a derivation.
- **The transaction id authenticates against accident, not against an adversary.** It is a plain
  SHA-256 over the record with no secret, so anyone able to edit the file can re-sign an edited
  record. It catches corruption and careless editing; it is not tamper-proofing.
- **Two optional fields were exercised, one synthetic.** Behaviour across more than two is untested.
- **The driver probe is tested at its own method boundary**, plus the real CLI end-to-end above.
  There is no unit test that constructs a full `ReplicaRun` and calls `_continue` directly.
- **This is a validation and records change.** It alters no dynamics, no acceptance rule, no
  reservoir semantics and no schedule, and none of this is evidence about sampling.
