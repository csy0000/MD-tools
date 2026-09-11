# Appending to v2 files written before the optional seed field

**Date** 2026-08-31
**Instruction** `docs/claudecode-instructions/20260831_rest2-rrest2-edge-contracts.md` (follow-up)
**Branches** `MD-templates: fix/openmm-md-rest2-v2-append-compatibility` · `MD-project: dev8-openmm-md-rest2-v2-append-compatibility`

## SHAs

| repository | starting SHA | final SHA |
|---|---|---|
| MD-templates | `b60184f63adc5f745834f1cfcabf06a8b2c942d0` | `14faa9213bbd926785f39a26d2522bec144a92ed` |
| MD-project | `c33d0b76d770955c4c59b6e0214559d396082ad6` | the final commit of this branch |

Both starting heads were fetched and verified before anything changed. Nothing was rebased, no
history was rewritten, and the source branches were not altered.

---

## The defect, reproduced before it was fixed

The previous branch added `reservoir_velocity_seed[exchange]` to
`md-templates-replica-exchange/v2`. New files carried it and readers tolerated its absence, but
`write_exchange()` wrote it unconditionally. So the current runtime could **read** an older v2 file
and then fail the moment it appended a row to one:

```
legacy has the optional variable? False
legacy READS fine, seeds -> [-1, -1, -1]
resume it -- append one more exchange:
  DEFECT REPRODUCED -- KeyError: 'reservoir_velocity_seed'
```

A file written by an earlier build of the same schema was unresumable and unextendable.

## Why v2 was kept

v2's **meaning** did not change; it gained a record. A v3 bump would have been the heavier answer
to a strictly additive change, and would have made every existing v2 file unreadable by a runtime
that can in fact read it. Inspection showed an in-place extension can be implemented safely — the
`exchange` dimension is unlimited, so a variable created later can be sized and filled for the rows
that already exist — so the schema version is unchanged.

`-1` is the field's **meaning** for those rows, not a placeholder. Files that predate the field
also predate working Maxwell refreshes, and a stored-velocity refresh draws no momenta, so those
rows genuinely used no velocity seed.

## What was implemented

Schema knowledge stays in `replica_storage.py`. `OPTIONAL_EXCHANGE_FIELDS` declares the field once
and `create()` drives off it, so a new file and a migrated one cannot drift apart.

- `inspect_optional_exchange_fields()` — **read-only**; reports each field as present and usable,
  absent, or malformed.
- `ensure_optional_exchange_fields()` — rank 0, append mode; creates what is missing, initialises
  existing rows, syncs, and returns what it did.

`write_exchange()` no longer papers over absence: reaching a row write without the field means the
continuation path skipped the migration, and that is a schema error worth naming.

The driver decides *when*, and never creates a NetCDF variable itself. `_continue` is split:

```
PHASE 1 -- read only
  validate_replica_output(...)            every authoritative file, read-only
  open mode="r": identity, counters, optional-field inspection; close
  identity comparison
  checkpoint read
  counters, schedule, --resume/--extend request
PHASE 2 -- write
  open mode="a" on rank 0 only
  ensure_optional_exchange_fields()       once, before any step is integrated
  rewind uncommitted rows
  broadcast the decision
  propagate
```

The ordering is the safety property: **a file is never modified merely because it could be
opened.** An identity mismatch, a corrupt or truncated file, a bad completion marker, a checkpoint
or schedule disagreement, an invalid continuation request, an unsupported schema, or a malformed
existing field all refuse in phase 1 and leave every byte unchanged. An existing field is validated
and never deleted or replaced — wrong dimensions or a non-integer type is a refusal, because
silently redefining a variable whose provenance is unknown would destroy a record.

Migration is never lazy. It happens before propagation, so a schema error surfaces before new
dynamics exist.

Provenance is recorded **twice**: in the run-state sidecar while the continuation runs, and in the
completion manifest as `storage_migration`. The sidecar is rewritten on completion, so without the
manifest copy a reader of the finished run would have no record that the file predated the field —
which is exactly what explains a leading run of `-1` in a seed history. Both carry
`previous_committed_exchanges`.

---

## Environment

| | |
|---|---|
| conda env | `$DATA_ROOT/software/md-stack/envs/openmm-rest2` |
| python / openmm | 3.12.14 / 8.6.0 |
| netCDF4 / numpy / mdtraj | 1.7.4 / 2.4.6 / 1.11.1 |
| mpi4py | 4.1.2 (OpenMPI) |
| snakemake | 9.26.1 (separate venv) |
| GPUs | 9 visible: device 0 RTX A5000, devices 1–8 RTX 3080; driver 580.173.02 |

**Linting, honestly.** The repository still defines no lint or format configuration, and ruff,
flake8 and black are not installed in the project environment. `ruff 0.16.5` was used from a
throwaway venv as an external check only, and no reformatting was applied. On the files this branch
touches: **19 findings before, 19 after — none added, none removed**; **0** added lines exceed 99
characters. The new test files trigger only `I001` and `RUF100`, the same rules the repository's
existing test files trigger under ruff's defaults.

---

## Test suites

| suite | result |
|---|---|
| MD-templates `tests/` | **672 passed**, 0 failed, 0 skipped (6m44s) |
| MD-project `tests/` | **360 passed**, 0 failed, 0 skipped |

`tests/test_v2_append_compatibility.py` is new: 19 tests, **16 of which fail on `b60184f6`**, the
head this branch starts from. Legacy files are built by copying a real one and dropping the
variable, so everything except the missing field is genuinely what this runtime produces.

The MD-project suite skipped 2 before the MD-templates branch was pushed: `test_component_pin.py`
verifies that the remote actually carries the requested ref, which it could not until the push.
They were re-run afterwards and pass; the 360 above is that final run, with no check left
unexercised.

MD-project adds `tests/test_v2_append_capability.py`: the pinned runtime is still v2, declares its
optional fields, exposes both inspection and migration, and can actually append to a file built the
way an earlier build wrote one.

---

## Migration checks

### Unit — synthetic legacy files

Read; `--verify-only` byte-preserving; append refused without migration; migration adds the field
and initialises rows; every other record logically unchanged; stored rows record `-1` and a Maxwell
refresh records its exact seed; an existing field is not recreated or reset; wrong dimensions and
an incompatible type are refused with the file unchanged; a read-only handle is refused; new files
create the field directly; the schema version is unchanged.

### Real — a run produced by the preceding branch

`ALA_imp_PRESEED` and `ALA_imp_PRESEED_LONG` were generated **and run** with MD-templates checked
out at `6cb693f748a2b63e5a0a49a9500ae295433fc2e0`, the revision before the seed field existed, via
a temporary git worktree (since removed). Both produced
`schema: md-templates-replica-exchange/v2` with `reservoir_velocity_seed present? False`.

`--verify-only` on the real legacy file:

```
sha before: a6ec922f192ba7d5a3b79d34    mtime before: 07:46:36.778149712
verify exit=0
sha after:  a6ec922f192ba7d5a3b79d34    mtime after:  07:46:36.778149712
field added as a side effect? False
```

`--resume` on the interrupted legacy run (interrupted at step 43000 of 150000):

```
# storage migrated : added ['reservoir_velocity_seed'] to this v2 file and set 43 existing
                     row(s) to {'reservoir_velocity_seed': -1}; 43 exchange(s) were committed
                     before this continuation
# exchanges        : 150
run_status: completed
```

Then two consecutive extensions → 159 exchanges, 32 whole frames, 318 solute frames,
`--verify-only` VALID, seed history 159 rows **all `-1`** — the correct record for a stored-mode
run — and the second continuation correctly reported `fields_added: []`.

The same sequence on the shorter legacy run: 6 → 9 → 11 exchanges, all `-1`.

---

## CPU, CUDA and MPI evidence, kept separate

| check | platform | result |
|---|---|---|
| unit migration tests | none (NetCDF only) | 19 passed |
| legacy generate + run | CUDA | completed, no seed field |
| legacy `--verify-only` | CUDA | VALID, byte-identical |
| legacy `--resume` + 2 extensions | CUDA | migrated once, VALID |
| legacy `--extend` under 6 ranks | CUDA | migrated once; the migration line appears in rank 0's report only, and in none of `rank01`–`rank05` |
| serial vs 6-rank continuation seed history | **CPU** | **identical** |

The serial/6-rank seed comparison was run on the deterministic **CPU** platform, where arithmetic
is reproducible:

```
serial seeds: [-1, 1735653950, -1, 1406449197, -1, 1259816624, -1]
6-rank seeds: [-1, 1735653950, -1, 1406449197, -1, 1259816624, -1]
IDENTICAL seed history: True
```

This is a storage and ordering change; it does not alter dynamics. No new scientific smoke matrix
was run for it, and none of the above is convergence evidence.

---

## Workflow and pin

`snakemake -s workflow/rest2.smk --configfile workflow/rest2.config.yaml -n`:

```
generate 8   equilibrate 8   run_source_stage 3   run_stage 8   verify 8   replica_all 1   total 36
```

Acyclic and dry-run capable. The workflow passes **no** continuation flag anywhere — conventional
cMD rules included.

`components.yaml` requests `fix/openmm-md-rest2-v2-append-compatibility`; `components.lock.yaml`
pins `14faa9213bbd926785f39a26d2522bec144a92ed`; every `configs/systems/*.sys.config.yaml` carries
the same full 40-character SHA; `tests/devtest/test_component_pin.py` holds it as an independent
second copy.

---

## Limitations

- **This is a storage-compatibility change, not a scientific one.** It alters no dynamics, no
  acceptance rule and no default. Nothing here is evidence about sampling.
- **Only one optional field exists.** The mechanism generalises, but it has been exercised for
  `reservoir_velocity_seed` alone.
- **Migration is one-way.** A migrated file cannot be returned to its pre-field form, and this
  runtime provides no tool to do so. `--verify-only` remains the read-only way to inspect one.
- **The `-1` in migrated rows is a claim about what those runs did**, justified by the fact that
  they predate working Maxwell refreshes. It is correct for every file this repository produced; a
  v2 file from some other source that used Maxwell without recording seeds would be
  indistinguishable from a stored-mode run, and nothing can recover those seeds.
- **A pre-existing field with an incompatible definition is refused, not repaired.** Such a file
  cannot be continued by this runtime at all.
- **No linter is configured or installed in the project environment**; the ruff figure is an
  external check with a delta of zero, not a project standard being met.
- Heterogeneous-GPU MPI runs remain non-bitwise-comparable to serial ones; the CPU comparison above
  is what establishes ordering correctness.
