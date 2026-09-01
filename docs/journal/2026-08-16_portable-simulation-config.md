# Committed-generation recovery, and a canonical simulation configuration

**2026-08-16 (fourth pass).** Status: **Milestone 1 complete and gated by a real subprocess crash
test. Milestone 2 substantially complete and driving preparation. Milestone 3 NOT STARTED. CI NOT
ADDED.** 285 tests pass plus one slow crash-recovery test run separately. The incomplete items are
listed at the end and are not claimed.

Request: `claudecode-instructions/20260816_portable-simulation-config.md`, three ordered
milestones, with a hard constraint that the configuration redesign must not begin until the
committed-generation recovery defects are fixed and tested. That ordering was followed.

## Milestone 1 — the committed generation is the sole authority (COMPLETE)

### The defect

The code documented `restart/committed.json` as the only authority for completed state, and then
derived the resume chunk by counting `chunk_*/done.json`. Those two disagree in exactly one window,
and it is a real one: a chunk finishes writing its outputs and its `done.json`, and the process
dies before the commit. Resuming there restored the older committed state **and advanced the chunk
counter past it** — skipping physical propagation, silently, in a run that afterwards looks
complete.

### What changed

`resume_boundary()` reads the commit record and nothing else: committed generation N means the next
chunk is N+1; no committed generation means chunk 0. `done.json`, directory counts, glob counts,
log length and status text can no longer move it.

Everything at or beyond the boundary is an **uncommitted tail** and is moved under `recovery/` —
not deleted, not appended to. It is the product of a process that died, so it may hold the only
copy of something worth seeing, but it is not part of the run's committed history. Everything
*below* the boundary is validated to still exist; a chunk the record calls finished with no outputs
is corruption and stops the run.

REST2 additionally requires every replica to reach the committed boundary, and treats an exchange
log **shorter** than its `attempts_committed` watermark as corruption — previously only a longer
log was handled.

The restored state is checked against the record before any reporter opens: a checkpoint that loads
cleanly but sits at the wrong step would reintroduce the same skipped-physics failure through a
different door. Time is checked against step × timestep with a documented 1e-6 ps tolerance.

Durability: checkpoints are fsynced before the rename and the containing directory is fsynced
after, so a commit record cannot point at a checkpoint a power loss truncates.

Three invocation-provenance defects, all fixed: `stdout.log`/`stderr.log` were opened for **write**
and truncated every earlier invocation's record — now append with a visible banner;
`run_manifest.json` was rewritten on every invocation, erasing the original start time — now
written once, with later invocations in append-only history; and `production.md.seed` /
`production.remd.seed` joined the continuity contract, since a changed production seed gives a
different trajectory from the same state.

### The gate

All ten required interruption scenarios are tested, plus missing committed artifacts, wrong-step
restarts and inconsistent time. Scenario 4 — outputs and `done.json` present, generation
uncommitted — **fails under the previous behaviour**, which is what makes it a regression test
rather than a description.

And the one that decides whether "crash-safe" may be said at all: a **real subprocess**, SIGKILLed
mid-run after a generation committed, then resumed.

```console
$ python -m pytest tests/test_crash_recovery.py -q -m slow
1 passed in 106.93s
```

It asserts recovery continues from the last committed chunk with contiguous attempt indices,
non-decreasing steps, no duplicated attempt, and a contiguous chunk sequence.

## Milestone 2 — canonical configuration (SUBSTANTIALLY COMPLETE)

### Architecture

Four sections with **independent schema versions** — `SystemSpec`, `BuildSpec`, `ProtocolSpec`,
`ExecutionSpec` — each with its own canonical projection and hash:

| projection | consequence of a change |
|---|---|
| system/build | a new bundle |
| protocol (excluding `n_chunks`) | a new run |
| execution | recorded, never compared |

`n_chunks` is excluded from the continuity projection deliberately: extension must not look like a
different calculation, or resuming becomes impossible.

Pydantic v2, `extra="forbid"` at every level, discriminated method models so `md` rejects
`exchange_interval` and `rest2` rejects `scale_factor`.

### Canonicalisation and hashing

Sorted keys, tight separators, quantities as an explicit `{value, unit}` pair. Pydantic's
`model_dump` flattens a NamedTuple to a bare list, which would have made the canonical form
`[0.004,"ps","time","4 fs"]` — ordering-dependent and unreadable as a quantity — so the model is
walked instead and quantities survive to canonicalisation.

### Units

A fixed-table parser, **no `eval`**. Quantities normalise into OpenMM's MD unit system before
hashing. A unitless number is refused where a unit is required, because `timestep: 2` meant fs in
the old manifests and ps in OpenMM's convention.

### Profiles and precedence

Five packaged versioned profiles carrying the **current defaults verbatim** — nothing was retuned.
Precedence is profile → document → `--set`, and every resolved leaf records its source layer,
including subtrees the profile never defined (a gap the tests caught).

Default selection requires an explicit `is_default` flag. Without it, "first match in sorted order"
silently chose `cpu-smoke-v1` — picoseconds of deliberately unvalidated settings — for any ligand
REST2 document, which the migration of the shipped RGD manifest walked straight into.

### One engine, not two

`md-openmm prepare --config DOCUMENT` builds a bundle from a canonical document through the **same
builder** the manifests use; `bundle.prepare` accepts a pre-resolved configuration. The manifest
pair is the legacy front end, migrated into this model by `config migrate`, carrying no semantics
of its own.

Two gaps the first real canonical run exposed: the model could not express the `simple`
equilibration protocol at all, so the smoke profile inherited a 1000 ps equilibration and the first
prepare ran ten minutes before being killed; and canonical SMILES had to be derived when absent.
Both fixed rather than worked around.

### Persistent-format changes

* new: `restart/committed.json` gains `attempts_committed`, `exchange_phase`, `walker_by_replica`,
  `rng_state`, `steps` (versioned, refused if unknown);
* new: `run_state.json` invocation history, append-only;
* new: `recovery/` for quarantined uncommitted tails;
* new: `canonical_configuration.json` inside a bundle prepared from a canonical document;
* unchanged: bundle layout, run layout, manifest schema versions (system 1, experiment 2).

Migration: `config migrate` converts the shipped manifests and reports every semantic change.
A run directory from the immediately preceding refactor has no `run_state.json` and is refused for
resume with that stated, never silently treated as continuable.

## Validation actually run

```console
$ python -m pytest tests/ -q -m "not slow"        285 passed, 3 deselected
$ python -m pytest tests/test_crash_recovery.py -q -m slow    1 passed (106.93 s)
$ python -m compileall -q src scripts             clean
$ python -m build --wheel                         md_tools-0.1.0-py3-none-any.whl
$ unzip -l <wheel>                                5 profiles, 4 systems, 3 experiments present
```

From a **clean cloned environment**, wheel-installed, in `/tmp/outside` with no checkout:

```console
$ md-openmm --version                             md-tools 0.1.0
$ md-openmm config list-profiles                  all five profiles
$ md-openmm config init/validate/resolve          YAML and JSON hashes identical
$ md-openmm smoke --platform CPU                  rc=0, completed, 4/4 rounds
```

Canonical document → bundle → run, on CPU:

```console
$ md-openmm prepare --config run.yaml --out-root ./runs --platform CPU
  bundle ... profile cpu-smoke-v1  system_build c01d4e09da83        (57 s)
$ md-openmm rest2 --bundle <bundle> --out-root ./runs --run-name canonrun --platform CPU
  rc=0  status: completed  exchange rounds 4/4
```

Re-preparing the same document reproduces the same `system_build` hash `81fec93e7231`.

## NOT DONE — do not read this as complete

**Milestone 3, the transferable bundle contract: not started.** No `bundle inspect` command, no
`original_inputs/`, no `checksums.json`, no `forcefield_provenance.json`, no relocation test, no
absolute-path audit, no topology-atom vs OpenMM-particle count separation. The bundle format is
unchanged apart from the added `canonical_configuration.json`. **Portability across relocation is
therefore NOT demonstrated in this pass**, and must not be claimed.

**CI: not added.** No GitHub Actions workflow exists. Every result above is local evidence, which
the repository's own guidance says is not a substitute for configured CI.

**Milestone 2 remainder:** `md` and `rest2` do not yet accept `--config` (only `prepare` does), so
a run still reaches the canonical model through the bundle rather than directly. The `pdb` route is
not supported by `prepare --config`. There is no `changelog`, and supported Python/OpenMM versions
are not yet documented in a released form.

**Deferred by instruction:** the Amber-style namelist adapter. The boundary is defined — register a
loader, return data shaped like the canonical model, reject every unmappable key, add no defaults
and no second validation — and `register_loader` exists for it. No parser was written, as required.

## Limitations that remain true

* State fallback preserves positions, velocities, box, time and statistics but **not** the
  stochastic integrator's stream; it is announced and recorded, never silent.
* The crash test kills one process at one moment. It is real evidence, not exhaustive coverage of
  every interleaving.
* Nothing here is scientific validation. No ladder is validated, the 2 fs / 4 fs equivalence gate
  is still open, and every run above is picoseconds.
