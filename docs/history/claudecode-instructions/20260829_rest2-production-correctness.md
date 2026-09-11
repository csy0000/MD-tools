# REST2 production-correctness correction

Work across:

```text
git@github.com:csy0000/MD-templates.git
git@github.com:csy0000/MD-project.git
```

Expected starting points:

```text
MD-templates
  branch: feat/openmmtools-rest2
  commit: f2c3c48b06f3f8492f7ce0776a05b75c4fc15757

MD-project
  branch: dev4-openmmtools-rest2
  commit: ca03f478d0dcaa92f5ebbcf881092de363a6e3fb
```

Use authenticated Git access already available in the environment. Read and follow every
applicable `CLAUDE.md`, the completed REST2 instruction, and its execution journal before editing.

This is a focused correctness pass on the existing feature branches. Do not merge into `dev` or
`main`, do not force-push, and do not rewrite history.

## Purpose

The OpenMMTools REST2 implementation is structurally sound, but four production claims currently
exceed what the code guarantees:

1. An interrupted run has authoritative NetCDF storage but no final `restart.json`; resume requires
   that missing manifest and therefore cannot recover the run.
2. The project completion rule checks only that NetCDF paths exist. It does not prove the files are
   readable, internally coherent, or complete.
3. The completion manifest reports only the last mixing call's proposal and acceptance matrices,
   not lifetime neighbouring-pair statistics.
4. Documentation says OpenMMTools owns every exchange decision, but the version-pinned sampler
   overrides `_attempt_swap()` and executes a project-owned Metropolis decision.

There is also a round-trip diagnostic error: a walker that begins at the hottest state can be
credited with a complete cold-to-hot-to-cold round trip after only reaching the cold state once.

Correct these issues without redesigning the concise Python protocol, Amber-like file interface,
REST2 Hamiltonian, `$MD_DATA` layout, or OpenMMTools NetCDF storage model.

## Non-goals

Do not:

- change the REST2 scaling convention;
- change tau values, force fields, HMR, timestep, temperature, pressure, or reporting intervals;
- turn REST2 into temperature REMD;
- implement AIS;
- add production trajectory data to Git;
- migrate historical datasets;
- delete the legacy contract-managed REST2 layout in this correction;
- merge the feature branches.

## Phase 1: establish the real OpenMMTools contract

Inspect the exact installed and upstream OpenMMTools 0.26.0 source before editing, including:

- `ReplicaExchangeSampler.create`, `run`, `extend`, and `from_storage`;
- `MultiStateReporter` metadata, iteration, checkpoint, sampler-state, mapping, energy, and mixing
  statistic readers;
- `swap-all` and `swap-neighbors` behavior;
- the installed NumPy compatibility problem recorded by the current implementation;
- MPI ownership of reporter writes and sampler state.

Record which APIs are public and which are private. Do not infer a NetCDF variable layout when a
reporter method already owns it.

Before choosing the final mixing implementation, test these alternatives:

1. stock OpenMMTools `swap-all`;
2. stock neighbour exchange on a fixed upstream revision or dependency combination;
3. the current locally patched neighbour implementation.

Prefer a stock OpenMMTools exchange implementation. `swap-all` is acceptable if its different
proposal semantics are documented and the scientific configuration explicitly selects it.

If neighbour exchange still requires project code, state the ownership honestly: OpenMMTools owns
propagation, energy evaluation, state storage and restart, while `MD-templates` owns the narrow
neighbour-pair bug fix and Metropolis call. Do not claim that OpenMMTools owns a decision made by
`rest2_openmmtools.py`.

The final code must not override `_attempt_swap()` merely to preserve an inaccurate ownership
statement. Either use the stock decision path or explicitly accept and document project ownership.

## Phase 2: make interrupted-run resume real

The scientific identity required to resume must exist before production propagation begins.

At initial creation:

1. Construct the complete scientific identity before creating the sampler.
2. Store that identity in OpenMMTools reporter metadata using the supported API.
3. Write an atomic run-state sidecar before propagation, with status `initialized` or `running`.
4. Record the analysis and checkpoint NetCDF names, requested budget, exchange stride, reporting
   intervals, generator identity, and input checksums.
5. Only rank 0 may write shared sidecars.

The final `restart.json` remains the completed-run manifest, but it must not be the sole source of
resume identity.

On `--resume`:

1. Require the authoritative analysis NetCDF.
2. Open it through `MultiStateReporter`/`ReplicaExchangeSampler.from_storage`.
3. Read the original scientific identity from reporter metadata.
4. Compare it with the current request before appending.
5. Inspect the last committed analysis and checkpoint iterations.
6. Continue to the original total budget without resetting iteration, mappings, seeds, or
   statistics.
7. Permit resume when `restart.json` is absent because the previous invocation was interrupted.
8. Refuse resume if neither valid reporter metadata nor an independently authenticated run-state
   record can establish the original identity.

On `--extend N`, require a completed run, validate its identity, and add exactly `N` mixing events.
Keep resume-to-existing-budget and extension-beyond-budget distinct.

On exceptions, signals, or controlled interruption, atomically update the run-state sidecar to
`failed` or `interrupted` without claiming completion. Preserve usable OpenMMTools storage.

Add tests that stop a run after a committed iteration, leave no completion manifest, resume it,
and prove continuity of iteration, walker-to-state mapping, storage, and requested budget.

## Phase 3: authoritative NetCDF validation

Implement one validator in `MD-templates`; `MD-project` must call it rather than reimplementing the
NetCDF schema.

Expose it through a documented command such as:

```bash
openmm-rest2 --verify-only \
    -x REST2/rest2.nc \
    --checkpoint REST2/rest2_checkpoint.nc \
    -r REST2/restart.json
```

The exact CLI may differ if a subcommand fits the current parser better, but every documented
command must exist.

Validation must open and read the files, not merely call `Path.is_file()`.

Verify at minimum:

- analysis NetCDF opens successfully through `MultiStateReporter`;
- checkpoint NetCDF opens successfully;
- recorded OpenMMTools version is supported;
- scientific identity metadata matches the completion manifest;
- last committed iteration equals the completed budget;
- the expected checkpoint iteration is readable;
- sampler states at the last checkpoint are readable;
- thermodynamic-state mapping has the expected shape;
- every mapping row is a permutation of all state indices;
- reduced potentials at the last completed iteration have the expected shape and contain no
  invalid values;
- mixing statistics exist for every mixing event expected by the configured stride;
- analysis and checkpoint storage names match the manifest;
- a manifest cannot point outside its run directory;
- the completion manifest is internally consistent with reporter metadata.

Reject truncated files, valid NetCDF files missing required variables, a copied manifest from a
different run, a missing last checkpoint, an incomplete exchange budget, or a changed scientific
identity.

Update `workflow/rest2.smk` so the verification rule delegates to this validator. The workflow may
create `.verified` only after the validator succeeds.

Do not let the project workflow import private OpenMMTools objects or parse raw NetCDF variables.

## Phase 4: lifetime mixing statistics

Build lifetime proposal and acceptance statistics from the authoritative per-iteration reporter
history.

For every configured state pair, report:

- total proposals;
- total accepted swaps;
- lifetime acceptance fraction;
- the iteration range covered.

For a neighbour scheme, show every adjacent pair. For `swap-all`, clearly report its actual pair
proposal semantics rather than labelling the result a neighbouring sweep.

Do not infer attempts solely as:

```python
sampler.iteration // exchange_stride
```

Count actual stored mixing events and cross-check them against the configured schedule. Skipped
non-exchange iterations must carry zero proposals and must not be counted as exchange attempts.

Store lifetime statistics in `restart.json` as a derived summary while keeping NetCDF authoritative.
The readable `.out` summary must label them as lifetime statistics, not “last attempt”.

Resume and extension must preserve and extend the same lifetime statistics. Add a split-run versus
uninterrupted-run comparison.

## Phase 5: correct round trips

Define a complete round trip as:

```text
cold state visited
→ hottest state visited later
→ cold state visited later still
```

A walker that starts at the hottest state has not completed a round trip until it first visits the
cold state, subsequently reaches the hottest state, and subsequently returns to cold.

Implement this as a small tested state machine. Cover walkers starting at the cold, hottest, and
intermediate states, repeated endpoint residence, incomplete half-trips, and multiple complete
trips.

## Phase 6: documentation truthfulness

Update `MD-templates` and `MD-project` documentation to match the final implementation exactly.

In particular:

- describe who owns the actual exchange decision after the Phase-1 choice;
- distinguish reporter metadata, running-state sidecar, checkpoint NetCDF and completed manifest;
- document interrupted resume separately from completed extension;
- state that `.verified` requires successful NetCDF parsing and structural checks;
- replace last-attempt acceptance output with lifetime statistics;
- describe the corrected round-trip definition;
- retain the walker-versus-state explanation;
- retain the explicit warning that smoke runs do not validate ladder quality or convergence.

Do not preserve a stronger claim merely because it already appears in several files.

## Required tests

Add focused tests for:

1. scientific identity stored before production;
2. interruption before completion-manifest creation;
3. resume from valid NetCDF without `restart.json`;
4. resume refusal after a scientific-identity change;
5. resume refusal for incompatible OpenMMTools storage;
6. completed-run extension by exactly `N` mixing events;
7. readable valid analysis and checkpoint NetCDF;
8. truncated analysis NetCDF;
9. truncated checkpoint NetCDF;
10. missing required NetCDF variables;
11. invalid final mapping;
12. non-finite reduced potentials;
13. incomplete stored iteration budget;
14. manifest/storage identity mismatch;
15. lifetime proposal and acceptance aggregation;
16. equality of split-run and uninterrupted lifetime statistics;
17. exchange-stride accounting;
18. corrected round trips for all starting states;
19. MPI rank-0 ownership of shared records;
20. the `MD-project` Snakemake rule invoking the authoritative validator;
21. refusal to create `.verified` for every corrupt/incomplete case above.

Use temporary files and tiny systems. Unit and contract tests must not require a GPU.

## Runtime acceptance

Run a short CPU REST2 job and deliberately interrupt it after at least one committed checkpoint.
Resume it without a completion manifest and complete the original budget.

Then extend it by a small number of mixing events and verify:

- exact final iteration;
- exact number of actual mixing events;
- continuous mapping history;
- readable final checkpoint;
- lifetime statistics spanning all invocations;
- a valid atomic completion manifest;
- successful authoritative validation.

If CUDA is available, run the existing ALA CUDA smoke. If MPI and at least two GPUs are available,
run a short multi-rank smoke and confirm device assignment. A skipped GPU/MPI test is not a pass;
record it as unavailable with the reason.

Do not run a new long production simulation.

## Cross-repository order

1. Implement and verify `MD-templates` first.
2. Commit and push `feat/openmmtools-rest2`.
3. Record its new full SHA.
4. Update `MD-project` component intent, lock, examples and active configuration pins to that SHA.
5. Preserve historical dataset provenance.
6. Replace project-side existence checks with delegation to the generated authoritative validator.
7. Run all `MD-project` tests.
8. Commit and push `dev4-openmmtools-rest2`.

## Completion procedure

Before finishing:

1. Run formatting and linting.
2. Run focused REST2 unit and storage tests.
3. Run the complete non-GPU `MD-templates` suite.
4. Run available GPU tests.
5. Run the interruption/resume/extend acceptance test.
6. Run authoritative validation against valid, truncated, incomplete and mismatched storage.
7. Run the complete `MD-project` suite.
8. Validate all changed YAML files.
9. Confirm Snakemake parses and dry-runs.
10. Inspect `git status` in both repositories.
11. Confirm no NetCDF, checkpoint, trajectory, cache, local environment or generated data was
    accidentally committed.

Create logical commits and push only the two existing feature branches. Do not merge.

Create the matching execution journal:

```text
docs/journals/20260829_rest2-production-correctness.md
```

The final report must include:

- starting and final SHAs;
- the chosen exchange-ownership design and why;
- interrupted resume behavior;
- NetCDF validation contract;
- lifetime statistics implementation;
- round-trip correction;
- tests and exact results;
- CPU/GPU/MPI runtime evidence;
- unavailable tests and reasons;
- remaining limitations;
- commits and branches pushed.
