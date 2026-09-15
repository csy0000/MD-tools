# Torsion collective variables

**Version 1 supports torsions and nothing else.** No biasing, no PLUMED, no second engine. A
collective variable here is an *observation* of a run, and the whole design follows from that.

## What it does not do

It adds no `Force` to the System.

That is worth stating first because OpenMM would happily compute a torsion for you — a
`CustomTorsionForce` with a zero force constant reports one every step — and reaching for it is
the obvious implementation. It is also wrong, and not for performance reasons. Adding a Force
changes the System: its serialisation, its force-group layout, the checkpoint that records them,
and, through the force groups an integrator asks for, the dynamics themselves. A run with
reporting enabled would no longer be the same experiment as the run without it, which makes every
comparison between the two invalid and makes *"does reporting change the physics?"* a question
nobody can answer from the outputs.

So the entire evaluation is arithmetic over positions the reporting point already had. Nothing in
`md_tools.cv` imports OpenMM, and a test asserts that through the AST rather than by text search —
the modules discuss `CustomTorsionForce` at length explaining why it is the wrong tool, and a text
search finds the explanation and reports the very thing it says is not there.

The suite asserts the invariance directly: identical System serialisation, identical force
inventory, identical force groups and an identical single-point energy with reporting on and off.
Coordinates after dynamics are compared against a *measured* reproducibility floor — two runs of
the same configuration are not bit-identical on the CPU platform, so an exact-equality assertion
there would be testing platform determinism rather than anything about collective variables.

## The definition file

```yaml
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
  - name: psi
    type: torsion
    atoms:
      - {chain: A, residue: "1", atom: N}
      - {chain: A, residue: "1", atom: CA}
      - {chain: A, residue: "1", atom: C}
      - {chain: A, residue: "2", atom: N}
```

Each entry needs a unique CSV-safe `name`, `type: torsion`, and **exactly one** of `atom_indices`
(four distinct, in-range, zero-based indices in bonding order `i-j-k-l`) or `atoms` (four
selectors, each with an explicit `chain`, `residue` and `atom`).

Everything about this schema is strict, and the strictness is the point. A CV series that names
the wrong four atoms is indistinguishable from one that names the right four: same column
heading, same range, same plausible distribution, and nothing downstream catches it. So there is
no first-match on an ambiguous selector, no clamping an out-of-range index, no suffixing a
duplicate name, no ignoring an unknown field, and no accepting a schema version this build does
not read. `residue` is compared as a *string* against the topology's own residue id, because a PDB
residue identifier may carry an insertion code and `int()` would conflate `"52A"` with `52`.

An unsupported `type` is refused **by name**, with the schema version it was refused under, so a
`distance` written against a future schema fails loudly on an old build instead of silently
producing a shorter CSV.

## Values and conventions

| | |
|---|---|
| units | degrees |
| range | `[-180, 180)` — half-open, so exactly 180 reports as −180 and no value has two spellings |
| sign | IUPAC/MDTraj: looking along `j → k`, positive turns the `i` end clockwise onto the `l` end |
| periodicity | triclinic minimum image, applied to the three sequential **bond vectors** |

The minimum-image convention applies to the bonds, not to the four positions. Wrapping positions
independently into the primary cell is the intuitive move and it is wrong: it can place two bonded
atoms in opposite corners, which is exactly the artefact the convention exists to remove. For a
triclinic cell a per-axis `round(v / L)` is not sufficient either — shortening along one axis can
lengthen along another — so the reduction is followed by a neighbour search, and a test brute-forces
the true minimum over a 9×9×9 neighbourhood and additionally proves the result is a genuine lattice
translate of the input.

## Cadence

`collective_variables.interval_steps` is **independent** of `crd_printout_solute` and
`info_printout`, and may be finer than either. That is the whole reason for a separate series: a
torsion is cheap to evaluate and a frame is expensive to store.

The schedule must be exact, and is refused rather than rounded:

| protocol | requirement |
|---|---|
| cMD | divides each dynamics stage's step count |
| REST2 / rREST2 | divides `rest2.exchange_interval_steps` |
| AIS | a multiple of `ais.parameter_update_interval_steps` **and** divides `ais.switching_steps` |

Step 0 and the final step appear exactly once each, in **every** protocol including the REST2
and rREST2 ladders. Minimisation produces no series — its
iterations have no timestep, so a `time_ps` for them would be a fiction, and the intermediate
geometries lie on no physical trajectory.

The AIS rule is two conditions, not one. Dividing `switching_steps` alone would allow an
observation *between* two parameter updates, at a tau the path never held, because tau is
piecewise constant across an update interval. Sitting on the update grid alone would allow a final
partial gap.

An interval that does not divide leaves that partial gap, so the last observation sits at an
irregular spacing from its predecessor — and every downstream time-series analysis
(autocorrelation, block averaging, spectral density) assumes uniform spacing while none of them
can detect the violation.

## Output

UTF-8 CSV, stable column order, **no comment lines** — a `#` line is the specific thing that
breaks the naive readers a CV series is actually read by. The provenance goes in a JSON sidecar
instead, where it can be complete: schema version, original `cv.yaml` digest, resolved atom
indices, units, wrapping, periodic and sign conventions, and the column order. The CSV is
readable without the topology or the definition that produced it.

| protocol | file | leading columns |
|---|---|---|
| cMD | `<stage>.cv.csv` + `<stage>.cv.json` | `step,time_ps,trajectory_frame_index` |
| REST2 / rREST2 | `cv_state<N>.csv` + `cv_state<N>.json` | `step,time_ps,exchange_attempt,state_index,tau,walker_index,exchange_phase,trajectory_frame_index` |
| AIS | `path_NNNN/cv.csv` | `path_index,source_frame_index,protocol_step,time_ps,tau,observation_index,coordinate_frame_index` |
| AIS | `AIS_cv.csv` | the aggregate, same columns |

The CV names follow, in definition order.

### States, not walkers

`cv_state2.csv` holds whatever configuration **occupied state 2**, whichever walker supplied each
one — the same rule the state trajectories follow. A ladder's result is a property of a rung
("the distribution at tau = 0.3") and a walker visits many rungs, so a per-walker series is a
series over a changing Hamiltonian and is not an ensemble average of anything. `walker_index`
records which walker supplied the configuration, which is what makes the exchange history
reconstructible from the CV files alone.

### Pre-exchange, everywhere

A row landing on an exchange boundary describes the configuration the walker actually
**propagated** to that step, taken before any swap is applied. Post-exchange would report, against
that step, a configuration that arrived from another rung and was never integrated there — so a
state's series would contain values from trajectories that never visited it.

The convention is enforced structurally (`cv` precedes `exchange` in `EVENT_ORDER`), written into
every row as `exchange_phase`, and stated in every sidecar, so a reader never has to infer it and
a future post-exchange row would be distinguishable rather than silently mixed in.

Step 0 carries `exchange_attempt = -1`, because no attempt has been made yet, and the identity
state-to-walker mapping in a fresh run.

### What `trajectory_frame_index` means in a ladder

**A row that names frame `k` was evaluated on exactly `whole_state<i>_prod<N>.nc[k]`** -- that
state's own trajectory, named by `state_trajectory_name`. That is the whole meaning
of the column, and it is why the field is often empty at a step where a frame *was* written.

The state trajectory frame is written from the **post-exchange** occupant; the CV row describes
the **pre-exchange** one. For a state whose walker the exchange did not move, those are the same
configuration and the frame is named. For a state that was swapped, they are different
configurations — no frame in that file holds what that row measured — and the field is left
empty. It is a **per-state** decision at the same step: at one exchange, some states name the
frame and others do not.

The field is never `-1`. `-1` is not a frame index, and writing it invites a reader to index from
the end of the file.

### rREST2: the reservoir refresh

A refreshed state's CV row holds the configuration the state **propagated**, not the reservoir
sample that replaced it. The complete walker-indexed configurations are snapshotted before the
exchange, so no later swap or refresh can mutate what a row is computed from.

A refreshed state also names **no** trajectory frame, even though its walker index did not change:
the frame about to be written holds the reservoir sample, and the row holds the propagated
coordinates. Same conclusion as an accepted swap, reached by a second route that the mapping alone
cannot see.

### AIS alignment

`observation_index` and `coordinate_frame_index` are **empty** on rows whose cadence does not
coincide with a saved observation. They are never filled with a nearest neighbour. When they are
present, the values were measured on exactly that saved coordinate — attaching a CV measured at
`x_j` to a different observation's coordinate is the same class of error the AIS two-probe
separation exists to prevent, and it is invisible in the output.

`AIS_cv.csv` is assembled **only from paths whose completion manifest verified**, the same gate the
work and HS tables use. A path that crashed mid-write leaves a plausible-looking `cv.csv` and no
manifest.

## Portability, restart and identity

The `file` path is resolved relative to **the MD configuration file**, not the working directory,
and the definition is then copied content-addressed into the generated directory. A generated tree
is meant to be moved — to a cluster, into an archive beside its results — and an absolute path to a
definition elsewhere survives none of that. It survives it *silently* in the worst case, where the
path exists on the target machine and holds a different file.

The definition and its schedule are bound into the run identity, the continuation fingerprints,
the output inventories, the checkpoint generation and the completion manifests.

### The committed prefix

A checkpoint records more than a row count. A count detects a file that is *shorter* than it
should be; it says nothing about whether the rows it does hold are still the rows that were
committed. Edit a value in place, renumber a step, change an identifier — the count still agrees,
the continuation appends onto them, and the finished series is part measurement and part edit with
nothing marking the boundary.

So the generation stores a **digest of exactly the header plus the committed rows**, in the same
transaction as the Context state. Not of the whole file: after a crash the file is legitimately
*longer* than the checkpoint, because rows are flushed as they are written and the commit happens
afterwards, and hashing the uncommitted tail would make every ordinary crash look like corruption.

Before appending, that prefix is validated — digest, exact columns, finite values, a strictly
increasing grid on the declared interval, and the identifiers the series must carry (state index
and tau for a ladder, path and source-frame index for AIS). All of it **before a byte is
truncated**, because a continuation that has already truncated cannot decide afterwards that it
should have refused. Ladder prefixes are recorded per state rather than as one combined hash: a
combined hash cannot say which file changed, and two states' files being swapped would leave it
unchanged while every series became another's.

A checkpoint with no prefix record refuses with a compatibility message rather than guessing.

### Completion

A ladder's `restart.json` records, for every state: index and tau, CSV and sidecar names, digests
and byte sizes, row count, exact header, first and final step, interval, expected grid, schema
version and definition digest, resolved atom indices and column order, units, wrapping, periodic
convention, exchange phase, and the CV cost. Completion is **refused** if any series fails
verification, and `validate_replica_output` re-reads them afterwards, so a file edited after the
fact is caught by the same rules that let it be written. An extension checks its parent read-only
before anything local exists.

**Walker identity and the ladder permutation.** Every ladder CV row's `walker_index` must be an
integer with `0 <= walker < number_of_states`, and its state index, tau, exchange phase and
observation step must match the owning series and the schedule. Across the set, at each common
observation step the walkers in all state files must form an exact permutation of
`0 .. number_of_states - 1`: an exchange permutes walkers among rungs and never creates, destroys
or duplicates one. Two rungs claiming one walker, or a walker occupying none, is invisible from
inside any single file — every row there is internally consistent, on the right step, at the
right tau, with a walker in range. This runs before completion is committed, whenever a completed
ladder is verified, and before an extension touches its parent.

**Provenance.** A ladder's per-state CV CSV and sidecar are named in three places: the completion
manifest, the outer machine-readable `-log` output inventory, and — being files under the run
root — the registry's `SHA256SUMS`. The `-log` inventory is what `registry.discovery.check_lineage`
indexes by digest to connect one stage's outputs to the next stage's inputs, so a series missing
from it is invisible to every lineage check even while present on disk. Each record carries the
path relative to the run root, the digest and byte size the manifest recorded, the state index and
its tau, and the CV definition digest; CSV and sidecar are separate roles, because the sidecar is
how the CSV is read. The inventory is built from the validated manifest and never from a glob: a
`cv_state*.csv` glob would record a file left behind by an earlier run into the same directory as
this run's provenance.

An AIS path validates its series structurally against the **schedule** rather than the file:
`switching_steps / cv_interval_steps + 1` rows on the grid `0, interval, …, switching_steps`, both
endpoints exactly once, with path and source-frame identity on every row. A file that is
self-consistently wrong cannot satisfy a requirement it did not supply.

A CV-disabled run records `collective_variables: null` explicitly. Omission would be
indistinguishable from a manifest predating the field.

### Cost

Three counters, with exact meanings, because two of them used to share a name:

| field | meaning |
|---|---|
| `cv_observations` | configurations on which the reporter evaluated the **complete** configured CV set — one per reporter call |
| `cv_evaluations` | **scalar** CV values evaluated. One observation of a definition holding `N_cv` torsions adds `N_cv` |
| `wall_seconds` | measured time spent evaluating the CV set, excluding CSV serialisation, hashing and validation |

`cv_evaluations` previously counted reporter *calls* — it incremented by one per observation
regardless of how many torsions the definition held, so a two-torsion run reported half the
scalar work it had done, under a name that says otherwise. The two counters are numerically
identical for a single-CV definition, which is how the misnomer survived a full test suite; every
test that asserts on them now uses at least two torsions, where a call counter and a scalar
counter cannot agree.

Each is recorded in **two scopes**:

```yaml
collective_variable_cost:
  schema_version: 2
  segment:                 # this invocation only
    cv_observations: ...
    cv_evaluations: ...
    wall_seconds: ...
  cumulative:              # the complete logical simulation, across every invocation
    cv_observations: ...
    cv_evaluations: ...
    wall_seconds: ...
```

On a fresh run the two are equal. On a continuation the cumulative counters are restored from the
committed prefix **before any new evaluation**, and only the segment counters reset — so a second
or later interruption neither loses nor double-counts what earlier segments spent. Both halves
stay visible: a single number replacing the pair would make an interrupted run look cheaper than
an identical uninterrupted one. Verifying or skipping already-complete work counts as neither;
re-entering a finished run leaves an empty segment.

The counters are restored as one typed object (`CommittedPrefix`, carrying the row count *and*
the cost) rather than as loosely related integers, so a caller cannot restore the rows while
silently discarding the counters — which is exactly what one did.

**Aggregation.** A ladder reports the **sum over thermodynamic states** and an AIS run the **sum
over completed paths**; both say so in an `aggregation` field and both retain the per-state or
per-path records beside the total, so the sum is auditable rather than a number to be trusted.
Counters are integers; every counter is finite and non-negative, and an AIS path whose cumulative
counters disagree with its row count and its `N_cv` is refused before its completion marker is
committed.

**A resume is the same trajectory, not merely a valid one.** Positions, velocities and box are
the complete *physical* state of a Langevin walker, and they are what every checkpoint stores —
which keeps a checkpoint readable on any device. They are not enough to continue a trajectory:
the integrator's pseudo-random stream has a position within it that coordinates do not carry, so
a walker resumed from coordinates alone draws different noise from that point on. The result is
correct and statistically exact, and it is a *different* trajectory — which means a resumed CV
series cannot be compared value-by-value with an uninterrupted reference.

cMD and AIS always resumed through `loadCheckpoint` and so were always bitwise. The ladder did
not, and every continued REST2/rREST2 run diverged from the resume point onward. Ladder
checkpoints now store each rung's OpenMM context checkpoint **alongside** the coordinates,
together with the platform and precision that produced them. They are an optimisation and never a
requirement: a continuation that finds them, written by the platform it is running on, restores
them and reproduces the reference exactly; one on a different device ignores them and falls back
to coordinates exactly as before. The checkpoint is never made unusable on another machine, and
which path was taken is announced on the run's output and recorded.

This costs checkpoint size. A ladder checkpoint now carries one OpenMM context checkpoint per
rung in addition to the coordinates, so the file grows roughly in proportion to the number of
states — the checkpoint is rewritten whole and replaced atomically, as before, so the cost is
per write rather than cumulative. The coordinates remain the authoritative record and a
continuation never requires the blobs, so a deployment that would rather not pay this is
losing bit-for-bit reproducibility and nothing else.

One caveat belongs with this. OpenMM's **CPU platform** sums its force reductions in
thread-completion order, so it is only reproducible at a fixed thread count: two replicate runs
with the same pinned seed diverge by the first observation with the default pool. That is a
property of the platform, not of any protocol here. Tests that compare series value-by-value pin
`OPENMM_CPU_THREADS=1` and say why; nothing pins a thread count in production.

**Steps are absolute.** OpenMM's `loadCheckpoint` restores the Context's step count, so
`simulation.currentStep` is the single authority after a restore and nothing adds the
already-completed count to it. A resumed run's grid is identical to an uninterrupted one's.

**A ladder checkpoint records the committed CV row count.** A checkpoint that does not — one
written by a build that reported CVs without binding their count into the transaction — refuses
continuation with a compatibility message rather than guessing from the file length. Which rows
are durable is exactly what the count exists to say.

A changed definition refuses continuation: different atom indices, a reordered definition, a
different interval, units, wrapping, sign or periodic convention, exchange-boundary convention,
state index, tau, definition digest, or column set. Each would make the appended rows a different
measurement sharing a column heading with the old ones.

**A rejected continuation preserves the prior run entirely.** Every authoritative CV record a
continuation depends on -- the committed prefix and its cost, each per-state entry, the
checkpoint's aggregate, and for AIS every selected path's completion manifest -- is validated in
a read-only phase before the run creates or replaces anything. That includes `.out`, `.log`, the
run-state record and `resolved.config`: those are the prior run's machine-readable provenance and
completion status, and a rejected attempt must not become the authoritative account of a run it
never started. The refusal is written to stderr and names the record, the scope and the field.
Once the phase passes, ordinary runtime logging resumes unchanged.

An absent cost on a CV-enabled record is not a zero cost. It is corruption or unsupported legacy
data, and it is refused rather than restored as a clean history of no work -- the distinction is
made from the run's own configuration and schema, never from whether a stored value happens to be
truthy. A genuinely CV-disabled run records its documented null field and is unaffected.

**CV output failure is simulation failure.** Reporting is never silently disabled.

## Cost

Evaluation count and wall time are recorded under their own `cv_*` names, deliberately apart from
any energy-evaluation counter. A position-only torsion is not an energy evaluation, and folding it
in would corrupt the one number that says how expensive the Hamiltonian is — the number used to
compare protocols and to size a machine allocation.


## Where the sidecar is, and what it is not

The output sidecar is `<name>.cv.json`, beside the CSV it describes. Three files are deliberately
not conflated:

| file | what it is |
|---|---|
| `cv.yaml` | the **input** definition a person writes |
| `cv.<digest>.yaml` | the content-addressed **copy** in the generated directory, which makes the tree movable |
| `<name>.cv.json` | the **output** sidecar saying how to read the CSV beside it |

Every one of them is named in exactly one place in the code. The inventory once named
`<stage>.cv.yaml` — a file that never existed — so the real sidecar was in no inventory: not
collision-checked, not removed by `--overwrite`, and free to survive a definition change and
describe the new CSV with the old atom mapping.

`md-openmm md-run` writes its own `resolved.config` into `-odir` and carries the definition there
with it, so `-odir` holds everything needed to read its own output and both routes resolve the
definition by the same single rule: *beside `resolved.config`*.
