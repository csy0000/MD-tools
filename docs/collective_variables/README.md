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

`collective_variables.interval_steps` is **independent** of `solute_printout` and
`system_printout`, and may be finer than either. That is the whole reason for a separate series: a
torsion is cheap to evaluate and a frame is expensive to store.

The schedule must be exact, and is refused rather than rounded:

| protocol | requirement |
|---|---|
| cMD | divides each dynamics stage's step count |
| REST2 / rREST2 | divides `rest2.exchange_interval_steps` |
| AIS | a multiple of `ais.parameter_update_interval_steps` **and** divides `ais.switching_steps` |

Step 0 and the final step appear exactly once each. Minimisation produces no series — its
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
| cMD | `<stage>.cv.csv` | `step,time_ps,trajectory_frame_index` |
| REST2 / rREST2 | `remd<N>.cv.csv` | `step,time_ps,exchange_attempt,state_index,tau,walker_index,exchange_phase,trajectory_frame_index` |
| AIS | `path_NNNN/cv.csv` | `path_index,source_frame_index,protocol_step,time_ps,tau,observation_index,coordinate_frame_index` |
| AIS | `AIS_cv.csv` | the aggregate, same columns |

The CV names follow, in definition order.

### States, not walkers

`remd2.cv.csv` holds whatever configuration **occupied state 2**, whichever walker supplied each
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
the output inventories, the checkpoint stream counts and the completion manifests. On resume the
series is truncated to the count the selected checkpoint generation committed before appending, so
a continuation produces neither a duplicate row nor a gap; a series *shorter* than the checkpoint
claims is refused rather than having the missing rows invented.

**CV output failure is simulation failure.** Reporting is never silently disabled.

## Cost

Evaluation count and wall time are recorded under their own `cv_*` names, deliberately apart from
any energy-evaluation counter. A position-only torsion is not an energy evaluation, and folding it
in would corrupt the one number that says how expensive the Hamiltonian is — the number used to
compare protocols and to size a machine allocation.
