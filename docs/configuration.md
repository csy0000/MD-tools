# Configuration

One canonical model. Every input syntax compiles into it before a System is built or a run
directory is opened, so file format is a front end and never a source of meaning.

## The four sections

| section | answers | changing it means |
|---|---|---|
| `system` | which molecule, by which **declared** route | a new bundle |
| `build` | parameterisation, solvation, nonbonded, constraints, HMR | a new bundle |
| `protocol` | integrator, equilibration, and the method's production plan | a new run |
| `execution` | platform, device, precision, reporting | nothing scientific |

Each carries its own `schema_version`, so one can move without forcing the others. Each has its own
canonical projection and hash:

```
system_build_sha256   change -> re-prepare the bundle
protocol_sha256       change -> start a new run (nothing is excluded: segment COUNT is no
                      longer a configuration field, so there is nothing to exclude)
execution_sha256      recorded in provenance, never compared
```

## Minimal input

Everything not stated comes from a versioned profile:

```yaml
system:
  system_id: my_ligand
  route: smiles              # declared, never inferred from file contents
  smiles: "CCO"
protocol:
  production:
    method: md               # 'md' or 'rest2'; selects which settings are meaningful
      duration_per_segment: 100 ps   # ONE segment; how many to run is the driver's choice
```

```bash
md-openmm config validate  my_run.yaml
md-openmm config resolve   my_run.yaml      # every value, its source, and the hashes
md-openmm prepare --config my_run.yaml --out-root ./runs --platform CPU
```

## Fully explicit input, and YAML/JSON equivalence

Any field may be stated instead of inherited. The same configuration in YAML and in JSON produces
**byte-identical canonical JSON and identical hashes** — verified by test and from the installed
wheel.

```yaml
profile: explicit-rest2-ligand-v1
system: {system_id: my_ligand, route: smiles, smiles: "CCO"}
build:
  solvation: {padding: 1.2 nm, box_shape: dodecahedron, ionic_strength_molar: 0.15}
  nonbonded: {cutoff: 1.0 nm, minimum_image_margin: 0.10 nm}
protocol:
  integrator: {timestep: 4 fs, temperature: 300 K, friction: 1 /ps}
  production:
    method: rest2
    duration_per_segment: 1 ns
    relaxation: 10 ps
    enhanced_region: {type: solute}
    tau_ladder: {minimum: 0.0, maximum: 0.5, count: 6, interpolation: linear}
    exchange: {number_of_exchanges_per_segment: 100}
    omega_exclusion: {enabled: true, definition: peptide_omega}
execution: {platform: CUDA, device: "0", precision: mixed}
```

## Units

Quantities are explicit: `2 fs`, `300 K`, `1 /ps`, `1.0 nm`, `1 bar`, `10 ps`. They normalise into
OpenMM's MD unit system (ps, nm, K, bar, amu, /ps, kJ/mol) before anything is hashed, and the
canonical document records both the normalised value and its unit.

A **unitless number is refused** where a unit is required. `timestep: 2` meant femtoseconds in the
old manifests and picoseconds in OpenMM's own convention; choosing silently is a factor of a
thousand. The parser is a fixed table, not `eval` — a configuration file is not a place to execute
arbitrary code.

## Resolution precedence

```
1. the versioned profile
2. the user's input document
3. --set dotted.path=value on the command line
```

`config resolve` prints every final value **and the layer that supplied it**, so "where did this
number come from" is answered by data.

## Profiles

```bash
md-openmm config list-profiles
```

| profile | route | method |
|---|---|---|
| `explicit-md-ligand-v1` | smiles | md |
| `explicit-md-peptide-v1` | pdb | md |
| `explicit-rest2-ligand-v1` | smiles | rest2 |
| `explicit-rest2-peptide-v1` | pdb | rest2 |
| `cpu-smoke-v1` | smiles | rest2 |

Omitting `profile:` selects the default for the declared route and method; the exact versioned
profile is printed and persisted. **`cpu-smoke-v1` is never selected automatically** — it is
picoseconds of deliberately unvalidated settings, and reaching it must be deliberate.

A profile is immutable. Changing a scientific value requires a new profile ID and version; the
existing one is never edited in place.

## Inspecting and changing configuration

```bash
md-openmm config init --method rest2 --route smiles --output run.yaml
md-openmm config validate run.yaml                    # schema + cross-field, no OpenMM
md-openmm config resolve  run.yaml --format yaml
md-openmm config diff     a.yaml b.yaml               # classified by consequence
md-openmm config explain  run.yaml protocol.integrator.timestep
md-openmm config migrate  SYSTEM --experiment EXPERIMENT --output canonical.json
```

`config diff` classifies every difference as **bundle-defining**, **continuity-defining**,
**extension-only** or **execution-only** — the useful question is whether you must re-prepare,
restart, or simply carry on.

`config migrate` converts the legacy system + experiment manifest pair. It attaches the unit each
old **field name** declared (`timestep_fs` → `4 fs`), never a guessed one, and reports every
semantic change — including values it deliberately does not carry, such as `ladder_status`, which
is a claim about evidence rather than a simulation setting.

## Unknown keys

Rejected at every nesting level, with the complete dotted path. A file that silently ignores a
misspelled key runs a calculation its author did not choose and believes they did.

Method models are discriminated: `md` rejects `exchange_interval`, `rest2` rejects `scale_factor`.

## The legacy front end

`--system`/`--experiment` still work and are the *legacy* path. They carry no semantics of their
own: `config migrate` lifts them into this model, which is the only configuration engine. New work
should use `--config`.

## The future Amber-style adapter

A namelist front end will register in the loader registry, parse the foreign syntax, and return
data shaped like this model. It must **reject every key it cannot map** rather than ignore it, and
must not add defaults, reinterpret values, or grow its own validation. It is an adapter into these
semantics, never a second set of them. Not implemented; the boundary is defined so that it can be
added without touching the model or the runners.


## Segments, repetition, and completed work

Three different things used to be conflated in one input. They are now kept apart:

| | where it lives | example |
|---|---|---|
| length of ONE segment | scientific JSON | `duration_per_segment: 5 ns` |
| how many segments to run | driver script | `NUMBER_OF_SEGMENTS=2` in Bash |
| how many actually committed | run manifest | `committed.json` |

Segment count is **not** a configuration field. When it was one, asking for a longer run changed the
configuration hash, so an extended run looked like a different calculation. `duration_per_segment`
*is* hashed, because it is the restart granularity: silently changing it would put segments of two
different lengths inside one run.

Every duration must convert to a whole number of integrator steps. A duration that does not is
**refused**, not rounded, and the error names the nearest durations that would work.

```
duration_per_segment            5 ns
timestep                        2 fs
steps per segment               2,500,000
number_of_exchanges_per_segment 100
steps per exchange round        25,000       (= 50 ps)
```

If either division is inexact the configuration is rejected: an exchange round landing mid-step
drops or duplicates an attempt across a segment boundary, and the committed watermark stops
agreeing with the exchange history.

## The REST2 ladder is parameterised by tau

The public and persisted ladder parameter is `tau`, in the Amber style. The Hamiltonian is the same
one the package has always applied:

```
one_minus_tau = 1 - tau
s             = (1 - tau)^2      solute-solute terms
sqrt(s)       = 1 - tau          solute-environment terms
                1                environment terms
```

`tau = 0` is the cold, physical replica (`s = 1`), and a ladder must start there — a ladder that
never samples the unscaled Hamiltonian has no replica whose trajectory is the physical ensemble.

`tau` is the only input. `s`, `sqrt(s)` and effective temperatures are derived by one shared
function and recorded as **labelled diagnostics**; they are never accepted back as input, because
two ways to state the same ladder is how a ladder drifts.

A linear `tau` ladder is an evenly spaced `sqrt(s)` ladder, which is the spacing that gives roughly
even exchange acceptance along the chain — `tau` is exactly `1 - sqrt(s)`.

```yaml
tau_ladder: {minimum: 0.0, maximum: 0.5, count: 10, interpolation: linear}
```

Migrating an existing `scale_factors` ladder: `tau = 1 - sqrt(s)` for each rung. A ladder that is
not linear in `tau` is **refused** rather than respaced, because respacing it changes exchange
acceptance and therefore the run.
