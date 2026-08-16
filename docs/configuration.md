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
protocol_sha256       change -> start a new run (n_chunks excluded: extension is not a change)
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
    n_chunks: 10
    chunk: 100 ps
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
    n_chunks: 2
    chunk: 1 ns
    exchange_interval: 10 ps
    relaxation: 10 ps
    scale_factors: [1.0, 0.79, 0.6, 0.44, 0.31, 0.25]
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
