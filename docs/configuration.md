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

Method models are discriminated: `md` rejects `exchange`, `rest2` rejects `scale_factor`.

### A REST2 segment: what is stated and what is derived

A segment is stated by its **duration** and the **number of exchanges** in it. The interval is a
consequence of those two and is derived, in integer step space:

```
steps_per_segment  = production.duration_per_segment / integrator.timestep
steps_per_exchange = steps_per_segment / exchange.number_of_exchanges_per_segment
exchange_interval  = steps_per_exchange * integrator.timestep
```

```json
"production": {
  "method": "rest2",
  "duration_per_segment": "5 ns",
  "exchange": {"number_of_exchanges_per_segment": 1000}
}
```

At a 4 fs timestep that is 1,250,000 steps per segment and 1,250 per exchange round -- a 5 ps
interval. Both divisions must be exact and are **refused rather than rounded**: an exchange interval
off by a step drifts the schedule out of alignment with the committed watermark while the run still
looks healthy. A count that does not divide is rejected with nearby counts that do.

`duration_per_segment` is the length of ONE segment. How many segments to run is an execution choice
made by the driver script (`NUMBER_OF_SEGMENTS`), and how many were committed is runtime state in
the run manifest. A segment count in the scientific input would move the configuration hash and make
a longer run look like a different calculation.

The earlier form -- `n_exchange_per_segment` plus `exchange_interval` -- is retired and refused with
a migration message naming both replacements.

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


### `forcefield.json`

Every system bundle carries a human-readable record of how it was parameterised. It is written by
`MD_system_gen.py` and is what a reader consults instead of deserialising `system.xml`:

| field | meaning |
|---|---|
| `route` | the resolved parameterisation route: `ligand` or `peptide` |
| `input_route` | how the molecule entered: `smiles` or `pdb` |
| `system_type` | `ligand`, `protein`, or `protein-ligand` |
| `xml` | the OpenMM force-field files actually loaded |
| `water` | the water force field, e.g. `amber19/opc.xml` |
| `protein_forcefield` | `null` on a ligand route -- so an accidental protein load is an error |
| `ligand` | small-molecule force field, charge method, net and formal charge, atom count |
| `nonbonded` | method, cutoff, switching, dispersion correction, Ewald tolerance |
| `hmr` | target hydrogen mass, how many hydrogens were repartitioned, scope, total mass |
| `constraints_note` | states that constraints and hydrogen mass belong to the **built System**, so changing either needs a new bundle rather than a new protocol |

The water model is recorded twice on purpose: `forcefield.water` is the model that was **simulated**,
while `system_manifest.json`'s `water.packing_model` records whose pre-equilibrated box supplied the
starting coordinates. They differ for models OpenMM cannot build a box for -- OPC is packed with
TIP4P-Ew geometry and parameterised by `amber19/opc.xml` -- and recording both keeps that
substitution visible instead of leaving a reader to infer it.

## Configuration ownership: two files, one canonical model

The two public generators consume two configuration files with strictly separate ownership.

| | `system_config.json` | `md_config.json` |
|---|---|---|
| owns | `system.type`, `ligand_build`, `forcefield`, `solvation`, `system_build` | `protocol.integrator`, `protocol.equilibration`, `protocol.production`, `execution`, `reporting` |
| consumed by | `MD_system_gen.py` | `MD_input_gen.py` |

Each **rejects** the other's keys. Writing `production` into `system_config.json` raises an error
naming the file it belongs in, rather than being ignored -- an ignored setting is one the author
believes took effect.

Both resolve through the **same** canonical typed model. Stage JSON files are *projections* of the
resolved model, never independent sources of defaults: a value that could be set in two places is a
value that can disagree with itself.

### One known seam

`conventional_md.duration` and `minimization.restraint` in `md_config.json` are **generator-level**
keys. The canonical model describes one production method, while a staged protocol has both a `cMD`
stage and a `REST2` stage. Until the model grows a multi-stage production block these two live
outside it -- they are extracted before canonical resolution and recorded in `run_manifest.json`
with their values, so nothing is silent, but they do not participate in the configuration hashes.

### Conventional MD as a standalone method

`protocol.production.method = "md"` is a supported public method with its own stage graphs:

```
explicit md    : min -> eq_nvt -> eq_npt_1 -> eq_npt_2 -> cMD_1
implicit md    : min -> eq -> cMD_1
explicit rest2 : min -> eq_nvt -> eq_npt_1 -> eq_npt_2 -> cMD_1 -> REST2_1
implicit rest2 : min -> eq -> cMD_1 -> REST2_1
```

Four graphs from two independent facts: implicit solvent has no NPT stage, and conventional MD has
no REST2 stage. An MD-only project constructs **no** REST2 object -- no tau ladder, no exchange
schedule, no omega policy, and no seed for a stage that does not exist.

For an MD-only project the cMD segment length is the canonical
`protocol.production.duration_per_segment`. The generator-only `conventional_md.duration` applies
only to the pre-production cMD stage of a REST2 chain, and stating both is **refused** rather than
resolved by precedence -- two fields must never compete for one stage.

#### cMD runs in committed segments

`cMD_1` was single-shot: re-running it overwrote its outputs, which looked like a continuation and
was a restart. It now has the contract REST2 already had, built on the same `runstate` primitives:

* an atomic committed-generation record per segment;
* a binary checkpoint, preferred for continuation, with a serialized State as an **announced**
  fallback -- the State is physically valid but does not restore the integrator's random stream;
* continuity comparison **before** any output is opened for append;
* per-stream reporting watermarks, because an all-atom trajectory at 100 ps and a solute trajectory
  at 10 ps reach the same boundary with different frame counts;
* quarantine of any tail beyond the committed watermark.

Re-invoking the stage launcher continues the same run in the same directory. It never starts a
sibling run and calls that continuation.

```bash
CMD_NUMBER_OF_SEGMENTS=2 ./run_all.sh      # equilibrate once, then two cMD segments
cd cMD_1 && ./cMD_1.sh                     # add one more segment, later
```

`CMD_NUMBER_OF_SEGMENTS` and `REST2_NUMBER_OF_SEGMENTS` are separate because adding cMD segments and
adding REST2 segments are different requests. Both live in Bash and neither enters the scientific
JSON or the continuity hash: asking for a longer run must not change the configuration hash.

#### Reporting is declared per stage

The reporting cadence is a **production** cadence. An equilibration stage of 10 ps cannot produce a
frame at a 100 ps interval, so the generator omits that stream for that stage and records why,
rather than declaring an output nothing will write. The state log falls back to a cadence that fits,
so equilibration stays observable.

### Explicit water or implicit solvent

`solvation.mode` discriminates, and each mode rejects the other's fields **by name** rather than
ignoring them. A configuration stating `ionic_strength_molar` under implicit solvent describes an
experiment that would not be run, and nothing in the output would say so.

```json
{"solvation": {"mode": "implicit", "implicit_model": "GBn2", "radii": "mbondi3"}}
```

`GBn2` with `mbondi3` is the **only** publicly accepted combination. The pairing is not arbitrary:
GBn2 was parameterised against mbondi3, so another radius set is a different Hamiltonian that still
runs. OpenMM exposes HCT, OBC1, OBC2 and GBn, and ParmEd accepts several radius sets, but the energy
identity in this repository is pinned for one pair and public acceptance follows the evidence rather
than the library's capability. A recognised-but-unvalidated value is refused with that wording --
not "unknown", which would send a reader hunting for a typo. Spellings are canonicalised
case-insensitively, so a bundle records one name for one thing.

**When each is appropriate.** Explicit water is the reference treatment: it represents hydrogen
bonding, dielectric screening and hydrophobic packing with real molecules, at the cost of a box that
is usually 100x the solute. Implicit solvent replaces all of that with a continuum, which makes long
sampling of a small solute affordable and removes solvent viscosity so conformational transitions
happen faster in wall-clock terms. What it does not do is reproduce specific water bridges, ion
pairing, or any property that depends on discrete solvent structure. Choose it to sample a solute's
own degrees of freedom cheaply; do not choose it to study solvation itself.

#### What implicit mode does not have, and why the absence is enforced

No water, no box, no ions, no salt concentration, no PME, no real-space cutoff, no pressure and no
barostat. Implicit solvent has no volume, so pressure is undefined; an NPT stage or a `pressure`
field is refused **before generation**, with the stage graph spelled out, rather than ignored at run
time.

The implicit stage graph is therefore:

```
min -> eq -> cMD_1 -> REST2_1
```

There is no `eq_npt_1` or `eq_npt_2` and there cannot be.

#### The construction path is part of the Hamiltonian

The System is built through ParmEd:

```python
st = parmed.load_file(prmtop, xyz=rst7)
parmed.tools.changeRadii(st, "mbondi3").execute()
system = st.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds,
                         implicitSolvent=GBn2, removeCMMotion=True)
```

`AmberPrmtopFile.createSystem()` is **not** interchangeable with it. Measured on ACE-ALA-NME with
identical per-particle GB parameters and identical radii, the two agree to 0.0000 kJ/mol on every
force except `CustomGBForce`, where they differ by **16.05 kJ/mol**. Under REST2 that offset is
several kT of spurious work, so every replica in a ladder must sit on the same branch.

`changeRadii` runs unconditionally. Measured: a **no-op** for a tleap-built peptide topology
(max |dR| = 0.0000 Å, since tleap already wrote mbondi3) and **load-bearing** for an OpenFF/Sage
topology (max |dR| = 1.70 Å, since those carry no GB radii at all). The bundle manifest records
which case a given bundle was, so a reader does not have to trust the claim.

`system.prmtop` and `system.rst7` are construction intermediates and provenance for the OpenMM
System. **This repository has no Amber execution engine.**

#### Mass and timestep are not inherited

The implicit profiles do **not** repartition hydrogen mass. The pinned reference builds its base
System without repartitioning and the GBn2 energy validation is against an unrepartitioned System,
so 3.024 amu hydrogens would be a different build that still passes every structural check. Without
HMR, 4 fs is not stable for the fastest remaining motions, so the implicit profiles use **2 fs**.
That choice is stated in the profile, not inherited by accident.

#### Implicit REST2

The whole system is the enhanced region, and a partial selection is **refused**. A generalised-Born
energy is not separable per atom -- every Born radius depends on every other atom's position -- so a
partial region needs a validated treatment of the solute-environment cross terms, and there is none
here.

The complete `CustomGBForce` energy is scaled by `s = (1 - tau)^2` through an injected global
parameter, not by scaling charges. GBn2 has three energy terms and one is a non-polar correction
with no charge dependence: `charge x sqrt(s)` would leave it at full strength. Measured: at `tau = 0`
the scaled System is bitwise identical to the unscaled one, and at `tau = 0.1, 0.25, 0.4, 0.5` the GB
energy equals `s x` the unscaled GB energy to ~1e-13 kJ/mol. Exactness at *every* tau is what shows
all three terms scale.

Omega exclusion remains torsion-only and is applied after the enhanced region is resolved, so it
never alters GB or nonbonded scaling.

**Ladder sizes.** Implicit ladders are shorter than their explicit counterparts: 4 replicas for the
peptide route, 6 for the ligand route, against 10 for explicit water.

#### Named profiles

`implicit-md-peptide-v1`, `implicit-md-ligand-v1`, `implicit-rest2-peptide-v1`,
`implicit-rest2-ligand-v1`. Named rather than branching hidden inside the explicit profiles, so
choosing implicit solvent is a decision recorded in the configuration.

Adding them moved **no existing hash**: the canonical build document omits whichever solvent
treatment is absent, so `"implicit": null` never enters an explicit build. Carrying it would have
changed `system_build_sha256` for every existing explicit configuration and made every prepared
bundle look stale for a field that says nothing.

### Refusing to overwrite

Both generators check the destination **before doing any work** and stop if a file they would write
is already there. Parameterising a ligand costs half an hour; finding out at the publish step that
the destination was occupied throws all of it away.

The rule is stated in terms of the files being written, not the directory:

```
MD_system_gen: destination /path/to/bundle already holds 13 file(s) this system bundle would write:
    checksums.json
    config.json
    ...
  Refusing to write: a destination half-rewritten from a different configuration would run without
  complaint and mean nothing.
  Use --overwrite to replace it, or choose another destination.
```

A destination holding only *unrelated* files is not a reason to stop, and those files are preserved:
the staged output is moved in entry by entry rather than replacing the directory.

There are two ways to say yes, and they mean different things.

`--overwrite` replaces the **whole** destination directory, so it deletes files the generator never
wrote -- on a project that has run, that is every result:

```
  --overwrite replaces the WHOLE destination directory, which would also delete 228 file(s) it did
  not write:
    REST2_1/  (207 files)
    cMD_1/  (4 files)
    min/  (4 files)
    run.log
```

`--overwrite-generated` rewrites only the files the generator produces and leaves everything else
in place, so a project keeps its results, checkpoints, trajectories and logs. This is the right tool
when the **generator** changed and the protocol did not -- a fixed launcher script, a new stage
projection -- which is a real situation: a generated project is a snapshot, so fixing a generator
bug does not fix projects already written.

It is the wrong tool when the protocol changed, and that is refused:

```
MD_input_gen: refusing to keep the results in /path/to/project: they were produced under a
different protocol.
    recorded : 26c53711db497c51
    this run : 1ed16940d99eb985

  Use --overwrite to discard them, or generate a new project with
  --inherit <this project>/run_manifest.json:<stage> to reuse an endpoint without
  pretending the old results belong to the new protocol.
```

The comparison is on the recorded `protocol_sha256`, so it is the resolved protocol that decides,
not the config file's text.

Both the check and the publish step live in `md_templates.openmm.destination`, so every mode means
the same thing in both generators, and the check protects callers that never go through the command
line.

### Inheritance is not restart

`--inherit` takes the `run_manifest.json` of a previous generated run and records lineage: what this
project descends from, and the hash of that manifest. It never parses a log, and it is not a way to
continue a simulation.

With the optional `:<stage>` selector -- `--inherit ../parent/run_manifest.json:eq_npt_2` -- the
named stage's endpoint becomes this project's starting state, so a project that only varies the
production protocol does not re-run equilibration another project already did. Every stage up to and
including the named one is listed in `lineage.skipped_stages`; the inherited stage is skipped too,
because what is inherited is its *endpoint*. Naming them is what keeps the shortcut auditable, and an
unknown stage name is an error rather than a silent inheritance of nothing.

Continuing a REST2 run is a different thing entirely and happens **inside that run's own directory**,
through the committed-generation record. `REST2_1.sh` assembles the bundle and hands it to the
runner, passing the previous run directory if one exists; the runner reads its own record to find the
restart point. The stage layer never reads or writes that record. It remains the sole authority for
the restart boundary, reachable also as `md-openmm rest2 --resume-run`.

### CUDA preference and the CPU validation path

Execution prefers CUDA with mixed precision. Every generator has a `--dry-run` that resolves and
validates the full configuration without touching a GPU, and `python -m md_templates.openmm.stage
--validate` checks a single stage's inputs the same way. That is what CI uses.
