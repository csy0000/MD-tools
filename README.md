# MD and REST2 — OpenMM template

A standard, self-contained template for building and running molecular simulations of peptide-like
compounds and small-molecule ligands with OpenMM: parameterise from SMILES or a PDB, solvate (or
not), equilibrate, and run conventional MD or REST2 replica exchange from hash-checked,
transportable bundles.

**Supported scope**

| | explicit water | implicit solvent |
|---|---|---|
| methods | conventional MD, REST2 | conventional MD, REST2 |
| peptide route | ff19SB + OPC | ff19SB + GBn2 / mbondi3 |
| ligand route | OpenFF Sage + TIP3P (water follows the force field) | GBn2 / mbondi3 |
| production ensemble | **NPT** | **nonperiodic constant temperature** — no box, so neither NVT nor NPT |
| nonbonded | PME, 1.0 nm cutoff | `NoCutoff` |
| integrator | Langevin-middle, 300 K, **4 fs** | Langevin-middle, 300 K, **4 fs** |
| hydrogen mass | **3.024 amu**, solute scope | **3.024 amu**, solute scope |
| ligand charges | **Sage 2.2 + NAGL AM1-BCC** (`am1bcc_nagl`) | same |

Implicit solvent is **not** a side branch: it has its own profiles, its own stage graph, an audited
`CustomGBForce` scaling path for REST2, and worked examples under `test/ala/implicit/` and
`test/ala/cMD/implicit/`.

Tested against **OpenMM 8.6.0**, pinned by `short_version` *and* the release tag commit
`c6173db6e8edd705eb59172bd21e9ce69c572405`. The official builds report a `.dev-<sha>` suffix in
`full_version` because they ship `release = False`; that is a build stamp, not a development
snapshot, and it is recorded verbatim rather than relabelled.

Two force-field routes, both exercised end to end on this machine:

| route | input | force field | worked example |
|---|---|---|---|
| ligand | `smiles` | **Sage / openff-2.2.0** + AM1-BCC | `cyclo_rgdfv`, `small_macrocycle_smoke` |
| peptide | `pdb` | **ff19SB** (+ CMAP) | `ace_ala_nme` |

The route is **declared, never inferred**: a ligand manifest may not name a protein force field and
a peptide manifest may not name a small-molecule one, because that is precisely how a peptide
silently becomes a Sage run with the same system name.

**Water follows the force field**, because ff19SB and Sage were fitted against different water:
ff19SB gets **OPC**, Sage gets plain **TIP3P**, and an ff19SB + Sage complex defaults to OPC as a
documented compatibility choice. The rule reads the resolved force-field family rather than the
input label, and refuses rather than guesses for a force field it does not recognise. Both defaults
are the active `explicit-*-v2` profiles; the superseded `-v1` profiles remain name-resolvable for
reproducing the earlier TIP3P-FB runs. See `docs/configuration.md` for the primary sources.

Nothing here depends on the project it came from: no annealed-importance-sampling machinery, no
project-specific datasets or conclusions. (An earlier version of this line also claimed "no
implicit-solvent work", which has not been true for some time -- see the scope table above.)

## Install

```bash
conda env create -f docs/implementation/explicit_solvent/environment.yml   # -> md-templates
conda activate md-templates
pip install -e . --no-deps          # from source, for development
```

`--no-deps` is deliberate: OpenMM, OpenFF, AmberTools and RDKit come from conda, and letting pip
resolve them produces a different and usually broken stack. `environment.yml` therefore lists every
runtime dependency.

To consume the pipeline **without** a checkout, install the wheel and use the packaged catalog —
see [From an installed wheel, with no checkout](#from-an-installed-wheel-with-no-checkout) below, and
[docs/support-matrix.md](docs/support-matrix.md) for what travels.

```bash
md-openmm validate-env    --platform CUDA --device 0
md-openmm validate-system --system cyclo_rgdfv --experiment rgd_rest2_10rung
md-openmm smoke           --system small_macrocycle_smoke --out-root ./runs --platform CPU
md-openmm smoke           --system ace_ala_nme           --out-root ./runs --platform CPU
md-openmm prepare         --system cyclo_rgdfv --experiment rgd_rest2_10rung \
                                --out-root ./runs --platform CUDA --device 0
md-openmm rest2           --bundle ./runs/<TIMESTAMP>__cyclo_rgdfv__bundle__<HASH> \
                                --out-root ./runs --platform CUDA --device 0
md-openmm md              --bundle ./runs/<BUNDLE> --out-root ./runs --platform CUDA --device 0
```

### Bundles, portability and CI

A prepared bundle is self-contained: `md-openmm bundle validate|inspect|relocate-check`. What is
and is not guaranteed across machines is in **[docs/support-matrix.md](docs/support-matrix.md)**;
schema changes are in **[CHANGELOG.md](CHANGELOG.md)**.

```bash
md-openmm bundle validate BUNDLE --deep
md-openmm bundle relocate-check BUNDLE      # copies elsewhere and validates there
```

### Configuration

Simulations are described by one canonical model that YAML and JSON both compile into, with
versioned default profiles and explicit units. See **[docs/configuration.md](docs/configuration.md)**.

```bash
md-openmm config list-profiles
md-openmm config init --method rest2 --route smiles --output run.yaml
md-openmm config validate run.yaml
md-openmm prepare --config run.yaml --out-root ./runs --platform CPU
```

`--system`/`--experiment` remain as the legacy front end; `config migrate` lifts them into the same
model.

### Template catalog — metadata, not yet a dispatcher

`registry.yaml` indexes the templates this repository publishes, and each one describes itself in
`templates/<method>/<engine>/<variant>/template.yaml`:

| template | method | engine |
|---|---|---|
| `templates/conventional-md/openmm/explicit-water/template.yaml` | `conventional-md` (alias `md`) | `openmm` |
| `templates/rest2/openmm/explicit-water/template.yaml` | `rest2` | `openmm` |

**Nothing dispatches through these files.** Runs still go through `md-openmm` over
`md_templates.openmm`, and every descriptor states that in `implementation.dispatch: legacy-direct`.
What the descriptors add is a machine-readable answer to "what is this template, what can it do,
which persistent schemas does it read and write, and how far has it actually been validated?" —
with implementation status and scientific status kept as separate fields, because a template that
runs is not a template that has been validated.

A template's immutable identity is its repository URL, the **full 40-character commit SHA**, and its
exact path:

```
https://github.com/csy0000/MD-templates@<40-hex-sha>#templates/rest2/openmm/explicit-water/template.yaml
```

```python
from md_templates.core import load_catalog, resolve_identity
catalog = load_catalog(".")
resolve_identity(catalog, "rest2/openmm/explicit-water").canonical
```

Resolution **proves** provenance — the commit is never supplied by the caller. In a checkout the tree
must be clean and the commit is HEAD; outside a checkout only explicitly trusted build provenance is
accepted. A **dirty checkout has no immutable identity** and this raises rather than quietly resolving
to HEAD, because HEAD describes what was committed, not the files on disk. Branch names, tags,
abbreviated SHAs, package versions and timestamps are refused outright, and symlinked paths are
refused so the commit SHA always identifies the bytes actually parsed.

Loading and validating the catalog imports no OpenMM, OpenFF or RDKit, so it works on a machine that
could never run a simulation.

#### From an installed wheel, with no checkout

The catalog travels inside the distribution, so an installed copy can list templates and resolve
identities without a repository, without Git and without network access:

```python
from md_templates.core import load_packaged_catalog, resolve_packaged_identity
sorted(load_packaged_catalog().descriptors)          # both template IDs
resolve_packaged_identity("rest2/openmm/explicit-water").canonical
```

The identity always uses the **logical repository path** — `templates/rest2/openmm/explicit-water/
template.yaml` — never a path inside the wheel, so it stays lookupable in the repository it names.

What an installed copy may claim depends entirely on how it was built:

| source it was built from | packaged `source_state` | identity |
|---|---|---|
| clean Git checkout | `clean-git-checkout` | resolves to that exact commit |
| unmodified sdist from a clean checkout | `verified-source-archive` | resolves to the inherited commit |
| tree with uncommitted or untracked changes | `dirty-source-tree` | **refused** |
| tree whose `git status` failed | `unverifiable-git-status` | **refused** |
| no Git and no verified archive record | `no-verifiable-git-provenance` | **refused** |
| a tree inside an unrelated repository | `no-verifiable-git-provenance` | **refused** |

Provenance is never inherited from an enclosing repository: the Git worktree root must *be* the
source root, not merely contain it. A source tree unpacked inside another project — vendored, or in
an ignored directory — would otherwise report that project's HEAD, and report it clean.

An unmodified sdist inherits its commit only while **both** its catalog bytes and a digest over
**every regular file in the archive** still hash to what it recorded. That digest is closed-world:
everything is covered except generated artifacts (`__pycache__`, `*.pyc`, `*.egg-info/`, top-level
`build/` and `dist/`, `.git/`) and the provenance record itself. An archive edited anywhere — source,
README, build configuration, a referenced document — cannot carry the original commit forward.

A refusal raises `UnresolvedBuildProvenanceError` and names the state. Listing still works in every
case: knowing *what* a distribution contains is useful even when its bytes cannot be named.

Two records travel beside the catalog. `resource_manifest.json` hashes the exact bytes of every
packaged file; `build_provenance.json` names the commit and binds itself to that manifest by hash,
so a commit cannot be paired with a catalog it did not describe. Both are verified before any
identity is returned, and both are deterministic — two builds of one commit produce identical
metadata.

**Trust boundary, stated plainly.** This detects bytes that drifted, a resource that went missing, a
metadata record paired with the wrong catalog, and a build from a tree nobody can name. It does
**not** authenticate GitHub, and it does not defend against someone who can rewrite a wheel *and*
both of its metadata records consistently. Nothing here is a signature.

The single tracked source stays `registry.yaml` and `templates/**/template.yaml` at the repository
root. The packaged copy is generated at build time into the build tree and is never written back
into the working tree, so there is exactly one file to edit.

Still true, and worth repeating: **the catalog dispatches nothing**, and template identity is **not**
written into any bundle or run manifest.

### Run directories, and continuing a run

Both `md` and `rest2` take the same naming options:

```bash
md-openmm rest2 --bundle B --out-root ./runs --run-name production_a   # exactly ./runs/production_a
md-openmm rest2 --bundle B --out-root ./runs                           # timestamped default
md-openmm rest2 --bundle B --out-root ./runs --resume-run production_a # continue IN PLACE
```

* `--run-name` is used verbatim: no timestamp, system or hash is appended.
* A fresh run refuses an existing directory rather than reusing it.
* `--resume-run` continues the same directory -- never a sibling or a child -- and
  **Each invocation runs one segment**, so resuming extends the run. Segment count is not a
  configuration field: the driver script decides how many segments to request, and the run
  manifest records how many committed.
* Before a resume loads anything, the continuity contract in `run_state.json` is compared with the
  requested configuration. Force field, integrator, constraints, chunk length, ladder and the
  omega-exclusion setting must match; the number of chunks deliberately need not. A mismatch is
  refused with the differing fields listed, before a byte is appended.
* At each chunk boundary a restart *generation* (OpenMM checkpoint + portable State) is written and
  then committed by one atomic replacement. On resume the checkpoint is preferred; a corrupt or
  foreign-platform checkpoint falls back to the State, which is physically valid but not bitwise
  identical, and says so.
* REST2 reports **lifetime** and **this-invocation** exchange statistics separately; lifetime
  counters are rebuilt from the durable log and never restart at zero.

Conventional MD declares its own plan. An experiment with only a `rest2:` block is refused for
`md`, rather than falling back to the package default -- which is 1000 ns.

## What is here

```text
src/md_templates/
  openmm/                      the pipeline: 34 modules
    cli.py, cli_system_gen.py, cli_input_gen.py    the three public entry points
    config.py                  the canonical configuration model and its defaults
    water_policy.py            which water a force-field family was fitted against
    solvation.py               box geometry, solvation, ion accounting
    system.py                  System construction, REST2 scaling, omega policy
    equilibration.py           the staged build and the restraint conventions
    stage.py, cmd_segments.py  staged execution and the committed-generation contract
    runstate.py, dcdtail.py    restart records, and record-aware DCD tail recovery
    hashing.py                 one definition of a content hash
    schemas.py, bundle*.py     manifests, bundle validation and portability
    spec/                      profiles, resolution and the profile registry
    manifests/systems/         the shipped system definitions
  core/, engines/              the template catalog and its packaged provenance
docs/
  configuration.md             the configuration model, water policy and cMD contract
  support-matrix.md            what is guaranteed across machines
  journal/                     dated records of what was changed and why
tests/                         the suite; `-m "not slow"` is the fast gate
scripts/ci/                    fast_checks.sh and integration_cpu.sh, the same steps CI runs
```

The public surface is three commands -- `md-system-gen`, `md-input-gen`, `md-openmm` -- and a
generated project directory. Everything else is an implementation detail that the manifests record.

## Relationship to OpenMM

This package is a protocol and a provenance layer around OpenMM, not a reimplementation of it.
Where OpenMM already does something, it does it here:

| task | who does it |
|---|---|
| solvation, ion placement | `Modeller.addSolvent` |
| hydrogen placement at pH | `Modeller.addHydrogens` |
| box vectors for a shape | `Modeller._computeBoxVectors` |
| hydrogen mass repartitioning | `ForceField.createSystem(hydrogenMass=...)` |
| checkpoints and portable States | `Simulation.saveCheckpoint` / `saveState` |
| positional restraints | `CustomExternalForce` |
| topology output | `PDBFile`, `PDBxFile` |

Three things are deliberately **not** delegated, each for a reason that was measured rather than
assumed:

* **box sizing.** `addSolvent(padding=...)` knows nothing about the nonbonded cutoff. With a small
  padding -- 1.2 nm, rhombic dodecahedron, 1.0 nm cutoff -- it builds a 2.400 nm box whose reduced-box
  height is 1.697 nm, and `Context` construction then fails with *"the cutoff distance cannot be
  greater than half the periodic box size"*. `cutoff_fit_policy` grows the box until that height
  clears the limit by a recorded margin, or refuses with the padding that would work.
* **DCD tail recovery.** Nothing in OpenMM reads DCD records or truncates on a record boundary,
  which is what crash recovery needs.
* **the committed-generation contract.** Segment commits, watermarks and continuity identity have no
  OpenMM counterpart; they are what makes a run resumable rather than merely restartable.

Where a delegation exists, a test asserts it stays equivalent -- including for
`_computeBoxVectors`, which is a private OpenMM helper, so a local fallback survives and the test
fails loudly if upstream moves it.

### Box geometry and padding

The explicit default is **`padding_nm: 2.0` with `padding_semantics: "openmm"`**, reproducing
`Modeller.addSolvent`: a bounding-sphere radius `R` about the centre of the solute's axis-aligned
bounding box, then `width = max(2R + padding, 2*padding)`. **OpenMM has no numeric padding default of
its own** -- 2.0 nm is this repository's choice, expressed in OpenMM's semantics.

Three quantities are involved and they are **not** interchangeable:

| quantity | what it is | value |
|---|---|---|
| shortest lattice translation | distance from a point to its own periodic image | `width`, for all three shapes |
| minimum reduced-box height | what OpenMM's cutoff check uses, `min(a_x, b_y, c_z)` | `width`, `width/√2`, `√6/3·width` |
| solute-image clearance | conservative solute-surface to copy-surface distance | `shortest lattice translation − 2R` |

Clearance is measured against the **lattice translation**, not the height. Reading the height as an
image distance understates the real separation by 29% in a dodecahedron. The cutoff, separately, must
satisfy `min reduced-box height ≥ 2·cutoff + system_build.minimum_image_margin_nm`, and
`cutoff_fit_policy: grow` enlarges the box until it does.

At the 2.0 nm default both alanine and the RGD macrocycle get more than 2 nm of solute-image
clearance, comfortably beyond twice the 1.0 nm cutoff. `"solute-image-gap"` remains available for
callers who want `padding_nm` to mean the clearance directly.

Every number is recorded in the bundle's `geometry` block: requested padding, bounding radius,
requested and final width, shortest lattice translation, solute-image clearance, minimum reduced-box
height, cutoff, required cutoff height, margin, and whether the box was grown. Two legacy fields --
`min_image_distance_nm` and `solute_image_gap_nm` -- hold the reduced-box height and that height
minus the solute diameter; they are kept so old manifests still parse and are labelled in the
manifest as **not** the solute-to-copy distance.

## State of the science — read before quoting anything

* `cyclo_rgdfv` + `rgd_rest2_10rung` is **`ladder_status: pilot_supported`**. That is a claim about
  one molecule on declared pilot evidence. It is *not* `validated`, not converged, and not
  production-ready; the predeclared conjunctive acceptance rule was never formally satisfied.
* Any other macrocycle needs its own pair-resolved acceptance pilot — start from
  `macrocycle_pilot_8rung.yaml`, which is `ladder_status: unvalidated` by design.
* **The 2 fs / 4 fs hydrogen-mass-repartitioning gate is partly closed.** The *static* half is
  settled by measurement: repartitioning does not change the potential or its gradient at all
  (`0.000e+00` on both, mass conserved), so the pinned energy identities still hold. What remains
  open is the *dynamical* half — whether 4 fs reproduces 2 fs ensemble averages for these systems —
  and long production is still blocked on that.
* Restart safety and convergence are not established.
* The smokes prove installability and mechanical execution. Nothing more.
* **No ladder is validated for any peptide.** `ace_ala_nme` proves the ff19SB route runs; it says
  nothing about rung spacing for a peptide of any size.

## Source provenance — the wheel and this tree agree

*Historical, as of the hand-off.* The distributed wheel `md_templates-0.1.0-py3-none-any.whl`
(sha256 `50e9a1a51ecbd3680d43985786813727186d534c486a9501ebd4895fe639a183`) was built from commit
`10809c7` of the originating repository, and all nine modules `md_templates/openmm/` then contained
were **byte-identical** between that commit and the shipped wheel, verified by comparison rather
than assumed from a version string. The package has since grown to 34 modules and diverged
substantially, so this records where the tree came from rather than what it currently equals; the
live check is `scripts/ci/fast_checks.sh`, which rebuilds the wheel and compares the packaged
resources against the checkout on every run.

Two fixes sit on top of it, both found by running the pipeline rather than reading it:

* `environment.yml` asked for the conda package `build`, which conda-forge packages as
  `python-build`. With `channels: [conda-forge, nodefaults]` pinned, `conda env create` was
  unsatisfiable — the package could not be installed by following its own instructions.
* a `pdb`-route bundle did not copy the structure its own manifest points at, so it could not be
  re-validated and the peptide route could build a system but never use one.

## Historical validation evidence

The 2026-08-15 target-machine validation, the ff19SB peptide-route report and the ladder pilots
were produced by the package as it was BEFORE this refactor: a different distribution name, a
different console script, experiment schema v1, and the retired salt fields. They are accurate
about what they tested and are preserved in git history at `f885be5` -- `git show f885be5:reports/` --
but they are not kept in the working tree, because every command in them names an entry point this
template no longer has, and rewriting those names would turn a true record into a false one.

Re-running them against the current CLI is the way to restore an evidence directory here.

## Provenance

Extracted from the research repository. Published as a single initial commit: the tree is what
matters for a template, and the development history remains in the originating repository.
Implicit-solvent support has since been developed here and is part of the template.

## The two public entry points

Preparation and protocol generation are **separate commands**, because they answer different
questions and change at different times.

```bash
python MD_system_gen.py -i ALA.pdb -o alanine_system --config system_config.json
python MD_input_gen.py  --system alanine_system/system_manifest.json \
                        -o alanine_run --config md_config.json
```

| | `MD_system_gen.py` | `MD_input_gen.py` |
|---|---|---|
| answers | **what** the molecule is | **what is done** to it |
| owns | chemistry, force fields, solvent, ions, box, topology | minimisation, equilibration, cMD, REST2, reporting, execution |
| config | `system_config.json` | `md_config.json` |
| produces | an immutable system bundle | a staged project |
| **never** | runs minimisation or any dynamics | reparameterises, resolvates or rebuilds the system |

Each rejects the other's settings rather than ignoring them: a protocol block written into
`system_config.json` would never be applied, and silence about that is worse than an error.

Why separate: preparing a system costs real time — solvation, parameterisation and charge
derivation — while regenerating a protocol against an existing bundle is instant and needs no GPU.
One prepared system can back several protocols with no chance of one of them quietly re-solvating
it. (The charge step is much cheaper than it was: the default is now NAGL AM1-BCC, which takes about
a second for cyclo-RGDfV against roughly 40 minutes for single-conformer AmberTools `am1bcc`, still
selectable.)

### Supported inputs

| extension | reader | system type |
|---|---|---|
| `.pdb` | PDB | **must be declared** -- a PDB may hold a peptide, a ligand or a complex |
| `.mol`, `.mol2`, `.sdf` | molecule | ligand (unambiguous) |
| `.smi`, `.smiles` | SMILES | ligand, and `ligand_build` must state charge, stereochemistry, protonation, conformer generation, charge model and parameterisation route |

The extension chooses the *reader*, never the chemistry.

### The generated project

```
alanine_run/
    inputs/            immutable copy of the prepared system + checksums
    min/               min.json       min.sh
    eq_nvt/            eq_nvt.json    eq_nvt.sh
    eq_npt_1/          eq_npt_1.json  eq_npt_1.sh     position-restrained NPT
    eq_npt_2/          eq_npt_2.json  eq_npt_2.sh     free NPT
    cMD_1/             cMD_1.json     cMD_1.sh
    REST2_1/           REST2_1.json   REST2_1.sh
    run_all.sh         run_manifest.json      run.log
```

Conventional MD stages are `cMD_N` and replica exchange stages are `REST2_N`. The two NPT stages are
separate because they are different protocols, not one repeated: `eq_npt_1` holds the solute under
the 1 kcal/mol/A^2 positional restraint while the box relaxes, and `eq_npt_2` releases it.

Each stage JSON names the topology and input **State** it consumes and which stage produced it; a
State rather than a PDB, because positions alone would discard velocities and box vectors at every
boundary.

Every stage executes through `python -m md_templates.openmm.stage`. The single-shot stages run
in-process; **REST2 is delegated** to the runner, which owns the committed-generation restart
contract. The stage layer assembles the bundle the runner expects and calls it -- it decides nothing
about restarts, because a second implementation of a restart boundary is exactly what this
repository forbids.

Each launcher records the interpreter that generated the project and preflights it. A bare `python`
is not safe here: the stack's activation script puts AmberTools' interpreter first on PATH, and it
has neither openmm nor `md_templates`. Override with `PYTHON=... ./run_all.sh`.

Both generators refuse to write when a file they would produce already exists, naming the files
rather than just calling the directory non-empty, and they check *before* doing any work -- a bundle
build can cost half an hour. A destination holding unrelated files is not blocked, and those files
survive the write. `--overwrite` replaces the **whole** destination directory, so on a project that
has already run it deletes the results too; the error says how many files that is and where they
are, before you commit to it. `--overwrite-generated` instead rewrites only the generator's own
files and keeps results, checkpoints and logs -- correct when the generator changed and the protocol
did not, and refused when the protocol itself changed.

`--dry-run` validates a whole project without a GPU.

`--inherit <run_manifest.json>[:<stage>]` continues from an endpoint another project already reached.
Without the `:<stage>` selector it records lineage only; with it, the named stage supplies the
starting state and every stage up to and including it is recorded in `skipped_stages` rather than
silently omitted. It is not a checkpoint resume -- continuing a REST2 run happens inside that run's
own directory, under the runner's own record.

### Conventional MD

`production.method = "md"` generates a standalone conventional-MD project, ending at `cMD_1` with no
REST2 stage and no REST2 machinery. Explicit runs get the two NPT equilibration stages; implicit runs
get neither, because there is no box.

cMD runs in **committed segments** in one run directory: each segment ends with an atomic commit
holding a checkpoint (preferred for continuation) and a portable State (announced fallback), and
re-invoking the launcher continues the same run rather than restarting it. Trajectories and logs
append, with per-stream watermarks so a crash leaves an uncommitted tail that the next invocation
removes rather than appending after.

A continuation is checked before anything is opened for append. The on-disk record is
`cmd_schema_version: 3`, which binds the identity of what is being continued by exact bytes: the
predecessor State, `system_manifest.json` and `forcefield.json`, alongside the System, topology,
ordered atom identity, integrator, ensemble, restraint convention, segment length and reporting
cadence. Change any of them and the resume is refused with the field named, because two segments
that disagree about the Hamiltonian are two simulations sharing a trajectory file -- which is worse
than an error, since the file still looks continuous. A stream *shorter* than its watermark is
missing history and is refused rather than silently re-run; a longer one is a crash tail and is
truncated on real DCD record boundaries. Asking for more segments is the one change that is always
safe, so segment **count** lives in Bash and never enters the continuity hash.

```bash
CMD_NUMBER_OF_SEGMENTS=2 ./run_all.sh   # equilibrate once, then two production segments
cd cMD_1 && ./cMD_1.sh                  # add another segment later
```

Worked examples: `test/ala/cMD/explicit/` and `test/ala/cMD/implicit/`.

### Implicit solvent (GBn2 / mbondi3)

`solvation.mode: implicit` builds a generalised-Born System instead of a water box. Defaults are
`GBn2` with `mbondi3` radii -- the pairing GBn2 was parameterised against.

Implicit mode has **no** water, box, ions, salt, PME, cutoff, pressure or barostat, and the stage
graph is correspondingly shorter:

```
explicit:  min -> eq_nvt -> eq_npt_1 -> eq_npt_2 -> cMD_1 -> REST2_1
implicit:  min -> eq ->                             cMD_1 -> REST2_1
```

There is no NPT stage and there cannot be one: with no box there is no volume to equilibrate and
pressure is undefined. Stating one is refused before generation rather than ignored at run time.

The System is built through ParmEd -- `load_file`, `changeRadii`, `Structure.createSystem` -- and
**not** through `AmberPrmtopFile.createSystem`. The two differ by ~16 kJ/mol in `CustomGBForce` with
identical radii and identical per-particle parameters, which under REST2 is several kT of spurious
work, so the construction branch is part of the Hamiltonian. `system.prmtop` and `system.rst7` are
kept as construction provenance; there is no Amber execution engine here.

Only GBn2 with mbondi3 is publicly accepted; other models and radius sets are refused as not
validated. Implicit profiles repartition hydrogen mass to 3.024 amu and use a 4 fs timestep, the
same as explicit — stated explicitly in each profile rather than inherited. They previously did not,
on the argument that the GBn2 energy validation was against an unrepartitioned System; that was
measured and found not to hold, because mass enters the kinetic term only. Repartitioning
cyclo-(RGDfV)'s 38 hydrogens left the GBn2 potential and every force component unchanged to
`0.000e+00`. Implicit REST2 requires the whole system as the
enhanced region and scales the entire GB energy by `s`, including the non-polar term that charge
scaling alone would miss. Ladders are shorter: 4 replicas for the peptide route, 6 for the ligand
route.

Worked examples: `test/ala/implicit/` and `test/rgd/implicit/`.

### Current implementation status

OpenMM is implemented. The system manifest carries an `adapter_status` block that says plainly that
**Amber and GROMACS are not implemented** -- Amber would emit `prmtop`/`rst7`, GROMACS `top`/`gro`,
and a GROMACS `tpr` is stage-specific and would belong to input generation, not preparation.

## Branch policy

| branch | role |
|---|---|
| `main` | stable and default. Only ever receives `dev` at a validated milestone. |
| `dev` | integration. Everything lands here first. |
| `feature/*`, `fix/*`, `docs/*` | normal work. Branch **from `dev`**, target **`dev`**. |

Normal work never targets `main` directly. `dev` is merged into `main` only when a milestone has
been validated — the point of the split is that `main` is always a state someone can build on
without checking what happened to it that week.

The repository's previous primary branch was `openmm`; it was renamed to `main` using GitHub's
branch-rename operation, which retargets open pull requests and leaves redirects in place. It was
not simulated by force-pushing, so history is continuous across the rename.
