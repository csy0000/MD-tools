# Changelog

## 0.5.2 — 2026-09-11

**A finished run can leave this package behind.** `md-openmm export-reference -idata <run> -odir
<out>` writes a directory that runs on OpenMM alone — the System that was integrated, the topology,
the state the stage continued from, a standalone runner, provenance and a `SHA256SUMS` inventory.
Nothing in it imports `md_tools`, enforced by running the bundle under an import hook rather than
by reading the source. cMD and REST2; umbrella and AIS are not done.

A REST2 bundle does not reimplement the ladder. The modules that decide what happens — the
acceptance criterion, the sweep schedule, the reduced potential, the seed derivation — are copied
byte for byte, which took moving `BAR_NM3_TO_KJ_PER_MOL` and `driver._stream_seed` into
`remd/core.py`. Checked against the engine's own run: 10 of 10 exchanges with an identical
state-to-walker mapping, not merely a similar acceptance rate. Its `md_tools_commit` is the engine
that ran, as recorded; the commit the modules were copied from is `ladder_modules_from`. The rung
construction travels too: `rest2/hamiltonian.py` (OpenMM only, the one implementation) is copied
into the bundle, the solute atoms and omega bonds are recorded, and `verify_rungs.py` rebuilds
every rung from rung 0 and requires an identical System. The runner also performs the ladder's
per-state `equilibration_steps`, which it used to skip; a ladder with that setting above zero is
now reproduced exchange for exchange, and tested so.

**Every bundle carries `input/`**: the structure, the build-top configuration, the run's
`resolved.config`, the built System and topology every stage ran on, and each stage's `.in` -- each
proven against the run's own records before it is copied. `input/build_system.py` rebuilds the
System from the structure with OpenMM and its chemistry libraries alone, no md-tools, and checks it
is the same bytes; a test holds that for peptide and SMILES inputs in implicit and explicit
solvent. The README gives four ways to reproduce a run, starting with OpenMM alone, and says where
the structure came from -- for a capped peptide, the tleap `sequence` that writes it, checked by
running tleap at export.

**A ladder can equilibrate every rung under its own tau.** `rest2.equilibration_per_tau: true`
(REST2/rREST2, off by default) stops the tau = 0 chain at minimisation under implicit solvent, or
after its NPT stages under explicit solvent, which fix the box every rung shares; every rung, tau = 0
included, then runs `eq_nvt_posres`, `eq_nvt_posres_2` and `eq_nvt_free` under its own tau at fixed
volume, then `equilibration_steps`, then exchanges. The stages are done as the stage chain does them
-- a fresh integrator per stage seeded per rung, velocities carried, the chain's restraint set after
the configuration -- on a restrained copy of each rung, so the propagated rungs are unchanged. Each
rung's end state is kept and verified into `restart.json`; an interruption during it is refused by
`--resume` by name. A reference bundle carries the same code (`ladder/rung_equilibration.py`) and
reproduces such a ladder exchange for exchange. Off, every generated script, `run.sh`,
`_protocol.py` and the resume identity are byte-identical to before; each `.in` gains one line.

**Test datasets.** `docs/campaigns/test-systems-2026-09/` builds, runs, exports and registers ALA
and phenol, implicit and explicit, cold and hot cMD and REST2, 1 ns each, under
`$MD_DATA/2026/md-tools/<system>-test/`.

The first version of the cMD exporter carried the **build** System rather than the integrated one
— a different Hamiltonian at non-zero tau — plus the config seed instead of the derived one, the
built coordinates instead of the continuation state, and a restraint left on by `setState`. All
four ran perfectly and none was visible from reading the script. The test did not catch them
because it compared the export against a Context built in the test file, sharing all of its
assumptions. It compares against the engine now.

**A wheel knows its own commit.** `md_tools_commit` was `null` everywhere, because a wheel is built
from a directory and nothing in the build consults git. An in-tree PEP 517 backend bakes it at
build time; a dirty tree bakes nothing, since a commit that does not describe the wheel is worse
than no commit.

**Shared datasets leave the year.** `--common-data` registers to `common/{project}/{data}` rather
than `{year}/common/{project}/{data}`: a reference is used for as long as it is the best one
available, and a year segment means finding it requires knowing when it was made. `data_name` may
be several segments deep, each validated separately.

**`origin` named the wrong repository.** It ran git with no `-C`, so it recorded whichever
repository the person was standing in — three datasets claimed MD-tools' HEAD for another project's
campaign, passing both guards on the way. It resolves from `-idata` now and refuses when the data
are not in a repository, naming `--project-repo`. New `--notes TEXT`.

**The environment is `openmm-env`.** `environment-cuda.yml` creates `openmm-env` and the CPU
file `openmm-env-ci`; the README installs to `envs/openmm-env`. `md-openmm` is the command the
environment provides, and naming the environment after it made "install md-openmm" and "run
md-openmm" sentences about different objects.

**The development history and the ALA reference campaign live here now**, under `docs/history/`
and `docs/campaigns/ala-2026-09/`, copied from the project repository where that work was done.
Four tests cited "the MD-project journal" as their real-run evidence; they cite a file in this
repository instead, and nothing here refers to another repository's working tree.

**Three more tests that failed for reasons outside the code.** Two counted the machine's GPUs
with `nvidia-smi`, which ignores `CUDA_VISIBLE_DEVICES`: on a suite confined to five of nine cards
one asked for device 8, and a six-state ladder passed its own "needs six devices" guard and failed
later with two ranks on one card. They count the devices the process can use now. The third read a
directory another test created, and its module built the wheel inside the checkout, where parallel
workers collided and the collision was reported as a skip. Behind all of them, `conftest.py` wrote
each test's GPU assignment into the worker's environment and never took it back, so a test given
no assignment inherited the previous test's card; it now resets before every test.

**Three tests were not running and reported it as a fact about the software** — a stale filename
(`cMD.csv` for what is now `mdout.csv`) that had never once been satisfied, two tests reading
another test's output, and an interrupt driven by a 45-second timer. One skip remains, an opt-in
evidence writer. `--dist loadgroup` is now the default so an expensive module-scoped fixture is
built once rather than once per worker.

## 0.5.1 — 2026-09-10

**An interrupted CV-enabled REST2 ladder can be resumed.** It could not be, on any ladder whose tau
was not exactly representable at six decimal places — four, eight or twelve rungs. Three
implementations of one linear tau ladder disagreed in the seventh decimal: the ladder that RUNS
rounds to six places and its values are what a generated `_protocol.py` executes and what every
`cv_stateN.json` records, while the resume check recomputed an unrounded one and compared with a
1e-12 tolerance. The resume was refused for a serialisation artefact, with the data intact.

There is one implementation now. `run/continuation.py`'s inline copy is gone and
`rest2.linear_tau_ladder` delegates rather than rounding to match — two implementations that agree
are what produced this.

Found by interrupting a real four-rung ladder at step 2,610,250 of 5,000,000 and resuming it: the
deferred integration experiment from `docs/release-notes/20260907-readiness.md`. After the fix the
same run completed with 20001 of 20001 rows on every state, no gap and no duplicate.

Ladders of two, three, five or six rungs were unaffected, as was any ladder with CV reporting off.


## 0.5.0 — 2026-09-10

**Breaking.** MD-templates became MD-tools: a standalone, pip-installable package with one
executable and three commands. Every detail, with the tests that verify it, is in
[`docs/release-notes/v0.5.0.md`](docs/release-notes/v0.5.0.md).

### The interface

- one installed executable, `md-openmm`, with exactly `build-top`, `build-md` and `data-register`
- AIS is a `build-md` protocol, not a fourth command
- **removed**: `sys-config`, `sys-gen`, `md-gen`, `setup`, `show-default`, the `openmm-md`
  executable, and the `md-template` environment installer. `pip install md-tools` replaces the
  installer; nothing replaces the rest, because the three commands cover what they did.

### Torsion collective variables

- new `collective_variables: {file, interval_steps}` section on every MD protocol, off by
  default; supplying only one of the two keys is an error rather than a guess
- a strict `cv.yaml` (v1: torsions only) naming atoms by zero-based index or by
  chain/residue/atom selector; ambiguous selectors, duplicate names, unknown fields, duplicate
  YAML keys and unsupported types are all refused rather than resolved
- **observation only**: no `Force` is added, and `md_tools.cv` imports no OpenMM at all. The suite
  asserts that enabling reporting leaves the System serialisation, force inventory, force groups
  and single-point energy identical
- degrees, wrapped to `[-180, 180)`, IUPAC/MDTraj sign, triclinic minimum image applied to the
  three sequential bond vectors
- a cadence independent of the trajectory and state-data intervals and possibly finer, required
  to divide exactly: the cMD stage length, the REST2/rREST2 exchange interval, and for AIS to sit
  on the parameter-update grid *and* divide `switching_steps`
- one CSV per cMD stage, per REST2/rREST2 **thermodynamic state** (with `walker_index` and a
  documented **pre-exchange** boundary convention), and per AIS path plus a verified-manifest-only
  `AIS_cv.csv` aggregate; each with a JSON sidecar carrying units, conventions, digest and
  resolved indices
- the definition is resolved against the MD configuration file and copied content-addressed into
  the generated directory, so a generated tree stays movable; on resume the series is truncated to
  the checkpoint's committed count before appending
- CV evaluation count and wall time are recorded under `cv_*`, never merged into energy evaluations

See [`docs/collective_variables/README.md`](docs/collective_variables/README.md).

### Configuration

- `configs/{machine,sys,md}/` are ordinary browsable files at the repository root, one copy each,
  shipped as wheel data files and located through distribution metadata
- every length is an exact integer step count; logs derive ps/ns
- examples must resolve, through the real resolver, to the model's own defaults

### The dataset contract

- contract **v2**, owned by MD-tools: `$MD_DATA/{year}/{project}/{data}/`, no month segment
- `extension.yaml` ported with a newly required checkpoint digest
- `dataset-v2.0` and `extension-v2.0` schemas generated from the models and drift-checked
- **`md_data` is no longer imported at runtime**, and contract v1 is gone

### A stable import API, and generated files that are entry points

- `md_tools.md` (`PositionalRestraint`, `ReportingConfig`, `run_stage`, `run_generated_stage`,
  `run_generated_workflow`), `md_tools.rest2` (`REST2Scaler`, `ScalingSelection`), `md_tools.remd`
  (`REMDRunner`, `NeighborExchangeRule`, `run_remd`, `run_generated_remd`),
  `md_tools.remd.reservoir` (`ReservoirRefreshRule`) and `md_tools.ais` (`run_generated_ais`)
- generated Python files are now compact entry points that call those APIs; `resolved.config`
  beside them is the single resolved declaration, found from `__file__`, re-validated at execution
  and bound into the checkpoint fingerprint
- **one REST2 scaler** for fixed-τ cMD, REST2, rREST2 and AIS
- `openmm/templates/` removed: it held installed runtime code, not templates. `md_tools.runtime`
  remains as compatibility-only facades so pre-v0.5 generated scripts still run

### Scientific options

- box shape: `cube` and `octahedron` documented alongside the `dodecahedron` default
- ligand force field: GAFF available and **resolved to an exact installed version**
- crossed protein/water pairs warn instead of refusing, and the warning is recorded
- `hydrogen_mass_repartitioning: {enabled, hydrogen_mass_amu}` replaces the null-means-off scalar
- `dynamics.timestep_fs: auto` resolves from the masses in the built System

### One current MD configuration model

- `md_tools.build.md` is the only authority for MD workflow configuration. The retired `methods:`
  model — `md_defaults`, `ais_defaults`, a second `resolve_md_config` and a second `stage_plan`,
  in `duration_ns` and `switching_duration_ps` — is gone from `openmm/`
- `openmm/config.py` → `openmm/system_config.py` (the `build-top` system resolver),
  `openmm/defaults.py` → `openmm/system_defaults.py`, `write_yaml` → `openmm/yaml_io.py`;
  `openmm/stages.py` deleted, having had no importer
- `templates/openmm_md.py` → `templates/replica_executor.py`: an internal module must not be named
  after a retired executable
- one `ConfigError`, not two, so a refusal from the builders is the one the CLI handles

### Fixed

- `src/md_tools/build/` — the package implementing `build-top` and `build-md` — had never been
  committed, silently excluded by an unanchored `build/` ignore rule. A clone of this repository
  did not contain two of its three commands.
- a stage now records the parent state it continued from **with its digest**, so a parent rewritten
  after a child consumed it is caught rather than becoming false ancestry
- a 4 fs timestep is refused unless the masses in the System prove hydrogen mass repartitioning

---

Earlier releases are in Git history. The state immediately before the v0.5.0 cleanup is preserved
at the tag `pre-v0.5-doc-cleanup`; this file no longer carries hundreds of lines about commands and
an installer that no longer exist.
