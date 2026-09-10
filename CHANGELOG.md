# Changelog

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
