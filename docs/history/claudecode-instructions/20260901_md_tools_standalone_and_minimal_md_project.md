# Instruction: replace MD-templates with standalone MD-tools and reduce MD-project to a consumer

Date: 2026-09-01

## Repositories and starting points

This is a coordinated, breaking migration across sibling repositories. Begin from clean checkouts of:

- `csy0000/MD-templates`, branch `dev`, observed starting commit `bed236f7e3eae5ce2fc1d5fec9818109971231e6`;
- `csy0000/MD-project`, branch `dev`, observed starting commit `a020dc35118b1dbe11696133a7d6d9cacee6d713`;
- use `csy0000/MD-data`, branch `protect-md-project-dev-test`, commit `20d982eb463ed439095f1b95e00ff1b1d75906b4`, as the semantic source for the existing dataset contract.

Re-read the live branches before editing. Do not assume the observed commits are still heads. Do not overwrite unrelated or uncommitted work. If either working tree is dirty, report the overlapping paths and stop before any move, rename, or deletion.

The implementation order is mandatory:

1. make the future MD-tools package, CLI, data contract, and tests work while the remote repository may still be named `MD-templates`;
2. install and test that package from a built wheel in an environment with no source checkout on `PYTHONPATH`;
3. simplify MD-project and prove its examples consume only the installed package;
4. only then rename the GitHub repository and local checkout from `MD-templates` to `MD-tools`, update remotes and documentation, and repeat the acceptance tests.

Do not rename first and leave two broken repositories behind. A GitHub repository rename requires repository-administration permission. If unavailable, finish all code changes and report the single remaining manual rename rather than inventing a workaround.

## Target ownership

`MD-tools` is a standalone, pip-installable software package. For now its sole executable is:

```text
md-openmm
```

It owns:

- OpenMM topology/system construction;
- generation of readable OpenMM MD scripts;
- the OpenMM runtime used by those generated scripts, including cMD, REST2, and rREST2;
- human-readable and machine-validated run records;
- the MD-data dataset schemas and validators needed by registration;
- finish detection, inventory, checksumming, transactional movement, and local symlink creation;
- user/machine configuration discovery.

`MD-project` is a scientific project and consumer of the installed package. It must not vendor, pin, clone, import, or execute a sibling MD-tools checkout. It owns only the project aims, documentation, project source/configuration/input files, and ignored local data links or staging directories.

`MD-data` is no longer a runtime dependency of either repository. Port the relevant contract semantics, schemas, validation tests, and documentation from the named protected branch into MD-tools, with attribution in migration documentation. Do not import `md_data` at runtime. Do not silently edit, delete, or archive the MD-data repository as part of this task.

## Naming migration

Apply the new name consistently to maintained code and documentation:

| Old | New |
|---|---|
| GitHub repository `csy0000/MD-templates` | `csy0000/MD-tools` |
| local checkout folder `MD-templates` | `MD-tools` |
| distribution `md-templates` | `md-tools` |
| import package `md_templates` | `md_tools` |
| primary source root `src/md_templates/` | `src/md_tools/` |
| provenance component name `MD-templates` | software package `MD-tools` |

Remove the `md-template` console entry point. Do not create a second `openmm-md` executable. The only executable in this phase is `md-openmm`. Existing historical journals may retain old names when they describe historical facts, but current commands, package metadata, examples, tests, and architecture documents must use MD-tools.

Use a new breaking development version, at least `0.5.0.dev0`. Ensure the wheel contains all shipped config examples, schemas, and script templates through package-data declarations. Package resources must be located with `importlib.resources`, never by assuming a source checkout or repository root.

After all tests pass, rename the GitHub repository through GitHub settings or an authenticated equivalent, rename the clean local directory, and set `origin` to the new URL. Verify the new repository URL and `git remote -v`. Do not rely on GitHub's old-name redirect in maintained configuration.

## CLI surface

The top-level help must expose exactly these public work commands, plus standard version/help behavior:

```text
md-openmm build-top
md-openmm build-md
md-openmm data-register
```

Legacy `sys-config`, `sys-gen`, `md-gen`, `setup`, and MD-project-local `md-data-finish`/`md-data-register` are not part of the new public workflow. Remove them after migrating the behavior that is still required. No legacy command may remain as the only route to a required capability.

Use `argparse` or an equivalently strict parser. The user-requested spellings, including single-dash multi-character options, are contractual and must work. Standard double-dash aliases may also be provided where helpful. Every command and generated script must return nonzero on validation or execution failure and must support `-h` without importing a functioning CUDA platform.

## `md-openmm build-top`

Required interface:

```text
md-openmm build-top \
  -i INPUT.pdb|INPUT.smi \
  [-os built.xml] \
  [-op built.pdb] \
  [-log built.log] \
  [--config CONFIG]
```

Defaults:

- `-os`: `./built.xml`;
- `-op`: `./built.pdb`;
- `-log`: `./built.log`;
- protein force field: ff14SB;
- small-molecule force field: Sage 2.2;
- optional protein force field: ff19SB;
- solvent: TIP3P, with OPC and GBn2 as supported alternatives;
- explicit solvent: neutralize first, then reach 0.15 M using Na+ and Cl-;
- box: dodecahedron with 1.5 nm minimum solute padding;
- HMR: false.

Ship a comprehensively commented YAML-format example, despite the `.config` suffix, at:

```text
configs/openmm/md_build.config
```

Also include it in the wheel as a package resource. Document every accepted key, type, unit, enum, default, incompatible combination, and relevant scientific consequence. Reject unknown keys and invalid combinations rather than ignoring them.

Input handling must be deterministic:

- `-i` is an existing `.pdb` or `.smi` file, not an ambiguous inline value;
- a `.smi` file must contain exactly one non-comment SMILES record for this phase; preserve or deterministically assign molecule/residue identity and record it;
- `built.pdb` contains the final coordinates/topology after hydrogen addition, parameterization, solvation/implicit setup, ions, and box construction;
- `built.xml` is an OpenMM-serialized `System` whose particle order exactly matches `built.pdb`;
- fail before replacing outputs if parameterization is incomplete or particle identities/counts disagree;
- write outputs atomically and refuse accidental overwrite unless an explicit overwrite option is implemented and tested.

Sage is used only when a small-molecule topology needs it; do not pretend it parameterized a peptide-only input. Record the actual force fields and versions used, not merely the requested defaults. GBn2 is implicit solvent: it has no solvent box, salt ions, barostat, or NPT stage. Explicit solvent choices must produce periodic systems. The log must state these resolved facts.

HMR changes particle masses in the serialized System. Record whether it was applied, the target hydrogen mass, the resulting timestep recommendation, and enough mass-summary evidence to audit it. Do not silently infer that HMR was used merely because a later config requests 4 fs.

### `built.log`

Make `built.log` readable in the style of `leap.log`: show the command, normalized configuration, input interpretation, preparation actions, warnings, atom/residue/solvent/ion counts, box vectors, charge before/after ions, force-field resolution, output paths, and a concise completion summary.

The free-text log is not a safe database format. Include a delimited, versioned YAML or JSON machine-record block in the same file, or write a contractually paired structured record while retaining `built.log`. If a paired record is used, `built.log` must name and checksum it, and registration must require both. Never scrape arbitrary prose to decide whether data are complete.

The structured build record must include at least:

- record schema version and record type;
- status and timezone-aware start/finish timestamps;
- exact command and resolved config;
- input and output relative paths plus SHA-256 and sizes;
- OpenMM, Python, MD-tools, OpenFF Toolkit, openmmforcefields, AmberTools/ParmEd/RDKit and other actually used package versions;
- MD-tools distribution version and immutable source/build commit when available;
- platform, hostname, and relevant OpenMM platform properties without secrets;
- random seeds where randomness is used;
- particle/residue counts, periodicity, box, force fields, solvent, ion details, HMR facts, and validation results.

Do not place an absolute machine path in the portable dataset manifest. Execution logs may state the path actually used, but the structured inventory and final `dataset.yaml` use paths relative to their declared roots.

## `md-openmm build-md`

Required interface:

```text
md-openmm build-md [-odir ./md_script/] [--config CONFIG] [--all-in-one]
```

`-odir` defaults to `./md_script/`. `--config` is optional. Configs use YAML syntax, strict schema validation, and step counts for all simulation lengths and output intervals. Ship comprehensively commented examples at:

```text
configs/openmm/cMD.config
configs/openmm/REST2.config
configs/openmm/rREST2.config
```

All three must be included in the wheel and tested as package resources.

### Default cMD workflow

The no-config default is explicit-solvent cMD with a 2 fs timestep and no assumption of HMR:

| Stage | Default steps | Meaning at 2 fs |
|---|---:|---:|
| minimization | 1000 minimizer iterations | not a time interval |
| restrained NVT | 5000 | 10 ps |
| restrained NPT | 5000 | 10 ps |
| unrestrained NPT | 5000 | 10 ps |
| cMD production | 2,500,000 | 5 ns |

The positional restraint default is 1 kcal mol^-1 Å^-2. Restraint units, temperature, pressure, timestep, friction/collision parameter, barostat interval, nonbonded settings, and seeds remain physical/configuration quantities; the statement that workflow lengths are in steps does not make physical constants unitless.

Default reporting intervals, in steps:

```yaml
solute_printout: 1000
system_printout: 10000
checkpoint_printout: 10000
```

The config must store the authoritative counts as integer steps. It may include comments showing the duration under its configured timestep, but runtime logs are the authoritative place that records translated ps/ns. Every stage log must show both its step count and the derived physical time. Do not round away the exact step count.

For GBn2, replace pressure-coupled stages with honest NVT equivalents. Do not call an implicit-solvent stage `NPT`, do not add a barostat, and do not fabricate pressure. Generate clearly named implicit files such as `eq_nvt_posres_2.py` and `eq_nvt_free.py`, and document the difference. For explicit solvent, the expected default files are:

```text
md_script/min.py
md_script/eq_nvt_posres.py
md_script/eq_npt_posres.py
md_script/eq_npt_free.py
md_script/cMD.py
md_script/run.sh
```

With `--all-in-one`, generate:

```text
md_script/md.py
md_script/run.sh
```

`--all-in-one` must execute the same stages with the same resolved settings, stage boundaries, logs, checkpoints, and restart semantics as the split form. Test this equivalence at the configuration/record level.

Generated scripts must be small, readable Python entry points that import the installed `md_tools` runtime. They must not import a sibling checkout or embed an absolute path. They must be syntactically valid and runnable from outside either repository.

At minimum, each stage supports the requested Amber-like inputs:

```text
python md_script/min.py -p built.pdb -s built.xml -log min.log ...
```

Define and document consistent flags for output trajectory/state/checkpoint and for the previous stage's restart/checkpoint. Preserve the requested `-p`, `-s`, and `-log` spellings. `run.sh` must chain the generated stages with explicit paths, `set -euo pipefail`, safe quoting, restart-aware behavior, and no `eval`. A completed stage is not rerun silently. A partial stage resumes only from a validated compatible checkpoint; otherwise it refuses with a useful message.

Each simulation log follows the same human-readable plus structured-record rule as `built.log`. It must record input hashes, exact resolved settings, step range, timestep, derived ps/ns, ensemble, restraints, seeds, platform, versions, output hashes, checkpoint identity, continuation/parent record, completion state, and validation results. Mark completion only after outputs and final checkpoint are flushed, reopened, and verified. A log file merely existing is not evidence of completion.

### REST2 and rREST2

Do not regress the scientifically validated behavior already present on `dev`. The `REST2.config` and `rREST2.config` files select the protocol and expose the current tau ladder, exchange interval in steps, number of exchange attempts, output intervals in steps, group/solute selection, state-trajectory behavior, neighboring acceptance reporting, `rem.log`, continuation, and reservoir settings.

Keep the current REST2 scientific contract unless a failing test demonstrates an existing defect:

- all replicas are at the same physical temperature;
- bonds and angles are unscaled;
- eligible solute torsions and CMAP scale with `(1-tau)^2`;
- ordinary amide omega remains unscaled;
- solute-solute ordinary nonbonded and 1-4 terms scale with `(1-tau)^2`;
- solute-environment nonbonded terms scale with `(1-tau)`;
- GB contribution scales with `(1-tau)`;
- exchanges do not rescale velocities;
- the runtime is NVT;
- state trajectories and checkpoint/restart semantics remain verifiable;
- rREST2 reservoir configuration and velocity provenance remain explicit.

Generated filenames may be protocol-specific, but `run.sh`, split/all-in-one behavior where applicable, logging, and registration records must obey the same contract as cMD. Do not route REST2 through an obsolete hidden CLI.

## User and machine configuration

`md-openmm data-register --init` initializes registration configuration after installation.

Do not write user configuration into the source repository, package resources, or `site-packages`: those locations may not exist, may be read-only, and would mix machine secrets/paths with distributable files. Instead use this precedence:

1. explicit `--user-config PATH`;
2. `MD_TOOLS_CONFIG`;
3. `${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config`.

Ship a commented, non-secret example as `configs/user.config.example`. The initialized file is YAML and includes both scientific user identity and machine storage configuration, for example:

```yaml
schema_version: "1.0"
user:
  name: "..."
  person_id: "..."
  orcid: null
machine:
  md_data: "/absolute/path/to/MD_DATA"
```

The environment variable `MD_DATA`, when set, overrides `machine.md_data`; an explicit CLI root override, if provided, overrides both. Print which source supplied the resolved root. Never print secrets. `--init` supports an interactive TTY flow and a fully noninteractive tested form. It creates parent directories safely, uses restrictive permissions where supported, validates the path, and refuses to overwrite an existing config unless explicitly requested.

## `md-openmm data-register`

Required interface:

```text
md-openmm data-register \
  -idata DATA_DIRECTORY \
  -project_name PROJECT_NAME \
  -data_name DATA_NAME \
  -year YYYY \
  [--common-data]

md-openmm data-register --init
```

Support `-h`. Add `--dry-run` and `--verify-only` because they are necessary safety and test surfaces. Names are single safe path segments; reject separators, `.`/`..`, control characters, empty values, and platform-specific traversal. `year` is exactly four decimal digits and is recorded in the manifest. Document whether it denotes the creation/completion year; enforce one unambiguous rule rather than silently accepting contradictory timestamps.

Canonical destination paths are now:

```text
$MD_DATA/{year}/{project_name}/{data_name}/
$MD_DATA/{year}/common/{project_name}/{data_name}/   # with --common-data
```

There is no month segment. This is a breaking MD-data contract v2. Do not weaken or patch the v1 validator until it happens to accept the new shape. Create a new versioned model/schema and migration document. At minimum, port and adapt these v1 concepts from the protected MD-data branch:

- strict schema version rejection;
- stable dataset ID separate from the path;
- relative canonical path and no absolute machine paths in manifests;
- explicit role (`project` or `common`) consistent with path shape;
- creator identity;
- timezone-aware creation/completion timestamps;
- dataset status and component status;
- exact repository/package provenance;
- component uniqueness, non-nesting, method identity, and linked/read-only rules;
- complete/archived data are immutable;
- extension/continuation provenance and ordering rules;
- catalogue validation only if a catalogue remains part of the new design.

Place models, validators, exported JSON schemas, docs, and tests inside MD-tools, for example under `src/md_tools/data_contract/`, without a runtime dependency on `md-data`. Generated schemas must be reproducible and checked for drift.

### Registration safety behavior

Port the good safety behavior currently split across MD-project's `finish.py`, `finish_gate.py`, `register.py`, `cli_finish.py`, and `cli_register.py`. Registration must:

1. resolve and validate the configured `$MD_DATA` root;
2. resolve `-idata` without following an untrusted destination construction;
3. recursively discover structured build/run records while treating free text as display only;
4. reject missing, malformed, failed, incomplete, mutually incompatible, or actively written records;
5. validate stage/restart/extension lineage and exact input-output hashes;
6. inventory every registered file with relative path, size, and SHA-256;
7. derive a complete dataset manifest and a resolved provenance manifest;
8. compute the exact year-first destination without month;
9. refuse a collision unless the existing destination is provably byte-identical and contract-identical;
10. copy/stage transactionally, including across filesystems;
11. reopen the destination and verify its inventory/checksums;
12. remove the local source only after destination verification succeeds;
13. replace the original ignored local path with a symlink to the verified destination;
14. record durable transaction state so interruption can resume safely and idempotently;
15. perform a final validation and print the canonical relative destination.

If any check fails, preserve the source. Never merge two datasets in place, never overwrite completed data, never trust a success line in a log, and never use `shutil.move` as proof that a cross-filesystem transfer completed correctly. Dry-run performs no writes. Repeating a completed registration is idempotent and does not alter data.

Tests must cover interruption at each state boundary, same- and cross-filesystem operation, active writers, truncated logs, forged completion text without a valid record, symlinks, path traversal, collisions, checksum mismatch, resumed copy, identical re-registration, extension lineage, and no-source-removal-before-verification.

## Minimal MD-project

After MD-tools works from a wheel, reduce the root of MD-project to:

```text
.gitignore
README.md
data/
docs/
aims/
src/
```

`data/` is ignored except for a minimal tracked README or placeholder needed to preserve the directory. It is the local staging area before registration and the location of symlinks after registration.

Rename `research/` to `aims/`. Preserve useful human-owned material such as aims, questions, theory, literature, decisions, and progress. Do not silently rewrite scientific claims during the move.

`src/` is the main project-development tree. Consolidate only genuinely useful project-owned inputs, project-specific configs, analysis source, and workflow source beneath it. A suggested ALA example layout is:

```text
src/ALA/
  input/ALA.pdb
  config/build.config
  config/cMD.config
  config/REST2.config
```

Do not preserve generic framework scaffolding merely by hiding it one directory deeper. Before deleting a top-level path, classify whether it is obsolete framework code, generated/ignored data, historical documentation, or project-owned scientific source. Migrate the last two to `docs/` or `src/` as appropriate and record the mapping in a journal.

Remove MD-project's:

- `components/`, `components-dev/`, `components.yaml`, and `components.lock.yaml` model;
- local `md_project` package and local finish/register executables after their tested behavior is ported;
- duplicate data-contract schemas and validators after MD-tools owns them;
- generic initialization/environment/component framework that is no longer needed;
- runtime assumptions about an MD-tools source checkout;
- obsolete root directories and root metadata outside the exact target layout.

Do not delete historical documentation that is needed to understand accepted past work; move it under `docs/legacy/` if it no longer describes the active workflow. Do not rewrite old dataset provenance to claim it was generated by MD-tools if it was actually generated by an earlier MD-templates commit.

The root README is intentionally short. It contains only a one-paragraph prerequisite and three runnable, copy-paste examples:

1. ALA cMD: build topology, generate cMD scripts, and run them;
2. ALA REST2: reuse/build the topology, generate REST2 scripts, and run them;
3. register one completed ALA data directory into `$MD_DATA`.

Use only `md-openmm` and project paths under `src/ALA/` and `data/`. Do not mention component locks, `md-template`, `openmm-md`, `md-data-register`, `sys-gen`, `md-gen`, or hidden checkout paths. Commands shown in the README must be executed in a clean acceptance test, not merely reviewed visually.

## Tests and acceptance evidence

Preserve useful current tests, but rewrite them around the new public interface. Do not make tests pass by deleting scientific or safety assertions.

### MD-tools fast acceptance

At minimum:

```bash
python -m build
python -m pytest tests -m "not slow and not gpu"
```

Install the built wheel into a clean test environment with the source checkout absent from `PYTHONPATH`, then verify:

```bash
md-openmm -h
md-openmm --version
md-openmm build-top -h
md-openmm build-md -h
md-openmm data-register -h
```

Prove package resources load from the wheel. Run `build-top` on the committed small ALA input for explicit TIP3P and implicit GBn2, validate that PDB/XML particle counts and periodicity match, and inspect the structured build records. Generate split and all-in-one cMD scripts, compile them, and run a short smoke workflow. Generate and smoke-test REST2 and rREST2 without changing their Hamiltonian semantics. GPU tests must use CUDA and be marked/skipped honestly when unavailable; never report CPU execution as CUDA evidence.

### Data registration acceptance

Initialize config in a temporary XDG config directory and use a temporary `$MD_DATA`. Register a completed smoke dataset and prove:

- project path is `YYYY/project/data`;
- common path is `YYYY/common/project/data`;
- there is no month segment;
- `dataset.yaml` and resolved provenance validate against the new schema;
- destination hashes equal source hashes;
- source becomes a symlink only after verification;
- dry-run is pure;
- interruption and retry are safe;
- collision and incomplete/active data are refused without source damage.

### MD-project acceptance

Test from a checkout with no `components/` directory and no sibling source import. Install the MD-tools wheel, run the three README examples using shortened smoke configs, and verify the resulting logs/records and registration. Then assert the root contains only:

```text
.git
.gitignore
README.md
data
docs
aims
src
```

Ignore local tool metadata when checking this assertion, but do not commit it.

### Search gates

Search maintained code, configs, tests, and active docs for stale references. Historical journals are exempt only where the old name is factually required. No maintained workflow may depend on:

```text
MD-templates
md_templates
md-template
openmm-md
md-data-register
md-data-finish
components.lock.yaml
yyyy-mm
```

The migration instruction itself and migration journal may name old terms to explain the change.

## Journals and commits

Write an execution journal in each modified repository. Include:

- starting and ending commits;
- files moved/removed and why;
- v1-to-v2 data-contract differences;
- CLI and config examples;
- repository/package rename actions;
- tests actually run, their exact commands, duration, platform, and results;
- skipped GPU/long tests and why;
- any remaining manual GitHub rename or local-folder action.

Keep commits reviewable and ordered by dependency: package namespace/resources, CLI/build functions, data contract/registration, MD-project consumer cleanup, then remote/local rename documentation. Do not claim the migration complete while any acceptance item is untested or while the two repositories disagree about the public command surface.

## Explicit non-goals

- Do not redesign the established REST2 Hamiltonian or exchange algorithm in this migration.
- Do not add Amber or GROMACS executables yet.
- Do not keep MD-project as a generic project generator.
- Do not make MD-data a required installed package.
- Do not store user configuration in the installed package.
- Do not perform long production simulations merely to demonstrate the refactor; use scientifically meaningful smoke tests and retain the existing marked long validations.
- Do not delete or rewrite already registered scientific data.

## Completion definition

This task is complete only when `MD-tools` can be installed independently, a clean MD-project can build and run the documented ALA cMD and REST2 examples using only `md-openmm`, completed output can be safely registered into the new year-first `$MD_DATA` layout, and MD-project has the exact minimal root layout without component checkouts or duplicated registration code.
