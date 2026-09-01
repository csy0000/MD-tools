# Instruction: close MD-tools v0.5.0 release blockers and remove superseded repository burden

Date: 2026-09-01

## Starting point and scope

Work on `csy0000/MD-tools`, branch `dev`. The observed head when this instruction was written was:

```text
ab8c84c31c63812ee4fa350960daf6273b2987d0
```

Re-fetch and record the actual head before editing. Work from a clean tree. Do not overwrite unrelated work. Do not merge `dev` into `main`, publish to PyPI, create the final `v0.5.0` release tag, or edit registered scientific data in this task. The goal is a release candidate that can be reviewed before promotion.

This instruction supersedes the unfinished portions of the 2026-09-01 standalone migration. Complete all sections as one coherent cleanup because they meet at the same architectural boundary: MD-tools must have one three-command interface, one current contract, one visible configuration layout, and one small set of authoritative documentation.

## Non-negotiable target

The only installed executable is `md-openmm`. Its only public work commands are:

```text
md-openmm build-top
md-openmm build-md
md-openmm data-register
```

AIS is a protocol selected by a `build-md` configuration. Do **not** add `build-ais` or any fourth public work command.

At completion:

- there is no runtime import of `md_data`;
- there is no v1 `{namespace}/{yyyy-mm}/{dataset_name}` implementation in maintained code;
- `sys-gen`, `md-gen`, `sys-config`, `show-default`, `setup`, `openmm-md`, `md-data-register`, and `md-data-finish` have no compatibility shim or hidden active route;
- dataset contract v2 includes dataset and extension/continuation semantics;
- ordinary users can browse complete config examples at repository-root `configs/{machine,sys,md}/`;
- active docs describe only the current architecture;
- `CLAUDE.md` and CI cannot teach an agent or test runner to use retired commands.

## Phase 0: preserve and revalidate before cleanup

Before deleting any historical material:

1. verify the working tree is clean and fetch `origin/dev`;
2. record the starting commit in the release-candidate notes;
3. create and push an annotated preservation tag at the exact pre-cleanup commit:

   ```text
   pre-v0.5-doc-cleanup
   ```

4. confirm the tag resolves remotely to the recorded commit;
5. re-run reference/import searches for every deletion group below;
6. do not delete a file if an active test, package entry point, current report, build workflow, or maintained source module still depends on it; first migrate the dependency and record the replacement.

Git history and the preservation tag are the archive. Do not move stale material into another `legacy/`, `archive/`, `journal/`, or agent-readable directory in the active branch.

## Phase 1: make the configuration examples visible

The current root `configs` is a mode-120000 Git symlink to `src/md_tools/configs`. GitHub therefore shows it as a symlink blob instead of an ordinary directory. Remove that symlink and create real, browsable files:

```text
configs/
├── machine/
│   └── user.config.example
├── sys/
│   └── build-top.config
└── md/
    ├── cMD.config
    ├── REST2.config
    ├── rREST2.config
    └── AIS.config
```

All files use YAML syntax despite the `.config` suffix. They must be comprehensive examples: every accepted key, type, unit, default, enum, incompatible combination, and consequential scientific choice is explained in comments. Unknown keys remain errors.

### Configuration ownership

- `configs/machine/user.config.example` documents the file created by `md-openmm data-register --init`. It contains no real user identity or machine path. The actual config remains under `${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config`, subject to the documented override precedence.
- `configs/sys/build-top.config` contains only system-construction choices: input interpretation where relevant, protein and small-molecule force fields, water/GB model, salt/neutralization, ion identities, box shape and padding, cutoff, constraints, HMR, and parameterization options.
- `configs/md/*.config` contain protocol and runtime choices: timestep, stages, integer step counts, restraints, ensemble, thermostat, barostat where valid, output/checkpoint intervals in steps, seeds, platform policy, and protocol-specific sections.
- GBn2 has no pressure, periodic box, barostat, salt ions, or NPT stage. Its commented example names NVT stages honestly.
- HMR defaults false. A 4 fs timestep is refused unless the serialized System proves compatible HMR.
- cMD, REST2, rREST2, and AIS duration/output fields use integer steps. Logs also state the derived ps/ns.

The repository-root files are the canonical user-facing examples. Do not solve wheel packaging with repository symlinks or maintain two hand-edited copies. Include the examples in the wheel through a standard packaging mechanism, such as wheel data files, and locate installed examples through distribution metadata when a test needs them. Runtime defaults remain in validated Python models; tests must prove each example resolves to the same defaults/schema and that the built wheel contains all seven files.

Remove the old `src/md_tools/configs/` mirror once no runtime import depends on it. If a temporary generation step is needed during migration, make it deterministic and remove the generated source-tree copy before completion.

Update README paths, package-data declarations, `MANIFEST.in`, wheel tests, schema/example drift tests, and help text accordingly.

## Phase 2: migrate AIS to `md-openmm build-md`

AIS is the sole reason the retired `sys-gen`/`md-gen` path and contract v1 remain. Make AIS a first-class `protocol: AIS` accepted by the same strict `build-md` configuration resolver used by cMD, REST2, and rREST2:

```bash
md-openmm build-top -i INPUT -os built.xml -op built.pdb -log built.log \
  --config configs/sys/build-top.config

md-openmm build-md -odir ./md_script/ --config configs/md/AIS.config
```

The generated AIS bundle must use the installed `md_tools.runtime` machinery, accept the same Amber-like `-p`, `-s`, `-log`, restart/output conventions where applicable, contain no absolute source-checkout path, and work from an installed wheel outside the repository.

Preserve the existing tested scientific AIS behavior unless a test demonstrates a defect:

- source configurations are drawn from an explicitly identified equilibrium source ensemble;
- the source thermodynamic state and target state are recorded;
- each switching path has an independent deterministic seed and provenance;
- work increments are calculated at the Hamiltonian update using a stated before/after convention;
- cumulative work and reduced work are recorded per observation;
- paths are independently restartable and a completed path is not appended or rerun silently;
- no barostat is active during Hamiltonian switching;
- all generated outputs and completion records are validated before completion is declared.

### AIS configuration and coordinate rules

- Public/config/persisted protocol coordinates use `tau`, not a second public `s` variable. Internal scale factors may be derived from tau but are not an alternative persisted coordinate.
- Replace time-length fields such as `switching_duration_ps` with integer step fields such as `switching_steps` and `observation_interval_steps`. The log derives ps/ns from the resolved timestep.
- `AIS.config` documents number of paths, tau start/end, switching steps, observation interval steps, source trajectory/topology/state selection, source-frame policy, direction, seeds, output intervals, and platform.
- The schedule includes both endpoints, states precisely whether observation 0 occurs before any work, and refuses a non-integral or internally inconsistent schedule rather than rounding it.
- Do not make AIS part of the common cMD stage chain. It consumes an existing source ensemble, and `run.sh` must require that source explicitly.

Port the strong existing AIS GPU/scientific tests to the new public route. Tests must invoke the real `md-openmm build-md --config ...AIS.config` command, not a helper that bypasses the CLI.

## Phase 3: remove the retired generator and v1 contract

After AIS works through `build-md`, trace active imports and delete the entire obsolete chain rather than leaving it labeled legacy. Expected removal candidates include, subject to revalidation:

```text
src/md_tools/openmm/md_data_contract.py
src/md_tools/openmm/mdgen.py
src/md_tools/openmm/sysgen.py
```

Also remove:

- all `md_data` imports from maintained runtime and templates;
- the v1 dataset/preflight branch in `templates/preflight.py`;
- the `dataset:` block used only by the retired generation route;
- obsolete v1 constants, environment assumptions such as `MD_DATA_LOCAL`, and install hints;
- conftest routing of retired subcommand names;
- tests whose sole purpose was to exercise a deleted route, after migrating their still-valid provenance, scientific, integrity, and configuration assertions to the three-command implementation.

Do not remove tests merely because they fail. Classify every affected test assertion as migrated, obsolete with a named deleted feature, or still blocking. Preserve tests for force-field provenance, exact commits, dirty-checkout truthfulness, stage fingerprints, hashes, completion detection, REST2 scaling, AIS work, and registration safety.

Run a maintained-code search gate. Outside migration/release notes that explain history, there must be no active dependency on:

```text
md_data
MD_DATA_LOCAL
{namespace}/{yyyy-mm}/{dataset_name}
sys-gen
md-gen
sys-config
show-default
setup
openmm-md
md-data-register
md-data-finish
```

## Phase 4: complete contract v2 with extension semantics

The current v2 port contains only the dataset model/schema. Port and adapt the extension semantics from `csy0000/MD-data@20d982eb463ed439095f1b95e00ff1b1d75906b4` into `md_tools.data_contract` without importing `md_data`.

Create a versioned `extension.yaml` model, exported JSON schema, validators, documentation, and tests. Preserve these rules:

- completed and archived datasets are immutable;
- an extension of a completed/archived dataset creates a new dataset under the v2 year-first layout;
- the new dataset has a new stable dataset ID and records the parent dataset ID and component;
- new output goes into a component owned by the new dataset;
- any linked parent components are declared, read-only, and never the output component;
- the source checkpoint is copied if a runtime may overwrite it, and its hash is recorded;
- restart step, original length, extension length, final length, and combined length are explicit and mutually consistent;
- extension status has requested/running/complete/failed/abandoned semantics with valid timestamp ordering;
- a completed extension cannot finish after the dataset containing it was declared complete;
- no operation writes new output into a completed parent.

Adapt canonical paths to v2:

```text
$MD_DATA/{year}/{project_name}/{data_name}/
$MD_DATA/{year}/common/{project_name}/{data_name}/
```

There is no month segment and no reserved v1 `baseline/` namespace. The path year retains its current single meaning: completion year, or creation year while active.

Registration must recognize and validate extension records, verify parent/checkpoint identities without mutating the parent, and retain the same transactional guarantees: dry-run purity, collision refusal, resumable copy, destination checksum verification, source preservation until verification, and symlink creation only after success.

Export and drift-check at least:

```text
dataset-v2.0.schema.json
extension-v2.0.schema.json
```

## Phase 5: remove superseded documentation and agent burden

The active repository currently contains about 80 stale instruction/journal/example files, over one megabyte of text. Revalidate, then remove these groups from `dev`; do not move them to another active archive directory:

```text
docs/examples/
docs/implementation/
docs/journals/
docs/legacy/
claudecode-instructions/
```

For `docs/journal/`, preserve the current v0.5 migration evidence by consolidating it into the release-candidate note described below, then remove the entire directory, including all older entries. This instruction itself may be deleted in the final cleanup commit after it has been read and its execution evidence has been recorded; it remains recoverable from the preservation tag and Git history.

Remove `docs/md-defaults-scientific-rationale.pdf` as a generated duplicate unless an active distribution workflow proves it is a required release artifact. Keep the Markdown source and bibliography. If `scripts/render_markdown_pdf.py` then has no active use, remove it after reference checks.

Revalidate legacy helper scripts. In particular, remove `scripts/shrink_configs_for_ci.py` when CI no longer uses retired configs, and remove `scripts/retrofit_fair_v030.py` if it exists only for the retired v0.3/v1 contract. Preserve a script only with a named active entry point or reproducibility obligation.

### Final active docs

Reduce `docs/` to a small authoritative set equivalent to:

```text
docs/
├── README.md
├── data-contract.md
├── scientific-defaults.md
├── scientific-defaults.bib
├── replica-exchange.md
├── support-matrix.md
└── release-notes/
    └── v0.5.0.md
```

Exact filenames may vary only if there is a concrete reason, recorded in the release note. There must be no active `examples/`, `implementation/`, `journal/`, `journals/`, `legacy/`, or historical-instructions directory.

Requirements:

- `docs/README.md` is a short router declaring which documents are authoritative and telling agents not to search Git history unless a historical question requires it.
- consolidate the useful current FAIR boundary into `data-contract.md`, but correct ownership: MD-tools now owns validation and transactional registration; MD-data is not a runtime dependency.
- retain the scientific rationale and bibliography, renamed consistently if needed;
- retain and update the current REST2/rREST2 scientific contract in `replica-exchange.md`;
- retain a concise current `support-matrix.md`;
- `release-notes/v0.5.0.md` records the migration, tests, starting/final commits, config layout, contract v2, deliberate removals, and any remaining blocker. It is not a chronological dump of every intermediate edit.

Rewrite `CHANGELOG.md` so its top-level current section describes v0.5.0 and the three-command architecture. Old details remain in Git history; do not keep hundreds of lines about removed installers and commands in the active changelog.

Rewrite `CLAUDE.md` as concise durable guidance. It must state exactly three public commands, the `configs/{machine,sys,md}` ownership, one v2 contract, current scientific invariants, test lanes, and rules against project-specific paths. Remove every retired command and the false “six public commands” statement. Keep it short enough that an agent can read it before acting.

Update the root README to link only active documents and visible config examples. Do not make the README a second implementation manual.

## Phase 6: replace stale CI and release packaging

The current `.github/workflows/release.yml` still invokes `show-default`, `sys-config`, `sys-gen`, and `md-gen`; therefore it is not evidence for the current package. Replace it.

CI must run for pull requests and pushes to `dev` and `main`, plus manual dispatch. Tag-triggered release validation must use the new naming convention intended for MD-tools rather than the historical `openmm-v*` convention.

The CPU/package job must:

1. build the wheel and install it with no source checkout on `PYTHONPATH`;
2. prove `md-openmm -h`, `--version`, and help for exactly the three public commands;
3. prove retired subcommands fail;
4. locate all seven installed config examples from the wheel/distribution and validate them;
5. generate cMD, REST2, rREST2, AIS, and split/all-in-one bundles as applicable from outside the checkout;
6. compile generated Python and inspect expected files without claiming simulation evidence;
7. export/check dataset and extension schemas;
8. run the non-GPU test lane with unexpected skips treated as failures;
9. upload useful logs on failure.

Keep GPU dynamics as a separate locally recorded acceptance lane unless a CUDA runner is actually configured. Never substitute CPU execution for CUDA evidence.

Do not publish a final release in this task. A later reviewed release step will merge `dev`, set the stable version, create the final tag/release, and update downstream installation commands.

## Phase 7: acceptance tests

Run and record exact commands, environment, wall times, and results.

### Fast and package lanes

At minimum:

```bash
python -m build
python -m pytest tests -m "not slow and not gpu"
```

Install the wheel into a fresh environment and run all CLI/config/schema/generation checks from outside the checkout.

### CUDA/scientific lane

On an actual CUDA platform, run shortened but meaningful smoke tests for:

- explicit and GBn2 `build-top`;
- split and all-in-one cMD;
- REST2 state trajectories, exchange decisions, `rem.log`, checkpoint and continuation;
- rREST2 reservoir identity and velocity provenance;
- AIS source selection, tau schedule, work convention, independent paths, restart and completion detection.

Record the device and OpenMM platform. Preserve existing long marked validations; do not silently replace them with shorter tests or delete them.

### Registration and extension lane

Using temporary XDG config and `$MD_DATA` roots, prove:

- `data-register --init` behavior and precedence;
- project and common year-first paths with no month;
- dataset and extension schema validation;
- incomplete/active data refusal;
- active-writer refusal;
- path traversal and collision refusal;
- same- and cross-filesystem transactions;
- interruption at every durable transaction boundary and safe resume;
- destination checksum verification before source removal;
- idempotent identical re-registration;
- extension creates a new dataset and leaves a completed parent byte-for-byte unchanged.

### Documentation and search gates

Assert the final config and docs directory shapes. Search active code, CI, tests, README, CLAUDE, configs, and docs for retired names. Historical terms may occur only in the concise migration/release note to explain the change, not as runnable guidance.

## Commits and handoff

Use reviewable commits in dependency order:

1. visible config layout and packaging;
2. AIS through `build-md`;
3. removal of retired generator/v1 imports and migrated tests;
4. extension contract v2;
5. documentation/CLAUDE/changelog cleanup;
6. CI replacement and final acceptance fixes;
7. release-candidate notes and final search gates.

Push the completed work to `origin/dev`. Do not merge `main` and do not claim v0.5.0 released. The final report must list:

- starting and ending commits;
- preservation-tag verification;
- every removed group and its replacement/history location;
- exact config layout in source and wheel;
- proof that AIS uses the real three-command interface;
- proof that `md_data` and v1 active code are gone;
- dataset and extension schema identities;
- CI status;
- all tests run and all skipped tests;
- any remaining blocker stated plainly.

## Completion definition

This task is complete only when MD-tools has exactly three public commands, AIS is generated through `build-md`, the v1/import/shim route is gone, contract v2 covers extensions, root configs are ordinary browsable files under `machine/`, `sys/`, and `md/`, active documentation is small and current, CI tests the real wheel interface, and the complete release-candidate evidence is pushed on `dev` for review.
