# Changelog

Entries describe behaviour changes and the migration each one needs.

Entries below `0.2.0` predate the reduction to the six-command CLI in `ca29fcd` and describe a
registry/bundle/schema architecture that no longer exists. They are kept as history; they do not
describe the current package.

## Unreleased — fixed-tau MD, and two scaling/restart fixes

**`cMD.tau` runs a single walker on one fixed rung of the REST2 ladder.** `tau: 0` (the default) is
ordinary conventional MD and leaves the System byte-identical; `tau > 0` scales the solute
Hamiltonian through the same `rest2_scaling` module the ladder uses, with the same omega exclusion.
The thermostat is unchanged — this is Hamiltonian scaling, not high-temperature MD. Verified on
CUDA: energies and per-atom forces match ordinary cMD at tau = 0, and match the REST2 rung at
tau = 0.5, to `0.000e+00`.

**Implicit REST2 was not scaling `CustomGBForce`.** Every other term scaled while the entire
generalised-Born energy stayed at s = 1 — 64 kJ/mol on ACE-ALA-NME at tau = 0.5. The template
scaler now scales it, and audits every force so an unclassifiable energy-bearing term is refused
rather than left at the wrong scale. **Any implicit REST2 result produced before this is wrong at
every rung above tau = 0 and must be rerun.**

**Reporter output past the checkpoint is now discarded on resume.** An interrupted stage left
frames ahead of its checkpoint; resuming appended after them, duplicating that interval. A 1 ns run
killed and resumed produced 108 frames instead of 100. `cMD` and `REST2` both trim to the
checkpoint before opening anything for append. **Migration:** a trajectory from an interrupted run
that was resumed may contain duplicated frames; check for a non-monotonic step column.
## 0.3.1 — commit the test fixture

`tests/data/ALA.pdb` had never been committed. `.gitignore` carries `data/`, which matches
`tests/data/` at any depth, so the fixture every MD test depends on was invisible to git.

The consequence was larger than a red CI job: **a fresh clone could not run the test suite at
all**, and every passing test result reported from a working tree was passing against a file that
happened to exist on that machine. The suite was not reproducible from the repository.

`.gitignore` now negates `tests/data/**` explicitly, so the next fixture cannot disappear the same
way. No package code changed; the 0.3.0 wheel and commands were unaffected.

This is the first release whose CI run is green.

## 0.3.0 — one directory per stage

`md-gen` now writes one directory per stage instead of one per method, and the dependency between
them is a file on disk rather than a sequence inside a single script:

```
MD/
  minimization/  eq1_nvt_1kcal/  eq2_npt_1kcal/  eq3_npt_free/    the common chain
  cMD/  REST2/                                                    siblings, from the last of those
```

Under implicit solvent the chain is `minimization -> eq1_nvt_1kcal -> eq2_nvt_free`: there is no
box, so there is no barostat and no NPT stage.

* **Every stage is independently runnable** (`cd MD/eq2_npt_1kcal && ./run.sh`) and reads only its
  parent's `final_state.xml`. `MD/run_all.sh` is a convenience wrapper that calls those scripts; it
  does not reimplement them.
* **`final_state.xml` is the handoff; `checkpoint.chk` is not.** A checkpoint resumes the stage
  that wrote it, in the same OpenMM environment. It is written while that stage is still running,
  so consuming one downstream would mean starting from a partially finished parent while every
  file on disk looked normal. `final_state.xml` is written only on success, and atomically.
* **A missing parent names the file and the command that produces it** rather than starting from
  whatever it can find.
* **cMD no longer equilibrates.** It starts from the last common stage — the same state
  `REST2/equilibrate.py` starts from — so the two are siblings and neither has to run first.
* **REST2 no longer repeats the common preparation.** `equilibrate.py` relaxes each replica under
  its own tau-scaled Hamiltonian into `replica_NN/equilibration/`; `run.py` runs exchange
  production from there into `replica_NN/production/`. `extend.sh` extends production only. Per-tau
  equilibration is not counted as production: the total remains
  `duration_per_segment_ps x number_of_exchanges`.
* **Each stage records what actually ran** in `resolved_stage.yaml` — kind, ensemble, duration or
  iterations, restraint, temperature, pressure where applicable, timestep, seeds, input and output
  state, template commit. No manifests, checksums, schema versions or run database.
* **Directory names cannot lie about the restraint.** `eq1_nvt_1kcal` only when the restraint is
  1 kcal/mol/A^2; otherwise `eq1_nvt_restrained`.

**Configuration.** `equilibration` now names its stages explicitly —
`nvt_restrained_duration_ps`, `npt_restrained_duration_ps`, `npt_free_duration_ps`, and
`nvt_free_duration_ps` for implicit — with the inapplicable ones null. `REST2` gains
`equilibration_duration_ps` (default 1000 ps).

**Migration.** A project generated by `openmm-v0.2.0` has a different layout and will not gain
these directories; regenerate it with `md-openmm md-gen`. `inputs/` is unchanged, so `sys-gen` does
not need to run again.

### Corrections made before release

* **Equilibration is grouped under `MD/eq/`** — `eq/nvt_1kcal`, `eq/npt_1kcal`, `eq/npt_free`, and
  `eq/nvt_1kcal`, `eq/nvt_free` for implicit. The ordinal prefixes are gone: the group already
  says what these are, and an ordinal in a directory name is a second statement of order that can
  disagree with the chain. Minimisation stays at `MD/minimization/` — it is not equilibration.
  Stages now locate `MD/` by looking for `md.config.yaml` above themselves rather than counting
  `..`, so stages at different depths all resolve correctly.
* **Restart accounting.** Loading the stage's OWN checkpoint keeps its step count — it was
  interrupted, and that count is how far it has come. Loading the PARENT's state resets the count
  to zero, because the parent's count belongs to the parent. Previously the parent state was
  loaded unconditionally before the checkpoint, and a stage's own count was never reset when it
  should have been.
* **Completion is identified by a canonical SHA-256 of the whole `stage.yaml`**, recorded as
  `stage_config_sha256`, and there is no redo flag. An earlier `MD_REDO=1` shortcut was removed
  before release because it did not redo anything: a completed dynamics stage still holds its
  terminal checkpoint, so it loaded that, found zero steps remaining, and rewrote the completion
  artifacts without integrating a step from the parent. Redoing a stage means generating into a
  new directory or removing that stage's six runtime outputs by hand.
* **A finished stage is not silently rerun.** `final_state.xml` is what downstream stages have
  already consumed, so re-running reports completion and changes nothing. Redoing a stage means
  generating into a new directory or removing that stage's six runtime outputs by hand; there is
  no flag for it, because a completed dynamics stage still holds its terminal checkpoint and
  anything that "reran" in place would load it, find zero steps remaining and rewrite the
  completion artifacts without integrating anything — a redo in name only.
* **Completion is identified by a canonical SHA-256 of the whole `stage.yaml`**, recorded as
  `stage_config_sha256`. A hand-kept list of significant fields accepted a completed stage after
  its input state, pressure, seeds or solvent mode had changed, because those keys were not on it.
* **`md_config_hash` hashes the generated `MD/md.config.yaml`.** It hashed the input document,
  which is not the file the project runs.
* **REST2 per-tau equilibration is concurrent** across devices and sequential within one, the same
  rule exchange production uses. It ran one replica after another, idling every GPU but one.
* **Every common stage writes the complete artifact set**, minimisation included.

**Testing platform policy.** Every test that minimises or integrates a molecular system is marked
`gpu` and runs on CUDA; on a machine without a working CUDA platform they are deselected rather
than passed, because a CPU run of them exercises a different code path from the one the work is
done on. CPU and Reference remain in use for installation and platform probes and for tests that
build no system. The GitHub workflow is renamed `release-packaging`: it runs on a GPU-less runner,
so it generates a project and checks the tree but **runs no dynamics at all**, deselects every
`gpu` test, and prints a notice saying what it did not validate. Runtime acceptance comes from
`CUDA_VISIBLE_DEVICES=... pytest tests/ -q` on the GPU machine.

**Also fixed.** Environment validation no longer fails a CPU-only runner because OpenMM's CUDA
plugin cannot load without `libcuda.so.1` — the false failure that broke the `openmm-v0.2.0`
release job. CUDA plugin and platform failures stay fatal on machines where an NVIDIA driver or
device is actually detected.

## 0.2.0 — OpenMM six-command CLI

First release of the simplified package. `md-template init`, `md-template install`,
`md-openmm sys-config`, `md-openmm show-default`, `md-openmm sys-gen` and `md-openmm md-gen` are
the whole public surface, and `md-gen` writes ordinary standalone OpenMM scripts that do not import
this package.

**Scientific corrections since `openmm-v0.1.0`.** These change what a run does, not only how it is
spelled:

* REST2 `duration_per_segment_ps` is the propagation time BETWEEN exchange rounds. The runner
  previously divided one segment among the exchanges, so the default `10 ps x 1000` advanced 10 ps
  in total instead of 10 ns.
* Both methods, and every REST2 replica, now run the configured restrained minimisation ->
  restrained NVT -> restrained NPT -> unrestrained production. The `equilibration` block was
  previously read for cMD, ignored for REST2, and its positional restraint applied by neither.
* Positional restraints follow the System: `periodicdistance` under a box, plain Cartesian
  `(x-x0)^2+(y-y0)^2+(z-z0)^2` without one. A periodic restraint on an implicit GBn2 system made
  `System.usesPeriodicBoundaryConditions()` report True for a system with no meaningful box.
* A resumed NPT run restores the production barostat before loading its checkpoint. A barostat's
  frequency lives in the System, not the checkpoint, so a resumed run previously continued with the
  inactive barostat equilibration had left behind — NPT that was silently NVT.
* Every REST2 replica gets its own integrator, velocity and barostat seed; the shared constant
  integrator seed is gone.
* A two-replica ladder exchanges on every round. Phase 1 offers no pair with two rungs, so the
  runner falls back to the other phase.
* Whole-system and solute-subset trajectories are written at their own configured intervals, for
  cMD and per REST2 replica, and append on restart. `sys-gen` also writes `solute.pdb`, since a
  subset DCD cannot be read against the whole-system topology.
* REST2 replicas share one thermostat temperature and one beta and differ by Hamiltonian, so the pV
  terms cancel in the NPT acceptance criterion. Positions and box vectors travel together through
  the cross-energy evaluation and an accepted swap.
* Runs use CUDA by default and refuse a silent CPU fallback; name another platform with
  `MD_PLATFORM`.
* Multi-GPU REST2 propagates device groups concurrently and replicas sharing a device in turn.

**Packaging and tooling.**

* `md-template install` installs and validates the whole preparation stack — OpenFF, AmberTools,
  openmmforcefields, ParmEd, RDKit — not OpenMM alone, and `--validate PREFIX` checks an
  environment it did not create.
* One release-only CI workflow, triggered by `workflow_dispatch` and an `openmm-v*` tag.

**Migration.** There is no migration path from the pre-`ca29fcd` registry/bundle interface; it was
removed rather than deprecated. Prepare systems again with `md-openmm sys-config` and `sys-gen`.
Existing run directories produced by `openmm-v0.1.0` should be restarted rather than resumed: the
REST2 segment semantics and the equilibration sequence both changed, so a continued run would not
be one trajectory.

## Unreleased (pre-0.2.0, historical)

### Packaged template catalog and build provenance (metadata only)

The catalog introduced in the previous entry was usable only from a repository checkout. It now
travels inside the built distribution, and an installed copy can resolve a real immutable identity.
**No runtime behaviour changes**; `md-openmm` and every simulation path are untouched.

* `md_templates.core.load_packaged_catalog()` and `resolve_packaged_identity(template_ref)` read the
  catalog built into the installation — no repository root, no Git, no working-directory assumption
  and no network. `load_catalog(root)` is unchanged for checkouts.
* The wheel carries the exact canonical `registry.yaml` bytes, every registered descriptor,
  `resource_manifest.json` (schema version 1) and `build_provenance.json` (schema version 1). Both
  schema versions are **independent** of the registry, template, bundle, canonical-configuration and
  run-state versions.
* **Single source of truth.** The tracked root `registry.yaml` and `templates/**/template.yaml`
  remain the only editable copy; the packaged copy is generated into the build tree by a `build_py`
  hook and is never written back into the working tree.
* **Provenance is decided from the source being built**, never supplied: a clean checkout contributes
  its exact full `HEAD`; an unmodified sdist from a clean checkout inherits that commit; a dirty
  tree, a failed `git status`, or a tree with no Git and no verified archive record all record an
  unresolved state and make identity resolution raise `UnresolvedBuildProvenanceError`. There is no
  environment-variable override and no "allow dirty but trust HEAD" option.
* Build metadata is deterministic — no build time, hostname, user, absolute path or branch name — so
  two builds of one commit produce byte-identical records.
* Identities always use the **logical repository path**; a wheel-internal path can never appear in
  one.
* `repository_references` keep their source-tree existence check in a checkout. In a wheel they are
  recorded as `validated-at-build`, because those paths name documentation that does not ship; the
  catalog reports its `reference_policy` rather than silently skipping the check.
* `pyyaml` and `pydantic` are now **build** requirements as well as runtime ones: the build validates
  the catalog with the same strict loader it ships.
* Git provenance is **never inherited from an enclosing repository**: the detected worktree root must
  be the source root itself, so a tree unpacked inside an unrelated (and clean) repository reports no
  Git metadata rather than that repository's HEAD.
* Archive provenance binds **every regular file in the source archive** via `source_tree_sha256`,
  keyed by normalised logical path — closed-world, excluding only generated artifacts
  (`__pycache__`, `*.pyc`, `*.egg-info/`, top-level `build/` and `dist/`, `.git/`) and the provenance
  record itself. An unpacked sdist edited anywhere, including README, build configuration or a
  referenced document, cannot inherit the original commit.
* `scripts/ci/check_packaged_catalog.py`, called by `scripts/ci/fast_checks.sh`, exercises all of
  this against the installed wheel from outside the checkout with sockets blocked. It **requires
  resolved provenance, and `--expect-commit` is mandatory** in strict mode and must match the
  checkout's HEAD. `scripts/ci/expect_commit.sh` acquires that SHA before anything is built and fails
  loudly on an empty or malformed value; `--allow-unresolved`
  (`FAST_CHECKS_ALLOW_UNRESOLVED=1`) is a local-development mode that must be selected explicitly.

**Migration:** none. No bundle, run directory, manifest or hash gained a template identity.

### Template catalog and immutable template identity (metadata only)

First step of the multi-method migration. **No runtime behaviour changes.** `md-openmm` still
executes through `md_templates.openmm`; nothing dispatches through the catalog, and every descriptor
records `implementation.dispatch: legacy-direct` to say so in the file itself.

* `registry.yaml` at the repository root — a discovery index over the templates, with
  `schema_version: 1`, **independent** of the bundle, canonical-configuration and run-state versions.
* `templates/conventional-md/openmm/explicit-water/template.yaml` and
  `templates/rest2/openmm/explicit-water/template.yaml` — strict typed descriptors covering method
  and engine identity, input routes, operational capabilities, readable/writable bundle schema
  versions, restart guarantees, the current implementation binding, and implementation status and
  scientific status as **separate** fields.
* `md_templates.core` — engine-neutral registry, descriptor, identity and path modules. They import
  YAML and pydantic and nothing else, so the catalog can be listed and validated in an environment
  with no OpenMM, OpenFF or RDKit.
* Immutable template identity is `(canonical repository URL, full 40-character Git commit SHA,
  exact normalised template path)`, canonically
  `https://github.com/csy0000/MD-templates@<sha>#templates/<method>/<engine>/<variant>/template.yaml`.
  Branch names, tags, abbreviated SHAs, package versions, template semantic versions and timestamps
  are all refused. Resolution **proves** provenance: the commit is never supplied by the caller, a
  **dirty checkout raises** rather than resolving to HEAD, trusted provenance must equal HEAD inside
  a checkout, and outside one only explicitly trusted build provenance is accepted. Resolution reads
  local Git metadata only and makes no network request.
* Catalog loading **refuses symlinked paths** — descriptor, `repository_references` and
  `registry.yaml` alike. A symlink can point at mutable bytes outside the checkout while Git reports
  the tree clean, which would let a commit SHA name content that commit does not contain.
* Compatibility goldens under `tests/goldens/`, captured before any structural edit and regenerated
  only by the explicit `scripts/capture_goldens.py`.

**Migration:** none. No bundle, run directory, manifest or hash gained a template identity, and no
profile, default or schema moved.

### Bundle schema version 2

New bundles are written with `bundle_schema_version: 2`, a version **independent** of the system
manifest, experiment manifest, canonical-configuration and run-state versions.

Added to every new bundle:

* `checksums.json` — normalised relative POSIX paths to SHA-256 of file **bytes**, over an
  explicitly stated domain (excluding itself and `bundle_manifest.json`, with the reason recorded);
* `original_inputs/` — the exact configuration document, the exact PDB on the peptide route, an
  identity record on the SMILES route, and the legacy manifests when that front end was used;
* `forcefield_provenance.json` — resolved resources, their hashes where readable, charge method,
  toolkit versions and combination assumptions;
* `environment.json` — versions classified as required-for-exact-rebuild,
  required-for-supported-execution, or informational;
* `topology.cif` — an mmCIF topology alongside the PDB, since PDB cannot round-trip a triclinic
  cell faithfully;
* `canonical_configuration.json`, `resolved_runtime_config.json`;
* separated counts: `topology_atoms`, `openmm_particles`, `virtual_sites`, `massless_particles`,
  `constraints`, and the DOF formula.

**Migration:** none required. Version-1 bundles remain readable and runnable through the
compatibility path. `md-openmm bundle validate` reports them as `v1-compatibility` and lists the
guarantees they do not provide. To obtain the version-2 guarantees, re-prepare.

**Breaking within the manifest:** the conflated `n_atoms` count is replaced by the separate counts
above. Anything reading `counts.n_atoms` must choose which number it meant.

### Canonical configuration drives runs

* `md-openmm prepare --config` supports both declared routes, `smiles` and `pdb`. The PDB path
  resolves relative to the **configuration document**, not the working directory, and its declared
  hash is verified.
* `md-openmm md` and `md-openmm rest2` accept `--config` and `--set`. With neither, a version-2
  bundle's pinned canonical configuration is used.
* Before a run directory is created: the method discriminator, the system/build hash and the
  prepared-state hash must all match the bundle. Extension-only changes such as `n_chunks` are
  permitted.
* `--config` against a version-1 bundle is refused rather than ignored.

### Explicit seeds

* New canonical `randomness` section: `master_seed` plus optional per-stage overrides.
* The legacy derivation is preserved exactly (`master + offset`, structure 0, equilibration 1,
  md 2, rest2 3), so migrated inputs reproduce their existing trajectories.
* What is hashed is the **resolved stage seed**, not the master label: structure and equilibration
  seeds enter the prepared-state projection, the production seed enters run continuity.
* **Migration:** a legacy `master_seed` is carried into `randomness.master_seed` by
  `md-openmm config migrate`.

### Projection reclassification

`prepared_state_sha256` now covers the equilibration block, the integrator used to reach the stored
state, and the preparation seeds, alongside system and build. Classification follows scientific
consequence rather than model section.

**Consequence:** a configuration that changes equilibration is no longer accepted against an
existing bundle. This is deliberate — the stored starting state would be wrong.

### New commands

```
md-openmm bundle validate BUNDLE [--deep]
md-openmm bundle inspect  BUNDLE [--format text|json|yaml]
md-openmm bundle relocate-check BUNDLE
```

`validate-bundle --bundle B` is retained as a documented compatibility alias.

### CI

`fast` and `integration-cpu` GitHub Actions workflows, both CPU-only, calling
`scripts/ci/fast_checks.sh` and `scripts/ci/integration_cpu.sh` so local and CI behaviour cannot
drift. `environment-ci.yml` is the documented environment without the CUDA pin.
