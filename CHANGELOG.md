# Changelog

Entries describe behaviour changes and the migration each one needs. No release has been published.

## Unreleased

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
