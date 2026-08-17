# PR 2 — Package the template catalog and carry verified build provenance

Date: 2026-08-17  
Target repository: `csy0000/MD-templates`  
Base branch: `openmm`  
Suggested branch: `migration/pr2-packaged-catalog-provenance`  
Suggested PR title: `Package the template catalog with verified build provenance`

## Purpose

Implement the second small PR in the approved multi-method, multi-engine migration.

PR 1 introduced the engine-neutral catalog and immutable identity contract, but deliberately left the catalog usable only from a repository checkout. PR 2 must make that same catalog available from an installed wheel and allow an installed copy to resolve:

```
canonical repository URL + full Git commit SHA + exact logical template path
```

This remains a catalog and provenance change only. Do not activate generic dispatch, move the OpenMM implementation, or alter simulation behavior.

Read completely before editing:

1. `CLAUDE.md`;
2. `docs/journal/2026-08-17_multi-method-migration-pr1.md`;
3. `registry.yaml`;
4. both current `templates/**/template.yaml` descriptors;
5. `src/md_templates/core/{registry,template,identity,paths}.py`;
6. the catalog, identity, compatibility-golden, wheel, and CI tests;
7. `pyproject.toml` and the current build configuration.

Inspect `git status` first and preserve unrelated work.

## Non-negotiable invariants

1. The tracked root `registry.yaml` and tracked root `templates/<method>/<engine>/<variant>/template.yaml` files remain the canonical source and canonical Git paths.
2. Do not keep a second hand-maintained tracked copy of the catalog under `src/`. There must be one editable source of truth.
3. The logical identity path remains the root path recorded in the registry, for example:
   `templates/rest2/openmm/explicit-water/template.yaml`.
   A wheel-internal storage path must never replace it in an identity.
4. A clean source checkout may contribute its exact full `HEAD` SHA as trusted build provenance.
5. A dirty checkout must never stamp `HEAD`, a branch, a tag, a package version, a timestamp, `GITHUB_SHA`, or any caller-provided value as a resolved template identity.
6. A wheel built from dirty source may still be produced, but it must carry explicit unresolved provenance and packaged identity resolution must raise a typed, actionable error.
7. Provenance generation must be deterministic: no build time, hostname, username, absolute source path, branch name, or other machine-local value.
8. Loading and identity resolution must remain offline and engine-neutral. They may depend on the standard library, PyYAML, and Pydantic, but must not import OpenMM, OpenFF, RDKit, MDTraj, NumPy, SciPy, or the OpenMM implementation.
9. Preserve the current symlink and path-containment protections. Packaging must not create a way to validate mutable bytes outside the source tree.
10. Do not write template identity into bundles, run manifests, configuration hashes, continuity hashes, or scientific hashes in this PR.
11. Do not change `md-openmm`, its help text, runtime dispatch, profiles, scientific defaults, bundle formats, restart formats, seeds, REST2 behavior, or omega-exclusion behavior.
12. Do not move `src/md_templates/openmm/` yet.

## Required implementation

### A. Package the catalog without tracked duplication

Make the built wheel contain, at minimum:

- the exact canonical `registry.yaml` bytes;
- every descriptor named by that registry;
- a versioned resource-integrity manifest for the packaged catalog;
- the versioned build-provenance record described below.

Use a build-time staging/copy mechanism or an equivalently safe design. Generated resource copies may exist in the wheel/build staging area, but must not become a second tracked source tree.

Before copying resources, validate the source catalog with the existing strict model and source-tree checks. A build must fail clearly if:

- the registry is invalid;
- a registered descriptor is absent;
- a path is non-normalized or escapes the repository;
- a relevant source path is a symlink;
- registry and descriptor identities disagree;
- duplicate IDs or paths exist.

The resource-integrity manifest must:

- have its own integer schema version;
- use normalized logical repository paths;
- include SHA-256 for the exact packaged registry and descriptor bytes;
- be stable for identical inputs;
- contain no absolute paths or timestamps;
- be verified before packaged catalog bytes are trusted.

Do not silently weaken `repository_references` validation. In a source checkout, continue checking that each reference exists under the repository root. For an installed catalog, implement an explicit installed-resource policy: the build must validate references against the source commit before packaging, and the installed loader must distinguish “validated at build time” from “present in this wheel.” Do not pretend checkout-only documentation paths exist in site-packages. Record the policy in code and documentation.

### B. Add an installed-catalog API

Keep `load_catalog(root)` backward-compatible for repository checkouts.

Add a clearly named public API for loading the built-in packaged catalog without a checkout or caller-supplied repository root. Choose names that make the provenance boundary obvious; for example, a pair equivalent to:

- `load_packaged_catalog()`;
- `resolve_packaged_identity(template_ref)`.

The exact names may differ if a better small API fits the current design, but document the choice.

The packaged loader must:

- use standard package-resource facilities rather than `cwd`;
- work after installing the wheel into a clean environment and running from an unrelated directory;
- validate the registry schema, descriptor schemas, registry/descriptor agreement, normalized logical paths, uniqueness, and resource hashes;
- return the same two template IDs, methods, engines, aliases, features, statuses, and logical template paths as the source loader;
- fail with typed errors for missing, changed, extra-claimed, or hash-mismatched catalog resources;
- not require Git, a repository checkout, GitHub, or network access.

Do not make a caller pass a SHA to the normal packaged-catalog API.

### C. Generate trustworthy build provenance

Add a small, strict, independently versioned provenance record inside the built distribution. It must state at least:

- provenance schema version;
- whether immutable source provenance is resolved;
- the full lowercase 40-character commit SHA when and only when resolved;
- a finite machine-readable source state/reason such as clean Git checkout, dirty source tree, or no verifiable Git provenance;
- the catalog resource-integrity manifest hash or equivalent binding between provenance and the exact packaged catalog bytes.

The build process must decide provenance from the source being built:

- In a Git checkout, require a successful full-`HEAD` query and proof that the tree is clean before recording the SHA.
- Treat a failed or timed-out status check as unresolved, never clean.
- Count untracked non-ignored files as dirty when they are part of source state.
- Do not use an environment variable as an override.
- Do not expose an “allow dirty but trust HEAD” option.
- Do not write generated provenance into tracked source files.
- Do not leave the working tree modified after a build, apart from normal ignored build artifacts.

If supporting the standard `sdist -> wheel` route, preserve clean provenance through an unmodified sdist using versioned generated metadata plus resource hashes. A wheel made from a modified or unverifiable source archive must not silently claim more than can be proved. Document the exact trust boundary and do not claim cryptographic authentication of GitHub or of a maliciously modified artifact.

### D. Connect packaged identity resolution

For an installed wheel:

1. load and integrity-check the packaged catalog;
2. load and strictly validate packaged build provenance;
3. refuse unresolved provenance with `NoProvenanceError` or a more specific subclass;
4. bind the trusted commit to the validated packaged catalog;
5. resolve the selected registered template;
6. return the existing `TemplateIdentity` using the canonical repository URL, verified full commit SHA, and logical root template path.

Do not let an arbitrary caller-provided `TrustedProvenance` override the packaged record through this convenience API.

A clean source checkout must retain the current behavior: the current clean `HEAD` is authoritative, and any trusted metadata is only an equality cross-check.

### E. Update build and installed-wheel gates

Extend `scripts/ci/fast_checks.sh` and any relevant existing CI wrapper so the supported gate inspects and exercises the packaged catalog. It must build and install the wheel, leave the checkout, and then verify at least:

- catalog resources and both metadata records exist in the wheel;
- the packaged catalog loads without a repository checkout;
- both registered templates are listed;
- both identities resolve to the exact clean build commit;
- canonical identity strings retain root `templates/...` paths;
- resource and provenance schemas are valid;
- catalog loading and identity resolution work with network access blocked;
- importing/using the catalog does not load heavy scientific dependencies.

Do not redesign unrelated CI in this PR. If no GitHub Actions workflow is configured or observable, report that honestly; local scripts are evidence, not proof of a remote passing run.

## Required regression and acceptance tests

Add tests that fail against the merged PR 1 base and pass after this work.

### 1. Source and wheel equivalence

From a clean temporary Git checkout:

- load the source catalog;
- build a wheel;
- install it into a clean environment;
- run from a directory with no checkout on `sys.path`;
- load the packaged catalog;
- compare a canonical serialized projection of every registry entry and descriptor;
- assert the source and packaged projections are identical.

### 2. Clean build identity

Build from a clean committed temporary checkout and assert:

- provenance is resolved;
- the recorded SHA is the exact 40-character `HEAD`;
- both packaged template identities use that SHA;
- identities use the registry’s logical root template paths;
- no branch, version, timestamp, or wheel path appears in the identity.

The clean test must build after the relevant files are committed, or from a fresh clean clone/worktree. Building while implementation edits are uncommitted is expected to produce unresolved provenance and is not a valid clean-build test.

### 3. Dirty build refusal

Modify a tracked catalog or implementation file without committing, build a wheel, install it, and assert:

- the build does not claim resolved provenance;
- the dirty `HEAD` is not accepted as trusted identity;
- catalog listing may work if resource integrity is valid;
- packaged identity resolution fails with a typed error explaining that the source was dirty;
- passing the same `HEAD` through a convenience argument cannot bypass the refusal.

Also test an untracked relevant source file and a failed Git-status check.

### 4. No-Git installed operation

After installation, remove access to the source checkout and ensure:

- packaged catalog loading succeeds;
- clean-build identity resolution succeeds from packaged provenance;
- no `.git` directory is needed;
- no current-working-directory assumption exists.

### 5. Tamper and truncation detection

Create controlled installed-resource copies and independently test:

- changed registry bytes;
- changed descriptor bytes;
- missing descriptor;
- missing provenance;
- malformed provenance;
- unsupported provenance schema version;
- malformed or abbreviated SHA;
- resource manifest path traversal;
- resource hash mismatch.

Every case must fail before returning a resolved identity.

### 6. No-network and dependency boundary

In a subprocess with socket creation/name resolution replaced by raising stubs:

- load the packaged catalog;
- resolve a clean packaged identity.

Assert that heavy scientific modules are absent from `sys.modules`.

### 7. Compatibility

Assert that this PR does not change:

- all seven committed compatibility goldens;
- existing profile selection and hashes;
- configuration projections and hashes;
- seed derivation;
- bundle and run-state contracts;
- REST2 defaults;
- the existing `md-openmm` public help text;
- runtime imports under `src/md_templates/openmm/`;
- bundle/run manifests and continuity/hash paths.

Run the existing focused catalog/identity suite, compatibility-golden checks, the complete non-slow suite, the wheel-based fast gate, and the relevant CPU integration gate. Report exact commands, counts, durations, failures, reruns, and environment limitations. Do not call smoke testing scientific validation.

## Documentation

Update the smallest relevant public documentation, including:

- how to enumerate templates from an installed wheel;
- how immutable packaged identity is formed;
- the difference between clean, dirty, and unverifiable build provenance;
- the resource-integrity and trust boundaries;
- that the catalog still does not dispatch simulations;
- that identity is not yet persisted into bundles or runs.

Add a dated journal:

`docs/journal/2026-08-17_multi-method-migration-pr2.md`

The journal must include:

- scope and explicit non-goals;
- design and provenance state machine;
- tracked source versus generated wheel resource layout;
- schemas added;
- public APIs added;
- exact tests and results;
- source/wheel equivalence evidence;
- dirty-build evidence;
- installed/no-Git/offline evidence;
- compatibility results;
- CI status actually observed;
- limitations and deferred work.

## Commit and PR discipline

Use small reviewable commits. A reasonable sequence is:

1. tests defining the packaged-resource and provenance contracts;
2. build staging and deterministic resource manifest;
3. build provenance and installed-catalog APIs;
4. wheel/CI gate updates;
5. documentation and journal.

Do not regenerate compatibility goldens merely to make a failure pass. Investigate any difference and keep the baseline unchanged unless the user explicitly approves a contract change.

Inspect the final diff for unrelated edits, generated artifacts, credentials, large files, and duplicate catalog sources.

Push the branch and open the PR, but do not merge it. The PR description must state:

- the exact base and head SHAs;
- how single-source catalog packaging works;
- the clean/dirty/no-provenance rules;
- source-to-wheel equivalence results;
- exact test and gate results;
- runtime/configuration/hash compatibility;
- scientific-validation limitations;
- known limitations and deferred work.

## Deferred beyond PR 2

Do not implement these here:

- generic template dispatch or a new general CLI;
- routing `prepare`, `md`, or `rest2` through descriptors;
- moving OpenMM code to `src/md_templates/engines/openmm/`;
- template-local profiles;
- writing template identity into bundles or run manifests;
- changing bundle or restart schema;
- adding another simulation engine or method;
- the Amber-style input adapter;
- scientific validation claims.

Stop and ask before expanding into any deferred item or changing a scientific/runtime default.
