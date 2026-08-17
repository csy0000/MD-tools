# Claude Code instruction: multi-method migration PR 1 — catalog and immutable identity contracts

Work from the openmm branch of csy0000/MD-templates.

Read and follow CLAUDE.md first. Then inspect:

- README.md
- pyproject.toml
- registry/package-resource conventions already present in the repository
- src/md_templates/openmm/spec/
- src/md_templates/openmm/schemas.py
- src/md_templates/openmm/provenance.py
- src/md_templates/openmm/bundle.py
- src/md_templates/openmm/bundlev2.py
- src/md_templates/openmm/runstate.py
- all current tests
- docs/journal/2026-08-16_portable-simulation-config.md
- docs/journal/2026-08-16_rest2-verification-and-instruction-provenance.md
- docs/journal/2026-08-17_portable-bundle-and-ci.md

The compatibility baseline inspected for this instruction is commit:

    1e80a2061613ac40a10143c28044c661e1d5d4ac

If the branch has advanced, inspect every intervening commit before editing. Preserve later valid work rather than resetting to this SHA.

This is the first small PR in the approved migration from an OpenMM-specific repository layout to a reusable multi-method, multi-engine template repository. Implement PR 1 only. Stop before PR 2.

## Approved long-term boundary

The final architecture uses method/engine template paths under templates/, for example:

    templates/conventional-md/openmm/explicit-water/template.yaml
    templates/rest2/openmm/explicit-water/template.yaml

Reusable OpenMM implementation code will later be centralized under:

    src/md_templates/engines/openmm/

Engine-neutral catalog, identity, configuration, bundle and persistence primitives may later live under:

    src/md_templates/core/

Do not perform the OpenMM namespace migration in this PR. The current implementation under src/md_templates/openmm/ remains the runtime implementation and compatibility baseline.

## Goal of PR 1

Introduce and test only the contracts needed to identify and describe templates:

1. a root registry.yaml;
2. one template.yaml for conventional MD in explicit water with OpenMM;
3. one template.yaml for REST2 in explicit water with OpenMM;
4. strict typed validation for both descriptor formats;
5. immutable template identity resolution using full Git commit SHA plus exact template path;
6. compact compatibility golden fixtures that freeze current configuration and persistent-contract behavior;
7. a dated implementation journal.

This PR creates catalog metadata and validation. It must not activate generic execution dispatch or change simulation behavior.

## Non-goals and forbidden scope

Do not do any of the following in this PR:

- move, rename or delete modules under src/md_templates/openmm/;
- create src/md_templates/engines/openmm/ implementation copies;
- change md-openmm command behavior or add the general execution CLI;
- route prepare, MD or REST2 through the new descriptors;
- change canonical configuration models, resolution precedence or hash projections;
- change profile contents, profile IDs, default selection or scientific defaults;
- move or duplicate packaged profiles, systems, experiments or evidence;
- change bundle schema version 1 or 2;
- add template identity to existing bundles, run manifests or continuity hashes;
- change restart, checkpoint, committed-generation or uncommitted-tail behavior;
- change REST2 ladder values, exchange behavior, omega classification or the default omega-exclusion setting;
- change master-seed or stage-seed derivation;
- implement an Amber-style input adapter;
- claim new scientific validation;
- remove legacy compatibility paths;
- publish a release.

If a seemingly necessary change crosses one of these boundaries, document it as deferred and stop rather than widening the PR.

## Registry contract

Add repository-root registry.yaml with schema_version 1.

Keep the registry a minimal discovery index. Do not duplicate full template settings, profiles, scientific defaults or validation evidence there.

Required conceptual fields:

~~~yaml
schema_version: 1

repository:
  canonical_url: https://github.com/csy0000/MD-templates

identity:
  scheme: git-commit-plus-template-path
  canonical_form: "{canonical_url}@{full_commit_sha}#{template_path}"

templates:
  - template_id: conventional-md/openmm/explicit-water
    template_path: templates/conventional-md/openmm/explicit-water/template.yaml
    method: conventional-md
    method_aliases: [md]
    engine: openmm
    tags: [explicit-water, portable-bundle, restartable]

  - template_id: rest2/openmm/explicit-water
    template_path: templates/rest2/openmm/explicit-water/template.yaml
    method: rest2
    engine: openmm
    tags: [explicit-water, replica-exchange, portable-bundle, restartable]
~~~

Names may be adjusted only when needed for a coherent typed implementation. Document any deviation.

Required validation:

- reject unknown keys at every schema level;
- require an integer schema version and reject unsupported versions;
- require exactly one canonical repository URL;
- reject duplicate template IDs and duplicate template paths;
- require normalized repository-relative POSIX paths;
- reject absolute paths, empty components, dot components and parent traversal;
- require every template path to remain under templates/;
- reject Windows separators and platform-dependent normalization;
- require the template path to exist;
- require template_id to equal the descriptor directory relative to templates/;
- require path components to match the declared method and engine;
- validate aliases and tags as normalized nonempty identifiers;
- produce errors that include the complete dotted field path and invalid value where practical.

Do not store the current commit SHA in registry.yaml. The same registry content must be usable at later commits, with identity resolved from the immutable commit containing it.

## Template descriptor contract

Add:

    templates/conventional-md/openmm/explicit-water/template.yaml
    templates/rest2/openmm/explicit-water/template.yaml

Both use template schema_version 1 and the same strict model.

The descriptor must distinguish at least:

- stable logical template ID;
- display name and summary;
- method identity and method API version;
- engine identity and engine API version;
- identity mode;
- current implementation binding;
- supported input routes;
- operational capabilities;
- readable and writable bundle schema versions;
- restart/continuation guarantees;
- compatibility commands and Python namespaces;
- implementation-validation status;
- scientific-validation status and limitations;
- examples or evidence only when their scope is explicit.

PR 1 is catalog-only. The descriptors may bind to the current md_templates.openmm implementation, but they must clearly state that generic template dispatch is not active yet. Do not invent non-existent entry points.

Do not move or duplicate current profile files in order to make the descriptors appear complete. If profile resources are represented, use a clearly named transitional/current-provider reference that does not pretend the files are template-local. Template-local profile migration is deferred to later method activation PRs.

Scientific status must be honest:

- implementation smoke and regression testing is not scientific validation;
- the general conventional-MD template is not yet production scientifically validated merely because it runs;
- the general REST2 template remains scientifically unvalidated;
- the RGD ten-rung result is system-specific pilot-supported evidence, not general REST2 validation;
- the macrocycle eight-rung example remains unvalidated;
- CPU smoke profiles are prohibited as scientific evidence;
- the 2 fs versus 4 fs HMR equivalence gate remains open;
- arbitrary REST2 ladder suitability remains open.

The REST2 descriptor must record that omega exclusion is supported and enabled by default, while remaining explicitly disableable. This metadata must not modify runtime behavior.

Required descriptor/path consistency checks:

- template_id matches the directory below templates/;
- method ID matches the first path component below templates/;
- engine ID matches the second path component;
- descriptor filenames are exactly template.yaml;
- all repository-relative references are normalized and non-escaping;
- unknown fields are rejected;
- declared status values come from documented enums;
- schema version is independent of current bundle, configuration and run-state schema versions.

## Immutable template identity

Treat the immutable identity as the tuple:

    canonical repository URL
    full 40-character Git commit SHA
    exact normalized template.yaml path

The canonical string form is:

    https://github.com/csy0000/MD-templates@FULL_SHA#templates/METHOD/ENGINE/VARIANT/template.yaml

Requirements:

- accept only a full 40-character hexadecimal commit SHA;
- normalize hexadecimal SHA text consistently;
- never use a branch name, tag, abbreviated SHA, package version, template semantic version or timestamp as immutable identity;
- never hardcode the current SHA into registry.yaml or template.yaml;
- the same SHA and path resolve to the same identity;
- a different SHA with the same path is a different identity;
- the same SHA with a different path is a different identity;
- path normalization must occur before identity construction;
- refuse absolute, escaping or non-normalized paths;
- verify that the requested path is a registered template;
- preserve structured identity fields as well as the canonical string;
- do not add this identity to current scientific/configuration hashes in PR 1.

Dirty source checkouts:

- A dirty checkout may be inspected for development, but must not claim a resolved immutable template identity.
- Return an explicit unresolved/dirty result or raise a clear typed error.
- Do not silently use HEAD while ignoring uncommitted changes.
- Tests must not depend on the developer's actual working tree. Create isolated temporary Git repositories.

Non-Git/installed contexts:

- Design the resolver so a future wheel can supply trusted build provenance explicitly.
- PR 1 does not need to package the catalog or implement wheel build provenance; that is PR 2.
- Without Git metadata or explicit trusted full-SHA provenance, do not fabricate an immutable identity.

Keep Git command execution narrow, non-shell, timeout-bounded and testable. Do not make validation depend on network access or GitHub availability.

## Location and dependency discipline

Use a small engine-neutral module area for the new catalog contracts. A suitable shape is:

    src/md_templates/core/
      __init__.py
      identity.py
      registry.py
      template.py

A small catalog subpackage is also acceptable if it produces a cleaner dependency boundary. Explain the choice in the journal.

Requirements:

- catalog loading and descriptor validation must not import OpenMM, OpenFF, RDKit or other engine dependencies;
- listing/validation must be usable in a minimal Python environment containing the package's schema dependencies;
- reuse the existing safe YAML loader and strict Pydantic conventions where possible;
- do not add a new dependency unless unavoidable and documented;
- do not implement a general engine plugin framework in PR 1;
- keep error types specific and actionable;
- keep public models and identity semantics documented.

## Compatibility golden fixtures

Before structural edits, capture compact deterministic fixtures from the baseline behavior. Do not commit trajectories, checkpoints, System XML, large molecular artifacts, environments or generated run directories.

Goldens should cover the current contracts that later PRs must preserve:

1. canonical JSON and all current configuration projection hashes for representative:
   - conventional MD ligand/SMILES;
   - conventional MD peptide/PDB;
   - REST2 ligand/SMILES;
   - REST2 peptide/PDB;
2. selected default profile ID and profile hash for each supported method/route;
3. YAML/JSON canonical equivalence;
4. current master-seed to stage-seed derivation;
5. representative bundle-v2 manifest field structure and normalized logical roles;
6. current run continuity projection and committed-generation record structure;
7. REST2 defaults relevant to compatibility, including ladder and omega exclusion.

Use existing shipped examples where they already provide representative inputs. Store only stable, reviewable JSON/YAML/text fixtures. When exact bytes are platform-dependent, freeze semantic projections or independently calculated invariants rather than unstable artifacts.

The golden-generation procedure must be documented and deterministic. Tests normally consume committed fixtures; they must not silently regenerate expected files during ordinary test runs.

If existing tests already freeze a contract adequately, reference and preserve them instead of adding redundant large snapshots.

## Tests required

Add focused tests for at least:

### Registry

1. valid registry loads;
2. unknown root and nested fields fail;
3. unsupported schema version fails;
4. duplicate template ID fails;
5. duplicate template path fails;
6. missing descriptor fails;
7. absolute path fails;
8. parent traversal fails;
9. dot or non-normalized path fails;
10. Windows separator fails;
11. template ID/path mismatch fails;
12. method/path mismatch fails;
13. engine/path mismatch fails.

### Template descriptors

14. both shipped descriptors validate;
15. unknown fields fail at every modeled level;
16. unsupported template schema version fails;
17. invalid status enum fails;
18. repository-relative reference escape fails;
19. REST2 declares omega exclusion enabled by default;
20. smoke evidence cannot be marked scientifically validated;
21. RGD evidence cannot elevate the general REST2 status.

### Identity

22. same SHA and path is deterministic;
23. different SHA changes identity;
24. different template path changes identity;
25. abbreviated SHA fails;
26. non-hex SHA fails;
27. unregistered template path fails;
28. dirty temporary Git checkout cannot resolve immutable identity;
29. clean temporary Git checkout resolves its full SHA;
30. non-Git directory without explicit provenance fails clearly;
31. identity resolution makes no network request.

### Compatibility

32. every committed golden matches current behavior;
33. canonical hashes remain exactly unchanged;
34. profile selection and profile hashes remain unchanged;
35. existing bundle/restart contract tests still pass;
36. importing catalog modules does not import OpenMM.

Tests must use temporary directories for destructive or Git-state cases. Do not alter the real checkout to test dirty-state detection.

## Acceptance gates

PR 1 is complete only when all of the following are true:

- registry.yaml and both template descriptors validate;
- the strict negative tests pass;
- immutable identity tests pass in isolated temporary Git repositories;
- every captured compatibility golden matches the current implementation;
- the complete existing non-slow suite passes with no removed or weakened cases;
- focused current bundle and restart tests pass;
- no source file under src/md_templates/openmm/ was moved, renamed or behaviorally changed;
- md-openmm behavior and help remain unchanged;
- canonical bytes and hashes remain unchanged;
- profile contents/defaults/hashes remain unchanged;
- bundle and run persistent schemas remain unchanged;
- no new template metadata enters scientific or continuity hashes;
- the diff contains no unrelated cleanup;
- no generated simulation data or secrets are committed.

If a long OpenMM integration environment is unavailable, run every environment-independent test and clearly report the exact missing gate. Do not label PR 1 complete until the repository's supported full gate has been run.

## Documentation and journal

Add:

    docs/journal/2026-08-17_multi-method-migration-pr1.md

The journal must contain:

- request and exact scope;
- baseline and final commit;
- files changed;
- final registry and descriptor model decisions;
- identity algorithm and dirty/non-Git behavior;
- compatibility goldens and how they were generated;
- confirmation that template identity is not yet written into bundles/runs;
- confirmation that runtime dispatch remains legacy/current;
- exact test commands, counts, durations and results;
- checks not run and why;
- scientific-status wording;
- deviations from this instruction;
- deferred PR 2 work;
- known limitations.

Do not claim remote CI passed unless a GitHub Actions run was actually observed.

## Required completion report

Before finishing, inspect git status and the complete diff. Report:

1. concise implementation summary;
2. complete changed-file list;
3. exact test commands and results;
4. captured golden fixture names and key hashes;
5. proof that no existing canonical hash/default changed;
6. proof that no runtime module was moved;
7. schema and identity design decisions;
8. compatibility and scientific consequences;
9. journal path;
10. limitations and the recommended PR 2 instruction.

Use focused commits. Do not proceed to catalog packaging, a generic CLI, OpenMM namespace relocation, method activation or artifact identity persistence. Stop after PR 1.
