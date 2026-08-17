# Claude Code instruction: PR3–PR8 single migration campaign

Date: 2026-08-17  
Repository: `csy0000/MD-templates`  
Base: `openmm` at `b2b35fea38d2d30cecfdb819f967e5e8ee5cda5f` (merged PR2)  
Suggested branch: `migration/pr3-pr8-reusable-template-platform`

## Mission and workflow

Complete the approved PR3–PR8 migration continuously on one branch and open one final PR against `openmm`. Keep at least one focused, reviewable commit per phase and do not pause merely because a phase finishes.

Before editing, read `CLAUDE.md`, the PR1/PR2 journals, registry/catalog and packaging code, every `template.yaml`, current core/OpenMM/CLI/profile/bundle/persistence code, tests, workflows, goldens, and user-facing support docs. Repository behavior and committed artifacts are the source of truth; a stricter `CLAUDE.md` invariant wins and must be journaled.

Confirm the exact base above. Execute Phases 0, 3, 4, 5, 6, 7, and 8 in order; begin the next only after its gate passes. Maintain `docs/journal/2026-08-17_multi-method-migration-pr3-pr8.md`, recording per phase: files/interfaces, compatibility decisions, exact tests/results, compared artifacts/hashes, risks, and gate status. Continue without routine confirmation. Stop only for a contradiction requiring scientific-default or canonical-hash changes, incompatible artifact migration, destructive data handling, or authority outside scope. Push and open one final PR; do not merge.

## Invariants

- Do not change canonical scientific bytes/hash inputs or existing golden hashes. Any unavoidable schema evolution must be versioned with old readers retained.
- Do not change force fields, integrators, thermodynamic settings, seeds, chunks, omega semantics, REST2 ladders, restraints, timesteps, HMR, or other scientific defaults.
- Existing v1/v2 bundles remain readable and committed runs resumable under established committed-generation rules.
- Template/non-scientific metadata stays outside scientific hashes.
- General conventional-MD and REST2 remain scientifically unvalidated; RGD remains system-specific pilot-supported. CPU smoke is engineering evidence only.
- Core must not import OpenMM, OpenFF, RDKit, or another heavy engine dependency at import time.
- Identity is exactly full Git SHA plus exact logical repository path to `template.yaml`, never a short SHA or installed path.
- Fail explicitly on ambiguity, incompatible artifacts, unavailable requested platform/device, or incomplete committed state. Never silently fall back.
- Keep one authoritative source for each profile, catalog asset, and engine implementation.

## Phase 0 — characterize and freeze

- Run all existing non-slow and CI-equivalent checks. PR2’s reference was 527 passed, 17 deselected, and 7/7 goldens; explain differences.
- Record hashes for all seven committed goldens; capture public imports and `md-openmm` help/arguments.
- Select representative legacy v1/v2 bundle and committed-run read/resume fixtures.
- Add characterization tests for relied-upon behavior not locked down.
- Build/install a wheel outside checkout and record catalog/list/inspect/identity behavior.
- Map old modules/tests to planned destinations so movement cannot hide coverage loss.

Gate: baseline/hashes recorded and compatibility protected before structural edits.

## Phase 3 — engine-neutral core

Put genuinely engine-neutral typed config/resolution, precedence/source attribution, engine-free value normalization, canonical serialization/hash boundaries, bundle layout/checksums/manifests/schema dispatch, atomic persistence primitives, committed-generation bookkeeping, and template/provenance identity under `src/md_templates/core/`.

Use one canonical typed config; adapters translate legacy/generic inputs but resolution happens once. Keep compatibility re-exports. Add import-boundary tests proving core, listing, descriptor inspection, and identity do not import heavy dependencies. Add differential tests proving equivalent legacy/new inputs yield identical canonical values, scientific bytes, and hashes.

Gate: existing tests/goldens plus new core/differential/import-boundary tests pass.

## Phase 4 — reusable OpenMM provider

Create the sole reusable implementation under `src/md_templates/engines/openmm/`, with clear build, config/bundle adapter, runtime, method, persistence, and platform-selection boundaries. Move code; do not copy it. `src/md_templates/openmm/` becomes thin documented compatibility imports/wrappers only.

Preserve public imports, exceptions, `md-openmm`, CLI behavior, scientific hashes, bundle layout, and restart semantics. Keep method invariants explicit while sharing engine plumbing. Add a public-import compatibility matrix. Optional dependencies fail actionably only when engine functions are requested.

Gate: legacy namespace/CLI, goldens, readers, and resume fixtures pass; only one OpenMM implementation exists.

## Phase 5 — activate conventional MD

Activate `templates/conventional-md/openmm/explicit-water/template.yaml`. Make its directory authoritative for profiles/examples/template evidence and package it in wheel/sdist. Replace legacy-direct/non-local descriptor status with real provider/dispatch and template-local status.

Add a generic `md-templates` CLI that lists, inspects, and reports exact identity without heavy imports, and explicitly dispatches prepare/run/resume. Choose syntax consistent with the current CLI and document it; keep `md-openmm` compatible. Route PDB and SMILES preparation through the centralized provider. Prove source, installed-wheel, and relocated-bundle operation without changing defaults/hashes.

Gate: generic/legacy routes produce equivalent canonical/scientific artifacts; CPU prepare/run/resume works from a wheel outside checkout; source/wheel assets agree.

## Phase 6 — activate REST2

Activate `templates/rest2/openmm/explicit-water/template.yaml`; make its directory authoritative for profiles/examples/evidence and update dispatch/provider/local-profile status. Preserve ladder, replica order, exchange schedule, RNG, omega, lifetime phase continuity, checkpoints, and hashes. Test missing/default omega separately from explicit false. Keep REST2 validation distinct from engine plumbing and run tiny installed-wheel CPU prepare/run/resume. Do not elevate scientific claims.

Gate: source/wheel generic REST2 works; exchange/restart/RNG/lifetime tests pass; legacy-equivalent inputs retain identical scientific output.

## Phase 7 — artifact dispatch, identity, restart

- New artifacts record engine, method/template dispatch, schema version, and exact full-SHA-plus-logical-`template.yaml` identity.
- Physical installed paths never become identity; metadata stays outside scientific hashes.
- Legacy artifacts remain readable/resumable. Label inferred provenance honestly, never fabricate identity, and never rewrite old artifacts merely by opening them.
- Separate persistence/restart concerns enough to test atomic writes, committed-generation discovery, checkpoint validation, recovery, and method restoration independently.
- Exercise all established interruption cases: partial metadata/checkpoint, stale candidates, corrupted newest generation, fallback to last committed generation, relocation, state fallback, and real SIGKILL.
- Generic resume dispatches deterministically from metadata or fails actionably for genuinely ambiguous legacy artifacts.

Gate: v1/v2 and historical resume pass; new artifacts have exact identity/dispatch; all crash/relocation tests pass with unchanged scientific hashes.

## Phase 8 — consolidate tests, docs, evidence, CI

Organize tests by core, engine, template/method, compatibility, packaging, and integration; preserve/improve coverage and map every moved/superseded test. Archive one-off scripts only when evidence and active coverage remain. Update architecture, CLI, compatibility, bundle/restart, contributor, and template-author docs. Publish a support matrix separating implemented+CI-tested, implemented+conditional/manual, system-specific pilot-supported, and scientifically validated.

CI must test an installed wheel outside checkout and retain strict fast checks plus tiny CPU PDB/SMILES, conventional-MD, REST2, resume, crash, relocation, and state-fallback integration. Verify source/wheel/sdist catalog and assets. Remove only non-public temporary adapters; keep legacy namespace/CLI.

Gate: clean-checkout and installed-wheel matrices pass, docs match evidence, and no duplicate implementation/profile source remains.

## GPU/platform compatibility gate

GPU support is an operational contract although normal CI is CPU-only. Preserve `CUDA`/`OpenCL`, explicit `--device` validation, OpenMM `DeviceIndex`, `Precision` values `single`/`mixed`/`double` and current effective default, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, explicit unavailable-platform/device failure, and REST2 one-process/one-selected-device behavior.

1. Mocked/unit contract tests must prove generic and legacy routes select the same platform and pass identical `Precision`/`DeviceIndex`.
2. Preserve observable legacy defaults. Shipped profiles describe CPU while operational legacy commands may default to CUDA; document precedence rather than silently “fixing” it. Any normalization must be compatibility-proven.
3. Generic execution must make platform/device selection explicit and never silently fall back from CUDA/OpenCL to CPU.
4. Do not add implicit multi-GPU replica distribution. External one-process-per-GPU launching remains the approach.
5. If a CUDA runner exists, run optional real-GPU prepare plus tiny conventional-MD and REST2 run/resume; record GPU model, platform, precision, device, and result.
6. No GPU runner is not a CPU-CI failure, but report “implemented, contract-tested, real-GPU not run” unless real tests passed. Mocks/CPU are not CUDA validation.

## Final acceptance

- Seven goldens are byte/hash unchanged; every prior test runs or has a reviewed replacement mapping.
- Fast, slow, crash/restart, packaging, and CPU integration pass.
- Wheel/sdist install and run outside checkout; source/wheel/sdist catalog/assets agree.
- Core/catalog-only commands avoid heavy imports.
- Legacy imports, `md-openmm`, v1/v2 reads, and committed-run resume work.
- New artifacts carry exact identity without scientific-hash changes.
- Generic listing/inspection/identity works engine-free; conventional-MD and REST2 execute.
- CPU PDB, SMILES, both methods, relocation, interruption, and resume smokes pass.
- CUDA/OpenCL/device/precision is contract-tested and real-GPU status stated honestly.
- No duplicate implementation/profile source or generated outputs, caches, secrets, environments, or unrelated diff.

## Out of scope

No new engine, Amber adapter, sampling method, scientific default/force field/timestep/HMR/REST2 ladder, multi-GPU REST2, removal of `md-openmm`/`md_templates.openmm`, removal of legacy readers, broad reformat, or unrelated dependency upgrade. The 2 fs versus 4 fs/HMR question and general ladder suitability remain open scientific questions.

## Final delivery

Push the branch and open one PR against `openmm`; do not merge. Its body and Claude report must give the phase commit list, architecture/compatibility summary, exact test commands/counts, golden comparison, wheel/sdist evidence, legacy read/resume evidence, CPU/crash evidence, GPU contract and any real-hardware results, limitations/scientific status, justified deviations, and journal link.

Fix failed gates within the phase and continue. If an invariant cannot be preserved, stop before publishing and report the smallest reproducible conflict with exact files, hashes, and commands.
