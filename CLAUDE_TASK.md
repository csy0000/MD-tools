# Final MD-tools v0.5 cleanup: remove the residual legacy MD configuration stack

Status: active, temporary execution instruction  
Target branch: `dev`  
Starting point reviewed: `4115d8c8f4682a98cbc89232ad8a48b14e49bade`  
Do not merge to `main`, create `v0.5.0`, publish a package, or change the version in this task.

## Why this task exists

The v0.5 release-blocker work is functionally green, but inspection found a maintainability problem:
the public `md-openmm build-md` implementation in `src/md_tools/build/md.py` uses the new
`protocol:` model and integer step counts, while parts of `src/md_tools/openmm/defaults.py`,
`src/md_tools/openmm/config.py`, `src/md_tools/openmm/stages.py`, and their tests still describe
and validate the retired `methods:` model with fields such as `switching_duration_ps` and
`duration_ns`.

This is especially confusing for an AI coding agent because both models look authoritative. There
must be one current MD configuration model.

This task is a cleanup, not a redesign. Preserve the current public behavior, scientific
invariants, generated bundles, dataset contract, and successful tests.

## Non-negotiable current interface

The installed distribution has one console script:

```text
md-openmm
```

It has exactly three public work commands:

```text
md-openmm build-top
md-openmm build-md
md-openmm data-register
```

AIS remains `protocol: AIS` under `build-md`. Do not introduce another executable or command.

The only current configuration examples are:

```text
configs/machine/user.config.example
configs/sys/build-top.config
configs/md/cMD.config
configs/md/REST2.config
configs/md/rREST2.config
configs/md/AIS.config
```

The data contract remains v2, with no month component and no runtime import of `md_data`.

## 1. Map dependencies before deleting anything

Before editing, build an import/call map for:

- `src/md_tools/openmm/defaults.py`
- `src/md_tools/openmm/config.py`
- `src/md_tools/openmm/stages.py`
- `src/md_tools/openmm/templates/`
- `src/md_tools/build/top.py`
- `src/md_tools/build/md.py`
- `src/md_tools/runtime/`

Classify each symbol as:

1. required by current `build-top`;
2. required by current `build-md` or its generated runtime;
3. required only by the retired `sys-gen`/`md-gen` configuration route;
4. test-only compatibility residue;
5. uncertain.

Do not delete categories 1 or 2. Resolve category 5 from callers and tests rather than guessing.

Important: files under `openmm/templates/` are not automatically obsolete. Several are currently
imported by `md_tools.runtime`. Preserve or move active scientific/runtime helpers with their
behavior and tests intact.

## 2. Remove the duplicate retired MD configuration model

Make `src/md_tools/build/md.py` the only authority for current MD workflow configuration.

Remove obsolete MD-specific definitions and validation paths that exist only for the retired
`methods:` configuration, including as applicable:

- `md_defaults()` and `ais_defaults()` in their old form;
- old `methods` dispatch;
- old MD `resolve_md_config()` branches;
- `switching_duration_ps`, `duration_ns`, and the old equilibration-duration vocabulary;
- old stage-plan construction used only by `md-gen`;
- tests whose only purpose is to validate that retired model.

Do not mechanically delete whole mixed-purpose modules. `build-top` still uses system defaults and
`resolve_sys_config`; current runtime code also uses `write_yaml` and selected scientific
helpers. Either retain the focused system/runtime parts with clear module documentation or move
them to names that describe their current role and update all imports.

Afterwards, there must not be two callable functions both presented as the MD configuration
resolver. A search for the retired duration fields may find migration or release-history prose, but
must not find live configuration code.

## 3. Correct stale command and interface language

Audit every occurrence of:

```text
sys-config
sys-gen
md-gen
show-default
openmm-md
md-template
md-data-register
md-data-finish
```

Classify occurrences rather than blindly deleting them:

- Keep explicit negative tests proving a retired command is refused.
- Keep concise release/migration history that clearly says the name is retired.
- Remove or rewrite prose that implies a retired command is current.
- Rename misleading test files, test names, and docstrings where they claim the current runtime is
  launched through the retired `openmm-md` executable.
- An internal Python module may retain a technically justified name only if it cannot be confused
  with an installed executable; otherwise give it a role-based internal name and update imports.

Update `CLAUDE.md` so its rule is precise: retired names may occur in explicit negative tests and
release/migration history, but must never be recommended or used as current commands.

## 4. Decide the fate of the root reports directory

Audit `reports/explicit_solvent/20260815_ff19sb_peptide_route/`.

If those raw logs are not consumed by current tests, documentation, or the v0.5 validation record,
delete the directory. It remains recoverable through Git history and the
`pre-v0.5-doc-cleanup` tag.

If the logs are necessary release evidence, document exactly what current claim they support and
why the compact release-note evidence is insufficient. Do not keep them merely because they already
exist.

## 5. Fix repository metadata

If authenticated GitHub CLI access is available, update the repository description to:

```text
Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and registering MD datasets.
```

If permission or authentication is unavailable, do not work around it. Record the exact command the
owner should run:

```bash
gh repo edit csy0000/MD-tools --description "Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and registering MD datasets."
```

Metadata failure must not block the code cleanup.

## 6. Required verification

Run all of the following from a clean working tree after editing.

### Static/interface checks

- `pyproject.toml` contains exactly one console script, `md-openmm`.
- Its subcommands are exactly `build-top`, `build-md`, and `data-register`.
- All six root configuration examples resolve through the real public resolvers.
- `rg` finds no runtime `md_data` import.
- `rg` finds no live old MD configuration fields or old `methods:` dispatch.
- Every remaining retired-command occurrence is an explicit negative assertion or clearly labelled
  history.
- Generated scripts contain no checkout path or machine-specific absolute path.

### Tests and packaging

Run:

```bash
python -m pytest tests -m "not slow and not gpu" -q -rs
python -m pytest tests -m "gpu or slow" -q -rs
python -m build
```

No selected test may skip. GPU tests must genuinely use CUDA; do not substitute CPU execution.

Install the built wheel in a clean environment, change to a directory outside the checkout, and
verify:

```bash
md-openmm --version
md-openmm build-top -h
md-openmm build-md -h
md-openmm data-register -h
```

Locate all six installed examples through `md_tools.configs.example_root()`. Generate and compile
cMD split, cMD implicit, cMD all-in-one, REST2, rREST2, and AIS bundles. Confirm AIS has no
minimization/equilibration chain. Confirm the committed dataset and extension schemas match their
models.

Push the implementation to `dev` and confirm the GitHub Actions run for the final code commit is
green. Do not describe CI as scientific validation; the CUDA/slow lane is separate local evidence.

## 7. Documentation and final repository state

Update `CHANGELOG.md` and `docs/release-notes/v0.5.0.md` concisely with:

- what duplicate legacy model was removed;
- what active shared code was retained or moved;
- test counts, zero-skip status, wheel verification, CUDA environment, and final CI URL;
- the disposition of `reports/`;
- whether repository metadata was updated.

Do not add a journal, implementation diary, generated inventory, or another permanent instruction
directory.

This file is intentionally temporary. After every acceptance criterion is satisfied and the final
CI result is recorded, delete `CLAUDE_TASK.md` in the final documentation commit. The instruction
will remain in Git history without burdening future agents.

## Completion report

Return:

1. final `dev` commit SHA;
2. concise list of deleted, split, moved, and retained modules;
3. explanation of every remaining retired-name occurrence category;
4. disposition of `reports/`;
5. exact fast and GPU/slow test counts with zero-skip confirmation;
6. wheel/outside-checkout results;
7. final GitHub Actions URL and conclusion;
8. repository-description result;
9. confirmation that `CLAUDE_TASK.md` was deleted;
10. confirmation that `main`, package publication, and final release tagging were untouched.
