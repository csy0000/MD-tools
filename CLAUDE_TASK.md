# Final documentation consistency corrections

Status: active, temporary execution instruction  
Target branch: `dev`  
Reviewed starting point: `c68262a517043f0168ee9f2ae462308cac56e9de`

This is a narrowly scoped documentation and repository-metadata correction. The code cleanup is
already accepted and the current-head CI is green. Do not refactor implementation code, alter
scientific behavior, change the CLI, merge to `main`, publish a package, create `v0.5.0`, or
change version `0.5.0.dev0`.

## 1. Correct the packaging comments in pyproject.toml

The comments above `[tool.setuptools.package-data]` currently make two obsolete claims:

- the shipped configuration examples are located with `importlib.resources`;
- the repository-root `configs/` path is a symlink to a package-data directory.

Both claims are false. The authoritative current arrangement is:

- `configs/` is an ordinary repository-root directory, not a symlink;
- it is the only hand-edited copy of the six examples;
- setuptools installs those files through `[tool.setuptools.data-files]` beneath
  `share/md-tools/configs/`;
- `md_tools.configs.example_root()` locates them using installed-distribution metadata, with the
  documented source-tree fallback;
- `importlib.resources` remains appropriate for actual package resources such as runtime modules
  and contract schemas, but not for these configuration examples.

Rewrite or remove only the obsolete comments. Keep the existing package-data and data-files tables
functionally unchanged.

## 2. Correct src/md_tools/build/__init__.py

Its module docstring currently says:

```text
shipped configuration examples are located with importlib.resources
```

Replace that statement with a concise accurate description: neither command assumes a repository
root, and installed configuration examples are located through distribution metadata by
`md_tools.configs.example_root()`.

Do not introduce an import merely to support the docstring.

## 3. Correct the preservation statement in the v0.5 release notes

Near the beginning of `docs/release-notes/v0.5.0.md`, the text currently states that everything
removed below is reachable from `pre-v0.5-doc-cleanup`. This is too broad.

Most pre-cleanup material is recoverable from that tag, but
`reports/explicit_solvent/20260815_ff19sb_peptide_route/` was added later and is instead
recoverable from commit:

```text
1690140fc77eea95137654b1e107cf9cb6100613
```

Rewrite the introductory preservation paragraph so it states both recovery points without
contradiction. Keep the release note compact; do not add a journal or deletion narrative.

The result must not imply that the reports were present in the preservation tag.

## 4. Update the GitHub repository description

Using only normally configured authenticated GitHub access, attempt to set the description to:

```text
Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and registering MD datasets.
```

Preferred command:

```bash
gh repo edit csy0000/MD-tools --description "Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and registering MD datasets."
```

If access still returns HTTP 403 or lacks permission:

- do not bypass permissions;
- do not treat it as a code failure;
- report the exact failure and the command the owner must run;
- retain the existing repository-metadata note in the release notes.

If it succeeds, update the repository-metadata section of the release notes to state the current
description rather than preserving the old failure note.

## 5. Verification

Before committing, verify:

```bash
rg -n "configs.*symlink|symlink.*configs|configuration examples.*importlib\.resources|examples are located with.*importlib\.resources" \
  pyproject.toml src/md_tools/build/__init__.py README.md CLAUDE.md docs src
```

Review every result. Historical discussion of the former symlink in the release notes and explicit
rules saying never to add a symlink are valid. No current-description text may claim that
`configs/` is presently a symlink or that the config examples are package resources.

Also verify:

- `configs/` is a real directory in Git;
- the exact six expected config files remain unchanged;
- `[project.scripts]` still contains only `md-openmm`;
- version remains `0.5.0.dev0`;
- the release note names the preservation tag for pre-cleanup material and commit `1690140...`
  for the removed reports;
- `git diff --check` passes.

Because this task must not modify executable code or configuration examples, the full local CUDA
lane does not need to be repeated. Run the focused static/package checks affected by the edits, and
allow the existing GitHub Actions workflow to perform its complete packaging/non-GPU validation.
If any implementation or test file changes unexpectedly, stop and explain why rather than expanding
scope silently.

## 6. Commit, CI, and temporary instruction cleanup

Commit and push the corrections to `dev`. Wait for that commit's GitHub Actions run and require a
green conclusion.

Then delete this temporary `CLAUDE_TASK.md`, commit that deletion, push it to `dev`, and confirm
the resulting current-head GitHub Actions run is also green. The instruction remains available in
Git history without burdening future agents.

Do not merge or tag.

## Completion report

Return:

1. final `dev` commit SHA;
2. exact files changed;
3. the corrected meaning of each of the three documentation statements;
4. repository-description result, including the exact error if it still failed;
5. both GitHub Actions URLs and conclusions;
6. confirmation that `CLAUDE_TASK.md` was deleted;
7. confirmation that executable code, config examples, `main`, package publication, version, and
   final release tags were untouched.
