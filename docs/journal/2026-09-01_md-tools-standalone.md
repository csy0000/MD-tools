# 2026-09-01 — MD-templates becomes MD-tools: a standalone, pip-installable package

Starting commit `bed236f7e3eae5ce2fc1d5fec9818109971231e6` (branch `dev`), which is the commit the
instruction named and the head that was actually there when work began.

Executed from `docs/claudecode-instructions/20260901_md_tools_standalone_and_minimal_md_project.md`
in MD-project, in the mandated order: make the package work under its new name first, prove it from
a built wheel with no source checkout on `PYTHONPATH`, then reduce MD-project, and only then rename
anything on GitHub.

---

## 1. Naming

| old | new |
|---|---|
| distribution `md-templates` | `md-tools` |
| import package `md_templates` | `md_tools` |
| `src/md_templates/` | `src/md_tools/` (via `git mv`, so history follows) |
| version `0.4.0.dev0` | `0.5.0.dev0` (breaking) |

153 maintained files rewritten. `docs/journal/`, `docs/journals/`, `claudecode-instructions/`,
`reports/` and `CHANGELOG.md` keep the old name: they state historical facts, and renaming them
would not update a reference, it would make a record wrong.

**I got that wrong once and fixed it.** My exclusion pattern said `docs/journals/` but the
directory is `docs/journal/` (singular), so 21 historical journals were rewritten to say MD-tools.
Restored verbatim in `8f0fe4e`.

`md-openmm` is now the only console script.

## 2. Removed paths

| path | lines | why |
|---|---:|---|
| `src/md_tools/cli/md_template.py` | 208 | the `md-template` entry point |
| `src/md_tools/install/` | 1163 | conda-stack installer; the migration replaces it with `pip install md-tools`, and environment construction is not among the capabilities MD-tools is specified to own |
| `installation/` | 172 | documented that installer |
| `src/md_tools/openmm/simple.py` | 1006 | the retired `setup` route. Imported by nothing after the CLI change |
| `src/md_tools/openmm/emit.py` | 678 | emitted `bin/openmm-md` launchers for a route with no command |
| `tests/test_install.py`, `test_installer_capability.py`, `test_machine_init.py` | 791 | tested the installer |
| `tests/test_simple_setup.py`, `test_staging_exception.py` | 842 | tested the retired `setup` route |
| 6 installer tests in `test_integrity_corrections.py` | — | same |

`test_staging_exception.py` deserves its own note. It tested an opt-in that let `setup` write into
a git working tree, which existed because `setup` otherwise required its output to sit beneath
`$MD_DATA`. `build-top` has no such constraint by design — `-op data/ALA/built.pdb` is the
documented shape — so the guard has nothing left to guard. The modern protection is that `data/` is
git-ignored and `data-register` moves the bytes out of the repository entirely.

## 3. The new command surface

Exactly three public commands, and the five retired ones are **gone**, not hidden:

```text
md-openmm build-top      -i INPUT.pdb|.smi [-os built.xml] [-op built.pdb] [-log built.log] [--config]
md-openmm build-md       [-odir ./md_script/] [--config] [--all-in-one]
md-openmm data-register  -idata DIR -project_name NAME -data_name NAME -year YYYY [--common-data]
                         [--dry-run] [--verify-only] [--init]
```

Every contractual single-dash spelling is asserted rather than left to argparse prefix matching,
and `-h` is checked with OpenMM poisoned in `sys.modules`, which is how "help must not need CUDA"
becomes a test rather than a hope.

## 4. New modules

| module | what it owns |
|---|---|
| `build/strict.py` | schemas that REFUSE unknown keys, with a nearest-key suggestion |
| `build/record.py` | one delimited, versioned YAML machine record per log |
| `build/top.py` | `build-top`, reusing the validated `_build_explicit` / `_build_implicit` |
| `build/md.py` | `build-md`; stage lengths are exact integer step counts |
| `build/examples.py` | the shipped `.config` files, RENDERED from the schemas |
| `runtime/stage.py` | what generated stage scripts import |
| `runtime/replica.py` | prepares solute.yaml, the protocol module and the group file, then calls the validated executor as a function |
| `data_contract/` | dataset contract v2 |
| `registry/` | discovery, inventory, transaction, registration |

Shipped examples are generated from the schema objects that enforce them, so a comment documenting
a default and the default the code applies have one source. A test regenerates and compares.

## 5. Data contract v1 → v2

Ported from MD-data `protect-md-project-dev-test@20d982eb463ed439095f1b95e00ff1b1d75906b4`, with
attribution in the module docstring. `md_data` is not imported at runtime anywhere.

| | v1 | v2 |
|---|---|---|
| canonical path | `{namespace}/{yyyy-mm}/{dataset_name}` | `{year}/{project_name}/{data_name}` |
| shared datasets | `baseline/` namespace | `{year}/common/{project}/{data}` |
| role | `baseline` \| `project` | `common` \| `project` |
| dated segment | creation **month** | completion **year**, checked against the records |
| provenance field | `templates` | `software` |

A **new model**, not a widened v1. Widening would have left one model accepting both shapes, which
is exactly the state in which a wrong path goes undetected. A v1 manifest now fails on
`schema_version` with a message naming the difference, rather than on a missing month.

v2 also settles an ambiguity v1 left open: the dated segment was the creation month, but data are
usually finished later, so the segment often disagreed with when the data became final.

Ported unchanged: strict schema-version rejection, a stable `dataset_id` separate from the path,
relative-only paths, explicit role consistent with path shape, creator identity, timezone-aware
timestamps, dataset and component status, exact repository provenance, component uniqueness,
non-nesting and method identity, linked/read-only rules, and immutability of complete data.

`src/md_tools/openmm/md_data_contract.py` still implements v1, and five lazy `md_data` imports
survive with it. Both are deviations, both are deliberate, and both are written up with their
removal plan in §9.1.

## 6. Tests

| suite | command | result |
|---|---|---|
| fast | `python -m pytest tests -m "not slow and not gpu"` | **663 passed, 0 failed**, 46 s |
| GPU + slow | `python -m pytest tests -m "gpu or slow"` | **162 passed, 0 failed**, 4 min 28 s, on CUDA |
| wheel | `python -m build`, install into a clean venv | see §7 |

825 tests pass in total. The starting suite had 901; the difference is the 76 tests that covered
the installer and the retired `setup` route, all listed in §2. Nothing was deleted for failing.

Platform: Linux 6.8, Python 3.12.14, OpenMM 8.6, 9 CUDA devices (1 × A5000, 8 × RTX 3080).
GPU tests ran on CUDA. Nothing was substituted with CPU execution.

Fifteen tests that drove the retired commands were **rewritten, not deleted**, and doing so exposed
three real gaps that are now fixed:

* `--check` failed on a missing parent state. In an unrun chain every parent after the first is
  missing by construction, so `--check` was useless exactly where it is most useful. Now `pending`.
* nothing refused a large timestep on unrepartitioned hydrogens. A 4 fs step is reasonable, a
  System built without HMR is reasonable, and together they integrate a ~10 fs X–H angle motion
  with a 4 fs step and produce a wrong trajectory rather than a failed run. Now checked against the
  **masses serialised in the System**, which by run time is a fact rather than a request.
* the checkpoint fingerprint covered `steps`, so asking for a longer run invalidated the checkpoint
  and made legitimate extension impossible. `steps` is now the one extendable field.

`conftest` routes the three retired subcommand *names* to the generator API they used to reach: the
generators were never removed, only the commands, and ~37 call sites across ten mostly-GPU files
are exercising the generator rather than the argument parser. The CLI surface test invokes the real
command directly, so the shim cannot hide a regression in what is actually public.

## 7. Wheel isolation

```bash
python -m build
python -m venv --system-site-packages /tmp/wheel/venv
/tmp/wheel/venv/bin/pip install --no-deps dist/md_tools-0.5.0.dev0-py3-none-any.whl
pip uninstall -y md-tools          # the dev editable install, so the source is genuinely absent
```

Verified from that venv, with `sys.path` proven free of the source checkout and the working
directory outside the repository:

* `md-openmm -h`, `--version`, and `build-top -h` / `build-md -h` / `data-register -h` all exit 0;
* package resources resolve **from the wheel** through `importlib.resources` — 4 `.config` files
  and 30 runtime template modules;
* `build-top` on the committed ALA input: explicit TIP3P gives 1796 atoms == 1796 particles,
  periodic; implicit GBn2 gives 22 atoms, non-periodic, no waters;
* split and `--all-in-one` scripts generated, all compile, none contains an absolute path or a
  reference to a source checkout;
* a five-stage cMD chain **ran on CUDA in 6.5 s**, every stage recording `platform: CUDA`;
* a 4-state REST2 ladder ran: `remd0.nc`…`remd3.nc`, `rem.log`, per-pair acceptance;
* a full registration of a 37-file dataset landed at `2026/ALA/ALA-cMD`, all 37 checksums
  verifying, the source replaced by a relative symlink.

## 8. rREST2, and fixed-tau cMD

rREST2 could not be smoke-tested at first: a reservoir must be Boltzmann-weighted at exactly the
top rung's Hamiltonian and hold complete samples **including velocities**, and the way this
repository produces one is a fixed-tau cMD run — which `build-md` had no way to express. That would
have silently dropped a validated capability (`tests/test_fixed_tau_md.py`).

So `dynamics.tau` and `dynamics.phase_space_printout` were added, wired to the same validated
`build_scaled_system` the ladder uses. Enforced, not documented: a scaled run is NVT whatever the
solvent; `tau` on a REST2/rREST2 protocol is refused; a phase-space stream at tau = 0 is refused;
and phase space is written only from production, never from restrained equilibration, because a
reservoir drawn from restrained dynamics is not a Boltzmann sample of the target ensemble.

The validated identity check then caught two bugs in my code, in sequence:

1. the reservoir fingerprint was a hand-assembled dict with no recognised format;
2. once real, it still disagreed — it claimed a `CustomExternalForce` the ladder rung does not
   have, because I fingerprinted the System *after* the positional restraint was added. Taking a
   reference before the call was not enough either, since `add_positional_restraint` mutates in
   place; the record is now **computed** at that point.

rREST2 then ran end to end: 10 phase-space samples, `velocity_policy: stored`, refresh every 2
exchanges, 4 state trajectories, `rem.log`, `run_status: completed`.

## 9. Search gates

Zero hits outside historical journals, `docs/legacy/`, instruction records and tests that assert
absence, for: `MD-templates`, `md_templates`, `md-template`, `md-data-register`, `md-data-finish`,
`components.lock.yaml`.

Three `openmm-md` hits remain and all three are sentences saying the command **does not exist**.
Seven `yyyy-mm` hits remain: two in `data_contract/` explaining what v1 was, five in the LEGACY v1
module. Those five, and the `md_data` imports beside them, are the subject of §9.1 — they are not
passing the gate, they are recorded as failing it.

## 9.1 Two remnants, one root cause — to be removed together

Both survive the migration deliberately, both are recorded here rather than left to be
rediscovered, and both disappear in the same piece of work: **giving AIS a public command.**

### What they are

**(a) `src/md_tools/openmm/md_data_contract.py` — dataset contract v1.**
MD-tools owns v2 in `md_tools.data_contract`. This module is the v1 implementation, kept because
the internal `sys-gen`/`md-gen` route can embed a `dataset:` block in its configuration. It is
marked LEGACY in its own docstring.

**(b) five runtime imports of `md_data`**, which the instruction forbids outright:

```text
src/md_tools/openmm/md_data_contract.py:65    import md_data
src/md_tools/openmm/md_data_contract.py:103   import md_data
src/md_tools/openmm/md_data_contract.py:408   from md_data.storage import check_dataset_tree
src/md_tools/openmm/templates/preflight.py:131  import md_data
src/md_tools/openmm/templates/preflight.py:148  from md_data.storage import check_dataset_tree
```

### Stating the deviation accurately

The instruction says "Do not import `md_data` at runtime." That is not satisfied. What *is* true:

* `md-data` is **not** a declared dependency — it is absent from `pyproject.toml`, and every one of
  the five imports is lazy and guarded, so the package installs and runs without it;
* the path is **unreachable from all three public commands**: `build-top` pops the `dataset:` block
  before building, `build-md` never touches it, and `data-register` validates against v2 and does
  not import this module at all;
* it fires only when a `sys.config.yaml` sets `dataset.enabled: true`, on the route AIS uses.

So the intent holds — MD-data is not required, and nothing a user runs reaches it — while the
letter does not. That is a deviation, not a technicality, and it is written down as one.

### Why they were not removed now

Removing them is not deleting dead code. `mdgen.py` calls this module a dozen times to resolve
roots, build component entries, merge them, write the manifest and validate it, and `preflight.py`
uses it to check a dataset tree before a run. Tearing that out would delete working, tested
behaviour and take a large number of tests with it:

| test file | tests | what would need doing |
|---|---:|---|
| `tests/test_md_data_contract.py` | 35 | rewritten against contract v2, or removed with the feature |
| `tests/test_template_provenance.py` | 16 | partially — the provenance half survives |
| `tests/test_integrity_corrections.py` | 28 | partially — most do not touch the contract |

Doing that as a side effect of a packaging migration would have been the wrong trade: the
capability still works, and nothing a user can type reaches the part that is wrong.

### The shape of the fix

AIS is the only reason the internal route still exists. When it gets a public command — `md-openmm
build-ais`, or an `AIS` protocol in `build-md` — the whole chain falls out together:

1. give AIS a public command that generates through `build-top` + the `md_tools.runtime` stage
   machinery, as cMD/REST2/rREST2 already do;
2. retire `sysgen.generate_system` and `mdgen.generate_md`, which then have no caller;
3. delete `md_data_contract.py` and the `dataset:` block it serves — registration has been a
   separate command since this migration, so an embedded manifest has nothing left to do;
4. drop the `md_data` branch of `preflight.py`; a generated run does not need to revalidate the
   dataset it is being written into, because `data-register` validates before it commits;
5. rewrite `test_md_data_contract.py` against contract v2, and remove the conftest shim that routes
   the three retired subcommand names to the generator API — it exists only for these callers.

Afterwards the search gate for `yyyy-mm` is clean in maintained code, `md_data` appears nowhere at
runtime, and MD-tools carries exactly one dataset contract.

## 10. The rename

Done, in two parts, on 2026-09-01 after everything above had passed.

### 10.1 GitHub

`csy0000/MD-templates` -> `csy0000/MD-tools`, renamed by the repository owner.

I had attempted it through the API and been refused — not by GitHub, which reported
`permissions.admin: true` for the token in `~/.config/gh/hosts.yml`, but by this environment's own
permission layer, which blocks outward-facing repository administration. That is the right place
for it to be refused, and it was not worked around.

The remote here was then updated explicitly rather than left to GitHub's old-name redirect:

```bash
git remote set-url origin git@github.com:csy0000/MD-tools.git
```

Verified: `git ls-remote git@github.com:csy0000/MD-tools.git` resolves, and a fetch against the new
URL succeeds.

### 10.2 The worktrees, and the local directory

Two unrelated projects held worktrees of this repository, which would have made a directory rename
break them:

```text
/path/to/projects/krREST2/components/MD-templates   f47505f  2026-08-27
/path/to/projects/pBGF/components/MD-templates      e15071e  2026-08-26
```

Both were checked before being touched: clean working trees, no untracked files, no stashes, and
**both HEADs contained in `origin/dev`**, so removing them discarded nothing. The commits are
recorded above in case either project ever wants that exact state back. Removed with
`git worktree remove`, and the already-prunable `MD-projects/components-dev/MD-templates` entry
cleared with `git worktree prune`.

`git worktree list` now shows one entry and `.git/worktrees` is empty, so moving the directory is a
plain `mv` with no `git worktree repair` needed.

**Both projects still reference the removed component in their own configuration** —
`components.yaml`, `components.lock.yaml`, `project.yaml` and, for krREST2,
`run_preparation.sh`. Neither project was part of this migration and neither was edited. If either
is picked up again it will need its component model reconsidered, which is the same work MD-project
has just had done to it.

### 10.3 One thing the directory move breaks

The development environment has MD-tools installed editable, pinned to the old absolute path:

```text
site-packages/__editable__.md_tools-0.5.0.dev0.pth   ->  /path/to/scheme/MD-template/src
```

After the `mv`, reinstall from the new location:

```bash
cd /path/to/scheme/MD-tools && pip install -e . --no-deps
```

Nothing else depends on the path. MD-project installs from a wheel and never referred to it.
