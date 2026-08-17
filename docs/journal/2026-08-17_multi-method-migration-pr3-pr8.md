# PR3–PR8: one campaign to a reusable multi-method template platform

**2026-08-17.** One branch, `migration/pr3-pr8-reusable-template-platform`, from `openmm` at
`c2731a8` — which is the instructed base `b2b35fe` (merged PR 2) plus the campaign instruction file
itself, and nothing else. Phases run in order; each begins only after the previous gate passes.

This journal is written phase by phase as the work happens, and records for each: what moved, what
the compatibility decision was and why, the exact commands and their results, the artifacts and
hashes compared, the risks accepted, and the gate status. Where a phase is incomplete or a gate could
not be run, it says so rather than rounding up.

---

## Phase 0 — characterize and freeze

**Purpose.** Before a single module moves, make it impossible for the move to hide a loss. A
structural migration breaks interfaces far more easily than physics, and it breaks them quietly.

### Baseline, measured

| check | result | matches PR 2 reference? |
|---|---|---|
| `pytest tests/ -q -m "not slow"` | **527 passed**, 17 deselected, 39.40 s | yes — 527/17 exactly |
| `python scripts/capture_goldens.py --check` | **7/7 ok**, exit 0 | yes |

No differences to explain: the campaign starts exactly where PR 2 finished.

### What was frozen, and why each guard exists

**`tests/baseline/public_surface.json`** (36.9 kB) — 20 commands, 96 public names, 18 exception
types, 33 modules. Written by `scripts/capture_public_surface.py`, deterministic, with a `--check`
mode.

Recorded as *structure*, not prose. The CLI is captured as a parse tree — every command, option
string, destination, default, choice list, `nargs`, type and required flag — because comparing
rendered help text would tie the baseline to wording rather than behaviour. Public names are recorded
with the module each one actually comes from, so a symbol that silently relocates is visible even
while `from x import y` keeps working through a re-export. Exceptions are recorded as full MROs,
because `except` clauses in consuming code depend on the inheritance chain, not the name.

**Golden bytes.** `tests/test_phase0_characterization.py` pins the SHA-256 of all seven golden files
directly. `capture_goldens.py --check` recomputes and compares *content*; the byte check also catches
a regenerated-but-equivalent rewrite, which is exactly what a migration is tempted to do when a
golden fails.

```
bundle_contract.json       514b14c308a0e588...   profiles.json           ba524a6dd49679c5...
configuration_hashes.json  a42d15344e5871a4...   rest2_defaults.json     a9420bd1b1a7a922...
format_equivalence.json    eca1873de9e7e7d1...   runstate_contract.json  2ef132a8ca41664e...
seed_derivation.json       95beae1da61dd0ae...
```

**`tests/baseline/module_map.json`** — every module and test file with its planned destination,
owning phase, and an honest `status` of `planned` / `moved` / `unchanged`, plus `compat_shim` where
the old import path must keep working. Tests assert that the map covers every module that exists,
that entries are well formed, and that any destination claimed as `moved` is genuinely importable
along with its shim. A map nobody checks is a wish; this one fails the suite when it drifts.

### Engine-coupling survey, which shaped the phase plan

Measured rather than assumed — top-level heavy imports per module:

| module | top-level heavy imports | consequence for the plan |
|---|---|---|
| `spec/*` (7 modules) | **0** | already engine-neutral; Phase 3 is a move, not a rewrite |
| `runstate`, `schemas`, `bundlev2`, `provenance`, `fingerprint`, `bundlecheck`, `runner`, `envcheck`, `cli`, `bundle` | 0 (deferred imports inside functions) | movable to core with care; `bundlev2.topology_counts` needs an engine and stays with the provider |
| `config`, `equilibration`, `md`, `rest2`, `solvation`, `system` | 1–2 | genuinely engine code; Phase 4 material |

The useful finding is that the canonical configuration package is *already* free of engine imports at
import time, so Phase 3's risk is in the re-export surface and the persistence/bundle split, not in
the config models.

### Fixtures selected for legacy read and resume

Existing slow coverage is the fixture set, and it is named here so later phases cannot quietly stop
running it:

* `tests/test_bundle_portability.py` — v2 bundle preparation on both routes, relocation with the
  source directory deleted, `validate --deep`, inspect, relocate-check, and v1-compatibility reads;
* `tests/test_crash_recovery.py` — real subprocess interruption, committed-generation recovery,
  uncommitted-tail quarantine, and State fallback;
* `tests/test_explicit_portable.py` and `tests/test_explicit_baseline.py` — bundle v2 contract and
  the non-slow engine surface.

### Gate

**PASSED.** Baseline and hashes recorded, three independent guards committed and green, module map
covering all 33 modules, before any structural edit.

```console
$ python -m pytest tests/test_phase0_characterization.py -q
16 passed                                                                   1.89 s
```

```console
$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh        # strict, installed wheel, clean tree
543 passed, 17 deselected; identities resolved to 2f0c25d0...
fast checks: PASSED                                                        exit 0

$ python -m pytest tests/test_crash_recovery.py tests/test_bundle_portability.py -q -m slow
15 passed, 21 deselected                                                  273.08 s
```

One defect, found by the gate rather than by review: the surface capture used `dir()` on a package,
which gains an attribute for every submodule *anything* has imported. The baseline was recorded in a
fresh interpreter and checked inside a full pytest session, where half the tree is already imported,
so the numbers disagreed — 96 names against 76 that are actually API. Fixed in the capture, not the
assertion: submodules are inventoried separately and deterministically, so this section states the
API rather than import order.

---

## Phase 3 — engine-neutral core

**What moved.** The Phase 0 survey found **zero** top-level heavy imports in `spec/`, so this phase
is a move, not a rewrite.

| from | to | what it is |
|---|---|---|
| `openmm/spec/*` (7 modules + 5 profiles) | `core/config/*` | typed models, units, resolution, precedence and source attribution, canonical serialisation, the five hash projections |
| `openmm/runstate.py` | `core/persistence.py` | atomic writes, generation layout, committed-generation bookkeeping, quarantine |
| `openmm/bundlev2.py` | `core/bundle.py` | normalised relative paths, checksum domain, environment provenance |
| `openmm/fingerprint.py` | `core/fingerprint.py` | the projection, unchanged |
| *(extracted)* | `core/hashing.py` | `canonical_json` / `sha256_text` / `sha256_file` |

**Two splits, drawn where the engine actually starts.** `save_restart`/`load_restart` take a running
`Simulation` and serialize OpenMM objects, so they became `openmm/restart.py`; `topology_counts` and
`forcefield_provenance` need a built `System` — only the toolkit can say that a virtual site is not
a topology atom, or where a force-field XML resolved from — so they became `openmm/bundleinfo.py`.
Both are re-attached to the legacy module objects, so `runstate.save_restart(...)` and
`bundlev2.topology_counts(...)` still work from the paths every existing caller uses.

**`schemas.py` deliberately did not move**, against the initial plan. The shipped system and
experiment manifests are OpenMM assets, and core reaching into an engine's data directory would
invert the dependency the phase exists to establish. Only its hashing helpers were genuinely
neutral; they now have exactly one definition, which the manifest reader re-exports. Recorded in
`module_map.json` with the reason, rather than quietly left as planned.

**Compatibility decision: alias, do not re-export.** `md_templates.openmm.runstate` **is**
`md_templates.core.persistence` — the same object in `sys.modules`, not a module that copies names
out of it. A copying shim would let `monkeypatch.setattr("md_templates.openmm.runstate.f", ...)`
patch a shadow while the code under test reads the original: the test passes and tests nothing.
Shared module state and private names keep working for the same reason. A test asserts identity
(`is`), not merely importability.

The cost is real and is recorded rather than hidden: `__module__` on the moved objects now reports
the new location, so the full surface baseline was regenerated. The *names* are pinned separately in
`public_names_phase0.json`, which is never regenerated — a name that disappears is a compatibility
break, while an origin that moves is the architectural change this campaign exists to make. Nothing
was lost: 0 names, 0 commands, 0 exceptions missing from the Phase 0 pin.

**Catalog references updated** because the profiles travelled with their resolver:
`python_resource` and one `repository_references` entry in both descriptors, plus the `package-data`
key in `pyproject.toml`. The catalog's own reference check caught this immediately — the build and
every catalog test failed until the descriptors matched the tree, which is the guard working.

### Tests, and what each proves

Two tests changed because the move made their question obsolete, both rewritten to state the
invariant more precisely rather than to pass:

* the catalog's engine-boundary test was a substring scan for `"from openmm"`, which flags every
  legitimate deferred import. It now parses the AST and looks only at **column-zero** imports across
  the whole core tree — the actual invariant is about import *time*.
* `test_identity_is_not_written_into_bundles_or_runs` asserted the runtime imports no `core` module
  at all. After Phase 3 that is false by design. Rescoped to what actually matters: the runtime must
  not reach the **catalog and identity** modules, so template metadata cannot enter a bundle, a
  manifest or a hash. A complement test asserts the runtime *does* use core, so the first cannot
  pass by the runtime importing nothing.

New in `tests/test_core_boundary.py` (26 tests):

* **import boundary, proven by removal.** A subprocess installs a `sys.meta_path` blocker that makes
  `openmm`, `openff`, `rdkit`, `mdtraj`, `parmed`, `openmmtools`, `numpy`, `scipy` and `pandas`
  *unimportable*, then imports core and resolves a real configuration, loads the catalog and reads
  Git provenance. A source scan asks "does it mention OpenMM"; this asks the question that matters.
  One test asserts the blocker itself fails on `import openmm`, because a guard that cannot fail
  proves nothing. Every core module is additionally imported alone under the blocker.
* **differential equivalence.** All four representative configurations resolved through the legacy
  path and the core path, comparing all five projection hashes, the profile record, the full source
  attribution map, and the canonical bytes. Plus a test that the two paths are the *same objects*,
  without which the differential test would be comparing a thing with itself.

### Gate

**PASSED.**

```console
$ python -m pytest tests/ -q -m "not slow"
574 passed, 17 deselected                                                  45.13 s

$ python -m pytest tests/test_core_boundary.py -q
26 passed                                                                   4.64 s

$ python scripts/capture_goldens.py --check
7/7 ok, exit 0                       (and byte-identical: the SHA-256 pins still match)

$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh        # strict, installed wheel
574 passed, 17 deselected; both identities on 566cbf17...
fast checks: PASSED                                                        exit 0
```

```console
$ python -m pytest tests/test_crash_recovery.py tests/test_bundle_portability.py -q -m slow
15 passed, 21 deselected                                                  273.64 s
```

Identical to the Phase 0 baseline: same 15 tests, same 273 s, unchanged v1/v2 reads and resume.

### Three defects, all found by the gates, none by review

The slow gate failed three times before it passed. That is the phase working as designed, and each
failure produced a fast test so the class cannot cost four minutes again.

**1. The adapter reached back into the engine.** `core/config/adapter.py` imported `DEFAULTS` from
the OpenMM package. It translates the canonical model into *one* engine's runtime dictionary, so it
was never core code — moved to `openmm/adapter.py`, still reachable as
`md_templates.openmm.spec.adapter`. This is also what settled the design of the `spec` compatibility
package: aliasing `spec` onto `core.config` would force `core.config` to carry `adapter` so the old
import kept working, reinstating the exact dependency the phase removed. A package of aliased
members keeps both promises.

**2. Two call sites still imported the old adapter path.** Deferred imports inside functions, so no
test touched them until an end-to-end prepare ran.

**3. A moved function used a name that was no longer in scope.** `forcefield_provenance` kept
referring to `BUNDLE_SCHEMA_VERSION` after moving to the engine — a `NameError` that fires only when
the function runs.

**The guards added in response**, `tests/test_import_integrity.py`, 44 tests in 1.7 s:

* every import in every module is resolved by walking the AST — **including deferred imports inside
  functions**, which is where all of this hid. Verified against the real regression: reintroducing
  the broken adapter import fails the guard in 1.5 s rather than four minutes.
* pyflakes over the whole package for undefined names. It immediately found a fourth, quieter
  defect nobody had noticed: `core/bundle.py` still advertised `topology_counts` and
  `forcefield_provenance` in `__all__` after they moved to the engine, so `from ... import *` would
  have failed on them.

**Risk accepted.** The legacy aliases mean `__module__` on moved objects reports the new location;
anything that pickles by qualified name, or asserts on `__module__`, sees the change. The names,
commands and exceptions are pinned separately and none was lost. No scientific value moved: the
seven goldens are byte-identical and the four representative configurations produce identical hashes
through both import paths.

---

## Phase 4 — reusable OpenMM provider

**What moved.** Sixteen modules and the shipped manifest data to `src/md_templates/engines/openmm/`,
methods in their own subpackage. `src/md_templates/openmm/` is now aliases and re-exports only — a
test asserts no module there *defines* a function or class.

| layer | modules |
|---|---|
| legacy front end | `schemas`, `config`, `adapter` |
| building | `system`, `solvation`, `equilibration` |
| methods | `methods/md`, `methods/rest2` |
| artifacts | `bundle`, `bundlecheck`, `bundleinfo`, `restart` |
| execution | `platform` (was `envcheck`), `provenance`, `runner`, `cli` |

`envcheck` was renamed `platform`, which is what it does: choose and validate a platform and device.

**Compatibility matrix**, checked rather than described: 26 legacy import paths, each asserted to
resolve to the *same module object* as its new home — 16 from this phase, 10 from Phase 3.

### Four defects, each caught by a gate

**1. `python -m md_templates.openmm.cli` printed nothing and exited 0.** Running a module as
`__main__` executes the shim file, not the alias target, so it imported, aliased and stopped. A
command that silently succeeds at doing nothing is worse than one that fails. The shim now carries an
explicit `__main__` guard and deliberately skips aliasing in that case, because rebinding
`sys.modules["__main__"]` would replace the running program.

**2. `bundlev2` inside the engine no longer meant one thing.** Phase 3 split the contract; call sites
each had to know which half they wanted. Assembled once now, in `bundlev2_module()`, as the core file
format plus this engine's counts and force-field provenance.

**3. Three test paths pointed at files the move had emptied.** The peptide PDB literal is the
instructive one: the fast suite stayed green while the PDB-route slow test died with
`FileNotFoundError` three minutes in. `test_explicit_baseline`'s module scan had a docstring warning
that naming one file would retire the check silently — which is nearly what happened.

**4. The engine depended on a compatibility shim's side effect.** This is the one worth remembering.
Phase 3 left `save_restart`/`load_restart` with the provider and had the legacy `runstate` shim
attach them onto `core.persistence`, so the historical API kept working. The methods then called them
*through the core module* — correct only while something had imported the shim. On the REST2 run path
nothing had:

```
AttributeError: module 'md_templates.core.persistence' has no attribute 'save_restart'
```

Four minutes into a run, in a subprocess whose stderr the crash test discarded. The methods now
import from `..restart` where the functions live; the shim keeps attaching them for legacy callers.

Two subprocess guards now state the invariant directly: importing **only** the engine, core must not
carry `save_restart` or `topology_counts` while the methods must have both; importing **only** the
legacy path, the historical API must still work. And the crash test now reports the child's exit
code and output, because "no generation was committed before the deadline" made an import error
inside the child indistinguishable from a slow machine.

### Gate

**PASSED.**

```console
$ python -m pytest tests/ -q -m "not slow"
676 passed, 17 deselected                                                  49.11 s

$ python -m pytest tests/test_engine_provider.py -q
36 passed                                                                   3.59 s

$ python -m pytest tests/test_crash_recovery.py tests/test_bundle_portability.py -q -m slow
15 passed, 21 deselected                                                  272.76 s

$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh        # strict, installed wheel
fast checks: PASSED                                                        exit 0
```

Goldens byte-identical; `md-openmm` help, `python -m`, and the entry point all verified; slow timing
matches the Phase 0 baseline (272.8 s against 273.1 s).
