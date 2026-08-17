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

PHASE0_GATES_PLACEHOLDER
