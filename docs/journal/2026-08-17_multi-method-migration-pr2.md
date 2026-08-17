# Multi-method migration, PR 2: packaging the catalog and carrying build provenance

**2026-08-17.** Status: **implemented and locally verified.** PR 1 left the catalog usable only from
a repository checkout; it now travels inside the built distribution, and an installed copy can
resolve a real immutable identity instead of correctly refusing for want of one. Still catalog and
provenance only — no dispatch, no relocation, no runtime change.

## Scope, and explicit non-goals

Implemented: the catalog packaged into the wheel with no second tracked copy; a versioned
resource-integrity manifest; a versioned build-provenance record decided from the source being
built; an installed-catalog API; packaged identity resolution; a wheel-based CI gate; regression
tests; this journal.

Not implemented, by instruction: generic template dispatch or a general CLI; routing `prepare`, `md`
or `rest2` through descriptors; moving OpenMM code to `src/md_templates/engines/openmm/`;
template-local profiles; writing template identity into bundles or run manifests; bundle or restart
schema changes; another engine or method; the Amber-style adapter; any scientific-validation claim.

Every invariant in the instruction's list was preserved. `md-openmm`, its help text, dispatch,
profiles, scientific defaults, bundle formats, restart formats, seeds, REST2 behaviour and omega
exclusion are untouched; `src/md_templates/openmm/` was not moved.

## Base and head

| | |
|---|---|
| base branch | `openmm` at `d4af9ad` (PR 1 merged as `2389f7e`, plus the PR 2 instruction) |
| feature branch | `migration/pr2-packaged-catalog-provenance` |

## Tracked source versus generated wheel resource

The tracked root `registry.yaml` and `templates/<method>/<engine>/<variant>/template.yaml` remain the
**only** editable catalog. Nothing was added under `src/`.

```
tracked                                   generated at build time (never in the working tree)
registry.yaml                    ──┐
templates/**/template.yaml       ──┤ copied ──▶ build/lib/md_templates/core/_packaged/
                                   │             registry.yaml
                                   │             templates/**/template.yaml
                                   │             resource_manifest.json
                                   └────────▶    build_provenance.json
```

Two tracked copies would be edited independently exactly once, after which the repository would
describe two catalogs and an identity — whose entire job is to say "these bytes, that commit" — would
be ambiguous about which bytes. A `build_py` hook does the copying; `_packaged` is underscored
because it is generated. A test asserts the tracked tree contains exactly one registry and two
descriptors, and that `src/md_templates/core/_packaged/` does not exist in the checkout.

## Schemas added

Both carry independent integer schema versions, because packaging changes on a different clock from
anything scientific. Neither is tied to the registry, template, bundle, canonical-configuration or
run-state versions.

**`resource_manifest.json`, schema version 1.** `resources` maps **logical repository path** →
SHA-256 of exact bytes; `references_validated_at_build` lists the `repository_references` the build
checked. Keyed by logical path rather than by location inside the wheel, because the logical path is
what an identity is made of and a packaging detail must never leak into one. Paths are validated as
normalised and non-escaping — traversal in a manifest is a read primitive.

**`build_provenance.json`, schema version 1.** `resolved`, `source_state`, `commit_sha`,
`canonical_url`, and `resource_manifest_sha256`, which binds the record to the exact catalog it
describes so a commit cannot be paired with a catalog it did not describe. The model enforces
`resolved` ⟺ a full lowercase 40-hex `commit_sha`, and only two states may resolve, so a record
cannot describe a dirty tree while carrying a SHA a careless reader would use anyway.

Neither record carries free text or any machine-local value. `source_state` is a closed set and the
human sentence is produced at read time — that is what makes the records deterministic.

## The provenance state machine

Decided from the source being built. Never supplied, never overridden.

```
              ┌─ git rev-parse HEAD fails ──▶ .catalog_source_provenance.json present?
              │                                 │                    │
              │                            yes, and its              no, or its manifest
              │                            manifest hash             hash disagrees, or it
              │                            still matches             was itself unresolved
              │                                 │                    │
              │                                 ▼                    ▼
              │                        verified-source-archive   no-verifiable-git-provenance
source ───────┤                              RESOLVED                 refused
              │
              └─ HEAD known ─┬─ git status failed/timed out ─▶ unverifiable-git-status   refused
                             ├─ uncommitted or untracked ────▶ dirty-source-tree         refused
                             └─ clean ───────────────────────▶ clean-git-checkout        RESOLVED
```

* a failed or timed-out status check is **unresolved, never clean** — a tree that cannot be shown
  clean has not been shown clean;
* untracked non-ignored files count as dirty, because they are part of what was built;
* **no environment variable overrides any of this**, and there is deliberately no "allow dirty but
  trust HEAD" option. That option is precisely the defect PR 1's review removed from
  `resolve_identity`; re-adding it one layer down would be the same mistake wearing a build hat.

**The sdist route.** `sdist` writes the checkout's provenance into the archive, and a wheel built
from that archive inherits the commit **only while the catalog bytes still hash to what the record
described**. An archive whose templates were edited after unpacking falls back to
`no-verifiable-git-provenance`, verified by test in both directions.

One ordering detail was a real bug before it was a design point: `sdist` creates its release tree
*inside* the project directory, so provenance computed during `make_release_tree` saw the build's own
scratch space and called every clean checkout dirty. Provenance is now decided in `run()`, before
anything is built. The tree is judged as it was, not as the build transiently made it.

**Trust boundary, stated plainly.** This detects bytes that drifted, a resource that went missing, a
metadata record paired with the wrong catalog, and a build from a tree nobody can name. It does
**not** authenticate GitHub, and it does not defend against an attacker who can rewrite a wheel *and*
both metadata records consistently. Nothing here is a signature, and the code says so where it
matters.

## Public API added

```python
from md_templates.core import (
    load_packaged_catalog,      # -> TemplateCatalog, from the installed distribution
    resolve_packaged_identity,  # -> TemplateIdentity, or a typed refusal
    load_packaged_provenance,   # -> BuildProvenance, resolved or not, for diagnostics
    verify_packaged_catalog,    # -> PackagedCatalog for an explicit root
    describe_packaged_catalog,  # -> a small summary dict, or None
    packaged_catalog_available, # -> bool, never raises
)
```

Names follow the instruction's suggested pair. `load_catalog(root)` is unchanged for checkouts.

**No commit may be supplied by the caller here either.** `resolve_packaged_identity` takes a template
reference and an optional `root` selecting *which* packaged catalog to check — never a SHA and never
a `TrustedProvenance`. A convenience API accepting one would reintroduce the PR 1 defect through a
new door: the point of an installed copy is that it can only claim what its build proved. A test
asserts the signature.

Three checks run before an identity is returned — integrity, then provenance, then binding. Integrity
also refuses a resource that is *present but unclaimed*, since a descriptor smuggled in beside the
manifest would otherwise be invisible to hashing while still looking like part of the catalog.
Listing succeeds even when provenance is unresolved: knowing what a distribution contains is useful
even when its bytes cannot be named. `UnresolvedBuildProvenanceError` subclasses `NoProvenanceError`
because to a caller it is the same refusal, discovered a different way.

**The `repository_references` policy.** A checkout keeps its existence check. A wheel cannot have
one — those entries name documentation that does not ship — so the build validates them against the
source commit, the manifest records which were validated, and `TemplateCatalog.reference_policy`
reports `validated-at-build` instead of `source-tree-existence`. Nothing pretends a `docs/` path
exists in site-packages, and nothing silently skips the check.

## Tests and results

Environment: `md-wheel-check` (Python 3.11.15, setuptools 84.0.0, build 1.5.0, pydantic 2.11.10) for
packaging; `escort-ais-explicit-target` for the full suite. Wheels are built `--no-isolation` and
installed with `pip install --no-deps --no-index --target`, so the packaging tests need no network.

```console
$ python -m pytest tests/test_packaged_catalog.py -q
47 passed                                                                  16.63 s

$ python -m pytest tests/test_template_catalog.py tests/test_template_identity.py -q
131 passed                                                                  6.19 s

$ python -m pytest tests/ -q -m "not slow"
491 passed, 17 deselected                                                  25.85 s

$ python scripts/capture_goldens.py --check
7/7 ok, exit 0

$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh          # from a CLEAN committed tree
wheel built, contents inspected (including all four packaged catalog paths), installed;
public commands run from outside the checkout;
packaged catalog: clean-git-checkout (resolved=True), both templates listed,
  reference policy validated-at-build, both identities on the build commit;
5 profiles validated; YAML/JSON hashes identical; 491 passed / 17 deselected
fast checks: PASSED                                                        exit 0
```

The gate was run twice deliberately. From the **uncommitted** working tree it reported
`dirty-source-tree` and exercised the refusal path — which is the correct outcome, and the reason the
script checks the refusal rather than skipping when provenance is unresolved. From a **clean
committed** tree (`bf85d42`, the implementation stashed away) it resolved both identities to that
exact commit:

```
https://github.com/csy0000/MD-templates@bf85d42411d3fadf6f33ac871054379d814e5a0f#templates/conventional-md/openmm/explicit-water/template.yaml
https://github.com/csy0000/MD-templates@bf85d42411d3fadf6f33ac871054379d814e5a0f#templates/rest2/openmm/explicit-water/template.yaml
```

That pair is the whole point of PR 2: an installed wheel, with no checkout and no Git, naming the
commit its bytes came from.

`491 = 444 (PR 1 head) + 47`. No existing test was deleted, skipped, weakened or rewritten.

**The new tests fail against the merged PR 1 base.** Run against a worktree at `d4af9ad` with only
the test file added: collection fails outright — `ModuleNotFoundError: No module named
'md_templates.core.packaged'`. That is a weaker demonstration than assertion failures would be, and
is reported as what it is: the module under test does not exist at the base, so nothing finer can be
observed there.

### Source-to-wheel equivalence

A canonical projection — canonical URL, registry schema version, identity scheme, every registry
entry's id/path/method/engine/aliases/tags, and every descriptor's full `model_dump(mode="json")` —
is compared between `load_catalog(<clean temp checkout>)` in-process and `load_packaged_catalog()` in
a **subprocess running from an unrelated directory** with only the install directory on `PYTHONPATH`.
They are equal. That is the test that makes "the same catalog" a fact rather than an intention.

### Clean-build evidence

From a clean committed temporary checkout: `resolved = true`, `source_state = clean-git-checkout`,
`commit_sha` exactly the 40-character `HEAD`. Both packaged identities carry that SHA and the logical
`templates/<method>/<engine>/explicit-water/template.yaml` path. A test scans the canonical string
for `main`, `master`, `0.1.0`, `site-packages`, `_packaged`, `.whl`, the install path and `2026` —
none appears. Two builds of the same commit produce **byte-identical** manifest and provenance, and
the build leaves the source tree unmodified.

### Dirty-build evidence

A tracked descriptor edited but not committed: `resolved = false`, `source_state =
dirty-source-tree`, `commit_sha` absent — and the dirty `HEAD` string appears nowhere in the metadata
bytes at all. Listing still works. `resolve_packaged_identity` raises
`UnresolvedBuildProvenanceError` naming the state and saying what to do. Passing the same `HEAD`
through `commit_sha=` or `trusted=` raises `TypeError`, because neither parameter exists. An
untracked non-ignored source file produces the same refusal. A failed `git status` is exercised by
monkeypatching the Git call in `decide_provenance` and yields `unverifiable-git-status`, unresolved.

### Installed, no-Git, offline evidence

A wheel is built and installed, then **the source repository is deleted outright**. From an unrelated
working directory, with only the install on `PYTHONPATH`: the catalog lists both templates and
identity resolves to the build commit. No `.git` exists anywhere in the install tree. A separate
subprocess replaces `socket.socket`, `create_connection` and `getaddrinfo` with raising stubs, loads
the catalog, resolves an identity, and asserts that none of openmm, openff, rdkit, mdtraj, parmed,
openmmtools, numpy, scipy or pandas reached `sys.modules`.

### Tamper and truncation

Seventeen cases, each required to fail before any identity is returned: changed registry bytes,
changed descriptor bytes, missing descriptor, missing provenance, malformed provenance, unsupported
provenance schema version, five malformed or abbreviated SHAs, manifest path traversal, manifest hash
mismatch, an unclaimed extra resource, a registered-but-unpackaged descriptor, a symlinked packaged
resource, a foreign repository URL, and two model-level cases (resolved without a SHA, unresolved
with one).

The case worth naming: **both records rewritten consistently.** Tampering with the manifest and then
repairing `resource_manifest_sha256` to match makes the binding check pass — so integrity has to
catch it, and does. A test that only ever corrupted one record would not have shown that.

## Compatibility

* All seven committed compatibility goldens still match: `capture_goldens.py --check` exits 0.
  **None was regenerated.**
* Profile selection, profile hashes, configuration projections and hashes, seed derivation, bundle
  and run-state contracts and REST2 defaults are unchanged — asserted by the existing 20 golden tests
  inside the 491.
* `git diff d4af9ad..HEAD -- src/md_templates/openmm/` is **0 lines**.
* `md-openmm --help` for the root command and all five subcommands is byte-identical to `d4af9ad`.
* No bundle, run manifest, checksum domain, continuity path or hash projection gained a template
  field. The existing PR 1 tests asserting that still pass.

The only changes outside `md_templates/core` and the build files are documentation.

## CI status actually observed

**Reported by the reviewer: `fast` and `integration-cpu`, run #10, both passed after review round
1.** That is the first remote run to exercise the strict gate introduced in round 1, and it resolves
the question left open by PR 1, whose head carried zero commit-status contexts.

**I could not independently confirm it, and am not claiming to have.** The API token on this machine
lacks Actions and Checks read permission — `actions/runs`, `commits/<sha>/check-runs` and
`commits/<sha>/status` all return **HTTP 403 "Resource not accessible by personal access token"**,
re-checked at the start of review round 2. So the workflow result is recorded as *reviewer-observed*;
every other result in this journal is local evidence produced and checked here. A first-hand check
needs the Actions tab, or a token carrying `actions: read`.

Run #10 predates review round 2, which changed both the digest and the gate's preconditions, so the
next Actions run is the first to exercise closed-world coverage and the mandatory `--expect-commit`
remotely.

## Review round 1: three findings, all real

Review on PR #2 raised three, each a case where something that looked verified was not.

**1. Provenance could be inherited from an enclosing repository.** `git -C <dir>` walks upwards, so a
source tree unpacked anywhere inside an unrelated repository reported *that* repository's HEAD — and
reported it **clean**, because the enclosing tree genuinely is clean while ignoring the copy. Every
check passed and the identity named a commit from a different project. `inspect_provenance` now
requires `git rev-parse --show-toplevel` to resolve to the directory it was asked about, compared by
`realpath`, checked before HEAD. A directory inside a repository is treated as having no Git metadata
at all, which is the honest description. This fixed source-tree identity as well as build provenance
— `resolve_identity(catalog, root=...)` had the same hazard.

**2. Archive verification bound only the catalog.** The resource manifest covers `registry.yaml` and
the descriptors, which leaves the code that reads and verifies them unbound: unpack an sdist, rewrite
`packaged.py`, `build_support/catalog.py` or any engine module, rebuild, and the wheel inherited the
archive's clean commit — a commit naming code it never contained. `BuildProvenance` gains
`source_tree_sha256`, one digest over `setup.py`, `pyproject.toml`, `MANIFEST.in`, `registry.yaml`
and everything under `src/`, `build_support/` and `templates/`. An archive is taken at its word only
when **both** digests agree. `.egg-info` and `__pycache__` are excluded because they appear *during* a
build; including them would make the digest depend on whether anything had been built before.

**3. The gate accepted either branch.** A clean build was verified and an unresolved one merely had
to refuse — so the gate passed on precisely the wheels that cannot name their own source, silently,
because an unresolved build looks healthy until something asks it for an identity. Strict is now the
default: resolved provenance is required and `--expect-commit` must match it exactly, with
`fast_checks.sh` passing `git rev-parse HEAD`. `--allow-unresolved` remains for a dirty tree but must
be selected explicitly via `FAST_CHECKS_ALLOW_UNRESOLVED=1`, defaulting to off, and the mode is
printed on every run.

Sixteen regression tests were added. The finding-1 group first asserts that plain `git -C` *does*
resolve to the outer repository from the vendored directory — the hazard is demonstrated before it is
fixed, so the test cannot pass vacuously.

```console
$ python -m pytest tests/test_packaged_catalog.py -q
63 passed                                                                  23.66 s

$ python -m pytest tests/test_template_catalog.py tests/test_template_identity.py -q
131 passed                                                                  5.71 s

$ python -m pytest tests/ -q -m "not slow"
507 passed, 17 deselected                                                  32.73 s

$ python scripts/capture_goldens.py --check
7/7 ok, exit 0

$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh       # clean tree, STRICT mode
  source state:       clean-git-checkout (resolved=True)
  mode:               strict
  expected commit:    21b7f3a2deff067264a536eb6f7fcad5a67da4df (matches)
  identity:           ...@21b7f3a2...#templates/conventional-md/openmm/explicit-water/template.yaml
  identity:           ...@21b7f3a2...#templates/rest2/openmm/explicit-water/template.yaml
  ok: offline, no heavy dependencies, no checkout, no Git
507 passed, 17 deselected
fast checks: PASSED                                                        exit 0
```

`507 = 491 + 16`. No golden was regenerated and no existing test was weakened.

## Review round 2: closed-world coverage and a mandatory expected commit

**1. The source digest was an allowlist, and allowlists rot.** Round 1 bound `src/`,
`build_support/`, `templates/` and four root files — which left `README.md`, build configuration, a
descriptor's referenced documentation and *any newly added file* invisible. An unpacked archive could
be edited in those places and still inherit the commit it named.

The shape is now inverted: hash **every regular file** in the tree, keyed by normalised logical path,
minus a short explicit list. Each exclusion earns its place by being **regenerated during a build** —
`__pycache__`, `*.pyc`/`*.pyo`, any `*.egg-info` directory, top-level `build/` and `dist/`, `.git/` —
because including them would make the digest depend on whether anything had been built before, and
two builds of one commit must stay byte-identical. The provenance record excludes itself for the
obvious reason: its own digest field is written from this value. Symlinks hash their link target
rather than what it points at, so replacing a file with a link moves the digest.

Coverage that has to be remembered is eventually forgotten. "Everything, minus a named list of
generated things" is the only version that stays true as the repository grows.

**2. `--expect-commit` was optional, so strict mode was weaker than it looked.** Resolved provenance
on its own only shows the wheel names *some* commit; without an expected value a stale wheel built
from an older checkout passed while proving something about source nobody was looking at. It is now
mandatory whenever `--allow-unresolved` was not selected.

The SHA is acquired in a new **step 0**, before anything is built, by `scripts/ci/expect_commit.sh`.
Two reasons for the separate script: a checkout that cannot name itself now fails in a second rather
than after a wheel build and an install, and the precondition is testable with a stubbed `git`
without building anything. It refuses an empty or non-40-hex SHA rather than letting either degrade
into "no expected commit" — which is exactly the check being skipped.
`FAST_CHECKS_ALLOW_UNRESOLVED=1` still permits both an unresolved build and a missing SHA, still
defaults to off, and is announced in step 0.

Twenty regression tests. The closed-world group edits `README.md`, `setup.cfg`, a referenced doc and
a brand-new file in an unpacked archive, each requiring unresolved provenance and a refused identity;
another confirms the opposite direction, that regenerating `.egg-info`, `__pycache__`, `build/`,
`dist/` and even rewriting the provenance record leaves the digest unchanged. The gate group runs the
whole shell gate with a failing `git` stub and asserts it refuses **before** the wheel step, in under
a minute.

```console
$ python -m pytest tests/test_packaged_catalog.py -q
83 passed                                                                  31.13 s

$ python -m pytest tests/test_template_catalog.py tests/test_template_identity.py -q
131 passed                                                                  5.84 s

$ python -m pytest tests/ -q -m "not slow"
527 passed, 17 deselected                                                  39.08 s

$ python scripts/capture_goldens.py --check
7/7 ok, exit 0

$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh       # clean tree, STRICT mode
  expected commit: d144e8c4d4275b9583493f4d1efa98e0d5580248
  source state:       clean-git-checkout (resolved=True)
  mode:               strict
  expected commit:    d144e8c4d4275b9583493f4d1efa98e0d5580248 (matches)
  ok: offline, no heavy dependencies, no checkout, no Git
527 passed, 17 deselected
fast checks: PASSED                                                        exit 0
```

`527 = 507 + 20`. No golden regenerated; no existing test weakened.

## Limitations and deferred work

* **Not a signature.** Integrity plus provenance detects drift, loss and mismatch. It cannot detect a
  consistent rewrite of a wheel and both its records by someone with write access to site-packages,
  and it authenticates nothing about GitHub.
* **`pyyaml` and `pydantic` are now build requirements.** Building with isolation fetches them, so a
  fully offline build needs `--no-isolation` in an environment that already has them. That is what
  the tests do; documented rather than worked around.
* **The reference policy is a record, not a re-check.** An installed catalog trusts that the build
  validated `repository_references`; it cannot re-verify paths that do not ship. It says so via
  `reference_policy` rather than implying otherwise.
* **`repository_references` are checked for existence, not meaning** — unchanged from PR 1.
* **Dirtiness remains whole-tree**, deliberately conservative, unchanged from PR 1.
* Nothing here is scientific validation, and no smoke test is described as one. Both descriptors
  remain `scientific_status: unvalidated` with the 2 fs / 4 fs HMR gate and REST2 ladder suitability
  open.

Deferred to later PRs, unchanged: generic dispatch and a general CLI; relocating the OpenMM
implementation; template-local profiles; writing identity into bundles and run manifests; further
engines or methods; the Amber-style adapter.
