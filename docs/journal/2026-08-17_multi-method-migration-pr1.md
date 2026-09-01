# Multi-method migration, PR 1: the template catalog and immutable identity

**2026-08-17.** Status: **PR 1 is implemented and locally verified.** It is catalog metadata plus
validation. No runtime module was moved, no default changed, no hash moved, and nothing dispatches
through the new descriptors — every one of them says so in its own `implementation.dispatch` field
rather than leaving a reader to infer it.

## Request and exact scope

`claudecode-instructions/20260817_multi-method-migration-pr1.md`, PR 1 only, on a feature branch
`migration/pr1-template-catalog` cut from `openmm` at `4d21838`. Seven deliverables: a root
`registry.yaml`, two `template.yaml` descriptors, strict typed validation for both formats, immutable
template identity from a full Git commit SHA plus the exact template path, compatibility goldens
captured before any structural edit, the tests, and this journal.

Everything in the instruction's forbidden list was left alone: no module under
`src/md_templates/openmm/` was moved, renamed or behaviourally changed; no
`src/md_templates/engines/openmm/` was created; `md-openmm` gained no command and changed no
behaviour; nothing routes `prepare`, `md` or `rest2` through a descriptor; the canonical
configuration models, resolution precedence and hash projections are untouched; profiles, profile
IDs, default selection and scientific defaults are untouched; bundle schemas 1 and 2 are untouched;
no template identity was written into a bundle, run manifest or continuity hash; restart, checkpoint,
committed-generation and quarantine behaviour are untouched; the REST2 ladder, exchange behaviour,
omega classification and default omega exclusion are untouched; seed derivation is untouched; no
Amber-style adapter was written; no new scientific validation is claimed; no legacy path was removed;
nothing was released.

## Baseline and final commit

| | |
|---|---|
| instruction's stated baseline | `1e80a2061613ac40a10143c28044c661e1d5d4ac` |
| branch head when work started | `4d21838009804a47047348e80c8ba352e3d8539e` |
| feature branch | `migration/pr1-template-catalog`, cut from `4d21838` |

The only commit between the stated baseline and `4d21838` is `4d21838` itself, which adds the
instruction file. Nothing intervening needed preserving.

## Files changed

Twenty-one files added, two edited.

```
registry.yaml
templates/conventional-md/openmm/explicit-water/template.yaml
templates/rest2/openmm/explicit-water/template.yaml
src/md_templates/core/{__init__,paths,template,registry,identity}.py
scripts/capture_goldens.py
tests/goldens/{README.md,configuration_hashes,profiles,format_equivalence,
               seed_derivation,bundle_contract,runstate_contract,rest2_defaults}.json
tests/{test_compat_goldens,test_template_catalog,test_template_identity}.py
docs/journal/2026-08-17_multi-method-migration-pr1.md
CHANGELOG.md      (edited: an Unreleased entry)
README.md         (edited: a "Template catalog" section)
```

Nothing else. `git diff 4d21838..HEAD -- src/md_templates/openmm/` is **zero lines**, and
`git diff --find-renames --diff-filter=RD` over the same range reports nothing, so no file was moved
or deleted anywhere in the repository.

## Registry and descriptor model decisions

**The registry is a discovery index and nothing else.** It answers which templates exist and where
each descriptor lives. Settings, profiles, scientific defaults and evidence live in the descriptors,
because a registry that duplicated them would be a second copy of the truth and the two copies would
disagree the first time someone edited one.

**No commit SHA appears in `registry.yaml`.** The same content has to remain valid at every later
commit, with identity resolved from whichever immutable commit contains the file. A SHA written into
the registry would be stale the instant it was committed, since committing it changes the commit. A
test asserts the file contains no 40-hex string and no `sha:`-like key.

**Validation runs in both directions between the registry and the tree.** The registry's own schema,
duplicate IDs, duplicate paths and path normalisation are checked; then the descriptor must exist,
must parse, and must agree with its registry entry on `template_id`, `method` and `engine`. A
registry that validates while disagreeing with the descriptor it points at is worse than one that
fails, because it looks correct.

**Descriptor fields chosen for what they let a consumer decide**, not for completeness: stable
logical ID, display name and summary; method identity with API version and aliases; engine identity
with API version; identity mode; the current implementation binding; declared input routes;
operational capabilities with an explicit `supported` flag; method features; readable and writable
bundle schema versions; restart guarantees and limitations; the profile provider; and validation.

Three of those deserve their reasons stated.

*Implementation status and scientific status are separate fields with separate enums.* They are
different claims and they move independently. "The code runs, is covered by tests, and produces
bundles that relocate and resume" says nothing about whether the physics is validated for
production. A single collapsed `status` is precisely how a template that merely executes comes to be
described as validated. Both shipped descriptors are `implementation_status:
implemented-and-tested`, `scientific_status: unvalidated`.

*Evidence carries its own kind and scope, and the model refuses to let weak evidence raise a status.*
`kind: cpu-smoke` may never establish general scientific validation — CPU smoke profiles are
picoseconds of unvalidated settings, and the instruction prohibits them as scientific evidence.
`kind: regression` may not either: regression tests show the implementation did not change. Nor may
anything whose `scope` is not `general`, which is what keeps the RGD ten-rung result system-specific.
And `scientific_status: validated` requires at least one evidence entry that does establish it, so
the status cannot be raised by editing one field. These are validators, not conventions, because the
next editor is the person the rule has to hold against.

*Profiles are a named provider reference with `template_local: false`.* The profile files are
package resources inside the OpenMM implementation and PR 1 moves and copies nothing. A descriptor
listing them as template-local resources would describe a layout that does not exist, so the model
refuses `template_local: true` outright until they actually move. Template-local profile migration is
deferred to the method-activation PRs.

**Deviations from the instruction's suggested names:** none in the registry. The instruction's
example fields (`schema_version`, `repository.canonical_url`, `identity.scheme`,
`identity.canonical_form`, `templates[].{template_id,template_path,method,method_aliases,engine,tags}`)
are used verbatim.

**Schema versions are validated in `mode="before"`.** Pydantic's lax coercion turns `"1"`, `1.0` and
`True` into the integer `1`, so a registry declaring a string or a boolean version would have
validated silently. A schema version is a discrete contract identifier, not a number to be converted
into. The template schema version is independent of the bundle (2), canonical-configuration (1) and
run-state (1) versions; a test asserts the independence at the source level — the catalog package
does not name any of the runtime's schema constants.

## Module layout, and why four modules rather than three

The instruction suggested `core/{__init__,identity,registry,template}.py`. I added `core/paths.py`.

The registry and identity both need the same answer to "is this a legal reference to a file inside
this repository?", and identity is constructed from the **normalised** path. Two copies of that rule
would eventually diverge, and the failure mode is specific and bad: if the registry accepted
`templates/rest2/openmm/../openmm/explicit-water/template.yaml` while identity accepted the collapsed
form, two strings would name one template and the identity would stop being a function of the
template. One module, one rule, both callers.

The rule rejects rather than repairs. Nothing collapses, resolves or normalises a path on the
caller's behalf, because repairing would mean two spellings map to one identity while a reviewer
reading the registry sees only one of them. `..` is refused even when it would resolve inside the
repository. Backslashes are refused rather than converted, since `os.path.normpath` splits `a\b`
into one component on POSIX and two on Windows — a registry validated on one platform would describe
a different tree on the other.

**Dependency floor: YAML and pydantic.** Nothing in `md_templates.core` imports OpenMM, OpenFF,
RDKit, mdtraj, numpy or scipy, so listing and validating the catalog works in a minimal environment
on a machine that could never run a simulation. Two tests hold the boundary: a source scan, and a
subprocess that imports the package, loads the catalog, and asserts none of those top-level modules
appears in `sys.modules`. The subprocess matters — this test process has already imported OpenMM
through other tests, so an in-process check would prove nothing.

No new dependency was added.

## The identity algorithm, and its refusals

Identity is the tuple `(canonical repository URL, full 40-character Git commit SHA, exact normalised
template path)`, canonically

```
https://github.com/csy0000/MD-templates@<40 hex>#templates/<method>/<engine>/<variant>/template.yaml
```

Both the structured fields and the string are kept: the string is what gets quoted in a report, the
fields are what a consumer compares, so nobody has to parse the string back apart to ask "same
repository, different commit?".

**The caller cannot supply the commit.** `resolve_identity` has no `commit_sha` parameter: an
identity asserts "these bytes are in that commit", and a caller-supplied SHA asserts nothing. Where
the commit comes from is decided by whether the root is a Git checkout:

1. **in a checkout** — the tree must be clean and the commit is HEAD. A `TrustedProvenance` may also
   be given, but it must *equal* HEAD; it is a cross-check on build metadata, never an override;
2. **outside a checkout** — only explicit `TrustedProvenance` is accepted, and absent that this
   raises rather than inventing an identity.

`build_identity` remains for purely syntactic construction — parsing, formatting, comparison — where
nothing is being claimed about a working tree.

Refusals, each of them typed:

* **Not a full 40-character hex SHA → `IdentityError`.** Abbreviations are refused however long,
  because a prefix that is unique today may collide after the next commit and a stored short SHA is
  a latent ambiguity. Branch names, tags, package versions, template semantic versions and timestamps
  are refused for the reasons that make them not identities: they move, or they cover many commits,
  or they name nothing.
* **Dirty checkout → `DirtyWorkingTreeError`.** HEAD describes what was committed; the files on disk
  may be something else. Silently resolving to HEAD would stamp an artifact with an identity that
  does not reproduce it, and later the wrong answer is indistinguishable from the right one. The
  refusal names HEAD in its message — it is available for a developer to look at, it is just not
  usable as identity. `inspect_provenance()` stays available for that inspection, explicitly
  labelled unresolved. Untracked files count as dirty: a new untracked template would change what
  the tree contains.
* **No Git metadata and no trusted SHA → `NoProvenanceError`**, whose message names what will *not*
  be substituted, so nobody reads it as "supply a version instead".
* **A path that is not registered → `UnknownTemplateError`.** An identity is only ever minted for a
  template the registry lists.
* **A `git status` that fails or times out is read as DIRTY**, not clean. A tree that cannot be shown
  to be clean has not been shown to be clean.

Git is invoked with an explicit argument list, `shell=False`, `check=False` and a 10-second timeout;
a test asserts all four on every call. **Nothing touches the network.** The canonical URL is a name,
not an endpoint, and a validation step that needed GitHub reachable would fail exactly where
reproducibility matters most — an offline compute node, a locked-down runner. The no-network test
runs in a subprocess with `socket.socket`, `create_connection` and `getaddrinfo` replaced by raising
stubs.

Template identity is **not** part of any scientific or continuity hash in PR 1, and a test asserts
that no runtime module under `src/md_templates/openmm/` references `md_templates.core` at all.

**Catalog loading refuses symlinks.** `is_file()`, `read_text()` and `exists()` all follow symlinks
silently, so a committed `template.yaml` *symlink* could point at mutable bytes outside the checkout
while Git still reported the tree clean — the link itself is unchanged. SHA plus path would then name
content the commit does not contain, which is the one thing the identity exists to prevent.

`resolve_within_repository()` walks every component below the repository root and refuses any that is
a symlink — components, not just the final name, because a symlinked *directory* redirects everything
beneath it just as effectively. It then verifies containment against `realpath` as a second line,
which also catches a link swapped in between the walk and the read. It guards the descriptor path,
every `repository_references` entry, and `registry.yaml` itself.

Links that stay inside the repository are refused too. An internal link is representable in the
commit, so that case is arguably safe; permitting it would mean deciding per link whether the target
is both inside the tree and covered by the same commit, which is easy to get subtly wrong and buys
the catalog nothing. The root itself is not component-checked, since a repository legitimately sits
under a symlinked parent — containment is checked against its resolved form instead, which handles
that correctly.

## Compatibility goldens, and how they were generated

Captured **before** any structural edit, and committed as the first commit on the branch
(`22e2880`), so they describe the baseline rather than the result.

`scripts/capture_goldens.py` writes seven JSON files into `tests/goldens/`, 36 kB in total:

| file | contract frozen |
|---|---|
| `configuration_hashes.json` | canonical-JSON hash, all five projection hashes, selected profile and profile hash, and the scientifically meaningful resolved values, for MD and REST2 on both declared routes |
| `profiles.json` | every packaged profile's schema version, route, method, `is_default` flag and hash, plus what the `default` alias selects for each route/method pair |
| `format_equivalence.json` | the same document as YAML and as JSON canonicalises to identical bytes and identical hashes |
| `seed_derivation.json` | `master_seed + index in STAGE_ORDER`, with explicit per-stage overrides preserved |
| `bundle_contract.json` | bundle schema version, the logical-role map, the checksum and original-input locations, the separated count fields |
| `runstate_contract.json` | run-state schema, restart/committed/quarantine locations, continuity and extension path sets, restart member naming |
| `rest2_defaults.json` | shipped ladders, omega exclusion, proline classification |

Each of the four representative configurations declares **only** the system and the method
discriminator, so everything else in its hashes comes from the shipped default profile. A profile
edit therefore moves a golden, which is the point of capturing them.

Where exact bytes are environment-dependent — a serialized `System`, a checkpoint, a prepared bundle,
a trajectory — the fixture freezes the semantic contract instead. Freezing those bytes would produce
failures caused by the OpenMM build and the toolkit versions rather than by this repository's
contracts. Both such generators say so in their docstrings, and the goldens' README says it again.
No trajectory, checkpoint, System XML, environment or run directory is committed.

Determinism: every input is a literal in the capture script or a resource shipped inside the package.
Nothing is timestamped, nothing is read from a run directory, nothing depends on the working tree.

Regeneration is only ever the explicit `python scripts/capture_goldens.py`. The tests consume the
committed files, and one test asserts that calling every generator rewrites nothing on disk — a suite
that regenerated its own expectations would pass whatever the code did. `--check` compares without
writing, for CI.

## Confirmations the instruction asks for explicitly

**Template identity is not yet written into bundles or runs.** Nothing in
`src/md_templates/openmm/` imports `md_templates.core`; a test scans every runtime module for that
import and for the string `template_identity`. No bundle manifest, run manifest, checksum domain,
continuity path or hash projection gained a template field, and a golden test asserts the string
`template` appears in none of the frozen configuration projections.

**Runtime dispatch remains legacy.** `md-openmm` is unchanged and both descriptors record
`implementation.dispatch: legacy-direct`, `implementation.binding: current-openmm-implementation`,
`python_namespace: md_templates.openmm`, `cli_command: md-openmm`. `--help` output for the root
command and for `prepare`, `md`, `rest2`, `bundle` and `config` was captured from a worktree at
`4d21838` and from the branch head and diffed: **identical, all six.**

## Test commands, counts, durations, results

Environment: the documented conda environment (`escort-ais-explicit-target`), Python 3.11.15,
OpenMM 8.5.1, pydantic 2.11.10, with the environment's `bin` on `PATH` so AmberTools' `sqm` and
`tleap` resolve. `PYTHONPATH=$PWD/src`.

```console
# baseline, before any edit on this branch
$ python -m pytest tests/ -q -m "not slow"
293 passed, 17 deselected                                                   4.37 s

# after PR 1, including the two review fixes
$ python -m pytest tests/ -q -m "not slow"
444 passed, 17 deselected                                                   9.80 s

$ python -m pytest tests/test_template_catalog.py tests/test_template_identity.py -q
131 passed                                                                  7.03 s

$ python -m pytest tests/test_compat_goldens.py -q
20 passed                                                                   1.91 s

$ python scripts/capture_goldens.py --check
7/7 ok, exit 0

# the focused bundle and restart gate
$ python -m pytest tests/test_crash_recovery.py tests/test_bundle_portability.py -q -m slow
15 passed, 21 deselected                                                  271.82 s
```

The slow gate needs AmberTools on `PATH` for AM1-BCC charges. A first attempt ran with the
environment's `python` invoked by absolute path but its `bin` not on `PATH`, and `prepare` refused
with `sqm: not on PATH`. That is the environment check working as intended, not a defect; the run
above has `PATH` set and passes.

```console
# the repository's supported full gate, against the INSTALLED WHEEL with no checkout on the path
$ env -u PYTHONPATH bash scripts/ci/fast_checks.sh
wheel built and inspected, installed, public commands run from outside the checkout,
all 5 profiles validated, YAML/JSON hashes identical, 444 passed / 17 deselected
fast checks: PASSED

$ env -u PYTHONPATH bash scripts/ci/integration_cpu.sh
both canonical routes prepared, relocated with their source directories deleted, validated
--deep, inspected, relocate-checked; REST2 run fresh and resumed; MD run fresh and resumed;
contiguous attempt indices, one CSV header, contiguous chunks, 2 invocations recorded;
a corrupted committed checkpoint forced the State fallback and it was announced;
validate and inspect run with the network disabled
CPU integration: PASSED
```

The full gate is the meaningful one: it runs against the **installed wheel** with no checkout on the
path, and it refuses to start if `md_templates` resolves into the repository. It passing after PR 1
is what shows the catalog changed no runtime behaviour — the same prepare, relocate, run, resume and
offline-validate sequence produces the same result it did before.

The 151 new tests are 87 catalog + 44 identity + 20 compatibility. No existing test was deleted,
skipped, weakened or rewritten; the non-slow count moves by exactly the number added (293 → 444).

The slow gate and `integration_cpu.sh` were run before the review fixes and not rerun after. Both
exercise `md_templates.openmm` only — the review fixes touch `md_templates.core` exclusively, which
no runtime module imports, and the 0-line runtime diff below still holds. The four gates the review
asked to rerun were all rerun.

One descriptor defect was found by a test rather than by reading: the conventional-MD template listed
`cpu-smoke-v1` among its provided profiles, and that profile is `method: rest2`. Fixed in `bd4fe43`.
The cross-check lives in a test, not in the model — the catalog package may not import the OpenMM
implementation, so a descriptor cannot validate its own profile references against the shipped files.
That pairing is what keeps the dependency boundary from becoming an unchecked claim.

## Review round 1: two identity-integrity defects, both real

Review on PR #1 requested changes on two counts. Both were genuine holes in the property the whole
design exists to provide — that an identity names bytes that reproduce — and both are fixed.

**1. The explicit-SHA shortcut bypassed provenance.** `resolve_identity(..., commit_sha=...)` skipped
the dirty-tree check *and* any comparison with HEAD, so a dirty checkout could pass its own HEAD and
receive a resolved identity, and a clean checkout could be stamped with any unrelated 40-hex string.
The parameter is removed rather than documented against: an identity asserts a relationship between
bytes and a commit, and a caller-supplied SHA asserts nothing. Resolution now proves provenance — in
a checkout the tree must be clean and the commit is HEAD, with trusted provenance permitted only as
an equality cross-check; outside a checkout only trusted build provenance is accepted.
`build_identity` keeps the syntactic path, with a docstring saying plainly what it does not check.

**2. Catalog loading followed symlinks.** Described above under the identity algorithm. Fixed with
`resolve_within_repository()`, applied to the descriptor path, every `repository_references` entry,
and `registry.yaml` itself.

**Nine regression tests, verified to fail against the pre-fix code.** They were run against a
worktree at `b7f25c1`, the commit under review: 9 failed, 122 passed. Against the fix: 131 passed.
A regression test that has never been seen to fail is a hope, not a test.

* identity: the `commit_sha` parameter is gone from the signature; dirty checkout naming its own
  HEAD; dirty checkout with matching trusted provenance; clean checkout with a mismatched trusted
  SHA; clean checkout with an agreeing one; trusted provenance outside Git.
* symlinks: external descriptor symlink (which still *parses*, so only the symlink rule catches it);
  symlinked directory component; external `repository_references` symlink; an internal symlink; a
  symlinked `registry.yaml`; and a check that the shipped catalog contains no symlinked path.

The reviewer's remaining observation is accepted as stated: the PR head carries **zero commit-status
contexts**, no GitHub Actions run has been observed, and all evidence in this journal is labelled
local. Whether Actions is enabled for the repository should be settled before merge.

## Checks not run, and why

**GitHub Actions was not observed.** The repository is private, `gh` is not installed here, and no
`GH_TOKEN`/`GITHUB_TOKEN` is available; SSH authenticates `git` but not the Actions REST API. The
same limitation is recorded in the 2026-08-17 portable-bundle journal. Nothing in this PR should be
read as evidence that a remote workflow passed. The workflows call the same two scripts run above, so
local and CI behaviour cannot drift, but their remote status is unknown to me.

**No GPU run.** CI and every gate here are CPU-only, unchanged from the existing support matrix.

**No production-length simulation.** Out of scope, and it would not be evidence about a catalog.

## Scientific status, stated as the instruction requires

* Implementation smoke and regression testing is **not** scientific validation.
* The general conventional-MD template is **not** production-scientifically-validated merely because
  it runs. `scientific_status: unvalidated`.
* The general REST2 template remains **scientifically unvalidated**. `scientific_status: unvalidated`.
* The RGD ten-rung result is **system-specific pilot-supported evidence**, not general REST2
  validation, and the descriptor model refuses to let it be marked otherwise.
* The macrocycle eight-rung example remains **unvalidated**.
* CPU smoke profiles are **prohibited** as scientific evidence, and the model enforces it.
* The **2 fs versus 4 fs HMR equivalence gate remains open** and is listed in both descriptors.
* **Arbitrary REST2 ladder suitability remains open** and is listed in the REST2 descriptor.

The REST2 descriptor records that omega exclusion is supported, enabled by default and explicitly
disableable. That field *records* the runtime default; it does not set it. The default lives in
`md_templates.openmm.config.DEFAULTS` and in the shipped REST2 profiles, PR 1 changes neither, and a
test asserts the descriptor and the runtime agree with the runtime as the authority.

## Deviations from the instruction

1. **Four modules under `core/` rather than three** — `paths.py` was added. Reasoning above; the
   instruction explicitly invites a different shape with an explanation.
2. **`CHANGELOG.md` and `README.md` were edited.** Not named in the instruction, but adding a
   repository-root `registry.yaml` and a `templates/` tree without documenting them would leave the
   two files that describe this repository silently wrong. Both edits are additive and describe the
   catalog as metadata that dispatches nothing.

No other deviation.

## Deferred to PR 2 and later

* Packaging the catalog: shipping `registry.yaml` and `templates/` inside the wheel via
  `package-data`, and reading them as package resources rather than from a checkout root.
* **Wheel build provenance.** `TrustedProvenance` exists and is honoured, but nothing writes it. PR 2
  is what records a clean build's full commit SHA into the wheel so an installed copy can resolve an
  identity without `.git`. Until then an installed copy correctly raises `NoProvenanceError`.
* Generic execution dispatch, the general CLI, and routing `prepare`/`md`/`rest2` through descriptors.
* Relocating the OpenMM implementation under `src/md_templates/engines/openmm/`.
* Template-local profiles, which is why `profiles.template_local` is `false` and the model refuses
  `true`.
* Writing template identity into bundles and run manifests.
* The Amber-style input adapter, still deferred from the previous instruction.

## Known limitations

* **The catalog is not packaged.** `load_catalog` takes a repository root. In a wheel-installed
  environment with no checkout there is nothing to load — by design in PR 1, and the reason
  packaging is PR 2's first item.
* **Descriptor content is asserted, not derived.** The tests check that the descriptors agree with
  the runtime where a runtime fact exists to compare against — the omega-exclusion default, the
  bundle schema version, the profile IDs. Prose fields such as capability descriptions and restart
  guarantees are reviewed text, and drift in them would not fail a test.
* **`repository_references` are checked for existence, not for meaning.** A reference can point at a
  file that has stopped being relevant.
* **Symlinks are refused, not resolved.** A repository that legitimately wanted a symlinked template
  directory could not use this catalog. No such case exists here, and the stricter rule is the safer
  default.
* **Dirtiness is whole-tree.** Any uncommitted or untracked path makes the checkout dirty, including
  one unrelated to the template being resolved. That is deliberately conservative — narrowing it to
  "paths this template depends on" would require deciding what a template depends on, which is
  exactly the question the catalog does not yet answer.
* **Nothing here is scientific validation**, and the descriptors say so in the file rather than only
  in this journal.
