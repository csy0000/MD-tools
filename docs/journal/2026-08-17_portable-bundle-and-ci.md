# The transferable bundle contract, canonical run integration, and CI

**2026-08-17.** Status: **every requirement of the instruction is implemented and locally
verified.** Both canonical routes were prepared, relocated, validated, run and resumed from a clean
wheel-installed environment outside the checkout, with the originating directories deleted. CI
workflows are **configured and locally reproduced but NOT yet observed running on GitHub** — that
distinction is kept throughout and no CI result is claimed.

Request: `claudecode-instructions/20260817_portable-bundle-and-ci.md`, continuing the previous
instruction rather than redoing it. The accepted-as-complete list was preserved: committed-generation
recovery, tail quarantine, State fallback, append-only logs, canonical model, `is_default`
selection, and the scientific defaults were not retuned.

## Completed requirements

### Explicit master and stage seeds

A canonical `randomness` section with its own schema version: `master_seed` plus optional per-stage
overrides. The legacy derivation is preserved exactly — `master + offset`, structure 0,
equilibration 1, md 2, rest2 3 — and a test drives the shipped RGD manifests down both the legacy
and canonical paths to prove they agree (20260814/15/16/17 either way).

**What is hashed is the resolved stage seed, not the label.** Structure and equilibration seeds
enter the prepared-state projection because they shape the stored artifacts; the production seed
enters run continuity because the same state advanced under a different seed is a different
trajectory; `master_seed` is classified `seed-label` in `config diff`. The consequence the
instruction asks for therefore falls out rather than being special-cased: a changed master seed
normally moves both physical hashes, but pinning every affected stage seed to its old value leaves
them unchanged. Verified in both directions.

### Projection audit

`prepared_state_sha256` now covers the equilibration block, the integrator used to reach the stored
state, and the preparation seeds, alongside system and build. Classification follows scientific
consequence, not model section: treating equilibration as "protocol" would have left a bundle
reusable across a change that produced a different starting state.

### Canonical configuration for both routes and both run commands

`prepare --config` supports `smiles` and `pdb`. The PDB path resolves relative to the
**configuration document**, never the working directory, its declared hash is verified, and the
structure is copied into the bundle so preparation does not depend on the input surviving.

`md` and `rest2` accept `--config` and `--set`; with neither, a version-2 bundle's pinned
configuration is used. Before a run directory exists, the method discriminator, the system/build
hash and the prepared-state hash must all match. Extension-only changes such as `n_chunks` pass.

### Bundle schema version 2

Independent of every other schema version. Adds `checksums.json` (bytes, normalised relative POSIX
paths, explicit domain excluding itself and the manifest with the reason recorded),
`original_inputs/`, `forcefield_provenance.json`, `environment.json` (classified by what each
version is needed FOR), `topology.cif`, and counts kept apart:
`topology_atoms`, `openmm_particles`, `virtual_sites`, `massless_particles`, `constraints`, with
the DOF formula recorded beside the number. `n_atoms` is gone.

Version-1 bundles remain readable and are reported as `v1-compatibility` with their missing
guarantees named — never relabelled.

### Bundle CLI

`bundle validate [--deep]`, `bundle inspect [--format]`, `bundle relocate-check`. Default validation
constructs no Context: a portability check that needs a working simulation environment cannot run
where it is most needed. `relocate-check` copies the bundle elsewhere and validates it **there**,
because validating in place cannot detect dependence on the original location.

## Three defects found by running, not reading

**Method mismatch was swallowed.** The first canonical run path wrapped resolution in a
try/except that fell back to the legacy resolver on any incompatibility, so `rest2` ran happily
against an md-configured bundle. The fallback now triggers only on the genuine ABSENCE of a
canonical configuration.

**Auto-migration guessed a method.** Attaching a canonical record to legacy bundles made
`migrate_manifests` choose between an `md:` and a `rest2:` block by `if/elif` order. The shipped
smoke experiment declares both, so a legacy REST2 bundle became "md" and refused to run as REST2 —
caught when the crash-recovery subprocess died in 60 s under the wheel. Ambiguity is now refused,
and an explicit legacy `--experiment` outranks a record the tool synthesised.

**A destructive test destroyed its neighbours.** The source-removal test deleted the directory
holding a module-scoped fixture's bundle, so every later test read that bundle as version 1. It now
prepares its own.

## Local evidence — commands and results

All run in a **clean cloned environment with the wheel installed and no checkout on the path**.

```console
$ python -m pytest tests/ -q -m "not slow"                  291 passed, 17 deselected
$ python -m pytest tests/test_crash_recovery.py -q -m slow    1 passed (107.70 s)
$ python -m pytest tests/test_bundle_portability.py -q -m slow  14 passed (163.91 s)
$ bash scripts/ci/fast_checks.sh                            fast checks: PASSED
$ bash scripts/ci/integration_cpu.sh                        CPU integration: PASSED
```

The integration gate is the completion requirement, and it does all of this in one run: prepares a
tiny canonical SMILES bundle and a tiny canonical PDB bundle, copies both to an unrelated
directory, **deletes the directories they were built in**, validates `--deep`, inspects,
relocate-checks, runs REST2 fresh and resumed, runs MD fresh and resumed, asserts contiguous
attempt indices / one CSV header / contiguous chunks / two recorded invocations, corrupts a
committed checkpoint and requires the State fallback to be **announced**, then validates and
inspects with the network disabled.

Wheel contents were inspected: 5 profiles, 4 system manifests, 3 experiment manifests, the peptide
structure, and the spec and bundle modules are all present.

## CI evidence

**None yet.** Two workflows are configured — `fast` and `integration-cpu`, both CPU-only, both
calling the same scripts run above so local and CI behaviour cannot drift. They parse as valid
YAML, every job pins `runs-on` and an explicit timeout, and `environment-ci.yml` is the documented
environment with the CUDA pin removed.

**Remote observation was attempted and is not possible from this environment.** After pushing
`015c88f` to `openmm` (push verified: local and remote HEAD agree, and both workflow files are
present on the remote branch), the runs were queried three ways:

```console
$ gh run list                       gh CLI is not installed
$ curl .../actions/runs             HTTP 404
$ curl .../repos/csy0000/MD-templates   HTTP 404 -- the repository is private
$ echo $GH_TOKEN $GITHUB_TOKEN      both unset
```

Pushing works because the remote is SSH (`git@github.com:...`); the Actions REST API needs a token,
and none is available here. So the workflows' status on GitHub is **unknown to me**, and nothing in
this repository should be read as evidence that they passed.

**The environment risk was measured rather than left as a worry.** A cold dry-run solve of
`environment-ci.yml` on this machine: **exit 0, 419 packages, 711 s (11 min)**, resolving Python
3.11.15, OpenMM 8.5.1, pydantic 2.11.10, RDKit 2025.03.6 and AmberTools 24.8 -- the same versions
the gates above were run against, and without the CUDA pin. Against the 45-minute `fast` and
90-minute `integration-cpu` timeouts that leaves ample margin, and `cache-environment: true` makes
it a first-run cost. The 11 minutes is the solve alone; a cold runner also downloads roughly 4 GB,
so the first run costs more than later ones. No trimming of the CI environment is warranted on this
evidence.

What someone with access should check, in order: that Actions is enabled for the repository at all;
that `fast` and `integration-cpu` were triggered by the push to `openmm`; and whether the two third-party actions
(`actions/checkout@v4`, `mamba-org/setup-micromamba@v2`) resolve on the runner. Those are the only
failure modes the local evidence cannot cover. If a job fails for environment rather than code
reasons, `environment-ci.yml` is the file to adjust -- though the solve measurement above says it
should not need to be.

Until then the correct description is **configured and locally reproduced, awaiting remote
observation.**

## Portability claims, stated precisely

* transferred prepared artifacts are **byte-identical when the checksums match**;
* binary checkpoints are **environment-specific** and are never described as portable;
* a serialized State is a **physically valid portable fallback**, not a bitwise continuation of
  stochastic dynamics, and its use is announced and recorded;
* rebuilding from original inputs **may be scientifically consistent without being bitwise
  identical** unless the recorded environment is reproduced;
* **cross-machine bitwise reproducibility is not claimed** in any configuration.

## Limitations

* The support matrix lists one Python (3.11) and one OpenMM (8.5.1). No second version is listed
  because none has been run.
* `bundle validate --deep` cross-checks counts and box but does not re-derive parameters; a
  force-field resource that could not be located on disk has a null hash, and that is visible
  rather than assumed away.
* The crash test kills one process at one moment. Real evidence, not exhaustive coverage.
* Nothing here is scientific validation: no ladder is validated, the 2 fs / 4 fs gate is open, and
  every run is picoseconds.

## Deferred: the Amber-style adapter

Not implemented, by instruction. The boundary stands: register a loader for the suffix, parse the
foreign syntax, return data shaped like the canonical model, and **reject every key that cannot be
mapped**. It must add no defaults, reinterpret no values, and grow no validation of its own — an
adapter into these semantics, never a parallel set of them. `register_loader` exists for it.

## Historical instruction audit

Per instruction, the tracked instruction files under `claudecode-instructions/` were not hidden
with `.gitignore` and not edited to change a match count. The active audit is scoped to package
code, current documentation, examples, manifests, tests, packaging metadata and workflows, where it
returns **0 matches**. The instruction directory is the documented exception: those files spell the
retired identity because their content is the directive to remove it.
