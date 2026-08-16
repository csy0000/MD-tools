# From a validation run to a standalone template repository

**2026-08-15 → 2026-08-16.** Status: **this repository exists, both force-field routes are
exercised end to end, and the source it ships is provably the source its wheel was built from.**
Three defects were found along the way, each by running something rather than reading it.

Starting state: the explicit-solvent pipeline lived on a branch of the `escort-ais` research
repository, its package had been handed off as `portable_rest2_handoff_10809c7.tar.gz`, and the
only record of its validation was a prose report. Ending state: `csy0000/MD-templates`, branch
`openmm`, carrying the implementation, both worked examples, and the evidence for every claim.

## 1. Target-machine validation of the handoff package

Ran `claudecode-instructions/20260815-explicit-test-2.md` against the shipped package on
`<host>` (Ubuntu 24.04.3, driver 580.173.02, GPU 0 = RTX A5000). Verdicts:

```text
handoff integrity: verified          CUDA portability: verified
clean environment: FAILED            RGD preparation: verified
wheel installation: verified         ten-rung RGD mechanical smoke: verified
CPU portability: verified            long-production readiness: NO
```

The single failure was the shipped `environment.yml`: line 47 requests the conda package `build`,
which conda-forge packages as `python-build`, and `channels: [conda-forge, nodefaults]` is pinned so
nothing else can supply it. `conda env create` — the package's own first documented step — exits 1
after 736 s with `build =* * does not exist`. **The package could not be installed by following its
own README.**

Root cause is in the wheel's own metadata: `Requires-Dist: build; extra == "dev"`. It is a
PyPI→conda name mismatch in a **developer-only** extra — `build` exists to *produce* the wheel and
no consumer needs it, so a tool nobody installing the package requires was blocking everybody
installing the package.

Everything else passed: CPU smoke 75 s, CUDA smoke 28 s, RGD preparation 787 s, ten-rung mechanical
smoke 6.6 s with all nine neighbouring pairs attempted and relaxation correctly excluded from
exchange accounting.

**The from-scratch RGD preparation reproduced the historical Phase-A system exactly** — 1047 waters,
3226 particles, box 3.66182 nm, minimum-image 2.5893 nm, Na⁺ 3 / Cl⁻ 3 — where §7 of the
instruction promises only "scientifically consistent, not bitwise identical". One derived number
differs, `realized_ionic_strength_molar` 0.143483 against the cited 0.15903, from identical ion and
water counts: the convention changed from ions-per-55.5-mol-water to ions-per-box-volume. The
current number is the better-defined one; the manifest's documented expectation is stale.

Two instruction/CLI mismatches were recorded rather than worked around: `validate-system` never
prints `charge_method` although §4 asks it to be confirmed, and `validate-env --precision` cannot
verify itself — it calls `Platform.getNumPropertyNames`, which does not exist in OpenMM 8.5.1 (the
API is `getPropertyNames`). The setting is applied correctly; only its check is broken.

## 2. The evidence problem, and why this repository exists

The previous validation lived on a branch as **prose**: hashes and acceptance tables typed into
markdown, with every artefact — package, environment export, run directories, `status.json`,
exchange CSVs — in an untracked scratch folder on one machine. Nothing on the branch could be
re-checked against the run that produced it.

Everything here now travels with its artefacts: console logs, environment exports, bundle and run
manifests, resolved configs and exchange CSVs. Trajectories are deliberately excluded — 2 ps smoke
output is regenerable and says nothing.

The branch was then cut down to what explicit-solvent actually needs. The first cut used an import
trace and was **wrong**: `explicit_baseline.run_rest2_remd` imports `methods.md_run` and
`systems.topology_prep` *inside the function*, so no trace taken before a run can see them. The
pristine checkout imported cleanly, prepared a bundle cleanly, then died at the first exchange with
`ModuleNotFoundError: No module named 'escort_ais.methods'`. The closure is 23 modules, established
by running the smoke. **An import trace cannot see a deferred import; only executing the path can.**

## 3. ff19SB: implemented, never run, and broken

The package always advertised two routes. Only one had ever been executed:

| | ligand (`smiles`) | peptide (`pdb`) |
|---|---|---|
| shipped systems | 2 | **0** |
| tests | behaviour + guards | **guards only** |
| ever run end to end | yes | **no** |

All three peptide tests assert *refusals* — that a peptide cannot silently become a Sage run, that a
PDB hash mismatch is rejected. They prove ff19SB is unreachable by accident; none proves it is
reachable on purpose.

Running it exposed a real defect. `ace_ala_nme` (ACE-ALA-NME, built by `tleap` from
`leaprc.protein.ff19SB`) parameterised, solvated, minimised and equilibrated — the physics was never
the problem — and then produced a bundle that could not be read back:

```text
manifest error: …/system.yaml: input.pdb does not exist:
                …__ace_ala_nme__bundle__dad2e5f8f0ff/ace_ala_nme.pdb     (exit 3)
```

The `smiles` route carries its input inline; the `pdb` route **points at a sibling file**, and
`bundle.py` copied `system.yaml` without it. Since `validate-bundle` re-reads that manifest and
`rest2 --bundle` does so before running, **the peptide route could build a system and never use
one.** The shipped wheel `10809c7` has the identical defect, so ff19SB was unusable in every
version.

Fixed by copying the structure into the bundle under its declared name, which keeps both the
relative path and `input.pdb_sha256` resolving. `ace_ala_nme` now ships as a first-class system, and
`manifests/systems/*.pdb` was added to package-data — without it the manifest travels into the wheel
and its structure does not, reproducing the bug through another door.

After: `smoke` rc=0, `completed`, 4/4 rounds; `validate-bundle` passes; manifest records
`amber19/protein.ff19SB.xml` with `small_molecule_forcefield` and `charge_method` null; 883 atoms,
22 solute, 287 waters.

**This is the argument for validating before advertising.** "Implemented" and "works" were different
states, and only executing the path distinguished them.

## 4. Recovering the wheel's source

The published tree carried two declared gaps. Both turned out to be recoverable, and one was never
a gap at all.

The wheel was built from commit `10809c7`, which existed in **no clone** on this machine. It was
found on `pharma-jay` in `projects/escort-ais-alanine` on `flatbottom-basin-cft-mixture` and fetched
read-only. **All nine modules of `escort_ais/explicit/` are byte-identical between that commit and
the shipped wheel**, compared file by file rather than inferred from a version string. That brought
in `fingerprint.py`, the dodecahedron box, `ladder_status: pilot_supported`, the exact ten scale
factors and 492 lines of portable tests.

The same commit contains
`reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/rgd_simbox.json` — the file
`cyclo_rgdfv.yaml` cites as the durable copy of the box geometry, which had been reported as tracked
nowhere. It hashes to `ea1c14edb7c9…`, exactly the value the manifest declares. It was never lost;
it had simply never reached the target machine. Worth noting the order of events: the from-scratch
preparation in §1 reproduced that file's geometry *before* anyone could read the file.

## 5. Package re-issue, and one more test defect

`portable_rest2_handoff_10809c7-envfix1.tar.gz` fixes the one token and **nothing else**. The wheel
is byte-identical to the superseded package's, deliberately not rebuilt: `50e9a1a5…` is the artefact
that passed validation, and shipping a freshly built one would distribute code nothing had tested —
which is exactly how the superseded package went wrong, built *after* the report that appeared to
bless it and quietly differing in box shape, ladder status and scale-factor precision. The shipped
`environment.yml` is byte-identical to the file that actually built the validated 429-package
environment.

Running the suite here surfaced a last defect, in a test rather than the package.
`test_shipped_manifests_are_not_hidden_from_git` asserted `"/data/" not in str(d)` against the
**absolute** path of the manifests directory. The property it means to check is that no *package*
directory is named `data`, which `.gitignore` erases at any depth. This checkout lives at
`/path/to/MD-templates`, so the absolute path contains `/data/` and the test failed with
nothing wrong — as it would for any clone under a directory named `data`, an ordinary layout on
shared clusters. Now checks the package-relative path. Suite: **160 passed**.

## 6. Where things ended up

| | |
|---|---|
| this repository | `csy0000/MD-templates`, branch `openmm` |
| AIS repository | one `main`, implicit-only — **zero** explicit-solvent files |
| archived branches | `archive/premerge-2026-08-16/*` tags on the AIS remote |
| scratch test folder | deleted, after its unique artefacts were rescued here |

The superseded package and the earlier `d9ff9d02…` wheel are kept under
`reports/explicit_solvent/packages/` because both validation reports make claims about specific
bytes; keeping the bytes is what makes those claims checkable rather than asserted. Neither is a
deliverable. The only thing lost with the scratch folder was 46 MB of 2 ps smoke trajectories.

## 7. Outstanding

* **No AIS-side migration note exists.** `git grep "MD-templates"` on AIS `main` returns nothing, so
  that repository records nowhere that its explicit-solvent code moved here. Phase 6 of
  `202608-repo-merge.md` requires target repository, branch/commit, original SHAs and manifest
  location. This is the largest open gap.
* **`reproducibility/integration_authority.yaml` and `report_inventory.{tsv,json}` were never
  written** (Phases 2 and 5 of the same runbook). The closure audit was meant to run *before* any
  checkout was deleted, and deletions have since happened.
* **The distributed wheel still contains both fixes' absence** — the `environment.yml` blocker and
  the `pdb`-route bundle defect. Rebuilding from this tree is now safe and would close both; that
  has not been done, and the shipped wheel remains `50e9a1a5…`.
* **`macrocycle_pilot_8rung` has no worked example**, though it is the manifest handed to anyone
  starting a new macrocycle.
* **The package is still named `escort_ais`** with an `escort-explicit` console script — a research
  project's name inside a general-purpose template. Cheap to change now, expensive once consumed.
* Four journals here still narrate branches that no longer exist. Historical logs under `reports/`
  legitimately name them; journals read as current instructions and do not.

## What none of this establishes

Mechanical portability only. `rgd_rest2_10rung` is `pilot_supported` for one molecule; no ladder is
validated for any peptide; the 2 fs / 4 fs hydrogen-mass-repartitioning gate is open; restart safety
and convergence are unestablished. Every acceptance figure quoted above comes from picoseconds and
must not be read as a statistic.
