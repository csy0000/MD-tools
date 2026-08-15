# The handoff package — current revision, and a gap you must know about

## Current revision: `envfix1`

| | |
|---|---|
| tarball | `reports/explicit_solvent/packages/portable_rest2_handoff_10809c7-envfix1.tar.gz` |
| tarball sha256 | `85f2acc570fb834b3e8782a96759eca74868bd2f7b50087b6ab72894e115e711` |
| wheel sha256 | `50e9a1a51ecbd3680d43985786813727186d534c486a9501ebd4895fe639a183` |
| supersedes | `portable_rest2_handoff_10809c7.tar.gz` |
| validated | 2026-08-15 on `<host>` — see `reports/explicit_solvent/20260815_target_machine_validation_run2/` |

Exactly one substantive file changed against the superseded package: `environment.yml` line 47
asked for the conda package `build`, which conda-forge packages as `python-build`. With
`channels: [conda-forge, nodefaults]` pinned, nothing could supply it and
`conda env create -f environment.yml` — the package's own documented first step — was
**unsatisfiable**. The previous package could not be installed by following its own README.

**The wheel was deliberately NOT rebuilt.** It is byte-identical to the superseded package's. This
is the important property of the re-issue: `50e9a1a5…` is the exact artefact that passed
target-machine validation, so shipping any other wheel would hand out something nothing had
tested. That is precisely how the superseded package went wrong — it was built after the report
that appeared to bless it, and quietly differed in box geometry, ladder status and scale-factor
precision.

The shipped `environment.yml` is byte-identical (`90d18f4e…`) to the file that actually built the
429-package environment in which the whole pipeline was exercised. The fix is verified, not
assumed: conda-forge's `python-build 1.5.0` provides the `build` module and `python -m build
--version` reports `build 1.5.0` from the environment's own `site-packages`.

`SHA256SUMS` was regenerated; the wheel and all five manifests hash unchanged.

### Worth deciding, not inheriting

`build` is a **`dev` extra** in `pyproject.toml` (`Requires-Dist: build; extra == "dev"`). It
exists to *produce* the wheel; no consumer of the wheel needs it. A developer-only tool was
therefore blocking environment creation for every consumer while being required by none of them.
Renaming keeps the packaging toolchain the file's own comment says it wants; deleting the line
would serve consumers equally well. The rename was chosen because it is the minimal change and
matches the fix already on `validation/explicit-portability-20260815` byte-for-byte.

## THE GAP: this branch's source is BEHIND the shipped wheel

**The wheel was built from commit `10809c7`, which does not exist in this repository**, and no
clone on `<host>` contains it. Only the built artefact survives here. Concretely, the wheel
contains code and settings this branch's `src/` does not:

| | this branch's `src/escort_ais/explicit/` | the shipped wheel |
|---|---|---|
| `fingerprint.py` | **absent** | present — the build-defining-projection guard |
| `cyclo_rgdfv` box shape | `cube` | `dodecahedron` |
| `rgd_rest2_10rung` ladder status | `validated` | `pilot_supported` |
| ten scale factors | six-decimal truncations | exact, test-locked to the sqrt rule |
| `common/gpu_lock.py`, `analysis/window_strat_convergence.py` | absent | present |

Roughly twenty modules differ in total.

### What follows from that

* **Do not rebuild the wheel from this branch.** It would regress to the cube box, drop the
  fingerprint guard, and re-assert the stronger `validated` ladder claim — silently undoing
  everything that makes the current package better, while keeping the same version number.
* **The committed tarball is currently the only copy of that code** anywhere on this machine.
  That is why the artefact is tracked here rather than left loose in a scratch directory.
* **This branch is not yet "everything you need for explicit solvent" at source level.** It has
  every explicit-solvent document, script, report and the full validation evidence, and its
  `environment.yml` is correct — but its `src/` is one generation behind the package it ships.

### Closing the gap

The `10809c7` source must be recovered from wherever the package was built, then merged here.
Until then, treat the wheel as the authority on behaviour and this branch's `src/` as history.
Reconstructing the source by unpacking the wheel is possible — every `.py` is in there — but it
would be a reconstruction with no commit history, no tests, and no `pyproject.toml` diff, and it
should not be done silently or without a decision record.
