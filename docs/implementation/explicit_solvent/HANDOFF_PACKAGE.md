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

## Source provenance — RESOLVED

The wheel was built from commit `10809c7` of the originating repository, recovered from
`pharma-jay` on 2026-08-16. All nine modules of `escort_ais/explicit/` are **byte-identical**
between that commit and the shipped wheel, so this tree is the wheel's source, not a generation
behind it. `explicit/fingerprint.py`, the dodecahedron box, `ladder_status: pilot_supported` and the
exact ten scale factors are all present here.

`reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/rgd_simbox.json` — which
`cyclo_rgdfv.yaml` cites as the durable copy of the box geometry — was recovered in the same commit
and hashes to `ea1c14edb7c9fc949d892fb41fb733750179555c235b3d6a0a54c7bda821ac61`, exactly the value
the manifest declares. An earlier revision of this document reported both as missing; they were
never lost, they had simply not reached the machine the package was validated on.

Rebuilding the wheel from this tree is therefore safe, and would additionally pick up the
`environment.yml` and `pdb`-route bundle fixes that the shipped wheel predates.
