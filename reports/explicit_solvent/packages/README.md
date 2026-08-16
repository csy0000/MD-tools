# Distributed packages, current and superseded

Three artefacts, kept because the two validation reports in this directory's parent make claims
about specific bytes. Without the bytes, those claims cannot be checked.

| file | wheel inside | status |
|---|---|---|
| `portable_rest2_handoff_10809c7-envfix1.tar.gz` | `50e9a1a5…` | **current** — install from this one |
| `superseded__portable_rest2_handoff_10809c7.tar.gz` | `50e9a1a5…` | superseded — cannot be installed |
| `superseded__escort_ais-0.1.0-d9ff9d02.whl` | — | superseded — a different, earlier build |

Checksums: `portable_rest2_handoff_10809c7-envfix1.SHA256SUMS` for the current package's contents,
`…tar.gz.sha256` for the tarball itself, and `SUPERSEDED.sha256` for the two superseded artefacts.

## Why the superseded package cannot be installed

Its `environment.yml` line 47 requests the conda package `build`; conda-forge packages that
distribution as `python-build`. The file pins `channels: [conda-forge, nodefaults]`, so nothing can
supply it and `conda env create -f environment.yml` — the package's own documented first step —
fails:

```text
error libmamba Could not solve for environment specs
    └─ build =* * does not exist (perhaps a typo or a missing channel).
```

`envfix1` changes that one token and nothing else of substance. **Its wheel is byte-identical**,
which is the point: `50e9a1a5…` is the artefact that passed target-machine validation, so shipping
a freshly built one would distribute code nothing had tested.

## Why the older wheel is kept

`d9ff9d02…` is the wheel the **first** target-machine validation tested
(`../20260815_target_machine_validation.md`). It is not a slightly different build of the same
thing — it differs in ways that change results:

| | `d9ff9d02…` | `50e9a1a5…` |
|---|---|---|
| `cyclo_rgdfv` box | cube | dodecahedron — the geometry the ladder pilots ran in |
| RGD bundle config hash | `a3173143d762` | `abf1d9c2f819` |
| prepared composition | 1554 atoms, 491 waters | 3226 atoms, 1047 waters |
| `rgd_rest2_10rung` | `ladder_status: validated` | `pilot_supported` |
| ten scale factors | six-decimal truncations | exact, test-locked to the sqrt rule |
| `fingerprint.py` | absent | present |

That first report is accurate about what it tested and is not retracted; it simply describes a
different artefact from the one now distributed. Keeping the wheel is what makes the difference
verifiable rather than merely asserted.

Neither superseded artefact should be handed to anyone. They are evidence, not deliverables.
