# cMD review fixes and the OPC default

**2026-08-21.** Seven findings from the review of `20260821_cmd-corrections-and-1ns-validation`.
That journal is **superseded**: its implicit hashes were produced under a restraint that read box
vectors an implicit System should not have, so they do not describe the code that is now on `dev`.
Its explicit results stand, but its explicit default does not.

Baseline `07d2458`, head **`d1ff1c3`**. Eight commits, no branch, no PR, `main` untouched.

## Findings

| # | finding | fix | proving test | evidence |
|---|---|---|---|---|
| 1 | `_add_positional_restraints` used `periodicdistance` in every solvent mode, so an implicit restraint read OpenMM's default box vectors and made its own System report periodic boundaries | expression chosen from the *unmodified* System; post-add assertion that periodicity did not change; convention returned and recorded | `test_restraint_periodicity.py` (6) | `System.usesPeriodicBoundaryConditions()` False→True before the fix; implicit restraint energy now invariant under a 1 nm vs 5 nm box, and equals `k·0.9²` exactly |
| 2 | cMD continuity compared a config fingerprint that bound neither the System nor the topology, so a resume could continue a different Hamiltonian under the same run name | `CMD_CONTINUITY_VERSION = 2` hashing serialized System, topology, bundle identity, forcefield provenance, restraint convention and ordered atom identity | `test_cmd_continuity_and_crash.py` (18) | `continuity_hash` committed per generation; explicit `6997c9e1…`, implicit `de031be0…` |
| 3 | DCD truncation computed a frame as three coordinate blocks, missing the 56-byte unit-cell record, and corrupted every periodic trajectory it touched | new `dcdtail` module walking real Fortran records, validating every marker, updating offsets 8 and 20 together, atomic replace | `test_dcd_tail_recovery.py` (17) | measured 96 assumed vs 152 actual bytes for a 6-atom periodic frame; verified with a second, independently written DCD reader |
| 4 | committed outputs were not checked before append; a trajectory shorter than its own watermark was treated as resumable | `inspect_committed_outputs` / `assert_committed_outputs_intact`: longer is truncated to the watermark, **shorter is refused** as missing history | `test_cmd_continuity_and_crash.py`, `integration_cpu.sh` step 13 | torn tails removed with committed frames byte-identical, both periodic and nonperiodic |
| 5 | invocation accounting could disagree with the data, and two invocations could share a run directory | invocation record, watermarks and continuity hash written inside the *same* atomic commit; `close_reporters` refuses to commit on an unclosable stream; exclusive `flock` per run directory | `test_cmd_continuity_and_crash.py` incl. `test_two_invocations_cannot_share_one_run_directory` | history `[1000, 2000, 2000]` before, contiguous `(1, 0, 1000) (2, 1000, 2000)` after |
| 6 | validation gates deferred, and remote CI unverified | staged cMD gated on CPU from the installed wheel: normal resume, forced State fallback, crash/tail recovery | `integration_cpu.sh` steps 10–13 | all gates below; remote CI **green for `d1ff1c3`** (`fast` #28, `integration-cpu` #28) |
| 7 | TIP3P-FB was the active explicit default although ff19SB was parameterised against OPC | four `-v2` OPC profiles become the defaults; `-v1` kept name-resolvable, `is_default: false`, `superseded_by`, COMPATIBILITY ONLY | `test_opc_default.py` (23) | `default` resolves to `-v2` for all four explicit route/method pairs; goldens changed for exactly those four, implicit untouched |

## A defect the OPC change exposed

`forcefield.water` and `solvation.water_model` are independent settings. A manifest naming a water
force field but no packing model leaves the packing model at its default — and once that default
became 4-site OPC, a config naming `amber19/tip3pfb.xml` was packed with 4-site geometry. OpenMM
then fails inside `addSolvent` with *"No template found for residue 3 (HOH). The residue contains
extra sites"*, which names neither setting involved. Caught by `test_bundle_portability`'s peptide
route in the slow suite, not by the fast suite.

`reconcile_water_model` compares site counts and lets the force field win, because the force field
assigns the parameters and therefore decides which model is simulated. Same-site-count pairs are
left exactly as declared — `tip3pfb` parameters packed from a `tip3p` box is the documented
arrangement, and rewriting it would change the resolved configuration, and the hash, of every
existing v1 run. The substitution is printed and recorded in the bundle provenance.

## What was deliberately not repointed at OPC

Reproduction records, labelled rather than changed: `cyclo_rgdfv.yaml` and `test/rgd/` reproduce
completed RGD production trajectories, `small_macrocycle_smoke.yaml` has recorded hashes, and
`cpu-smoke-v1` pins 3-site water to stay fast on CPU. Historical journals and the run table in
`baseline_setups.md` are left as written.

## Gates

| gate | command | result |
|---|---|---|
| per-finding focused tests | `pytest tests/test_restraint_periodicity.py tests/test_dcd_tail_recovery.py tests/test_opc_default.py tests/test_cmd_continuity_and_crash.py tests/test_cmd_segments.py` | **83 passed**, 96.14 s |
| non-slow suite | `pytest -m "not slow"` | **801 passed, 119 deselected**, 79.45 s |
| complete suite | `pytest` | **920 passed, 0 failed, 0 error, 0 skipped**, 3 warnings, 1434.20 s (23:54) |
| fast CI | `scripts/ci/fast_checks.sh` | **PASSED**, 1 m 36 s — wheel + sdist, packaged-resource inspection, public commands from outside the checkout, all 13 profiles listed with the four v2 defaults |
| CPU integration | `scripts/ci/integration_cpu.sh` | **PASSED**, 13 steps — relocation with the source deleted, offline validation, chunked resume, State fallback, and the three new staged-cMD gates |
| committed artifacts | `git ls-files` scan | clean: **2.1 MiB across 205 files**, no trajectory, checkpoint or State, nothing over 1 MiB |

Two pre-existing blockers in the CI scripts were fixed to reach those results: `fast_checks.sh`
step 6 and both `integration_cpu.sh` documents still used the retired `n_chunks`/`chunk`/
`scale_factors` fields, so each aborted before running anything.

## Final CUDA validation

Regenerated from scratch — bundles, projects and runs — under the corrected code.
`CUDA_DEVICE_ORDER=PCI_BUS_ID` throughout: without it CUDA orders `FASTEST_FIRST` and OpenMM's
device 0 is not `nvidia-smi`'s index 0, so a recorded device identity is approximately right and
actually wrong. All nine GPUs were idle beforehand (0 MiB in use, no compute processes); the two
runs used separate devices.

| | explicit | implicit |
|---|---|---|
| GPU | index 0, RTX A5000, `GPU-7a14ba65-b0a3-66bd-536d-881e08b55da1` | index 1, RTX 3080, `GPU-96ce533d-9d42-acf6-8384-5e27150e9a85` |
| force field | ff19SB / **OPC**, 0.15 M NaCl, dodecahedron, 1.2 nm padding | ff19SB / GBn2 / mbondi3 |
| stages | min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1 | min, **eq**, cMD_1 |
| restraint convention | `minimum-image (periodicdistance)` | **`cartesian (nonperiodic)`** |
| timestep | 4 fs, HMR to 3.024 amu | 2 fs, no HMR |
| ensemble | NPT, 300 K, 1 bar | NVT, 300 K, no barostat |
| production | 2 × 125,000 steps | 2 × 250,000 steps |
| absolute step / time | 250,000 / 1000.0 ps | 500,000 / 1000.0 ps |
| committed generations | 2 | 2 |
| invocations | (1, 0→125,000), (2, 125,000→250,000) | (1, 0→250,000), (2, 250,000→500,000) |
| restart source, segment 2 | checkpoint | checkpoint |
| periodic | true | **false** |
| box volume | 18.53 nm³ | **null** |
| all-atom frames | 10 (2438 atoms) | 10 (22 atoms) |
| selected frames | 100 (22 atoms) | 100 (22 atoms) |
| log header / rows | 1 / 10, strictly increasing | 1 / 10, strictly increasing |
| NaN or Inf | none | none |
| continuity hash | `6997c9e1f73fcb0249bf629d70f1a50a78d574cc30486a6d6a20ee8b9f135db1` | `de031be03a20cbe7929dc55c086eb3b236419f3b18cc1ef1469ed2319cd3563b` |

Output hashes (sha256), recomputed under the final code:

```
explicit  cMD_1_all_atoms.dcd       c9ba7e850910bda696c98f6be7287aa6829cb6621821a41d45014039a0aa9f47
explicit  cMD_1_selected_atoms.dcd  5502340e3ded7624e5c161851d771acf9b0fbfa93101c9026104cc48c0f1d8c3
explicit  cMD_1.log                 bb524b4abdb442680b8f0e12efd34829335bfc96144cee5345cdc5e1bae4f843
implicit  cMD_1_all_atoms.dcd       bd6d854e4da54bcd8a92d83842d77df79fb357769a2d7ca796518306a1bd129f
implicit  cMD_1_selected_atoms.dcd  dbf4cea982b761c5599fe7efa7444ace52177ac9dc0707524eb641da0314f925
implicit  cMD_1.log                 d4863bc48b7b742c9226106998266bb2899cdd317e08a8f6ad2f848e53f3c6b4
```

The implicit hashes are **new** and must not be compared with the previous journal's: Finding 1
changed that Hamiltonian. Driver 580.173.02, OpenMM 8.5.2, ParmEd 4.3.1, Python 3.12.13, CUDA mixed
precision. Runs live outside the repository, under `MD-test/cmd-validation-20260821/`.

The implicit potential energy steps at the `eq` → `cMD_1` boundary (−109.2 → −126.9 kJ/mol) because
the positional restraint is released there. Expected, not drift; the explicit graph shows no such
step because `eq_npt_2` is already unrestrained.

## Commits

```
bf970b8  fix(equilibration): restraints measure the distance their System actually has
8ef61d7  fix(dcd): truncate trajectories by their real records, not by coordinate arithmetic
9856b71  fix(cmd): bind continuity to real artifacts, verify commits, and lock the run directory
b1d0647  feat(profiles)!: OPC is the active explicit-solvent default, versioned as -v2
270d8b8  fix(solvation): make the packing model follow the water force field
e1f94ba  fix(ci): step 6 was still building a spec from retired production fields
9d0d4db  test(ci): gate the staged cMD path, and migrate the integration documents
```

## Remote CI

Both workflows trigger on pushes to `dev` and both are **green for the final SHA `d1ff1c3`**.

| workflow | run | conclusion | duration | URL |
|---|---|---|---|---|
| `fast` | #28 | **success** | 2 m 04 s | https://github.com/csy0000/MD-templates/actions/runs/32476211126 |
| `integration-cpu` | #28 | **success** | 3 m 57 s | https://github.com/csy0000/MD-templates/actions/runs/32476210988 |

Every step succeeded in both jobs; the only skipped steps are the `if: failure()` diagnostic
uploads. `integration-cpu` ran the full 13-step gate, the real subprocess crash-recovery slow tests
and the bundle-portability slow tests.

**The baseline was red.** `fast` and `integration-cpu` both **failed** at `07d2458` (#27) and at
`7684854` (#26), and the last green pair before this work was #25 at `5471471` on `main`. The cause
was the one recorded above: both CI scripts still constructed their documents from the retired
`n_chunks` / `chunk` / `scale_factors` production fields, so each aborted before running anything.
That is why the previous round reported these gates as unverified rather than failing — the failure
was in the harness, not in the code under test.

Reading these results required **Actions: Read** on the fine-grained PAT, which was absent while
this work was done and added afterwards. Until then every Actions route returned HTTP 403
"Resource not accessible by personal access token", with GitHub naming the missing permission in
the `x-accepted-github-permissions: actions=read` response header.
