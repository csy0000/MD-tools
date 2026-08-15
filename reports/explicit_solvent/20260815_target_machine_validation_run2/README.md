# Target-machine validation — portable explicit-solvent REST2, handoff `10809c7`

Executed against `claudecode-instructions/20260815-explicit-test-2.md` on 2026-08-15.

This validates the **handoff package `portable_rest2_handoff_10809c7.tar.gz`**, whose wheel hashes
`50e9a1a5…`. It is **not** the package validated by the earlier report
`reports/explicit_solvent/20260815_target_machine_validation.md`, which tested wheel `d9ff9d02…`
built on this machine that morning. The two differ in ways that change results — see
[§10](#10-differences-from-the-earlier-validated-wheel).

## Verdicts

```text
handoff integrity:                verified
clean environment:                failed
wheel installation:               verified
CPU portability:                  verified
CUDA portability:                 verified
RGD preparation:                  verified
ten-rung RGD mechanical smoke:    verified
long-production readiness:        NO
```

`clean environment: failed` is the only failure and it is reproducible, specific and one token
wide: the shipped `environment.yml` names a conda package that does not exist. Everything
downstream was executed in a **fresh** environment built from a corrected copy, never from a
pre-existing project environment. The deviation is recorded in [§3](#3-clean-environment).

## 1. Target machine

| | |
|---|---|
| host | `<host>` |
| OS | Ubuntu 24.04.3 LTS |
| kernel | `Linux 6.8.0-124-generic #124-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 13:00:45 UTC 2026 x86_64` |
| driver / CUDA | 580.173.02 / CUDA 13.0 |
| GPUs | 9 — index 0 `NVIDIA RTX A5000` (`GPU-7a14ba65-b0a3-66bd-536d-881e08b55da1`), indices 1–8 `NVIDIA GeForce RTX 3080` |
| device used | **0**, the A5000 — confirmed by OpenMM's own report, see [§6](#6-cuda-validation) |
| solver | `mamba 2.5.0` — `micromamba` is **not installed** on this host, so the instruction's stated fallback applies |
| CPU | 48 cores; load average ~5 throughout, from unrelated jobs |

**Unrelated work was running and was not disturbed.** GPUs 6, 7 and 8 carried three foreign compute
processes (PIDs 2881804, 2691238, 2677584, all `envs/escort-ais/bin/python` — the cBAR and
window-strat repeats). Every CUDA step below was preceded by an `nvidia-smi` compute-process
preflight; device 0 was confirmed to carry **zero** compute processes each time, and no step ever
targeted 6–8.

Machine record: `console_logs/01_machine.txt`.

## 2. Handoff integrity — **verified**

All nine required files present (`console_logs/02_handoff_files.txt`), and every checksum
verifies (`console_logs/03_checksums.txt`):

```console
$ cd /path/to/explicit-test-20260815/portable_rest2_handoff && sha256sum -c SHA256SUMS
escort_ais-0.1.0-py3-none-any.whl: OK
environment.yml: OK
cyclo_rgdfv.yaml: OK
macrocycle_pilot_8rung.yaml: OK
rgd_rest2_10rung.yaml: OK
small_macrocycle_smoke.yaml: OK
smoke.yaml: OK
HANDOFF_README.md: OK
                                                                       # exit 0
```

The wheel independently hashes to the expected value:

| file | sha256 |
|---|---|
| `escort_ais-0.1.0-py3-none-any.whl` | `50e9a1a51ecbd3680d43985786813727186d534c486a9501ebd4895fe639a183` ✓ **matches the expected value in §2 of the instruction** |
| `environment.yml` | `593bac40be510105e1986f760289cdc1e8ef1e7228fee62f14b7e24f35b5c2c4` |
| `cyclo_rgdfv.yaml` | `ab6bd85ebf2cb1907c44b0a3449061b6a3fbe0107e01a4d313d10bcde195e789` |
| `rgd_rest2_10rung.yaml` | `a41a73be487d3f59b95b4d88f08016ab8488e2d8bc62b73668110ea6eb6201e7` |
| `macrocycle_pilot_8rung.yaml` | `c9097bfb7fe94bcbb1800d5526c2882b444b034846be4c1ed7e6f177a0928be5` |
| `small_macrocycle_smoke.yaml` | `91f8b59a86380eac67cf8e0a3bbf7a0cd076dc0bb49281c32d6999fdd925a04a` |
| `smoke.yaml` | `06dcdd5ee0e713625075668c9c5e3d80348785bb173e39773bcbec6aa7cfaf92` |
| `HANDOFF_README.md` | `6d0a749aa111795488a9b255eced6845701bfeb595c772b4e7f01f535be3db28` |

The five loose manifests are byte-identical to the copies inside the wheel, so they are genuinely
readable mirrors and not a second, drifting source of truth.

## 3. Clean environment — **FAILED**, then deviated

### 3.1 As shipped — fails

```console
$ mamba env create -y -n escort-ais-explicit-target -f "$HANDOFF/environment.yml"
error    libmamba Could not solve for environment specs
    The following package could not be installed
    └─ build =* * does not exist (perhaps a typo or a missing channel).
critical libmamba Could not solve for environment specs
EXITCODE=1        19:01:35Z -> 19:13:51Z  (736 s)
```

No environment was created. Log: `console_logs/04_env_create_asshipped.log`.

`environment.yml:47` requests `build`; on conda-forge that distribution is packaged as
`python-build`. The file pins `channels: [conda-forge, nodefaults]`, so no other channel can supply
it and the solve is unsatisfiable.

**Root cause, from the wheel's own metadata.** `escort_ais-0.1.0.dist-info/METADATA` declares:

```text
Requires-Dist: pytest; extra == "dev"
Requires-Dist: build;  extra == "dev"
```

`environment.yml` was transcribed from `pyproject.toml`, and the PyPI name `build` was carried over
unchanged. Two things follow. First, it is a PyPI→conda naming mismatch, not a missing dependency.
Second — and this is what makes it worth fixing rather than working around — **`build` is in the
`dev` extra**. It exists to run `python -m build`, which is how the wheel is *produced*, not how it
is *consumed*. A developer-only tool therefore blocks environment creation for every consumer of
the package while being needed by none of them. Renaming it to `python-build` or deleting the line
from the consumer-facing file would both work.

**This blocker was already fixed once and did not reach the package.** Commit `31f8134` on
`validation/explicit-portability-20260815` corrected
`docs/implementation/explicit_solvent/environment.yml` on 2026-08-15. The handoff tarball was built
from `10809c7`, which does not carry that fix, and `github/flatbottom-basin-cft-mixture` still has
`- build` at line 47 today. The fix exists on exactly one branch and is merged nowhere.

A note for whoever hits this next: the previous session's identical solve failed in **seconds**;
this one took **12 minutes and ~4 GB of RSS** to emit the same message, on the same host with the
same solver. Same outcome, very different experience — do not assume a long silence means progress.

### 3.2 Deviation, and its limits

The remainder of the validation ran in a **new** environment created from a corrected copy of the
shipped file, differing by exactly one token:

```diff
47c47
<   - build                     # `python -m build` -> the wheel a consuming repository installs
---
>   - python-build              # `python -m build` -> the wheel a consuming repository installs
```

`environment/environment.target-corrected.yml`, sha256
`90d18f4e81984c23ba76a3de53067ea1e7eba489c398a9f6e9d5cf000ee9b4f6`.

This is the *same* deviation the previous session made, and it is bounded: §3 of the instruction
forbids falling back to a **pre-existing project environment**, and no pre-existing environment was
used. `escort-ais-explicit-target` was created from scratch for this run. The environment that was
tested is therefore not the environment as shipped, and no claim here should be read as validating
the shipped `environment.yml`.

```console
$ mamba env create -y -n escort-ais-explicit-target -f environment/environment.target-corrected.yml
EXITCODE=0        19:11:27Z -> 19:23:56Z  (749 s)     # 9.5 GB, 429 packages
```

Log: `console_logs/05_env_create_corrected.log`.

Every primary version pinned in the file resolved exactly as declared:

| package | pinned | resolved |
|---|---|---|
| python | 3.11 | 3.11.15 |
| openmm | 8.5.1 | 8.5.1 |
| openmmtools | 0.26.0 | 0.26.0 |
| openmmforcefields | 0.16.0 | 0.16.0 |
| openff-toolkit | 0.17.1 | 0.17.1 |
| openff-interchange | 0.4.5 | 0.4.5 |
| ambertools | 24.8 | 24.8 |
| parmed | 4.3.1 | 4.3.1 |
| rdkit | 2025.03.6 | 2025.03.6 |
| pymbar | 4.2.0 | 4.2.0 |
| mdtraj | 1.11.1 | 1.11.1 |
| numpy / scipy | 2.4.6 / 1.17.1 | 2.4.6 / 1.17.1 |
| cuda-version | 13.0 | 13.0 |

Exports required by §3: `environment/conda-list-explicit.txt` (425 lines),
`environment/conda-env-from-history.yml`, `environment/conda-list-full.txt` (429 packages).

**The `--no-deps` contract holds.** Every runtime dependency the wheel declares (`numpy`, `scipy`,
`pandas`, `matplotlib`, `tqdm`, `pyyaml`, `scikit-learn`, `numba`) is present in `environment.yml`.
Nothing is silently missing; OpenMM, RDKit and the OpenFF stack are deliberately absent from
`Requires-Dist` because conda supplies them.

## 4. Wheel installation — **verified**

```console
$ mamba run -n escort-ais-explicit-target python -m pip install \
    "$HANDOFF/escort_ais-0.1.0-py3-none-any.whl" --no-deps
Successfully installed escort-ais-0.1.0                                # exit 0

$ mamba run -n escort-ais-explicit-target python -c "import escort_ais; print(escort_ais.__file__)"
/path/to/miniforge3/envs/escort-ais-explicit-target/lib/python3.11/site-packages/escort_ais/__init__.py
```

| requirement | result |
|---|---|
| resolves from the environment, not a checkout | ✓ path is inside the target env's `site-packages` |
| no editable install | ✓ `pip show` reports `Location: …/escort-ais-explicit-target/lib/python3.11/site-packages` |
| source repository not on `PYTHONPATH` | ✓ `PYTHONPATH` is empty |
| python | 3.11.15 |
| `sqm` | `/path/to/miniforge3/envs/escort-ais-explicit-target/bin/sqm` — inside the env |

Log: `console_logs/06_wheel_install.log`.

**The package proves its own identity.** `validate-env` reports the sha256 of the wheel it was
installed from — `50e9a1a5…`, matching §2. That is an unusually good audit property: the running
code states which artefact it came from, rather than leaving it to be inferred.

## 5. CPU validation and smoke — **verified**

```console
$ escort-explicit --version                       # escort-ais 0.1.0                exit 0
$ escort-explicit --help                          #                                 exit 0
$ escort-explicit validate-env --platform CPU --route smiles                      # exit 0
$ escort-explicit validate-system --system "$HANDOFF/cyclo_rgdfv.yaml" \
                                  --experiment "$HANDOFF/rgd_rest2_10rung.yaml"   # exit 0
```

Every field §4 requires was confirmed (`console_logs/09_validate_system_rgd.txt`):

| required | reported |
|---|---|
| `system_id: cyclo_rgdfv` | ✓ |
| `route: smiles` | ✓ |
| `formal charge: 0` | ✓ |
| small-molecule ff `openff-2.2.0` | ✓ |
| protein ff `null` | ✓ `protein=None` |
| box shape `dodecahedron` | ✓ |
| rungs `10` | ✓ |
| ladder status `pilot_supported` | ✓ — and the word `validated` appears nowhere |
| charge method `am1bcc` | **not printed by `validate-system`** — see [§9](#9-gaps-between-the-instruction-and-the-cli) |

Canonical identity reported: molecule hash `59d4422635f77292ed94c609a76808df5c97110be09cd9779819270363143159`,
SMILES `CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)[C@H](CCCNC(N)=[NH2+])NC1=O`.

### CPU smoke

```console
$ escort-explicit smoke --system "$HANDOFF/small_macrocycle_smoke.yaml" \
    --experiment "$HANDOFF/smoke.yaml" --out-root "$TARGET_ROOT/cpu_smoke" --platform CPU
[smoke] status: completed  exchange rounds 4/4                          # exit 0, 75 s
```

| §5 acceptance | result |
|---|---|
| exit code zero | ✓ |
| `status.json` reports `completed` | ✓ |
| at least two exchange rounds | ✓ **4** |
| every configured replica propagates | ✓ 3 replica directories, s = 1.0 / 0.5625 / 0.25 |
| every expected neighbouring pair attempted | ✓ (0,1) ×2 and (1,2) ×2 |
| no NaN or constraint error | ✓ zero matches |
| `validate-bundle` passes | ✓ exit 0 |
| no generated path points to the source repository | ✓ zero matches for `escort-ais-prest2` / `src-checkout` |
| installed package inside the target environment | ✓ |

Bundle `20260815T192514Z__small_macrocycle_smoke__bundle__66321c39db51`, run
`20260815T192606Z__small_macrocycle_smoke__rest2__66321c39db51`, both under `cpu_smoke/`.
Logs: `console_logs/10_cpu_smoke.log`, `…/11_cpu_validate_bundle.txt`.

## 6. CUDA validation — **verified**

Preflight before launch: three foreign compute processes on GPUs 6/7/8, **zero on device 0**.

```console
$ export CUDA_DEVICE_ORDER=PCI_BUS_ID
$ escort-explicit validate-env --platform CUDA --device 0 --precision mixed --route smiles
[  ok  ] cuda runtime: 9 device(s): 0, NVIDIA RTX A5000, 580.173.02; 1..8, RTX 3080, 580.173.02
[  ok  ] device: 0: 0, NVIDIA RTX A5000, 580.173.02
[  ok  ] CUDA_DEVICE_ORDER: set to PCI_BUS_ID at launch so --device matches nvidia-smi numbering
[ warn ] precision: could not query CUDA: 'Platform' object has no attribute 'getNumPropertyNames'
environment OK.                                                          # exit 0
```

**The GPU OpenMM selected matches device 0 from `nvidia-smi`**: both report the RTX A5000, and the
tool sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` in-process so the indices cannot diverge.

The `[warn]` is a defect in the *check*, not the setting — see [§9](#9-gaps-between-the-instruction-and-the-cli).
`resolved_config.json` of every CUDA run records `"precision": "mixed"`, so mixed precision was in
force.

### CUDA smoke

```console
$ escort-explicit smoke --system "$HANDOFF/small_macrocycle_smoke.yaml" \
    --experiment "$HANDOFF/smoke.yaml" --out-root "$TARGET_ROOT/cuda_smoke" \
    --platform CUDA --device 0
[smoke] status: completed  exchange rounds 4/4                          # exit 0, 28 s
```

Same acceptance table as §5, all ✓: 3 replicas, pairs (0,1) ×2 and (1,2) ×2, no NaN, `validate-bundle`
exit 0, `status: completed`. 28 s against 75 s on CPU for the identical 585-atom system.

Bundle `20260815T192710Z__small_macrocycle_smoke__bundle__66321c39db51`, run
`20260815T192735Z__small_macrocycle_smoke__rest2__66321c39db51`, both under `cuda_smoke/`.
Logs: `console_logs/12_validate_env_cuda.txt`, `…/13_cuda_smoke.log`, `…/15_cuda_validate_bundle.txt`.

The CPU and CUDA smokes produced the **same bundle config hash `66321c39db51`**, which is correct:
the platform is a runtime property and must not change the identity of the prepared system.

## 7. Target-generated portable RGD bundle — **verified**

```console
$ escort-explicit prepare --system "$HANDOFF/cyclo_rgdfv.yaml" \
    --experiment "$HANDOFF/rgd_rest2_10rung.yaml" \
    --out-root "$TARGET_ROOT/rgd_prepare" --platform CUDA --device 0
EXITCODE=0        19:27:51Z -> 19:40:58Z  (787 s)
```

**Label: target-generated portable RGD bundle.** It is *not* the historical Phase-A or ladder-pilot
bundle, and must never be described as one.

| | |
|---|---|
| path | `rgd_prepare/20260815T192751Z__cyclo_rgdfv__bundle__abf1d9c2f819` |
| config hash | `abf1d9c2f819` |
| prepared-system fingerprint | `ecba4b5a44d31dac1d186a73b3f9e51fddfda11ab7d0694719f4fbae00337969` |
| composition | 3226 atoms (79 solute, 1047 waters, 6 ions), 6496 DOF, 3179 constraints |
| built | `2026-08-15T19:40:57Z`, escort-ais 0.1.0, openmm 8.5.1 |

All thirteen §7 manifest requirements pass (`console_logs/17_rgd_bundle_fieldcheck.txt`):

| required | value |
|---|---|
| canonical molecular hash matches the transferred manifest | ✓ `59d4422635f7…` |
| formal charge zero | ✓ `0` |
| Sage 2.2 recorded | ✓ `openff-2.2.0` |
| AM1-BCC recorded | ✓ `am1bcc` |
| protein force field null | ✓ `null` |
| water force field TIP3P-FB | ✓ `amber19/tip3pfb.xml` |
| box shape dodecahedron | ✓ |
| requested padding 1.2 nm | ✓ |
| nonbonded cutoff 1.0 nm | ✓ |
| minimum-image margin 0.10 nm | ✓ |
| ten exact scale factors | ✓ all ten within 1e-12 of `rest2_ladder(1.0, 0.25, 10, "sqrt")` |
| ladder status `pilot_supported` | ✓ |
| bundle file hashes pass | ✓ `validate-bundle` exit 0 |
| prepared-system fingerprint present | ✓ |

### It reproduces the historical system's composition exactly

`cyclo_rgdfv.yaml` documents the Phase-A prepared system that fed all six matched ladder pilots. A
rebuild from SMILES on a different machine is expected to be *scientifically consistent but not
bitwise identical*. It is in fact considerably closer than that:

| quantity | Phase-A, cited in the manifest | this target-generated bundle |
|---|---|---|
| waters | 1047 | **1047** |
| total particles | 3226 | **3226** |
| ions | Na⁺ 3 / Cl⁻ 3 | **Na⁺ 3 / Cl⁻ 3** |
| box width | 3.66182 nm | **3.66182 nm** |
| minimum-image distance | 2.5893 nm | **2.5893 nm** |
| grown for cutoff | false | **false** |

**One derived figure differs, and it is a convention change, not a discrepancy in the system.**
The manifest cites `realised_ionic_strength_molar 0.15903`; this bundle records
`realized_ionic_strength_molar 0.143483` — from identical ion counts, identical water count and
identical box. The Phase-A figure is consistent with counting ions per 55.5 mol of water
(3 / (1047/55.5) = 0.1590); this wheel computes concentration from the box volume
(3 / (34.72 nm³ · N_A) = 0.1435), which is the definition of ionic strength. The current number is
the better-defined one, but anyone verifying a fresh bundle against the manifest's own documented
expectation will see a ~10 % mismatch and should not read it as a preparation failure.

Log: `console_logs/14_rgd_prepare.log`; checks `…/16_rgd_validate_bundle.txt`, `…/17_rgd_bundle_fieldcheck.txt`.

## 8. Ten-rung RGD mechanical smoke — **verified**

**This is a mechanical smoke. It is not a scientific run**, not a ladder validation, not convergence,
and it does not close the 2 fs / 4 fs hydrogen-mass-repartitioning gate.

The shipped `rgd_rest2_10rung.yaml` was **not** launched unchanged (it requests 2 ns/replica).
`rgd_target_mechanical_smoke.yaml` copies it and changes runtime fields only.

### The first attempt was refused, correctly

Following §8's template literally — including `master_seed: 20260815` — the run was rejected in
0.4 s:

```console
$ escort-explicit rest2 --bundle <bundle> --experiment rgd_target_mechanical_smoke.yaml ...
incompatible experiment: the requested experiment describes a DIFFERENT prepared System than this
bundle contains.
setting                    bundle       proposed
equilibration.seed         20260815     20260816
structure.etkdg.seed       20260814     20260815
                                                                        # exit 8
```

**§8's template is internally inconsistent with the package.** It instructs "changing only runtime
fields" and then prescribes a new `master_seed`, but `master_seed` is build-defining: it derives
`structure.etkdg.seed` and `equilibration.seed`, which determined the System and the equilibrated
state already stored in the bundle. Applied literally, §8 can never run against the §7 bundle; it
would silently require another ~790 s preparation.

**Deviation taken:** `master_seed` held at the bundle's `20260814`; only the `rest2:` block differs
(`exchange_interval_ps` 10.0→1.0, `relaxation_ps` 10.0→1.0, `total_ns_per_replica` 2.0→0.002,
`chunk_ns` 1.0→0.001). This is the faithful reading of "runtime fields only" and preserves §8's own
premise of running against the §7 bundle. Integrator, platform, force field, solvation, cutoff,
constraints, HMR and box shape are untouched.

This refusal is a **feature working**, and it is new in this wheel: `escort_ais/explicit/fingerprint.py`
exists precisely to stop a run being *described* by one configuration and *performed* under another.
It caught a real mismatch on its first exposure to one.

### The run

```console
$ escort-explicit rest2 --bundle rgd_prepare/20260815T192751Z__cyclo_rgdfv__bundle__abf1d9c2f819 \
    --experiment rgd_target_mechanical_smoke.yaml --out-root "$TARGET_ROOT/rgd_smoke" \
    --platform CUDA --device 0
[remd] 10 replicas, s = [1.0, 0.891975308642, …, 0.25],
       T_eff = [300, 336, 380, 432, 496, 575, 675, 803, 972, 1200] K, exchange every 1.0 ps
status: completed  exchange rounds 2/2                                  # exit 0, 6.6 s
```

Run: `rgd_smoke/20260815T194400Z__cyclo_rgdfv__rest2__eacb2418947d`.

| §8 acceptance | result |
|---|---|
| ten replicas created | ✓ `replica_00` … `replica_09` |
| the exact ten-rung ladder recorded | ✓ all ten within 1e-12 of the sqrt rule |
| exit code zero | ✓ |
| `status` reports `completed` | ✓ |
| exactly two production exchange rounds | ✓ at t = 1.0 and 2.0 ps |
| relaxation excluded from exchange accounting | ✓ first attempt at t = 1.0 ps = `relaxation_ps`; relaxation stored separately in `_relaxation/` |
| all nine neighbouring pairs attempted | ✓ (0,1)…(8,9), one attempt each |
| only neighbouring pairs attempted | ✓ every pair satisfies j − i = 1 |
| no NaN, constraint or CUDA error | ✓ zero matches |
| prepared-system fingerprint matches the bundle | ✓ `ecba4b5a44d3…` identical in run and bundle |
| experiment clearly labelled a mechanical smoke | ✓ `rgd_target_mechanical_smoke` |

2 of 9 attempts accepted. **This number carries no information** — nine attempts over 2 ps is far too
little to estimate an acceptance rate, and it must not be compared with the pilots' pair-resolved
statistics.

Log: `console_logs/18_rgd_ten_rung_smoke.log`; checks `…/19_rgd_smoke_acceptance.txt`.

## 9. Gaps between the instruction and the CLI

Neither is a failure; both are recorded so the next session does not rediscover them.

1. **`charge method: am1bcc` is required by §4 but `validate-system` never prints it.** Confirmed
   directly from `cyclo_rgdfv.yaml:30` (`charge_method: am1bcc`) and from the prepared bundle's
   manifest (`parameterization.charge_method = "am1bcc"`). Worth adding to the command's output:
   it is the one field that distinguishes AM1-BCC from a silent fallback, and `validate-env`
   already treats a missing `sqm` as fatal for exactly that reason.
2. **`validate-env --precision` cannot verify itself.** It emits
   `[warn] precision: could not query CUDA: 'Platform' object has no attribute 'getNumPropertyNames'`.
   The OpenMM 8.5.1 API is `Platform.getPropertyNames()`; `getNumPropertyNames` does not exist. The
   *setting* is applied correctly — every CUDA run records `"precision": "mixed"` — so this is a
   broken check, not a broken run. It does mean the one flag §6 asks you to pass is the one flag the
   tool cannot confirm.
3. **§8's `master_seed` prescription cannot be followed** against a §7 bundle. See [§8](#8-ten-rung-rgd-mechanical-smoke--verified).

Unlike the previous session's finding, `escort-explicit rest2 --smoke` is **not** an issue here:
every flag this instruction uses exists in this wheel. Instruction-2 was evidently written against
`10809c7`.

## 10. Differences from the earlier validated wheel

The earlier report validated wheel `d9ff9d02…`. This package ships `50e9a1a5…`. They are not
interchangeable, and the differences are scientific, not cosmetic:

| | earlier wheel `d9ff9d02…` | this wheel `50e9a1a5…` |
|---|---|---|
| `cyclo_rgdfv` box shape | **cube** | **dodecahedron** — the geometry the six matched pilots actually ran in |
| RGD bundle config hash | `a3173143d762` | `abf1d9c2f819` |
| prepared composition | 1554 atoms, 491 waters, 2 ions | 3226 atoms, 1047 waters, 6 ions |
| `rgd_rest2_10rung` ladder status | `validated` | `pilot_supported` — deliberately downgraded; the predeclared conjunctive rule was never formally satisfied |
| ten scale factors | six-decimal truncations | exact, test-locked to the sqrt rule at 1e-12 |
| smoke box-growth abort | worked around with 1.2 nm padding | fixed generally via `system_build.minimum_image_margin_nm` |
| new modules | — | `explicit/fingerprint.py`, `common/gpu_lock.py`, `analysis/window_strat_convergence.py` |

Two consequences worth stating plainly:

* **The previously prepared RGD bundle cannot be reused with this package.** The earlier handoff
  note suggested a further REST2 run would need no re-parameterisation because
  `…__cyclo_rgdfv__bundle__a3173143d762` was already prepared and hash-checked. That bundle was
  built as a **cube**. Under this wheel the config hash differs and the fingerprint guard would
  refuse it. Paying the 787 s again is correct behaviour, not waste.
* **The earlier report's "portable CUDA execution verified" does not transfer to the shipped
  package.** It is accurate about what it tested; it tested a different wheel with a different box
  geometry and a stronger ladder claim.

## 11. What this does *not* establish

A successful smoke establishes mechanical portability only. Specifically **not** established:

* restart safety;
* the 2 fs / 4 fs hydrogen-mass-repartitioning equivalence gate — still open;
* ladder validity for any macrocycle other than cyclo-RGDfV, which itself is only `pilot_supported`;
* convergence of anything;
* long-production readiness — **NO**.

The acceptance figures in §8 are from 2 ps of simulation and must not be quoted as statistics.

## 12. Artefact inventory

```text
/path/to/explicit-test-20260815/
  target_machine_validation.md                    this report
  rgd_target_mechanical_smoke.yaml                the §8 experiment, runtime fields only
  environment/
    environment.target-corrected.yml              the one-token deviation
    conda-list-explicit.txt                       425 lines
    conda-env-from-history.yml
    conda-list-full.txt                           429 packages
  console_logs/
    01_machine.txt  02_handoff_files.txt  03_checksums.txt
    04_env_create_asshipped.log                   the FAILED as-shipped solve
    05_env_create_corrected.log  06_wheel_install.log  07_help.txt
    08_validate_env_cpu.txt  09_validate_system_rgd.txt
    10_cpu_smoke.log  11_cpu_validate_bundle.txt
    12_validate_env_cuda.txt  13_cuda_smoke.log  15_cuda_validate_bundle.txt
    14_rgd_prepare.log  16_rgd_validate_bundle.txt  17_rgd_bundle_fieldcheck.txt
    18_rgd_ten_rung_smoke.log  19_rgd_smoke_acceptance.txt
  cpu_smoke/     20260815T192514Z__small_macrocycle_smoke__bundle__66321c39db51
                 20260815T192606Z__small_macrocycle_smoke__rest2__66321c39db51
  cuda_smoke/    20260815T192710Z__small_macrocycle_smoke__bundle__66321c39db51
                 20260815T192735Z__small_macrocycle_smoke__rest2__66321c39db51
  rgd_prepare/   20260815T192751Z__cyclo_rgdfv__bundle__abf1d9c2f819
  rgd_smoke/     20260815T194400Z__cyclo_rgdfv__rest2__eacb2418947d
```

Environment: `escort-ais-explicit-target` under `/path/to/miniforge3/envs/` (9.5 GB, 429
packages). It is built; do not rebuild it unless deliberately re-testing §3.

## 13. Recommended next actions

1. **Fix `environment.yml` in the package, not only in the repo.** `31f8134` corrected the repo copy
   but the handoff tarball predates it and `flatbottom-basin-cft-mixture` is still broken. Until
   both are fixed, the documented first step of both the branch and the handoff README fails.
2. **Land `31f8134` on `flatbottom-basin-cft-mixture`.** It is contained in no other ref; if that
   branch is ever deleted the fix is lost.
3. **Have `validate-system` print the charge method**, and repair the precision probe
   (`getPropertyNames`, not `getNumPropertyNames`).
4. **Reconcile §8's `master_seed` prescription** with the fingerprint contract, or state in the
   instruction that a new master seed implies re-preparation.
5. **Reconcile the ionic-strength convention** between the manifest's cited Phase-A figure and the
   value this code now computes, so the documented expectation matches a fresh rebuild.
