# Target-machine validation — portable explicit-solvent REST2

> **SUPERSEDED AS A STATEMENT ABOUT THE SHIPPED PACKAGE — the results below remain accurate about
> what they tested.** This report validated the wheel built on this machine that morning,
> sha256 `d9ff9d02e2a4…`. The handoff package now distributed,
> `portable_rest2_handoff_10809c7.tar.gz`, ships a **different** wheel, sha256 `50e9a1a51ecb…`,
> and the differences change results: `cyclo_rgdfv` moved from a **cube** to a **dodecahedron**
> box (so the RGD bundle hash moved `a3173143d762` → `abf1d9c2f819` and its composition from 1554
> atoms / 491 waters to 3226 / 1047), `rgd_rest2_10rung` was downgraded from `ladder_status:
> validated` to `pilot_supported`, and the ten scale factors went from six-decimal truncations to
> exact values. Nothing here is retracted; it simply does not transfer to the package being handed
> out. For the shipped package see
> [`20260815_target_machine_validation_run2/README.md`](20260815_target_machine_validation_run2/README.md).
>
> One consequence is practical: the prepared bundle this report recommends reusing
> (`…__cyclo_rgdfv__bundle__a3173143d762`) **cannot** be reused with the current package — the
> fingerprint guard refuses it, and re-preparation is required.

**Instruction:** `claudecode-instructions/20260815-explicit-test-1.md` ·
**Package under test:** `40f222a` *feat(explicit): portable REST2 — installed CLI, versioned
manifests, hash-checked bundles* · **Run:** 2026-08-15

## Verdict

```
portable CUDA execution verified — with one blocker in environment.yml
```

The package installs from a wheel and runs from a directory outside its source checkout, on CPU and
on CUDA, including the ten-rung RGD bundle. **The shipped `environment.yml` does not solve.** Every
result below was obtained after correcting that file, which is a deviation from the instruction and
is described in full in §1.

A successful smoke establishes mechanical portability only. It says nothing about restart safety,
timestep validity, ladder validity for other macrocycles, convergence, or long-production
readiness.

## Machine

| | |
|---|---|
| host | `<host>` |
| OS | Ubuntu 24.04.3 LTS, Linux 6.8.0-124-generic x86_64 |
| GPU used | device 0, **NVIDIA RTX A5000** (devices 1–8 are RTX 3080) |
| driver | 580.173.02 |
| OpenMM | 8.5.1.dev-f7fa0c2, platforms Reference / CPU / CUDA / OpenCL, precision `mixed` |
| solver | mamba 2.5.0 (no `micromamba` on this host) |

## Input hashes, recorded before installation

| file | sha256 |
|---|---|
| `environment.yml` | `593bac40be510105e1986f760289cdc1e8ef1e7228fee62f14b7e24f35b5c2c4` |
| `cyclo_rgdfv.yaml` | `90d299ae9c3059a7756b3acce12918e65ce56af0d25fd165d657080d63e1c792` |
| `small_macrocycle_smoke.yaml` | `28ca0d5db645ba8ec3998191923fd7fd4e842267a51242b7d4450434bf2041c8` |
| `rgd_rest2_10rung.yaml` | `2aab03b2b87612c38f71227cdd09db3701340ed4a0ae124e42f9b08229302e0f` |
| `escort_ais-0.1.0-py3-none-any.whl` (built here) | `d9ff9d02e2a495628c0b1898d5b19b3362c9d74107c9d3fe81e7a0d91f1ebf4c` |

## 1. Environment — **BLOCKER**

```
$ mamba env create -n escort-ais-explicit -f docs/implementation/explicit_solvent/environment.yml
error    libmamba Could not solve for environment specs
    The following package could not be installed
    └─ build =* * does not exist (perhaps a typo or a missing channel).
critical libmamba Could not solve for environment specs
```

`environment.yml:47` requests **`build`**. On conda-forge that package is **`python-build`**:

```
$ mamba search -c conda-forge --override-channels build         -> No entries matching "build" found
$ mamba search -c conda-forge --override-channels python-build  -> 0.10.0, 0.1.0, 0.0.4
```

The file declares `channels: [conda-forge, nodefaults]`, so no other channel can supply it. `build`
is the PyPI name; the conda-forge name has the `python-` prefix. Fixed on this branch, one word.

**It is the only blocker.** A diagnostic dry-run solve with the substitution succeeds — 420 packages,
4 GB — so nothing else in the pinned set is unsatisfiable on linux-64 today.

**Deviation.** The instruction says to report the constraints and not continue using a pre-existing
environment. I did not fall back to one: a NEW environment was created from the corrected file.
Everything below therefore validates the package, not the shipped environment file.

After creation:

```
$ escort-explicit validate-env                                    exit 0
[  ok  ] escort-ais 0.1.0        [  ok  ] openmm 8.5.1
[  ok  ] rdkit 2025.03.6         [  ok  ] openff.toolkit 0.17.1
[  ok  ] openmmforcefields 0.16.0
[  ok  ] openmm platforms: Reference, CPU, CUDA, OpenCL
[  ok  ] sqm: .../envs/escort-ais-explicit/bin/sqm
$ python -c "import openmm, openmmtools, openff.toolkit, openmmforcefields, rdkit, pymbar, numba"
                                                                  exit 0
```

## 2. CPU smoke, from a directory outside the source checkout

```
$ cd /path/to/explicit-test-20260815/consuming-repo
$ escort-explicit validate-system --system cyclo_rgdfv            exit 0
$ escort-explicit smoke --system small_macrocycle_smoke \
      --out-root "$PWD/portable_rest2_test" --platform CPU        exit 0, 74 s
```

| acceptance | result |
|---|---|
| exit code zero | ✓ |
| every configured replica propagates | ✓ 3 replicas, s = 1.0 / 0.5625 / 0.25 |
| at least two exchange rounds written | ✓ 4 of 4 |
| no missing neighbouring pair | ✓ (0,1) and (1,2), 2 attempts each |
| run directory hash-named | ✓ `20260815T083038Z__small_macrocycle_smoke__rest2__66321c39db51` |
| `status.json` says completed | ✓ |
| bundle validation passes | ✓ exit 0 |

**Not verified: run-directory immutability.** The directory is `drwxrwxr-x` — writable. Nothing in
the run re-writes it, but the acceptance criterion as written ("immutable") is not enforced by
permissions.

## 3. CUDA smoke

```
$ export CUDA_DEVICE_ORDER=PCI_BUS_ID
$ escort-explicit smoke --system small_macrocycle_smoke \
      --out-root "$PWD/portable_rest2_test" --platform CUDA --device 0   exit 0, 31 s
```

3 replicas, 4/4 exchange rounds, `completed`. **No NaN, constraint or CUDA errors.** Device selected
was device 0 as requested (RTX A5000); no silent substitution.

## 4. RGD bundle, ten rungs

```
$ escort-explicit prepare --system cyclo_rgdfv --experiment rgd_rest2_10rung \
      --out-root "$PWD/portable_rest2_test" --platform CUDA --device 0   exit 0, 752 s
  bundle 20260815T084251Z__cyclo_rgdfv__bundle__a3173143d762
  composition 1554 atoms (79 solute, 491 waters, 2 ions), 3148 DOF
$ escort-explicit validate-bundle --bundle <bundle>                      exit 0
$ escort-explicit rest2 --bundle <bundle> --experiment rgd_rest2_10rung \
      --out-root "$PWD/portable_rest2_test" --platform CUDA --device 0   exit 0, 575 s
```

| acceptance | result |
|---|---|
| all bundle hashes match before execution | ✓ `validate-bundle` exit 0, config hash `a3173143d762` |
| ten replicas, exact expected ladder | ✓ `[1.0, 0.891975, 0.790123, 0.694444, 0.604938, 0.521605, 0.444444, 0.373457, 0.308642, 0.25]`, T_eff 300→1200 K |
| relaxation excluded from exchange accounting | ✓ first exchange at 10.0 ps = `relaxation_ps` |
| at least two exchange rounds | ✓ **200 of 200** |
| all neighbouring pairs, even/odd schedule | ✓ 900 attempts over exactly the 9 pairs (0,1)…(8,9); phases 500 / 400 |
| no NaNs, constraint failures, CUDA errors | ✓ zero matching lines |
| manifests identify wheel and bundle hashes | ✓ `run_manifest.json` carries `wheel_sha256 d9ff9d02…` and bundle `a3173143d762` |

Exchange acceptance 0.282 across both chunks (0.289, 0.282). Reported for the record only — this run
is 2 ns per replica and is not evidence about the ladder.

**`--smoke` does not exist.** The instruction's §4 command uses `escort-explicit rest2 … --smoke`;
the shipped CLI rejects it (`unrecognized arguments: --smoke`, exit 2). No workaround was needed:
`rgd_rest2_10rung.yaml` is already `total_ns_per_replica: 2.0`, so the shipped experiment *is* a
short run and nothing was modified to shorten it.

## 5. Free cold and hot walkers

Not reachable through `escort-explicit` — its subcommands are `validate-env`, `validate-system`,
`validate-bundle`, `prepare`, `rest2`, `smoke`. Stage (c) lives in
`escort_ais.systems.explicit_baseline.run_md`, driven by
`docs/implementation/explicit_solvent/scripts/md.py`. That module **is** importable from the
installed wheel, so the capability is portable even though the console script does not expose it.

Both walkers ran from the wheel on the prepared RGD bundle:

| walker | s | T_eff | result |
|---|---|---|---|
| cold | 1.00 | 300 K | exit 0, 2 chunks, mean **2827 ns/day** |
| hot | 0.25 | 1200 K | exit 0, 2 chunks, mean **2892 ns/day** |

**Deviation.** The shipped cold walker default is 1000 ns in 100 ns chunks and the hot 200 ns in
1 ns chunks. Both were cut to 0.02 ns in 0.01 ns chunks — a mechanical smoke, labelled as such in
the config. This is a modified scientific default and the throughput figures are the only thing
those runs establish.

## Portability defects found

1. **`environment.yml` does not solve** — `build` should be `python-build` on conda-forge. Blocker;
   fixed on this branch.
2. **`escort-explicit rest2 --smoke` is documented in the instruction but does not exist.** Either
   the flag or the instruction is stale.
3. **`escort-explicit` does not expose the free cold/hot walkers**, so the documented pipeline
   (a)–(d) cannot be driven end-to-end by the console script alone; stages (c) require the source
   `scripts/` directory or a direct import.
4. **Run directories are not write-protected**, though the acceptance criteria call them immutable.

## Output paths

```
/path/to/explicit-test-20260815/
  src-checkout/                     worktree at 40f222a (source of the wheel)
  dist/escort_ais-0.1.0-py3-none-any.whl
  environment_diagnostic.yml        the corrected environment file
  env_create.log, env_create2.log, probe.log
  consuming-repo/portable_rest2_test/
    20260815T082947Z__small_macrocycle_smoke__bundle__66321c39db51
    20260815T083038Z__small_macrocycle_smoke__rest2__66321c39db51   (CPU)
    20260815T083206Z__small_macrocycle_smoke__rest2__66321c39db51   (CUDA)
    20260815T084251Z__cyclo_rgdfv__bundle__a3173143d762
    20260815T085712Z__cyclo_rgdfv__rest2__a3173143d762
  walkers/{cold_smoke,hot_smoke}/   stage (c) smoke output
```
