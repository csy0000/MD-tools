# explicit-test-20260815 — portable explicit-solvent REST2, target-machine run

**Read this first if you are picking the work up.** It says what exists on disk, what was proven,
what is broken, and what the next session should do. The formal result is
`reports/explicit_solvent/20260815_target_machine_validation.md`; this file is the working context
around it.

## Where you are

| | |
|---|---|
| branch | `validation/explicit-portability-20260815` (pushed), branched from `40f222a` on `flatbottom-basin-cft-mixture` |
| commit | `31f8134` — the environment fix + the validation report |
| machine | `<host>`, Ubuntu 24.04.3, 9 GPUs (device 0 is an RTX A5000, 1–8 are RTX 3080) |

This branch does **not** contain the RGD/cBAR work — that is on `results/rgd-derivatives-cbar`, a
different line off a different base. Do not expect `reports/rgd/...` to be here.

## What was asked, and what came back

Run the portable package from a clean environment outside its source repository, per
`claudecode-instructions/20260815-explicit-test-1.md`, and demonstrate cold MD, hot MD and REST2.

**Verdict: portable CUDA execution verified — with one blocker in `environment.yml`.**

The blocker: line 47 asked for `build`, which does not exist on conda-forge (the package is
`python-build`), and the file pins `channels: [conda-forge, nodefaults]` so nothing else could
supply it. Fixed in `31f8134`. It was the **only** blocker — a dry-run solve with the substitution
resolves 420 packages / 4 GB.

Everything else passed: CPU smoke 74 s, CUDA smoke 31 s, RGD ten-rung bundle prepared in 752 s and
run in 575 s with 200/200 exchange rounds over exactly the nine neighbouring pairs, no NaN or
constraint or CUDA errors, and both free walkers (s = 1.0 and s = 0.25) executing from the wheel.

## What is on disk, and still usable

```
/path/to/explicit-test-20260815/          224 MB
  src-checkout/                worktree of THIS branch (a git worktree of the main repo)
  dist/escort_ais-0.1.0-py3-none-any.whl          sha256 d9ff9d02e2a4...
  environment_diagnostic.yml   the corrected environment file used to build the env
  consuming-repo/portable_rest2_test/
    ...__small_macrocycle_smoke__bundle__66321c39db51
    ...__small_macrocycle_smoke__rest2__66321c39db51   (CPU, then CUDA)
    ...__cyclo_rgdfv__bundle__a3173143d762             the prepared RGD bundle
    ...__cyclo_rgdfv__rest2__a3173143d762              the ten-rung run
  walkers/{cold_smoke,hot_smoke}/                 stage (c) output
```

The environment is **`escort-ais-explicit`** under `/path/to/miniforge3/envs/` — 9.5 GB,
420 packages. It is already built; do not rebuild it unless you are deliberately re-testing step 1.

```bash
export PATH=/path/to/miniforge3/envs/escort-ais-explicit/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /path/to/explicit-test-20260815/consuming-repo
escort-explicit validate-env
```

The RGD bundle is prepared and hash-checked, so a further REST2 run needs no re-parameterisation —
which is the expensive part, 752 s of AM1BCC and equilibration:

```bash
B=$(ls -d portable_rest2_test/*__cyclo_rgdfv__bundle__* | tail -1)
escort-explicit rest2 --bundle $B --experiment rgd_rest2_10rung \
  --out-root "$PWD/portable_rest2_test" --platform CUDA --device 0
```

## Three things that are not typos, and were deliberately not fixed

1. **`escort-explicit rest2 --smoke` does not exist.** The instruction uses it; the CLI rejects it
   (exit 2). No workaround was needed — `rgd_rest2_10rung.yaml` is `total_ns_per_replica: 2.0`, so
   the shipped experiment IS the short run. Decide whether the flag or the instruction is stale.
2. **The CLI does not expose the free cold/hot walkers.** Subcommands are `validate-env`,
   `validate-system`, `validate-bundle`, `prepare`, `rest2`, `smoke`. Stage (c) lives in
   `escort_ais.systems.explicit_baseline.run_md` and is driven by
   `docs/implementation/explicit_solvent/scripts/md.py`. The module IS in the wheel, so the
   capability is portable, but the documented (a)–(d) pipeline cannot be driven by the console
   script alone.
3. **Run directories are `drwxrwxr-x`,** though the acceptance criteria call them immutable.

## Two deviations, so nothing here is over-read

* The environment was created from a **corrected** `environment.yml`. The instruction says not to
  fall back to a pre-existing environment, and I did not — but the shipped file was not what was
  tested.
* The cold and hot walkers were cut from **1000 ns / 200 ns to 0.02 ns** for a mechanical smoke.
  Their only result is throughput: 2827 and 2892 ns/day on 1554 atoms. The REST2 path used shipped
  defaults unmodified.

A successful smoke establishes mechanical portability only — not restart safety, timestep validity,
ladder validity for other macrocycles, convergence, or long-production readiness.

## What the next session could do

1. **Open the PR** for `validation/explicit-portability-20260815`, or merge the one-word
   `environment.yml` fix straight to `flatbottom-basin-cft-mixture` — it blocks that branch's own
   documented first step.
2. **Decide the three design questions above.** Each is a small change; none should be guessed at.
3. **A real explicit run, if wanted.** The bundle is prepared and the ladder is
   `ladder_status: validated` for cyclo_rgdfv. `rgd_rest2_10rung` at 2 ns/replica took 575 s on one
   RTX A5000, so 200 ns/replica ≈ 16 h on that card — an overnight run, not a campaign.
4. **Do not confuse this with the held TIP3P work.** A separate explicit-solvent campaign exists on
   `results/rgd-derivatives-cbar`: boxes built at 1.0 nm padding for all four macrocycles
   (`data/<slug>/topology_tip3p/`), a runner at `scripts/run_explicit_rest2_remd.py`, a frame-rate
   decision record, and a launcher held at the user's request. That path builds its own boxes from
   the implicit prmtops; THIS package builds its own from SMILES. They are not interchangeable and
   should not be merged without deciding which parameterisation route is canonical.

## Unrelated work running on this machine right now (do not kill it)

GPUs 6, 7, 8 are running cBAR + window-strat repeats 2 and 3 for the three RGD macrocycles, from
`results/rgd-derivatives-cbar`. As of 2026-08-15 20:30 repeat 2 is complete on all three
(rgd 52/36 generations, the derivatives 71/55) and repeat 3 is at 15, 51 and 54 cold-window
generations. GPUs 0–5 are free. The six implicit REST2 repeats finished earlier today.
