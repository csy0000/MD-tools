# Validation matrix, 2026-08 — throughput report

**Date** 2026-08-31
**Runtime** MD-templates `dev` @ `80eb91c`, MD-project `dev` @ `df84e67` (both merged today)
**Hardware** 8 × NVIDIA RTX 3080 (10 GB), driver 580.173.02; an RTX A5000 held in reserve
**Environment** OpenMM 8.6.0, CUDA, mixed precision, 2 fs timestep

## What ran

2 systems × 2 solvents × 5 MD types = **20 simulations**, all completed.

| | ALA (ACE-ALA-NME) | Phenol (CCD id IPH) |
|---|---|---|
| TIP3P explicit | 1790 particles | 1808 particles |
| GBn2/mbondi3 implicit | 22 particles | 13 particles |

Per cell: cMD 5 ns, hot-cMD 5 ns at τ = 0.5, AIS (100 paths × 1 ps, τ 0.5 → 0), REST2 5 ns
with 4 replicas, and a REST2 extension of a further 5 ns. **200 ns of dynamics in 3.48
GPU-hours of production**, a mean of **57 ns/h per GPU**.

## Throughput

ns/day per GPU. A REST2 figure is per replica; multiply by 4 for the ladder's aggregate.

| system | solvent | type | atoms | ensemble | GPUs | min | **ns/day/GPU** |
|---|---|---|---|---|---|---|---|
| ALA | GBn2 | cMD | 22 | NVT | 1 | 2.6 | **2769** |
| ALA | GBn2 | hot-cMD | 22 | NVT | 1 | 2.6 | **2769** |
| ALA | GBn2 | REST2 | 22 | NVT | 4 | 5.0 | **1440** |
| ALA | GBn2 | REST2 ext | 22 | NVT | 4 | 5.4 | **1333** |
| ALA | TIP3P | cMD | 1790 | NPT | 1 | 4.6 | **1565** |
| ALA | TIP3P | hot-cMD | 1790 | NPT | 1 | 4.7 | **1532** |
| ALA | TIP3P | REST2 | 1790 | NVT | 4 | 6.4 | **1125** |
| ALA | TIP3P | REST2 ext | 1790 | NVT | 4 | 7.0 | **1029** |
| Phenol | GBn2 | cMD | 13 | NVT | 1 | 2.1 | **3429** |
| Phenol | GBn2 | hot-cMD | 13 | NVT | 1 | 2.2 | **3273** |
| Phenol | GBn2 | REST2 | 13 | NVT | 4 | 4.9 | **1469** |
| Phenol | GBn2 | REST2 ext | 13 | NVT | 4 | 4.3 | **1674** |
| Phenol | TIP3P | cMD | 1808 | NPT | 1 | 4.3 | **1674** |
| Phenol | TIP3P | hot-cMD | 1808 | NPT | 1 | 4.4 | **1636** |
| Phenol | TIP3P | REST2 | 1808 | NVT | 4 | 6.7 | **1075** |
| Phenol | TIP3P | REST2 ext | 1808 | NVT | 4 | 5.6 | **1286** |

Raw data: `results/validation-2026-08/throughput.csv`.

## A correction to an earlier measurement

Earlier today a 6-replica ALA/TIP3P ladder at a **2 ps** exchange interval was measured at
**7.4 ns/h per GPU (177 ns/day)**, and that figure was reported. The same system here, at a
**10 ps** interval with 4 replicas, runs at **46.9 ns/h per GPU (1125 ns/day)** — **6.3×
faster**.

A 5× reduction in exchange barriers does not account for 6.3×, and the honest reading is
that the earlier number was **contaminated**: that run shared the machine with repeated full
test-suite executions, several of which run OpenMM on the CPU platform. It measured a
loaded machine, not the ladder. The 177 ns/day figure should not be used for anything.

This also settles the question raised at the time — whether ~177 ns/day was too slow for a
system this size. It was. The hardware delivers 1000–1700 ns/day on these systems, which is
in the expected range.

## What the numbers show

**The exchange barrier dominates small-system REST2.** ALA/TIP3P cMD at 1565 ns/day against
its 4-replica ladder at 1125 ns/day per replica: the ladder gives up ~28% of single-walker
throughput to synchronise 4 ranks every 10 ps. On the implicit systems the loss is far
larger — 2769 → 1440 ns/day for ALA, a 48% cost — because at 13–22 particles the dynamics
between barriers is nearly free and the barrier is almost the whole cost.

**System size barely matters here, but solvent does.** 1790 vs 22 particles changes cMD
throughput by only 1.8× (1565 → 2769 ns/day). Neither system saturates a 3080; both are
dominated by per-step kernel-launch overhead rather than by force evaluation. The larger
gap is between explicit and implicit, and it is mostly the particle count in PME.

**NPT costs little.** The explicit cMD cells ran NPT with a `MonteCarloBarostat`; the
explicit ladders ran NVT. At 1565 (NPT, 1 GPU) vs 1125 (NVT, 4-rank ladder) the barostat's
volume moves are not separable from the exchange cost in this data, so no NPT-vs-NVT number
is claimed here — it would need a matched pair, which this matrix does not contain.

**Extensions cost the same as the parent**, within noise: 1029 vs 1125 (ALA/TIP3P), 1286 vs
1075 (phenol/TIP3P). Continuing out-of-place is not a penalised path.

**Scheduling.** Implicit half: 14 jobs in 11.9 min. Explicit half: 14 jobs in 17.3 min. Two
4-replica ladders exactly saturate 8 GPUs; single-GPU cMD jobs fill the gaps. Measured
utilisation during the explicit phase was 90–99% on all eight devices. The pool did sit idle
between the two campaign halves — a scheduling gap of the driver being invoked twice, not a
hardware limit.

## Correctness checks that accompanied the timings

- All 20 runs report `run_status: completed`; all four AIS runs report `status: completed`
  with 100/100 paths, 21 observations each, and no failed path.
- Extension contract on all four ladders: parent pinned by sha256 (manifest, analysis,
  checkpoint and all four state trajectories), boundary gap exactly one 10 ps output
  interval, 500 + 500 frames, 10000 ps chains.
- cpptraj V7.6.2 concatenated `REST2/remd0.nc` + `REST2_ext1/remd0.nc` for every ladder into
  1000 frames spanning 10–10000 ps, uniformly spaced, zero duplicate times; and parsed all
  eight `rem.log` files as type Hamiltonian.

## What this does not establish

- **No convergence or sampling claim.** 5 ns per state is a mechanism and performance test.
  Nothing here says the ladders mixed adequately or that any ensemble is converged.
- **Implicit-solvent ligand results are experimental.** `implicit.py:321` labels GBn2 ligand
  systems as not Amber igb=8 parity, and the repository's own phenol REST2 example declines
  to offer implicit ligand REST2 at all. The two phenol/GBn2 cells exercise the machinery;
  they are not a validated model.
- **The two implicit cMD cells ran NVT, not the requested NPT.** `simple.py:429` refuses NPT
  for implicit solvent, because a system with no box has no volume for a barostat to control.
  The explicit cMD cells did run NPT as intended.
- **No NPT-vs-NVT cost figure**, for the reason given above.
- Local measurement only, on one machine, one run each. No error bars.
