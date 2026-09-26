# AIS: annealing the TYK2 pocket back to full strength

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a
run executed as written, at that version, on one NVIDIA RTX 3080.

!!! warning "This page measures a limit, not a result"
    The estimate it produces is **not usable**: a Kish effective sample size of **5.6 of 64
    paths**, with one path carrying 39% of the weight. The page is here because that ceiling is
    worth knowing and cheap to hit by accident, and because the same machinery **does** work on
    this system with a smaller region — see [the route that works](#the-route-that-works).

A switching calculation over **TYK2 + ejm_31**: sample the pocket with the ligand and its first
shell weakened, then anneal rigorously back to the physical Hamiltonian. The region is the ligand
plus the 7 residues within 0.35 nm of it — 97 scaled atoms out of 53 030.

**Starts from a built system.** Do [the system page](index.md) first.

## What AIS does here

Two end states with the same atoms and different parameters:

* **V0** is the complex with the ligand and the 0.35 nm shell scaled at τ = 0.4 — solute–solute
  interactions at (1−0.4)² = 0.36, solute–environment at 1−0.4 = 0.6.
* **V1** is the ordinary, physical complex.

md-tools runs ordinary MD on V0 (the **source ensemble**), then starts switching paths from frames
of it. Along each path `V(λ) = (1−λ)V0 + λV1` with λ: 0 → 1. Each endpoint is a configuration of
V1 but not a sample of it; weighting by `exp(−βW)` recovers V1's distribution (Jarzynski). The
method: [AIS](../../openmm_methods/AIS/README.md).

This is the switching counterpart of [the REST2 ladder](REST2.md), deliberately over the same kind
of region, so the two are comparable as sampling strategies rather than as different experiments.

## 1. The two end states

`build/scaler.config` — both end states, written by one scaler run:

```yaml
method: AIS
schedule: {kind: linear, n_states: 2, tau_min: 0.0, tau_max: 0.4}
sidechain_scaling_list: ":17,23,40,92-94,140"
ligand_scaling_dict:
  L31:
    mask: ":291"
    torsion_exclusions: auto
```

The sidechain mask comes from the pocket helper at a **0.35 nm** cutoff, not the 0.5 nm the REST2
page uses:

```bash
python -m md_tools.rest2.pocket build/built.pdb --ligand ":291" --cutoff-nm 0.35
```

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config
```

```text
  schedule                    linear, 2 state(s), tau 0.0 .. 0.4
  selection                   EXPLICIT (md-tools-selective-rest2/1): 97 nonbonded atom(s), 23 torsion central bond(s), 0 CMAP term(s) scaled
  system_state0.xml      tau 0.0       (1-tau)^2 1.000000   1-tau 1.000000
  system_state1.xml      tau 0.4       (1-tau)^2 0.360000   1-tau 0.600000
```

`0 partially owned` central bonds: a torsion is scaled only when **both** its owning residues are
in the region, so nothing straddling the boundary is heated.

**State 0 is τ 0 — V1, the physical end state. State 1 is τ 0.4 — V0, what the source samples.**
The index follows τ; the V-number follows λ, and they run opposite ways. The table is on
[the alanine page](../ALA/AIS.md#2-build-the-two-end-states).

## 2. Configure, with 100 ps paths

`AIS.config` at the dataset root. The part that matters most is `switching_steps`:

```yaml
protocol: AIS
solvent: explicit

dynamics:
  tau: 0.4                       # a claim about build/AIS/, checked against scaler.yaml
  timestep_fs: auto              # 2 fs: this System has no HMR
  temperature_K: 300.0
  seed: 20260924

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 25000
  restrained_npt_steps: 25000     # run at fixed volume: a scaled run is NVT throughout
  unrestrained_npt_steps: 25000
  production_steps: 1000000       # 2 ns of MD on V0: the source ensemble

reporting:
  crd_printout_solute: 500
  crd_printout_whole: 10000       # a whole-system frame every 20 ps: 100 source frames
  info_printout: 500
  checkpoint_printout: 5000

ais:
  number_of_paths: 64
  switching_steps: 50000          # 100 ps per path, NOT 20
  observation_interval_steps: 2500
  parameter_update_interval_steps: 1

ais_source:
  generate: true
  first_frame: 5
  selection: evenly_spaced

collective_variables:
  file: cv.hot-region.yaml        # the ligand's torsions and the pocket's, 886 of them
  interval_steps: 2500
```

!!! tip "Do not ask for every solute torsion here"
    `collective_variables.generate: all_solute_torsions` writes one series per proper torsion of
    the solute. On this 4701-atom solute that is **12 636 series**, and the reporting cost measured
    **371 s of a 417 s stage — 89% of the run**. The file above is the hot region only: the
    ligand's torsions plus the 7 pocket sidechains, 886 series, which brought the same stage to
    56.9 s.

```bash
md-openmm build-md -odir ./AIS-run1 --config AIS.config
```

## 3. Run it

Paths are independent, so they parallelise. The AIS phase is **sync-bound, not compute-bound** —
λ moves every step and each move is a host–device synchronisation, so one worker leaves the card at
about 21%. Four workers sharing one card under MPS reach 100%:

```bash
export CUDA_MPS_PIPE_DIRECTORY="$HOME/.cache/mps/pipe"   # short: a Unix socket caps at 108 bytes
export CUDA_MPS_LOG_DIRECTORY="$HOME/.cache/mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
CUDA_VISIBLE_DEVICES=<your card> nvidia-cuda-mps-control -d

cd AIS-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NPROC=4 ./run.sh   # the SERVER's numbering

echo quit | nvidia-cuda-mps-control
```

That took **2 h 7 m** for 64 paths of 100 ps. Path identity does not depend on the worker count:
global path *n* always writes `AIS_traj000n.nc`, so a campaign resumed on a different number of
cards lands on the same files.

!!! warning "`AIS.out` shows only this rank's paths"
    Under `NPROC=4`, `AIS.out` holds rank 0's 16 paths and the others go to `AIS.out.rank01`…
    `AIS_paths.csv` is the authoritative table with all 64. Reading the summary as if it were the
    campaign is an easy way to quote a quarter of your data — it happened while this page was
    being written.

## 4. Results

From `AIS_paths.csv`, all 64 paths:

```text
paths                    64
work  mean / min / max   -1171.4 / -1186.4 / -1154.2 kJ/mol
dF (Jarzynski, V0 -> V1) -1178.3 kJ/mol  (-472.4 kT)
reduced work sd          3.16 kT
Kish effective samples   5.6 of 64
heaviest single path     39% of the total weight
```

**5.6 of 64 is not a usable estimate.** One path in sixty-four carries 39% of the weight, so the
answer rests on a handful of rare low-work configurations and the remaining paths contribute almost
nothing. A free energy quoted from this would have an error bar nobody can compute from 64 samples.

For reference, on the same estimator: [paracetamol](../paracetamol/AIS.md) with 20 scaled atoms
gives ESS 35.5 of 64 and a work spread of 0.90 kT.

## The route that works

Scale the **ligand alone** — 32 atoms, no pocket sidechains — and the same machinery works:

| region | scaled atoms | switch | work sd | ESS of 64 | heaviest path |
|---|---|---|---|---|---|
| ligand only (`sidechain_scaling_list` omitted) | 32 | 20 ps | 3.17 kT | **16.9** | 11% |
| ligand + 0.35 nm pocket (this page) | 97 | 100 ps | 3.16 kT | 5.6 | 39% |

The two have almost the same work spread and very different effective sample sizes, because ESS
depends on the shape of the distribution's tail and not only its width. ΔF for the ligand-only
transformation is **−625.3 kJ/mol (−250.7 kT)**.

To run it, drop one line from the scaler configuration — no `sidechain_scaling_list` at all, since
an empty mask is refused and omitting the key is how you say "heat no sidechains".

## Where the switching time came from

100 ps rather than 20 ps, because at 20 ps the same region is far worse:

| switch | work sd | ESS of 64 | heaviest path | ΔF (kJ/mol) |
|---|---|---|---|---|
| 20 ps | 5.73 kT | 1.4 | 84% | −1180.4 |
| **100 ps** | **3.16 kT** | **5.6** | 39% | **−1178.3** |

Five times slower switching halves the spread — a 1.8× narrowing, close to the √5 ≈ 2.2 that
dissipation-limited theory predicts — and the two ΔF values agree to 2 kJ/mol, which is the sign
that the improvement is real rather than a reshuffling of weights. Decomposing the variance into a
dissipative part (falling as 1/τ_switch) and a configurational floor gives a floor of **2.07 kT**
for this region: slower switching than 100 ps would keep helping, but it cannot go below that.

That floor is what makes the region size the binding constraint rather than the schedule. At 193
scaled atoms (the 0.5 nm pocket) the floor is **5.34 kT** and no switching time reaches a usable
ESS at all.

## Next

* the exchange counterpart over the same kind of region: [selective REST2](REST2.md)
* the same method where it works cleanly: [AIS: paracetamol](../paracetamol/AIS.md)
* what the scaled states are: [REST2](../../openmm_methods/REST2/README.md)
