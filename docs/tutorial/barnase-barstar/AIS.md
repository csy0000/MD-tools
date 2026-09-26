# AIS: annealing the barnase–barstar salt bridges back to full strength

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a
run executed as written, at that version, on one NVIDIA RTX 3080.

!!! danger "This page measures a limit, and this one has no working variant"
    The estimate is **not usable**: a Kish effective sample size of **5.2 of 64 paths**, with one
    path carrying 34% of the weight. Worse, the 20 ps and 100 ps runs disagree on ΔF by
    **22 kJ/mol** when they should agree exactly — direct evidence that neither is converged.

    Unlike [the TYK2 pocket](../tyk2-ejm31/AIS.md), there is **no smaller region to fall back to**.
    A protein–protein interface cannot be described by 30 atoms, and 30 atoms is roughly where this
    estimator stops working. Read this page for the ceiling, and use
    [selective REST2](REST2.md) for this system.

A switching calculation over the **barnase–barstar** interface: sample it with the salt-bridge
sidechains weakened, then anneal back to the physical Hamiltonian. The region is six residues —
92 scaled atoms out of 29 923.

**Starts from a built system.** Do [the system page](index.md) first.

## Why six residues, and not a cutoff

The region has to be as small as possible, because the work spread grows with it. For this
interface, **hand-picking beats every cutoff**:

| region | residues | scaled atoms |
|---|---|---|
| **salt bridges only** (this page) | **6** | **92** |
| interface at 0.30 nm | 10 | 105 |
| interface at 0.35 nm | 20 | 226 |
| interface at 0.50 nm | 39 | 400 |

A cutoff takes *whole sidechains* of whatever it touches, so 0.35 nm is more than twice the size of
a deliberate choice. The six are barnase **Arg59, Arg83, Arg87** and catalytic **His102** against
barstar **Asp39** and **Glu76** — the salt bridges this complex is known for, chosen by what they
do rather than by a distance.

## 1. The two end states

`build/scaler.config`:

```yaml
method: AIS
schedule: {kind: linear, n_states: 2, tau_min: 0.0, tau_max: 0.4}
sidechain_scaling_list: ":57,81,85,100,146,184"
```

Those are one-based **topology** residue indices, which are not the author numbering — Arg59 is
topology residue 57. [The REST2 page](REST2.md#1-which-residues-line-the-interface) shows how to
get them from the structure rather than counting by hand.

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config
```

```text
  schedule                    linear, 2 state(s), tau 0.0 .. 0.4
  selection                   EXPLICIT (md-tools-selective-rest2/1): 92 nonbonded atom(s), 19 torsion central bond(s), 0 CMAP term(s) scaled
  system_state0.xml      tau 0.0       (1-tau)^2 1.000000   1-tau 1.000000
  system_state1.xml      tau 0.4       (1-tau)^2 0.360000   1-tau 0.600000
```

State 0 is τ 0 — **V1**, the physical end state. State 1 is τ 0.4 — **V0**, what the source
ensemble samples. The index follows τ and the V-number follows λ; they run opposite ways, and the
table is on [the alanine page](../ALA/AIS.md#2-build-the-two-end-states).

## 2. Configure, with 100 ps paths

`AIS.config` at the dataset root:

```yaml
protocol: AIS
solvent: explicit

dynamics:
  tau: 0.4
  timestep_fs: auto              # 2 fs
  temperature_K: 300.0
  seed: 20260924

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 25000
  restrained_npt_steps: 25000    # fixed volume: a scaled run is NVT throughout
  unrestrained_npt_steps: 25000
  production_steps: 1000000      # 2 ns on V0

reporting:
  crd_printout_solute: 500
  crd_printout_whole: 10000
  info_printout: 500
  checkpoint_printout: 5000

ais:
  number_of_paths: 64
  switching_steps: 50000         # 100 ps per path
  observation_interval_steps: 2500
  parameter_update_interval_steps: 1

ais_source:
  generate: true
  first_frame: 5
  selection: evenly_spaced
```

No `collective_variables` block here: this run exists to measure the work distribution, and
`generate: all_solute_torsions` on a 3132-atom solute would spend most of the wall clock writing
torsion series nothing reads.

```bash
md-openmm build-md -odir ./AIS-run1 --config AIS.config
```

## 3. Run it

Four workers sharing one card under MPS, because the AIS phase is sync-bound rather than
compute-bound — λ moves every step, and each move is a host–device synchronisation:

```bash
export CUDA_MPS_PIPE_DIRECTORY="$HOME/.cache/mps/pipe"
export CUDA_MPS_LOG_DIRECTORY="$HOME/.cache/mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
CUDA_VISIBLE_DEVICES=<your card> nvidia-cuda-mps-control -d

cd AIS-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NPROC=4 ./run.sh

echo quit | nvidia-cuda-mps-control
```

64 paths of 100 ps took **1 h 12 m**.

Read the results from `AIS_paths.csv`, not `AIS.out`: under `NPROC=4` the latter holds only rank
0's 16 paths.

## 4. Results

```text
paths                    64
work  mean / min / max   -1829.8 / -1851.3 / -1806.0 kJ/mol
dF (Jarzynski, V0 -> V1) -1843.6 kJ/mol  (-739.1 kT)
reduced work sd          4.25 kT
Kish effective samples   5.2 of 64
heaviest single path     34% of the total weight
```

## 5. Why this is not converged, shown rather than argued

ΔF is a property of the two end states. It does **not** depend on how fast you switch between them,
so two switching times must give the same number. They do not:

| switch | work sd | ESS of 64 | heaviest path | ΔF (kJ/mol) |
|---|---|---|---|---|
| 20 ps | 3.82 kT | 4.4 | 33% | **−1821.4** |
| 100 ps | 4.25 kT | 5.2 | 34% | **−1843.6** |

**22 kJ/mol apart — about 9 kT — for a quantity that should be identical.** That disagreement is
the cleanest possible evidence that neither estimate has converged, and it is worth more than any
error bar computed from within one of the runs.

Note also that 100 ps did **not** help here: the spread is marginally *wider*, and decomposing the
variance between the two switching times gives a *negative* dissipative term, meaning none is
detectable. The spread is configurational — different interface configurations genuinely have
different V1−V0 — and no switching schedule removes that.

Compare [the TYK2 pocket](../tyk2-ejm31/AIS.md#where-the-switching-time-came-from), where 100 ps
halved the spread and the two ΔF values agreed to 2 kJ/mol. Same method, same τ, similar region
size, opposite behaviour: whether slower switching helps is a property of the system and the
region, not something to assume.

## What this says about AIS on interfaces

Collecting every region measured on this estimator, all with 64 paths:

| system | region | scaled atoms | ESS of 64 |
|---|---|---|---|
| paracetamol | whole molecule | 20 | 35.5 |
| TYK2 + ejm_31 | ligand only | 32 | 16.9 |
| **barnase–barstar** | **salt bridges** | **92** | **5.2** |
| TYK2 + ejm_31 | ligand + 0.35 nm pocket | 97 | 5.6 |
| chignolin | whole peptide | 138 | 2.1 |
| TYK2 + ejm_31 | ligand + 0.50 nm pocket | 193 | 2.1 |

The usable ceiling is somewhere between 32 and 92 scaled atoms. A ligand fits under it; a
protein–protein interface does not, and the smallest chemically meaningful region for this one —
six charged sidechains — is already 92 atoms. That is a statement about this transformation, not
about barnase: for **sampling** this interface, use [selective REST2](REST2.md), which heats all 39
interface residues and works.

## Next

* the method that does work on this system: [selective REST2](REST2.md)
* the unbiased reference: [cMD](cMD.md)
* AIS where the statistics are comfortable: [paracetamol](../paracetamol/AIS.md)
