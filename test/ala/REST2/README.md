# Worked example — alanine dipeptide, six-replica REST2

Neutral capped alanine dipeptide, `ACE-ALA-NME`, run end to end through the OpenMM-centric
configuration: restrained minimisation, restrained NVT, restrained NPT, 1 ns conventional MD, then
10 ns of REST2 per replica delivered as two 5 ns segments.

This is a **route and protocol example**, not a scientific result. Alanine dipeptide in explicit
water says nothing about macrocycle conformational ladders, and nothing measured here should be
quoted as if it did.

## System

| | |
|---|---|
| solute | `ACE-ALA-NME`, 22 atoms, neutral |
| solute force field | ff19SB (`amber19/protein.ff19SB.xml`) |
| water | **OPC** (`amber19/opc.xml`) |
| box | dodecahedron, 1.2 nm solute-image padding |
| salt | 0.15 M NaCl, counterions counted separately from added pairs |
| nonbonded | PME, 1.0 nm real-space cutoff |
| constraints | bonds to hydrogen; rigid water |
| HMR | **hydrogen mass 3.024 amu**, solute scope; water never repartitioned |
| integrator | `LangevinMiddleIntegrator`, 300 K, friction 1/ps, **4 fs** (enabled by HMR) |

**Why OPC and not TIP3P.** ff19SB's backbone parameters were fit with OPC, and that pairing is the
published recommendation. The packaged profile defaults to `tip3pfb`, so this example overrides it
deliberately rather than inheriting a mismatched pair.

**OPC is a four-site model.** It carries a virtual site per water, so the OpenMM particle count
exceeds the topology atom count. Anything that indexes particles must use the resolved indices
recorded in the bundle, not an atom count.

## The tau ladder

Six replicas, `tau` linearly spaced from 0.0 to 0.5:

| replica | tau | derived s | derived sqrt(s) | derived T_eff (K) |
|---|---|---|---|---|
| 0 | 0.0 | 1.00 | 1.0 | 300 |
| 1 | 0.1 | 0.81 | 0.9 | 370.4 |
| 2 | 0.2 | 0.64 | 0.8 | 468.8 |
| 3 | 0.3 | 0.49 | 0.7 | 612.2 |
| 4 | 0.4 | 0.36 | 0.6 | 833.3 |
| 5 | 0.5 | 0.25 | 0.5 | 1200 |

`tau` is the source parameter and the only one that is an input. `s`, `sqrt(s)` and `T_eff` are
derived by one shared function and recorded as labelled diagnostics. The Hamiltonian is unchanged
from the `s` formulation:

```
solute-solute terms       s       = (1 - tau)^2
solute-environment terms  sqrt(s) = (1 - tau)
environment terms         1
```

Replica 0 is the physical, unscaled Hamiltonian; its trajectory is the one that is the answer.

**Omega exclusion is on by default.** Peptide omega torsions in the enhanced region stay unscaled,
because softening them lets the backbone sample cis-amide states that are an artefact of the
scaling rather than physics. Disable it explicitly with
`production.omega_exclusion.enabled: false` if that is what you want.

## Segments and exchange

The JSON declares the length of **one** segment. It does not say how many to run — that is the
driver script's job, and how many actually committed is recorded in the run manifest.

```
duration_per_segment            5 ns             stated
number_of_exchanges_per_segment 1000             stated
timestep                        4 fs
steps per segment               1,250,000        (5 ns / 4 fs, must divide exactly)
steps per exchange round        1,250            (1,250,000 / 1000, must divide exactly)
exchange_interval               5 ps             DERIVED, not stated
```

Both divisions happen in integer step space and both are refused rather than rounded: an exchange
interval off by a step drifts the schedule out of alignment with the committed watermark while the
run still looks healthy. An exchange count that does not divide is rejected with nearby counts that
do.

The segment length is **derived by multiplication**, never stated. That is the point: a product is
exact, so the only quantity that can fail to divide is the interval itself -- which is the one you
chose directly. Stating a duration and an exchange count instead would make the interval a quotient
that might not divide, and `duration_per_segment` is therefore **refused** for REST2.

10 ns per replica is therefore `NUMBER_OF_SEGMENTS=2`.

## Reporting

Two trajectory streams, as required by the protocol:

| stream | interval | steps |
|---|---|---|
| full system | 100 ps | 25,000 |
| solute / selected atoms | 10 ps | 2,500 |

## Hydrogen-mass repartitioning

Mass is moved from heavy atoms onto their bonded hydrogens until each hydrogen reaches
**3.024 amu** (3 x 1.008). The heavy partner loses exactly what the hydrogen gains, so total mass is
conserved and centre-of-mass dynamics are untouched — the implementation asserts this.

This is what buys the 4 fs timestep: it lowers the frequency of the bond-*angle* motions involving
hydrogen, which are the fastest remaining degrees of freedom once `constraints: HBonds` has removed
the bond *stretches*. HMR without those constraints would not help.

**Water is never repartitioned.** Rigid water is fully constrained, so its hydrogen masses do not
limit the timestep, and changing them would alter water's rotational dynamics — and therefore its
diffusion constant and dielectric relaxation — for no benefit. `hmr_scope: solute` enforces this.

**HMR is applied to the base System before any tau scaling**, so every replica has identical masses.
That matters for exchange validity: REST2 swaps configurations between replicas, and the acceptance
criterion used here is potential-energy-only, so masses must not differ across the ladder.

HMR changes the equations of motion, not the potential energy surface. Thermodynamic averages are
unaffected; kinetic quantities such as diffusion constants and rate constants are **not** directly
comparable to an unrepartitioned run.

## Setup with the public generators

Preparation and protocol generation are separate commands, because they answer different questions
and change at different times. Preparing the system derives charges and solvates -- expensive, and
unchanged when you alter a protocol. Generating a protocol is cheap and needs no GPU.

```bash
# 1. prepare the molecular system. Runs NO dynamics.
python MD_system_gen.py -i ace_ala_nme.pdb \
       -o ala_system --config system_config.json

# 2. generate the staged protocol from it.
python MD_input_gen.py --system ala_system/system_manifest.json \
       -o ala_run --config md_config.json
```

The PDB is a peptide, so `system.type` is declared: a PDB does not say what it holds.

This produces:

```
ala_run/
    inputs/       immutable copy of the prepared system + its checksums
    min/          min.json      min.sh
    eq_nvt/       eq_nvt.json   eq_nvt.sh
    eq_npt_1/     eq_npt_1.json eq_npt_1.sh    restrained NPT
    eq_npt_2/     eq_npt_2.json eq_npt_2.sh    free NPT
    cMD_1/        cMD_1.json    cMD_1.sh
    REST2_1/      REST2_1.json  REST2_1.sh
    run_all.sh    run_manifest.json    run.log
```

Each stage owns its configuration, its launcher and (once run) its outputs. Each stage JSON names
the topology and the input **State** it consumes and which stage produced it, so a stage cannot
silently start from the wrong coordinates -- running one before its predecessor fails with exactly
that message.

Run everything, or one stage at a time:

```bash
cd ala_run && ./run_all.sh          # all stages in order
cd ala_run/min && ./min.sh          # or just one
```

Every stage runs through its own `.sh`. The single-shot stages execute in-process; **`REST2_1` is
delegated** to the runner, which owns the committed-generation restart contract. `REST2_1.sh`
assembles the bundle and calls it, passing the previous run directory if there is one -- so running
`REST2_1.sh` again continues the chain rather than restarting it. The stage layer decides nothing
about restarts.

Each launcher records the interpreter that generated the project and preflights it before running:
the stack's activation script puts AmberTools' interpreter first on PATH and that one cannot import
`openmm` or `md_templates`. Override with `PYTHON=... ./run_all.sh`.

`NUMBER_OF_SEGMENTS` in `run_all.sh` controls how many REST2 segments run. It is an execution
choice and never appears in the scientific JSON.

## Commands

```bash
source /path/to/md-stack/activate-md-stack.sh     # or activate your own environment
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# inspect what the JSON resolves to before running anything
md-openmm config resolve --input alanine_rest2.json

# the whole chain
./run_all.sh
```

`run_all.sh` takes the GPU list and the segment count from the environment:

```bash
MD_DEVICES=1,2,3,4,5,6 NUMBER_OF_SEGMENTS=2 ./run_all.sh
```

Replicas are mapped to devices deterministically and round-robin. Six replicas on six GPUs is one
each; six replicas on two GPUs is three each, which is allowed and recorded rather than refused.

## Expected outputs

```
outputs/
  bundle/                     prepared system, hashes, provenance
  md/                         1 ns conventional MD
  rest2/                      REST2 run directory
    replica_00 .. replica_05
    exchanges.csv             durable exchange history
    committed.json            the atomic restart boundary
```

`outputs/` is gitignored. Do not commit trajectories, checkpoints or serialized States.

## Runtime

Not measured. The full protocol is 6 replicas x 10 ns at 4 fs in explicit water; estimate from your
own hardware before committing to it. A short CPU smoke version exists for CI — see
`tests/` — and a smoke run is not evidence of convergence or of anything scientific.

## Extension

See `extension/` for requesting further 5 ns segments from the committed checkpoints of this run.
