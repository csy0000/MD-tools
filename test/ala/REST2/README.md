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
| constraints | bonds to hydrogen; rigid water; **no** hydrogen-mass repartitioning |
| integrator | `LangevinMiddleIntegrator`, 300 K, friction 1/ps, **2 fs** |

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
duration_per_segment            5 ns
timestep                        2 fs
steps per segment               2,500,000        (exact; a non-integer count is refused)
number_of_exchanges_per_segment 100
steps per exchange round        25,000           (= 50 ps)
```

10 ns per replica is therefore `NUMBER_OF_SEGMENTS=2`.

## Reporting

Two trajectory streams, as required by the protocol:

| stream | interval | steps |
|---|---|---|
| full system | 100 ps | 50,000 |
| solute / selected atoms | 10 ps | 5,000 |

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

Not measured. The full protocol is 6 replicas x 10 ns at 2 fs in explicit water; estimate from your
own hardware before committing to it. A short CPU smoke version exists for CI — see
`tests/` — and a smoke run is not evidence of convergence or of anything scientific.

## Extension

See `extension/` for requesting further 5 ns segments from the committed checkpoints of this run.
