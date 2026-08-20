# Worked example — cyclo-RGDfV, ten-replica REST2

cyclo-(Arg-Gly-Asp-D-Phe-Val) run end to end: restrained minimisation, restrained NVT, restrained
NPT, 1 ns conventional MD, then 10 ns of REST2 per replica delivered as two 5 ns segments.

## The molecule is vetted, not inferred

**Do not infer this chemistry from the string "RGDfV".** The definition used here is taken verbatim
from the packaged manifest `src/md_templates/openmm/manifests/systems/cyclo_rgdfv.yaml`, which
records:

| | |
|---|---|
| topology | **cyclic** pentapeptide, head-to-tail |
| sequence | cyclo-(Arg-Gly-Asp-D-Phe-Val) — note the **D**-Phe |
| stereocentres | 4 |
| charge state | zwitterion: Asp side chain deprotonated `[O-]`, Arg guanidinium protonated `[NH2+]` |
| total formal charge | **0** |
| formula | C26H38N8O7 |
| SMILES sha256 | `59d4422635f77292ed94c609a76808df5c97110be09cd9779819270363143159` |

The example JSON repeats the SMILES and its hash so that a mismatch is a failure rather than a
silent substitution.

## The force-field route is a ligand route, and that is deliberate

cyclo-RGDfV is parameterised through **openff-2.2.0 (Sage) + AM1-BCC**, not as an ff19SB peptide.
The manifest states this explicitly and sets `protein_forcefield: null` so an accidental ff19SB load
is an error. Its wording is worth repeating: it is a cyclic pentapeptide, so "peptide" is a
defensible guess and a wrong one — an ff19SB run of this molecule would be a different calculation
with the same name.

**Water is TIP3P for the same reason.** Sage's vdW parameters were trained against condensed-phase
properties in TIP3P, and AM1-BCC charges were derived to be consistent with TIP3P-era additive
force fields. OPC is the matched partner for ff19SB, not for Sage. Pairing Sage with OPC would be
off-model.

**Documented deviation from the task instruction.** The instruction's shared-preparation block
specifies ff19SB, OPC water and a truncated-octahedral box. This example uses the vetted
openff-2.2.0/AM1-BCC route, TIP3P, and a **dodecahedral** box instead. The instruction itself scopes
ff19SB to "where supported by the vetted topology route", and this route is a ligand
parameterisation. The box shape is evidence-bound: the manifest records that the ten-rung ladder was
selected on a dodecahedron-prepared system, and attaching that evidence to a different solvent
geometry would silently change the conditions the acceptance statistics were measured in.

| | instruction | used here | why |
|---|---|---|---|
| solute FF | ff19SB | openff-2.2.0 + AM1-BCC | vetted route; ff19SB would be a different calculation |
| water | OPC | TIP3P | matched to Sage/AM1-BCC; OPC is matched to ff19SB |
| box | truncated octahedron | dodecahedron | the ladder evidence was measured in a dodecahedron |

## System

| | |
|---|---|
| box | dodecahedron, 1.2 nm solute-image padding |
| salt | 0.15 M NaCl, counterions counted separately from added pairs |
| nonbonded | PME, 1.0 nm real-space cutoff |
| constraints | bonds to hydrogen; rigid water |
| HMR | **hydrogen mass 3.024 amu**, solute scope; water never repartitioned |
| integrator | `LangevinMiddleIntegrator`, 300 K, friction 1/ps, **4 fs** (enabled by HMR) |

## The tau ladder

Ten replicas, `tau` linearly spaced from 0.0 to 0.5. This resolves to exactly the ladder the
project already validated — converting the shipped profile reproduced every rung to within 3.7e-13:

```
tau  0.000000  0.055556  0.111111  0.166667  0.222222
     0.277778  0.333333  0.388889  0.444444  0.500000

s    1.000000  0.891975  0.790123  0.694444  0.604938
     0.521605  0.444444  0.373457  0.308642  0.250000
```

`tau` is the input; `s = (1 - tau)^2` and `sqrt(s) = 1 - tau` are derived. Replica 0 is the
physical, unscaled Hamiltonian.

## Segments and exchange

```
n_exchange_per_segment          1000
exchange_interval               5 ps
timestep                        4 fs
steps per exchange round        1,250            (5 ps / 4 fs, must divide exactly)
steps per segment               1,250,000        (= 1000 x 1,250, a PRODUCT)
duration_per_segment            5 ns             DERIVED, not stated
```

The segment length is **derived by multiplication**, never stated. That is the point: a product is
exact, so the only quantity that can fail to divide is the interval itself -- which is the one you
chose directly. Stating a duration and an exchange count instead would make the interval a quotient
that might not divide, and `duration_per_segment` is therefore **refused** for REST2.

10 ns per replica is `NUMBER_OF_SEGMENTS=2`.

## Device mapping

Ten replicas rarely match the GPU count. The mapping is round-robin over an ordered device list and
is deterministic, so several replicas sharing a device is allowed and recorded rather than refused:

```
MD_DEVICES=1,2,3,4,5,6,7,8   ->  replicas 0..7 one each, replicas 8 and 9 share devices 1 and 2
```

Inspect the hardware first; do not assume every installed GPU should be mixed into one synchronous
run. On a mixed machine, pin `CUDA_DEVICE_ORDER=PCI_BUS_ID` and select devices of a single
architecture — ranks on different GPU generations make the slowest device set the pace and can
complicate reproducibility.

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
python MD_system_gen.py -i cyclo_rgdfv.smi \
       -o rgd_system --config system_config.json

# 2. generate the staged protocol from it.
python MD_input_gen.py --system rgd_system/system_manifest.json \
       -o rgd_run --config md_config.json
```

The SMILES is the vetted one from `manifests/systems/cyclo_rgdfv.yaml`, and `ligand_build` states the charge, stereochemistry, protonation, conformer, charge model and parameterisation route explicitly -- none of which a SMILES string implies.

This produces:

```
rgd_run/
    inputs/       immutable copy of the prepared system + its checksums
    min/          min.json      min.sh
    eq_nvt/       eq_nvt.json   eq_nvt.sh
    eq_npt/       eq_npt.json   eq_npt.sh
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
cd rgd_run && ./run_all.sh          # all stages in order
cd rgd_run/min && ./min.sh          # or just one
```

`min`, `eq_nvt`, `eq_npt` and `cMD_1` execute directly. **`REST2_1` is delegated** to the expert
CLI, which owns the committed-generation restart contract -- `REST2_1.sh` prints the exact command.

`NUMBER_OF_SEGMENTS` in `run_all.sh` controls how many REST2 segments run. It is an execution
choice and never appears in the scientific JSON.

## Commands

```bash
source /path/to/md-stack/activate-md-stack.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID

md-openmm config resolve --input rgdfv_rest2.json
MD_DEVICES=1,2,3,4,5,6,7,8 NUMBER_OF_SEGMENTS=2 ./run_all.sh
```

## Expected outputs

```
outputs/
  bundle/                     prepared system, hashes, provenance
  md/                         1 ns conventional MD
  rest2/                      REST2 run directory
    replica_00 .. replica_09
    exchanges.csv             durable exchange history
    committed.json            the atomic restart boundary
```

`outputs/` is gitignored.

## Execution status

**Not executed.** This example is generated and its configuration resolves and validates, but the
protocol has not been run. Nothing here is a scientific result, and a resolved configuration is not
a completed simulation.

## Extension

See `extension/` for requesting further 5 ns segments from the committed checkpoints of this run.
