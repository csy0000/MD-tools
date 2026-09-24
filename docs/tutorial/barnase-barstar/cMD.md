# cMD: barnase–barstar from a deposited crystal structure

**Tested against md-tools `0.5.4`.** Every command and every number on this page comes from a
run executed as written, at that version. It has not been re-run for 0.6.1.

Ordinary molecular dynamics on **barnase–barstar**
([PDB 1BRS](https://www.rcsb.org/structure/1BRS)), the ribonuclease and its inhibitor, from the
deposited crystal structure to 10 ns of production. This is the first protein–protein tutorial:
two chains, an interface, crystal waters, residues with missing side-chain atoms, and PROPKA3
protonation. Every command below was run exactly as written and every number is copied from the
files that run produced (one NVIDIA RTX 3080, CUDA, mixed precision).

Building took **16 s**; minimisation, equilibration and 10 ns of production took **40 min**.

Read [cMD: paracetamol](../paracetamol/cMD.md) first; this page explains only what is new.

**Starts from a built system.** Do [the system page](index.md) first: it downloads 1BRS,
chooses assembly 3, assigns protonation and writes `build/built.xml`, `built.pdb` and
`built.log`.

## 1. Generate and run

A 30,000-atom protein complex gets longer equilibration than the shipped example's 10 ps stages.
`cMD.config` at the dataset root:

```yaml
protocol: cMD
solvent: explicit

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 50000      # 100 ps
  restrained_npt_steps: 50000      # 100 ps
  unrestrained_npt_steps: 50000    # 100 ps
  production_steps: 5000000        # 10 ns at 2 fs

reporting:
  crd_printout_solute: 5000        # 10 ps
  info_printout: 5000
  checkpoint_printout: 50000
```

```bash
cd ..                                            # 1BRS/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./run.sh
```

It finished in 39 min 51 s:

```text
run.sh: all stages reported completion
```

## 2. Result

From `cMD-run1/cMD.out`:

```text
Averages
  over                         1000 report(s) in mdout.csv
  Temperature (K)              mean 300.29   rms fluctuation 1.71354
  Density (g/mL)               mean 1.02756   rms fluctuation 0.00270213
  Speed (ns/day)               mean 375.622   rms fluctuation 12.6314
```

The structure of the complex over the run, from
[`barnase_barstar_analysis.py`](barnase_barstar_analysis.py) (run it from `1BRS/`):

```text
frames 1000, time 310-10300 ps
complex CA RMSD (A): mean 0.86, last 1 ns 0.88, max 1.13
chain index 0 (108 CA): CA RMSD mean 0.68 A, last 1 ns 0.67 A
chain index 1 (89 CA): CA RMSD mean 0.83 A, last 1 ns 0.86 A
interface heavy-atom contacts <4.5 A: start 311, mean 297, last 1 ns 286
fraction of initial contacts kept: mean 0.64, last 1 ns 0.64
```

Both proteins stay within 1 Å of the prepared structure and the complex as a whole within 1.1 Å;
the interface keeps its size (311 heavy-atom contacts at the start, 286 in the last nanosecond),
while about a third of the individual contacts exchange for others -- side chains at an interface
move even when the complex does not.

!!! warning "Image the trajectory before measuring anything between the chains"
    The solute trajectory holds the chains as the periodic simulation holds them, so barnase and
    barstar can sit in different periodic images. Measured naively, this run shows a complex RMSD
    of 25 Å and 2700 "contacts". The analysis script makes each molecule whole and puts barstar in
    the image nearest barnase first (`mdtraj.Trajectory.image_molecules`).

**What 10 ns shows, and what it does not.** It shows that the prepared complex -- the chosen
assembly, the built side chains, the PROPKA protonation -- is stable in explicit water on this
timescale. It says nothing about binding affinity, and it does not sample the slower motions of the
interface; for either, the run length and the analysis are a study of their own.

## What it wrote

```text
1BRS/
├── build/     1BRS.cif  build-top.config  built.xml  built.pdb  built.solute.pdb  built.log
│              assembly.json      the expanded chains, their author chains and operators
├── input/  min/
└── cMD-run1/  cMD.out  cMD.log  cMD.xml  solute_prod1.nc  mdout.csv  eq/  ...
```

## Next

* a protein with a bound ligand, from its parameter package:
  [the TYK2 complex](../tyk2-ejm31/index.md)
* register the finished directory as a dataset: [Registering a finished run](../../basics/data-register/index.md)
