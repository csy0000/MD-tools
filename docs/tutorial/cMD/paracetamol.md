# cMD: paracetamol in explicit water

A complete conventional MD run of one small molecule, from a SMILES string to 200 ps of NPT
production, with md-tools **0.5.3**. Every command below was run exactly as written, and every
number on this page is copied from the files that run produced (one NVIDIA RTX 3080, CUDA, mixed
precision).

It takes about two minutes, most of it the partial-charge calculation.

## What you need

* md-tools 0.5.3 installed, with CUDA — see [Installing](../../install.md). Check it:

  ```bash
  md-openmm --version          # md-tools 0.5.3
  ```

* a machine configuration — see [Machine configuration](../../machine-configuration.md). CUDA is
  the default and is mandatory; nothing here falls back to the CPU.

## 1. The dataset root

One directory holds everything about this system. `build/` is the system itself; every run of it
sits beside `build/` and shares it.

```bash
mkdir -p PARA/build
cd PARA/build
```

The molecule, as a one-record SMILES file (`paracetamol.smi`):

```text
CC(=O)Nc1ccc(O)cc1 paracetamol
```

and a build configuration (`build-top.config`) that says what the input *is*:

```yaml
solute:
  kind: ligand
```

`kind: ligand` reads `-i` as a small molecule and parameterises it with the small-molecule force
field. Everything else — Sage 2.2.1, AM1-BCC charges, TIP3P, a 1.5 nm dodecahedron, 0.15 M NaCl,
HBonds constraints — is the documented default, and the build log lists each value with
`(default)` beside it. See [Building a system](../../build-top.md).

## 2. Build the system

```bash
md-openmm build-top -i paracetamol.smi \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

What the log reports:

```text
Input interpretation
  interpreted as              single-molecule SMILES
  small-molecule FF           sage-2.2.1
  solvent treatment           explicit TIP3P, periodic
Preparation
  solvation    : 592 waters, ions {'NA': 2, 'CL': 2}, box dodecahedron (19.1 nm^3)
Validation
  particle agreement          1800 atoms == 1800 particles  OK
  HMR                         NOT applied; masses are the force field's own, recommend 2.0 fs
Outputs
  built.xml  built.pdb  built.solute.pdb  built.sdf
Summary
  built 1800 particles, 597 residues, explicit TIP3P
  status: completed
```

`built.xml` (the serialised OpenMM System) and `built.pdb` (the matching coordinates) are a pair:
the build refuses to write either if their atoms disagree. `built.sdf` keeps the molecule's bond
orders, which a topology cannot carry. cMD does not need it; a REST2 run of this molecule does —
see [REST2: paracetamol](../REST2/paracetamol.md).

## 3. Generate the run

Back at the dataset root, a cMD configuration (`cMD.config`). These lengths are the shipped
example's — short enough to watch finish, not a production recommendation:

```yaml
protocol: cMD
solvent: explicit

stages:
  minimization_iterations: 1000
  restrained_nvt_steps: 5000
  restrained_npt_steps: 5000
  unrestrained_npt_steps: 5000
  production_steps: 100000

reporting:
  crd_printout_solute: 500
  info_printout: 5000
  checkpoint_printout: 5000
```

Every length is an integer step count. At the 2 fs timestep the System's masses allow, that is
10 ps for each of the three equilibration stages and 200 ps of production.

```bash
cd ..                                            # PARA/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
```

```text
PARA/
├── build/        built.xml  built.pdb  built.sdf  built.log  ...
├── input/        min.in  eq_1.in  eq_2.in  eq_3.in  cMD.in     <- shared by every run here
├── min/          min.py                                          <- one minimised structure, shared
└── cMD-run1/     run.sh  cMD.py  resolved.config  run.config  build-md.log  eq/
```

`input/` and `min/` belong to the system, not to this run. See
[The run layout](../../run-layout.md).

## 4. Run it

```bash
cd cMD-run1
./run.sh
```

`run.sh` is five `md-openmm md-run` calls, in order:

| stage | what | ensemble |
|---|---|---|
| `min` | energy minimisation, 1000 iterations | — |
| `eq_1` | solute heavy atoms restrained | NVT |
| `eq_2` | solute heavy atoms restrained | NPT |
| `eq_3` | restraint released | NPT |
| `cMD` | production | NPT |

It finished in 35 s and ended with:

```text
run.sh: all stages reported completion
```

Choose the device with `CUDA_VISIBLE_DEVICES` (this run used `CUDA_VISIBLE_DEVICES=1`). Run it
again and every stage is skipped, because each already reports `status: completed` in its own
machine record; an interrupted stage resumes from its last checkpoint.

## 5. What it wrote

```text
cMD-run1/
├── cMD.out            the readable summary of production -- read this first
├── cMD.log            the machine record (provenance; what completion is read from)
├── cMD.xml            final state: positions, velocities, box
├── solute_prod1.nc    solute trajectory, AMBER NetCDF, a frame every 500 steps
├── mdout.csv          the state table: energies, temperature, volume, density, speed
├── energy_components.csv
├── cMD.checkpoints/
└── eq/                eq_1..eq_3: .out  .log  .xml  solute_eq_<n>.nc  mdout_eq_<n>.csv
```

From `cMD.out`:

```text
System
  atoms                        1800
  residues                     597 (HOH 592, CL 2, NA 2, UNL 1)
  box                          3.000 x 3.000 x 2.121 nm  volume 19.092 nm^3
Method
  nonbonded                    PME, cutoff 1.0 nm, Ewald tolerance 0.0005
  constraints                  1785 bond(s)
  barostat in system           MonteCarloBarostat
Run
  timestep                     2.0 fs
  steps                        100000 (200 ps)
  ensemble                     NPT
  platform                     CUDA
Averages
  Temperature (K)              mean 300.743   rms fluctuation 7.13978
  Density (g/mL)               mean 1.00088   rms fluctuation 0.0101666
  Speed (ns/day)               mean 1225   rms fluctuation 297.767
Summary
  cMD: 100000 steps completed, 200 ps
status               completed
```

The temperature sits at the 300 K thermostat and the density at that of water, which is what a
correctly built and equilibrated box looks like. 200 ps samples nothing in particular; for a real
study, raise `production_steps`.

!!! note "The residue name in 0.5.3"
    `build-top` logs `residue name PAR (assigned deterministically)`, but the System and every
    output name the molecule `UNL`. In 0.5.3 the assigned name is recorded and not applied. It
    changes nothing about the simulation.

## Next

* the same molecule with enhanced sampling: [REST2: paracetamol](../REST2/paracetamol.md)
* another small molecule, same steps: [cMD: Chinolin](chinolin.md)
* register the finished directory as a dataset: [Registering a finished run](../../data_register/README.md)
