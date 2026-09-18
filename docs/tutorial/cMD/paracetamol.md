# cMD: paracetamol in explicit water, parameterised once

!!! note "Requires the md-tools release after 0.5.4"
    This page uses `build-top --parameterize` and `solute.parameters` naming a package **path**,
    neither of which is in 0.5.4. It was run with md-tools at commit `e9a89db`, on one NVIDIA
    RTX A5000 (CUDA, mixed precision). Every command below was run exactly as written and every
    number is copied from the files that run produced.

A complete conventional MD run of one small molecule, from a SMILES string to 200 ps of NPT
production — and, on the way, **the parameter package every other paracetamol tutorial reuses**.
This is the page to read first if you want to know where ligand parameters come from.

It takes about a minute: the AM1-BCC charge calculation is most of it, and the 200 ps run is 14 s.

## What you need

* md-tools installed, with CUDA — see [Installing](../../install.md). CUDA is the default and is
  mandatory; nothing here falls back to the CPU.
* a machine configuration — see [Machine configuration](../../machine-configuration.md).

## 1. The dataset root

One directory holds everything about this system. `build/` is the system itself; every run of it
sits beside `build/` and shares it.

```bash
mkdir -p PARA/build
cd PARA/build
```

The molecule, as a one-record SMILES file (`TYL.smi`):

```text
CC(=O)Nc1ccc(O)cc1 paracetamol
```

`TYL` is paracetamol's chemical component id in the PDB, which is what makes it a good residue
name: it is unique, and a structure downloaded from the RCSB already uses it.

## 2. Parameterise the molecule

This is the step that costs something, and the one you do **once per molecule**.

`parameterize.config`:

```yaml
solute:
  kind: ligand
  compound_id: CHEMBL112
```

```bash
md-openmm build-top --parameterize -i TYL.smi --resname TYL \
    --config parameterize.config \
    -op parameter/TYL.pdb -os parameter/TYL.xml -log parameterize.log
```

```text
Command
  input                       TYL.smi
  package directory           parameter
  residue name                TYL
Resolved configuration
  solute.ligand_forcefield    sage-2.2.1   (default)
  solute.ligand_charge_method am1bcc   (default)
  solute.compound_id          CHEMBL112   (set)
  coordinates                 embedded from the SMILES CC(=O)Nc1ccc(O)cc1 (ETKDGv3 seed 20260814,
                              2 conformers, MMFF94s-minimised, lowest kept)
Parameters
  package                     CHEMBL112/param_e932f4c4f371   (created)
  charges                     am1bcc / am1bcc by ambertools-sqm
  force field                 openff-2.2.1
Outputs
  TYL.pdb  TYL.sdf  TYL.xml  metadata.json  molecule.sdf  parameter.config  parameters.ffxml
  re-read check               CHEMBL112/param_e932f4c4f371 verifies in place  OK
```

`-op` and `-os` must name files in **one** directory, and that directory is the package:

```text
build/parameter/
    molecule.sdf       the exact chemical state -- explicit hydrogens, formal charges, bond orders
    parameters.ffxml   the parameters themselves
    metadata.json      identities, charges, provenance, digests
    parameter.config   what a later build must match to REUSE these parameters
    TYL.sdf  TYL.pdb  TYL.xml     readable copies, named for --resname
```

The first four files are the package; the three named for `--resname` are the copies a person, a
tutorial and the next command line point at. **The directory moves as a unit** — the metadata
declares the readable copies, and a load verifies they are there.

`compound_id: CHEMBL112` is the substance. It is not required (a compound no database lists gets
`LOCAL-<InChIKey block>`), but giving it makes the package findable by a name other people use.

Add `--register` to copy the finished package into the shared catalog under
`$MD_DATA/parameters/ligands/`. That is optional throughout: every build below reads the folder
directly. Registering is what lets another dataset name the package **without a path**, as
`parameters: CHEMBL112/param_e932f4c4f371`.

!!! note "The parameter id is a digest, and it does not depend on the conformer"
    `param_e932f4c4f371` is a digest of the chemical state and of the parameters, so the same
    molecule parameterised the same way lands on the same id. Parameterising this SMILES twice,
    parameterising an SDF instead, and importing the parameters out of an existing 0.5.3 System all
    produce `CHEMBL112/param_e932f4c4f371`, with identical `parameters.ffxml` and charges differing
    by 0.0 e — even though the embedded conformers differ, because coordinates are not part of the
    identity.

    That is a **measured fact about paracetamol**, not a promise about every molecule. AM1-BCC is
    computed on one conformer; for a flexible molecule two conformers can give different charges,
    and then the parameter id differs too. That is the system working — different numbers, different
    package — and it is the reason to reuse a package rather than regenerate one: a reused package
    cannot drift.

## 3. Build the system

Now the box, from the package you just made. `build-top.config`:

```yaml
solute:
  kind: ligand
  parameters: ./parameter
```

```bash
md-openmm build-top -i parameter/TYL.sdf \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

`parameters` is a **path**, relative to this configuration, so nothing has to be registered and no
catalog has to exist. Give `CHEMBL112/param_e932f4c4f371` instead to take it from the catalog by
name; `search` (the default) looks for a package whose recorded criteria match this build, and
`generate` parameterises the molecule whatever the catalog holds.

```text
Input interpretation
  interpreted as              single-molecule SDF
  coordinates                 supplied by the SDF, used as given (no embedding, no minimisation)
  solvent treatment           explicit TIP3P, periodic
Preparation
  parameters   : CHEMBL112/param_e932f4c4f371 (reused (stated reference))
  solvation    : 592 waters, ions {'NA': 2, 'CL': 2}, box dodecahedron (19.1 nm^3)
Validation
  particle agreement          1800 atoms == 1800 particles  OK
  HMR                         NOT applied; masses are the force field's own, recommend 2.0 fs
Summary
  built 1800 particles, 597 residues, explicit TIP3P
  status: completed
```

**`reused (stated reference)` is the point of this page.** No charge is generated here: the step
that cost a minute happened once, in section 2, and every later build of this molecule — this box,
the REST2 ladder, the bromodomain complex — reads those same numbers.

Everything not stated is the documented default (Sage 2.2.1, TIP3P, a 1.5 nm dodecahedron, 0.15 M
NaCl, HBonds constraints), and the build log lists each with `(default)` beside it. See
[Building a system](../../build-top.md).

## 4. Generate the run

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
├── build/        parameter/  built.xml  built.pdb  built.log  ...
├── input/        min.in  eq_1.in  eq_2.in  eq_3.in  cMD.in     <- shared by every run here
├── min/          min.py                                          <- one minimised structure, shared
└── cMD-run1/     run.sh  cMD.py  resolved.config  run.config  build-md.log  eq/
```

`input/` and `min/` belong to the system, not to this run. See
[The run layout](../../run-layout.md).

## 5. Run it

```bash
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./run.sh
```

Without `PCI_BUS_ID`, CUDA may number the cards differently from `nvidia-smi`.

`run.sh` is five `md-openmm md-run` calls, in order:

| stage | what | ensemble | elapsed |
|---|---|---|---|
| `min` | energy minimisation, 1000 iterations | — | 1.0 s |
| `eq_1` | solute heavy atoms restrained | NVT | 1.5 s |
| `eq_2` | solute heavy atoms restrained | NPT | 1.7 s |
| `eq_3` | restraint released | NPT | 1.6 s |
| `cMD` | production, 200 ps | NPT | 14.0 s |

```text
run.sh: all stages reported completion
```

Run it again and every stage is skipped, because each already reports `status: completed` in its
own machine record; an interrupted stage resumes from its last checkpoint.

## 6. Result

From `cMD-run1/cMD.out`:

```text
System
  atoms                        1800
  residues                     597 (HOH 592, CL 2, NA 2, TYL 1)
  net charge                   +0.000 e
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
  over                         20 report(s) in mdout.csv
  Temperature (K)              mean 297.375   rms fluctuation 5.58951
  Density (g/mL)               mean 0.994745   rms fluctuation 0.0120426
  Speed (ns/day)               mean 1183.6    rms fluctuation 283.627
Summary
  cMD: 100000 steps completed, 200 ps
status               completed
```

The temperature sits at the 300 K thermostat and the density at that of water, which is what a
correctly built and equilibrated box looks like. The molecule is named **TYL** throughout, because
`--resname` is applied to the package and to every build that reads it. 200 ps samples nothing in
particular; for a real study, raise `production_steps`.

## What it wrote

```text
PARA/
├── build/
│   ├── TYL.smi  parameterize.config  parameterize.log
│   ├── parameter/     the package: seven files, 80 KB, the only expensive thing here
│   ├── build-top.config  built.xml  built.pdb  built.solute.pdb  built.log
│   └── ligands/       a copy of the package used, so the build is self-contained
├── input/  min/
└── cMD-run1/  cMD.out  cMD.log  cMD.xml  solute_prod1.nc  mdout.csv  energy_components.csv  eq/
```

## Next

* the same molecule with enhanced sampling: [REST2: paracetamol](../REST2/paracetamol.md)
* the same parameters in a protein pocket:
  [cMD: a bromodomain with paracetamol](bromodomain-paracetamol.md)
* what a package is, in full: [Ligand parameter packages](../../ligand-packages.md)
* register the finished directory as a dataset:
  [Registering a finished run](../../data_register/README.md)
