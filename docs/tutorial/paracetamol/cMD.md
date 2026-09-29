# cMD: paracetamol in explicit water, parameterised once

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a
run executed as written, at that version.

A complete conventional MD run of one small molecule, from a SMILES string to 200 ps of NPT
production — and, on the way, **the parameter package every other paracetamol tutorial reuses**.
This is the page to read first if you want to know where ligand parameters come from.

It takes about a minute: the AM1-BCC charge calculation is most of it, and the 200 ps run is 13.5 s.

## What you need

* md-tools installed, with CUDA — see [Installing](../../install.md). CUDA is the default and is
  mandatory; nothing here falls back to the CPU.
* a machine configuration — see [Machine configuration](../../structure/machine-config.md).

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
  aliases: [paracetamol, acetaminophen]
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
`aliases` are the searchable names — they are **names, not identities**, so they do not enter the
parameter id, and a package written with different aliases is the same package.

!!! note "Set the aliases before you register"
    A package's aliases are fixed when it is written. `data-register --ligand-package` is
    write-once, so an aliased copy of a package already in the catalog is kept out, and a build
    that finds the package by search reuses it with the names it already has. Both say so on
    stderr (`NOTE: ... were NOT added` / `were NOT applied`), and the build records the dropped
    names in built.log as `stated_aliases_not_applied`. `data-register --find-ligand` also
    searches the compound id, the residue name and the canonical SMILES, so a package without
    aliases can still be found as `CHEMBL112` or `TYL`, but not as "paracetamol". To change the
    aliases of a registered package, remove its catalog entry first.

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
the REST2 ladder, a protein–ligand complex — reads those same numbers.

Everything not stated is the documented default (Sage 2.2.1, TIP3P, a 1.5 nm dodecahedron, 0.15 M
NaCl, HBonds constraints), and the build log lists each with `(default)` beside it. See
[Building a system](../../basics/build-top/index.md).

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
[The run layout](../../basics/run-layout.md).

## 5. Run it

```bash
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./run.sh
```

Without `PCI_BUS_ID`, CUDA may number the cards differently from `nvidia-smi`.

`run.sh` is five `md-openmm md-run` calls, in order:

| stage | what | ensemble | elapsed |
|---|---|---|---|
| `min` | energy minimisation, 1000 iterations | — | 0.9 s |
| `eq_1` | solute heavy atoms restrained | NVT | 1.4 s |
| `eq_2` | solute heavy atoms restrained | NPT | 1.5 s |
| `eq_3` | restraint released | NPT | 1.4 s |
| `cMD` | production, 200 ps | NPT | 13.5 s |

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
  Temperature (K)              mean 298.784   rms fluctuation 7.29157
  Density (g/mL)               mean 0.99488    rms fluctuation 0.013941
  Speed (ns/day)               mean 1238      rms fluctuation 295.476
Summary
  cMD: 100000 steps completed, 200 ps
status               completed
```

The temperature sits at the 300 K thermostat and the density at that of water, which is what a
correctly built and equilibrated box looks like. The molecule is named **TYL** throughout, because
`--resname` is applied to the package and to every build that reads it. 200 ps samples nothing in
particular; for a real study, raise `production_steps`.

## 7. What 200 ps sampled, and a coordinate that is not what it looks like

The ring's rotation about the **1-3-4-5** torsion — atoms C2–N1–C3–C4, in the numbering
`scaler.yaml` and the unscaled-torsion picture use — is the slow coordinate of this molecule, and
it comes with a trap.

**The ring is para-substituted, so turning it 180° maps the molecule onto itself.** Two
orientations separated by 180° are not two states; they are the same state, reached by relabelling
two equivalent carbons. That has three consequences, and a tutorial that ignores them reports
numbers that look fine and mean nothing:

* The raw distribution shows **two copies of every feature**, so its spread measures the gap
  between the copies rather than the width of the basin.
* An arithmetic mean of the raw angle returns the point midway between the copies, which is the
  **barrier**, not the preferred geometry.
* A "transition" from one copy to the other changes no physical property at all.

The fold is a property of the molecular graph, not of any one conformation: it is the number of
symmetry operations that fix the first three atoms of the torsion and carry the fourth onto another
atom bonded to the third. For this ring that orbit is {C4, C8}, the two ortho carbons, so the fold
is **2**. The same test gives **3** for a methyl rotation and **1** for the amide ω and the
hydroxyl O–H, which have no such degeneracy.

Folding the angle into its unique range, −90° to +90°, gives the coordinate that means something:

| | raw spread | **folded spread** |
|---|---|---|
| 200 ps cMD — this page | 157.1° | — *(one orientation only; see below)* |
| 10 ns cMD | 137.2° | **25.1°** |
| 10 ns REST2, state 0 | 116.1° | **25.5°** |
| **1 µs cMD — the reference** | 120.6° | **25.6°** |

The 1 µs run is here only as a reference: it is not part of the workflow this page teaches, and
nothing below asks anyone to run one. Its job is to say what the answer IS, so the short runs can
be marked against it rather than against each other. On the folded coordinate **10 ns of plain cMD
is already within 0.5° of the microsecond**, and the ladder within 0.1°. The physical width of this
torsion is not what either method struggles with.

**In 200 ps the ring never rotated once.** The run has no information about this coordinate at
all, and the histogram it produces is a picture of one half of a symmetric distribution. Nothing in
the run says so: the temperature, the density and the completion record are all exactly as they
should be. This is a correct 200 ps simulation, and 200 ps is simply not long enough for this
rotation.

!!! note "Symmetry gives a free convergence test, and it is not a test of accuracy"
    Because the two orientations are equivalent, a fully converged run must spend **exactly 50%**
    of its time on each side. Measuring that costs nothing and needs no reference: 200 ps gives
    0.00%, 10 ns gives 26.5%, [the ladder](REST2.md) gives 48.5% in the same 10 ns.

    **The 1 µs reference gives 44.2%, not 50%** — and that is the useful part. It crossed the
    barrier 165 times in a microsecond, and an occupancy estimated from N crossings carries an
    uncertainty of roughly 1/sqrt(N), here about 8%. So 44.2% is consistent with the exact answer
    and pins it no more tightly than that. **This test converges with the number of crossings, not
    with simulation length**: a thousand times the sampling bought about 80 times the crossings,
    because the rate is what it is. Quoting "1 µs" says nothing about how well this coordinate was
    sampled; quoting 165 crossings says everything.

    It is a test of **ergodicity over the rotation**, not of any physical quantity. On the folded
    coordinate — the one that carries physics — the 10 ns run and the ladder agree to half a
    degree. A run can be a long way from 50/50 and still give the right answer for every
    observable, because the states it failed to interconvert are indistinguishable.

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

* the same molecule with enhanced sampling: [REST2: paracetamol](../paracetamol/REST2.md)
* the same parameters in a protein pocket:
  [the TYK2 complex](../tyk2-ejm31/index.md)
* what a package is, in full: [Ligand parameter packages](../../basics/ligand-packages.md)
* register the finished directory as a dataset:
  [Registering a finished run](../../basics/data-register/index.md)
