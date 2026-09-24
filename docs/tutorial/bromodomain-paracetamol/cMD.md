# cMD: a bromodomain with paracetamol, from a reused parameter package

**Tested against md-tools `0.5.4`.** Every command and every number on this page comes from a
run executed as written, at that version. It has not been re-run for 0.6.1.

10 ns of ordinary MD on the [bromodomain–paracetamol complex](index.md), on one RTX 3080.
**Production: 34 min 18 s.**

**Starts from a built system.** Do [the system page](index.md) first: it downloads 4A9K, strips
the crystallisation additives, finds or builds the paracetamol package, and writes
`4A9K/build/built.xml`, `built.pdb` and `built.log`.

## 1. Generate and run

`cMD.config` is the barnase–barstar one: 5000 minimisation iterations, three 100 ps equilibration
stages, 10 ns of production at 2 fs, solute frames every 10 ps.

```bash
cd ..                                            # 4A9K/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 ./run.sh
```

```text
run.sh: all stages reported completion
```

## 2. Result

From `cMD-run1/cMD.out`:

```text
Averages
  over                         1000 report(s) in mdout.csv
  Temperature (K)              mean 300.199   rms fluctuation 1.92983
  Density (g/mL)               mean 1.01856   rms fluctuation 0.00308122
  Speed (ns/day)               mean 438.282   rms fluctuation 14.9757
```

Protein and ligand over the run, from
[`bromodomain_paracetamol_analysis.py`](bromodomain_paracetamol_analysis.py) (run it from `4A9K/`):

```text
frames 1000, time 310-10300 ps
protein CA RMSD (A): mean 1.21, last 1 ns 1.49, max 2.00
paracetamol heavy-atom RMSD after protein alignment (A): mean 1.34, last 1 ns 1.51, max 3.73
protein heavy atoms within 4 A of the ligand: start 22, mean 16, last 1 ns 18
residues in contact for more than half the run: ASN1168 (98%), VAL1174 (79%), PRO1110 (57%)
```

The ligand stays in the pocket for the whole run, about 1.5 Å from its crystallographic pose once
the protein is aligned, and its most persistent contact is **Asn1168** -- the conserved bromodomain
asparagine that recognises an acetyl-lysine, here contacting paracetamol's acetamide. That is the
expected binding mode, which is what makes it a check on the preparation rather than a result.

!!! warning "Image the trajectory before measuring protein–ligand distances"
    The solute trajectory holds the ligand as the periodic simulation holds it, so it can sit in a
    different periodic image from the protein. The analysis script makes molecules whole and puts
    the ligand in the image nearest the protein first (`mdtraj.Trajectory.image_molecules`); without
    that step the ligand RMSD and the contact counts are meaningless.

**What 10 ns shows, and what it does not.** The prepared complex -- assembly 1, the stripped
additives, the built side chains, the PROPKA protonation, the reused ligand parameters -- is stable
in explicit water, and the ligand keeps its crystallographic binding mode. It is not a binding
affinity, not a residence time, and not evidence that the pose is the global minimum.

## What it wrote

```text
4A9K/
├── catalog/    CHEMBL112/param_e932f4c4f371/    the package this build reuses (see below)
├── build/      4A9K.cif  4A9K-prepared.cif  build-top.config  built.xml  built.pdb  built.solute.pdb  built.log
│               built.prepared.pdb   the expanded and completed structure the build read
│               assembly.json        the expanded chain and its operator
│               ligand_mapping.json  the instance, its package and its atom map
│               ligands/             a copy of every package used, so the build is self-contained
├── input/  min/
└── cMD-run1/   cMD.out  cMD.log  cMD.xml  solute_prod1.nc  mdout.csv  eq/  ...
```

## Next

* the protein–protein counterpart, with no ligand: [cMD: barnase–barstar](../barnase-barstar/cMD.md)
* register the finished directory as a dataset: [Registering a finished run](../../basics/data-register/index.md)
