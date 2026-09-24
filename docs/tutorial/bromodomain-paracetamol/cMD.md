# cMD: a bromodomain with paracetamol, from a reused parameter package

**Tested against md-tools `0.5.4`.** Every command and every number on this page comes from a
run executed as written, at that version. It has not been re-run for 0.6.1.

!!! note "Requires the md-tools release after 0.5.4"
    This page uses `solute.kind: complex`, a `ligands:` entry selecting by `resname` with the
    package given as a path, `input.assembly`, `input.missing_atoms` and
    `protonation.method: propka`, none of which are in 0.5.4.

    The 10 ns run below was produced at commit `fa288ff`, whose configuration differed in two
    ways that have since changed: it deleted the crystallisation additives through a retired
    `input.remove` section, and it named the ligand by chain and residue id. Both commands on this
    page were re-executed on the current tree and build the same System **byte for byte** --
    `built.xml` sha256 `f5908732...4e9c0df0`, the System the production run integrated -- so every
    number here still stands.

A protein–ligand complex from a deposited crystal structure: the **CREBBP bromodomain with
paracetamol** ([PDB 4A9K](https://www.rcsb.org/structure/4A9K), 1.81 Å), 10 ns of production. The
paracetamol parameters are **not generated here**: the run loads the same parameter package a
ligand-only build made, unchanged, and keeps the deposited pose. Every command below was run
exactly as written and every number is copied from the files that run produced (one NVIDIA RTX
3080, CUDA, mixed precision).

Building took under a minute; minimisation, equilibration and 10 ns of production took
**34 min 18 s**.

Read [cMD: paracetamol](../paracetamol/cMD.md) and [cMD: barnase–barstar](../barnase-barstar/cMD.md) first: this
page explains only what a ligand adds.

## 1. The structure, and what is in it

```bash
mkdir -p 4A9K/build
cd 4A9K/build
curl -O https://files.rcsb.org/download/4A9K.cif      # sha256 e34a4ae9...4d223f8e
```

The asymmetric unit holds two copies of the complex, and the entry defines each as a biological
assembly:

| assembly | protein | paracetamol | also in the assembly |
|---|---|---|---|
| 1 | chain A, 115 residues | TYL A:2200 | 2 × ethylene glycol (EDO), 1 thiocyanate (SCN), 150 waters |
| 2 | chain B, 112 residues | TYL B:2198 | 1 K⁺, 117 waters |

This tutorial builds **assembly 1**. EDO and SCN are crystallisation additives, not part of the
biology, so strip them from the structure file before building:

```bash
grep -v -E "^HETATM .* (EDO|SCN) " 4A9K.cif > 4A9K-prepared.cif   # sha256 a796ec9f...3535eeab
```

md-tools does not delete residues for you: removing an additive is an edit to your own file, and a
non-standard residue that is neither mapped to a parameter package nor removed refuses the build by
name. Two consequences worth knowing:

* **The edit is on the deposited ids, before `input.assembly` expands anything.** Here that makes no
  difference -- assembly 1 is one copy of chain A -- but for an assembly with several copies of a
  chain, stripping a residue from the file removes it from every copy. That is usually what you want
  for an additive.
* **The build no longer records the removal, so you should.** Write down what you removed and why,
  as this page does: three residues, `EDO A:2198`, `EDO A:2199` and `SCN A:2201`, 11 atoms in all,
  none of them part of the biology.

Four residues have alternate conformations (the first is kept, and they are recorded), and four
residues lack heavy atoms: Lys1083, Ile1084, Gln1194 and the C-terminal Gly1197 (its OXT).

## 2. The build configuration

`build-top.config`:

```yaml
solute:
  kind: complex
ligands:
  - {resname: TYL, parameter: ../catalog/CHEMBL112/param_e932f4c4f371}
input:
  assembly: "1"
  missing_atoms: add
protonation:
  method: propka
  ph: 7.0
forcefield:
  protein: ff14SB
solvent:
  model: TIP3P
  padding_nm: 1.5
  ionic_strength_molar: 0.15
```

* `kind: complex` reads the structure as protein chains plus ligand instances. The protein takes
  ff14SB; each ligand instance takes the parameters of an existing package.
* `resname: TYL` selects the ligand by residue name, and `parameter` is a **path** to a package
  directory, relative to this configuration. Nothing has to be registered and no catalog has to be
  configured: the build reads that folder. To name a package in the shared catalog instead, write
  `parameters: CHEMBL112/param_e932f4c4f371` -- one or the other, never both.
* **A `resname` entry maps every residue of that name**, on the reasoning that copies of one
  compound take the same parameters. Assembly 1 has a single TYL, so this entry selects exactly the
  one ligand; had the page built both assemblies, the same entry would cover both copies. To name
  one copy instead, select it by `{chain, resid, insertion_code}`, which must match exactly one
  residue.
* A selector by chain names the **expanded** chain ids; `build/assembly.json` maps each back to its
  author chain and operator. (The additive strip above is the other way round: it edits the file
  before expansion, so it names the deposited ids.)
* **Where the package comes from.** This page reuses one that already exists. To make it yourself,
  see [cMD: paracetamol](../paracetamol/cMD.md), which parameterises the molecule from its SMILES string
  and writes exactly this package -- same parameter id, same parameters, measured.

## 3. Build

```bash
md-openmm build-top -i 4A9K-prepared.cif \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

```text
Command
  assembly                    1: 1 chain(s), 0 on-axis copy(ies) dropped
  missing atoms               12 atom(s) added to 4 residue(s) (input.missing_atoms: add)
Input interpretation
  interpreted as              protein-ligand complex (.cif), 1 ligand instance(s) mapped to parameter packages
  ligand TYL                  resname 'TYL' -> CHEMBL112/param_e932f4c4f371
  coordinates                 the deposited pose; ligand hydrogens from each package
  small-molecule FF           not run: parameters loaded from the packages
Preparation
  ligand       : TYL resname 'TYL' -> CHEMBL112/param_e932f4c4f371
  protonation  : pH 7.0, 9 -> 1265 hydrogens; 1 ligand instance(s) kept as packaged
  protonation  : method propka, PROPKA 3.5.1
  solvation    : 7200 waters, ions {'NA': 23, 'CL': 20}, box dodecahedron (255.4 nm^3)
Counts
  atoms                       24036
  solute atoms                1943
```

**"not run: parameters loaded from the packages"** is the point of this page. No charge is
generated: `build/ligands/CHEMBL112/param_e932f4c4f371/` is copied into the build, so the run is
self-contained, and `build/ligand_mapping.json` records the package digests, the deposited-atom to
package-atom map, the symmetry verdict and the resolved topology indices.

**The ligand's hydrogens come from the package, and its heavy atoms do not move.** The deposited
pose has no hydrogens; the package supplies them and only they are relaxed. The protein's hydrogens
are then added around it -- `1 ligand instance(s) kept as packaged` says the ligand was not touched,
and the build asserts that afterwards, atom by atom.

**Protonation.** PROPKA ran on the complex with its crystal waters. This bromodomain construct has
no histidine and no residue whose predicted pKa is within a pH unit of 7, so there is nothing to
flag; a histidine within 5 Å of the ligand or an ion would be printed with its distances. Had the
prediction suggested a different ligand protonation state, it would be reported as a warning and
nothing would change: a different chemical state is a different package, never the old parameters
with new hydrogens.

## 4. Generate and run

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

## 5. Result

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

## Where the package comes from

`CHEMBL112/param_e932f4c4f371` is the paracetamol package: one chemical state (explicit hydrogens,
formal charges, bond orders), its Sage 2.2.1 parameters and AM1-BCC charges, and the digests that
identify them. A ligand-only build makes it once; this build loads it. The parameter id is a digest
of the parameters themselves, so the same chemical state and the same method always produce the
same id, and a build that reuses it runs no charge generation at all.

A package is a folder, and `parameter:` above points straight at it -- nothing needs to be
registered for this build to run. Registering it in the shared catalog under
`$MD_DATA/parameters/ligands` is what makes it reusable by NAME from anywhere
(`parameters: CHEMBL112/param_e932f4c4f371`, with no path); see
[Ligand parameter packages](../../basics/ligand-packages.md).

## Next

* the protein–protein counterpart, with no ligand: [cMD: barnase–barstar](../barnase-barstar/cMD.md)
* register the finished directory as a dataset: [Registering a finished run](../../basics/data-register/index.md)
