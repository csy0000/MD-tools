# Bromodomain with paracetamol

**Tested against md-tools `0.5.4`.** A protein–ligand complex: a bromodomain from PDB 4A9K with
paracetamol in its acetyl-lysine site.

![The bromodomain with paracetamol bound](images/bromodomain.png)

*4A9K, assembly 1. The ligand is orange; the residues within 5 Å of it are drawn as a translucent
surface. Crystallographic waters, ethylene glycol and ions are removed by the build.*

## Why this system

It is the smallest honest protein–ligand complex in this set — a real binding site, a real ligand,
and small enough to build and run on one card. It is also where a ligand parameter package made for
one purpose is REUSED in another: the same paracetamol package that the ligand-only tutorials
produce is the one this complex consumes.

## Is the ligand package already there?

Paracetamol may already be parameterised — by the [ligand-only tutorial](../paracetamol/index.md),
by a sibling project, or in the machine catalog. Parameterising again costs a minute and, for a
flexible molecule, risks landing on different charges, so look first.

```bash
# the machine catalog, if $MD_DATA is configured
ls $MD_DATA/parameters/ligands/                       # compound ids
ls $MD_DATA/parameters/ligands/CHEMBL112/             # its parameter ids

# a sibling project's build keeps its own copy
ls ../*/build/ligands/*/                              # <compound-id>/<parameter-id>/
```

A package is a directory of four files — `molecule.sdf`, `parameters.ffxml`, `metadata.json`,
`parameter.config` — so finding one is finding that directory.

Three ways to use what you find, and one to make what you do not:

| in `build-top.config` | |
|---|---|
| `parameter: ../catalog/CHEMBL112/param_e932f4c4f371` | a **path**, relative to the configuration. Nothing needs to be registered |
| `parameters: CHEMBL112/param_e932f4c4f371` | that exact package from the catalog |
| `parameters: search` | let the build find one whose recorded criteria match — topology, protonation, charge method **and its implementation**, and the force field. A near match is a difference |
| `parameters: generate` | parameterise regardless of what exists |

`built.log` records which route ran — `reused (stated reference)`, `reused (catalog search)` or
`created` — so reuse is a claim the build record supports or refutes.

If nothing matches, make one: [parameterising a ligand](../../basics/build-top/parameterization.md),
or the worked example in [paracetamol / cMD](../paracetamol/cMD.md), which writes exactly the
package this page reuses — same parameter id, same parameters, measured.

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
  see [paracetamol / cMD](../paracetamol/cMD.md), which parameterises the molecule from its SMILES string
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

## Where to go next

| method | what it is for | |
|---|---|---|
| [**cMD**](cMD.md) | the complex simulated unbiased, with the ligand's package reused | published |
| **REST2** | heating the ligand, or the ligand and its pocket | see [TYK2](../tyk2-ejm31/index.md) for this on a kinase |
| **Umbrella sampling** | a potential of mean force along the ligand–protein centre-of-mass distance | planned, 0.6.2 |
| **Alchemical (TI, FEP)** | absolute and relative binding free energies | planned, 0.7.0 |

*Umbrella sampling arrives in 0.6.2 and the alchemical pages in 0.7.0. They are listed here so the
shape of the set is visible; neither is written yet, and neither is linked to a page that does not
exist.*
