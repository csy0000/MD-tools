# cMD: barnase–barstar from a deposited crystal structure

!!! note "Requires the md-tools release after 0.5.4"
    This page uses `input.assembly`, `input.missing_atoms` and `protonation.method: propka`,
    which are not in 0.5.4. It was run with md-tools at commit `cbc617a` (as `52499b0`, before a
    rebase that changed only the build-log text of protein-ligand builds, which this page does not
    use).

Ordinary molecular dynamics on **barnase–barstar**
([PDB 1BRS](https://www.rcsb.org/structure/1BRS)), the ribonuclease and its inhibitor, from the
deposited crystal structure to 10 ns of production. This is the first protein–protein tutorial:
two chains, an interface, crystal waters, residues with missing side-chain atoms, and PROPKA3
protonation. Every command below was run exactly as written and every number is copied from the
files that run produced (one NVIDIA RTX 3080, CUDA, mixed precision).

Building took **16 s**; minimisation, equilibration and 10 ns of production took **40 min**.

Read [cMD: paracetamol](paracetamol.md) first; this page explains only what is new.

## 1. The structure, and the choices it needs

```bash
mkdir -p 1BRS/build
cd 1BRS/build
curl -O https://files.rcsb.org/download/1BRS.cif      # sha256 1c98dcc3...5704f153
```

The asymmetric unit holds **three** barnase–barstar complexes, and the entry defines each as a
biological assembly: 1 is chains A + D, 2 is B + E, 3 is C + F. They are not interchangeable:

| assembly | barnase | barstar | as deposited |
|---|---|---|---|
| 1 | A | D | barstar residues 64–65 **missing inside the chain** |
| 2 | B | E | barstar residues 64–65 **missing inside the chain** |
| 3 | C | F | no gap; **44 side-chain atoms missing** over 15 residues |

md-tools refuses a gap inside a chain -- building one is loop modelling, which it does not invent --
so this tutorial builds **assembly 3**. Its missing side-chain atoms are built with PDBFixer and
every one of them is recorded; they carry no crystallographic evidence.

The barstar in 1BRS is an engineered double mutant, C40A/C82A (`SEQADV` records), so it has no
cysteine. Barnase residues 1–2 and the C-terminal residues missing from the deposit are not built:
each chain starts and ends at its first and last observed residue.

## 2. The build configuration

`build-top.config`:

```yaml
# 1BRS barnase-barstar: biological assembly 3 (author chains C = barnase, F = barstar), the only
# pair without a chain break; its 44 missing side-chain atoms are built and recorded.
solute:
  kind: peptide
input:
  assembly: "3"
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

* `input.assembly: "3"` expands that biological assembly from the mmCIF before anything else reads
  it. Each copy of a chain gets its own chain id -- here **A is barnase (author chain C)** and
  **B is barstar (author chain F)** -- and `build/assembly.json` maps each back to its author chain,
  label_asym ids and symmetry operator.
* `input.missing_atoms: add` builds the missing heavy atoms. The default, `refuse`, stops the build
  and lists them.
* `protonation.method: propka` predicts pKa values with PROPKA 3.5.1 on the prepared structure and
  assigns residue variants by one stated rule (see [Building a system](../../build-top.md)).
  The default, `openmm`, is what earlier releases did.
* The 90 crystal waters of assembly 3 are kept, and placed after the protein in the topology.

## 3. Build

```bash
md-openmm build-top -i 1BRS.cif \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

What the log reports:

```text
Command
  input                       1BRS.cif
  assembly                    3: 2 chain(s), 0 on-axis copy(ies) dropped
  missing atoms               44 atom(s) added to 15 residue(s) (input.missing_atoms: add)
Preparation
  protonation  : pH 7.0, 0 -> 1730 hydrogens
  protonation  : method propka, PROPKA 3.5.1
    A:18 HIS pKa 6.04 -> HID (propka (neutral) + openmm hydrogen-bond heuristic for HID/HIE) [near_ph]
    A:73 GLU pKa 6.69 -> GLU (propka) [near_ph]
  solvation    : 8757 waters, ions {'NA': 28, 'CL': 24}, box dodecahedron (315.5 nm^3)
Counts
  atoms                       29725
  solute atoms                3132
  waters                      8847
  ions                        {'NA': 28, 'CL': 24}
```

**Reading the protonation.** PROPKA predicts every titratable group; md-tools prints the ones worth
a second look. Barnase His18 (pKa 6.04) and Glu73 (6.69) are within one pH unit of 7, so their
assigned states -- neutral His, deprotonated Glu -- should be read as uncertain. The other two
histidines, barnase His102 (4.83) and barstar His17 (4.24), are predicted neutral too. PROPKA does
not decide between the neutral tautomers HID and HIE; OpenMM's hydrogen-bond heuristic chose HID for
all three. If a tautomer matters to your question, set it explicitly:

```yaml
protonation:
  method: propka
  overrides:
    - select: {chain: A, resid: "102"}
      variant: HIE
```

Every prediction, assignment, override and warning is in the machine record at the end of
`built.log` under `protonation`, and PROPKA's own report is kept.

The added side-chain atoms are listed under `structure_completion`: barnase Lys19, Asp22, Glu29,
Gln31, Lys39, Val45, Lys49, Ser67 and Arg110, and barstar Lys22, Glu28, Glu46, Glu64, Asn65 and
Ser89 -- surface residues whose side chains were disordered in the crystal.

## 4. Generate and run

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

## 5. Result

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
  [cMD: a bromodomain with paracetamol (4A9K)](bromodomain-paracetamol.md)
* register the finished directory as a dataset: [Registering a finished run](../../data_register/README.md)
