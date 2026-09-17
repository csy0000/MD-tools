# cMD: chignolin in explicit water

!!! note "Requires md-tools 0.5.4 or later"
    `build-md` validates the whole stage chain when it generates it, so the built System has to
    exist first. Earlier releases generated a run in an empty directory.

Ordinary molecular dynamics on **chignolin**, the designed ten-residue miniprotein
([PDB 1UAO](https://www.rcsb.org/structure/1UAO), sequence GYDPETGTWG), from the deposited NMR
structure to 1 ns of production. Every command below was run exactly as written and every number is
copied from the files that run produced. md-tools **0.5.4** at commit `afb537f`, on one NVIDIA
RTX 3080 with CUDA and mixed precision.

The whole thing took **1 min 3 s**.

This is the peptide counterpart of [cMD: paracetamol](paracetamol.md), which explains each step in
more detail. The differences are that a peptide is built from a structure rather than a SMILES
string, and that hydrogen mass repartitioning lets it run at 4 fs.

## What you need

```bash
md-openmm --version          # md-tools 0.5.4
```

## 1. The dataset root and the structure

```bash
mkdir -p CHI/build
cd CHI/build
```

1UAO is a solution-NMR entry with **18 models**, and a build needs one:

```bash
curl -O https://files.rcsb.org/download/1UAO.pdb
awk '/^MODEL/{m++} m==1{print} /^ENDMDL/{if(m==1) exit}' 1UAO.pdb \
    | grep -E '^(ATOM|TER)' > chignolin.pdb
echo END >> chignolin.pdb
```

That leaves 138 atoms: the ten residues with their hydrogens, free termini, nothing else.

`build-top` also takes the sequence instead of the structure, as a `.seq`. That builds an
**extended** conformation rather than the folded one, which is a different starting point and a
much larger box — see
[the note on the REST2 page](../REST2/chignolin.md#1-the-dataset-root-and-the-structure).

## 2. Build the system

`build-top.config`:

```yaml
solute:
  kind: peptide
solvent:
  model: TIP3P
  padding_nm: 1.5
hydrogen_mass_repartitioning:
  enabled: true
```

```bash
md-openmm build-top -i chignolin.pdb -os built.xml -op built.pdb \
    -log built.log --config build-top.config
```

From `built.log`:

```text
  interpreted as              peptide/protein PDB
  small-molecule FF           not used (peptide-only input)
  solvent treatment           explicit TIP3P, periodic

Counts
------
  atoms                       2553
  residues                    819
  solute atoms                138
  waters                      803
  ions                        {'NA': 4, 'CL': 2}

  HMR                         applied, target 3.024 amu, recommend 4.0 fs
```

Four Na⁺ against two Cl⁻ because chignolin carries −2, from Asp3 and Glu5. The box is a rhombic
dodecahedron of 28.7 nm³.

## 3. Generate the run

`cMD.config` at the dataset root:

```yaml
protocol: cMD
solvent: explicit

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 25000    # 100 ps
  restrained_npt_steps: 25000    # 100 ps
  unrestrained_npt_steps: 50000  # 200 ps
  production_steps: 250000       # 1 ns at 4 fs

reporting:
  crd_printout_solute: 500       # a solute frame every 2 ps
  info_printout: 2500
  checkpoint_printout: 25000
```

Every length is an integer step count; the picoseconds in the comments are what they work out to at
the timestep the System's masses allow. `timestep_fs` is left at `auto`, and because HMR was applied
the run resolves it to **4.0 fs** — read from the masses in `built.xml`, not from this file.

```bash
cd ..                                            # CHI/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
```

`build-md` checks the whole chain against the built System before writing anything: the masses
behind the timestep, the box behind each ensemble, and the paths every stage writes.

## 4. Run it

```bash
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./run.sh
```

`run.sh` is five `md-openmm md-run` calls, in order:

```text
md-openmm md-run -i ../input/min.in
md-openmm md-run -i ../input/eq_1.in
md-openmm md-run -i ../input/eq_2.in
md-openmm md-run -i ../input/eq_3.in
md-openmm md-run -i ../input/cMD.in
```

```text
== min ==     min: 0 steps completed, 0 ps
== eq_1 ==    eq_nvt_posres: 25000 steps completed, 100 ps
== eq_2 ==    eq_npt_posres: 25000 steps completed, 100 ps
== eq_3 ==    eq_npt_free:   50000 steps completed, 200 ps
== cMD ==     cMD:          250000 steps completed, 1000 ps
```

The inputs are **shared**: they live at `CHI/input/`, not inside the run, because an input says what
a method was asked to do and that belongs to the system rather than to one repeat. `min/` is shared
too — every run on this system minimises to the same structure.

## 5. What it wrote

```text
CHI/
├── build/          built.xml, built.pdb, built.log
├── input/          min.in, eq_1.in, eq_2.in, eq_3.in, cMD.in
├── min/            min.xml, min.log, min.out, min.py
└── cMD-run1/
    ├── eq/                    eq_1.xml … eq_3.xml and their logs
    ├── solute_prod1.nc        solute trajectory, AMBER NetCDF, 500 frames (855 kB)
    ├── mdout.csv              the state table, 100 rows
    ├── energy_components.csv  the potential, term by term
    ├── cMD.xml                the final state
    ├── cMD.checkpoints/       the committed generations a resume would continue from
    └── cMD.log                the machine-readable record
```

From `cMD.out`:

```text
Run
------------------------------------------------------------------------------
  timestep                     4.0 fs
  steps                        250000 (1000 ps)
  already done                 0
  ensemble                     NPT
  platform                     CUDA

  Speed (ns/day)               mean 2349.4   rms fluctuation 251.236
  elapsed                      36.6 s
```

**The step counter is absolute.** `mdout.csv` opens at step 102500, 410 ps, not at zero: minimisation
and the three equilibration stages came to 100000 steps before production began, and the counter
carries across the chain rather than restarting. That is what makes a resumed run's numbers line up
with an uninterrupted one's.

1 ns is a demonstration, not a study. Chignolin folds and unfolds on the microsecond scale, so for
anything real raise `production_steps` — or use the ladder, which is what
[REST2: chignolin](../REST2/chignolin.md) is for.

## Next

* the same peptide across six scaled states: [REST2: chignolin](../REST2/chignolin.md)
* the same steps for a small molecule, explained in more detail: [cMD: paracetamol](paracetamol.md)
* what every configuration key means: [Configuration](../../md-configuration.md)
