# cMD: Chinolin (quinoline) in explicit water

The same workflow as [cMD: paracetamol](paracetamol.md), for a second small molecule: Chinolin,
C₉H₇N — quinoline in English. It is here to show that nothing in the steps is specific to one
molecule: only the SMILES line changes. Run with md-tools **0.5.3**; every number is copied from
that run (one NVIDIA RTX 3080, CUDA, mixed precision).

If you have not done the paracetamol tutorial, read it first; this page does not repeat its
explanations.

## 1. Inputs

```bash
mkdir -p CHINOLIN/build
cd CHINOLIN/build
```

`chinolin.smi`:

```text
c1ccc2ncccc2c1 chinolin
```

`build-top.config`:

```yaml
solute:
  kind: ligand
```

## 2. Build

```bash
md-openmm build-top -i chinolin.smi \
    -os built.xml -op built.pdb -log built.log --config build-top.config
```

```text
Input interpretation
  interpreted as              single-molecule SMILES
  smiles                      c1ccc2ncccc2c1
  small-molecule FF           sage-2.2.1
  solvent treatment           explicit TIP3P, periodic
Preparation
  protonation  : pH 7.0, 7 -> 7 hydrogens
Counts
  atoms                       1806
  solute atoms                17
  waters                      595
  ions                        {'NA': 2, 'CL': 2}
Summary
  status: completed
```

`7 -> 7 hydrogens`: the ring nitrogen is left unprotonated at pH 7 (the conjugate acid's pKa is
about 4.9), so the molecule is neutral — 9 carbons, 1 nitrogen and 7 hydrogens, 17 atoms.

## 3. Generate and run

`cMD.config` at the dataset root is identical to the paracetamol one:

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

```bash
cd ..                                            # CHINOLIN/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
cd cMD-run1
./run.sh
```

```text
== min ==
== eq_1 ==
== eq_2 ==
== eq_3 ==
== cMD ==
...
  cMD: 100000 steps completed, 200 ps
  status: completed
run.sh: all stages reported completion
```

The whole chain took 36 s.

## 4. Result

From `cMD-run1/cMD.out`:

```text
System
  residues                     600 (HOH 595, CL 2, NA 2, UNL 1)
Averages
  Temperature (K)              mean 298.616   rms fluctuation 5.13951
  Density (g/mL)               mean 0.99185   rms fluctuation 0.00689213
  Speed (ns/day)               mean 1245.2   rms fluctuation 312.025
```

The outputs have the same names and meanings as in the paracetamol tutorial:
`solute_prod1.nc` is the trajectory, `mdout.csv` the state table, `cMD.xml` the final state, and
`cMD.log` the record completion is read from.
