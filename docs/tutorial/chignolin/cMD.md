# cMD: chignolin in explicit water

**Tested against md-tools `0.5.4`.** Every command and every number comes from a run executed as
written, at that version, on one RTX 3080. It has not been re-run for 0.6.1.

1 ns of ordinary MD on the folded NMR structure of [chignolin](index.md). **Total time: 1 min 3 s.**

## 1. The structure

```bash
mkdir -p CHI/build && cd CHI/build
curl -O https://files.rcsb.org/download/1UAO.pdb
awk '/^MODEL/{m++} m==1{print} /^ENDMDL/{if(m==1) exit}' 1UAO.pdb \
    | grep -E '^(ATOM|TER)' > chignolin.pdb
echo END >> chignolin.pdb
```

1UAO holds 18 NMR models; the build needs one. That leaves 138 atoms.

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

Writes the System, the structure that matches it and the build record. From `built.log`:

```text
  atoms                       2553
  solute atoms                138
  waters                      803
  ions                        {'NA': 4, 'CL': 2}
  HMR                         applied, target 3.024 amu, recommend 4.0 fs
```

Four Na⁺ against two Cl⁻ because chignolin carries −2. What every key means:
[build-top](../../basics/build-top/index.md).

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

```bash
cd ..                                            # CHI/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
```

Lengths are integer step counts. `timestep_fs` stays `auto` and resolves to **4.0 fs**, read from
the masses in `built.xml` rather than from this file — HMR has to be proved, not claimed. What
`build-md` writes and checks: [build-md](../../basics/build-md/index.md). Every key:
[the configuration reference](../../basics/build-md/configuration.md).

## 4. Run it

```bash
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./run.sh
```

`run.sh` is five `md-run` calls in order — minimisation, three equilibration stages, production:

```text
== min ==     min: 0 steps completed, 0 ps
== eq_1 ==    eq_nvt_posres: 25000 steps completed, 100 ps
== eq_2 ==    eq_npt_posres: 25000 steps completed, 100 ps
== eq_3 ==    eq_npt_free:   50000 steps completed, 200 ps
== cMD ==     cMD:          250000 steps completed, 1000 ps
```

## 5. What it wrote

```text
CHI/
├── build/          built.xml, built.pdb, built.log
├── input/          min.in, eq_1.in, eq_2.in, eq_3.in, cMD.in
├── min/            min.xml, min.log, min.out, min.py
└── cMD-run1/
    ├── eq/                    eq_1.xml … eq_3.xml and their logs
    ├── solute_prod1.nc        solute trajectory, 500 frames (855 kB)
    ├── mdout.csv              the state table, 100 rows
    ├── energy_components.csv  the potential, term by term
    ├── cMD.xml                the final state
    ├── cMD.checkpoints/       the committed generations a resume continues from
    └── cMD.log                the machine-readable record
```

`input/` and `min/` are siblings of the run, not inside it: they belong to the system, and every
run on it shares them. Why, and what each file is: [the run layout](../../basics/run-layout.md).

From `cMD.out`:

```text
  timestep                     4.0 fs
  steps                        250000 (1000 ps)
  ensemble                     NPT
  platform                     CUDA
  Speed (ns/day)               mean 2349.4   rms fluctuation 251.236
  elapsed                      36.6 s
```

`mdout.csv` opens at step 102500, not at zero: the step counter is absolute across the chain.

## Next

1 ns is a demonstration. Chignolin folds on the microsecond scale, so raise `production_steps` —
or use the ladder:

* [REST2: chignolin](REST2.md) — the same peptide across six scaled states
* [the system page](index.md) — the other methods for chignolin
