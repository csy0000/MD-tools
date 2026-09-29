# cMD: one microsecond of TYK2 + ejm_31

**Tested against md-tools `0.6.1`.** Every command and every number on this page comes from a
run executed as written, at that version, on one NVIDIA RTX A5000.

**1 µs** of unbiased NPT dynamics on the kinase–inhibitor complex: 500 000 000 steps at 2 fs,
234 ns/day, **4 days 6 h 44 m** on a single card. This is the reference the enhanced-sampling pages
on this system are compared against — the point of a long plain run is to be the thing that needs
no defending.

**Starts from a built system.** Do [the system page](index.md) first: it prepares 4GJ3, builds or
reuses the ejm_31 parameter package, and writes `tyk2/build/built.xml`.

## 1. Generate the run

`cMD.config` at the dataset root:

```yaml
protocol: cMD
solvent: explicit

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 25000     # 50 ps
  restrained_npt_steps: 25000     # 50 ps
  unrestrained_npt_steps: 200000  # 400 ps
  production_steps: 500000000     # 1 us at 2 fs

reporting:
  crd_printout_solute: 25000      # a solute frame every 50 ps: 20 000 frames
  crd_printout_whole: 0           # no whole-system trajectory
  info_printout: 250000           # a state row every 500 ps: 2000 rows
  checkpoint_printout: 250000
```

```bash
cd ..                                            # tyk2/
md-openmm build-md -odir ./cMD-run1 --config cMD.config
```

**The reporting cadences are the part to think about at this length.** A microsecond is 500 million
steps, and a cadence that is fine for 10 ns will produce something unusable here:

| cadence | interval | what it produces |
|---|---|---|
| `crd_printout_solute: 25000` | 50 ps | 20 000 frames of the 4701 solute atoms — **1.1 GB** |
| `crd_printout_whole: 0` | — | nothing. Every atom every 50 ps would be ~12 GB |
| `info_printout: 250000` | 500 ps | 2000 rows in `mdout.csv` |
| `checkpoint_printout: 250000` | 500 ps | the resume granularity: at most 500 ps lost to a crash |

`crd_printout_whole: 0` is a deliberate choice, not an omission: the solute trajectory is what any
analysis of the ligand or the pocket needs, and the waters are 16 079 of the 16 462 residues.
Explicit `0` says "no whole-system stream" rather than leaving it to a default.

## 2. Run it

```bash
cd cMD-run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./run.sh
```

`run.sh` is five `md-run` calls in order — minimisation, three equilibration stages, production.
For a run of this length, two things about the runtime matter more than they do on a short one:

* **An interrupted stage continues automatically.** A stage whose committed checkpoint is short of
  its step count resumes from that checkpoint, with every appendable stream truncated to the counts
  that generation vouches for. You do not pass `--resume`, and you do not want `--overwrite`, which
  starts clean and discards the committed work.
* **CUDA is mandatory and there is no silent fallback.** A run that quietly moved to the CPU would
  finish this in about two years, having reported success at every stage.

## 3. Result

From `cMD-run1/cMD.out`:

```text
System
  atoms                        53030
  residues                     16462 (HOH 16079, NA 48, CL 44, LEU 35, GLU 23, GLY 20, ...)
  net charge                   -0.000 e
  degrees of freedom           108522
Method
  nonbonded                    PME, cutoff 0.9 nm, Ewald tolerance 0.0005
  constraints                  50565 bond(s)
  barostat in system           MonteCarloBarostat
Selections
  solute                       4701 atom(s) (restraint and REST2 region)
Run
  timestep                     2.0 fs
  steps                        500000000 (1e+06 ps)
  ensemble                     NPT
  platform                     CUDA
Averages
  over                         2000 report(s) in mdout.csv
  Potential Energy (kJ/mole)   mean -714483   rms fluctuation 844.308
  Temperature (K)              mean 300.322   rms fluctuation 1.29169
  Box Volume (nm^3)            mean 530.506   rms fluctuation 1.06434
  Density (g/mL)               mean 1.02056   rms fluctuation 0.00204735
  Speed (ns/day)               mean 234.001   rms fluctuation 5.29792
Summary
  cMD: 500000000 steps completed, 1e+06 ps
status               completed
```

Temperature holds at 300.3 K with an rms fluctuation of 1.3 K, and the density at 1.021 g/mL with
0.002 — both an order of magnitude tighter than the same quantities over 10 ns, because 2000 reports
over a microsecond average away the fluctuations a short run shows. That tightness is a property of
the averaging, not evidence that the run is better behaved.

**Speed is 234 ns/day with an rms of 5.3**, i.e. steady to about 2% over four days. A single
sustained number like this is worth more than a short benchmark: a 10-minute window on the same
system and card reports a higher rate, because set-up cost is amortised differently and the card is
not yet in a thermal steady state.

## 4. What it wrote

```text
tyk2/
├── build/          built.xml, built.pdb, ligands/, REST2/ or AIS/ if you built scaled states
├── input/          min.in, eq_1.in, eq_2.in, eq_3.in, cMD.in
├── min/            min.xml, min.log, min.out, min.py
└── cMD-run1/
    ├── eq/                    eq_1.xml … eq_3.xml and their logs
    ├── solute_prod1.nc        20 000 frames of 4701 atoms, 1.1 GB
    ├── mdout.csv              2000 rows, 280 kB
    ├── energy_components.csv  the potential term by term
    ├── cMD.xml                the final state
    ├── cMD.checkpoints/       committed generations, 500 ps apart
    └── cMD.log                the machine-readable record
```

## 5. What a microsecond is and is not enough for

It is enough to say the prepared complex is stable, to give the pocket's sidechains time to find
their rotamer populations, and to serve as the unbiased baseline the ladder and switching pages are
read against.

It is **not** a binding free energy, and no amount of plain MD on a bound complex will be one — the
ligand does not leave, so nothing here samples the unbound state. That needs an alchemical
transformation, which is 0.7.0.

It is also not obviously enough for the ligand's own slow degrees of freedom. The honest check is
whether a quantity you care about has stopped drifting over the last third of the run, measured
rather than assumed; the enhanced-sampling pages on this system exist because for the pocket
torsions the answer was no.

## Next

* the same region heated instead of simulated: [selective REST2](REST2.md), and
  [choosing the ladder](choosing-the-ladder.md) for how its rung count was measured
* the same region annealed: [AIS](AIS.md) — and where that method's statistics run out
* [the system page](index.md) — the other methods for this complex
