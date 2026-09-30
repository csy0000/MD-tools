# Hydration free energy: ethanol out of water, in twelve windows

**Tested against md-tools `0.6.4`.** Every command and every number on this page comes from a run
executed as written, at that version, on four NVIDIA RTX 3080 GPUs with CUDA and mixed precision.
The whole ladder took **31 minutes** of wall clock.

**Starts from a built system.** Do [the system page](index.md) first — you need
`build/built.{xml,pdb,log}` and the ligand package they were built from.

## What this computes, and what it does not

ΔG_hyd is the free energy of moving one ethanol molecule from vacuum into water. This calculation
gets it by going the other way and changing sign: **switch off every interaction between the
molecule and its surroundings**, measure the work, and negate it.

```text
s = 0    ethanol interacting with water normally
s = 1    ethanol interacting with nothing outside itself
         (its OWN bonds, angles, torsions and internal pairs unchanged at both ends)

dG_hyd = -dG(s: 0 -> 1)
```

!!! warning "One leg, not two"

    A decoupling cycle is usually drawn with two legs — one in solvent and one in vacuum — and the
    vacuum leg here is **exactly zero by construction**. The plan keeps the molecule's own
    Hamiltonian physical at both ends, so in vacuum the potential does not depend on λ at all:
    ΔG = 0 with no variance. Running it would burn GPU hours sampling a Hamiltonian that does not
    change, and reporting it as a measured leg would dress up a definition as a result. So: one leg.

## 1. Declare what the far end of the path is

An alchemical calculation needs two end states. Here the second one is not another molecule — it is
**this molecule, absent**. That is `mode: decoupling`.

`decouple.config`:

```yaml
# decouple.config -- endpoint B is ethanol ABSENT
format: md-tools-combine-topology/1
mode: decoupling

endpoints:
  A:
    parameters: ./build/ligands/ethanol     # a PATH, relative to this file

environment:
  system: ./build/built.xml
  topology: ./build/built.pdb
  record: ./build/built.log
  ligand:
    resname: EOH
```

```bash
md-openmm combine-topology --config decouple.config -odir ./plan --check
md-openmm combine-topology --config decouple.config -odir ./plan
```

```text
combine-topology: decoupling plan for LOCAL-LFQSCWFLJHTTHZ/param_d66453ef683a -- endpoint B is the ligand absent, plan_sha256 b663f7bf42daf492...
combine-topology: plan written: plan
```

There is **no `endpoints.B`, no `map` and no `b_pose`**, and each of them is refused by name if you
write one. That is not pedantry: a `map` under `mode: decoupling` is almost always a mutation
somebody wrote and then set the wrong mode on, and silently ignoring it would run a different
calculation without a word.

`--check` first, always. It builds the plan, runs every check, and writes nothing — not even the
output directory.

## 2. Place the windows

`hydration.config`:

```yaml
# hydration.config -- twelve fixed-lambda windows decoupling ethanol from TIP3P water
protocol: alchemical

alchemical:
  cycle: AHFE                     # absolute hydration; names the run directory
  leg: 1                          # -> AHFE-leg1-run<N>/ and input/AHFE-leg1.in
  plan: ./plan
  lambda_path: linear
  number_of_windows: 12
  window_steps: 500000            # 1 ns of production per window, at 2 fs
  equilibration_steps: 10000      # 20 ps, discarded, at the window's OWN Hamiltonian
  report_interval_steps: 500
  checkpoint_interval_steps: 50000
  minimize_iterations: 500

dynamics:
  temperature_K: 300.0
  pressure_bar: 1.01325
  timestep_fs: 2.0
```

Nothing here restates anything the plan already holds. The two end states, the atom numbering and
the environment are all in `plan/plan.json`, with their own digests; this file says only which path
is walked across it, where the windows sit, and how long each one samples.

**`equilibration_steps` is not optional in spirit.** Every window starts from a configuration that
was equilibrated at some *other* λ, so its first frames are not from its own ensemble. Those 20 ps
are thrown away before a single sample is taken.

```bash
md-openmm build-md --config hydration.config -odir ./AHFE-leg1-run1
```

An alchemical run directory is `<cycle>-leg<i>-run<N>`, and the name is checked rather than
suggested. **An absolute hydration has ONE leg** — the vacuum leg of this construction is
identically zero and is not run — so there is only ever `AHFE-leg1`, and a second repeat is
`AHFE-leg1-run2`.

```text
alchemical
----------
  plan                        plan.b663f7bf42da (decoupling, retain-all)
  end states                  LOCAL-LFQSCWFLJHTTHZ/param_d66453ef683a -> the ligand ABSENT (decoupled)
  environment                 explicit
  windows                     12 at s = 0, 0.0909091, 0.181818, 0.272727, 0.363636, 0.454545, 0.545455, 0.636364, 0.727273, 0.818182, 0.909091, 1
  path                        linear
  softcore                    sc True, amber18, scalpha 0.5, scbeta 12 nm^2, 1-4 scaled
  per window                  500000 steps = 1000 ps production, after 10000 steps equilibration
```

## 3. Run the windows, in any order, on any number of cards

```bash
cd AHFE-leg1-run1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<card> WINDOWS="w000 w001 w002" ./run.sh &
CUDA_VISIBLE_DEVICES=<card> WINDOWS="w003 w004 w005" ./run.sh &
CUDA_VISIBLE_DEVICES=<card> WINDOWS="w006 w007 w008" ./run.sh &
CUDA_VISIBLE_DEVICES=<card> WINDOWS="w009 w010 w011" ./run.sh &
wait
```

```text
== w000 ==
w000: fresh, 1001 rows
== w001 ==
w001: fresh, 1001 rows
== w002 ==
w002: fresh, 1001 rows
run.sh: all windows reported completion
```

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Without it CUDA orders devices by speed, so
`CUDA_VISIBLE_DEVICES=1` need not be the card `nvidia-smi` calls 1 — on a machine with mixed models
every index can shift by one, and you will spend an hour on the wrong card wondering why it is
contended.

**The windows are independent.** No ordering, no restart handed from one to the next. `./run.sh`
with no `WINDOWS` runs all twelve in series; splitting them is the same campaign. A window that
already completed and verifies is **skipped**, so re-running after an interruption finishes what is
left rather than starting over.

!!! note "A window that minimises is not reproducible on a card"

    `minimize_iterations: 500` means each window minimises before drawing velocities, and
    `LocalEnergyMinimizer` is **not deterministic on CUDA**. Measured on this Hamiltonian: across
    ten fresh Contexts the energies and forces at the same positions are bit-identical — spread
    exactly zero — and yet five minimisations from that same start scatter by 16 kJ/mol in mixed
    precision and 24 in double. There is no seed for it; minimisation has none.

    So the System, the plan and `resolved.config` are reproducible, and what follows from them on a
    card is a new realisation of the same calculation. This is the same class of statement as an
    AIS resume being exact in committed state and not in trajectory. It is also why two `--cpu`
    runs agreeing tells you nothing about a card.

## 4. Read the result

Analysis is deliberately not a `build-md` setting: the engine produces samples, and which estimator
you trust over them is your decision. The script beside this page is ~70 lines.

```bash
python ethanol_analysis.py AHFE-leg1-run1
```

```text
dG_hyd(MBAR)            -3.648 +/- 0.107 kcal/mol

the same samples, four ways:
  MBAR         dG_hyd   -3.648 kcal/mol
  BAR          dG_hyd   -3.699 kcal/mol
  TI           dG_hyd   -4.195 kcal/mol
  EXP_forward  dG_hyd   -3.838 kcal/mol
  EXP_reverse  dG_hyd   -3.622 kcal/mol

smallest neighbour overlap  0.178   (a ladder is in trouble below ~0.03)
poor overlap anywhere       False
samples per window          247-430 after decorrelation
```

## 5. What the numbers mean

| | ΔG_hyd, kcal/mol | what it is |
|---|---|---|
| **this run, MBAR** | **−3.648 ± 0.107** | one ladder, 12 windows × 1 ns |
| **three repeats, MBAR — the reference** | **−3.523 ± 0.025** | the [validation campaign](../../openmm_methods/alchemy/validation.md), error-barred by the repeat spread |
| experiment (FreeSolv `mobley_2310185`) | −5.00 | measured |

**−3.648 and −3.523 are the same number here.** They differ by 0.125, which is about one standard
error of the single run, and the three campaign repeats themselves span 0.087 (−3.483, −3.515,
−3.570). One ladder lands within the scatter of three; that is what agreement looks like at this
sample size, and a tighter match would be luck rather than a better result.

**The 1.4 kcal/mol gap to experiment is force field, not machinery.** The campaign found the same
offset in the same direction for a hydrocarbon, an alkyl halide and an alcohol — 0.65 to 1.48
kcal/mol, all too positive. A systematic shift across three chemistries is what a force field and
water model that were not fitted to these numbers look like. It is reported, never gated on.

## 6. Why TI disagrees, and why that is the useful line

Look again at the estimator block. MBAR, BAR and both EXP directions sit between −3.62 and −3.84.
**TI gives −4.195** — half a kcal/mol away from the rest.

That is not a bug and it is not noise. TI integrates ⟨∂U/∂λ⟩ over λ with a trapezoid through twelve
points. Near s = 1, where the last of the molecule's interactions with water are switched off,
⟨∂U/∂λ⟩ varies sharply, and a trapezoid drawn across a 1/11-wide gap in that region cannot follow
it. MBAR does not integrate anything — it reweights configurations between states — so it is not
subject to that error at all.

**The lesson to take to a harder system:** when the estimators over one set of samples disagree,
the disagreement tells you which assumption failed. TI away from MBAR and BAR means the λ SPACING
is too coarse somewhere. EXP forward away from EXP reverse would mean the exponential average is
being carried by its tail. MBAR away from BAR would mean something is wrong with the samples
themselves. Four estimators over identical samples are not four measurements — they are four probes
of one, and their pattern is a diagnostic.

For this calculation MBAR is the number to quote, and the overlap gate is what licenses it: 0.178
at the tightest neighbouring pair, against a floor of about 0.03. Twelve evenly spaced windows are
enough for ethanol.

## What it wrote

```text
EOH/
  build/                        built.xml, built.pdb, built.log
  plan/                         plan.json, system_a.xml, system_b.xml, combined.pdb, positions.npy
  AHFE-leg1-run1/
    resolved.config             AUTHORITATIVE: the resolved declaration the windows re-read
    plan.b663f7bf42da/          the plan, COPIED IN and content-addressed
    leg/
      leg.json                  endpoint_a: coupled, endpoint_b: decoupled, the s values
      system.xml                the mixed System, built ONCE at generation, digest-checked
      topology.pdb
      windows/r1/
        w000.samples.csv        every window's energy at EVERY state, plus dU/dlambda at its own
        w000.complete.json      rows, digests, the self-check and the fingerprint
        w000.checkpoints/
    w000.py .. w011.py          two-line entry points, one per window
    run.sh
  input/AHFE-leg1.in            the shared input this leg's repeats read
```

`w000.samples.csv` is where the free energy actually lives: each row is one configuration's energy
evaluated at all twelve states, which is exactly what MBAR consumes.

## Next

* [What the alchemical runtime has been shown to compute](../../openmm_methods/alchemy/validation.md)
  — the ethane → chloroethane cross-check, where decoupling two molecules separately reproduces the
  result of mutating one into the other
* [Alchemical topology](../../openmm_methods/alchemy/README.md) — `single`, `hybrid` and `dual`
  plans, for turning one ligand into another rather than removing it
* [The system page](index.md) — the build, and why there is no salt in the box
