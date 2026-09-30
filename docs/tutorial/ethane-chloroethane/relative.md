# Relative hydration free energy: ethane → chloroethane, through a cycle

**Tested against md-tools `0.6.4`.** Every command and every number on this page comes from a run
executed as written, at that version, on four NVIDIA RTX 3080 GPUs with CUDA and mixed precision.
Four ladders, 102 windows, 102 ns, about two hours of wall clock.

**Starts from two built boxes.** Do [the system page](index.md) first — you need a vacuum build and
a solvent build. Both legs run under one `<system>`.

## 1. Map one molecule onto the other

A relative calculation needs to know which atom becomes which. `mode: hybrid` keeps a mapped core
plus endpoint-unique atoms on both sides, as separate particles, and never maps a hydrogen to a
heavy atom.

`mutate.config` — one per box, pointing at that box's build:

```yaml
# mutate.config -- ethane -> chloroethane, hybrid topology
format: md-tools-combine-topology/1
mode: hybrid

endpoints:
  A:
    parameters: ./parameter/ethane
  B:
    parameters: ./parameter/chloroethane

environment:
  system: ./build/built.xml
  topology: ./build/built.pdb
  record: ./build/built.log
  ligand:
    resname: ETA

map:
  automatic: true
```

```bash
md-openmm combine-topology --config mutate.config -odir ./plan --check
md-openmm combine-topology --config mutate.config -odir ./plan
```

```text
combine-topology: hybrid plan LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4 -> LOCAL-HRYZWHHZPQKTII/param_5930c10b0577, 7 mapped atom pairs, plan_sha256 f140fe5bf3511d5d...
combine-topology: plan written: plan
```

**`map: {automatic: true}` PROPOSES; it does not decide.** The proposal is written beside the plan
as `plan.map.yaml`, for you to read:

```yaml
map:
  a_to_b:
    '0': 0
    '1': 1
    '2': 5
    '3': 3
    '4': 4
    '6': 6
    '7': 7
```

Seven pairs, and note `'2': 5` — the proposal does not assume the two molecules are numbered the
same way. Handed back as `map: {file: plan.map.yaml}` it reproduces the same plan, with the same
`plan_sha256`, which is what makes "review the proposal, then commit to it" a checkable step rather
than a hope.

Do this for **both boxes**. The two plans have different `plan_sha256` — they combine into
different boxes — but they must share one **ligand Hamiltonian**, and step 4 checks that they do.

## 2. Place the windows, and not evenly in vacuum

`mutate-md.config` for the solvent leg:

```yaml
# mutate-md.config -- the solvent leg
protocol: alchemical
alchemical:
  cycle: RHFE                     # relative hydration; names the run directory
  leg: 1                          # -> RHFE-leg1-run<N>/ and input/RHFE-leg1.in
  plan: ./plan
  lambda_path: linear
  lambda_values: 0.0, 0.0666666667, 0.1333333333, 0.2, 0.2666666667, 0.3333333333, 0.4, 0.4666666667, 0.5333333333, 0.6, 0.6666666667, 0.7333333333, 0.8, 0.8666666667, 0.9333333333, 1.0
  window_steps: 500000
  equilibration_steps: 10000
  report_interval_steps: 500
  checkpoint_interval_steps: 50000
  minimize_iterations: 500

dynamics:
  temperature_K: 300.0
  pressure_bar: 1.01325
  timestep_fs: 2.0
```

The vacuum leg is `leg: 2`, and is otherwise identical except for its plan and **two extra windows
near s = 0**:

```yaml
  lambda_values: 0.0, 0.0166666667, 0.0333333333, 0.0666666667, 0.1333333333, 0.2, ...
```

`1/60` and `1/30` sit inside the first ordinary gap. They are there because the appearing chlorine
enters against a partner it has no repulsive core against yet, and ⟨∂U/∂λ⟩ rises sharply over the
first few percent of the path — a region a 1/15-wide gap steps straight over. This is what
`lambda_values` is for: even spacing is a default, not a rule, and a ladder should be finer where
the integrand is steep.

```bash
md-openmm build-md --config mutate-md.config -odir ./RHFE-leg1-run1
```

The directory name is **checked, not suggested**. `-odir ./md_script` is refused, naming
`RHFE-leg1-run<N>`, because the combination step in section 4 takes one directory per repeat and a
name that does not say which leg it is turns assembling a cycle into guesswork over six
directories.

```text
  plan                        plan.ad309cab11ed (hybrid, retain-all)
  end states                  LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4 -> LOCAL-HRYZWHHZPQKTII/param_5930c10b0577
  environment                 vacuum
  windows                     18 at s = 0, 0.0166667, 0.0333333, 0.0666667, 0.133333, 0.2, ...
```

**`environment` is read from the plan, not declared here.** The plan carries the environment it was
combined into, whose build record says `solvent.treatment`. A vacuum leg runs NVT and a solvent leg
NPT, and neither is something you set twice.

!!! note "Two legs, one system, two inputs"

    `input/<cycle>-leg<i>.in` is shared by every REPEAT of one leg and by nothing else, so the two
    legs coexist without competing for a file they would each resolve differently. Give a second
    leg the same `cycle` and a different `leg`, and it gets its own input beside the first.

    A *repeat* is different again, and shares the input deliberately — see section 3.

## 3. Run both legs, and three repeats of each

```bash
cd RHFE-leg1-run1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<card> WINDOWS="w000 w004 w008 w012" ./run.sh &
...
wait
```

**A REPEAT IS A RUN DIRECTORY, not a name inside one.** A generated run always writes `r1`; two
repeats sharing a directory would be two experiments over one set of paths, and a run directory is
an identity. So a second repeat is a second `build-md`, differing in `dynamics.seed` and in nothing
else:

```yaml
dynamics:
  seed: 202          # run2; 303 for run3
  temperature_K: 300.0
  ...
```

```bash
md-openmm build-md --config mutate-md-run2.config -odir ./RHFE-leg1-run2
```

The shared `input/alchemical.in` is *not* refused for a repeat, because the seed is deliberately
not written into it — the `.in` says what the method is, and the seed says which repeat this is.

The vacuum legs take minutes; the solvent legs about forty minutes each on four cards.

## 4. Close the cycle

```bash
python relative_analysis.py \
    RHFE-leg2-run1,RHFE-leg2-run2,RHFE-leg2-run3 \
    RHFE-leg1-run1,RHFE-leg1-run2,RHFE-leg1-run3
```

```text
ddG_hyd = dG_hyd(LOCAL-HRYZWHHZPQKTII/param_5930c10b0577) - dG_hyd(LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4)

  +1 x solvent  solvent (3 repeats)      dG =   -3.385 +/- 0.020 kcal/mol
  -1 x vacuum   vacuum (3 repeats)       dG =   -1.639 +/- 0.000 kcal/mol

ddG_hyd =   -1.746 +/- 0.020 kcal/mol   (3 repeat(s), MBAR)
  legs independent; variances add

matched_legs: one ligand Hamiltonian c61e4e0de022f93b... across both legs
```

**`matched_legs` is what licenses the subtraction**, and it runs before a single sample is read.
Two legs that do not carry one ligand Hamiltonian are refused. Without that check, a vacuum leg
built from one parameterisation and a solvent leg from another would subtract just as cleanly and
mean nothing at all — the two ΔG values would be perfectly good numbers describing different
molecules.

## 5. The result, and what agrees with what

| route | ΔΔG_hyd, kcal/mol | what it shares with this run |
|---|---|---|
| **this run — mutation, 3 repeats** | **−1.746 ± 0.020** | — |
| mutation, driven from Python (0.7.0 M2, 3 repeats) | −1.743 ± 0.026 | everything but the driver |
| **decoupling each molecule** (3 repeats each) | **−1.639 ± 0.074** | almost nothing |
| experiment (FreeSolv) | −2.46 | — |

**−1.746 against −1.743 is not an independent check.** It is the same Hamiltonian, the same window
grid and the same lengths, driven from a generated directory instead of a Python script. Agreeing
to 0.002 says the command-line path is faithful — which is worth establishing, and is all it says.

**−1.639 ± 0.074 is the independent one.** It comes from decoupling ethane and chloroethane
separately and subtracting their absolute hydration free energies: a different plan mode, a
different Hamiltonian, one leg per molecule instead of two, no atom map at all. It sits **0.107**
from this run, inside its own error bar. Two routes sharing almost nothing arrive at the same
number, and an error in the softcore, the estimator or the cross-state bookkeeping would have to
corrupt both by the same amount in the same direction to survive that.

**The 0.7 kcal/mol gap to experiment is force field**, not machinery, and both routes show it. It
is reported, never gated on.

### The vacuum leg prints ±0.000, and that is not a bug

Three repeats gave **−1.6389, −1.6384, −1.6389** — 0.0005 apart, which rounds to ±0.000 at three
decimals. Eight atoms with no solvent, 18 windows of 1 ns: there is very little left for three
seeds to disagree about.

The solvent leg's repeats are **−3.3868, −3.4189, −3.3506**, a spread of 0.068. Expect essentially
all of your run-to-run scatter to come from the water leg, and size your repeats accordingly —
there is little point running the vacuum leg more times than the solvent one.

`combine_repeats` reports the **larger** of the estimator's uncertainty and the repeat spread, so
you get the honest error bar without choosing.

## What it wrote

```text
<system>/
  solvent-build/  vacuum-build/     the two boxes
  plan-solvent/   plan-vacuum/      one plan per box, + its .map.yaml
  input/
    RHFE-leg1.in                    the solvent leg, shared by its three repeats
    RHFE-leg2.in                    the vacuum leg
  RHFE-leg1-run1/  RHFE-leg1-run2/  RHFE-leg1-run3/
  RHFE-leg2-run1/  RHFE-leg2-run2/  RHFE-leg2-run3/
```

Six run directories under one system, each a complete and self-describing experiment: its own
`resolved.config`, its own copy of the plan (content-addressed), its own `leg/` whose System was
built once at generation, and its own seed in `run.config`. The name says which leg of which cycle
it is, so six directories sort into two legs without being opened.

## Next

* [Absolute hydration by decoupling](../ethanol/hydration.md) — the other route, and the one that
  gives numbers you can compare with FreeSolv directly
* [What the alchemical runtime has been shown to compute](../../openmm_methods/alchemy/validation.md)
* [Alchemical topology](../../openmm_methods/alchemy/README.md) — the modes, and what each claims
