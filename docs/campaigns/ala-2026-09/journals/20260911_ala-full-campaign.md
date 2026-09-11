# 2026-09-11 — the full ALA campaign: 300 ns hot, 3 × 1000 ns cold, 300 ns REST2, both solvents

Ten runs in `data/ala-campaign-full-20260910/` (untracked: `data/` is generated output). MD-tools
`d157c63`, which is `0.5.1` plus its release note.

## Facts established before acting

| question | answer | how |
|---|---|---|
| throughput, implicit | **2831 ns/day** | 20,000 steps timed on one RTX 3080 after a warm-up |
| throughput, explicit | **1837 ns/day** | same |
| can explicit cold be NVT? | **no** | `build/md.py`: `fixed_volume = implicit or scaled`; no `ensemble` key in the schema |
| can hot (τ=0.5) be NPT? | **no, and it should not be** | `stage_main` refuses: "the barostat's volume move is defined for the unscaled Hamiltonian" |
| does REST2 equilibrate its box? | yes, at τ=0 | `rest2_explicit/eq_npt_posres.in` carries `tau = 0.0` |
| as-built explicit density | **0.947 g/cm³**, 3.8% under TIP3P | box vectors of `explicit.xml`, mass ÷ volume |
| after NPT equilibration | **0.9934 g/cm³** | `cold_explicit_r1/eq_npt_free.xml` |

## What exists

| run | protocol | solvent | ensemble | target |
|---|---|---|---|---|
| `hot_implicit` | cMD, τ=0.5 | GBn2 | NVT | 300 ns |
| `hot_explicit` | cMD, τ=0.5 | TIP3P + 0.15 M NaCl | NVT | 300 ns |
| `cold_implicit_r{1,2,3}` | cMD, τ=0 | GBn2 | NVT | 1000 ns each |
| `cold_explicit_r{1,2,3}` | cMD, τ=0 | TIP3P + 0.15 M NaCl | **NPT** | 1000 ns each |
| `rest2_implicit` | REST2, 4 rungs, τ 0→0.5 | GBn2 | NVT | 300 ns/state |
| `rest2_explicit` | REST2, 6 rungs, τ 0→0.5 | TIP3P + 0.15 M NaCl | NVT | 300 ns/state |

Repeats differ by `dynamics.seed` (1, 2, 3). Same seed would have been one run copied three times.

Common: 2 fs, 300 K, ff14SB. Solute written every 5 ps (2500 steps); whole system every 50 ns
(25,000,000 steps) for explicit only. REST2 exchanges every 10 ps (5000 steps) × 30,000 attempts.

Ladders complete: `rem.log` carries *n_states + 1* lines per exchange, so 150,006 lines ÷ 5 and
210,006 ÷ 7 are both exactly **30,000 exchanges = 300 ns per state**.

## Deviations from the instruction — three, and why

**1. `hot_explicit` does not equilibrate its own box.** A scaled run is fixed-volume throughout,
so it cannot run a barostat at any stage; the runtime refuses it, correctly. Its chain therefore
starts from `cold_explicit_r1/eq_npt_free.xml` — unscaled, NPT-equilibrated, 0.9934 g/cm³ — which
is the same shape REST2 uses (τ=0 equilibration, then scaling). Started via a hand-written
`hot_explicit/run_from_equilibrated.sh`, because `run.sh` has no way to accept a starting
structure for cMD: its third positional argument is the AIS source-trajectory slot.

Without this the run would have sampled `build-top`'s box at **0.947 g/cm³, 3.8% under density**,
for 300 ns, with nothing able to relax it. Nothing warns about this.

**2. The implicit runs used 100 ps of restrained NVT equilibration, not 20 ps.** They were
generated and launched before the equilibration protocol (explicit: restrained NVT 10 ps +
restrained NPT 10 ps; implicit: restrained NVT 20 ps; k = 1.0 kcal/mol/Å² throughout) was stated.
Longer rather than shorter, restrained throughout, and the production stage is identical, so they
were left to finish rather than discarding roughly 25 GPU-hours. The explicit runs follow the
protocol exactly.

**3. The explicit colds are NPT while the explicit hot and both ladders are NVT.** Requested as
NVT, changed to NPT deliberately once it was established that explicit cold NVT is not
expressible. **A consequence to carry into any analysis:** every rung of a REST2 ladder is NVT,
including τ=0, so `cold_explicit_*` is not the same ensemble as `rest2_explicit`'s bottom rung.
The implicit colds have no such mismatch — implicit solvent is NVT everywhere.

## Defects found, not fixed here

* **A scaled explicit cMD run started through `run.sh` silently samples an unequilibrated box.**
  The refusal to equilibrate under a scaled Hamiltonian is right; producing a run at build-time
  density without saying so is not. Belongs upstream in MD-tools — either refuse a scaled explicit
  run that has no `-c`, or say in `run.sh` that one is required.
* **`run.sh` cannot take a starting structure for cMD.** Worked around with a local script.
* `src/ALA/config/REST2.config` in this repository uses the pre-rename reporting keys
  (`solute_printout`, `system_printout`) and is refused by MD-tools ≥ `7d76b05`. It matches this
  project's own pin (`61d5b35`), which predates the rename — so moving the pin requires updating
  the configs. Not touched here.

## One thing that was attempted and reverted

`build/md.py` was changed so that explicit runs always equilibrate under NPT, on the reasoning that
a scaled run's *volume* being fixed is no argument for never having equilibrated it. `stage_main`
refused the result, and was right to: the barostat's Metropolis move evaluates the unscaled
Hamiltonian, so there is no correct way to run one at τ=0.5. MD-tools is unmodified by this
campaign. The legitimate route is the one REST2 already uses and deviation 1 now follows.

## Left undone

* `cold_explicit_r{1,2,3}` and `hot_explicit` were at 2% and 6% when this was written; the three
  implicit colds at 66–84%. Nothing is analysed yet.
* No analysis, no estimator, no comparison between arms. This journal records what was generated.
* The ensemble mismatch in deviation 3 is recorded, not resolved. If the explicit cold is ever
  needed in NVT, MD-tools needs a way to ask for it.

## Registration, 2026-09-11

Three runs had finished by the afternoon — `hot_implicit`, `cold_implicit_r2`, `cold_implicit_r3` —
and are registered as shared reference data:

```
$MD_DATA/common/reference/2026-09/ALA-dipeptide-implicit/cMD-hot/run1
$MD_DATA/common/reference/2026-09/ALA-dipeptide-implicit/cMD-cold/run2
$MD_DATA/common/reference/2026-09/ALA-dipeptide-implicit/cMD-cold/run3
```

Each holds `bundle/` (runnable with OpenMM alone), `data/` (solute trajectory, `mdout.csv`,
`energy_components.csv`), `provenance/` (the engine's machine records), a README, `dataset.yaml`
and `SHA256SUMS`. The paths under `data/reference-staging-20260911/` are now symlinks into the
store, which is what `data-register` leaves behind.

`common/` no longer sits under a year segment. A project dataset is year-first because a project
happened in a year; a reference simulation is used by whoever needs it for as long as it is the
best one available, and filing it under `2026/` means finding it requires already knowing when it
was made. This needed a change in MD-tools' data contract and is released there.

### The exporter was wrong first, and the first test agreed with it

`md-openmm export-reference` writes the MD-tools-free bundle. Its first version exported
`implicit.xml` as `system.xml` — the *build* System, which no stage integrates. Between that file
and the Context, MD-tools scales the solute at τ, adds the positional-restraint force, and adds
the barostat for explicit solvent. **At τ = 0.5 that is a different Hamiltonian.** Three more of
the same kind: the integrator seed is `derive_seed(config_seed, stage_name)` and not the config
value; a production stage continues from the state its `-c` named rather than from the built
coordinates; and `Context.setState` restores global parameters, so a state written by a restrained
equilibration comes back still restrained unless the runner resets it.

None of it was caught by the first test, which compared the exported runner against a Context
built inside the test file — sharing every one of the bundle's assumptions, and agreeing with all
four mistakes. The test now runs a real scaled stage through the engine and requires the bundle's
serialised end point to equal the engine's `cMD.xml` element for element, single-threaded on both
sides. **A second implementation checked against a third written by the same hand is not a check.**

### Not yet exported

The two finished ladders. A REST2 bundle needs the per-rung scaled Systems and the exchange loop
(about 150 lines; acceptance is one line, `log α = [u_ii + u_jj] − [u_ij + u_ji]`), with its own
equivalence test against the engine's `rem.log`. Until that exists there is no honest way to hand
out a ladder that runs without MD-tools.

### Worth improving

The registered manifests say *"No build record was present to describe the system"* under
`system:`, because the staged directories carry the stage records but not `implicit.log` from
`build-top`. Copying the build log in beside them would let the manifest describe the system
instead of declining to.
