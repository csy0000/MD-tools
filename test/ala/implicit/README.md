# Alanine dipeptide, implicit solvent (GBn2 / mbondi3)

ACE-ALA-NME, the complete solute, in generalised-Born implicit solvent. **Four replicas.**

A smoke run here is engineering validation: it shows the chain executes, continues and records what
it did. It is **not** convergence, and it is not scientific validation of the protocol.

## What makes this different from `../REST2/`

| | explicit water | implicit (this) |
|---|---|---|
| solvent | OPC box, ions, PME | GBn2 with mbondi3 radii |
| box | dodecahedron, 1.2 nm padding | **none** |
| stages | min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1, REST2_1 | **min, eq, cMD_1, REST2_1** |
| barostat | on the NPT and production stages | **never** |
| timestep | 4 fs with HMR to 3.024 amu | **2 fs, no HMR** |
| replicas | 6 | **4** |

There is no NPT stage and there cannot be one. Implicit solvent has no box, so there is no volume to
equilibrate and pressure is undefined; an NPT stage or a `pressure` field is refused before
generation rather than ignored at run time.

Hydrogen mass is **not** repartitioned. The pinned reference builds its base System without
repartitioning and the GBn2 energy validation is against an unrepartitioned System, so the implicit
profile does not inherit the explicit-water HMR. Without it, 4 fs is not stable for the fastest
remaining motions, which is why the timestep is 2 fs. That choice is stated in
`implicit-rest2-peptide-v1`, not inherited by accident.

## The Hamiltonian

Built through ParmEd, which is part of the scientific identity rather than a style choice:

```python
st = parmed.load_file("system.prmtop", xyz="system.rst7")
parmed.tools.changeRadii(st, "mbondi3").execute()
system = st.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds,
                         implicitSolvent=GBn2, removeCMMotion=True)
```

`AmberPrmtopFile.createSystem()` is **not** interchangeable with this. Measured on this system,
with identical per-particle GB parameters and identical radii, the two agree to 0.0000 kJ/mol on
every force except `CustomGBForce`, where they differ by **16.05 kJ/mol**. Under REST2 that offset
is several kT of spurious work, so every replica must sit on the same branch.

`changeRadii` is a **no-op here** — tleap has already written mbondi3, measured max |dR| = 0.0000 Å.
It runs anyway because it is load-bearing for the OpenFF route, whose topologies carry no GB radii.

`system.prmtop` and `system.rst7` are construction intermediates and provenance for the OpenMM
System. This repository has no Amber execution engine.

## Commands

```bash
# 1. prepare the system (no dynamics)
python MD_system_gen.py -i ace_ala_nme.pdb -o ala_implicit \
    --config test/ala/implicit/system_config.json

# 2. generate the protocol
python MD_input_gen.py --system ala_implicit/system_manifest.json -o ala_implicit_run \
    --config test/ala/implicit/md_config.json

# 3. run it
cd ala_implicit_run && ./run_all.sh
```

`md_config_smoke.json` is the same protocol at smoke sizing, for checking an installation.

## Stage graph

```
ala_implicit_run/
    inputs/        the prepared system, copied
    min/           restrained minimisation
    eq/            restrained equilibration, 20 ps -- velocities initialised HERE, once
    cMD_1/         unrestrained NVT conventional MD
    REST2_1/       replica exchange, one run containing its segments
    run_all.sh     run_manifest.json     run.log
```

There is no NPT stage and no stage called NVT. Equilibration still happens -- `eq` runs restrained
constant-temperature dynamics for 20 ps -- but it carries no ensemble label, because "NVT" fixes a
volume and this System has none.

Velocities are created once on entering `eq`, the first stage that integrates; minimisation writes a
state with no velocities, which is what triggers that single initialisation.

## Exchange derivation

The segment is stated by its duration and its exchange count; the interval is derived, in whole
steps, and refused rather than rounded if either division is inexact.

```
duration_per_segment            2 ns      stated
number_of_exchanges_per_segment 200       stated
timestep                        2 fs
steps per segment               1,000,000 (2 ns / 2 fs)
steps per exchange round        5,000     (1,000,000 / 200)
exchange_interval               10 ps     DERIVED
```

## The tau ladder

`s = (1 - tau)^2`, `sqrt(s) = 1 - tau`, linear in tau from 0 to 0.5 over 4 replicas:

```
tau    0.000   0.167   0.333   0.500
s      1.000   0.694   0.444   0.250
T_eff  300     432     675     1200    K
```

`tau = 0` is `s = 1`, the unscaled physical Hamiltonian, and the implicit System at `s = 1` is
bitwise identical to the unscaled GBn2 System.

Under implicit solvent the **whole system is the enhanced region**, and a partial selection is
refused. A Born radius depends on every other atom's position, so a partial enhanced region needs a
validated treatment of the solute-environment cross terms, and there is none here.

The complete `CustomGBForce` energy is scaled by `s`, not just its charge-dependent part: GBn2 has
three terms and one is a non-polar correction with no charge dependence, which `charge x sqrt(s)`
would leave untouched.

## Extension

`REST2_NUMBER_OF_SEGMENTS` in `run_all.sh` controls how many segments run. It is an execution choice and
never appears in the scientific JSON — putting it there would move the configuration hash and make a
longer run look like a different calculation.

To extend an existing run, invoke the stage again in the same directory:

```bash
cd ala_implicit_run/REST2_1 && ./REST2_1.sh
```

Each invocation continues the same run: the runner reads its own committed-generation record to find
the restart point, and lifetime exchange statistics accumulate while per-invocation statistics do
not.

## Status meanings

`completed` means the requested work for that invocation reached its committed boundary. A smoke run
that finishes is a completed smoke run, not a completed production run, and the manifest says which.
