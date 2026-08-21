# cyclo-RGDfV, implicit solvent (GBn2 / mbondi3)

The vetted cyclo(Arg-Gly-Asp-D-Phe-Val) macrocycle in generalised-Born implicit solvent.
**Six replicas.**

The SMILES, stereochemistry, protonation and charge model are the ones already vetted for the
explicit-water example in `../REST2/` and are not reinvented here. Only the solvent treatment and
the protocol differ.

A smoke run here is engineering validation. A macrocycle's conformational ensemble is exactly the
thing a short run does not converge, so nothing in this directory is evidence about its
conformational preferences.

## What differs from the explicit example

| | explicit water | implicit (this) |
|---|---|---|
| solvent | OPC box, ions, PME | GBn2 with mbondi3 radii |
| box | dodecahedron, 1.2 nm padding | **none** |
| stages | min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1, REST2_1 | **min, eq, cMD_1, REST2_1** |
| barostat | on the NPT and production stages | **never** |
| timestep | 4 fs with HMR to 3.024 amu | **2 fs, no HMR** |
| replicas | 10 | **6** |

## Why `changeRadii` matters here and not for alanine

This is the case the unconditional `changeRadii` call exists for.

The ligand route parameterises the macrocycle with OpenFF/Sage and AM1-BCC charges, then serialises
those parameters to Amber files through ParmEd. That prmtop carries **no GB radii at all**, so
without `changeRadii(st, "mbondi3")` the radii would be zero and the GB energy meaningless. For the
tleap-built alanine topology the same call changes nothing (max |dR| = 0.0000 Å) because tleap
already wrote mbondi3.

Applying it unconditionally is therefore safe for the first case and required for this one, which is
cheaper than deciding per route and getting the decision wrong. The bundle manifest records what the
call actually did — `radii_change_was_a_no_op` and `radii_max_change_angstrom` — so a reader can see
which case a given bundle was, rather than trusting a claim.

The System is then built through the same ParmEd path as every other implicit bundle:

```python
system = st.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds,
                         implicitSolvent=GBn2, removeCMMotion=True)
```

`system.prmtop` and `system.rst7` are construction intermediates and provenance for the OpenMM
System. There is no Amber execution engine here.

## Cost

Preparation runs AM1-BCC through AmberTools `sqm` for the whole macrocycle, which takes minutes to
tens of minutes. `--dry-run` validates the input, the configuration and the routing without paying
for it, and the destination check refuses an occupied output directory **before** the
parameterisation rather than after.

## Commands

```bash
# 1. prepare (expensive: AM1-BCC)
python MD_system_gen.py -i test/rgd/implicit/cyclo_rgdfv.smi -o rgd_implicit \
    --config test/rgd/implicit/system_config.json

# 2. generate the protocol
python MD_input_gen.py --system rgd_implicit/system_manifest.json -o rgd_implicit_run \
    --config test/rgd/implicit/md_config.json

# 3. run it
cd rgd_implicit_run && ./run_all.sh
```

## Stage graph

```
rgd_implicit_run/
    inputs/        the prepared system, copied
    min/           restrained minimisation
    eq/            restrained equilibration, 20 ps -- velocities initialised HERE, once
    cMD_1/         unrestrained NVT conventional MD
    REST2_1/       replica exchange, one run containing its segments
    run_all.sh     run_manifest.json     run.log
```

No NPT stage, and the equilibration stage is called `eq` rather than `eq_nvt`: it runs the same
restrained constant-temperature dynamics, but "NVT" fixes a volume this System does not have.

## Exchange derivation

```
duration_per_segment            2 ns      stated
number_of_exchanges_per_segment 200       stated
timestep                        2 fs
steps per segment               1,000,000 (2 ns / 2 fs)
steps per exchange round        5,000     (1,000,000 / 200)
exchange_interval               10 ps     DERIVED
```

Both divisions are exact or refused, never rounded.

## The tau ladder

`s = (1 - tau)^2`, `sqrt(s) = 1 - tau`, linear in tau from 0 to 0.5 over 6 replicas:

```
tau    0.00    0.10    0.20    0.30    0.40    0.50
s      1.00    0.81    0.64    0.49    0.36    0.25
T_eff  300     370     469     612     833     1200    K
```

The whole system is the enhanced region, and a partial selection is refused: a Born radius depends
on every other atom's position, so a partial region would need a validated treatment of the cross
terms. The complete `CustomGBForce` energy scales by `s`, including the non-polar term that charge
scaling alone would miss.

Omega exclusion is enabled. It is torsion-only and is applied after the enhanced region is resolved,
so it never alters GB or nonbonded scaling — it keeps the peptide-bond torsions of ordinary amides
physical so the hot replicas do not manufacture cis/trans isomerisation. For this macrocycle that
matters more than for a linear peptide, because a cis amide changes the ring geometry the whole
calculation is about.

## Extension

`REST2_NUMBER_OF_SEGMENTS` in `run_all.sh` sets how many segments run; it is an execution choice and never
appears in the scientific JSON. Re-invoking `REST2_1/REST2_1.sh` continues the same run from its
committed-generation record, accumulating lifetime exchange statistics.

## Status meanings

`completed` means the requested work for that invocation reached its committed boundary — not that
the ensemble converged. Nothing here is scientific validation.
