# Alchemical endpoint fixtures

Version 1 (`v1/`, `FIXTURE_VERSION = "alchemy-endpoints/1"` in `tests/alchemy_fixtures.py`). A
fixture is versioned: if any file here changes, the directory becomes `v2/`, and every result
citing `v1` is re-derived rather than reinterpreted.

## What is here

| path | what it is |
|---|---|
| `v1/inputs/{ETA,CLE,EOH}.sdf` | the three input molecules: RDKit `AddHs`, ETKDGv3 seed 20260919, MMFF-optimised |
| `v1/packages/<compound>/<param_id>/` | three complete parameter package directories (all seven files) |
| `v1/ethane-tip3p/built.{xml,pdb}` | ethane in a 1.9 nm cube of TIP3P, PME 0.9 nm, HBonds, rigid water, no ions |
| `v1/ethane-tip3p/build.config` | the `build-top` configuration that produced it |

| molecule | SMILES | package | residue |
|---|---|---|---|
| ethane | `CC` | `LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4` | ETA |
| chloroethane | `CCCl` | `LOCAL-HRYZWHHZPQKTII/param_5930c10b0577` | CLE |
| ethanol | `CCO` | `LOCAL-LFQSCWFLJHTTHZ/param_d66453ef683a` | EOH |

All three are neutral, openff-2.2.1, AM1-BCC from AmberTools sqm (`backend_id:
ambertools-sqm`), generated charges -- real packages, not test doubles. They share the ethyl atom
names `C1 C2 H1 H2 H3 H4 H5`; ethane's `H6` sits where chloroethane has `Cl1` and ethanol has
`O1` (ethanol's hydroxyl hydrogen is its `H6`). Ethane -> chloroethane is the one-atom
substitution; ethane -> ethanol adds a two-atom dummy group.

## How they were made

At `3fe32b8` (`work/0.7.0-topology`), OpenMM 8.6, openmmforcefields 0.16.0, in a scratch
directory:

```bash
# each molecule: ETA, CLE, EOH
printf "solute:\n  kind: ligand\n  parameters: generate\n" > para.config
md-openmm build-top --parameterize -i ETA.sdf --config para.config \
    -op parameter/ETA.pdb -os parameter/ETA.xml -log parameterize.log --resname ETA

# the environment, reusing the ethane package (build.config in v1/ethane-tip3p/)
md-openmm build-top -i input/ETA.sdf --config build/build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

The package directories were then copied into `<compound>/<param_id>/` catalog layout, whole.
Logs are not committed: they carry machine paths.

**No `$MD_DATA` is involved.** `parameters: generate` searches no catalog (`search: null` in the
log) and the environment names its package by path (`catalog_searched: []`). The whole procedure
was re-run on 2026-09-19 with `MD_DATA` set to an empty temporary root: every file came back
byte-identical except `metadata.json`'s `created_utc`, so the parameter ids, the parameter
digests and `built.{xml,pdb}` are reproduced exactly. Nothing here is registered anywhere.
