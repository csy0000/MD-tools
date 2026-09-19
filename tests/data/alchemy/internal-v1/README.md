# Unique-group internal nonbonded fixture, version 1

One package, added beside `../v1/` without changing it:

| molecule | SMILES | package | residue |
|---|---|---|---|
| n-pentane | `CCCCC` | `LOCAL-OFBQJSOFQDEBGM/param_2ef743cf164f` | PNT |

openff-2.2.1, AM1-BCC from AmberTools sqm (`backend_id: ambertools-sqm`), generated charges.
Atoms `C1 C2 H1 H2 H3 H4 H5` carry the same names as ethane's ethyl in `../v1/`, so the ethyl CORE
map applies; the propyl `C3 C4 C5 H6..H12` is then one unique group attached by C2-C3, with
internal 1-4 exceptions and non-excluded 1-5 pairs -- the terms S0 ruled stay physical at a
dummy group's dummy end, which the v1 groups (Cl1; ethanol's O1-H6) cannot show.

Made on 2026-09-19 at `4ec63bd` with `MD_DATA` set to an empty temporary root (still empty
afterwards), `inputs/PNT.sdf` from RDKit `AddHs`, ETKDGv3 seed 20260919, MMFF-optimised:

```bash
printf "solute:\n  kind: ligand\n  parameters: generate\n" > para.config
md-openmm build-top --parameterize -i PNT.sdf --config para.config \
    -op parameter/PNT.pdb -os parameter/PNT.xml -log parameterize.log --resname PNT
```

The package directory was copied, whole, into `packages/<compound>/<param_id>/`. Nothing is
registered anywhere.
