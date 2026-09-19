# X-H-only fixture, version 1

| molecule | SMILES | package | residue |
|---|---|---|---|
| methane | `C` | `LOCAL-VNWKTOKETHGBQD/param_7e58d629eea7` | MTH |

Every bond of methane is a C-H bond, so an environment built around it constrains the same pairs
under HBonds and AllBonds and cannot say which policy it was built with. The topology builder
refuses such an environment when endpoint B has a bond between heavy atoms, whose treatment
depends on the answer; this package is what lets that refusal be tested.

openff-2.2.1, AM1-BCC (AmberTools sqm). Made on 2026-09-19 at `d71f0f5` with `MD_DATA` set to an
empty temporary root (still empty afterwards), `inputs/MTH.sdf` from RDKit `AddHs`, ETKDGv3 seed
20260919, MMFF-optimised, by

```bash
md-openmm build-top --parameterize -i MTH.sdf --config para.config \
    -op parameter/MTH.pdb -os parameter/MTH.xml -log parameterize.log --resname MTH
```

with `para.config` = `solute: {kind: ligand, parameters: generate}`. Nothing is registered.
