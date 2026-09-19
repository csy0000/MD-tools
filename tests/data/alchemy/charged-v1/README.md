# Equal-nonzero-charge fixture, version 1

| molecule | SMILES | package | residue | net charge |
|---|---|---|---|---|
| acetate | `CC(=O)[O-]` | `LOCAL-QTBSBXVTEAMEQO/param_c565813e02ae` | ACT | -1 |
| propanoate | `CCC(=O)[O-]` | `LOCAL-XBDQKXXYIPTUBI/param_fa700052a552` | PRP | -1 |

`acetate-tip3p/`: acetate in a cube of TIP3P with one Na+ added by build-top to neutralise it
(PME 0.9 nm, HBonds, rigid water, ionic strength 0), 617 particles.

Acetate -> propanoate preserves the net charge, which `validate_map` allows; acetate's
C1 C2 O1 O2 H1 H2 map onto propanoate's C2 C3 O1 O2 H4 H5 (O1 is the double-bonded oxygen in
both), leaving acetate's H3 and propanoate's methyl C1 H1 H2 H3 unique.

openff-2.2.1, AM1-BCC (AmberTools sqm). Made on 2026-09-19 at `8261fe1` with `MD_DATA` set to an
empty temporary root (still empty afterwards); inputs from RDKit `AddHs`, ETKDGv3 seed 20260919,
MMFF-optimised; packages by `build-top --parameterize` (`solute: {kind: ligand, parameters:
generate}`), the environment by `build-top -i input/ACT.sdf --config build/build.config`, with
the package path rewritten here to `../packages/...`. Nothing is registered.
