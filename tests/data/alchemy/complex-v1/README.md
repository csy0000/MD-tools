# Complex-leg fixture, version 1

Capped alanine (ACE-ALA-NME, chain A, from `tests/data/ALA.pdb`) and one ethane (chain B, resid
201, package `LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4` from `../v1/`), solvated in a cube of
TIP3P: ff14SB, PME 0.9 nm, HBonds, rigid water, no ions, 582 particles.

| file | what |
|---|---|
| `input/complex.pdb` | the structure given to build-top: ALA.pdb as chain A, ethane heavy atoms as HETATM chain B 201 |
| `build.config` | the `kind: complex` build configuration (catalog path rewritten to `../v1/packages`) |
| `built.xml`, `built.pdb`, `ligand_mapping.json` | what build-top wrote |

Made on 2026-09-19 at `7d48140` with `MD_DATA` set to an empty temporary root (still empty
afterwards). The ethane package was reused from a TEMPORARY catalog directory holding a copy of
the v1 package -- no registered package, nothing from a real catalog -- with

```bash
md-openmm build-top -i input/complex.pdb --config build/build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

**No CMAP.** The intended fixture was ff19SB + OPC, which carries CMAP. `build-top` refuses it:
ff19SB's XML states `coulomb14scale="0.833333"`, packages store the exact 5/6, and
`ligands.mapping.check_forcefield_compatibility` compares the two exactly:

```text
MappingError: package LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4 uses 1-4 scales coulomb
0.8333333333333334 / LJ 0.5, the force field it is combined with uses 0.833333 / 0.5.
```

So no ligand package can currently be built into an ff19SB system, and the complex leg is
exercised with ff14SB, which has no CMAP.
