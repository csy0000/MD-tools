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

**No CMAP.** The intended fixture was ff19SB + OPC, which carries CMAP. `build-top` refuses it in
`ligands.mapping.check_forcefield_compatibility`:

```text
MappingError: package LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4 uses 1-4 scales coulomb
0.8333333333333334 / LJ 0.5, the force field it is combined with uses 0.833333 / 0.5.
```

The rounded `0.833333` is NOT ff19SB's: `amber19/protein.ff19SB.xml` writes the exact
`0.8333333333333334`, and the rounded value is `amber19/opc.xml`'s (also `amber14/opc.xml`,
`opc3.xml`). `openmm.build_forcefield` calls `ForceField("amber19-all.xml", "amber19/opc.xml")`
-- protein first -- but `amber19-all.xml` is a manifest of `<Include>`s, and OpenMM parses the
included files after the remaining top-level file, so OPC's `NonbondedForce` is registered first
and its rounded scale wins (measured with OpenMM 8.6.0.dev-c6173db: `ForceField("amber19/protein.
ff19SB.xml", "amber19/opc.xml")` gives 0.8333333333333334, `ForceField("amber19-all.xml",
"amber19/opc.xml")` gives 0.833333). The 0.6.0 ligand check compares that exactly with a
package's 5/6. The fix belongs to 0.6.0's ligand module and is not worked around here; until it
lands the complex leg is exercised with ff14SB, which has no CMAP.

## Build record (added 2026-09-19)

`built.log` is that build's record, reproduced on released 0.6.0 (`ab69961`) with
`MD_DATA` an empty temporary root: the rebuild gave byte-identical `built.xml` and `built.pdb`
; `ligand_mapping.json` was replaced by the rebuild's, which 0.6.0 writes with an empty `aliases` field per package (nothing here reads it). Three machine-specific values are replaced by `<redacted>` -- `environment.hostname`,
`environment.user`, and the absolute path of the invoked `md_openmm.py` in `command` -- and
nothing else differs. `Environment.from_files` reads the applied 1-4 scales from it and checks its
`outputs` sha256 against the two files.
