# Complex leg with CMAP, version 1

The `complex-v1` structure (capped alanine, chain A; ethane `LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4`,
chain B 201) built with ff19SB + OPC: `CMAPTorsionForce` present, OPC's four-site water (180
virtual sites), 750 particles, PME 0.9 nm, HBonds, rigid water, no ions.

OpenMM applies the FIRST NonbondedForce definition it registers to every 1-4 pair, and with
`amber19-all.xml` (an include manifest) + `amber19/opc.xml` that is OPC's rounded
`coulomb14scale="0.833333"`. The ligand's 1-4 exceptions in `built.xml` are therefore at 0.833333,
not the package's 5/6. `built.log` records both (`forcefield_record.ligand.nonbonded_compatibility`),
and the topology builder compares the environment's ligand against package A with its 1-4
exceptions rescaled to that recorded, applied value.

Made on 2026-09-19 with md-tools at `ab69961` (released 0.6.0 with the OPC 1-4 fix merged into
0.7.0), `MD_DATA` an empty temporary root (still empty afterwards), the ethane package reused from
a temporary catalog copy:

```bash
md-openmm build-top -i input/complex.pdb --config build/build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

`built.log` is the build's own record with three machine-specific values replaced by
`<redacted>`: `environment.hostname`, `environment.user`, and the absolute path of the invoked
`md_openmm.py` in `command`. Nothing else differs; its `outputs` sha256 match `built.xml` and
`built.pdb`, which `Environment.from_files` checks.
