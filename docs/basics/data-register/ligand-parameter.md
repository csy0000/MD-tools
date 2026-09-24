# Registering a ligand parameter package

A [parameter package](../build-top/parameterization.md) works perfectly well as a directory — any
build can point at it by path. Registering it puts it in the **catalog**, so later builds can find
it by compound id, or find it themselves through `search`.

## Register while parameterising

```bash
md-openmm build-top --parameterize -i TYL.sdf --config parameterize.config \
    -op build/parameter/TYL.pdb -os build/parameter/TYL.xml \
    -log build/parameterize.log --resname TYL --register
```

`--register` copies the verified package into the catalog under
`<compound-id>/<parameter-id>/`.

## Register an existing package

Point a later build at it by path and let that build register it, or copy the package directory
into the catalog root yourself — the catalog is a directory layout, not a database.

## Where the catalog is

Resolved in this order:

1. `ligand_catalog.path` in the build configuration, relative to that file;
2. the machine catalog under `$MD_DATA`.

Both are searched, in that order, and `built.log` records which one supplied the package.

!!! danger "Aliases cannot be added later"
    The catalog identifies a package by its **parameter digest** and its **chemical-state digest**.
    Neither includes the names. So re-registering the same molecule with `aliases` filled in is
    accepted and **does nothing**: the command reports success, the existing package is kept, and
    the names are still missing.

    Set `solute.aliases` before you parameterise. A package registered without them can only ever
    be found by its compound id — never by the name you think of it by.

## Reuse, and proving it happened

```yaml
solute:
  kind: ligand
  parameters: CHEMBL112/param_e932f4c4f371   # or: search, generate, or a path
```

`built.log` records the route — `reused (stated reference)`, `reused (catalog search)` or
`created`. A build that quietly regenerated charges says so in its own record, which is the point:
reuse is a claim the build record either supports or refutes.

Full reference: [ligand parameter packages](../ligand-packages.md).
