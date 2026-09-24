# Parameterising a ligand

A protein is built from residue templates a force field already contains. A small molecule is not:
its parameters have to be produced, once, and then **reused**. `build-top --parameterize` exists to
produce them and nothing else.

## The command

```bash
md-openmm build-top --parameterize -i TYL.sdf --config parameterize.config \
    -op build/parameter/TYL.pdb -os build/parameter/TYL.xml \
    -log build/parameterize.log --resname TYL
```

| flag | |
|---|---|
| `-i` | `.sdf`, `.mol2` or `.smi`. A `.pdb` is **refused**: parameterisation needs bond orders, and a PDB has none |
| `--resname` | the residue name the molecule gets, three or four characters |
| `-os` / `-op` | the serialised System and the structure that matches it |
| `-log` | the build record: what ran, what it produced, and every digest |
| `--register` | also place the package in the catalog (see [registering a package](../data-register/ligand-parameter.md)) |

`parameterize.config` states the chemistry, not the geometry:

```yaml
solute:
  kind: ligand
  compound_id: CHEMBL112                       # or omit, and a LOCAL- id is derived
  aliases: [paracetamol, acetaminophen]        # searchable names -- see the warning below
parameters:
  small_molecule_forcefield: openff-2.2.1
  charge_method: am1bcc
```

## What comes out

```text
<compound-id>/<parameter-id>/
    molecule.sdf       the exact chemical state; atom i is package atom i
    parameters.ffxml   self-contained OpenMM parameters for that molecule
    metadata.json      identities, charges, provenance and artifact digests
    parameter.config   what a later build must match to REUSE these parameters
```

That directory is a **parameter package**, and it is the unit of reuse: a complex build, a solvent
box and a vacuum leg all point at the same package, so the ligand's intramolecular Hamiltonian is
identical in each.

## Identity is chemistry, not coordinates

The parameter id is a digest of the **parameters**, not of the conformer. Paracetamol has been
parameterised by three routes that share no code path — imported from an older System, generated
from an SDF, generated from a SMILES with an embedded conformer — and all three land on
`param_e932f4c4f371` with identical `parameters.ffxml` bytes, while `molecule.sdf` differs between
the routes that embed their own conformer.

That is one compound and not a guarantee. AM1-BCC is computed on one conformer, so for a **flexible**
molecule two conformers can give different charges and therefore a different package. That is the
system working — different numbers, different package — and it is exactly why a package is saved
rather than a recipe.

!!! warning "Set the aliases before you parameterise"
    `solute.aliases` defaults to empty, and a package's aliases cannot be changed afterwards: a
    re-registration with new names is accepted and **inert**, because the catalog compares the
    parameter and chemical-state digests and neither includes the names. A package made without
    aliases can only ever be found by its compound id.

## Reusing it

In any later build, `parameters` takes one of four forms:

| value | meaning |
|---|---|
| `./build/parameter` | a path, relative to the configuration. Nothing needs to be registered |
| `<compound-id>/<parameter-id>` | that exact package from the catalog |
| `search` (default) | find a package whose recorded criteria match this build; parameterise on any difference |
| `generate` | parameterise regardless of what the catalog holds |

`built.log` records which route was taken — `reused (stated reference)`, `reused (catalog search)`
or `created` — so a build that silently regenerated charges is visible in its own record.

Full reference: [ligand parameter packages](../ligand-packages.md).
