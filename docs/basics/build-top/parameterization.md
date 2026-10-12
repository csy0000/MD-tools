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

## Protonation comes first

The molecule that is parameterised is the one **at a stated pH**, not whichever protomer the input
happens to spell. Before the charges are calculated, Open Babel's pH model (`obabel -p`) is applied
to the input at `protonation.ph`, **7.0 by default**:

```yaml
protonation:
  ph: 7.0                  # default; also the pH PROPKA uses for a protein
  ligand_method: openbabel # default; `none` parameterises the input exactly as it encodes it
```

| input | at pH 7.0 |
|---|---|
| `CC(=O)O` (acetic acid) | `CC(=O)[O-]` — a carboxylic or sulfonic acid becomes the anion |
| `CCN` (ethylamine) | `CC[NH3+]` — an aliphatic amine, amidine or guanidine becomes the cation |
| `C[C@H](N)C(=O)O` (alanine) | the zwitterion, stereocentre intact |
| paracetamol, benzene, a tertiary amide | unchanged — no rule applies |

It applies to a `.smi`, `.sdf` or `.mol2`, to `--parameterize` and to a `kind: ligand` or
`kind: peptide-like` build alike. It does **not** apply to `kind: peptide` (PROPKA decides there) or
to the ligands of a `kind: complex`, whose hydrogens are their package's and are never changed.

What it does and does not do is worth knowing before you rely on it:

- **It is a table of functional-group rules, not a pKa predictor.** It ignores the molecule's
  environment, so a phosphate, an imidazole or a phenol made acidic by a nitro group is left as it
  came. Where that matters, choose the protomer yourself and set `ligand_method: none`.
- **It moves hydrogens and formal charges and nothing else.** A result whose heavy-atom graph or
  number of specified stereo elements differs from the input is refused, and an SDF or mol2 keeps
  its heavy atoms where they were (checked to 1e-3 Å). When the model changes nothing, the molecule
  you supplied is used as is — a docked pose keeps every hydrogen it arrived with.
- **It chooses no tautomer.**
- **Open Babel missing is an error**, never a quiet fall back to `none`: the build would otherwise
  describe a different species than its record says was asked for.
- **The record says what happened:** `built.log` prints `ligand protonation  Open Babel 3.2.1 at pH
  7.0: net charge +0 -> -1; CC(=O)O -> CC(=O)[O-]` and the machine record carries the per-atom
  changes.

A package is one exact protonation state (see below), so an input Open Babel changes is a
**different species** and takes its own package. A package built earlier from the neutral acid no
longer matches a build of the same `.smi` at pH 7; it is not edited, and it is still there to be
reused by `parameters: <id>` from a molecule that really is that protomer.

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
