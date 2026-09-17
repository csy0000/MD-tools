# Ligand parameter packages

One compound, one set of parameters, in every environment it is simulated in. A package holds the
parameters of ONE chemical state of a small molecule as they were generated, once: charges,
Lennard-Jones, bonds, angles, torsions and 1-4 exceptions. Every build that uses the compound
loads that file, whether the environment is water, a protein pocket or anything else `build-top`
builds. Nothing is regenerated, so results from different environments rest on identical ligand
parameters.

This is not only a matter of speed. The OpenFF toolkit computes AM1-BCC charges on a conformer it
generates itself, without a seed (`docs/backlog.md`, entry 6), so a flexible molecule parameterised
twice can get two different sets of charges. Before packages, a ligand build ran AM1-BCC again at
each preparation step: hydrogens, solvent and System.

## What a package is

```text
<compound-id>/<parameter-id>/
    molecule.sdf       the exact chemical state; atom i is package atom i
    parameters.ffxml   self-contained OpenMM parameters for that molecule
    metadata.json      identities, charges, provenance and artifact digests
```

Three identities are kept apart:

| identity | example | what it names |
|---|---|---|
| compound | `CHEMBL112` | the substance. Tautomers and protomers share it. `aliases` (paracetamol, acetaminophen, TYL) are searchable names, not identities. A compound no database lists uses `LOCAL-<first block of its standard InChIKey>`. |
| parameters | `param_e932f4c4f371` | one exact chemical state with one saved parameter set |
| instance | chain B, resid 201 | one copy of a ligand in one structure |

A compound id never authorises reuse on its own. A configuration names a package by its full
reference, `CHEMBL112/param_e932f4c4f371`.

### The chemical state

`molecule.sdf` defines the state exactly: every hydrogen as an explicit atom, formal charges, bond
orders, and stereochemistry (assigned from the 3D coordinates). A molecule with an unassigned
stereocentre is refused. Its identity is recorded two ways:

* `chemical_state.digest`: sha256 of `md-tools-chemical-state/1|` followed by RDKit's canonical
  isomeric SMILES of the explicit-hydrogen graph. That string depends on the RDKit version, which
  the metadata records.
* `fixed_h_inchi`: the fixed-hydrogen InChI, which keeps tautomers apart and is stable across
  toolkit releases. It is checked on every load, whatever RDKit is installed.

Two protomers of one compound are two states and two packages, even though ChEMBL, a standard InChI
or a normalising toolkit would merge them. The coordinates in `molecule.sdf` are a reference
conformer only. They never replace the pose of a ligand in a structure.

**Atom identity.** Package atom i is the i-th atom block of `molecule.sdf`. Atom names are the
record's `MDT_ATOM_NAMES` property: element plus a per-element count (`C1`, `O1`, `H1`, ...) unless
the creator supplied names. Every built topology uses these names, so one compound carries the same
atom names in every environment.

### The parameter identity

```text
parameter_id = "param_" + sha256(canonical_json({"chemical_state": <state digest>,
                                                 "parameters":     <parameter digest>}))[:12]
```

The **parameter digest** is computed from the parameters themselves, not from the file's bytes. The
ffxml is loaded into OpenMM, and a System is created for the molecule alone (no cutoff, no
constraints). Every value is read back in package atom order:

* per atom: element, mass, charge, sigma, epsilon;
* bonds and angles;
* proper and improper torsions, classified from the bond graph;
* 1-4 exceptions;
* the 1-4 scales.

That table (schema `md-tools-ligand-parameters/1`) is serialised as canonical JSON and hashed.
Whitespace or generated names cannot change the identity, and a changed charge always does.
`load_package` recomputes all of it. A directory name, a metadata field or a file digest that
disagrees with the contents is refused.

Atom-type and template names are derived from the parameter id (`MDT_param_e932f4c4f371-7`). Two
packages of the same molecule can therefore load into one ForceField. openmmforcefields' own names
come from the SMILES and would collide.

**Supported force terms** are NonbondedForce, HarmonicBondForce, HarmonicAngleForce and
PeriodicTorsionForce. A molecule whose parameterisation needs anything else (a virtual site, a
CMAP, a custom force, parameter offsets) is refused rather than packaged without it. Only SMIRNOFF
force fields (Sage) can create packages at present. GAFF is refused by name: its ffxml shares atom
classes with `gaff.xml` instead of carrying per-atom parameters, and a per-atom conversion that
keeps the improper ordering has not been validated. A GAFF build keeps its previous route and
writes no package.

**Nonbonded conventions.** A package stores the electrostatic 1-4 scale as exactly 5/6. SMIRNOFF
files write it as `0.8333333333`, and OpenMM silently keeps the first file's value when two agree
within 1e-5, so a ligand loaded after ff14SB would otherwise get slightly different exceptions than
the same ligand built alone. Combining a package with a force field whose 1-4 scales differ, or
that uses a custom Lennard-Jones representation, is refused before the package is loaded.

### The metadata

`metadata.json` holds:

* compound id, aliases and residue name;
* the chemical state;
* every atom's name, element, formal charge and partial charge;
* the force-field resource;
* the charge method, scheme and backend (AmberTools `sqm`, OpenEye, or a NAGL model with its
  sha256), and whether the charges were generated or imported;
* the nonbonded conventions, and any normalisation applied to them;
* the parameter digest;
* software versions;
* provenance;
* the sha256 of `molecule.sdf` and `parameters.ffxml`.

## Making a package

**In a build.** A single-molecule build (`solute.kind: ligand` or `peptide-like`) with no
`solute.parameters` creates a package from its prepared molecule before any force field is built.
The charges are generated once, and every step of that build loads the saved parameters:

```yaml
solute:
  kind: ligand
  residue_name: TYL
  compound_id: CHEMBL112
  aliases: [paracetamol, acetaminophen]
```

The package is written to `build/ligands/CHEMBL112/param_<...>/`, beside `built.xml`, and the
`ligand_packages` block of `built.log` records it with `how: created`.

**From an existing build.** `md_tools.ligands.import_package_from_system` recovers a package from an
UNSCALED built System without computing a charge. It reads the charges off the System's
NonbondedForce and assigns every other term from the force field with those charges fixed. The
result must reproduce every term the System carries for the ligand atoms: Lennard-Jones, exceptions,
angles, torsions and every unconstrained bond. A bond the System constrained must have the
constraint's length. A REST2- or AIS-scaled state fails this comparison by construction and is
refused. Masses are not compared, because hydrogen mass repartitioning changes them; the package
carries the force field's own masses.

The paracetamol of the 0.5.4 AIS tutorial (`tutorial-runs/AIS-0.5.4/PARA/build`, built at
`3da35d0`) was recovered this way from `built.xml` and `TYL.sdf`. The result was
`CHEMBL112/param_e932f4c4f371`. A fresh AM1-BCC package made from the same `TYL.sdf` gets the same
parameter id.

## Reusing a package

```yaml
solute:
  kind: ligand
  residue_name: TYL
  parameters: CHEMBL112/param_e932f4c4f371
ligand_catalog:
  path: ../catalog          # optional; $MD_DATA/parameters/ligands is searched after it
```

The input `.smi` or `.sdf` must describe the package's exact chemical state: the same graph with
every hydrogen, charge and bond order, and the same stereochemistry in its coordinates. Anything
else is refused. The atoms are put into package order with package names, the permutation is
recorded, and no charge is computed. `solute.ligand_forcefield` and `solute.ligand_charge_method`
may not be stated alongside `parameters`, because the package brings its own.

Rebuilding the tutorial paracetamol this way reproduces its `built.xml`, `built.solute.pdb` and
`TYL.sdf` byte for byte, and `built.pdb` apart from its date line. It took 7 s; the original build
took 86 s.

## Protein-ligand complexes: `kind: complex`

```yaml
solute:
  kind: complex
ligands:
  - select: {chain: B, resid: "201", insertion_code: ""}
    parameters: CHEMBL112/param_e932f4c4f371
  - select: {chain: D, resid: "201"}
    parameters: CHEMBL112/param_e932f4c4f371      # a second copy: same package
  - select: {chain: B, resid: "202"}
    parameters: LOCAL-XXXXXXXXXXXXXX/param_...    # a different species
solvent:
  model: TIP3P
```

`-i` is a `.pdb` or a `.cif`. Before anything is written, the build does the following, in order,
and refuses on the first failure:

1. **Packages.** Every `parameters` reference is loaded from `ligand_catalog.path`, then from
   `$MD_DATA/parameters/ligands`, and verified.
2. **Selectors.** Each selector names exactly one residue. `chain` is the chain id the file
   carries; for mmCIF that is the AUTHOR chain (`auth_asym_id`), which is what OpenMM reads. The
   label chain is not accepted in its place. `resid` is a quoted string. A structure whose chains
   reuse an id (an assembly expanded without new chain ids) makes the selector ambiguous and is
   refused.
3. **Coverage.** Every residue that is not a standard protein residue, water or ion must be
   covered by an entry. An unmapped one is refused, never guessed and never deleted.
4. **Graph match.** The residue's heavy atoms must have the package's elements and connectivity.
   Connectivity is perceived from the coordinates (covalent radii + 0.045 nm), since a PDB has no
   bond orders. The deposited atom order and names do not matter.
5. **Symmetry.** Every match is enumerated. Several matches are accepted only when each alternative
   is a symmetry of the package's full chemical graph (a phenyl ring flip) or leaves every package
   parameter unchanged (a carboxylate whose two oxygens are treated identically). Anything else,
   such as the two oxygens of a carboxylic acid, is refused unless the entry adds
   `atom_map: {deposited name: package name}` for every heavy atom.
6. **Stereochemistry.** The package's stereochemistry must be the deposited pose's. A mirror-image
   pose is refused.
7. **Hydrogens.** The package's hydrogens are placed on the deposited heavy atoms by local
   superposition and relaxed alone (heavy atoms fixed, ligand in vacuum, package parameters,
   Reference platform). The heavy atoms are never moved. If the file already has hydrogens on the
   ligand, they are kept only when every heavy atom carries exactly the package's number of
   hydrogens; any other count is a different protonation state and is refused.
8. **Attachments.** A ligand heavy atom within covalent distance of another residue's heavy atom is
   refused: a covalently attached ligand is not a single-residue package. A declared bond to a metal
   is dropped and recorded, and metal contacts within 0.30 nm are listed. Coordination belongs to
   the metal-site model, not to the ligand package.

The rest of the build then runs on the result:

* Protein hydrogens are added to every residue except the mapped instances, which are frozen.
* Solvent and ions are added.
* The System is created with `residueTemplates` naming each instance's template explicitly.

After protonation, after solvation and on the final `built.pdb`, every instance is checked
unchanged: atoms, names, elements, bonds, and positions within PDB precision.

Outputs, beside `built.xml`:

* `ligands/<compound>/<parameter>/`: each package used, copied, so the build does not depend on the
  catalog;
* `ligand_mapping.json`: each instance's selector, deposited residue name, package reference and
  digests, heavy-atom map (package index, package name, deposited name), hydrogen source, symmetry
  verdict, stereochemistry check, contacts and dropped bonds, and its resolved chain, residue and
  atom indices in `built.pdb`.

A metal site, such as the zinc sites of 1TYL prepared with MCPB.py, is **not** a ligand package. It
is a separate artifact with its own residue templates and cross-residue bonded terms. The complex
route does not model metal coordination.

Implicit solvent (`GBn2`) is refused for a complex.

## The catalog

```text
$MD_DATA/parameters/ligands/<compound-id>/<parameter-id>/
```

The catalog is not a registered dataset and does not follow the dataset layout: a package belongs to
a compound and is shared across projects. A build only reads the catalog. Registration is the one
thing that writes to it:

```bash
md-openmm data-register --ligand-package build/ligands/CHEMBL112/param_e932f4c4f371
md-openmm data-register --ligand-package <dir> --dry-run      # check, write nothing
md-openmm data-register --ligand-package <dir> --verify-only  # re-verify the catalog copy
md-openmm data-register --find-ligand paracetamol             # search ids, aliases, SMILES
```

A package is verified before it is registered. Registration is write-once. If the identity is
already present with the same parameter digest and chemical state, the catalog copy is kept; a
creation time or reference conformer may differ, and those are not identity. The same parameter id
with a different digest or state is refused. Nothing is ever replaced.

## Export

`export-reference` copies each package a build used into the bundle's `input/ligands/`, and a
complex's `ligand_mapping.json` into `input/`. Before copying, each is verified against the digest
the build-top record holds, and `SHA256SUMS` lists them.

* **Running the bundle** (route 1) needs OpenMM only. The serialised System is the Hamiltonian that
  ran.
* **Rebuilding a single-molecule system from its structure** (route 2, `input/build_system.py`)
  loads the package's ffxml and applies build-top's recorded atom permutation. It runs with
  md_tools, OpenFF and openmmforcefields all absent, and gives the same System bytes.
* **Rebuilding a complex from its deposited structure** needs md-tools' ligand mapping, so the
  bundle says so rather than shipping a partial script. The System, the packages and the mapping
  are all in the bundle.

## Python API

```python
from md_tools.ligands import create_package, import_package_from_system, load_package
from md_tools.ligands.catalog import find_package, register_package, search_catalog
from md_tools.ligands.mapping import (map_ligands, load_packages_into, assert_instances_unchanged,
                                      unmapped_residues)
```

`MappedStructure` has the following members:

* `frozen_residues` gives `(chain, resid, insertion_code)` for every instance;
* `residue_templates(topology)` gives the `{Residue: template}` mapping to pass to
  `addHydrogens`, `addSolvent` and `createSystem`;
* `resolve(topology)` finds each instance again in a later topology by residue identity and package
  atom names. Instances are never bound to atom indices, which shift whenever a step adds atoms.
