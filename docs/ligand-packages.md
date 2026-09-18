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

**What reproducing looks like when it works.** Paracetamol has now been parameterised by three
routes that share no code path: imported from a 0.5.3 System without generating a charge,
generated from an SDF with `--parameterize`, and generated from a SMILES with the conformer
embedded. All three give parameter id `param_e932f4c4f371`, the same `parameters.ffxml` bytes and
charges equal to 0.0 in every atom, while `molecule.sdf` differs between the routes that embed
their own conformer -- coordinates are not part of the identity. That is the behaviour a package
is for, measured rather than assumed.

**It is one compound, not a guarantee.** Paracetamol is small and nearly rigid, which is why it
lands on the same charges every time. The unseeded-conformer warning above is still the honest
statement for a flexible ligand, and it is the reason a package is saved rather than a recipe.

## What a package is

```text
<compound-id>/<parameter-id>/
    molecule.sdf       the exact chemical state; atom i is package atom i
    parameters.ffxml   self-contained OpenMM parameters for that molecule
    metadata.json      identities, charges, provenance and artifact digests
    parameter.config   what a build must match to REUSE these parameters
```

The first three files are required. `parameter.config` is derived from `metadata.json`, so a
package written before it existed -- by a build on an earlier commit, or a copy carried in a build
directory or a reference bundle -- loads exactly as it did, with its criteria derived on the spot.
Such a package is older, not incomplete, and it is searchable like any other.

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

## Finding a package: what a build searches, and what counts as a match

`parameter.config` is the file a search reads. It is YAML, like every configuration here, and it is
DERIVED from `metadata.json` and re-derived on every load, so the two cannot drift.

It states the SCHEMA it was written in, and that decides how it is read:

* the same schema: the stored and derived documents must be equal exactly. A difference means the
  file was edited, and the package is refused -- this is the check that protects a catalog;
* another schema: the file was written in a different vocabulary, which is not the same as
  disagreeing with its package, so it is RE-DERIVED rather than compared. This is what an ABSENT
  declaration already does, and the two now behave alike. A package's `criteria` records
  `declaration_schema` and `declaration_is_current` so a reader can tell.

The alternative -- compare strictly and bump the schema when the shape changes -- was tried and
does not hold: adding one field made every declaration already on disk read as a contradiction,
with the symptom "the parameters do not support this file" for packages that matched perfectly. It declares four
things, and a build reuses the package only if ALL of them match what it needs. A near match is a
difference:

| compared | what it is | what it deliberately ignores |
|---|---|---|
| `topology` | the heavy-atom skeleton: which atoms, bonded how | hydrogens, formal charges, bond orders -- so tautomers and protomers share it |
| `protonation` | the exact chemical state: every hydrogen, formal charges, bond orders, stereochemistry | nothing |
| `charges` | the method, the scheme it resolved to, and the IMPLEMENTATION that ran (`ambertools-sqm`, `openeye`, `openff-nagl`, with the NAGL model file and its digest) | the versions of that software, which are recorded but not compared |
| `forcefield` | the exact small-molecule resource, e.g. `openff-2.2.1` | nothing |

The charge implementation is part of the identity because a method name is not a number: AM1-BCC
through AmberTools' `sqm`, AM1-BCC ELF10 through OpenEye, and NAGL's graph model trained to predict
it are three different results for the same molecule. A package being WRITTEN must say which one
produced it: `import_package_from_system` requires `backend_id`, and creation records it.

**A package written before that was recorded still loads.** Its numbers are whatever they are and
nothing about them changed, so a configuration may name it explicitly and the build proceeds. It is
never the answer to a SEARCH, because what is missing is exactly the thing a search compares. The
two cases are distinguishable in the criteria -- `charges.backend_recorded` is `false` and
`charges.backend_id` is `null` -- and in the build record, which carries `charge_backend_recorded`,
so a reader is never told that reuse of unknown charges was reuse of known ones. The search reports
such a package as considered, with "the package does not record which implementation produced its
charges, so it can be used only by naming it explicitly".

Regenerating such a package is one command and gives the same parameter id, so accommodating them
further was deliberately not done.

Software versions are recorded (`charge_software`) and NOT compared. Requiring them to be equal
would force a regeneration on every AmberTools or toolkit upgrade; a build that cares can read them.

`md_tools.ligands.match.matches` is the one definition of all of this. The catalog search and the
"differs, so parameterise" branch both call it, because two implementations of "the same ligand"
would drift and the drift would be invisible.

The search itself compares against every `parameter.config` under the catalog roots, in order, and
LOADS only the package that matched -- which verifies its parameters, digests and identity in full
before anything reuses it. A catalog entry whose declaration cannot be read is reported as skipped
rather than passed over silently. The report lists every candidate with the reason it did or did
not match, and a build record keeps it.

## Making a package

**With `build-top --parameterize`**, which exists to produce a package and nothing else:

```bash
md-openmm build-top --parameterize -i TYL.sdf --config para.config \
    -op build/parameter/TYL.pdb -os build/parameter/TYL.xml \
    -log build/parameterize.log --resname TYL [--register]
```

`-i` reads a `.sdf`, a `.mol2` or a `.smi`. `.mol2` is accepted only here, because parameterisation
needs a molecular graph with bond orders and that is what several docking and preparation tools
write; a `.smi` states the chemistry and no coordinates, so a conformer is EMBEDDED exactly as
`build-top` embeds one (ETKDGv3 from the run seed, then MMFF), and the log says which route the
coordinates came from. A structure without bond orders, such as a `.pdb`, is refused.

**The conformer is not part of the identity.** A package is identified by its chemical state and
its parameters, so the same molecule parameterised from a `.smi` and from a `.sdf` gets the SAME
parameter id with different `molecule.sdf` coordinates. A conformer can only move the id by moving
the charges, which is a property of the molecule rather than of the input format.

`-op` and `-os` must name files in ONE directory, and that directory is what is written:

```text
build/parameter/  molecule.sdf       the package
                  parameters.ffxml   the package
                  metadata.json      the package
                  parameter.config   the package
                  TYL.sdf            a readable copy of the molecule, named for --resname
                  TYL.pdb            the molecule as a topology (-op)
                  TYL.xml            the molecule ALONE as a serialised System (-os)
```

The first four ARE a package, so the directory registers and a catalog search finds it. The last
three are what a person, a tutorial and a later command line point at; they are declared in the
metadata with their digests, and a package whose copies were modified is refused. `-log` must be
written OUTSIDE the directory, because a package holds nothing it does not declare.

**A package directory moves as a UNIT: all seven files.** The declared copies are part of what the
metadata promises, so copying only the four "package" files leaves the package unloadable --
"the metadata declares the readable copy 'TYL.pdb', which is not there". Copy or move the
directory, never a selection of its contents.

`TYL.xml` is the molecule alone: no solvent, no box, no cutoff, no constraints. It is a parameters
artefact for reading and comparing, not a system to integrate -- a run's Hamiltonian comes from a
`build-top` build that loads this package.

The directory needs no particular name: `build/parameter/` is a perfectly good local package, and
`<compound>/param_<id>/` is only what the CATALOG requires. `--register` is optional and goes
through the same write-once path as `data-register --ligand-package`; it is not a second
registration.

`solute.parameters` applies here too, so a molecule the catalog already holds in this exact state,
with these charges and this force field, is REUSED rather than charged again. What the mode
guarantees is a package directory, not a charge calculation.

The configuration must say `kind: ligand`, and `peptide-like` is refused HERE only. A package
records a chemical state and its parameters; it records no kind, and nothing a build matches
against depends on one. `peptide-like` is a build-time property -- the same force field and the
same charges as `ligand`, plus a validated peptide-chemistry map over the result, which is what
drives the mbondi3 corrections under implicit solvent. So a peptide-like solute is parameterised
with `kind: ligand`, and a `kind: peptide-like` build reuses the package unchanged.

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

`solute.parameters` says where a build's parameters come from, and has three values:

```yaml
solute:
  kind: ligand
  residue_name: TYL
  parameters: search                       # the default
ligand_catalog:
  path: ../catalog          # optional; $MD_DATA/parameters/ligands is searched after it
```

* `search` (the default) looks in the catalogs for a package whose declared criteria match this
  build, REUSES it on a match and PARAMETERISES the molecule on any difference;
* `CHEMBL112/param_e932f4c4f371` reuses exactly that package from the catalogs and searches
  nothing. This is the explicit override: it is how a tutorial or a campaign pins the parameters
  it means to use;
* a PATH to a package directory -- `parameters: ./parameter` -- reuses the package there and
  searches nothing. Absolute, or relative to the configuration file, so a build works from
  `build-top --parameterize` output beside it with no catalog at all. The directory keeps whatever
  name its build gave it;
* `generate` parameterises the molecule whatever the catalog holds.

Either kind of reuse requires the same thing of the input:

the `.smi` or `.sdf` must describe the package's exact chemical state: the same graph with
every hydrogen, charge and bond order, and the same stereochemistry in its coordinates. Anything
else is refused. The atoms are put into package order with package names, the permutation is
recorded, and no charge is computed. `solute.ligand_forcefield` and `solute.ligand_charge_method`
may not be stated alongside `parameters`, because the package brings its own.

`built.log` records the decision under `ligand_packages.attached`: `how` (`reused (catalog search)`,
`reused (stated reference)` or `created`), `matched_on` with the four comparisons, and `search` with
every candidate considered and the reason it did or did not match. A reuse with no record of what it
matched is not evidence, so the record carries it.

Rebuilding the tutorial paracetamol this way reproduces its `built.xml`, `built.solute.pdb` and
`TYL.sdf` byte for byte, and `built.pdb` apart from its date line. It took 7 s; the original build
took 86 s.

## Protein-ligand complexes: `kind: complex`

```yaml
solute:
  kind: complex
ligands:
  # Every copy of a ligand, by residue NAME, taking one package. The common case.
  - {resname: TYL, parameters: CHEMBL112/param_e932f4c4f371}
  # A package that is not in any catalog: a path to its directory, relative to this file.
  - {resname: EOH, parameter: ../build/parameter}
  # Or one named residue, when copies must differ.
  - select: {chain: B, resid: "202", insertion_code: ""}
    parameters: LOCAL-XXXXXXXXXXXXXX/param_...
solvent:
  model: TIP3P
```

An entry says WHICH residues and WHICH package:

* the selector is `{resname: TYL}`, `{chain: B, resid: "201"}`, or any combination of them -- each
  stated key narrows. It may be written directly in the entry or under `select`, but not both.
  **A `resname` selector maps EVERY residue of that name**, which is what several copies of one
  compound need. A stated `resid` still names exactly one residue: two residues answering to one
  number is an ambiguous structure, not an instruction to map both, and it is refused. A selector
  that matches nothing is refused, and two entries naming one residue are refused;
* the package is `parameters: <compound>/param_<id>` (looked up in the catalogs) or
  `parameter: <path to a package directory>` (for a build that has one locally and no catalog).
  Exactly one of the two. A local directory need not be named `param_<id>`; that shape is what the
  CATALOG requires, and `<system>/build/parameter/` is a perfectly good local package.

`-i` is a `.pdb` or a `.cif`. Before anything is written, the build does the following, in order,
and refuses on the first failure:

1. **Packages.** Every `parameters` reference is loaded from `ligand_catalog.path`, then from
   `$MD_DATA/parameters/ligands`; every `parameter` path is loaded from where it points. Both are
   verified.
2. **Selectors.** Each selector names at least one residue, and one whose `resid` is stated names
   exactly one. `chain` is the chain id the file carries; for mmCIF that is the AUTHOR chain
   (`auth_asym_id`), which is what OpenMM reads. The label chain is not accepted in its place.
   `resid` is a quoted string. A structure whose chains reuse an id (an assembly expanded without
   new chain ids) makes a `chain`/`resid` selector ambiguous and is refused; `resname` is the way
   to map every copy deliberately.
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
