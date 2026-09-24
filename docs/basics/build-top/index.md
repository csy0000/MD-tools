# Building a system

`md-openmm build-top` turns one input structure into one parameterised OpenMM System. It decides
the **physics** — force field, charges, radii, constraints, solvent — and writes `built.xml`,
which is the Hamiltonian every later command integrates.

```text
md-openmm build-top -i INPUT [-os PATH] [-op PATH] [-log PATH] [--config PATH] [--overwrite]
```

| flag | what it is | default |
|---|---|---|
| `-i` | the input structure: a `.pdb`, `.cif`, `.seq`, `.smi` or `.sdf` **file** | required |
| `-os` | the serialised OpenMM System | `./built.xml` |
| `-op` | the final coordinates and topology | `./built.pdb` |
| `-log` | the readable log, which carries the machine record | `./built.log` |
| `--config` | the build configuration (YAML, `.config` suffix) | built-in defaults |
| `--overwrite` | replace existing outputs instead of refusing | off |

`-i` names a **file**, never an inline string: the input has to be unambiguous and hashable into
the provenance record, so a SMILES or a sequence typed on the command line is not accepted.

## The inputs, and what separates them

```text
.pdb   a peptide or protein, with residue names      parameterised by the protein force field
.seq   a peptide, as one line of residue names       coordinates BUILT by tleap (extended chain)
.smi   one molecule, as SMILES                       coordinates GENERATED (ETKDGv3 + MMFF)
.sdf   one molecule, with coordinates                coordinates USED AS GIVEN
.cif   protein chains and ligands (kind: complex)    each ligand mapped onto a parameter package
```

The format is not a free choice: it follows `solute.kind`, which says what the solute *is*.

| `solute.kind` | `.pdb` | `.cif` | `.seq` | `.smi` | `.sdf` |
|---|:---:|:---:|:---:|:---:|:---:|
| `peptide` (default) | ✅ | ❌ | ✅ | ❌ | ❌ |
| `ligand` | ❌ | ❌ | ❌ | ✅ | ✅ |
| `peptide-like` | ❌ | ❌ | ❌ | ✅ | ✅ |
| `complex` | ✅ | ✅ | ❌ | ❌ | ❌ |

Anything else is refused by name, before any output directory is created.

### A peptide, from a PDB

The default. The protein force field parameterises it by residue template, and the small-molecule
force field never touches it.

```bash
md-openmm build-top -i ALA.pdb \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

### A peptide, from a sequence

`.seq` holds exactly **one** record: residue names separated by whitespace, as tleap's residue
library spells them. `#` comment lines and blank lines are allowed; a second record is refused
rather than joined or dropped.

```bash
printf '# alanine dipeptide\nACE ALA NME\n' > ALA.seq

md-openmm build-top -i ALA.seq \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

AmberTools' tleap builds the chain with `sequence { ACE ALA NME }`, from the residue library that
matches `forcefield.protein` (`leaprc.protein.ff14SB` for ff14SB, `leaprc.protein.ff19SB` for
ff19SB), and writes it as a PDB. From that PDB on, the build is **exactly** the `.pdb` route —
hydrogens, box, solvent, ions and System under explicit solvent, tleap and GBn2 under implicit —
so every setting that applies to a `.pdb` peptide applies here. `tests/data/ALA.pdb` *is* tleap's
output for this sequence, and the two builds give the same residues, atom names and masses.

**The conformation is extended**: tleap places each residue with its library geometry, and nothing
is sampled or minimised. The log says so, and minimisation and equilibration start from it.

Which residue names exist is tleap's decision, not this command's. A name its library does not
define is refused by tleap — before any output exists — and the end of tleap's log is shown:

```text
build-top: tleap refused the sequence { ACE XYZ NME } under leaprc.protein.ff14SB. ...
    sequence: Illegal UNIT named: XYZ
```

Termini are what you write: `ACE ... NME` caps the chain, while an uncapped chain needs tleap's
terminal units (`NALA ... CALA`). The record keeps the residues, the tleap commands, the digest of
tleap's log and the digest of the PDB it wrote, under `sequence`.

### A small molecule, from SMILES

`.smi` holds exactly one record — a SMILES string, optionally followed by a name. The chemistry is
stated and the **conformer is generated**: ETKDGv3 embeds it from a recorded seed, MMFF94s
minimises every embedding, and the lowest in energy is kept. That seed is what makes the build
reproducible, and it is written into the record.

```bash
printf 'CCO ethanol\n' > ethanol.smi

cat > ligand.config <<'YAML'
solute:
  kind: ligand
  ligand_forcefield: sage-2.2.1
  ligand_charge_method: am1bcc
solvent:
  model: GBn2
YAML

md-openmm build-top -i ethanol.smi --config ligand.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

A file holding several molecules is refused rather than silently built from its first line.

The molecule is **one residue, and it is named**: `solute.residue_name` if the configuration states
one, otherwise a name assigned deterministically from the `.smi` name field or the file stem —
`ethanol` becomes `ETH`. That name is written into `built.pdb`, `built.solute.pdb` and the
serialised topology, and it names the prepared molecule beside the System (`build/ETH.sdf`). A name
that already means water, an ion or a protein residue (`HOH`, `NA`, `ALA`) is refused when stated
and passed over when assigned, because solvent selection and the omega classifier read residue
names; so is a stated name that is not three letters or digits, and any stated name under
`kind: peptide`.

### A small molecule, from an SDF

`.sdf` carries the chemistry **and** the coordinates, and the coordinates are used exactly as
supplied — no embedding, no minimisation. That is the point of the route: a docked pose, a
crystallographic conformer or another pipeline's output is the structure somebody chose, and
replacing it with an MMFF minimum would be a different experiment reported under the same name.

```bash
md-openmm build-top -i ligand.sdf --config ligand.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

`structure.etkdg` and `structure.mmff` are not read on this route. They are not ignored — they
describe a step that does not happen — and the log says so:

```text
Input interpretation
--------------------
  interpreted as              single-molecule SDF
  coordinates                 supplied by the SDF, used as given (no embedding, no minimisation)
```

There is no seed to record, because nothing was sampled. What makes the build reproducible instead
is the identity of the input file, so its SHA-256 is recorded in the seed's place.

An SDF is refused, by name and before anything is created, when it:

* holds **more than one** molecule record — this phase builds one System from one molecule;
* carries **no conformer**, so it supplies no coordinates at all;
* has a conformer flagged **two-dimensional** — a flat molecule is not a starting structure;
* has **no explicit hydrogens** — this route parameterises what it is given and adds none, so an
  implicit-hydrogen file would be built as the heavy-atom skeleton alone;
* cannot be read as an SDF at all, an empty file included.

### `peptide-like`

The same whole-molecule route as `ligand` — same force field, same charges — plus a validated
peptide-chemistry map over the result. It exists for a head-to-tail cyclic peptide, whose residues
are real amino acids but which a single-residue ligand representation cannot describe, so that
residue-keyed corrections such as mbondi3's do not silently miss it. It never loads a protein
force field. A molecule that is not a peptide fails the map — `no amide bond was found` — though
not as cleanly as a bad suffix does; see [Refusals](#refusals).

### A protein with ligands: `kind: complex`

A `.pdb` or `.cif` holding protein chains and ligand instances. Each ligand instance is listed under
`ligands` with a selector (`chain`, `resid`, `insertion_code`) and the parameter package it takes.
The package is loaded as saved and no charge is computed. Every non-standard residue must be listed,
and the ligands' hydrogens come from their packages. See
[Ligand parameter packages](../ligand-packages.md).

A single-molecule build (`ligand`, `peptide-like`) now also works through a package. It either
reuses the one `solute.parameters` names, or creates one from the prepared molecule and writes it to
`ligands/` beside `built.xml`. Either way the charges are computed at most once per build.

### Parameters without a System: `--parameterize`

```text
md-openmm build-top --parameterize -i MOLECULE.{sdf,mol2,smi} --resname NAME \
    -op DIR/NAME.pdb -os DIR/NAME.xml -log LOG [--config PATH] [--register]
```

The third mode of this command, beside the build and `--rest2-scaler`. It writes ONE directory
holding a reusable ligand parameter package and the readable copies named for `--resname`. No
box, no solvent and no System to integrate: see [Ligand parameter packages](../ligand-packages.md).

**The input must carry BOND ORDERS.** `.sdf` and `.mol2` supply coordinates as well; a `.smi`
states the chemistry and the conformer is embedded. A structure alone — a `.pdb`, a `.cif` — is
refused by name, because bond orders cannot be recovered from coordinates.

**`--register` is optional and it is what makes a package reusable by NAME.** It places the
finished package into the machine catalog under `$MD_DATA/parameters/ligands`, through the one
registration path, so a later configuration can name it `<compound>/<parameter>` with no path from
anywhere on the machine. Without it the package is a directory like any other, and a configuration
reuses it by pointing `parameter` at that directory.

## What a build writes

```text
build/  built.xml           the serialised OpenMM System -- the Hamiltonian
        built.pdb           the topology and final coordinates
        built.solute.pdb    the solute alone, for solute-only trajectories
        built.log           the readable log, ending in the machine record
        <RESNAME>.sdf       the prepared molecule, e.g. ETH.sdf -- MOLECULAR INPUTS ONLY
        ligands/            the parameter packages the build loaded -- ligand and complex builds
        ligand_mapping.json each ligand instance's selector, package and atom map -- COMPLEX ONLY
```

`built.pdb` and `built.xml` are a **pair**: the System's particle order matches the PDB exactly,
and the build refuses to write either if they disagree, re-reading both after placing them.

**`<RESNAME>.sdf` is written for a `.smi` or `.sdf` input and not for a peptide**, and its absence
is itself the signal that the molecular route does not apply. It is named for the residue the
topology carries, and that name is also its molecule title. It is not a copy of your input: it is
the prepared molecule, written by one writer for both routes. It is kept because bond orders are
not recoverable from a topology, and four later consumers need them — the ligand route of the
unscaled-torsion classifier (which of the molecule's bonds are aromatic or double), the
peptide-like map, `build-top --rest2-scaler`, and the run-time preflight, which finds it beside
the System by reading the one non-solvent residue name from `<system stem>.pdb`. The record
lists it under `outputs.solute_sdf`, with its digest and `residue_name`.

**Before 0.5.4 this file was `built.sdf`** — `<system stem>.sdf` — and the residue was RDKit's
`UNL` whatever the configuration said. The preflight still reads `<system stem>.sdf` first, so an
older build directory keeps working unchanged. For the same reason `build-top` refuses to write a
new System beside such a file, even under `--overwrite`: this build would not replace it, and it
would be read in place of the molecule the new System was built from. Move it aside first.

## Scaled states: `--rest2-scaler`

The second mode of `build-top` scales a System it already built. It is the ONLY place a REST2
Hamiltonian is made: REST2 ladders, hot cMD stages and AIS integrate the files it writes and scale
nothing themselves.

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config --check
```

```yaml
method: REST2              # REST2, cMD or AIS; also the output directory build/<method>/
schedule: {kind: linear, n_states: 4, tau_min: 0.0, tau_max: 0.5}
unscaled_torsions: true    # default
sdf_filelist: null         # {RESNAME: path.sdf}, relative to this file
```

It writes `build/<method>/system_state<i>.xml`, `scaler.yaml` (the source System's sha256, each
state's tau and sha256, the solute, every unscaled central bond and improper) and
`<RESNAME>-unscaled.png` for each small molecule, with its unscaled torsions' bonds in red. With
`unscaled_torsions: true` a torsion that cannot be classified is refused. `--check` validates and
writes nothing; `--overwrite` moves an existing set aside. The structure flags `-i`, `-os`, `-op`
and `-log` are refused in this mode, and `-s`/`-p`/`--check` without it. See
[REST2](../../openmm_methods/REST2/README.md#the-scaled-states-build-top-rest2-scaler).

## Refusals

**An input refused for its shape creates nothing at all** — not even the output directory. The
suffix, the `solute.kind` × format rule, the shape of an SDF, a `.smi` or a `.seq`, a residue
name that cannot be applied, and tleap's verdict on a sequence are all checked before anything is
written, because a directory holding a `built.log` is indistinguishable from a build that was
attempted and died halfway. These exit **2**.

Existing outputs are refused the same way, unless `--overwrite` is passed: a build that silently
replaced a System would leave any trajectory already produced against the old one unexplainable.

**A chemistry mismatch discovered during parameterisation is different**, and it behaves
less well. `solute.kind: peptide-like` over a molecule that is not a peptide gets as far as the
map before failing:

```text
build-top: PeptideMapError: no amide bond was found; this is not a peptide
```

That is a statement about the input just as much as a bad suffix is, but it exits **1** rather
than 2 and leaves a `built.log` in the output directory. Until it is fixed: a `build-top` that exits 1 has left a directory behind, and the
`.log` in it describes an attempt rather than a System.

## Next

* [Configuration reference](../build-md/configuration.md) — every `build-top` key, generated from the
  schemas: force fields, solvent, box, ions, constraints, hydrogen-mass repartitioning.
* [Scientific defaults](../../scientific-defaults.md) — why each default is what it is, and the limits of
  the evidence for it.
* [Methods](../../openmm_methods/README.md) — what to generate next, from the System you just built.
