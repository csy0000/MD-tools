# Building a system

`md-openmm build-top` turns one input structure into one parameterised OpenMM System. It decides
the **physics** — force field, charges, radii, constraints, solvent — and writes `built.xml`,
which is the Hamiltonian every later command integrates.

```text
md-openmm build-top -i INPUT [-os PATH] [-op PATH] [-log PATH] [--config PATH] [--overwrite]
```

| flag | what it is | default |
|---|---|---|
| `-i` | the input structure: a `.pdb`, `.smi` or `.sdf` **file** | required |
| `-os` | the serialised OpenMM System | `./built.xml` |
| `-op` | the final coordinates and topology | `./built.pdb` |
| `-log` | the readable log, which carries the machine record | `./built.log` |
| `--config` | the build configuration (YAML, `.config` suffix) | built-in defaults |
| `--overwrite` | replace existing outputs instead of refusing | off |

`-i` names a **file**, never an inline string: the input has to be unambiguous and hashable into
the provenance record, so a SMILES typed on the command line is not accepted.

## Three inputs, and what separates them

```text
.pdb   a peptide or protein, with residue names      parameterised by the protein force field
.smi   one molecule, as SMILES                       coordinates GENERATED (ETKDGv3 + MMFF)
.sdf   one molecule, with coordinates                coordinates USED AS GIVEN
```

The format is not a free choice: it follows `solute.kind`, which says what the solute *is*.

| `solute.kind` | `.pdb` | `.smi` | `.sdf` |
|---|:---:|:---:|:---:|
| `peptide` (default) | ✅ | ❌ | ❌ |
| `ligand` | ❌ | ✅ | ✅ |
| `peptide-like` | ❌ | ✅ | ✅ |

Anything else is refused by name, before any output directory is created.

### A peptide, from a PDB

The default. The protein force field parameterises it by residue template, and the small-molecule
force field never touches it.

```bash
md-openmm build-top -i ALA.pdb \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

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

## What a build writes

```text
build/  built.xml           the serialised OpenMM System -- the Hamiltonian
        built.pdb           the topology and final coordinates
        built.solute.pdb    the solute alone, for solute-only trajectories
        built.log           the readable log, ending in the machine record
        built.sdf           the prepared molecule -- MOLECULAR INPUTS ONLY
```

`built.pdb` and `built.xml` are a **pair**: the System's particle order matches the PDB exactly,
and the build refuses to write either if they disagree, re-reading both after placing them.

**`built.sdf` is written for a `.smi` or `.sdf` input and not for a peptide**, and its absence is
itself the signal that the molecular route does not apply. It is not a copy of your input: it is
the prepared molecule, written by one writer for both routes. It is kept because bond orders are
not recoverable from a topology, and three later consumers need them — the ligand route of the
omega classifier, the peptide-like map, and the run-time preflight, which looks for `<system
stem>.sdf` beside the System.

## Refusals

**An input refused for its shape creates nothing at all** — not even the output directory. The
suffix, the `solute.kind` × format rule and the SDF's shape are all checked before anything is
written, because a directory holding a `built.log` is indistinguishable from a build that was
attempted and died halfway. These exit **2**.

Existing outputs are refused the same way, unless `--overwrite` is passed: a build that silently
replaced a System would leave any trajectory already produced against the old one unexplainable.

**A chemistry mismatch discovered during parameterisation is different**, and today it behaves
less well. `solute.kind: peptide-like` over a molecule that is not a peptide gets as far as the
map before failing:

```text
build-top: PeptideMapError: no amide bond was found; this is not a peptide
```

That is a statement about the input just as much as a bad suffix is, but it exits **1** rather
than 2 and leaves a `built.log` in the output directory. See `docs/backlog.md` entry 19 in the
repository. Until it is fixed: a `build-top` that exits 1 has left a directory behind, and the
`.log` in it describes an attempt rather than a System.

## Next

* [Configuration reference](md-configuration.md) — every `build-top` key, generated from the
  schemas: force fields, solvent, box, ions, constraints, hydrogen-mass repartitioning.
* [Scientific defaults](scientific-defaults.md) — why each default is what it is, and the limits of
  the evidence for it.
* [Methods](openmm_methods/README.md) — what to generate next, from the System you just built.
