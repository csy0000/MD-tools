# Ligands are protonated at pH 7 before they are parameterised, and a ligand's amide is scaled

**Implemented 2026-10-12, in the working tree; not committed, not released.** Two changes the user
asked for together, plus the environment they need. Each is a decision about what the Hamiltonian
is, so both change results for some inputs; this note says which.

## 1. Open Babel is in `openmm-env`

`openbabel` 3.2.1 was installed from conda-forge into the shared `openmm-env` (the solver also moved
`openssl` 3.6.4 → 3.6.5, and nothing else). It is listed in `environment.yml`. Open Babel was not in
any environment before; the only copy on the machine was a 3.0.0 binary at `/usr/local/bin/obabel`
that the environments cannot see.

## 2. A small molecule is protonated at a stated pH, by default 7.0

```yaml
protonation:
  ph: 7.0                   # default; the same key the protein half uses
  ligand_method: openbabel  # default; `none` keeps the input's protomer
```

`OBMol.AddHydrogens(False, True, pH)` is what `obabel -p` calls, so the CLI and the library cannot
disagree; the record carries the library version. It runs on the molecule as supplied — before the
conformer is embedded, before the charges are calculated, before the package identity is computed —
for `kind: ligand` and `kind: peptide-like`, for `--parameterize`, and for `.smi`, `.sdf` and
`.mol2` alike.

What changes: `CC(=O)O` is now built as acetate, an aliphatic amine as the ammonium, a guanidine as
the guanidinium, at pH 7. What does not: paracetamol, benzene, tertiary amides — nothing in the
table applies, and the molecule you supplied is used as the same object, so a docked pose keeps its
hydrogens.

**Limits, stated because a default is an action.** It is a table of functional-group rules and not
a pKa predictor. Measured with the installed version: a phosphate (`OP(=O)(O)O`) and an imidazole
(`c1c[nH]cn1`) are left as written; a carboxylic acid is ionised and an amine protonated at 7.0, and
reversed at pH 2 and 12. It never changes the heavy-atom graph or the number of specified stereo
elements — a result that does is refused, which the tests exercise with a molecule that does — and
a 3D input's heavy atoms stay within 1e-3 Å (measured 5e-5 Å, the molfile precision). Open Babel
missing is an error naming `ligand_method: none`; it never falls back.

**Refused by name:** `protonation.ligand_method` on `kind: peptide` (PROPKA decides there) and on
`kind: complex` (a mapped ligand's hydrogens are its package's). `protonation.method`, `overrides`
and the histidine keys are still refused for a molecule.

**What this does to existing work.** A package is one exact chemical state, so a `.smi` that Open
Babel now changes no longer matches a package built from the unchanged molecule. That package is
not edited and is still reusable by `parameters: <id>` from a molecule that really is that protomer;
a `search` simply finds nothing and parameterises the new species. Anything built from an input the
model leaves alone is unchanged.

## 3. REST2 convention v4: the ω protection is for peptides

`rest2-unscaled-torsions` goes from v3 to **v4**. An ordinary amide C–N inside a **non-protein
residue** is now *scaled*, and recorded under `ligand_amide_scaled_bonds` (as ureas already are, so
that a decision never looks like a miss). A protein residue keeps the protection, and so does a
residue named in the new scaler key `peptide_like_residues`.

To the scaler a `solute.kind: peptide-like` molecule is one non-protein residue read from an SDF,
which is also what a ligand looks like. So `peptide_like_residues` is **unset by default and read
from the build record**: `build/built.log` carries `resolved_config.solute.kind`, and for
`peptide-like` the solute's non-protein residues keep the protection; any other kind names none. A
stated list wins, an empty one included. With no readable record the answer is "none" and the
scaler's `ligand amides` log line says so and where the list came from
(`peptide_like_residues_source` in `scaler.yaml`). The build record for a `peptide-like` build is
classified the same way.

The rule is "inside one residue". A bond *between* two residues is a peptide bond whatever either
is called: a hydroxyproline in a protein chain still reads bond orders or refuses, as before. The
first version of this change got that wrong — it scaled HYP's backbone — and
`test_proline_like_is_configurable_and_hydroxyproline_can_be_declared` caught it.

To protect one ligand amide anyway: its central bond in `unscaled_list`.

**v3 ladders are continued, not refused.** The identity a ladder stores embeds the convention and is
compared whole on resume and extension, so a plain bump would have made every in-flight v3 ladder
unresumable once this is installed. `rest2.scaler.rest2_implementations_agree` is the one named
compatibility branch: recorded v3, current v4, every other field equal. The states are files, so the
Hamiltonian a v3 ladder integrates is unchanged, and `ReplicaRun.adopt_recorded_convention` keeps
the continuation recording v3 rather than relabelling states it did not build.

**Tutorials.** The paracetamol REST2 and AIS pages and the TYK2 REST2 page record v3 runs and say so
in a notice at the top; their numbers are unchanged and were not regenerated, because that needs the
GPU runs.

## What was checked, and what was not

Run, CPU only, pinned to cores 35–38: the omega, scaler, selective-REST2, identity, ligand,
build-top and configuration test files, plus the whole `not slow and not gpu` lane (see the
handover for its counts). New tests: `tests/test_ligand_protonation.py`, additions to
`test_build_top_protonation.py` and `test_unscaled_torsions.py`; sixteen existing assertions were
migrated, each one a statement that a ligand's amide is protected, and none removed.

**Also run by hand, CPU, GBn2:** a `kind: peptide-like` cyclic peptide (cyclo-GDR) built and passed to
`--rest2-scaler`. With `peptide_like_residues` unset the scaler read the build record and protected
its 3 backbone ω bonds (`from build record built.log (solute.kind: peptide-like)`); with
`peptide_like_residues: []` it scaled all 3. Open Babel left that molecule unchanged, as it is
already ionised.

**Fast lane:** `not slow and not gpu`, pinned to cores 35-38, 2467 tests; the three failures it showed
were this change's own (a regenerated manual, and a pinned digest of the generated `_protocol.py`
that a docstring edit of mine moved -- reverted, because that file is content-addressed), and both
were fixed and re-run. The text in that template's docstring still says "amide omega" without the
peptide qualifier, deliberately.

**Not run:** the `slow` and `gpu` lanes, in particular `test_peptide_like_hamiltonian.py` and the GPU
REST2 lanes that build and integrate real systems under the new rule. No CUDA evidence is claimed
for any of this.
