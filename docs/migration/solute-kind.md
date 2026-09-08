# Migration: `solute.peptide` becomes `solute.kind`

## Nothing you have breaks

Every configuration already on disk keeps working, including every `resolved.config` inside a
generated directory. `solute.peptide` is read and folded into `solute.kind`, and a file with
neither key still builds a peptide, exactly as before.

## The change

| before | now |
|---|---|
| `solute: {peptide: true}` | `solute: {kind: peptide}` |
| `solute: {peptide: false}` | `solute: {kind: ligand}` |
| — | `solute: {kind: peptide-like}` |

`peptide-like` is new and has **no boolean spelling**. It is the ligand route — same
small-molecule force field, same charge method, same builder — with a validated peptide-chemistry
map over the result. Use it for a head-to-tail cyclic peptide built from SMILES whose residues
are real amino acids.

## Why bother

Because residue-keyed corrections cannot reach a solute with one invented residue name. The
concrete case is `mbondi3`, whose Arg/Asp/Glu radius adjustments are selected by residue and atom
name: against a single `UNL`/`RGD` residue they match nothing, the radii silently reduce to
mbondi2, and the build still reports `mbondi3`. With `kind: peptide-like` the same corrections are
applied from the mapped chemistry, and the build records which atoms changed and by how much.

## Rules worth knowing

- **Stating both is allowed only when they agree.** `kind: peptide` with `peptide: true` is
  redundant and accepted; `kind: ligand` with `peptide: true` is refused with a message telling
  you to delete the boolean. `peptide-like` with either boolean is always refused.
- **The alias is resolved before defaults**, so a `peptide: false` you wrote is never compared
  against a `kind: peptide` you did not.
- **One authoritative field.** `kind` decides the route. The derived `peptide` boolean is
  recomputed from it wherever older readers need it and is never a second opinion.

## If your solute is not a cyclic peptide

Keep `kind: ligand`. `peptide-like` refuses anything it cannot describe completely — a linear
peptide, a modified side chain, unresolved stereochemistry — and names the residue in the
refusal, because a map that covered four residues out of five would correct some radii and
silently not others.

See `docs/integration/rgdfv-peptide-like-mbondi3.md` for the measured result on a real system.
