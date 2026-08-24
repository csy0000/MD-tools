# Delegating to OpenMM, and one hash function

**2026-08-24.** Four changes on `dev`, from `9ae5419` to `006799e`. The theme is subtraction: this
package should be a protocol and provenance layer over OpenMM, and several pieces of it had quietly
become a second implementation of things OpenMM already does.

```
8b9177e  refactor: delegate box vectors and hydrogen mass repartitioning to OpenMM
de07510  feat(solvation)!: OpenMM's padding semantics is the default
a8a057f  refactor: one definition of the file hash
006799e  docs(readme): bring it back in line with the code
```

## What was delegated, and how it was checked

Both replacements were verified equivalent **before** the switch, not after.

**Box vectors.** `_box_vectors` was a line-for-line copy of `Modeller._computeBoxVectors`, identical
to 0.00e+00 for cube, dodecahedron and octahedron. It now calls OpenMM's. That helper is *private*,
which is a real trade rather than a free win: the copy survives as `_box_vectors_fallback` and
`test_box_vectors_match_openmm` asserts the two still agree, so if upstream moves it the failure is
loud and the fallback keeps the build working.

**Hydrogen mass repartitioning.** `createSystem(hydrogenMass=...)` does the same arithmetic and skips
residues it made rigid, so with `rigidWater=True` its behaviour *is* the "solute" scope this package
wants. Measured on a solvated alanine system: 3,450 masses identical to 0.000e+00 amu, same total.
The duplicate loop is gone. What remains is `verify_hydrogen_mass_repartitioning`, which checks the
three things OpenMM does not:

* every eligible hydrogen actually reached the target;
* water was left alone;
* no heavy atom was stripped to 1 amu or below -- OpenMM will do that silently if the target is too
  high for a methyl, and a nearly massless carbon integrates happily and wrongly.

## What was not delegated, and why

`addSolvent(padding=...)` cannot size the box, because it knows nothing about the nonbonded cutoff.
Tested at this package's own defaults -- 1.2 nm padding, rhombic dodecahedron, 1.0 nm cutoff:

```
addSolvent(padding=1.2, boxShape='dodecahedron') -> width 2.400 nm, min-image 1.697 nm
Context FAILED: NonbondedForce: The cutoff distance cannot be greater than half the periodic box size.
```

So the `grow`/`refuse` policy and the geometry provenance stay. `dcdtail` stays too -- nothing in
OpenMM reads DCD records or truncates on a record boundary -- as does the committed-generation
contract, which has no OpenMM counterpart.

One candidate was rejected on inspection rather than adopted for tidiness. `_kabsch_rmsd` looked
replaceable by `openmm.RMSDForce`, but that class is documented as a biasing collective variable for
use with `CustomCVForce`: using it for a diagnostic number would mean building a `System` and
`Context` per evaluation. Eight lines of numpy is the better answer, and saying so is more useful
than a delegation that makes the code worse.

## The default is now OpenMM's padding semantics

`solvation.padding_semantics` defaults to `"openmm"`, reproducing `Modeller.addSolvent`: a
bounding-sphere radius about the centre of the solute's axis-aligned bounding box, then
`width = max(2*radius + padding, 2*padding)`. A box built here is now the box OpenMM would build.

`"solute-image-gap"` remains available and unchanged. The two differ in any non-cubic box, because
OpenMM's padding sizes the **width** while the minimum image is `width/sqrt(2)` for a dodecahedron --
so `padding = 1.2` there delivers roughly 0.4 nm of real clearance, not 1.2. That gap is why the
other semantics was written; making OpenMM's the default is a decision to match upstream rather than
to reinterpret it, taken deliberately and not as a side effect of the refactor.

Measured consequence: under `"openmm"` the box is usually sized by `cutoff_fit_policy` rather than by
padding, since the grow step enlarges it until the minimum image clears `2 * cutoff` by
`minimum_image_margin_nm`.

| system | radius | openmm | solute-image-gap |
|---|---|---|---|
| alanine | 0.473 nm | 2.970 nm | 2.970 nm |
| cyclo-(RGDfV) | 0.802 nm | 2.970 nm | 3.965 nm |

For alanine they coincide. For the macrocycle the OpenMM box is smaller and leaves a 0.05 nm margin
before OpenMM's hard minimum-image abort, so **a run that dies with that error needs a larger margin,
not larger padding.** Recorded in the default's own comment, where someone changing it will read it.

## A real error found by a new test

The delegation test asserted the minimum-image factor for each shape and caught that the truncated
octahedron's is **sqrt(6)/3 = 0.8165**, not `sqrt(3)/2 = 0.8660` -- from the smallest diagonal
element of OpenMM's reduced vectors `(w,0,0)`, `(w/3, 2sqrt2 w/3, 0)`, `(-w/3, sqrt2 w/3, sqrt6 w/3)`.
The wrong value overstates the minimum image by 6%, which would let through a box OpenMM rejects.

Here it was only in a test expectation. The same mistake was **live** in the sibling analysis
prototype's box check and is fixed there too. No result was affected: every system in the current
campaign is a rhombic dodecahedron.

## One definition of a content hash

The same eight-line function existed seven times across seven modules under five names --
`sha256_file`, `_sha256`, `_file_sha256`, `sha256_of`, and a wrapper that already delegated -- plus
four places hashing text or bytes inline with their own encoding choice.

They agreed. That was verified through all seven entry points on four files before anything changed
and again afterwards, and the goldens are unchanged, which for a refactor touching hashing is the
result that matters. The risk was never disagreement today; it was that one copy would later acquire
a different block size or encoding, and two bundles would then disagree about a file for a reason
nobody could find. A package whose entire identity story rests on content hashes should not have
seven places to introduce that.

`hashing.py` holds `sha256_file`, `sha256_text` and `sha256_bytes`. Every public spelling survives
and delegates, so no caller changed. `seeds.py` is the one deliberate exception -- it needs the raw
digest *bytes* to derive an integer seed, a different operation rather than another copy -- and a
test asserts it is the only remaining direct user of `hashlib.sha256`, so the duplication cannot grow
back quietly.

## README

"What is here" described a source tree that no longer exists: all four paths it named were removed
several refactors ago, and its "23 modules" is now 50. Replaced with the real layout.

Added the section this work made obviously missing -- **Relationship to OpenMM** -- tabulating what
OpenMM does and naming the three things deliberately not delegated, each with the measured reason.
Added box geometry and padding, documented nowhere before despite now being an OpenMM-semantics
default. Updated the cMD section to the version-3 contract. Marked the wheel-provenance paragraph
historical: "all nine modules byte-identical" was true at hand-off and says nothing about a tree that
now has 34.

Every path and command the README names was checked to exist -- one dead link to a
`HANDOFF_PACKAGE.md` that does not exist was found and replaced -- and the documented defaults were
read back from the code rather than trusted.

## State

864 fast tests pass, 1,030 collected. Goldens unchanged by both refactors, which is the point; the
`config_defaults.json` fixture changed once, for the padding default, as it should have.

The full slow suite has not been run since these four commits. The fast suite and goldens are clean,
and CI runs the full gate on push.

---

## Correction, 2026-08-24 (later the same day)

**The geometry described above conflates two different quantities, and one statement in it is
wrong.** The record is left as written and corrected here rather than edited, because it is what was
believed at the time and the error is instructive.

Where this entry says *"the minimum image is `width/sqrt(2)` for a dodecahedron"* and treats that as
the distance from the solute to its periodic copy, it is describing the **minimum reduced-box
height** -- `min(a_x, b_y, c_z)`, which is the quantity OpenMM's cutoff legality check uses. It is
not how far a point sits from its own image.

The **shortest nonzero lattice translation** is what governs periodic-copy separation, and by
enumeration over integer lattice combinations it equals the box **width** for all three shapes
OpenMM builds -- cube, rhombic dodecahedron and truncated octahedron alike. So the conservative
solute-image clearance is

```
shortest_lattice_translation - 2*R          not      min_reduced_box_height - 2*R
```

The two differ by 29% in a dodecahedron. On the 2026-08-21 campaign's macrocycle boxes that was the
difference between an apparent 0.7 nm clearance and an actual 2.4 nm one, and it produced a false
verdict that a valid 850 ns trajectory was unusable. The measured nearest solute-image *atom*
distance in that run was 2.134 nm, against a lattice-translation bound of 2.361 nm -- consistent, and
irreconcilable with the height-based figure.

The correction, the separated quantities and their provenance fields, and eight tests that establish
each independently are in `2026-08-24_ala-rgdfv-1us-cmd-2nm.md`.

One further clarification to this entry: it says the default was changed to OpenMM's padding
semantics, which remains true, but should have said that **OpenMM has no numeric padding default of
its own**. The numeric value is this repository's choice; only the semantics are OpenMM's.
