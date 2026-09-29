# The scaler's refusal withholds the picture that would answer it

**Implemented 2026-09-29.** Found while measuring the amidate gap. The sections below state the
defect and the measurement; *What was built* at the end records what shipped and which of the
options was taken.

## What happens

`build-top --rest2-scaler` refuses an unclassified torsion and writes nothing at all:

```text
src/md_tools/build/scaler.py:601   except UnclassifiedTorsionError: raise ConfigError(...)
src/md_tools/build/scaler.py:722   depictions = depict_unscaled_torsions(...)
```

The refusal is raised 121 lines before the picture is drawn, and the output directory is staged and
renamed into place only on success. A refused build therefore leaves no `<RESNAME>-unscaled.png`, no
`scaler.yaml`, nothing.

## Why that is the defect

The refusal names a bond by topology atom index:

```text
bond 6-8: FOL0 C7 -> FOL0 N1: RDKit found no ordinary-amide match for this C-N bond in the SDF
```

The only artefact that maps such an index onto an atom a reader can see is the depiction, which
annotates every atom with its topology index (`scaler.py:341`, `scaler.py:347`, `scaler.py:447`,
`atomNote` = `topology_index`) and captions the unscaled bonds as index pairs. That file is exactly
what someone needs in order to answer the refusal — and the refusal is what prevents it from being
written.

So a user is told which bond is undecided and given no way to find out which bond that is, short of
writing their own RDKit script against the SDF. This, not the refusal itself, was the dead end in the
folate case: the refusal was correct, and unanswerable.

The refusal should stay. Making it answerable is a separate change from weakening it, and this item
is only the first.

## Measured

Four molecules through `unscaled_torsions(..., enforce=True)`, CPU only. Reproducible from these
SMILES, embedded with RDKit and typed through OpenFF:

| molecule | SMILES | charge | classifier |
|---|---|---|---|
| acylsulfonamide | `CC(=O)NS(=O)(=O)c1ccccc1` | 0 | ACCEPTED, unscaled `[(1,3), ring…]` |
| acylsulfonamide | `CC(=O)[N-]S(=O)(=O)c1ccccc1` | −1 | REFUSED — `bond 1-3 … no ordinary-amide match` |
| hydantoin | `O=C1NC(=O)NC1` | 0 | REFUSED — `carbon 3 is bonded to 2 nitrogens (urea-like)` |
| hydantoin | `O=C1[N-]C(=O)NC1` | −1 | REFUSED — urea-like ×2, plus the amidate bond |

Every one of those refusals produced no picture.

The neutral hydantoin row matters most. It is not an exotic protonation state; it is a plain drug
scaffold (phenytoin, phenobarbital), refused for a reason that has nothing to do with charge, and the
user is handed a bond index with nothing to look at.

## The tension, and why it dissolved

A staged directory that is never renamed is the scaler's atomicity guarantee: `build/<method>/`
never appears half-built. Three placements were weighed — leave the staging directory behind and
name it; an opt-in `--explain-unclassified` flag; a temporary directory — and each traded either
usability or the surrounding rules (`--check` creates nothing; a refused continuation is read-only).

None was needed. The context a reader actually wants is textual, and putting it in the message
costs no file at all. The picture then has no reason to live in the output tree: it belongs beside
the System, next to the SDF it is drawn from, where it neither pollutes `build/<method>/` nor can
be reaped. See *What was built*.

## Still outstanding

**The undecided bond is not drawn in its own colour.** Red currently means "already classified",
and the message says so, so the picture shows what *was* decided while the undecided bond has to
be found by its index annotation. Giving it a third colour was in the original plan and was not
done; red must keep meaning exactly one thing, so this needs a palette decision rather than a
quick addition.

## Scope

Self-contained. It does not depend on, and should not be bundled with:

- the amidate SMARTS widening `[#6X3](=[OX1])-[#7X2,#7X3]`, a separate open item that must be
  re-checked against the three widenings the 0.6.2 notes pin as rejected;
- whether the urea-like abstention stays (it is deliberate — `openmm/system.py:1012`);
- making `torsion_exclusions` reachable outside `ligand_scaling_dict`, so a user can *answer* an
  unclassified bond rather than only add protection to one already classified.

## What was built

Neither (a), (b) nor (c): the tension they were arguing over turned out to be avoidable. The
refusal now carries the **local environment in the message itself**, which needs no file at all —

```text
  bond 2-1: URE0 C -> URE0 N: carbon 2 is bonded to 2 nitrogens (urea-like)
      C [2] bonded to N [1], O [3], N [4]
      N [1] bonded to C [0], C [2], H [9]
        already classified here: N1-C3 aromatic_ring; N1-C5 aromatic_ring
```

Everything there is already computed at the point of refusal: the bond graph, and what the other
rules decided about the neighbourhood. It cannot be reaped, needs no flag, and works over ssh and
in a CI log where an image helps nobody. The `already classified here` line is what makes an
overlap legible — on N-acetylimidazole it says immediately that the nitrogen is an aromatic ring
atom, which is *why* the amide pattern cannot reach it.

Every atom carries its INDEX as well as its name, because a small molecule's atoms are routinely
named after their element and `C bonded to N, O, N` identifies nothing. The index is what the rest
of the message quotes and what the depiction annotates.

**And the picture is drawn after all**, beside the System where the SDF it is drawn from already
lives — not into the staged output directory, which is what made option (a) awkward:

```text
Every atom in these is annotated with the topology index this message quotes; red bonds are the
ones already classified:
  <build>/URE-unscaled.png
```

It never overwrites an earlier one: `URE-unscaled.png`, then `URE-unscaled.1.png`, and so on. A
refusal is something you iterate on, and the previous attempt's diagnostic is what you compare
against. The suffix goes *before* the extension so every copy is still a `.png` a viewer opens.

Drawing can never replace the refusal it explains — every failure inside it is caught and reported
as prose, because a molecule with no usable SDF is exactly the case that refuses, and an exception
raised while explaining an exception hides the real one.

The output tree is untouched: a refusal still creates no `build/<method>/`, which a test asserts.

## Tests

`tests/test_rest2_scaler.py`: the picture is drawn beside the System and named in the message; a
second refusal writes `.1.png` and leaves the first bytes intact; the no-SDF case reports prose and
draws nothing; and end to end through `build_scaled_states` the refusal still creates no output
directory.
