# The scaler's refusal withholds the picture that would answer it

Proposed for 0.6.3, not implemented. Found 2026-09-29 while measuring the amidate gap.

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

## The fix

Draw the depiction before enforcing, and name its path in the refusal.

1. Call the classifier with `enforce=False` (`openmm/system.py:1388`), draw, then enforce. The
   classifier already returns the unclassified list without raising, so no new code path is needed to
   get the data.
2. Draw unclassified bonds in a second colour. Red currently means "unscaled"; an undecided bond is a
   third state and must not read as either.
3. Name the file in the refusal: `… see <path>/FOL-unscaled.png, where bond 6-8 is drawn in orange`.

## The tension to settle first

A staged directory that is never renamed is the scaler's atomicity guarantee: `build/<method>/` never
appears half-built. Writing a diagnostic into the staging directory and leaving it does not violate
that — the output tree still never appears half-built — but it does leave a directory behind on a
refusal, and the neighbouring runtime rules (`--check` creates nothing; a refused continuation is
read-only) lean the other way.

Three options:

- **(a) Leave the staging directory on refusal and say where it is.** Cheapest, one message change.
  Costs a stale directory per refused attempt, which needs a documented name and a cleanup story.
- **(b) Write the picture only when asked**, e.g. `--explain-unclassified <dir>`. Keeps the
  write-nothing property exactly. Costs the user a second command.
- **(c) Draw into a temporary directory and print the path.** Preserves both properties, litters the
  temporary directory, and the file the user is told to open can be reaped out from under them.

(a) is the smallest change and matches what the scaler already does. (b) concedes nothing to the
surrounding invariants. This needs a decision before the work starts.

## Scope

Self-contained. It does not depend on, and should not be bundled with:

- the amidate SMARTS widening `[#6X3](=[OX1])-[#7X2,#7X3]`, a separate open item that must be
  re-checked against the three widenings the 0.6.2 notes pin as rejected;
- whether the urea-like abstention stays (it is deliberate — `openmm/system.py:1012`);
- making `torsion_exclusions` reachable outside `ligand_scaling_dict`, so a user can *answer* an
  unclassified bond rather than only add protection to one already classified.

## Test to write first

Build the acylsulfonamide anion, run the scaler, and assert that it refuses **and** that the named
picture exists and marks the unclassified bond. That test fails today on the second half only, which
is the whole of this item.
