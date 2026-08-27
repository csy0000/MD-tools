# GBn2 pairs with ff14SB, not ff19SB

Date: 2026-08-27
Branch: `dev`
Version: `0.4.0.dev0` (unchanged)

A scientific default change, requested on the basis of the literature.

## The problem

The implicit route built every peptide topology with `leaprc.protein.ff19SB`, hardcoded, and the
configuration's `forcefield.protein` was never consulted on that path at all.

ff19SB's amino-acid-specific CMAP corrections were fit **in explicit OPC water**, and no
generalised Born model has been reparameterised against them. GBn2 was developed and validated in
the ff99SB/ff14SB lineage (Nguyen, Roe & Simmerling, *JCTC* 2013). Pairing them combines a backbone
trained in explicit solvent with a solvation model tuned for a different one.

The failure mode is the dangerous kind: nothing errors, the run completes, and the trajectory
silently describes a Hamiltonian nobody validated.

## What changed

**Defaults are solvation-aware.** `sys_defaults()` now returns the force field that matches the
solvation model, in the namespace the builder for that route actually consumes:

| solvation | `forcefield.protein` | `forcefield.water` |
|---|---|---|
| explicit OPC | `amber19-all.xml` (OpenMM XML) | `opc.xml` |
| implicit GBn2 | `leaprc.protein.ff14SB` (tleap resource) | `null` |

Explicit solvent is untouched: ff19SB + OPC is the pairing ff19SB was parameterised for.

**The configuration now reaches tleap.** `build_implicit_bundle_inputs` passes
`forcefield.protein` through instead of letting the hardcoded default win, and that default is now
ff14SB. Verified in the generated `preparation/tleap.in`:

```
source leaprc.protein.ff14SB
```

**The mismatched pair is refused, not warned about.** `resolve_sys_config` rejects an ff19SB-family
protein force field with an implicit GB model, naming both dotted paths, both values, the reason,
and the matched replacement. A warning in a log is not read by whoever reads the trajectory a year
later.

```
forcefield.protein = 'amber19-all.xml' is not parameterised for implicit_solvent.model = 'GBn2'.
  ff19SB's amino-acid-specific CMAP corrections were fit in explicit OPC water, and no GB model
  has been reparameterised against them. ...
  Use the matched pair:
      forcefield.protein: leaprc.protein.ff14SB
```

**`forcefield.json` no longer falls back to ff19SB.** It records whatever tleap was actually given.
`mbondi3` radii are unchanged — they remain the set recommended for GBn2.

## Files changed

`defaults.py`, `config.py`, `implicit.py`, `forcefield_record.py`, `tests/test_config_generation.py`,
`README.md`, this journal.

## Validation

End-to-end implicit build of ACE-ALA-NME:

```
sys.config.yaml   forcefield.protein: leaprc.protein.ff14SB
tleap.in          source leaprc.protein.ff14SB
forcefield.json   protein.tleap_resource  leaprc.protein.ff14SB
                  protein.openmm_resource null
                  implicit_solvent        GBn2 / mbondi3
```

```
pytest -m "not gpu"   121 passed, 38 deselected
pytest -m "gpu"        38 passed, 121 deselected   (CUDA devices 1-4)
```

## Consequences for existing data

**Implicit-solvent results produced before this commit used ff19SB + GBn2.** They are not
invalidated by the change, but the pairing is the one this repository no longer recommends, and the
force field is recorded in each bundle's `forcefield.json` (or, for 0.3.x, in
`resolved_sys.config.yaml`) so affected runs can be identified rather than guessed at. Deciding
whether any given result needs repeating is a scientific judgement, not something this change makes
for anyone.

Explicit-solvent results are unaffected.

## Limitations

* Only the ff19SB/amber19 family is refused. Other mismatched pairs — an older GB model with a
  modern force field, say — are not detected; the check encodes the one pairing that was wrong here,
  not a general compatibility matrix.
* `mbondi3` with ff14SB/GBn2 follows the GBn2 recommendation and was not re-derived here.
* No implicit trajectory was re-run for comparison; this change is to defaults, validation and
  provenance, not a revalidation of the implicit protocol.
