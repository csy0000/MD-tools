# Bespoke torsion refinement in ligand parameterisation — design

**Status 2026-10-13: design, not implemented.** The next release (0.6.5) is about ligand
parameterisation only: Open Babel protonation, non-peptide ω scaling (both done on this branch's
parent) and this. The umbrella release is postponed and nothing from it belongs here.

## The option

```yaml
solute:
  torsion_refinement: none | bespokefit | mace-omol-bespokefit | mace-off23-bespokefit   # none
  bespokefit_env: /path/to/env     # required for `bespokefit` (Psi4 reference)
  mace_env: /path/to/env           # required for the two mace-* options
```

| value | QC reference the torsion drives are run with |
|---|---|
| `bespokefit` | Psi4 B3LYP-D3BJ/DZVP, BespokeFit's own default |
| `mace-omol-bespokefit` | MACE-OMOL-0 (`extra_large`) |
| `mace-off23-bespokefit` | MACE-OFF23 (neutral molecules only) |

Everything else is BespokeFit's default workflow (WBO fragmentation, 15° 1D TorsionDrive,
ForceBalance fit of k1–k4). The base force field is `ligand_forcefield`; BespokeFit refines it.

## What is NOT in `environment.yml`

BespokeFit, QCEngine, Psi4, torch and both MACE models. The user installs them in a separate
environment and names it. `build-top` never imports any of them: it runs a small driver with that
environment's interpreter, as a subprocess, and reads back one `.offxml`. A missing, unreadable or
wrong environment is an error naming the path and what was missing; it never falls back to the
stock force field.

## Contracts

1. **Explicit, never implicit.** A fit takes hours and a GPU. `torsion_refinement` is `none` by
   default and nothing else turns it on.
2. **Identity.** `parameter_id` already hashes the parameter table, so a refined package differs
   from a stock one by construction. The package metadata and the search criteria also carry a
   `refinement` record (`method`, `reference`, base force field, environment package versions,
   model name and weights hash, `.offxml` sha256), so a search for a MACE-OMOL package never
   returns a Psi4 one and a reuse of a refined package never re-fits.
3. **MACE-OFF23 takes no charge or spin.** A charged molecule is refused for it, by name, with the
   two ways out (`protonation.ligand_method: none`, or `mace-omol-bespokefit`). Open Babel's pH 7
   default makes this reachable, so it is stated rather than discovered.
4. **A supplied `.offxml` is allowed.** `torsion_refinement` plus a stated file skips the fit and
   records the file's digest as provenance only, because MD-tools cannot know how it was made.
5. **No new command.** This is `build-top --parameterize` and `build-top` on a molecule.
6. **Tests need no GPU.** The driver contract is tested with a fake environment whose
   interpreter writes a canned `.offxml`; the real fit is a separate, GPU lane and is not claimed
   by the fast lane.

## Reference material (read-only, not a dependency)

`MD-tools-torsionator/bespokefit-mace/run_bespoke.py` and `prototypes/qcengine_mace.py` show how
the QC specification and the MACE harness are wired. MD-tools carries its own driver.
