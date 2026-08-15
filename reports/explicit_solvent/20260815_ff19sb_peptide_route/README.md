# The ff19SB peptide route, exercised for the first time — 2026-08-15

Both force-field routes were implemented; only one had ever been run. This report is the record of
running the other, and of the defect that had been sitting behind it.

| | Sage 2.2 ligand route (`smiles`) | ff19SB peptide route (`pdb`) |
|---|---|---|
| before this run | exercised: CPU, CUDA, RGD prepare, ten-rung REST2 | **never run end to end** |
| shipped systems | `cyclo_rgdfv`, `small_macrocycle_smoke` | none |
| tests | behaviour + guards | **guards only** |

The three peptide tests — `test_peptide_route_cannot_silently_become_a_sage_ligand_route`,
`test_pdb_hash_mismatch_is_rejected`, `test_rgd_config_rejects_a_pdb_route_before_parameterisation`
— all assert *refusals*. They prove ff19SB cannot be reached by accident. None proves it can be
reached on purpose.

## The system

ACE-ALA-NME (alanine dipeptide), 22 atoms, neutral, built by AmberTools:

```bash
source leaprc.protein.ff19SB
pep = sequence { ACE ALA NME }
savepdb pep ace_ala_nme.pdb          # sha256 dd50bd9cb687…
```

Chosen because every residue is one ff19SB recognises, so the route resolves to `peptide` from the
input itself rather than from a system preset, and CMAP is exercised. It is a **route test, not a
science system** — alanine dipeptide in explicit water says nothing about macrocycle ladders.

## What the first run found

`prepare` completed everything and then produced a bundle that could not be read back:

```text
manifest error: …/system.yaml: input.pdb does not exist:
                …/20260815T214138Z__ace_ala_nme__bundle__dad2e5f8f0ff/ace_ala_nme.pdb
                                                                              (exit 3)
```

The physics was never the problem. ff19SB parameterisation, CMAP, solvation, minimisation, NVT and
NPT all completed — the equilibration trace is in `logs/ff19sb_smoke.log`. The failure is in
packaging: the `smiles` route carries its input **inline** in the manifest, while the `pdb` route
**points at a sibling file**. `bundle.py` copied `system.yaml` into the bundle and nothing else, so
`input.pdb` — resolved relative to the manifest, which now lives inside the bundle — pointed at a
file that was not there.

That breaks the bundle the moment anything re-validates it, which `validate-bundle` does explicitly
and `rest2 --bundle` does before it runs. **The peptide route could build a system and never use
one.** The shipped wheel `10809c7` contains the identical defect, so ff19SB was unusable in every
version of the package, not only in this source tree.

## The fix, and the result

The structure is now copied into the bundle under its declared name, so both the relative path and
`input.pdb_sha256` still resolve. After it:

| check | result |
|---|---|
| `smoke` exit code | **0** |
| `status` | `completed`, 4/4 exchange rounds |
| `validate-bundle` | **passes** — the bundle now contains `ace_ala_nme.pdb` |
| recorded `protein_forcefield` | `amber19/protein.ff19SB.xml` |
| recorded `small_molecule_forcefield` / `charge_method` | `null` / `null` — no Sage, no AM1-BCC |
| composition | 883 atoms (22 solute, 287 waters, 0 ions), 1773 DOF |
| config hash | `dad2e5f8f0ff` |

`ace_ala_nme` now ships as a first-class system, and `manifests/systems/*.pdb` was added to
package-data — without that the manifest travels into the wheel while its structure does not,
reproducing the same bug through a different door.

## What this does and does not establish

**Does:** the ff19SB peptide route parameterises, solvates, equilibrates, produces a self-contained
bundle, and runs REST2 exchanges. Both routes now have a worked example.

**Does not:** anything scientific. Three rungs over 2 ps on a 22-atom solute is a mechanical check.
No ladder is validated for any peptide, the 2 fs / 4 fs gate is still open, and restart safety and
convergence remain unestablished. `smoke.yaml` is `ladder_status: unvalidated` by design.

## Files

```text
logs/build.leap, tleap.log        how the structure was made (0 errors, 0 warnings)
logs/validate_system.log          manifest accepted on the pdb route
logs/ff19sb_smoke.log             the FAILING first run (exit 3)
logs/ff19sb_smoke_fixed.log       the passing run after the fix
bundle/                           the peptide bundle's manifests, config and its structure
run/                              status, resolved config and exchange attempts
```
