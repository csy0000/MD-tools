# 0.6.1 status

Updated 2026-09-18 by the coordinator session (S0). The coordinator owns this file; workers report
in [`handoffs/`](handoffs/) and never edit it.

| | |
|---|---|
| branch | `0.6.1` |
| baseline | `e524e0e` on the 0.6.0 development line (released as `v0.6.0` at `3927105`; the `dev-0.6.0` branch has since been deleted) (the commit carrying the 20260918 instruction) |
| contract commit | `a531ce7` on this branch — [shared contracts](../shared-contracts.md) |
| aims | [AIMS.md](AIMS.md) |
| worker branch | `work/0.6.1-selection` (session S1) |
| release state | **not released, not merged, not tagged.** 0.6.0 beneath it is still under the user's testing |

## Milestones

| | milestone | state | evidence |
|---|---|---|---|
| S1-A | mask parser and resolved selection record | IMPLEMENTED | [S1](handoffs/S1.md) rows 1, 10, 17 |
| S1-B | backbone / sidechain membership and torsion ownership | IMPLEMENTED | rows 2, 8 (ff19SB and ff14SB) |
| S1-C | partial-selection Hamiltonian construction | IMPLEMENTED | rows 3–7: tau 0 and no-selector byte-identical to 0.6.0; PME energy `a²E_SS + aE_SE + E_EE` to 1e-9 on Reference |
| S1-D | ligand instances and exclusion files | IMPLEMENTED against a synthetic mapping record; the registered-package row is BLOCKED (sandbox) | rows 2, 19 |
| S1-E | versioned CMAP rule | IMPLEMENTED | rows 2, 9 |
| S1-F | identity, resume and refusal | IMPLEMENTED | rows 11–15: a ladder run by 0.6.0 code extends; selective states refused by name; exports verify; each integration patch reverted fails its tests |
| S1-G | CUDA explicit-solvent ladders | **VALIDATED** 2026-09-20, card 5 (user-granted, in a window hpREST2 freed): `tests/test_selective_rest2_cuda.py` 12 passed, 0 skipped, `--error-on-skip`, 73 s. All four ladders (backbone, sidechain, ligand, combined) completed on CUDA/mixed, `selection_mode: explicit`, fingerprint `md-tools-hamiltonian-identity/v3`, read from `restart.json`. Exchange energies recomputed on CUDA at stored coordinates: worst |u| 0.00582 kT, worst cross 0.00901 kT, against a 0.05 kT tolerance calibrated beforehand — 3–5× the CPU figures, which is what mixed precision costs. Each test also re-checked that the whole-solute Hamiltonian at the same taus is REJECTED, so no pass is vacuous | [S1](handoffs/S1.md) row 18 |

State values are NOT STARTED, IN PROGRESS, IMPLEMENTED (code and deterministic tests), VALIDATED
(evidence on the required platform), or BLOCKED (with the blocker named). IMPLEMENTED is not
VALIDATED, and neither is released.

## Integration commits

| commit | what | checks |
|---|---|---|
| `e36ff56` | merge S1 at `d013ca9`, with its integration patches in `remd/driver.py`, `run/preflight.py`, `remd/executor.py`, `reference/export.py`, `reference/rest2_export.py`, `remd/protocol.py` | code byte-identical to S1's tested tree; S0 reran the non-CUDA selective tests, slow included, CUDA hidden, `MD_DATA` at an empty temp root: 173 passed. Held: `rest2/regions.py::explicit_selection` unclassified in the CUDA matrix |
| (this merge) | merge S1 at `20f4036`: that site classified non-CUDA | inventory, stale-entry and constructor guards PASS, run directly on the in-tree file |
| `62adec3` | merge released **0.6.0** (`dev-0.6.0` = `main` = `v0.6.0` → `3927105`): $MD_DATA test isolation, alias reporting, `--extend-from` refusal writes nothing, OPC 1-4 scale | fast lane: 2270 passed, 3 skipped, 2 known failures (build-top example; `--check` needs a device). `test_selective_rest2_integration.py`: 8 passed, **1 failed as expected** — `test_an_extension_of_the_0_6_0_run_onto_selective_states_is_refused_on_its_hamiltonian` reads `extended/REST2.out`, which a refused `--extend-from` no longer writes; S1 updates it |
| `4c2002e` (+ handoff `4585d14`) | merge S1: caller-supplied ladder rungs (user decision) — `rung_source` declared with a reason, per-rung origins in `restart.json`, the `rungs` identity key with a legacy compatibility branch, run-time re-derivation refused; `solute.yaml` unchanged | fast lane (CUDA hidden, `MD_DATA` temp root, empty afterwards): 2288 passed, 2 environmental failures; ladder/rung/selective/saved-state files: 46 passed; both inventory guards PASS; S1 shows each of 8 guards failing when removed |

## Blockers

- Open for S0: the compact `L01: <path>` form, which needs an instance name in `md-tools-ligand-mapping/1`. The `--extend-from` defect is fixed in released 0.6.0, merged here (`62adec3`), and S1's extension test now asserts the new behaviour.
- Registration: BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every session
  (2026-09-19); the registration half of S1-G is reported BLOCKED, not passed.
- S1-G: DONE, validated on CUDA (see the milestone row). What remains BLOCKED is the registered-package ligand row and dataset registration (the `$MD_DATA` sandbox rule), and the two TYK2 ladders, which need S2's prepared complex and 4 cards.
- Integration: done. Resume and export of a selective ladder work on `0.6.1`.

## Notes

- No AMBER-mask parser exists anywhere in `src/` today; S1-A is new code, not a wiring job.
- `src/md_tools/rest2/` is identical between `dev` and this baseline, so older notes about those
  files still apply. `src/md_tools/cli/md_openmm.py` and `src/md_tools/openmm/system.py` are
  **not** identical — read them here, not in an older checkout.
