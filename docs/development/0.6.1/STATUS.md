# 0.6.1 status

Updated 2026-09-18 by the coordinator session (S0). The coordinator owns this file; workers report
in [`handoffs/`](handoffs/) and never edit it.

| | |
|---|---|
| branch | `0.6.1` |
| baseline | `e524e0e` on `dev-0.6.0` (the commit carrying the 20260918 instruction) |
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
| S1-G | CUDA explicit-solvent ladders | BLOCKED — no card granted; `tests/test_selective_rest2_cuda.py` (12 tests, one card) is ready. A CPU run of that harness is not CUDA evidence | row 18 |

State values are NOT STARTED, IN PROGRESS, IMPLEMENTED (code and deterministic tests), VALIDATED
(evidence on the required platform), or BLOCKED (with the blocker named). IMPLEMENTED is not
VALIDATED, and neither is released.

## Integration commits

| commit | what | checks |
|---|---|---|
| `e36ff56` | merge S1 at `d013ca9`, with its integration patches in `remd/driver.py`, `run/preflight.py`, `remd/executor.py`, `reference/export.py`, `reference/rest2_export.py`, `remd/protocol.py` | code byte-identical to S1's tested tree; S0 reran the non-CUDA selective tests, slow included, CUDA hidden, `MD_DATA` at an empty temp root: 173 passed. Held: `rest2/regions.py::explicit_selection` unclassified in the CUDA matrix |
| (this merge) | merge S1 at `20f4036`: that site classified non-CUDA | inventory, stale-entry and constructor guards PASS, run directly on the in-tree file |

## Blockers

- Open for S0: `build-md` forwarding of the three selection keys; the compact `L01: <path>` form (needs an instance name in the mapping record); the `--extend-from` log-before-refusal defect, which MD-tools-0.6.0 is fixing on `dev-0.6.0`.
- Registration: BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every session
  (2026-09-19); the registration half of S1-G is reported BLOCKED, not passed.
- S1-G: no GPU is available to this wave. Cards 0–4 are reserved by the user for the 0.6.0 gate
  and 5–8 belong to hpREST2. Blocked is not passed; the deterministic CPU work continues.
- Integration: done. Resume and export of a selective ladder work on `0.6.1`.

## Notes

- No AMBER-mask parser exists anywhere in `src/` today; S1-A is new code, not a wiring job.
- `src/md_tools/rest2/` is identical between `dev` and this baseline, so older notes about those
  files still apply. `src/md_tools/cli/md_openmm.py` and `src/md_tools/openmm/system.py` are
  **not** identical — read them here, not in an older checkout.
