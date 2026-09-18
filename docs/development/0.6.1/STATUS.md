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
| S1-A | mask parser and resolved selection record | NOT STARTED | — |
| S1-B | backbone / sidechain membership and torsion ownership | NOT STARTED | — |
| S1-C | partial-selection Hamiltonian construction | NOT STARTED | — |
| S1-D | ligand instances and exclusion files | NOT STARTED | — |
| S1-E | versioned CMAP rule | NOT STARTED | — |
| S1-F | identity, resume and refusal | NOT STARTED | — |
| S1-G | CUDA explicit-solvent ladders | NOT STARTED | — |

State values are NOT STARTED, IN PROGRESS, IMPLEMENTED (code and deterministic tests), VALIDATED
(evidence on the required platform), or BLOCKED (with the blocker named). IMPLEMENTED is not
VALIDATED, and neither is released.

## Integration commits

None yet.

## Blockers

- None recorded. The CUDA ladders in S1-G need GPU allocation agreed with the other sessions and
  with the hpREST2 project before they run.

## Notes

- No AMBER-mask parser exists anywhere in `src/` today; S1-A is new code, not a wiring job.
- `src/md_tools/rest2/` is identical between `dev` and this baseline, so older notes about those
  files still apply. `src/md_tools/cli/md_openmm.py` and `src/md_tools/openmm/system.py` are
  **not** identical — read them here, not in an older checkout.
