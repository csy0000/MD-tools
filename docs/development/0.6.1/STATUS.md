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
| S1-A | mask parser and resolved selection record | IN PROGRESS | `work/0.6.1-selection` (local), handoffs/S1.md |
| S1-B | backbone / sidechain membership and torsion ownership | NOT STARTED | — |
| S1-C | partial-selection Hamiltonian construction | NOT STARTED | — |
| S1-D | ligand instances and exclusion files | NOT STARTED | — |
| S1-E | versioned CMAP rule | NOT STARTED | — |
| S1-F | identity, resume and refusal | NOT STARTED | — |
| S1-G | CUDA explicit-solvent ladders | BLOCKED | no card: 0–4 reserved by the user for the 0.6.0 gate, 5–8 are hpREST2's (2026-09-19) |

State values are NOT STARTED, IN PROGRESS, IMPLEMENTED (code and deterministic tests), VALIDATED
(evidence on the required platform), or BLOCKED (with the blocker named). IMPLEMENTED is not
VALIDATED, and neither is released.

## Integration commits

None yet.

## Blockers

- Registration: BLOCKED (sandbox). The user has put `$MD_DATA` out of reach of every session
  (2026-09-19); the registration half of S1-G is reported BLOCKED, not passed.
- S1-G: no GPU is available to this wave. Cards 0–4 are reserved by the user for the 0.6.0 gate
  and 5–8 belong to hpREST2. Blocked is not passed; the deterministic CPU work continues.
- Integration: S1's selection record changes three runtime call sites outside its files —
  `ReplicaRun.compare_identity` (`remd/driver.py`), the identity callers in `run/preflight.py` and
  `remd/driver.py`, and state rebuilding in `reference/export.py` and `reference/rest2_export.py`
  — plus a passthrough in `remd/protocol.py` (`c39b5da`). S1 submits them as separate patch
  commits; S0 reviews and lands them at integration. Until then, resume and export of a
  selective ladder do not work on any branch.

## Notes

- No AMBER-mask parser exists anywhere in `src/` today; S1-A is new code, not a wiring job.
- `src/md_tools/rest2/` is identical between `dev` and this baseline, so older notes about those
  files still apply. `src/md_tools/cli/md_openmm.py` and `src/md_tools/openmm/system.py` are
  **not** identical — read them here, not in an older checkout.
