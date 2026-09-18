# 0.7.0 — active execution index

The one active instruction index for branch `0.7.0`. If a document is not linked from here, it is
not an active instruction for this branch.

| | |
|---|---|
| shared roadmap | [20260918 parallel 0.6.1–0.7.2 development](../20260918_parallel-0.6.1-0.7.x-development.md) |
| what this branch is for | [aims](../../development/0.7.0/AIMS.md) |
| where it stands | [status](../../development/0.7.0/STATUS.md) |
| records shared with 0.6.1 | [shared contracts](../../development/shared-contracts.md) |
| worker reports | [handoffs/](../../development/0.7.0/handoffs/) |
| repository rules | [CLAUDE.md](../../../CLAUDE.md) — every invariant in it still applies |

## Sessions

| session | branch | owns |
|---|---|---|
| S0 (coordinator) | `0.7.0` | `src/md_tools/cli/md_openmm.py`, shared configuration and schema authorities, data-contract changes, root docs, packaging, CI, and any file two jobs touch |
| S2 | `work/0.7.0-topology` | [assignment](S2.md) — endpoint mapping and topology construction |
| S3 | `work/0.7.0-hamiltonian` | [assignment](S3.md) — Amber18 softcore, energies, derivatives |
| S4 | `work/0.7.0-execution-analysis` | [assignment](S4.md) — windows, reporting, estimators, cycles |

## How three sessions work on one branch without waiting

S3 and S4 do **not** wait for S2. S3 builds against the agreed miniature topology-plan fixtures;
S4 builds against frozen energy and derivative fixtures at lambda 0, 0.25, 0.5, 0.75 and 1. Only
real end-to-end execution waits for A1 and A2.

That only holds while the fixtures are shared and versioned. A session that quietly edits a
fixture to make its own code pass has broken the other two without telling them — fixture changes
go through S0, with a version bump, and every result citing the old version is re-derived.

## Session sandbox (the user, 2026-09-19)

Every session works inside its own worktree and nothing else:

- **Push only your own branch.** A worker pushes its `work/...` branch; the coordinator pushes the
  four integration branches. Nobody pushes `main`, `dev`, `dev-0.6.0` or another session's branch.
- **No `$MD_DATA`.** Do not register, retrieve, search or read anything under the machine's
  `$MD_DATA` — no `data-register`, no catalog lookup against it, no registered dataset as a
  fixture. A test that needs a catalog or a dataset root builds one in a temporary directory and
  points `MD_DATA` at it explicitly; unsetting the variable is not isolation, because the user
  configuration then finds the machine root again.
- Nothing outside the worktree is written except the session's own temporary/scratch directories
  and its own environment: not another worktree, not the `site-packages` of a shared environment,
  not the user configuration.

Consequently **registration evidence is out of reach in this wave.** Every acceptance row that says
"registration" is BLOCKED (sandbox) until the user lifts this, and is reported that way — never as
passed, and never satisfied by registering into the real catalog "just once".

## Rules that apply to every session on this branch

- Work only in your own worktree and on your own worker branch.
- Do not merge into `0.7.0`, push any branch but your own, force-push, edit another session's worktree, or mark a
  shared milestone complete.
- Do not change existing cMD, REST2 or AIS behaviour. A new alchemical Hamiltonian does not
  redefine the two-System AIS contract, and `dU/dlambda = U1 - U0` does not transfer to a softcore
  path.
- Unsupported combinations are refused by name until validated. A successful short run does not
  create support.
- No release tag, no package publication, no merge to `main`. 0.7.0 is released only after 0.6.1.
- Announce GPU use before taking a card, and never terminate another session's job or MPS service.
