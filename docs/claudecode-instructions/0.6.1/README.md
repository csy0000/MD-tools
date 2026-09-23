# 0.6.1 — active execution index

The one active instruction index for branch `0.6.1`. If a document is not linked from here, it is
not an active instruction for this branch.

| | |
|---|---|
| shared roadmap | [20260918 parallel 0.6.1–0.7.2 development](../20260918_parallel-0.6.1-0.7.x-development.md) |
| what this branch is for | [aims](../../development/0.6.1/AIMS.md) |
| where it stands | [status](../../development/0.6.1/STATUS.md) |
| records shared with 0.7.0 | [shared contracts](../../development/shared-contracts.md) |
| worker reports | [handoffs/](../../development/0.6.1/handoffs/) |
| repository rules | [CLAUDE.md](../../../CLAUDE.md) — every invariant in it still applies |

## Sessions

| session | branch | owns |
|---|---|---|
| S0 (coordinator) | `0.6.1` | `src/md_tools/cli/md_openmm.py`, shared configuration and schema authorities, data-contract changes, root docs, packaging, and any file two jobs touch |
| S1 | `work/0.6.1-selection` | [assignment](S1.md) |

## Session sandbox (the user, 2026-09-19)

Every session works inside its own worktree and nothing else:

- **Push only your own branch.** A worker pushes its `work/...` branch; the coordinator pushes the
  four integration branches. Nobody pushes `main`, `dev` or another session's branch.
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
- Do not merge into `0.6.1`, push any branch but your own, force-push, edit another session's worktree, or mark a
  shared milestone complete. The coordinator integrates, in dependency order.
- A change to a shared contract is requested from the coordinator, who lands one commit updating
  the document and its fixtures. Never a private variant.
- No release tag, no package publication, no merge to `main`.
- Announce GPU use before taking a card, and never terminate another session's job or MPS service.
