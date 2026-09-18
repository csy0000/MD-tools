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

## Rules that apply to every session on this branch

- Work only in your own worktree and on your own worker branch.
- Do not merge into `0.6.1`, force-push a shared ref, edit another session's worktree, or mark a
  shared milestone complete. The coordinator integrates, in dependency order.
- A change to a shared contract is requested from the coordinator, who lands one commit updating
  the document and its fixtures. Never a private variant.
- No release tag, no package publication, no merge to `main`.
- Announce GPU use before taking a card, and never terminate another session's job or MPS service.
