# Task instructions

One file per task. Each defines the scope of a single piece of work; `CLAUDE.md` at the repository
root defines the durable rules that outlive any of them.

## Current

- `20260825_focused-scientific-runtime-correction.md` — the scientific and execution corrections to
  the generated scripts, the installer, CI and the working tree.
- `20260825_eq-grouping-restart-gpu-correction.md` — equilibration grouped under `MD/eq/`, restart
  accounting, the no-overwrite guard, and the GPU-only runtime testing policy.

## Historical — not executable

**Every other file here targets an architecture that no longer exists.** The registry, bundle,
schema, profile, canonical-spec and runtime-export machinery they instruct against was removed in
`ca29fcd`, which reduced the repository from 20,216 to 4,211 lines. Their commands will not run and
their file paths do not resolve.

They are kept for provenance: `docs/journal/` cites them by name, and a journal entry that names an
instruction file nobody can read is a record of nothing. Read them as evidence of what was asked
for and when — never as a description of how this repository works now, and never as a task to
execute.

If you are looking for how something works today, the order is: `CLAUDE.md`, the root `README.md`,
then the most recent entries in `docs/journal/`.
