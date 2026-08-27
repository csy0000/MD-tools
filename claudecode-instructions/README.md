# Task instructions

One file per task. Each defines the scope of a single piece of work; `CLAUDE.md` at the repository
root defines the durable rules that outlive any of them.

## Current

- `20260827_ais-method-and-release-gap-correction.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_ais-method-and-release-gaps.md`. Closed the explicit force-field/water
  pairing gap, established an exact stable OpenMM 8.6.0 acceptance environment (and corrected what
  "exact" has to be decided from), and added AIS as a third method through the existing six
  commands.
- `20260827_ais-method-and-release-gap-correction.md` — **next task**; close edited explicit-pair
  validation and exact OpenMM 8.6.0 acceptance gaps, then add standalone forward AIS using the
  existing REST2 tau scaling, trajectory/time-window source selection, 21 endpoint-inclusive
  coordinate/work observations, and focused CUDA validation.
- `20260827_evidence-based-scientific-defaults.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_evidence-based-defaults-and-rationale.md`. Changed the method-development
  explicit default to ff14SB/Sage 2.2.1/TIP3P, retained ff19SB/OPC as `--solvent OPC`, set the
  dodecahedral padding default to 1.5 nm, exposed `common.barostat_frequency_steps`, measured and
  labelled the implicit ligand scope, and produced
  `docs/md-defaults-scientific-rationale.{md,pdf}` with `docs/md-defaults-references.bib`.
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
