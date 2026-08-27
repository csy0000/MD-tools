# Task instructions

One file per task. Each defines the scope of a single piece of work; `CLAUDE.md` at the repository
root defines the durable rules that outlive any of them.

## Current

- `20260827_provenance-and-contract-readiness-correction.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_provenance-and-contract-readiness-correction.md`. Made one canonical
  template identity propagate through every generated record, refused dirty contract-managed
  generation before system construction, made preflight require every applicable provenance
  record, verified the ACTUALLY INSTALLED MD-data source commit rather than echoing the intended
  pin, made contract-support readiness an explicit three-state capability, and replaced the
  size-based source-hashing test with a path-specific guard.
- `20260827_md-data-contract-preflight-and-iterload.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_md-data-contract-preflight-and-iterload.md`. Made generated projects
  optionally conform to MD-data dataset v1 using MD-data's own imported validator, added a shared
  automatic preflight that runs before any OpenMM Context (and `--check`, which runs it and stops),
  switched AIS source loading to bounded `mdtraj.iterload`, made AIS source tau explicit evidence,
  and made a truncated or missing path DCD refuse to count as complete.
- `20260827_final-md-data-integrity-correction.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_final-md-data-integrity-correction.md`. Stopped hashing the production
  AIS source, replaced the DCD header count with physical frame reading, made force-field preflight
  route-aware and exact, made `--check` recompute current and parent stage fingerprints, made
  `dataset.templates.commit` be checked against established generator provenance, strengthened AIS
  topology comparison to full atom identity and bond connectivity, and pinned the MD-data validator
  to an exact commit over HTTPS.
- `20260827_ais-method-and-release-gap-correction.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_ais-method-and-release-gaps.md`. Closed the explicit force-field/water
  pairing gap, established an exact stable OpenMM 8.6.0 acceptance environment (and corrected what
  "exact" has to be decided from), and added AIS as a third method through the existing six
  commands.
- `20260827_evidence-based-scientific-defaults.md` — **executed 2026-08-27**; see
  `docs/journal/2026-08-27_evidence-based-defaults-and-rationale.md`. Changed the
  method-development explicit default to ff14SB/Sage 2.2.1/TIP3P, retained ff19SB/OPC as
  `--solvent OPC`, set the dodecahedral padding default to 1.5 nm, exposed
  `common.barostat_frequency_steps`, measured and labelled the implicit ligand scope, and
  produced `docs/md-defaults-scientific-rationale.{md,pdf}` with
  `docs/md-defaults-references.bib`.
- `20260825_focused-scientific-runtime-correction.md` — the scientific and execution corrections
  to the generated scripts, the installer, CI and the working tree.
- `20260825_eq-grouping-restart-gpu-correction.md` — equilibration grouped under `MD/eq/`,
  restart accounting, the no-overwrite guard, and the GPU-only runtime testing policy.

## Historical — not executable

**Every other file here targets an architecture that no longer exists.** The registry, bundle,
schema, profile, canonical-spec and runtime-export machinery they instruct against was removed in
`ca29fcd`, which reduced the repository from 20,216 to 4,211 lines. Their commands will not run
and their file paths do not resolve.

They are kept for provenance: `docs/journal/` cites them by name, and a journal entry that names
an instruction file nobody can read is a record of nothing. Read them as evidence of what was asked
for and when — never as a description of how this repository works now, and never as a task to
execute.

If you are looking for how something works today, the order is: `CLAUDE.md`, the root `README.md`,
then the most recent entries in `docs/journal/`.
