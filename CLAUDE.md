# CLAUDE.md

## Repository purpose

Project-independent, reusable OpenMM workflows for explicit-water and implicit-solvent molecular
simulation. Two supported public methods: conventional MD, and REST2 replica exchange with omega
exclusion enabled by default.

The repository must stay usable from unrelated consuming projects. Do not introduce assumptions,
paths, terminology, datasets or scientific conclusions belonging to one research project.

Reliability, reproducibility and explicit failure matter more than convenience.

## The architecture, and what not to rebuild

Six public commands, and no seventh:

```
md-template init      md-template install
md-openmm show-default    md-openmm sys-config    md-openmm sys-gen    md-openmm md-gen
```

`generate_system()` lives in `src/md_templates/openmm/sysgen.py`, `generate_md()` in
`src/md_templates/openmm/mdgen.py`. There are no root-level `MD_system_gen.py` or
`MD_input_gen.py`.

A previous architecture — a template registry, a bundle format, schema-migration engines, a
committed-chunk run database and a workflow manager — was deleted in `ca29fcd`, taking the
repository from 20,216 to 4,211 lines. **Do not rebuild any of it.** If a task seems to need a
registry, a schema version, a lock manager or a transaction log, the answer is almost always a
plain file with a documented format.

## Cross-repository ownership

| repository | owns |
|---|---|
| [MD-templates](https://github.com/csy0000/MD-templates) | system construction, force-field record, resolved protocol, seeds, generated scripts, execution provenance |
| [MD-data](https://github.com/csy0000/MD-data) | permanent dataset ID, `$MD_DATA` storage, complete archive checksums, metadata, retention and access lifecycle |
| [MD-analysis](https://github.com/csy0000/MD-analysis) | analysis configuration, software identity, consumed dataset IDs, derived-result lineage |
| project-template | project inputs, configs, workflows and component locks — **not yet available** |

This repository never assigns a dataset ID, writes into `$MD_DATA`, uploads, deletes or registers
anything. See `docs/FAIR_HANDOFF.md`.

## Scientific safety

- Never silently guess a molecular route, force field, charge model, water model, protonation
  state, ion definition, ensemble or enhanced-sampling method.
- The protein force field and the water model are ONE selection, never two independent defaults.
  `--solvent TIP3P` is ff14SB + TIP3P and is the default; `--solvent OPC` is ff19SB + OPC;
  `--solvent GBn2` is ff14SB + GBn2/mbondi3 with no SASA term. The ligand force field is
  OpenFF Sage 2.2.1 (`openff-2.2.1`) with standard AM1-BCC through AmberTools.
- Every consequential default is argued in `docs/md-defaults-scientific-rationale.md`, with its
  evidence classified. Do not change one without updating that document and its evidence label.
- Reject unknown configuration keys and incompatible combinations, naming the full dotted path and
  the value.
- Keep neutralising counterions separate from salt ion pairs.
- Distinguish topology atom count from OpenMM particle count; virtual sites make them differ.
- Keep the four box distances apart and record them under names that say which is which: requested
  solute-to-box padding, solute-to-periodic-COPY clearance, shortest reduced-box height, and
  `2*cutoff + margin`. The default padding is 1.5 nm and 2.0 nm is the conservative option; the
  built-system cutoff gate is what is actually enforced.
- The implicit route is supported for peptides and proteins. A Sage ligand is recorded as
  `support_status: "experimental"` with the measured GB parameter coverage, and no Amber `igb=8`
  parity is claimed for it.
- The barostat attempt interval is the public `common.barostat_frequency_steps` (default 25, which
  is OpenMM's own). It is declared once, in `defaults.py`, and reaches every NPT stage and every
  REST2 replica. Presence of a barostat Force is not the same as it being active.
- 2 fs with unmodified hydrogen masses is the baseline. HMR at 3.024 amu with 4 fs is an explicit
  option, requires constrained hydrogen bonds and rigid water, and is never presented as validated
  for kinetics.
- Preserve periodic box vectors in structures and restart states.
- Durations and intervals must convert to exact integer steps. Reject rather than round.
- REST2 is parameterised by `tau`. `s = (1-tau)^2` and the solute–environment coupling is
  `sqrt(s) = 1-tau`, from one shared function. `tau` is the persisted source parameter; `s` and
  effective temperatures are labelled derived and never accepted back as input.
- Every REST2 replica shares one thermostat temperature and one beta and differs only by
  Hamiltonian, so the pV terms cancel in the NPT exchange criterion. Positions and box vectors are
  one configuration and travel together.
- Scientific configuration states the length of ONE segment, never a segment count.
- Do not change a scientific default without an explicit task requirement, documentation and tests.
- Never call a smoke test validation, convergence or proof of production suitability.

## Generated projects

`md-gen` writes standalone OpenMM scripts. They must:

- not import `md_templates`, and not name this checkout;
- use paths relative to the generated project, so moving `inputs/` and `MD/` together is enough;
- carry the small helpers they need (`md_stages.py`, `rest2_scaling.py`) as copies.

The stage chain is one directory per stage, with the dependency on disk:

```
inputs -> minimization -> eq/nvt_1kcal -> eq/npt_1kcal -> eq/npt_free -> {cMD, REST2}
```

Implicit solvent resolves to `minimization -> eq/nvt_1kcal -> eq/nvt_free`: no box, so no barostat
in the System and no NPT stage. Each stage reads only its parent's `final_state.xml`; a checkpoint
resumes the stage that wrote it and is never consumed downstream. A completed stage refuses to
rerun, identified by a SHA-256 of its whole `stage.yaml`.

Generated runs default to CUDA and refuse a silent CPU fallback.

## FAIR provenance

`sys-gen` writes `original_inputs/`, `forcefield.json`, `provenance.yaml` and `SHA256SUMS`.
`md-gen` writes `provenance.yaml` with parent-system lineage, seeds and the stage plan, plus
`generated-files.sha256`. Runtime writes `resolved_stage.yaml`, `resolved_run.yaml` and append-only
`invocations.jsonl`.

Rules:

- record the value where it is decided, not by parsing a log's prose afterwards;
- `null` means "does not apply here", and is written explicitly;
- `unknown` means "not recorded" and is never upgraded to a guess;
- paths in records are relative; absolute paths may be optional observations only;
- do not hash trajectories at runtime — record path, size and frame count; MD-data hashes them once
  at archival;
- a failed or interrupted invocation is never labelled completed.
- results are archived WITH the prepared inputs that produced them: `system.xml` records what was
  built, the configuration records only what was requested, and the two can disagree. An archive
  holding trajectories alone cannot say which Hamiltonian produced them.

## Testing

- Every test that minimises or integrates a molecular system runs on **CUDA**. CPU and Reference
  are for installation probes and tests that build no system.
- Such tests carry the `gpu` marker and are deselected — not silently passed — where no CUDA
  platform exists.
- Every behaviour change needs a test that would have failed before it.
- Keep the suite small and direct. Prefer one test of an invariant over a matrix of field checks.
- Smoke sizes are picoseconds. Do not add nanosecond runs to the suite.
- The GitHub workflow runs on a GPU-less runner and validates **packaging only**; it must never
  claim scientific runtime validation. GPU acceptance comes from the local GPU machine.

## Git and reporting

- Do not modify unrelated files or discard user changes; inspect the full diff before finishing.
- Do not commit generated run data, trajectories, checkpoints, environments or caches. Test
  fixtures under `tests/data/` ARE source and are tracked.
- Do not rewrite history, force-push, or move an existing tag.
- Do not push, merge to `main`, or tag unless explicitly asked.
- Report what changed, why it is scientifically safe, the exact tests and their results, and the
  remaining limitations. Do not claim a check that was not run.
