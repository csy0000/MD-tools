# CLAUDE.md

## Repository purpose

Project-independent, reusable OpenMM workflows for explicit-water and implicit-solvent molecular
simulation. Three supported public methods: conventional MD, REST2 replica exchange with omega
exclusion enabled by default, and AIS -- annealed importance sampling along the same REST2 tau
path.

The repository must stay usable from unrelated consuming projects. Do not introduce assumptions,
paths, terminology, datasets or scientific conclusions belonging to one research project.

Reliability, reproducibility and explicit failure matter more than convenience.

## The architecture, and what not to rebuild

Six public commands, and no seventh. AIS is a `--method`, not a command:

```
md-openmm build-top   md-openmm build-md   md-openmm data-register
md-openmm show-default    md-openmm sys-config    md-openmm sys-gen    md-openmm md-gen
```

`generate_system()` lives in `src/md_tools/openmm/sysgen.py`, `generate_md()` in
`src/md_tools/openmm/mdgen.py`. There are no root-level `MD_system_gen.py` or
`MD_input_gen.py`.

A previous architecture — a template registry, a bundle format, schema-migration engines, a
committed-chunk run database and a workflow manager — was deleted in `ca29fcd`, taking the
repository from 20,216 to 4,211 lines. **Do not rebuild any of it.** If a task seems to need a
registry, a schema version, a lock manager or a transaction log, the answer is almost always a
plain file with a documented format.

## Cross-repository ownership

| repository | owns |
|---|---|
| [MD-tools](https://github.com/csy0000/MD-tools) | system construction, force-field record, resolved protocol, seeds, generated scripts, execution provenance |
| [MD-data](https://github.com/csy0000/MD-data) | permanent dataset ID, `$MD_DATA` storage, complete archive checksums, metadata, retention and access lifecycle |
| [MD-analysis](https://github.com/csy0000/MD-analysis) | analysis configuration, software identity, consumed dataset IDs, derived-result lineage |
| project-template | project inputs, configs, workflows and component locks — **not yet available** |

The boundary is not "MD-tools never touches `$MD_DATA`". It is narrower and more useful:

- MD-tools **may** write simulation files and a contract-valid `dataset.yaml` into the ONE
  active dataset directory the user explicitly selected, and nowhere else. It writes there because
  that is where the run's own outputs go; writing them somewhere else first and moving them later
  would break the stage chain's on-disk dependency.
- MD-data owns everything about that directory as a dataset: the schema, the validator, the
  permanent identity, the lifecycle (`active` -> `complete` -> `archived`), the catalogue, aliases,
  extensions, archival checksums and retention. MD-tools imports MD-data's validator; it never
  reimplements it, never invents a field, and never changes a lifecycle status.

Concretely, this repository still never assigns a dataset ID, never walks or hashes `$MD_DATA`,
never touches a dataset other than the selected one, never marks a dataset complete, and never
uploads, deletes, aliases or registers anything. See `docs/FAIR_HANDOFF.md`.

## The MD-data dataset contract

Optional, off by default, and when on it is MD-data's contract v1 — not a schema of ours.

- Import `md_data.validate_dataset` and `md_data.storage.check_dataset_tree`. Never vendor, copy
  or reimplement them, and never add a field the contract does not define.
- The canonical path is `{year}/{project_name}/{data_name}` relative to `$MD_DATA`. Components
  are top level and a component's `path` equals its `name`. `eq/nvt_1kcal` is a STAGE inside the
  `eq` component, never a component.
- `MD_DATA_LOCAL` names the ONE dataset this invocation may write into. Refuse a symlink (aliases
  are MD-data's), refuse a path escaping `MD_DATA`, refuse a status that is not `active`.
- The identity is declared once, in `sys.config.yaml`. `md-gen` reads it back from
  `common/resolved_sys.config.yaml`. Never ask for it twice.
- `dataset_id`, `namespace`, `dataset_name`, `role`, `system`, `created_by.*`, `origin.*` and
  `templates.commit` come from the user. Commits must match `^[0-9a-f]{40}$`. Never fabricate one,
  never abbreviate one, never resolve `HEAD` on the user's behalf.
- Never write `$MD_DATA`'s value into a generated file. The record must survive the tree moving.
  `provenance.yaml` is the one exception, and only because it records the command line verbatim.

## Preflight

Every generated launcher runs one shared preflight before any Context, integrator, worker process,
checkpoint or trajectory exists. `--check` runs the same gate and stops.

- Bounded by construction: a fixed list of named files. Never walk `$MD_DATA`, never enumerate
  datasets, never open or hash a trajectory. A test greps `preflight.py` for `rglob`, `os.walk`,
  `glob.glob`, `iterdir` and `.dcd` and fails if any appear.
- A CUDA platform with zero visible devices FAILS. Listing the platform is not having a device.
- `--check` writes nothing at all, including the stage's `run.log`.
- Missing parent output is `skip` under `--check` (it has not run yet) and `FAIL` for a real run.
  An *inconsistent* parent fails in both.

## AIS sources

- Read source frames with `mdtraj.iterload` in bounded chunks. Never `mdtraj.load` — peak memory
  must not depend on the length of the source trajectory.
- Source tau is evidence: a companion record, or an explicit `AIS.source.source_tau`. Refuse when
  it is absent, when the two conflict, and when it does not equal `path.tau_start`.
- A path counts as complete only if the completion record, CSV rows, DCD existence, DCD frame
  count, frame mapping and configuration identity all agree. A healthy JSON beside a truncated DCD
  is the case this exists to catch.
- NEVER hash the production source trajectory — not at generation, preflight, preparation or run
  time. Record bounded observations instead: path, byte size, frame count from the `iterload`
  survey, frame timing, selected indices, chunk size and chunks read. MD-data hashes it once at
  archival. `AIS/inputs/sources.dcd` is a small generated input and may be digested; the two files
  are named and recorded differently on purpose.
- A DCD header is not evidence of completeness. `NSET` survives an interrupted write intact, so
  the SMALL GENERATED files — `AIS/inputs/sources.dcd` and every `observations.dcd` — are
  validated by READING every frame with bounded `iterload`: exact count, final frame readable,
  finite coordinates, and a non-degenerate box under explicit solvent. Never point that at a
  production trajectory.
- The selected frames are materialised once into `AIS/inputs/sources.dcd` + `sources.yaml`, before
  any path runs, and are NOT deleted afterwards. After preparation the source trajectory is never
  opened again: a rerun must survive the source being archived or deleted.
- Prepared inputs that disagree with the configuration are REFUSED, naming the field. Never
  silently re-prepare — that deletes the configurations a finished path started from.
- `sources.dcd` holds positions and box vectors only. Velocities are NOT stored and must not be:
  DCD cannot carry them, `setVelocitiesToTemperature` gives constraint-satisfying momenta from a
  recorded per-path seed, and the canonical distribution factorises so a fresh momentum draw is
  correct. These are starting configurations; a `final_state.xml` is a restart. Do not blur them.
- Box vectors are stored in `sources.yaml` as exact reduced numbers. The DCD's cell is a
  convenience for viewers; never recover the propagation box from its lengths and angles.

## Identity checks that must recompute

A stored hash that is merely PRESENT proves nothing. Every one of these recomputes and compares.

- One canonical stage fingerprint, `md_stages.stage_config_sha256`, imported by preflight rather
  than reimplemented. Two subtly different hashes over "the stage request" means the run writes one
  and the check compares another.
- `--check` recomputes the current stage's fingerprint from its `stage.yaml` and the PARENT's from
  the parent's, and compares the parent's `final_state.xml` against the digest recorded for it.
  A changed request or a changed handoff fails before a Context exists.
- Force-field preflight is route-aware and exact. An ABSENT expected field FAILS — that is the case
  where what was built is least knowable. A peptide system must claim no ligand force field, a
  ligand system no protein one, and an implicit system neither a water model nor a barostat.
- There is ONE resolution of "which MD-tools is this": `provenance_min.template_identity()`.
  Never reach past it for `implementation_identity()["git_commit"]` — that is null in a
  VCS-installed package whose `direct_url.json` proves the commit, and it is how `dataset.yaml`
  came to carry a verified commit while the `stage.yaml` files beside it carried nulls.
- The same established commit goes into `dataset.yaml`, both `provenance.yaml` files,
  `resolved_sys.config.yaml`, `md.config.yaml`, every `stage.yaml` and every method record.
  Contract-managed preflight REQUIRES every one of them and fails on a missing field, not only a
  mismatched one.
- Contract-managed generation refuses three states before the System is built: a claimed commit
  that is not the generating one, an install with no establishable commit, and **a dirty working
  tree** — its HEAD is real but checking it out gives different code, which is a false provenance.
  Never derive a commit from a version, a branch or a date.
- Unregistered generation from a dirty tree is permitted and must record `git_dirty`,
  `reproducible_from_commit: false` and a plain statement that the commit alone does not reproduce
  it. Never present it as contract-verified.
- MD-data readiness is verified against the INSTALLED distribution's PEP 610 metadata, never by
  echoing the intended pin. Report three states — runtime ready, contract support ready, contract
  support unavailable — and never call contract support ready without import, both validator entry
  points, a matching contract version and a proven source commit.
- AIS compares full per-index atom identity — chain, residue index and id, residue name, atom name,
  element — plus bond connectivity. Atom names repeat within a protein, so names and counts cannot
  tell two topologies apart.
- The MD-data pin lives in ONE place, `md_data_contract.MD_DATA_REPOSITORY` / `MD_DATA_COMMIT`, as
  HTTPS and an exact 40-hex commit. Never SSH, never a branch, never a second copy of the URL.

## Scientific safety

- Never silently guess a molecular route, force field, charge model, water model, protonation
  state, ion definition, ensemble or enhanced-sampling method.
- The protein force field and the water model are ONE selection, never two independent defaults.
  `--solvent TIP3P` is ff14SB + TIP3P and is the default; `--solvent OPC` is ff19SB + OPC;
  `--solvent GBn2` is ff14SB + GBn2/mbondi3 with no SASA term. The ligand force field is
  OpenFF Sage 2.2.1 (`openff-2.2.1`) with standard AM1-BCC through AmberTools. `resolve_sys_config`
  refuses a hand-edited crossing (ff19SB with TIP3P, ff14SB with OPC) before a System exists --
  generating the pair correctly is not the same as building it correctly.
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
- AIS switches the SAME Hamiltonian along a tau path. There is one source of truth for the scaling
  rules -- `templates/rest2_scaling.py` -- and `TauSwitcher` restores the unscaled parameters from
  a private base System before every change, so a switched Context at tau equals a separately built
  `build_scaled_system(..., tau)` exactly. Never implement endpoint energy interpolation.
- AIS work is `delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)`: parameters first at frozen
  coordinates, then propagate. Physical work in kJ/mol, reduced work `beta*W` at the one common
  beta. Switching is FIXED VOLUME with no barostat, and pV work is not included.
- AIS observations are coordinate frames, not integration steps. The update count must divide
  exactly by `number_of_observations - 1`; reject rather than round.
- AIS never starts from the common equilibration chain and is never in `run_all.sh`. Its source is
  an equilibrium trajectory whose tau must equal `tau_start`, established from that run's own
  record and refused if it cannot be. A frame index is never a time.
- Each AIS path is an independent realisation: its own directory, its own DCD, its own seeds. Never
  concatenate them, and never append a second path to an existing `observations.dcd`.
- Scientific configuration states the length of ONE segment, never a segment count.
- Do not change a scientific default without an explicit task requirement, documentation and tests.
- Never call a smoke test validation, convergence or proof of production suitability.

## Generated projects

`md-gen` writes standalone OpenMM scripts. They must:

- not import `md_tools`, and not name this checkout;
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
- OpenMM 8.6.0 means the exact stable release, decided from the INSTALLED PACKAGE identity
  (`conda-meta/openmm-*.json`). `openmm.version.version` is not a release marker: conda-forge's
  8.6.0 release reports `8.6.0.dev-c6173db`, and `short_version` is `8.6.0` for a release and any
  prerelease of it. Record all of them; judge on the package.

## Git and reporting

- Do not modify unrelated files or discard user changes; inspect the full diff before finishing.
- Do not commit generated run data, trajectories, checkpoints, environments or caches. Test
  fixtures under `tests/data/` ARE source and are tracked.
- Do not rewrite history, force-push, or move an existing tag.
- Do not push, merge to `main`, or tag unless explicitly asked.
- Report what changed, why it is scientifically safe, the exact tests and their results, and the
  remaining limitations. Do not claim a check that was not run.
