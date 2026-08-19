# Claude Code instruction: establish main/dev and validate OpenMM peptide MD/REST2 templates

Date: 2026-08-19  
Repository: `csy0000/MD-templates`  
Current primary branch at instruction creation: `openmm`  
Expected current base commit: `9e08dffd406d86384a1a4ff4e899f7186cb3dcda` or a direct descendant containing this instruction

## Goal

Make the repository's branch model simple, then implement and demonstrate one OpenMM-centric
protocol on two peptide systems:

- neutral capped alanine dipeptide, ACE-ALA-NME, with six REST2 replicas;
- RGDfV, using a vetted user-provided structure and ten REST2 replicas.

For each system the worked protocol is:

1. restrained minimization, maximum 1,000 iterations;
2. restrained 10 ps NVT equilibration;
3. restrained 10 ps NPT equilibration;
4. 1 ns conventional NPT MD;
5. 10 ns REST2 production per replica.

The public configuration is OpenMM-centric. Amber and GROMACS are future adapters; do not design
this implementation around Amber `mdin` variable names. REST2 is configured with Amber-style
`tau`, however, and resolves to the existing OpenMM `s`/`sqrt(s)` Hamiltonian exactly as described
below.

This task replaces the previous `n_chunks`/`chunk_ns` input contract. Scientific JSON defines one
segment with `duration_per_segment`; Bash controls how many segments to execute; runtime manifests
record what actually completed.

## Read before editing

Read completely:

- `CLAUDE.md`;
- `README.md` and `pyproject.toml`;
- `docs/journal/2026-08-19_md-stack-installation.md`;
- the latest relevant configuration, bundle, continuity, MD, and REST2 journals;
- `src/md_templates/openmm/config.py`;
- `src/md_templates/openmm/schemas.py`;
- `src/md_templates/openmm/equilibration.py`;
- `src/md_templates/openmm/md.py`;
- `src/md_templates/openmm/rest2.py`;
- `src/md_templates/openmm/runner.py`;
- `src/md_templates/openmm/runstate.py`;
- all related tests and packaged example/profile resources;
- `https://github.com/csy0000/partitioned-REST2/blob/main/scripts/run_explicit_rest2_remd.py`.

The external script is a scientific reference, not code to copy blindly. It uses TIP3P, a geometric
`s` ladder, older package names, rounded step conversions, and incomplete restart handling. Preserve
only the relevant verified scaling, exchange, selection, and two-trajectory-stream ideas.

## Phase 0: branch migration

This phase is explicitly authorized. Perform it before implementation.

1. Confirm the worktree is clean and fetch all remotes.
2. Record the current remote `openmm` SHA and verify it contains this instruction.
3. Check open pull requests and branch protection/default-branch settings. Record anything that
   needs attention after the rename.
4. Rename the remote branch `openmm` to `main` using GitHub's branch-rename operation. Do not
   simulate a rename by force-pushing unrelated history.
5. Verify that `main` points to exactly the recorded commit and that the repository default branch
   is `main`.
6. Update the local branch/upstream configuration to follow `origin/main`.
7. Create remote `dev` at exactly the same commit as `main`.
8. Verify the final state: stable/default branch `main`; integration branch `dev`; no remote
   `openmm` branch.
9. Do not delete, rewrite, merge, or repurpose any existing feature/migration branches.
10. Add a short branch-policy section to the contributor-facing documentation: normal work branches
    from `dev` and targets `dev`; `dev` is merged into `main` only at a validated milestone.

After the migration, create `feature/openmm-peptide-rest2-examples` from `dev`. All implementation
commits go there. Push it and open a draft PR against `dev`; do not merge it.

If the branch rename cannot be performed safely because of permissions, protection, an unexpected
default branch, or a moving base, stop before implementation and report the exact state and commands
needed. Never force the migration.

## Scope guard

- Keep this OpenMM-centric and method-development friendly.
- Reuse and simplify the existing package rather than creating a parallel application.
- Do not implement Amber or GROMACS execution in this task.
- Do not implement the general molecular-input generator in this task.
- Do not copy production code from `partitioned-REST2`; port the required behavior into the current
  `md_templates` package with tests and current persistence invariants.
- Do not commit large trajectories, checkpoints, serialized states, environments, licensed Amber
  content, or generated caches.
- `test/` in this instruction contains worked user examples. Keep automated tests under the existing
  `tests/` directory.
- A short smoke test is not evidence of convergence or scientific validation.

## System definitions

### Alanine

Use neutral capped alanine dipeptide `ACE-ALA-NME`. The complete peptide is the default solute and
REST2 enhanced region.

### RGDfV

Do not infer the chemistry from the string `RGDfV`. Before building it, locate a vetted input in the
repository or user-provided local inputs and record:

- whether it is linear or cyclic and, if cyclic, the exact closure bond;
- the identity and stereochemistry of every residue, especially D-Phe;
- terminal/capping state;
- protonation states and total formal charge;
- any non-standard residue/template mapping or manual bond.

If no vetted structure or complete chemical definition exists, do not invent one. Complete the
generic implementation and alanine example, create the RGDfV example skeleton with a clear
`INPUT_REQUIRED.md`, and report RGDfV execution as blocked. Do not claim the RGDfV protocol ran.

### Shared preparation

Use the installed and validated stack described in the installation journal. Prefer CUDA over
OpenCL. Use:

- OpenMM 8.5.2;
- ff19SB for peptide parameters where supported by the vetted topology route;
- OPC water;
- truncated-octahedral periodic box with at least 12.0 A solute padding;
- explicit 0.15 M NaCl, with neutralizing counterions and added salt pairs counted separately;
- PME with a 1.0 nm real-space cutoff;
- bonds involving hydrogen constrained;
- rigid water;
- no hydrogen-mass repartitioning;
- 300 K;
- 1 bar for NPT stages;
- 2 fs timestep for dynamics;
- `LangevinMiddleIntegrator`, friction 1/ps;
- explicit nonzero seeds recorded per stage and replica;
- CUDA mixed precision by default and `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

Do not set Amber `saltcon`; this is explicit salt. Record requested and realized salt concentration,
ion species/counts, water count, box vectors, force-field identifiers, tool versions, preparation
commands, input/output hashes, warnings, and manual edits in the prepared bundle.

## OpenMM-centric configuration

Use explicit, typed, lowercase snake_case OpenMM concepts. Each method/stage has its own block and
rejects fields belonging to another stage. A representative shape is:

```json
{
  "engine": {
    "name": "openmm",
    "version": "8.5.2",
    "platform": "CUDA",
    "precision": "mixed",
    "device_indices": [0]
  },
  "integrator": {
    "type": "LangevinMiddleIntegrator",
    "temperature_kelvin": 300.0,
    "friction_per_ps": 1.0,
    "timestep_fs": 2.0
  },
  "minimization": {
    "max_iterations": 1000,
    "restraint": {
      "selection": {"type": "solute"},
      "force_constant_kcal_per_mol_angstrom2": 1.0
    }
  },
  "equilibration": {
    "nvt": {"duration_ps": 10.0},
    "npt": {
      "duration_ps": 10.0,
      "pressure_bar": 1.0,
      "barostat": "MonteCarloBarostat"
    }
  },
  "production": {
    "method": "rest2",
    "duration_per_segment_ns": 5.0
  },
  "reporting": {
    "full_system_interval_ps": 100.0,
    "selected_atoms_interval_ps": 10.0,
    "selected_atoms": {"type": "solute"}
  }
}
```

The exact schema may be improved while retaining these semantics. Do not retain `n_chunks`,
`chunk_ns`, or total-duration-derived chunk inputs in the active scientific JSON.

Update `CLAUDE.md`, schemas, profiles, examples, fingerprints, compatibility checks, tests, and
documentation together. Bump the affected schema/profile versions. Reject old chunk-based inputs
with a precise migration message; do not silently reinterpret them.

Scientific JSON defines one segment. Generated Bash scripts control repetition, for example with
`NUMBER_OF_SEGMENTS`. Runtime state must still record completed segments, total committed time,
absolute steps, frames, restart generations, and exchange attempts. Segment count is execution
state, not a scientific JSON input.

All durations/reporting intervals must convert to exact integer steps. Reject rather than round.

## Stage protocol

For both systems:

### Minimization

- maximum 1,000 OpenMM minimizer iterations;
- positional restraint on the complete solute, 1 kcal mol-1 A-2;
- initial prepared solute coordinates are the restraint reference;
- record minimizer tolerance and final energy/force sanity, recognizing that 1,000 is a maximum.

### NVT equilibration

- 10 ps = 5,000 steps;
- 300 K;
- same solute restraint and reference coordinates.

### NPT equilibration

- 10 ps = 5,000 steps;
- 300 K and 1 bar;
- `MonteCarloBarostat` with its explicit attempt frequency persisted;
- same solute restraint and reference coordinates;
- persist the final periodic box in the authoritative State and human-readable structure.

### Conventional MD

- 1 ns NPT production for each worked example;
- no positional restraint;
- start from that system's final NPT-equilibration State;
- save full-system coordinates every 100 ps;
- save solute/custom-selection coordinates every 10 ps;
- write state data and committed checkpoint plus serialized State at the segment boundary.

This 1 ns duration is specific to the worked examples. The versioned general conventional-MD
profile may retain `duration_per_segment_ns: 5.0` as its default.

### REST2

- start from the final conventional-MD coordinates and equilibrated box;
- run NVT REST2 production at 300 K for this task, matching the referenced potential-energy-only
  exchange criterion; do not keep an NPT barostat without implementing and testing the correct pV
  contribution;
- 10 ns per replica in each worked example;
- implement it as two 5 ns segments requested by the Bash script;
- save full-system coordinates every 100 ps and solute/custom-selection coordinates every 10 ps;
- commit restart artifacts and global exchange state after every 5 ns segment.

## Amber-style tau parameterization of the existing scaling

The public and persisted REST2 ladder is `tau`, not `s`.

Defaults:

```json
{
  "rest2": {
    "enhanced_region": {"selection": {"type": "solute"}},
    "tau_ladder": {
      "minimum": 0.0,
      "maximum": 0.5,
      "count": 6,
      "interpolation": "linear"
    },
    "exchange": {
      "number_of_exchanges_per_segment": 100
    },
    "omega_exclusion": {
      "enabled": true,
      "definition": "peptide_omega"
    }
  }
}
```

For each replica derive, using one shared tested function:

```text
one_minus_tau = 1 - tau
s = (1 - tau)^2
sqrt(s) = 1 - tau
```

Therefore preserve the existing Hamiltonian meaning:

```text
solute-solute terms:  s = (1 - tau)^2
solute-environment:   sqrt(s) = (1 - tau)
environment terms:    1
```

Do not implement a different Hamiltonian merely to rename the parameter. Persist `tau` as the
source parameter and `s`, `sqrt_s`, and effective temperatures as labelled derived diagnostics.

Ladders:

- alanine, six replicas: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`;
- RGDfV, ten replicas: ten linearly spaced inclusive values from 0.0 to 0.5.

Add exact mathematical tests that `s == (1-tau)^2` and `sqrt(s) == 1-tau` across both ladders,
plus force/energy-component tests showing the expected internal, cross, and environment scaling.

## Exchange contract

Keep the configuration straightforward:

```json
"production": {"duration_per_segment_ns": 5.0},
"exchange": {"number_of_exchanges_per_segment": 100}
```

Derive:

```text
steps_per_segment = duration_per_segment / timestep
steps_per_exchange = steps_per_segment / number_of_exchanges_per_segment
```

At 5 ns, 2 fs, and 100 exchanges this is exactly 2,500,000 steps per segment and 25,000 steps
(50 ps) between exchange rounds. Reject any configuration for which either division is not exact.

Use alternating nearest-neighbor phases and the repository's durable exchange history. Persist and
restore global attempt indices, phase, walker/replica mapping, RNG state, accepted/attempted counts,
and the committed watermark. The second segment and later extensions must continue, not restart,
the exchange sequence.

Do not hard-code one replica per GPU. Accept an ordered CUDA device list and map replicas
deterministically, allowing multiple replicas to share a device when replicas outnumber selected
GPUs. Record the resolved mapping. Inspect available hardware rather than assuming all installed
GPUs should be mixed in one synchronous run.

## Enhanced-region selection and omega exclusion

OpenMM consumes resolved zero-based atom/particle indices, not Amber masks directly.

Support at minimum:

- `{"type": "solute"}` as the default, resolved from the prepared bundle;
- explicit zero-based atom indices;
- an optional Amber-mask expression parsed through ParmEd, then resolved to indices before System
  construction.

Persist the original selection, selection language, resolved indices, atom/residue identities,
topology hash, and atom-order fingerprint. Reject empty, ambiguous, out-of-range, or virtual-site-
inconsistent selections.

Apply operations in this order:

1. resolve the enhanced solute region;
2. identify force terms affected by REST2;
3. identify peptide omega torsions in the enhanced region from topology/bonded terms;
4. keep those omega terms unscaled when omega exclusion is enabled;
5. build each tau replica;
6. persist the exact excluded torsion/central-bond identities.

Test that omega exclusion is enabled by default, can be disabled explicitly, changes only the
intended omega terms, and survives restart compatibility validation.

## Worked example layout

Create exactly these user-facing example roots:

```text
test/ala/REST2/
test/ala/REST2/extension/
test/rgd/REST2/
test/rgd/REST2/extension/
```

Each main example should contain small, reviewable inputs only:

- `README.md` explaining the system, every stage, tau ladder, replica count, device selection,
  expected runtime, outputs, and exact commands;
- the canonical JSON input and resolved example JSON;
- preparation/run scripts using repository CLIs rather than copied algorithms;
- expected directory layout and small validation summaries;
- checksums/provenance for committed molecular inputs when redistribution is allowed;
- an `outputs/` directory pattern that is gitignored.

Do not commit generated production trajectories, checkpoints, State XML, environments, or large
logs.

The REST2 example runner must execute the entire chain in order:

```text
prepare -> minimize -> 10 ps NVT -> 10 ps NPT -> 1 ns cMD -> 10 ns REST2
```

For alanine use six replicas. For RGDfV use ten replicas.

## Extension examples

Each `extension/` directory contains documentation and an executable `extend.sh` demonstrating how
to request additional 5 ns REST2 segments from the committed checkpoints of the parent run.

The extension directory contains the extension request, not a copied run. Continuation must write
into the original REST2 run directory in accordance with the repository's same-directory resume
contract. Do not copy checkpoints into `extension/` and do not create a scientifically separate
sibling run while calling it continuation.

The extension example must prove:

- binary checkpoint is preferred when compatible;
- serialized State fallback is explicit and recorded;
- physical time, steps, frames, exchange attempt indices, exchange phase, RNG state, walker mapping,
  and lifetime statistics continue monotonically;
- trajectory/state-data output appends without duplicated headers or overwritten files;
- incompatible tau ladder, enhanced region, omega policy, topology, integrator, or nonbonded settings
  are rejected before output is opened;
- a second extension can be run with the same command pattern.

## Automated tests

Add or update tests for:

1. schema separation among minimization, NVT, NPT, cMD, and REST2;
2. rejection/migration message for old `n_chunks` and `chunk_ns` inputs;
3. exact step conversion without rounding;
4. cMD one-segment execution and Bash-controlled repetition;
5. both tau ladders and tau-to-s mapping;
6. internal/cross/environment REST2 scaling;
7. enhanced-region selection and persisted resolved indices;
8. default-on and explicit-off omega exclusion;
9. exact exchange-step derivation from duration and exchange count;
10. deterministic device mapping when replicas exceed GPUs;
11. two trajectory streams at 100 ps full-system and 10 ps selected-atom cadence;
12. six-replica alanine and ten-replica RGDfV configuration resolution;
13. fresh REST2 run followed by same-directory extension;
14. checkpoint preference and explicit State fallback;
15. lifetime exchange history/statistics across extension;
16. extension incompatibility rejection before file append;
17. installed-wheel access to the updated schemas/profiles/examples;
18. CPU/mocked execution for normal CI without requiring ten GPUs or 10 ns trajectories.

Add a very short CPU end-to-end smoke version of each available system. Keep the full 1 ns/10 ns
CUDA examples outside the default CI suite and give exact commands.

## Full example execution

On the installed workstation, inspect the environment and selected GPUs first. If valid molecular
inputs are present and the estimated runtime is acceptable, execute the complete alanine and RGDfV
worked protocols, including the first 5 ns extension demonstration, and save only concise summaries
and hashes to the repository.

If a full run is not completed, report it as unexecuted or partial. Never label generated inputs, a
short smoke run, or a successful process launch as completion of 10 ns REST2.

Do not perform single-point Amber/OpenMM energy matching. Cross-engine ensemble comparison is not
part of this task.

## Documentation and journal

Update public documentation for:

- the `main`/`dev` branch policy;
- OpenMM-centric canonical configuration;
- removal of chunk count from scientific JSON;
- segment duration versus Bash-controlled repetition versus runtime completed-segment state;
- tau parameterization and its exact mapping to `s`/`sqrt(s)`;
- enhanced-region selection and omega exclusion;
- complete worked-example and extension commands.

Add a dated journal entry containing branch migration evidence, files changed, schema/profile
versions, scientific decisions, exact commands and results, smoke/full-run status, GPU mapping,
known limitations, and deferred work.

## Acceptance gate

Before opening the draft PR:

- `main` is the stable/default branch and `dev` exists at the migration base;
- the implementation branch was created from `dev` and targets `dev`;
- the current public config is OpenMM-centric and has no active `n_chunks`/`chunk_ns` inputs;
- minimization, NVT, NPT, cMD, and REST2 have distinct validated blocks;
- alanine resolves to six tau replicas and RGDfV to ten;
- tau is the persisted source parameter and scaling matches `(1-tau)^2`/`(1-tau)`;
- enhanced-region selection occurs before omega exclusion;
- 1 ns cMD and 10 ns REST2 worked protocols are reproducibly generated for each scientifically
  defined system;
- extension examples continue in the original run directories from committed restart generations;
- relevant focused tests, full supported tests, package/build checks, CLI help, and smoke commands
  are run and reported exactly;
- no large outputs, secrets, licensed files, absolute local paths, or unrelated changes are
  committed.

Push `feature/openmm-peptide-rest2-examples` and open one draft PR against `dev`. Do not merge it.
The final report must clearly separate implemented, tested, smoke-tested, fully executed, blocked,
and deferred items.
