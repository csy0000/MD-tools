# Claude Code instruction: correct OpenMM box geometry and rerun 1 us ALA/RGDfV cMD

Date: 2026-08-24  
Repository: `csy0000/MD-templates`  
Control branch: `dev`  
Expected starting commit: `7aadc466b08aadc820bb22a428494a99b9e21aa7`, or a direct descendant containing this instruction

## Outcome

Make **2.0 nm, with OpenMM padding semantics**, the active explicit-solvent default, correct the
current dodecahedral geometry interpretation, and then launch and complete a fresh conventional-MD
campaign:

| solute | solvent | independent replicates | production per replicate |
|---|---|---:|---:|
| alanine dipeptide | explicit | 3 | 1 us |
| cyclo-(RGDfV) | explicit | 3 | 1 us |
| alanine dipeptide | implicit GBn2/mbondi3 | 3 | 1 us |
| cyclo-(RGDfV) | implicit GBn2/mbondi3 | 3 | 1 us |

This is 12 independent 1 us production runs. The explicit and implicit simulations are separate
Hamiltonians and must never be pooled or described as equivalent replicates.

This instruction supersedes the simulation portion of
`claudecode-instructions/20260821_unattended-kinetics-and-alchemy.md`. Do **not** launch, resume,
develop, or analyze any alchemical transformation. Do not do Deeptime/MSM work in this task. Preserve
all previous campaign data, branches, prototypes, and reports; do not delete them.

## Branch and repository law

1. Work directly on `dev`. Do not create another development branch or PR.
2. Before changing anything, fetch the remote, confirm the checked-out branch is `dev`, confirm the
   working tree is clean, and record the exact starting SHA.
3. If remote `dev` is no longer the expected commit or a direct descendant, inspect and integrate
   the new commits safely. Never force-push or discard user work.
4. Make focused commits and push them to `dev`. The simulation bundles and campaign must identify
   the exact full 40-character commit containing the fixes and configuration.
5. Do not commit trajectories, checkpoints, serialized States, large logs, environments, or other
   production data.

Read completely before acting: `CLAUDE.md`, `README.md`, `docs/configuration.md`, the current
OpenMM configuration/profiles, solvation geometry, system/input generators, cMD runner,
committed-generation continuation code, relevant tests, the accepted ALA/RGD examples, and the
2026-08-21 and 2026-08-24 journals.

## Part 1 — correct the geometry before generating any new system

The current code and `docs/journal/2026-08-24_openmm-delegation-and-cleanup.md` conflate two
different geometric quantities for OpenMM's reduced triclinic boxes. Fix this before changing the
padding or launching simulations.

For the vectors produced by `Modeller._computeBoxVectors(width, shape)`:

- the **shortest nonzero lattice translation** must be calculated from integer combinations of the
  lattice vectors; for OpenMM's cube, rhombic dodecahedron, and truncated octahedron definitions it
  is `width`;
- the **minimum reduced-box perpendicular height used by OpenMM's cutoff legality check** is a
  different quantity. For the current reduced vectors it is represented by the minimum diagonal
  component: `width` for cube, `width/sqrt(2)` for dodecahedron, and
  `sqrt(6)/3 * width` for octahedron;
- OpenMM padding uses a solute bounding-sphere radius `R` and
  `width = max(2*R + padding, 2*padding)`;
- therefore the conservative bounding-sphere solute-to-periodic-copy clearance is
  `shortest_lattice_translation - 2*R`, not
  `minimum_reduced_box_height - 2*R`;
- cutoff fitting must continue to enforce
  `minimum_reduced_box_height >= 2*nonbonded_cutoff + minimum_image_margin`.

Do not merely rename one variable. Separate these quantities in calculations, validation messages,
manifests, schema documentation, and tests. New provenance must report at least:

- requested OpenMM padding;
- solute bounding radius;
- requested and final box width;
- shortest lattice translation;
- conservative solute-image clearance;
- minimum perpendicular/reduced-box height used for cutoff legality;
- cutoff, required cutoff height, margin, and whether the box was grown.

If compatibility requires retaining a legacy field, mark its exact historical meaning and do not
present it as the physical solute-image distance. Existing bundles and run manifests must remain
readable, but a run generated with the old protocol must not be resumed as though it used the new
one.

### Geometry tests required

Add tests that independently establish all of the following:

1. local/fallback vectors equal the vectors produced by the installed OpenMM version;
2. brute-force enumeration of neighboring integer lattice translations gives shortest translation
   `width` for cube, dodecahedron, and octahedron;
3. the reduced-box height/cutoff quantity has the three factors listed above;
4. `padding_semantics: openmm` reproduces OpenMM's width formula and provides at least the requested
   bounding-sphere solute-image clearance;
5. cutoff growth is based on the reduced-box height and passes real OpenMM `Context` construction;
6. a 1.0 nm cutoff remains legal after the expected NPT contraction margin;
7. representative ALA and RGDfV boxes report physically consistent width, volume, cutoff height, and
   solute-image clearance;
8. cube/dodecahedron volume ratios agree with OpenMM for equal periodic-copy spacing.

Correct the active README and configuration documentation. In the historical 2026-08-24 journal,
append a clearly dated correction rather than silently rewriting the historical record. State that
OpenMM itself has no numeric `Modeller.addSolvent` padding default: **2.0 nm is this repository's
chosen default using OpenMM semantics.**

## Part 2 — make 2.0 nm the explicit-solvent default

Set the active explicit-solvent defaults and named explicit profiles to:

```yaml
solvation:
  mode: explicit
  box_shape: dodecahedron
  padding_nm: 2.0
  padding_semantics: openmm
  cutoff_fit_policy: grow
  ionic_strength_molar: 0.15
  positive_ion: Na+
  negative_ion: Cl-
  neutralize: true

system_build:
  nonbonded_method: PME
  nonbonded_cutoff_nm: 1.0
  minimum_image_margin_nm: 0.10
```

Update active examples, generated configuration templates, fixtures, golden outputs, CLI help, and
documentation that claim the old 1.2 nm default. Do not rewrite historical run manifests or make old
results appear to have used 2.0 nm.

Confirm that the default does not leak into implicit mode. Implicit configurations must continue to
reject box, padding, salt, cutoff, pressure, PME, and barostat fields.

Run all fast tests, geometry tests, generator tests, profile/golden tests, and the available CPU
integration/slow tests that exercise explicit and implicit preparation. Do not launch production
until these pass. Commit and push this correction first, then use that exact commit for every new
bundle.

## Part 3 — scientific protocol

Use OpenMM 8.5.2 and the repository's current accepted public generator/runner. Do not create a
parallel experiment-only implementation.

### Explicit ALA

- accepted capped alanine-dipeptide structure and stereochemistry;
- ff19SB;
- OPC with its compatible ions;
- 0.15 M NaCl;
- rhombic dodecahedron;
- `padding_nm: 2.0`, `padding_semantics: openmm`;
- PME, 1.0 nm real-space cutoff, and the 0.10 nm cutoff-height margin;
- 300 K; NPT equilibration at 1 bar followed by the current accepted fixed-box cMD production
  ensemble;
- HMR to 3.024 amu, HBonds constraints, 4 fs Langevin-middle production timestep;
- CUDA mixed precision.

### Explicit cyclo-(RGDfV)

- repository's validated cyclic connectivity, D-Phe stereochemistry, and input route;
- OpenFF Sage 2.2 with AM1-BCC;
- plain TIP3P and its compatible ions, not OPC and not TIP3P-FB;
- otherwise the same box, padding, salt, cutoff, thermodynamic, HMR, timestep, and CUDA conventions
  as explicit ALA where supported by the accepted RGD profile.

Use the current production-quality staged explicit equilibration. Do not silently replace it with the
old 10 ps NVT + 10 ps NPT smoke-test protocol. Each replicate must be prepared and equilibrated
independently with its own master seed. A copied production checkpoint with three velocity reseeds is
not three independent replicates.

### Implicit ALA

- ff19SB;
- nonperiodic GBn2 with mbondi3 radii through the repository's validated ParmEd construction path;
- no water, ions, periodic box, cutoff, pressure, NPT stage, or barostat;
- 300 K;
- current accepted implicit minimization and restrained 20 ps constant-temperature equilibration;
- no HMR, HBonds constraints, 2 fs Langevin-middle timestep;
- CUDA mixed precision if the installed platform supports the validated implicit path.

### Implicit cyclo-(RGDfV)

- the same validated Sage 2.2/AM1-BCC solute parameters and stereochemistry as the explicit RGD
  system;
- nonperiodic GBn2 with mbondi3 radii through the validated implicit-ligand profile;
- otherwise the same implicit restrictions and 2 fs/no-HMR protocol as implicit ALA.

Assert in the generated System and manifest that implicit runs are nonperiodic, use `NoCutoff`,
contain the intended GBn2 force/radii provenance, contain no barostat, and have no solvent or ions.

## Part 4 — replicate, segment, seed, and output contract

Use these distinct master seeds unless an already-established seed registry in the repository
requires a noncolliding deterministic derivation; any alternative must be recorded:

| system | replicate seeds |
|---|---|
| ALA explicit | 20260824001, 20260824002, 20260824003 |
| RGDfV explicit | 20260824101, 20260824102, 20260824103 |
| ALA implicit | 20260824201, 20260824202, 20260824203 |
| RGDfV implicit | 20260824301, 20260824302, 20260824303 |

For every replicate:

- exactly 1 us of committed production;
- 20 committed segments of 50 ns in one persistent run directory;
- full-system coordinates every 100 ps: exactly 10,000 committed frames at completion;
- solute coordinates every 10 ps: exactly 100,000 committed frames at completion;
- state/energy records every 100 ps;
- checkpoint/State generations and atomic progress records at the repository's accepted safe cadence;
- one header, monotonic absolute step/time/frame/invocation accounting, no duplicate boundary frame,
  and no uncommitted DCD tail counted as production;
- finite energies, coordinates, temperature, and all applicable volume/density values;
- exact configuration, seed, input, force-field, software, CUDA, GPU UUID, bundle, and repository
  provenance.

For implicit solvent, the whole system is the solute. Preserve the public reporting contract, but
document that the all-atom and solute selections are identical; do not invent solvent atoms or
volume/density observables.

A padding/default change defines a new protocol. Generate fresh bundles and fresh run directories.
Do not resume or inherit production from the old 1.2 nm explicit runs. Reuse of immutable chemical
input files is allowed only with content hashes and provenance; old equilibrated endpoints are not.

## Part 5 — machine inspection and nine-GPU scheduling

Before launch:

1. inspect all GPUs with `nvidia-smi`: index, UUID, model, memory, utilization, and compute
   processes;
2. inspect CPU, RAM, free disk, CUDA driver/runtime, OpenMM version/platform properties, and required
   OpenFF/AmberTools/ParmEd dependencies;
3. set `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
4. map workers to idle physical GPU UUIDs and record the mapping;
5. never use an occupied GPU and never terminate an unrelated or previous campaign process;
6. enforce one simulation process per GPU using lock files or an equally reliable supervisor;
7. measure short end-to-end pilots for all four system/solvent combinations and estimate runtime and
   disk growth before the production queue;
8. verify enough disk capacity with a safety margin.

Use a fresh data root outside Git, for example:

```text
/path/to/MD-analysis-data/20260824_cmd_2nm/
  explicit/ala/{replicate_01,replicate_02,replicate_03}/
  explicit/rgdfv/{replicate_01,replicate_02,replicate_03}/
  implicit/ala/{replicate_01,replicate_02,replicate_03}/
  implicit/rgdfv/{replicate_01,replicate_02,replicate_03}/
```

If that path is unavailable or already occupied, choose a new non-destructive sibling root and
record it. Never overwrite the 2026-08-21 campaign.

Initially assign six idle GPUs to the six explicit runs because they are the long critical path.
Use up to three other idle GPUs for an implicit queue, then backfill any GPU released by a completed
worker. Do not assume indices 0-8 are idle. Use stable UUID mapping and one simulation process per
GPU.

Use durable named tmux sessions or an equally inspectable supervisor. Maintain a machine-readable
campaign manifest plus a human-readable `STATUS.md` containing run directory, PID, GPU UUID, command,
seed, segment, committed ns, frame watermarks, start/update times, last checkpoint, and exact status,
log, stop, and resume commands. A failed worker must not duplicate or terminate other runs.

## Part 6 — runtime validation and final acceptance

After each pilot and periodically during production, validate:

- the actual manifest values, not only the source configuration;
- explicit box vectors, volume, shortest lattice translation, cutoff height, and cutoff legality;
- explicit solute-to-periodic-copy distance using an exact neighboring-image calculation on
  representative frames, not the old diagonal-factor approximation;
- no direct solute-copy pair falls within the 1.0 nm real-space cutoff; warn below 1.2 nm;
- NPT equilibration never crosses OpenMM's cutoff box-height limit;
- implicit Systems remain nonperiodic and contain no explicit-solvent machinery;
- trajectory/log/checkpoint watermarks agree with committed progress;
- temperatures, energies, and explicit densities/volumes are finite and physically stable;
- no restraints remain in production unless explicitly part of the accepted production protocol.

Write `docs/journal/2026-08-24_ala-rgdfv-1us-cmd-2nm.md` with:

- correction of the geometry diagnosis and tests;
- exact repository commit and software versions;
- final resolved configuration for all four system/solvent combinations;
- input and bundle hashes;
- all 12 seeds and GPU UUID assignments;
- pilot throughput and disk estimates;
- per-replicate segment, time, frame, and QC table;
- failures, retries, checkpoint fallbacks, and whether continuation was bitwise checkpoint or portable
  State;
- exact monitoring and resume commands;
- an explicit statement that alchemy and Deeptime analysis were out of scope.

Do not return `PASS` merely because jobs launched or pilots succeeded. Return `PASS` only when:

1. the geometry interpretation is corrected and its complete test gate passes;
2. the active explicit default is 2.0 nm with OpenMM semantics;
3. all 12 independently prepared runs have exactly 1 us committed;
4. every expected frame/log/checkpoint count and continuity invariant passes;
5. explicit and implicit scientific identity checks pass;
6. runtime QC contains no unresolved NaN, instability, cutoff, periodic-image, or provenance failure;
7. the journal and final focused commits are pushed to `dev`.

If valid jobs are still active when the interactive Claude session must end, leave them running and
return `RUNNING`, never `PASS`, with the exact status and monitoring/resume commands. If an isolated
run fails, diagnose it and continue all independent safe work; report `PARTIAL` only when a real
blocker remains. Never shorten or relabel an incomplete trajectory as 1 us.
