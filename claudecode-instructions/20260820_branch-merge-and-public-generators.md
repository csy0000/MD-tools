# Claude Code instruction: finish branch integration, then add the two public generators

Date: 2026-08-20
Repository: csy0000/MD-templates
Current feature branch: feature/openmm-peptide-rest2-examples
Intended long-lived branches: main and dev

## Goal

Do two tasks, strictly in this order:

1. Complete and verify the branch migration and integration of the existing OpenMM peptide/REST2 work.
2. Only after that integration is complete, implement the public MD_system_gen.py and MD_input_gen.py entry points that were mistakenly excluded from the previous instruction.

The branch work is the priority. Do not begin generator implementation while the repository still uses openmm as the default branch or while feature/openmm-peptide-rest2-examples remains unintegrated.

## Important correction to the previous instruction and journal

The previous instruction explicitly excluded the general molecular-input generator. That scope guard was a mistake.

The staged-layout journal also reverses the intended responsibilities of the two generators. Use the following definitions as authoritative:

- MD_system_gen.py prepares the molecular system.
- MD_input_gen.py generates simulation protocols, stage inputs, and launch scripts from a prepared system.

Do not implement the reversed interpretation.

## Phase 0 — inspect before changing anything

Before any mutation:

1. Read CLAUDE.md, README.md, the previous instruction, and both journals:
   - claudecode-instructions/20260819_openmm-ala-rgdfv-rest2-protocol.md
   - docs/journal/2026-08-19_openmm-peptide-rest2-examples.md
   - docs/journal/2026-08-20_staged-run-layout-and-production-status.md
2. Fetch all remote refs and inspect:
   - the current default branch;
   - open pull requests and their base/head branches;
   - branch-protection rules, if accessible;
   - the ancestry of openmm and feature/openmm-peptide-rest2-examples;
   - local staged, unstaged, and untracked files.
3. Preserve all user work. Do not reset, force-push, rewrite published history, or delete branches.
4. Record the exact starting commit IDs.

If the worktree contains unrelated user changes, stop and report them before merging.

## Phase 1 — mandatory branch migration and merge

This phase must finish before Phase 2.

### Required final branch state

- main is the stable long-lived branch and the GitHub default branch.
- dev is the integration/development branch.
- feature/openmm-peptide-rest2-examples is fully merged into dev.
- main is not advanced with the unfinished development feature unless it already contains the same commits.
- future development branches start from dev and target dev.
- no active pull request targets the obsolete openmm branch name.
- repository documentation states that development is merged into dev and main is updated only when a coherent release/milestone is ready.

### Safe migration sequence

1. Use GitHub's real branch-rename operation to rename openmm to main. Do not merely create a second branch pointing at the same commit while leaving openmm as the default.
2. Verify that GitHub changed the default branch to main and retargeted open pull requests correctly. Correct any stale base branch explicitly.
3. Create dev from main if dev does not exist.
4. Merge feature/openmm-peptide-rest2-examples into dev with normal, non-rewritten history. Resolve conflicts conservatively and preserve the journals, executed examples, tests, and production-status records.
5. Run the complete test suite and golden/configuration checks required by the repository.
6. Push main/dev changes only after inspecting the exact commits and diff.
7. Verify the remote branch graph and PR bases after the push.
8. Do not delete feature/openmm-peptide-rest2-examples during this task. It may be removed later after the user verifies the integration.

If authenticated GitHub branch rename/default-branch access is unavailable, STOP HERE. Give the user the exact GitHub UI/API actions needed and do not begin the generators. The user explicitly prioritizes the merge.

### Phase 1 evidence

Record in the new journal:

- old and new default branch;
- main, dev, and feature commit IDs before and after;
- merge commit ID;
- PR bases inspected or changed;
- commands/tests run and results;
- any manual GitHub action still required.

## Phase 2 — create a clean implementation branch

Only after Phase 1 passes:

1. Switch to the updated dev branch.
2. Create a new branch named feature/public-system-and-md-generators from dev.
3. Confirm that it contains the merged REST2 implementation and has a clean worktree.
4. Implement the generators on this feature branch. Do not implement directly on main.

## Phase 3 — authoritative responsibility boundary

### MD_system_gen.py

This is the molecular/system-preparation front end.

Canonical example:

    python MD_system_gen.py -i ALA.pdb -o alanine_dipeptide_system --config system_config.json

Accept -o, -odir, and --outdir as aliases if this can be done without ambiguity.

Responsibilities:

- read PDB, MOL, or SMI input;
- determine file format from the extension, but do not infer the complete system type from extension alone;
- classify supported systems as ligand, protein, or protein-ligand complex;
- require system.type when a PDB is ambiguous;
- for SMI, require explicit ligand-build information: formal charge, stereochemistry policy, protonation policy, conformer generation, charge model, and parameterization route;
- prepare the chemistry, force fields, solvent, ions, box, topology, positions, and initial state before minimization;
- create a portable, immutable system bundle;
- write input hashes, resolved system configuration, preparation logs, system_manifest.json, and forcefield.json;
- record neutralizing ions separately from added salt pairs;
- never run minimization, NVT, NPT, cMD, or REST2.

For OpenMM, System.xml is not a topology. The prepared bundle must include:

- a topology-bearing PDB or mmCIF;
- System.xml;
- an initial State.xml containing positions and periodic box vectors;
- a human-readable forcefield.json;
- checksums and provenance.

Keep supported Amber/GROMACS output concepts in the manifest schema, but do not claim those adapters work until implemented and tested. Amber would use prmtop/rst7; GROMACS would use top/gro. A GROMACS tpr is stage-specific and belongs to MD input generation, not molecular preparation.

### MD_input_gen.py

This is the protocol/stage-generation front end.

Canonical example:

    python MD_input_gen.py --system alanine_dipeptide_system/system_manifest.json -o alanine_run --config md_config.json

It may also accept -odir and --outdir as output aliases.

Responsibilities:

- consume a prepared system_manifest.json and verify its checksums;
- consume one OpenMM-centric, typed scientific configuration;
- generate stage directories, resolved stage JSON, readable Bash launchers, run_all.sh, and a machine-readable run manifest;
- project the canonical configuration into the existing OpenMM runner rather than introducing another configuration engine;
- never reparameterize the molecule, resolvate it, change force fields, or silently rebuild the molecular system;
- do not run long production simulations by default;
- provide a validation/dry-run path that requires no GPU;
- make every stage's input topology, coordinates/state, and predecessor dependency explicit.

Optional --inherit must accept a machine-readable manifest produced by another generated run. It records lineage and permitted configuration inheritance. It is not a checkpoint-resume option and must not parse a free-form log.

## Phase 4 — staged output contract

MD_input_gen.py should generate a project with an explicit, inspectable structure similar to:

    alanine_run/
        inputs/
            system_manifest.json
            forcefield.json
            checksums and immutable system references/copies
        min/
            min.json
            min.sh
        eq_nvt/
            eq_nvt.json
            eq_nvt.sh
        eq_npt/
            eq_npt.json
            eq_npt.sh
        cmd_1/
            cmd_1.json
            cmd_1.sh
        rest2_1/
            rest2_1.json
            rest2_1.sh
        run_all.sh
        run_manifest.json
        run.log

Rules:

1. A generated stage owns its configuration and launcher. When executed, it also owns its logs, endpoint structure/state, checkpoint, and result summary.
2. Each stage JSON explicitly names its input system/topology and input state/coordinates.
3. Stage launchers are readable artifacts containing an explicit command. They must call package modules or thin stage commands; do not duplicate simulation physics in many scripts.
4. The next stage consumes the previous stage's endpoint State.xml so positions, velocities, and box vectors are preserved when scientifically appropriate.
5. REST2 extension continues in the same run directory through the existing committed-generation/runstate contract.
6. Scientific JSON contains duration_per_segment and number_of_exchanges_per_segment. It contains no number of chunks/segments to run. Bash controls repetition; the run manifest records completed segments.
7. Generation must be transactional: refuse an existing nonempty destination unless an explicit safe overwrite/update mode is designed and tested.

The exact spelling of conventional-MD stage directories may be normalized, but document it and keep it stable.

## Phase 5 — configuration ownership and defaults

Do not create two overlapping configuration systems.

- system_config.json owns chemistry and system-building choices.
- md_config.json owns minimization, equilibration, conventional MD, REST2, reporting, and execution choices.
- both resolve through package-level typed models;
- stage JSON files are projections of the resolved canonical MD model, not independent sources of defaults;
- preserve the existing profile/version/hash machinery and record every resolved value and source.

Keep distinct blocks for minimization, NVT, NPT, conventional MD, and REST2. Do not collapse them into one generic stage block.

Honor current repository defaults and worked-example values, including:

- OpenMM 8.5.2, CUDA preferred, mixed precision;
- PME and 10 angstrom real-space cutoff;
- 0.15 M NaCl plus explicit neutralization accounting;
- alanine ff19SB/OPC preparation as supported by the vetted route;
- the vetted RGDfV force-field/water/box decisions already documented in the journal;
- minimization maximum 1000 iterations;
- 1 kcal/mol/angstrom^2 positional restraints on the solute for minimization, 10 ps NVT, and 10 ps NPT;
- the existing conventional-MD reporting contract: whole system every 100 ps and solute/custom selection every 10 ps;
- REST2 tau input with s = (1-tau)^2 and cross scaling 1-tau;
- default tau_max 0.5, linear interpolation;
- enhanced-region selection followed by omega exclusion;
- exact integer conversion of durations, timesteps, reporting periods, and exchange counts, with refusal instead of rounding.

Do not silently undo the later, documented HMR/timestep decisions in the executed examples. If general defaults and worked-example overrides differ, make that distinction explicit in resolved configuration and documentation.

## Phase 6 — reuse and refactor, do not fork the architecture

The repository already has canonical configuration, bundle, runner, runstate, REST2 scaling, tau ladder, device mapping, and CLI code.

- Implement the two root scripts as thin, discoverable public entry points backed by package modules.
- Refactor existing preparation only as needed to separate molecular construction from equilibration.
- Preserve the existing md-openmm CLI as a compatible expert interface unless a tested migration is provided.
- Do not copy runner logic into MD_input_gen.py.
- Do not create a second restart authority.
- Do not create separate force-field implementations for examples.
- Do not make the generator scripts depend on the repository checkout's current working directory.
- Generated projects must remain relocatable and checksum-verifiable.

## Phase 7 — worked examples must use the public generators

Update the alanine and RGDfV worked examples so their documented setup begins with the two public commands.

At minimum demonstrate:

1. Generate or reference the prepared molecular bundle through MD_system_gen.py.
2. Generate min + 10 ps restrained NVT + 10 ps restrained NPT + 1 ns conventional MD + 10 ns REST2 stage layouts through MD_input_gen.py.
3. Alanine uses six REST2 replicas.
4. RGDfV uses ten REST2 replicas.
5. REST2 extension uses the existing checkpoint/State fallback and committed-generation contract in the same run directory.
6. Existing expensive trajectory/checkpoint data are not regenerated or committed merely to prove generation.
7. Clearly label previously executed unrestrained equilibration data; do not relabel it as satisfying the new restrained protocol.

## Phase 8 — required tests

Add tests for at least:

- both root scripts exist, import safely, and show useful --help;
- PDB/MOL/SMI routing and ambiguous PDB refusal;
- required SMI ligand-build fields;
- system bundle files, hashes, force-field provenance, and relocation;
- system generator stops before minimization;
- MD generator does not mutate/reparameterize the system bundle;
- distinct min/NVT/NPT/cMD/REST2 stage projections;
- the 1 kcal/mol/angstrom^2 restraint and reference-coordinate policy;
- explicit stage handoffs;
- no segment/chunk count in scientific JSON;
- Bash-owned segment repetition;
- REST2 tau ladder/scaling, selection, omega exclusion, and device mapping;
- --inherit lineage versus actual checkpoint resume;
- existing destination refusal and partial-generation cleanup;
- CPU-only dry generation for alanine and RGDfV;
- existing full test suite and golden artifacts.

Run tests before and after the refactor. Do not update goldens blindly; explain every intentional change.

## Phase 9 — documentation and journal

Update README.md and docs/configuration.md with:

- why preparation and protocol generation are separate;
- exact commands for both scripts;
- supported input formats and current system-type limitations;
- system_config.json versus md_config.json ownership;
- generated directory tree;
- forcefield.json contents;
- inheritance versus restart;
- CUDA preference and CPU validation path;
- current OpenMM-only implementation status and honest Amber/GROMACS limitations;
- main/dev development policy.

Add:

    docs/journal/2026-08-20_branch-integration-and-public-generators.md

The journal must distinguish:

- implemented and tested;
- generated but not executed;
- executed smoke tests;
- existing production data reused;
- deferred Amber/GROMACS adapters;
- any remaining branch-management action.

## Completion criteria

This task is complete only when:

1. GitHub's default branch is main.
2. dev exists and contains the previous REST2 feature work.
3. the generator implementation lives on feature/public-system-and-md-generators branched from dev.
4. MD_system_gen.py and MD_input_gen.py exist at repository root and have the authoritative responsibility split above.
5. alanine and RGDfV examples use them.
6. the complete test/golden/build checks pass, or failures are honestly recorded with exact evidence.
7. the journal records branch and implementation provenance.

Do not merge feature/public-system-and-md-generators into dev automatically. Push the feature branch and leave it ready for user review/PR unless the user separately authorizes that merge.
