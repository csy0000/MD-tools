# Corrective runtime closure and collective-variable reporting

Implementation baseline: 0498c22d84c58e80c3561d35276f806218465abb

Work on branch dev. Read CLAUDE.md and this file completely before editing. This is an implementation task, not another review. Add regression tests that reach the named runtime rule before changing behavior. Preserve all validated REST2, rREST2, AIS, trajectory-format, platform, MPI, restart, and MD-data contracts.

The purpose of this iteration is to close the remaining lifecycle holes found in the full acceptance audit and add first-class, analysis-ready torsion reporting from cv.yaml. Do not add an executable or a fifth md-openmm command. Generated Python scripts remain thin entry points into the installed runtime.

## 1. Completed cMD stages must be proven, not inferred

The current completion check observes that a fingerprint field exists but never recomputes or compares it. It checks only that the final restart and, for dynamics, the trajectory exist.

Replace this with one authoritative completion verifier. Before returning already completed, it must:

- recompute the current stage fingerprint from the resolved scientific configuration and the current topology and System digests;
- compare it with the recorded fingerprint;
- validate the complete expected output inventory for that stage;
- compare the current sha256 and size of every recorded immutable output;
- validate the final-state particle count;
- validate the committed checkpoint pointer, selected generation, checkpoint digest, fingerprint, step count, stage identity, seed, platform identity, and committed stream counts;
- validate DCD, state CSV, phase-space NetCDF, and CV CSV readability and exact expected counts when enabled;
- reject missing, additional, truncated, corrupt, foreign, or legacy-unverifiable output records with an actionable diagnostic;
- perform all verification read-only.

A changed built.pdb, built.xml, resolved scientific setting, final restart, DCD, state CSV, phase-space stream, CV stream, checkpoint pointer, checkpoint binary, or checkpoint sidecar must prevent the stage from being skipped.

The machine log must record the complete output manifest. The state CSV is currently omitted from the completion output record; correct that.

## 2. cMD overwrite and continuation

Use the StagePreflight output inventory as the single authority for replacement.

- --overwrite must remove or transactionally replace every owned output: out, log, state CSV, DCD, phase-space NetCDF, CV CSV and its resolved definition, restart, checkpoint pointer and every checkpoint generation.
- Cleanup must happen before any new output is opened.
- It must not delete unrelated files.
- A crash during cleanup/replacement must not leave old files presented as outputs of the new run.
- The existing automatic cMD continuation contract remains: a valid committed checkpoint short of the requested step count resumes automatically; --resume is not required.
- If --resume has no distinct cMD meaning, remove it from the cMD execution surface and documentation rather than accepting an inert flag. Do not change REST2, rREST2, or AIS semantics merely for cosmetic flag uniformity.

Add end-to-end tests for changed input digests, every damaged output type, a legacy/incomplete manifest, complete cleanup, unrelated-file preservation, and interruption during replacement.

## 3. AIS directory identity and resume semantics

An AIS output directory may never be adopted merely because AIS_run.json is absent.

Before any write:

- inspect the complete AIS inventory;
- if any owned AIS artifact exists but AIS_run.json is absent, unreadable, incomplete, or unverifiable, refuse the directory as orphaned/unknown;
- a truly empty or absent directory is fresh;
- a compatible directory whose every path is complete may be verified and skipped idempotently;
- a compatible directory with any incomplete path requires --resume;
- without --resume, refuse before changing any byte and explicitly say that --resume continues committed paths while --overwrite starts over;
- with --resume, verify and skip completed paths, resume incomplete paths from their last committed generation, and start paths that the same identified run had not yet begun;
- --resume against an absent/empty directory must refuse because there is nothing to resume;
- --overwrite and --resume remain mutually exclusive.

The preflight disposition and the boolean actually passed to run_one_path must be one decision. Do not label a run resume and then invoke the path runner with resume false.

## 4. AIS overwrite ownership

clear_run_directory currently receives an identity but does not use it, and treats every exact path_NNNN directory and AIS_trajNNNN.nc file as owned.

Replace name-only destructive inference with verified ownership:

- read and validate the previous AIS_run.json first;
- derive the expected path IDs, rank reports, tables, manifests, and trajectories from that identity and from verified per-path completion/checkpoint records;
- remove only artifacts proven to belong to that run;
- retain unrelated files and directories, including a user-created numeric-looking path that has no valid MD-tools ownership record;
- if ownership cannot be established, refuse automated cleanup and advise a new output directory or explicit manual cleanup;
- make the identity parameter meaningful;
- use a staged/transactional replacement so a crash cannot make remnants of two experiments look current.

Test a valid prior run, longer-to-shorter replacement, changed MPI size, orphan files, a user-owned path_0000, corrupt identity, and crash boundaries.

## 5. One complete ladder plan for every entry point

The generated REST2/rREST2 runtime and the directly callable grouped executor must construct the same complete typed LadderPreflight before output creation.

A ladder plan must contain, at minimum:

- resolved protocol and full scientific identity;
- topology and unscaled reference System facts;
- exact tau list;
- resolved timestep;
- solute indices and excluded omega bonds;
- complete force audit;
- an ordered immutable collection of prepared Systems, one for every tau rung;
- platform/device/MPI coordination;
- exchange and reporting schedules;
- initial continuation facts;
- output inventory and disposition;
- prepared exchange-rule identity;
- for rREST2, the prepared reservoir declaration and validated source facts.

Do not prebuild only the top rung and then call protocol.build_systems again. The driver must consume the exact prepared rung Systems. It may clone them only when the clone is scientifically identical and the reason is documented and tested.

The direct grouped executor must obtain all required scientific values from the group/protocol inputs and perform the same planning. It must not open its rank report and then load the protocol, solute selection, System, or reservoir for the first time.

Remove duplicate YAML parsing and scientific reconstruction from run_grouped and ReplicaRun.run. A missing or malformed protocol, solute document, scaling force, timestep, continuation, exchange rule, or reservoir must refuse before mkdir or report creation.

## 6. rREST2 reservoir preflight

Move reservoir_declaration_text and all predictable source validation before output creation. Preflight must validate read-only:

- source existence and readable phase-space format;
- nonzero frame count and valid time axis;
- positions, velocities, and periodic box availability as required;
- atom count and topology compatibility;
- tau/temperature/Hamiltonian identity of the source;
- top-rung System compatibility;
- velocity policy;
- continuation/reservoir box compatibility whenever the initial configuration makes this knowable;
- the exact declaration content and its digest.

Carry the prepared declaration in LadderPreflight. Rank 0 publishes that already-validated content atomically and every rank verifies its digest. Do not reopen the source merely to rediscover the same facts after output exists.

## 7. MPI failures inside the scientific run

The outer coordination.phase around an entire ladder is insufficient when the executor catches an exception locally while other ranks remain inside driver collectives.

- In a plural launch, any unexpected exception during Context construction, propagation, energy evaluation, exchange, reservoir refresh, trajectory reporting, checkpointing, manifest writing, or finalization must immediately invoke the one shared communicator fail/abort authority.
- Do not convert a rank-local exception into an ordinary local return code and then enter another collective.
- All ranks must execute collectives in the same order.
- Preserve a useful per-rank failure report where possible, but never prefer report completion over stopping a hung job.
- A clean collectively reached interruption may retain its distinct resumable status.
- md_tools.remd.mpi remains the only module that imports mpi4py or decides barriers, gathers, and aborts.

Use injected failures on individual non-root and root ranks inside actual propagation/exchange/report/checkpoint/finalization paths. Run them under a real MPI launcher with timeouts and prove that no rank survives or hangs.

## 8. Torsion collective-variable reporting from cv.yaml

Implement a first, deliberately narrow CV schema. Version 1 supports fixed torsions only. Do not add biasing, forces, PLUMED, or another simulation engine. CV reporting is observation and must not alter the Hamiltonian.

### Configuration

Add a strict top-level section to every MD protocol configuration:

    collective_variables:
      file: null
      interval_steps: 0

Both values disable reporting by default. Reporting is enabled only when file is non-null and interval_steps is greater than zero. Supplying only one is an error. Unknown fields and duplicate YAML keys are refused.

The path is resolved deterministically relative to the MD configuration file at build time. Copy a content-addressed definition into the generated script directory so the generated workflow remains portable, and record both the original facts and copied digest. At runtime the copied definition is revalidated and included in run/checkpoint fingerprints.

A cv.yaml example:

    schema_version: 1
    collective_variables:
      - name: phi
        type: torsion
        atom_indices: [4, 6, 8, 14]
      - name: psi
        type: torsion
        atoms:
          - {chain: A, residue: "1", atom: N}
          - {chain: A, residue: "1", atom: CA}
          - {chain: A, residue: "1", atom: C}
          - {chain: A, residue: "2", atom: N}

Each entry must have:

- a unique stable name valid as a CSV column;
- type torsion;
- exactly one of atom_indices or atoms;
- exactly four distinct zero-based indices, or four selectors that each resolve to exactly one topology atom;
- indices in range;
- selectors with explicit chain, residue identifier, and atom name; support insertion code only if represented unambiguously by the topology;
- no silent first-match behavior.

Only torsion is accepted in schema version 1. Unsupported CV types must be refused by name with a forward-compatible message.

Values are geometric dihedrals in degrees, wrapped to [-180, 180). For periodic systems, construct the three sequential bond vectors using the triclinic minimum-image convention before evaluating the dihedral. Test the sign convention against MDTraj or an independently calculated reference on nonperiodic, orthorhombic, and triclinic examples.

Do not add CustomTorsionForce or any other Force to the System. CV evaluation must not change forces, energies, force groups, checkpoints, or dynamics. It may retrieve positions already needed at the reporting point. Record CV evaluation count and wall time separately; do not count a position-only torsion evaluation as an energy evaluation.

### Exact schedules and outputs

The interval is independent of solute trajectory and system-state reporting. It may therefore be more frequent than the whole-system trajectory.

Require exact schedules:

- for every enabled cMD dynamics stage, interval_steps divides the stage step count;
- for REST2/rREST2, interval_steps divides exchange_interval_steps;
- for AIS, interval_steps is on the parameter-update grid and divides switching_steps;
- always include step 0 and the final step exactly once;
- minimization produces no CV time series.

Write ordinary UTF-8 CSV with a stable schema and no comment lines. Write a resolved YAML/JSON sidecar containing schema version, original cv.yaml digest, resolved atom indices, units, periodic convention, and output-column order.

cMD, one file per dynamics stage:

    <stage>.cv.csv

Columns begin with:

    step,time_ps,trajectory_frame_index

trajectory_frame_index is empty when no trajectory frame exists at that step, followed by the CV names.

REST2/rREST2, one file per fixed thermodynamic state:

    remd0.cv.csv ... remdN.cv.csv

Columns begin with:

    step,time_ps,exchange_attempt,state_index,tau,walker_index,exchange_phase,trajectory_frame_index

Define and document whether an exchange-boundary row is pre-exchange or post-exchange. Use one convention everywhere and prevent duplicate rows. State files follow thermodynamic states, while walker_index records which walker supplied the configuration.

AIS, one file per path plus one aggregate assembled only from verified completed manifests:

    path_NNNN/cv.csv
    AIS_cv.csv

Columns begin with:

    path_index,source_frame_index,protocol_step,time_ps,tau,observation_index,coordinate_frame_index

The observation/frame indices are empty when the independent CV cadence does not coincide with an AIS observation. When they are present, the CV must be evaluated on exactly that saved coordinate. Never attach a CV measured at x_j to a different observation coordinate.

### Restart, identity, and completion

CV definition and schedule are scientific analysis provenance and must be bound into:

- resolved.config;
- stage/ladder/AIS run identity;
- continuation fingerprints;
- output inventories;
- checkpoint stream counts;
- completion manifests and sha256 records;
- MD-data registration metadata.

On cMD and AIS resume, truncate CV CSVs to the count committed by the selected checkpoint generation before appending. REST2/rREST2 continuation must likewise avoid duplicate or missing CV rows. A changed CV definition, atom mapping, sign/unit convention, or interval must refuse continuation into the same run.

CV output failures are simulation failures; do not silently disable reporting.

## 9. Tests for CV reporting

Add CPU unit/integration tests and real CUDA tests covering:

- strict cv.yaml parsing, duplicate keys, unsupported types, duplicate names, invalid indices, ambiguous/missing selectors;
- atom-selector resolution with chain/residue identity;
- known torsion values and sign convention;
- periodic minimum-image behavior for orthorhombic and triclinic boxes;
- proof that enabling CVs does not change System serialization, force inventory, energies, or deterministic short-trajectory coordinates under the same seeds;
- independent CV cadence more frequent than trajectory cadence;
- exact step-0/final inclusion and no duplicates;
- cMD split and all-in-one output;
- REST2 and rREST2 state-index/walker-index correctness across accepted exchanges;
- AIS tau/path/source/observation/frame alignment;
- cMD, ladder, and AIS interruption/resume without duplicated CV rows;
- completion refusal after CV truncation, mutation, deletion, or definition changes;
- full overwrite cleanup and unrelated-file preservation;
- wheel-installed runs outside the checkout.

Use an analytically checkable four-atom geometry and an ALA torsion example. Do not validate only that a column exists; compare every reported value with an independent coordinate-based calculation.

## 10. CUDA, MPI, packaging, and evidence

Run the complete repository test matrix. Do not weaken, skip, xfail, delete, or mock scientific CUDA/MPI coverage to obtain green results.

Required lanes:

1. Complete fast/non-GPU suite.
2. Complete slow/GPU suite on real CUDA devices.
3. CUDA cMD, fixed-tau cMD, REST2, rREST2, and AIS, with CV reporting enabled and disabled.
4. Every source function that constructs a CUDA Context, integrates, minimizes, evaluates energy, updates Context parameters, saves/loads checkpoints, reports from CUDA-derived state, or derives scientific output from CUDA state must be classified in the machine-readable CUDA coverage inventory and exercised or explicitly proven not to reach a device.
5. Real MPI REST2/rREST2 and AIS at multiple ranks, including rank-to-device placement, exchanges, CV state/walker labels, checkpoint/resume, and injected rank-local failures with a timeout.
6. Build a wheel, install it into a clean environment, run outside the checkout, verify import origin, exercise all four public commands, and run short installed-wheel cMD, REST2, rREST2, and AIS workflows with cv.yaml.
7. Exact-head GitHub Actions after the cleanup commit.

Record exact commands, pass/fail/skip/deselect counts, wall times, GPU models/device IDs, OpenMM/CUDA/MPI versions, rank counts, and commit SHA. Separate local CUDA/MPI evidence from CPU-only GitHub Actions. Do not claim convergence or production sampling from short mechanism tests.

## 11. Documentation and delivery

Update CLAUDE.md, README, method documentation, all four example MD configurations, generated-input documentation, release notes, and MD-data contract documentation.

Document:

- the v1 torsion-only scope;
- cv.yaml examples using indices and selectors;
- output names and columns for cMD, REST2/rREST2, and AIS;
- state versus walker semantics;
- degrees and wrapping convention;
- periodic minimum-image convention;
- independent reporting cadence;
- restart and identity behavior;
- CV evaluation cost;
- the AIS non-scaled, sqrt-scaled, and lin-scaled work/potential columns for HS reweighting.

Do not rename lin-scaled to generic scaled: the latter is ambiguous. Preserve the exact AIS decomposition identity and work convention.

Commit coherent implementation and test slices to dev. Keep CLAUDE_TASK.md while working. Remove it only in a separate final cleanup commit after every acceptance criterion above passes, the installed-wheel tests pass, the real CUDA/MPI suite passes, and CI for the implementation is green. Then wait for CI on the exact cleanup SHA and require that run to be green.

The final report must provide:

- implementation commit SHA(s);
- cleanup commit SHA;
- exact-head CI URL;
- exact commands, counts, skips, wall times, and hardware;
- direct evidence for every corrected lifecycle refusal;
- CUDA-function coverage evidence;
- MPI non-hang/fail-closed evidence;
- CV numerical, alignment, restart, and installed-wheel evidence;
- AIS decomposition/HS evidence;
- every remaining limitation.
