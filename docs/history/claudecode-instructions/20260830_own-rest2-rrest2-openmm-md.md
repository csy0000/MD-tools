# Own REST2 and reservoir REST2 through openmm-md

Date: 2026-08-30

## Starting point

Work across these repositories:

- https://github.com/csy0000/MD-templates
- https://github.com/csy0000/MD-project

Before changing anything, fetch and verify:

- MD-templates branch feat/openmmtools-rest2 at 068ef2f0538a20260e2d40b566ca862e50c6b2c8;
- MD-project branch dev4-openmmtools-rest2 at the commit containing this instruction. Its completed production-correctness implementation parent is f76c498759bcba7ee11dcd64e4c1266cdf8d1804.

Read every applicable CLAUDE.md and the completed journals for the OpenMMTools REST2 work. Preserve its verified scientific and operational lessons. Create the new MD-templates branch from the exact commit above and the new MD-project branch from the dev4 head that contains this instruction:

- MD-templates: feat/openmm-md-rest2-rrest2
- MD-project: dev5-openmm-md-rest2-rrest2

Do not merge into dev or main. Do not rewrite or force-push history.

This instruction supersedes the earlier architectural decision that OpenMMTools ReplicaExchangeSampler must be the production REST2 controller. OpenMMTools may remain an optional reference oracle in tests, but the generated production calculation must not depend on ReplicaExchangeSampler, MultiStateReporter, or an openmm-rest2 executable.

## Purpose

Implement a small, owned replica-exchange runtime that supports:

1. conventional REST2;
2. Boltzmann-reservoir REST2, abbreviated rREST2 in this repository;
3. later exchange rules such as non-Boltzmann reservoirs and kinetic-reservoir REST2 without adding another executable.

There must be one simulation executor:

    openmm-md

Single-system MD continues to use the existing Amber-like file interface. Multi-replica methods use the same executable with an Amber-like group file:

    mpiexec -n 6 openmm-md \
        -ng 6 \
        --groupfile REST2/rest2.group \
        -o REST2/rest2.out \
        -x REST2/rest2.nc \
        -r REST2/restart.json \
        --checkpoint REST2/rest2_checkpoint.nc

A changed transition rule is selected explicitly:

    mpiexec -n 6 openmm-md \
        -ng 6 \
        --groupfile rREST2/rrest2.group \
        --exchange-rule rREST2/rrest2_exchange.py \
        --reservoir rREST2/reservoir.yaml \
        -o rREST2/rrest2.out \
        -x rREST2/rrest2.nc \
        -r rREST2/restart.json \
        --checkpoint rREST2/rrest2_checkpoint.nc

Do not retain openmm-rest2 as a public compatibility command. Generated documentation, launchers, tests, project workflow and help text must converge on openmm-md. If a temporary private adapter is required during migration, remove it before completion.

## Naming and scientific definitions

REST2 means Hamiltonian replica exchange at one physical thermostat temperature. Persist tau as the source parameter:

    s = (1 - tau)^2
    sqrt(s) = 1 - tau

Use the existing audited REST2 scaling implementation and omega-exclusion convention. Do not implement REST2 as temperature REMD.

rREST2 in this milestone means REST2 whose tau_max state is periodically refreshed from a finite, Boltzmann-weighted configuration reservoir sampled from an equilibrium fixed-tau cMD run at exactly the same tau, temperature, Hamiltonian and fixed-volume ensemble as the top REST2 state.

The intended source is the existing cMD_tau0p5-style run used to seed AIS when tau_max is 0.5. The source path is not evidence. Its companion runtime record, topology identity, tau, temperature, ensemble and frame-time map are evidence.

For v1:

- reservoir configurations are complete configurations, not solute-only structures;
- explicit-solvent reservoirs are NVT and carry positions plus exact periodic box vectors;
- implicit GBn2 reservoirs have no box;
- source and top rung must use the same tau and temperature;
- source and top rung must have identical topology, atom order, particle count and Hamiltonian identity;
- reservoir frames are drawn uniformly from a Boltzmann-weighted source window;
- velocities are not read from DCD and are redrawn from the Maxwell distribution at the common physical temperature with recorded deterministic seeds;
- a top-rung reservoir refresh is accepted with probability one only under the recorded Boltzmann-reservoir/same-state contract;
- the finite-reservoir approximation and the assumption that the selected source window represents the top-state equilibrium distribution must be stated in every resolved record and in the documentation.

Refuse rather than approximate:

- an explicit NPT reservoir;
- partial-solute insertion into an existing solvent configuration;
- a source at the wrong tau or temperature;
- missing source-state evidence;
- a topology, atom identity, force-field or box mismatch;
- a non-Boltzmann, clustered, weighted or kinetic reservoir without a separately implemented and tested acceptance rule.

Do not call rREST2 exact merely because the code accepts each refresh. Literature distinguishes Boltzmann-weighted and non-Boltzmann reservoirs and shows that a wrong reservoir criterion biases every replica. Cite and discuss at least:

- Roitberg, Okur and Simmerling, J. Phys. Chem. B 2007, DOI 10.1021/jp068335b
- Kasavajhala, Lam and Simmerling, J. Chem. Inf. Model. 2020, PMCID PMC7725893

## One executor, two modes

Preserve the existing single-run contract:

    openmm-md -i INPUT.py -p topology.pdb -s system.xml -c start.xml \
        -o stage.out -x trajectory.nc -r final_state.xml --checkpoint stage.chk

Add a grouped mode activated only by --groupfile:

    openmm-md -ng N --groupfile FILE [--exchange-rule RULE.py] \
        -o run.out -x run.nc -r restart.json --checkpoint run_checkpoint.nc

The executable remains generic. It may own path resolution, group parsing, MPI coordination, propagation scheduling, reporter/checkpoint lifecycle, failure handling and invocation of a transition rule. It must not contain a hard-coded REST2 ladder, reservoir policy, force field, temperature or scientific default.

Mode-specific validation must be explicit:

- --groupfile requires -ng;
- -ng equals the number of non-comment group lines;
- under MPI, the world size must match the supported execution policy and every rank records its assignment;
- grouped-only flags are rejected in single mode;
- single-only ambiguous combinations are rejected;
- --resume, --extend and --verify-only work for grouped mode through openmm-md;
- --force never combines with resume or extension;
- existing output refusal is preserved;
- all output/input alias and path-clobber checks are preserved.

The installed public scripts remain md-template and md-openmm for generation plus openmm-md for simulation. Do not add a method-specific executable.

## Group-file contract

Use a plain text, one-group-per-line format parsed with shlex. Never evaluate it with a shell. Blank lines and lines beginning with # are ignored.

A generated six-state REST2 group file should remain understandable:

    -i REST2/rest2.py -p common/topology.pdb -s common/system.xml -c eq/npt_free/npt_free.state.xml --group-index 0
    -i REST2/rest2.py -p common/topology.pdb -s common/system.xml -c eq/npt_free/npt_free.state.xml --group-index 1
    -i REST2/rest2.py -p common/topology.pdb -s common/system.xml -c eq/npt_free/npt_free.state.xml --group-index 2
    -i REST2/rest2.py -p common/topology.pdb -s common/system.xml -c eq/npt_free/npt_free.state.xml --group-index 3
    -i REST2/rest2.py -p common/topology.pdb -s common/system.xml -c eq/npt_free/npt_free.state.xml --group-index 4
    -i REST2/rest2.py -p common/topology.pdb -s common/system.xml -c eq/npt_free/npt_free.state.xml --group-index 5

All science remains in the concise, path-independent Python protocol. The group file owns concrete paths and initial group assignment. The launcher sources paths.sh and invokes openmm-md with explicit run-level output paths. Do not embed an absolute $MD_DATA path in committed or generated scientific input.

Group indices must be unique, contiguous zero-based integers. Shared input paths are allowed. Unknown group flags, missing values, duplicate fields and output flags inside a group line are refused with the line number. Run-level -o, -x, -r and --checkpoint occur once on the outer command, because they describe the coordinated run.

## Concise grouped protocol

Define the narrowest readable protocol contract. A generated REST2 input should be approximately this simple and contain no path:

    from replica_runtime import REST2Protocol

    protocol = REST2Protocol(
        tau=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        temperature_k=300.0,
        timestep_fs=4.0,
        segment_ps=2.0,
        exchange_interval_ps=10.0,
        whole_output_interval_ps=10.0,
        number_of_exchanges=1000,
    )

The exact symbol names may change if inspection shows a smaller contract, but generated input must not contain the exchange loop, NetCDF implementation, MPI plumbing, source parsing or concrete filesystem paths.

A grouped protocol describes thermodynamic states and propagation. An exchange-rule file describes proposals and acceptance. Do not build a plugin registry, dependency injector, workflow engine or general method database.

Generated projects remain standalone: they import no md_templates checkout. Copy only the small runtime modules they need.

## Owned replica runtime

Separate these responsibilities into small modules with tested interfaces:

1. protocol/state construction;
2. replica propagation;
3. reduced-potential evaluation;
4. walker-to-state mapping;
5. transition-rule proposal and acceptance;
6. reservoir preparation and access;
7. reporter and checkpoint storage;
8. restart/extension and validation;
9. cumulative statistics and provenance.

Do not produce one large REST2 class that also implements reservoirs, NetCDF, MPI and CLI parsing.

OpenMM owns System, Context, Integrator, State and energy evaluation. MD-templates owns the replica scheduling and transition rules. OpenMMTools is not imported by generated production code.

### Conventional REST2 rule

The default grouped transition rule is conventional neighboring REST2 exchange with a deterministic recorded odd/even pairing schedule. For states i and j at the one common beta, compute:

    log_alpha =
        -u_i(x_j) - u_j(x_i)
        +u_i(x_i) + u_j(x_j)

and accept when log(U) < min(0, log_alpha), using a dedicated recorded RNG stream.

The implementation must independently evaluate all four required reduced potentials. It must not infer cross energies by scaling the current energy. Positions and periodic box vectors are one configuration. Because every rung has the same physical temperature, do not apply temperature-REMD velocity rescaling.

Choose and document one representation consistently:

- fixed thermodynamic-state contexts with complete configurations exchanged; or
- continuous walkers whose state assignments exchange.

Whichever is chosen, record enough mapping to reconstruct both walker and thermodynamic-state views, and test that the two views agree. Every mapping row must be a permutation.

### Exchange-rule contract

A rule receives a bounded state object containing the iteration, thermodynamic states, walker mapping, required reduced potentials, dedicated RNG and optional prepared reservoir. It returns explicit proposals and decisions. It may persist a small JSON-serializable rule state.

At minimum support:

- the built-in conventional neighboring rule when --exchange-rule is omitted;
- the generated rREST2 rule selected with --exchange-rule.

Record rule name, format version, source-file SHA-256, parameters, proposal counts, acceptance counts, RNG identity and restart state. A resumed or extended run refuses any change.

Future non-Boltzmann and kinetic-reservoir methods must fit this boundary without changing openmm-md.

## Share the AIS source-ensemble implementation

The current generated AIS runtime already implements much of the correct reservoir-input boundary. Extract the reusable parts into one standalone generated helper used by both AIS and rREST2. Do not copy them into a second implementation.

The shared helper must cover:

- resolving source trajectory and topology paths;
- finding and reading the companion resolved_run.yaml;
- identifying a fixed-tau cMD or replica source;
- establishing physical frame times from authoritative records or explicit fields;
- inclusive start/end time-window selection;
- deterministic selection with and without replacement;
- bounded mdtraj.iterload surveying and reading;
- atom identity and connectivity comparison;
- topology and particle-count validation;
- source tau and temperature validation;
- exact explicit-solvent box-vector preservation and reduction;
- materializing selected frames once into a small prepared reservoir;
- recording selected source indices, times, frame count, source size, chunk size, chunks read and configuration identity.

Preserve the existing AIS safety rules:

- never mdtraj.load a production trajectory;
- never hash the production trajectory at runtime;
- never treat a frame index or DCD header as physical-time evidence;
- never infer source tau from a directory name such as cMD_tau0p5;
- after preparation, never reopen the production source;
- never silently regenerate prepared inputs that disagree with their manifest;
- prepared small files may be fully read and hashed;
- DCD carries no velocities;
- explicit box vectors come from recorded exact values, not reconstructed DCD lengths and angles.

Refactor AIS to use the shared helper without changing its scientific results, selected-frame determinism or generated public layout except where the shared prepared-source format is deliberately adopted and documented. Run all existing AIS tests.

For rREST2, materialize a prepared reservoir before propagation, for example:

    rREST2/reservoir/
        configurations.dcd
        reservoir.yaml

The exact names may change, but the manifest must state the source evidence, selection, topology identity, tau, temperature, ensemble, box convention, reservoir weighting and finite-reservoir assumptions.

## rREST2 transition rule

The rREST2 rule combines the conventional neighboring REST2 schedule with a separately scheduled reservoir refresh of the current walker occupying tau_max.

At each reservoir attempt:

1. establish which walker currently occupies the top thermodynamic state;
2. select a prepared reservoir frame with the rule's dedicated recorded RNG;
3. validate the prepared reservoir identity against the top state;
4. replace the complete top-state configuration, including box vectors under explicit NVT;
5. draw fresh velocities at the common temperature from a separate recorded seed;
6. record the selected reservoir frame, displaced walker, state index, seeds and outcome;
7. continue the normal REST2 mapping without pretending the reservoir is another propagated replica.

Define whether a reservoir attempt happens before or after the ordinary neighboring exchange at a coincident iteration. Choose one order, document it, persist it, and test it. Never let process scheduling decide the order.

Do not insert only the solute into existing explicit solvent. Do not minimize or relax a proposed frame silently. Do not fabricate intermediate frames.

## Owned storage and restart

Do not use MultiStateReporter as production storage. Implement one documented NetCDF schema owned by MD-templates, using netCDF4 or another minimal maintained dependency already justified by the repository.

The analysis file -x must contain or reference enough information for:

- real stored coordinate frames;
- iteration and physical time;
- thermodynamic-state identities and tau ladder;
- walker-to-state mapping;
- reduced potentials required for decisions;
- ordinary exchange proposals and acceptances;
- reservoir attempts and selected frame indices;
- cumulative statistics;
- reporter/schema version.

The grouped checkpoint must preserve all state required for bitwise-consistent logical continuation where the platform permits it:

- positions, velocities and box vectors for every walker;
- mapping;
- integrator or OpenMM checkpoint state as appropriate;
- propagation iteration and original budget;
- all RNG states, including exchange and reservoir selection;
- exchange-rule persistent state;
- reporter commit point;
- scientific identity and input checksums.

Retain the corrected three-record separation:

- scientific identity is committed before propagation;
- an atomic run-state sidecar records initialized, running, interrupted, failed or completed;
- -r is an atomic completed-run manifest and is not required to resume an interrupted run.

A signal or exception must leave the last fully committed iteration resumable. Resume completes the original budget. Extension requires a completed run and adds an explicit number of exchange attempts. Neither resets statistics or RNG streams.

Implement one authoritative validator reachable through:

    openmm-md --verify-only -x RUN.nc --checkpoint CHECKPOINT.nc -r restart.json

It must open and read the files, not check existence. MD-project delegates to this validator.

## Output and statistics

The human-readable -o file must be comparable to Amber output and include:

- resolved command and mode;
- input and output files;
- protocol and exchange-rule identity;
- system and force-field identity;
- tau ladder and derived s values;
- timestep, segment, exchange and output intervals;
- group/rank/device assignment;
- completed physical time per replica;
- ordinary and reservoir proposal/acceptance statistics;
- walker/state round trips;
- reservoir frame usage;
- resume/extension history;
- exact software versions;
- final completion status.

Lifetime statistics come from authoritative stored event history and survive resume and extension. A conventional round trip remains cold -> hot -> cold. A reservoir refresh is not itself a thermodynamic-state round trip.

## Required ALA acceptance systems

Generate and run four short correctness/smoke workflows. They are not convergence evidence.

### 1. ALA implicit REST2

- ACE-ALA-NME peptide;
- ff14SB + GBn2/mbondi3, no SASA;
- whole-system REST2 scaling;
- NVT;
- six states, tau 0.0 through 0.5;
- short CUDA propagation with multiple accepted/rejected exchange opportunities.

### 2. ALA implicit rREST2

- same system and ladder;
- reservoir from a preceding fixed-tau cMD_tau0p5 run;
- prepared by the shared AIS/rREST2 source reader;
- several reservoir attempts;
- verify frame selection, top-state replacement, velocity redraw, mapping continuity, storage and restart.

### 3. ALA explicit NVT REST2

- ACE-ALA-NME;
- ff19SB + OPC;
- fixed periodic box, no active barostat during REST2;
- six states, tau 0.0 through 0.5;
- complete configuration exchanges with box preservation.

### 4. ALA explicit NVT rREST2

- reservoir from fixed-box ff19SB/OPC cMD at tau=0.5;
- identical topology, System, temperature and box contract;
- full-system reservoir configurations;
- reject changed boxes, NPT source records and solute-only reservoirs.

Use picosecond smoke sizes. Every test that integrates runs on CUDA and carries the repository's gpu marker. CPU or Reference may test parsing and exact energy arithmetic only. Run a six-rank CUDA/MPI smoke when the machine supports it and record any genuine environmental limitation.

## Required focused tests

Add tests that would have failed before this change:

- openmm-md single mode remains backward compatible;
- openmm-rest2 is absent from installed and generated public commands;
- groupfile parsing uses shlex and never shell evaluation;
- group count/index/MPI mismatch is refused;
- grouped input/output aliasing is refused;
- unknown group and exchange-rule fields are refused;
- generated REST2 and rREST2 inputs contain no concrete paths;
- default neighboring pairing covers every eligible pair over odd/even phases;
- analytical and independently evaluated REST2 log acceptance agree;
- same-temperature exchanges do not rescale velocities;
- mapping rows remain permutations;
- walker and state views reconstruct consistently;
- exchange decisions repeat under a fixed seed;
- source tau comes from evidence, not the cMD_tau0p5 directory name;
- missing or conflicting source evidence is refused;
- AIS and rREST2 select the same frames for the same shared request and seed;
- production sources are read with bounded iterload and never hashed;
- prepared reservoirs remain usable after the production source is removed;
- changed prepared-source identity is refused rather than regenerated;
- implicit Boltzmann-reservoir refresh occurs only at tau_max;
- explicit refresh preserves the complete configuration and exact box;
- explicit NPT, box mismatch and solute-only insertion are refused;
- reservoir selection and velocity seeds survive restart;
- interrupted grouped runs resume without -r;
- extension preserves cumulative ordinary and reservoir statistics;
- corrupt/truncated analysis and checkpoint files fail authoritative validation;
- rREST2 cannot be selected without --exchange-rule and --reservoir;
- a non-Boltzmann reservoir declaration is refused by the v1 rule;
- MD-project workflow invokes openmm-md --groupfile and delegates verification.

Use OpenMMTools only in optional test code to compare conventional REST2 reduced potentials, mappings or distributions. The production package and generated runtime must work without OpenMMTools installed.

## MD-project integration

After MD-templates is implemented, tested, committed and pushed:

1. pin MD-project to the new full MD-templates commit;
2. replace openmm-rest2 calls with openmm-md --groupfile;
3. add separate ALA REST2 and rREST2 requests for implicit and explicit NVT smoke/dry workflows;
4. update Snakemake scheduling and verification without duplicating the owned NetCDF schema;
5. update README and architecture documentation with the Amber comparison;
6. document that $MD_DATA holds physical outputs and the repository contains portable requests and paths;
7. keep generated trajectories, reservoirs, checkpoints and component clones ignored.

Every documented command must exist.

## Non-goals

Do not implement in this milestone:

- non-Boltzmann or clustered reservoirs;
- kinetic-reservoir REST2;
- solute-only explicit-solvent grafting;
- NPT reservoir exchange;
- temperature REMD;
- asynchronous exchange;
- multiple reservoirs;
- adaptive tau ladders;
- production convergence claims;
- MD-analysis changes;
- an exchange-rule registry or workflow framework.

The architecture must make these later rules possible without changing openmm-md, but placeholder implementations are forbidden.

## Completion order

1. Inspect current code, journals and OpenMM/OpenMMTools reference behavior.
2. Implement and test MD-templates.
3. Run formatting, linting, packaging and all applicable non-GPU tests.
4. Run all applicable CUDA tests.
5. Run the four ALA acceptance workflows.
6. Run available MPI acceptance.
7. Commit logically and push feat/openmm-md-rest2-rrest2.
8. Update MD-project to the exact resulting commit.
9. Run project tests, validators and Snakemake dry runs.
10. Commit logically and push dev5-openmm-md-rest2-rrest2.
11. Do not merge either branch.

Create the execution journal:

    docs/journals/20260830_own-rest2-rrest2-openmm-md.md

It must record exact starting and ending SHAs, commands, dependency versions, test counts/results/skips, CUDA devices, MPI layout, generated examples, runtime evidence, storage schema, scientific assumptions, deviations, unresolved gaps and every pushed commit.

Stop and report instead of inventing behavior if source evidence, OpenMM semantics, a rigorous acceptance rule, NetCDF continuation, available GPU/MPI resources or a repository contract cannot safely support a requirement.
