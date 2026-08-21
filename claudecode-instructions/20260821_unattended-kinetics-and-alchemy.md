# Claude Code instruction: unattended alanine/RGD kinetics and alchemical prototype

Date: 2026-08-21  
Repository: `csy0000/MD-templates`  
Control branch: `dev`  
Expected starting commit: `0d469f881b53f45b177da8937f0f0d471f926488`, or a direct descendant containing this instruction

## Outcome

Use the next approximately two unattended days to:

1. run three independent 1 microsecond explicit-solvent conventional-MD replicates of alanine
   dipeptide;
2. run three independent 1 microsecond explicit-solvent conventional-MD replicates of cyclic
   RGDfV;
3. build a local prototype of a future `MD-analysis` repository and analyze the six trajectories
   with Deeptime;
4. prototype and, only after strict endpoint/pilot validation, run two RGDfV alchemical
   transformations with 21 lambda windows and 10 ns of sampled production per window:
   RGDfV -> reduced_RGDfV and RGDfV -> cilengitide;
5. analyze each valid alchemical leg with PyMBAR, FastMBAR, and thermodynamic integration.

There are nine GPUs on the machine. Use six initially for the six cMD replicates and three as a
dynamic alchemical worker pool. As cMD jobs finish, their GPUs may join the alchemical queue.

This is an unattended task. The user will not answer questions for two days. **Do not stop all work
to wait for clarification.** When one subtask is genuinely ambiguous or blocked, document the exact
blocker, make no unsafe scientific guess, and continue every independent task that can be completed.
Never fabricate a chemical structure, atom map, successful result, or validation.

## Relationship to the accepted MD template

Read completely before acting:

- `CLAUDE.md`, `README.md`, `docs/configuration.md`;
- `claudecode-instructions/20260821_close-remaining-cmd-acceptance-gaps.md`;
- `docs/journal/2026-08-21_remaining-cmd-acceptance-gaps.md` and its linked cMD journals;
- the current system/input generators, profiles, cMD stage/continuation implementation, reporting,
  run-state, deterministic bundle generation, tests, and worked alanine/RGD examples.

Verify remote `dev`, exact SHA, clean working tree, and the accepted test/CI state. Use the current
accepted generator and committed-generation continuation contract; do not reopen the architecture
or replace it with an experiment-only runner. If the experiment exposes a reproducible core defect,
write a regression test and a focused fix, but do not refactor unrelated code.

Record the exact `MD-templates` commit used by every generated bundle and run.

## Local prototype and data layout

Create a sibling local repository, not another branch inside `MD-templates`:

```text
../MD-analysis-prototype-20260821/
  README.md
  CLAUDE.md
  pyproject.toml or environment files
  configs/
  src/md_analysis/
    kinetics/
    alchemy/
  workflows/
  tests/
  reports/
  docs/journal/
  STATUS.md
```

Initialize it as a local Git repository with `main` and `dev`; work and commit on local `dev`. Do
not create or push a remote repository while the user is away. This prototype is intended for later
review and extraction into a separate `MD-analysis` repository.

Keep simulation data outside both Git worktrees, for example:

```text
../MD-analysis-data/20260821_unattended/
  cmd/ala/{replicate_01,replicate_02,replicate_03}/
  cmd/rgdfv/{replicate_01,replicate_02,replicate_03}/
  alchemy/rgdfv_to_reduced/
  alchemy/rgdfv_to_cilengitide/
```

The local analysis repository may contain small configuration, provenance, summary tables, plots,
tests, and reports. It must not contain DCDs, checkpoints, serialized States, large reduced-potential
matrices, environments, or copied simulation bundles. Reference data through manifests or relative
paths that can later be changed. Add appropriate `.gitignore` rules.

The permanent ownership boundary is:

- simulation generation/execution, including the eventual alchemical protocol: `MD-templates`;
- trajectory, kinetics, MBAR/FastMBAR/TI analysis: future `MD-analysis` repository.

Prototype code may temporarily coexist locally, but keep simulation and analysis modules separated
so they can be moved without rewriting them.

## Machine inspection and unattended execution law

Before launching anything:

1. inspect all GPUs with `nvidia-smi`, including index, UUID, model, memory, utilization, and active
   compute processes;
2. inspect CPU, RAM, free disk, installed OpenMM/platforms, CUDA driver/runtime, and available
   Deeptime/PyMBAR/FastMBAR/alchemical dependencies;
3. set `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
4. map logical workers to currently idle physical GPU UUIDs and record the mapping;
5. never start a simulation on an occupied GPU and never kill or disturb an unrelated process;
6. enforce one simulation process per GPU with a lock file or equally reliable supervisor;
7. estimate throughput and disk growth from a short measured pilot before launching the full queue;
8. verify sufficient disk space for all planned outputs with a safety margin.

Use a durable supervisor such as named tmux sessions plus a manifest/worker queue. Record session
names, PIDs, GPU UUIDs, commands, start times, log paths, current generation/window, and exact resume
commands in `STATUS.md`. A failed worker must not terminate or duplicate other jobs. Retry only
understood transient failures. Preserve checkpoints and diagnostics.

If work remains active when the Claude session must return, leave it running safely and return a
`RUNNING` handoff with commands for status, logs, stop, and resume. Do not terminate valid work just
to produce a final report.

## Part A: six independent 1 microsecond cMD replicates

### Alanine dipeptide: three replicates

- system: capped alanine dipeptide used by the accepted worked example;
- protein force field: ff19SB;
- water: OPC with compatible ions;
- salt: 0.15 M NaCl;
- periodic dodecahedral/truncated-octahedral geometry contract;
- minimum 12 angstrom solute padding;
- 10 angstrom real-space cutoff;
- 300 K and 1 bar;
- accepted explicit-production HMR/4 fs protocol and CUDA mixed precision;
- one independently generated/equilibrated production chain per replicate.

### Cyclic RGDfV: three replicates

- use the repository's validated RGDfV stereochemistry and cyclic connectivity;
- Sage/OpenFF 2.2 with its approved plain TIP3P default and compatible ions;
- otherwise use the same explicit-solvent box, salt, cutoff, temperature, pressure, HMR/4 fs, and
  CUDA mixed-precision conventions where supported by the validated RGD profile;
- do not silently fall back to ff19SB or OPC for the Sage-parameterized macrocycle.

### Replicate and reporting contract

For all six runs:

- three unique recorded master seeds per system;
- independently seeded equilibration and velocities, not three copies of one production checkpoint;
- exactly 1 microsecond production per replicate;
- 20 committed 50 ns segments in one run directory per replicate;
- full-system coordinates every 100 ps: 10,000 frames at completion;
- solute coordinates every 10 ps: 100,000 frames at completion;
- state/energy log at a documented cadence no coarser than 100 ps;
- checkpoint-preferred continuation, atomic generation commits, one log header, monotonic absolute
  step/time/frame/invocation accounting, and no duplicate boundary frame;
- finite energies, temperature, volume/density, and no NaN/Inf;
- exact bundle/configuration/seed/software/GPU provenance.

Launch the six runs concurrently on six distinct idle GPUs after short end-to-end pilots pass. If
the six microseconds do not finish within two days, continue from valid checkpoints later; never
shorten, concatenate incorrectly, or relabel an incomplete trajectory as 1 microsecond.

The HMR/4 fs trajectories validate the template and analysis workflow. Label their transition rates
as protocol-dependent; do not claim they are an HMR-independent physical kinetic benchmark.

## Part B: Deeptime kinetics analysis

Use the stable installed Deeptime release in an isolated reproducible environment and record exact
versions. Use MDTraj or MDAnalysis for topology-aware torsions, with tests on known coordinates.

Analyze each replicate separately and the three replicates for one molecule as a list of independent
trajectories. Never concatenate across replicate or restart boundaries. Confirm that restart
boundaries themselves do not create feature discontinuities.

### Alanine features and models

- backbone phi/psi, represented periodically with sine/cosine;
- direct phi/psi discretization as a transparent baseline;
- optional TICA/VAMP model as a comparison, not a mandatory source of artificial dimensions;
- multiple cluster counts and random seeds;
- reversible MSMs over lag times compatible with the 10 ps stride, beginning with
  10, 20, 50, 100, 200, 500, and 1000 ps and extending only where supported by data;
- implied-timescale, spectral-gap, connectivity, count, and Chapman-Kolmogorov diagnostics;
- PCCA+ macrostate count chosen from spectral/robustness evidence rather than hard-coded;
- stationary populations, free-energy surface, transition matrix, MFPT/rates, and uncertainties.

### RGDfV features and models

- all relevant cyclic-backbone and side-chain torsions with sine/cosine embedding;
- document the exact atom quadruplets and stereochemical convention;
- compare transparent torsional clustering with TICA/VAMP reduction;
- scan lag times from 10 ps into the ns regime as supported by observed relaxation and counts;
- do not force a Markov or PCCA+ model when the three microseconds lack state connectivity;
- provide the same population, implied-timescale, CK, PCCA+, MFPT/rate, and uncertainty diagnostics
  when statistically supported.

Use trajectory-aware block bootstrap or Bayesian MSM uncertainty. Include convergence at available
prefixes (for example 250, 500, 750, and 1000 ns) and leave incomplete-prefix analyses honest when
runs are still active. Generate machine-readable tables plus concise plots and a Markdown/HTML
report. Separate equilibrium population conclusions from kinetic conclusions.

## Part C: alchemical transformations

### Scientific target

Prototype:

1. cyclic RGDfV -> reduced_RGDfV;
2. cyclic RGDfV -> cilengitide, defined here as the same cyclic RGDfV scaffold with N-methylation
   at the Val amide nitrogen, retaining the validated D-Phe and all other stereochemistry.

Use Sage/OpenFF 2.2 endpoint parameters, plain TIP3P water, compatible ions, 0.15 M NaCl, and the
same explicit-solvent geometry/cutoff contract. Confirm endpoint net charges and do not proceed if a
transformation changes total charge without an explicitly implemented and validated correction.

For each transformation calculate two legs:

- explicit aqueous leg;
- nonperiodic vacuum leg.

Use the recorded sign convention

```text
Delta G = G_product - G_RGDfV
DeltaDelta G_hyd = Delta G_aqueous - Delta G_vacuum
```

A water-only result may be recorded as the solution-leg mutation free energy but must not be called
the relative hydration free energy.

### Resolve endpoint structures without guessing

Search the current repository and known project inputs/journals for exact, provenance-bearing RGDfV,
reduced_RGDfV, and cilengitide structures, including the known sibling project locations already
documented in this repository. Prefer an existing SDF/MOL/SMILES with explicit stereochemistry and
connectivity over generating a structure from a name.

`reduced_RGDfV` is chemically ambiguous from its name alone. If no unique structure with defensible
provenance is found:

- write `reports/BLOCKED_reduced_RGDfV_structure.md` describing where you searched and what exact
  structure information is missing;
- do not invent or infer the reduction;
- continue all cMD, Deeptime, infrastructure, and RGDfV -> cilengitide work;
- prepare the reduced transformation configuration/schema and validation tests so only the endpoint
  structure/atom map remains to be supplied later.

If the cilengitide structure is generated from the user's definition because no trusted file exists,
independently validate formula, N-methylation site, cyclic connectivity, D-Phe stereochemistry,
Val stereochemistry, net charge, and endpoint force-field assignment before any alchemical pilot.

### Hybrid topology and pilot gate

Do not trust a transformation merely because it runs. Evaluate available OpenMM-compatible
alchemical tooling, but do not hide the physics behind an unvalidated pre-alpha workflow. Record a
short technical decision explaining the selected implementation and exact versions.

Before production, require automated tests proving:

- explicit, reviewed atom mapping including appearing/disappearing atoms;
- independently constructed endpoint systems and energies;
- hybrid lambda=0 reproduces the independently built RGDfV endpoint;
- hybrid lambda=1 reproduces the independently built product endpoint;
- forces and energies are finite at endpoints and all intermediate states;
- constraints, masses, exclusions, 1-4 interactions, bonded terms, charges, Lennard-Jones terms,
  and periodic nonbonded settings transform as intended;
- soft-core treatment removes endpoint singularities for appearing/disappearing atoms;
- total charge is correct throughout;
- exact parameter derivatives are available and checked against finite differences for TI;
- a short dynamics smoke test at every lambda is stable;
- the neighboring-window overlap graph from a short pilot is connected.

If any gate fails, do not launch that transformation's 10 ns/window production. Diagnose and test
what can be fixed safely, record the exact blocker, and continue every other independent task.

### Lambda production contract

For every transformation and leg that passes the pilot:

- exactly 21 planned lambda states, with the complete electrostatic/steric/bonded schedule recorded;
- 10 ns of sampled production per window after equilibration; equilibration does not count toward
  the 10 ns;
- independent recorded seeds and checkpoint/resume per window;
- coordinates at a documented compact cadence, normally solute every 10 ps;
- reduced potentials and exact dU/dlambda frequently enough for reliable MBAR/TI after statistical
  inefficiency analysis, normally every 1 ps unless the pilot justifies another cadence;
- finite energies/derivatives and no NaN/Inf;
- dynamic worker queue across the three initially assigned GPUs, expanding to safely freed cMD GPUs;
- no silent lambda insertion, removal, or schedule change after production begins.

The base workload is 2 transformations x 2 legs x 21 windows = 84 window simulations, each with
10 ns sampled production. This instruction does not authorize three complete alchemical repeats;
run one validated schedule per transformation/leg. Preserve seeds and infrastructure so independent
repeats can be added later.

## Part D: PyMBAR, FastMBAR, and TI analysis

For every completed leg:

- construct and validate reduced-potential `u_kn`, sample counts `N_k`, temperature/beta, and units;
- decorrelate or subsample using a documented statistical-inefficiency procedure;
- use PyMBAR as the primary MBAR estimate;
- solve the same data with FastMBAR as an independent CPU/GPU implementation check;
- integrate exact mean dU/dlambda with a documented quadrature rule for TI;
- report Delta G, uncertainty, overlap matrix, connectedness, effective sample sizes, per-window
  sampling, dU/dlambda curves, cumulative estimates, and first-half/second-half convergence;
- compare PyMBAR, FastMBAR, and TI and diagnose disagreements rather than averaging them;
- combine aqueous and vacuum legs only after both independently pass convergence/overlap checks.

Keep large `u_kn` arrays outside Git; commit compact summaries and scripts that regenerate them.

## Autonomous decision policy while the user is away

Do not issue a blocking question and wait. Use this priority order:

1. preserve scientific correctness and existing user data;
2. continue independent simulations, analysis, tests, documentation, and infrastructure;
3. make reversible, conventional implementation choices and record them;
4. retry only safe, understood transient failures;
5. isolate a chemically or scientifically ambiguous subtask without guessing;
6. leave a precise blocker and smallest next action for the user on return.

Examples:

- missing reduced_RGDfV structure blocks only that transformation, not cilengitide or cMD;
- failed alchemical overlap blocks that production leg, not endpoint development or kinetics;
- one occupied/broken GPU reduces the worker pool, not the experiment's scientific scope;
- one failed replicate is diagnosed/restarted from its last valid commit while the others continue;
- unavailable FastMBAR blocks only its cross-check after reasonable isolated-environment installation
  attempts; PyMBAR/TI and all simulation work continue;
- incomplete 1 microsecond runs remain `RUNNING`, with partial convergence reports clearly labelled.

Never use destructive Git/filesystem operations, overwrite raw data, kill unrelated jobs, expose
credentials, or broaden the chemical definition to make a blocked transformation appear complete.

## Tests, provenance, reports, and handoff

Test analysis code on small synthetic trajectories and known free-energy examples before using
production data. Record exact commands, versions, random seeds, source hashes, input hashes, GPU
UUIDs, timings, failures/retries, and output hashes for compact defining artifacts.

Maintain continuously:

- local prototype `STATUS.md` with active/completed/blocked queue items;
- `docs/journal/2026-08-21_unattended-kinetics-and-alchemy.md` in the local prototype;
- a concise handoff journal in `MD-templates/docs/journal/2026-08-21_unattended-kinetics-and-alchemy.md`
  containing no large data and pointing to the local prototype/data paths and exact commits.

Commit focused code/docs changes to the appropriate local `dev` branch. Push only authorized
`MD-templates` changes to remote `dev`; do not create a new GitHub repository, branch, or PR. Do not
commit simulation outputs.

When returning, begin with exactly one status:

- `COMPLETE — all validated unattended experiments and analyses finished.`
- `RUNNING — valid unattended jobs remain active; no user response is required.`
- `PARTIAL — independent work continued, with the following isolated blockers.`
- `BLOCKED — no safe independent work could continue.`

Then give:

1. completed/running/blocked task matrix;
2. GPU/session/PID/window/segment status and exact monitoring commands;
3. cMD production time and frame counts per replicate;
4. alchemical validation/window/leg status;
5. preliminary analysis only where statistically valid;
6. local and remote commit SHAs;
7. exact next actions requiring the user, if any.
