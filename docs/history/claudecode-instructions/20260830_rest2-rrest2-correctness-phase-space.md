# Correct and harden the owned `openmm-md` REST2/rREST2 runtime

## Scope and starting points

Work in one Claude Code session across:

```text
git@github.com:csy0000/MD-templates.git
git@github.com:csy0000/MD-project.git
```

Before changing anything, fetch and verify these exact remote heads:

```text
MD-templates/feat/openmm-md-rest2-rrest2
5c4edfba40a9724e5737c60ca3da9cee55dbf5b0

MD-project/dev5-openmm-md-rest2-rrest2
the commit containing this instruction; record its exact full SHA before branching
```

Read and follow every applicable `CLAUDE.md`, repository instruction, and current runtime journal. Inspect the implementation rather than relying only on this instruction. If either starting head differs, stop and report the observed SHA instead of silently rebasing the work.

Create new branches from those heads:

```text
MD-templates: fix/openmm-md-rest2-rrest2-correctness
MD-project:   dev6-openmm-md-rest2-rrest2-correctness
```

Do not merge into `dev` or `main`, do not rewrite history, and do not alter the existing source branches. Make logical commits and push each new branch once after validation.

## Objective

Correct and harden the repository-owned REST2 and Boltzmann-reservoir REST2 runtime while preserving the established Amber-like interface:

```text
openmm-md --groupfile ... -ng ... [--exchange-rule ...] [--reservoir ...]
```

There must remain one executor, `openmm-md`. REST2, rREST2, and later reservoir variants must use the same coordinated runtime. A changed transition is selected with `--exchange-rule`; it must not require another public executor. Do not replace this runtime with OpenMMTools, do not revive `openmm-rest2`, and do not implement REST2 as temperature REMD.

This iteration fixes correctness, phase-space reservoir handling, event scheduling, restart behavior, MPI ownership, tests, and the MD-project pin. It does not implement kinetic-reservoir REST2 or a non-Boltzmann acceptance rule.

## Scientific defaults and required examples

Use the current MD-templates defaults as the source of truth:

- explicit default: **ff14SB + TIP3P**;
- explicit alternative: **ff19SB + OPC**, only when explicitly selected;
- implicit default: **ff14SB + GBn2 + mbondi3**, with no SASA term;
- REST2 uses one physical thermostat temperature at every rung;
- the default six-state ladder is `tau = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`;
- `s = (1 - tau)^2` scales solute-solute nonbonded terms and the established solute torsions;
- `sqrt(s) = 1 - tau` scales solute-environment interactions;
- environment-environment terms remain unscaled.

The required explicit ALA acceptance test is ff14SB/TIP3P, not ff19SB/OPC. ff19SB/OPC may be retained as a secondary compatibility smoke test but must not be described as the default or substitute for the required test.

## Preserve the architectural boundaries

Keep the current small modules and their ownership:

- protocol and integer-step scheduling;
- Hamiltonian construction/scaling;
- propagation contexts;
- transition rules;
- source/reservoir preparation;
- NetCDF storage/checkpointing;
- validation/statistics;
- the thin `openmm-md` grouped-mode adapter.

Keep the exchange-rule interface narrow. A rule decides proposals, swaps, and optional reservoir transitions from an explicit context. It must not open files, own MPI, propagate dynamics, or parse the CLI. Keep common cMD/AIS/rREST2 source-window parsing and atom/topology identity logic in one shared implementation where the contracts truly agree. Do not force coordinate-only AIS materialization and phase-space reservoir materialization into one ambiguous format.

## Required corrections

### 1. Remove user-facing `segment_ps`

`segment_ps` is an implementation detail and must no longer be required or emitted in user configuration, generated protocol files, examples, or normal documentation.

Users specify physical intervals such as:

- timestep;
- exchange interval;
- whole-system output interval;
- solute output interval;
- number of exchange attempts;
- rREST2 reservoir refresh interval in exchange attempts.

Convert each physical interval exactly to integer steps. Reject an interval that is not an exact positive number of integration steps; never round it silently. Drive propagation by absolute step and event counters. Advance to the next scheduled event among exchange, whole-system output, solute output, checkpoint, and termination. Simultaneous events at one step must execute once in a documented deterministic order.

An internal greatest-common-divisor quantum or variable propagation span is acceptable, but it must be derived, private, and not presented as a scientific input. Records should store the requested physical intervals, their exact step counts, absolute completed step, and event counters. If a legacy config contains `segment_ps`, fail with an actionable migration message rather than silently changing its meaning.

### 2. Separate exchange, whole-system, and solute output schedules

The current runtime writes coordinate frames only at `whole_output_stride` and does not implement the promised independent frequent solute stream. Implement the actual contract:

- exchange attempts occur only at the exact exchange steps;
- whole-system frames occur only at the exact whole-output steps;
- solute frames occur only at the exact solute-output steps;
- no schedule is inferred from another;
- no duplicate frame is written when events coincide;
- each stored frame has its absolute step and time;
- walker/state mappings remain reconstructable at every relevant observation.

Use a compact, validated NetCDF representation for the analysis and trajectory products. Do not fabricate intermediate coordinates. Document whether coordinate arrays are walker-indexed or state-indexed and provide tested inversion between the two views.

### 3. Make reservoir phase space explicit

The default rREST2 reservoir transition must install the stored phase-space sample:

```text
positions + velocities + periodic box when applicable
```

A DCD is not a phase-space reservoir because it cannot store velocities. Do not claim otherwise and do not infer or reconstruct stored velocities from a coordinate-only trajectory.

Add a dedicated, versioned, machine-readable phase-space reservoir source, preferably NetCDF, produced from the fixed-`tau_max` cMD source. It must record at least:

- positions in explicit units;
- velocities in explicit units;
- periodic box vectors for explicit solvent;
- absolute source steps and times;
- frame-to-source mapping;
- atom ordering and topology identity;
- source run and Hamiltonian identity;
- temperature, tau, ensemble, weighting, and selection-window evidence;
- format version and exact software/template identity.

Reuse the shared cMD/AIS source-window selection logic for time windows, bounded reading, atom identity, reproducible frame selection, and source evidence. Keep the resulting phase-space format distinct from AIS coordinate-only `sources.dcd`.

### 4. Stored reservoir velocity is the default

Define and validate a reservoir velocity policy:

```yaml
velocity_policy: stored
```

`stored` is the default. Under this policy, every selected reservoir frame must have finite velocities with the exact particle count. Missing or malformed velocities are a hard error before propagation.

An explicit opt-in alternative may be supported:

```yaml
velocity_policy: maxwell
```

Only this explicit policy may draw Maxwell velocities. It must use the common physical temperature and a recorded deterministic seed. There must be no silent fallback from `stored` to `maxwell`.

Ordinary neighbouring REST2 exchanges continue to exchange configurations/mappings without velocity rescaling because all states share one physical temperature. A stored rREST2 refresh installs the stored velocity unchanged. Do not remove center-of-mass momentum, rescale kinetic energy, or redraw a stored velocity unless the user explicitly selected a separately documented policy.

Record, per reservoir attempt, the frame index, source step/time, velocity policy, accepted status, refresh order, and enough identity to audit which phase-space sample was installed. Do not store large velocity arrays redundantly in JSON manifests.

### 5. Fix MPI reservoir ownership and synchronization

The current `_apply_reservoir()` path can update the top-state owner and then broadcast rank 0's stale configuration list, discarding the new velocity whenever the top rung is not owned by rank 0. Replace this with an explicit single-authority flow:

1. rank 0 chooses and records the reservoir event using the persisted exchange RNG;
2. the chosen phase-space sample, or an unambiguous serializable reference resolved identically, is broadcast;
3. every rank updates the same walker-indexed in-memory configuration;
4. the rank owning the top state installs that exact position/velocity/box state;
5. all ranks verify a compact digest or exact test representation before continuing.

Do not broadcast stale state. Do not depend on which MPI rank owns the top rung. Reinstalling configurations after the transition must not overwrite the new stored velocity.

### 6. Make continuation storage root-owned

During `--resume` and `--extend`, only rank 0 may open the authoritative analysis NetCDF in append/write mode. The current `_continue()` opens it on every MPI rank before closing non-root handles; fix this.

Rank 0 must validate scientific identity, read the checkpoint, decide rewind/extension behavior, and broadcast the validated continuation payload. Read-only per-rank access is allowed only where demonstrably required and safe. All writable reporters, checkpoints, run-state records, and final manifests remain single-writer/root-owned.

Resume from the last complete checkpoint, rewind committed analysis rows that have no checkpoint behind them, and preserve the exchange parity, reservoir schedule, absolute-step schedule, RNG state, frame counters, mapping, and selected velocity policy. `--extend N` must extend only a completed budget by exactly `N` exchange attempts.

### 7. Verify exact Hamiltonian identity

The probability-one Boltzmann reservoir rule is valid only when the source and top rung use the same configurational distribution. Tau, temperature, topology, atom order, and box are necessary but not sufficient.

Add an authoritative Hamiltonian identity check covering the actual serialized OpenMM System or an equivalently complete canonical identity, including force-field/protonation/constraints/masses, REST2 scaling/exclusions, nonbonded settings, and other force parameters. Reuse an existing authoritative cMD continuity/System fingerprint if it is complete. Otherwise implement one shared fingerprint producer and use it in both the source runtime record and the reservoir validator. Never compare two stored claims without recomputing the current side.

Refuse missing identity, a changed System, a crossed ff14SB/TIP3P versus ff19SB/OPC pair, different solute selection, different REST2 exclusions, or any other top-Hamiltonian mismatch. State clearly that a path or directory name such as `cMD_tau0p5` is not identity evidence.

### 8. Coordinate interruption under MPI

Make SIGINT/SIGTERM handling coordinated at safe event boundaries. One rank receiving a termination request must not leave other ranks blocked in collectives. The run must either commit a complete checkpoint/event boundary or leave the previous checkpoint authoritative. Only rank 0 writes `interrupted` state; no rank may write a completion manifest for a partial run.

Document the chosen behavior and test it with subprocess/MPI integration where the environment permits. If portable graceful handling of a particular MPI launcher is impossible, fail explicitly and document the limitation rather than claiming support.

### 9. Preserve extensibility without implementing kinetic reservoirs

Keep ordinary REST2 as the built-in neighbouring rule and Boltzmann rREST2 as its own rule. Retain a clean boundary for future reservoir or kinetic-reservoir rules. Do not implement kinetic-reservoir REST2 in this iteration, and do not reuse the probability-one Boltzmann rule for a clustered, biased, kinetic, or otherwise non-Boltzmann reservoir.

## Storage and provenance requirements

The authoritative outputs must distinguish these jobs:

- analysis/exchange history;
- whole-system trajectory;
- frequent solute trajectory;
- phase-space reservoir source;
- restart checkpoint;
- run-state status;
- final completion manifest.

Every format must be versioned and validated by reading, not merely by checking file existence. Store exact units, dimensions, indexing conventions, requested intervals, step/time arrays, mapping arrays, and identities. Detect truncation, inconsistent counters, non-finite values, wrong shapes, mismatched atom counts, and incomplete final writes.

Use atomic replacement for small JSON/YAML state records. NetCDF append operations must have a last-complete marker written only after the row or frame is fully committed. Restart must not duplicate exchange rows, whole frames, solute frames, or reservoir events.

Keep generated paths external and explicit through `paths.sh`, `$MD_DATA`, and the Amber-like flags. Generated Python protocol files remain concise and path-independent.

## Required tests

Add focused tests that fail on the current defects and pass only after the corrections.

### Protocol and scheduling

- no user-facing or generated `segment_ps`;
- actionable rejection/migration of legacy `segment_ps`;
- exact conversion of every interval to integration steps;
- independent exchange, whole, solute, and checkpoint event schedules;
- simultaneous events written once in deterministic order;
- nontrivial cadence example such as 2 ps solute, 10 ps exchange, and 100 ps whole output;
- no time/step drift over an extension.

### REST2 physics

- `tau=0` reproduces the unscaled System energy within a stated tolerance;
- solute-solute, solute-environment, environment-environment, and torsion scaling are verified against direct energy evaluations;
- exchange log acceptance matches the four-energy analytical expression;
- ordinary accepted exchanges do not rescale velocities;
- walker/state mappings and their inverses remain correct through exchanges and resume.

### Reservoir phase space

- phase-space NetCDF round trip preserves positions, velocities, boxes, source steps, and times;
- `velocity_policy: stored` is the default;
- missing, non-finite, or wrong-sized stored velocities fail before propagation;
- `velocity_policy: maxwell` works only when explicitly requested and is reproducible from its recorded seed;
- stored velocity is installed exactly and survives the post-refresh reinstall;
- frame selection and reservoir event records are identical across deterministic resume;
- coordinate-only DCD is refused as a `stored` phase-space reservoir;
- Hamiltonian, topology, atom order, tau, temperature, ensemble, weighting, box, and solute-selection mismatches are individually refused.

### MPI and restart

- a refresh works when rank 0 does not own the top state;
- all ranks agree on the refreshed configuration digest and mapping;
- only rank 0 opens writable NetCDF/reporters on new, resume, and extend paths;
- interrupted then resumed output is equivalent to uninterrupted output for mapping, RNG continuation, reservoir selection, positions/velocities at checkpoints, event counts, and final status;
- interruption immediately before and after a reservoir event is covered;
- extension adds the exact requested exchange attempts without duplicate rows or frames;
- truncated/corrupt analysis, phase-space, and checkpoint files are rejected;
- SIGINT/SIGTERM under available MPI does not deadlock or create a false completion manifest.

### Required ALA integration matrix

Run real short tests without claiming convergence:

| system | solvent/force field | method | requirement |
|---|---|---|---|
| ALA | ff14SB + GBn2 + mbondi3 | REST2 | required implicit smoke |
| ALA | ff14SB + GBn2 + mbondi3 | rREST2 | required implicit smoke using a real stored-velocity reservoir |
| ALA | ff14SB + TIP3P, NVT | REST2 | required explicit default smoke |
| ALA | ff14SB + TIP3P, NVT | rREST2 | required explicit default smoke using a real stored-velocity reservoir |
| ALA | ff19SB + OPC, NVT | REST2/rREST2 | optional secondary compatibility smoke; never the default substitute |

For fast CPU coverage, a two-state miniature ladder and very short duration are acceptable in addition to the required generated six-state contract. Where CUDA and six-rank MPI are available, run the six-state ladder `tau=[0.0,0.1,0.2,0.3,0.4,0.5]` and record rank-to-device assignment. Explicitly inspect the resolved force-field record so a mislabeled ff19SB/OPC run cannot satisfy the ff14SB/TIP3P requirement.

The rREST2 smoke source must be a real fixed-`tau=0.5` NVT cMD phase-space source produced by this implementation, not a fabricated array. Small synthetic arrays remain appropriate for unit tests only.

Unavailable CUDA or MPI may be reported as unavailable with exact evidence; do not mark an unrun test as passed. CPU/unit/storage tests must still run. Stop rather than weaken a scientific check merely to obtain a green result.

## MD-project integration

After MD-templates is complete and committed:

- update MD-project to pin the exact new full MD-templates commit SHA;
- update the REST2/rREST2 example and workflow to the corrected `openmm-md --groupfile` interface;
- keep all run paths supplied by sourced `paths.sh` variables under `$MD_DATA`;
- expose exchange, whole-output, and solute-output intervals, but not `segment_ps`;
- make the rREST2 example select the stored-velocity phase-space reservoir explicitly or inherit the documented `stored` default;
- keep generated data, component clones, NetCDF products, checkpoints, and trajectories ignored;
- keep small configs, source inputs, workflow definitions, and provenance records trackable;
- verify the workflow dry-run and every documented command.

Do not reimplement the simulation engine in MD-project.

## Documentation and execution record

Update the MD-templates documentation so it accurately explains:

- one `openmm-md` executor;
- `--groupfile`, `-ng`, `--exchange-rule`, and `--reservoir`;
- REST2 versus temperature REMD velocity behavior;
- stored reservoir phase space as the rREST2 default;
- the explicit `maxwell` opt-in;
- the event-driven interval model with no user-facing `segment_ps`;
- state-versus-walker indexing;
- restart/extension and root-owned MPI storage;
- the exact Boltzmann reservoir assumptions and finite-reservoir limitation;
- that kinetic/non-Boltzmann reservoirs remain future separately derived rules.

Add an MD-project execution journal recording:

- starting and final branch SHAs;
- the inspected defects and their root causes;
- files changed and interfaces preserved;
- exact test commands, counts, durations, platforms, GPU mapping, passes, skips, and failures;
- the four ALA matrix results and the resolved force fields;
- unavailable environmental capabilities;
- remaining limitations and deliberate non-goals.

## Completion procedure

Before pushing:

1. Run formatting, linting, and compile checks.
2. Run the complete feasible MD-templates test suite plus every new focused test.
3. Run the four required ALA smoke paths and available CUDA/MPI tests.
4. Exercise new-run, interruption, resume, verification, and extension paths.
5. Inspect all generated configs and commands for path independence and absence of user-facing `segment_ps`.
6. Verify ff14SB/TIP3P is the explicit default and the required explicit test actually used it.
7. Validate all NetCDF, YAML, JSON, checkpoint, and completion records by reading them.
8. Update and validate the MD-project component pin and workflow dry-run.
9. Inspect both working trees for accidental generated data or unrelated changes.
10. Make logical commits, then push both new branches once.

Do not merge into `dev` or `main`. Stop and report rather than inventing behavior if an existing source contract, OpenMM API, NetCDF backend, MPI environment, or available hardware cannot support a requested feature safely.
