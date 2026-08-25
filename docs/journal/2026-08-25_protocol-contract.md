# The protocol contract

Date: 2026-08-25
Branch: `dev`

What the public YAML means. The generated scripts implement this table; making the YAML and the
README agree with each other is not the same thing and was how the previous errors survived.

| Item | Required meaning |
|---|---|
| REST2 `duration_per_segment_ps` | propagation time **between consecutive exchange rounds** |
| REST2 `number_of_exchanges` | number of exchange rounds in one invocation |
| REST2 total production time | `duration_per_segment_ps * number_of_exchanges` |
| NVT | no active barostat |
| NPT | exactly one active Monte Carlo barostat |
| minimization/equilibration restraint | actually applied to the solute atoms in `solute.yaml` |
| production restraint | zero / off |
| whole-system interval | writes a whole-system trajectory |
| solute interval | writes an atom-subset solute trajectory |
| REST2 temperature | ONE physical temperature and ONE physical beta for every replica |
| REST2 ladder | different scaled Hamiltonians, not different thermostat temperatures |
| GPU count | `min(number_of_replicas, number_of_visible_GPUs)` |
| GPU execution | different GPUs propagate concurrently; replicas sharing one GPU propagate sequentially |

## What was wrong

**Duration.** The runner divided one segment among the exchanges, so
`duration_per_segment_ps: 10` with `number_of_exchanges: 1000` propagated 10 ps in total rather
than 10 ns — a factor of a thousand. `duration_per_segment_ps` is the time *between* exchanges.

**Equilibration.** The `equilibration` block was read for cMD and ignored for REST2, and neither
applied the configured positional restraint. A stage the YAML configures and the script skips is
worse than one that is absent.

**Beta.** The previous note claimed replicas differ in beta and that the `pV` term therefore does
not cancel. That is wrong. Every replica is thermostatted at the same temperature and differs only
in its *Hamiltonian*, so beta and pressure are common and the `pV` contributions cancel
algebraically when complete configurations are swapped:

```
log(alpha) = beta * [ U_i(x_i,V_i) + U_j(x_j,V_j) - U_i(x_j,V_j) - U_j(x_i,V_i) ]
```

The box-vector fix that accompanied the wrong explanation is still right and is kept: positions and
periodic box vectors are one configuration, so the box travels with the positions during
cross-energy evaluation and on an accepted swap. The correction is to the reasoning, not to that
behaviour.

## Restraint convention

```
U = 1/2 k |r - r0|^2
1 kcal mol^-1 A^-2 = 418.4 kJ mol^-1 nm^-2
```

Reference coordinates come from `initial_state.xml`. The `CustomExternalForce` stays in the System
for the whole run so the checkpoint layout does not change; its strength is a global Context
parameter set to zero before production.


## Execution platform

Added to the contract: **a generated run uses CUDA unless another platform is named.**

| Item | Required meaning |
|---|---|
| default platform | CUDA |
| CPU/Reference | only when `MD_PLATFORM` asks for it by name |
| missing CUDA | an error naming the override, never a silent fallback |

A run that lands on the CPU because the GPU was not visible to the process still minimises, still
writes a trajectory and still prints "complete" — two orders of magnitude later. That failure is
invisible in every artifact it produces, so it has to be refused rather than absorbed.
`resolve_platform()` in `md_stages.py` is the single place this is decided, and both generated
scripts go through it.

The test suite pins `MD_PLATFORM=CPU`. Its runs are ~30 steps of ALA and exist to check stage
order, barostat state, restraint values and restart bookkeeping; running them on a GPU would buy
nothing and would make the suite unrunnable on a GPU-less CI runner.

## What the corrected code does

`md_stages.py` is new and is copied into every generated project beside `run.py`. It holds the
pieces both methods need and neither should own a private copy of: the restraint, the barostat
activation, the seed derivation, the equilibration sequence, the production hand-off, the platform
decision and the GPU grouping.

- **Duration.** `duration_per_segment_ps` is now the time between exchange rounds. The runner
  propagates `segment_steps` and then attempts exchanges, `number_of_exchanges` times.
- **Equilibration.** Both methods, and every REST2 replica, run restrained minimisation ->
  restrained NVT (barostat frequency 0) -> restrained NPT (frequency 25) -> unrestrained
  production. Implicit systems skip NPT and never carry a barostat. What actually ran is written
  to `equilibration.yaml` / `replica_NN_equilibration.yaml` rather than only logged.
- **Two restraint constants.** `minimization.restraint_k_kcal_mol_a2` holds the solute while the
  initial clashes are relieved; `equilibration.restraint_k_kcal_mol_a2` holds it while the solvent
  relaxes. Both are in the public YAML, so both are applied; previously the second was ignored.
- **Restart.** A barostat's frequency lives in the System, not in the checkpoint, so
  `resume_production()` sets the production layout *before* loading. A resumed NPT run that only
  loaded its checkpoint would have continued with the inactive barostat equilibration left behind —
  NPT that is silently NVT. Setting the layout first also avoids a `reinitialize` afterwards, which
  would restart the integrator's random stream that the checkpoint exists to restore.
- **Seeds.** `derive_seed(base, *purpose)` gives every replica its own integrator, velocity and
  barostat seed, recorded in the startup log and in the equilibration record.
- **Two-replica ladder.** Phase 1 offers no pair when there are two rungs. The runner now falls
  back to the other phase, so every round exchanges and a run resumed on an odd attempt index
  cannot sit in the empty phase.
- **Trajectories.** `whole_system.dcd` / `solute.dcd` for cMD and `replica_NN_whole.dcd` /
  `replica_NN_solute.dcd` per replica, at their own configured intervals, appending on restart.
  `sys-gen` now also writes `solute.pdb` from the same indices as `solute.yaml`, because a subset
  DCD cannot be read against the whole-system topology.
- **GPU concurrency.** Replicas are grouped round-robin over `min(n_replicas, n_visible)` devices;
  groups propagate in a `ThreadPoolExecutor` and every group is awaited before exchanges are
  attempted. OpenMM releases the GIL while stepping, so the threads genuinely overlap.

## Validation

Explicit ALA/OPC, 5,722 particles, on this machine's GPUs:

- cMD: 50,000 steps (0.1 ns) at ~700 ns/day, NVT with 0 active barostats and NPT/production with
  exactly 1, restraint 418.4 then 836.8 then 0.0 kJ/mol/nm^2.
- REST2, 4 replicas x 25 rounds x 1,000 steps, three invocations: 75 contiguous exchange rounds,
  75,000 steps per replica, step counter monotone, history appended never rewritten; 75 solute
  frames and 15 whole-system frames per replica across the three runs.
- Concurrency, identical resumed work: 29.9 s on one GPU, 13.1 s on four.

Implicit GBn2, both methods, on GPU: no `MonteCarloBarostat` in `system.xml`, zero active
barostats at every stage, `usesPeriodicBoundaryConditions()` False, both ensembles resolved to NVT
and `npt_duration_ps` to 0. cMD 25,000 steps at ~2,680 ns/day; REST2 4 replicas x 2 invocations =
50 contiguous rounds, 50,000 steps per replica, trajectories appended. In implicit solvent the
solute is the whole system, so the two DCDs hold the same atoms at different intervals — redundant
rather than wrong.

Installed wheel, from outside the checkout: all six template files packaged, both generators
importable from the installed location, the six public commands and a full
`sys-config -> sys-gen -> md-gen -> cMD/REST2` pipeline run against the wheel. The only absolute
path in a generated file is the recorded interpreter in `run.sh`, which has a `python3` fallback
and is not a checkout path.

`pytest tests/` — 64 passed in 50 s.

## Not yet done

Sections 8 (installer dependency set), 9 (release-only CI), 10 (stale surfaces) and the wheel
completeness check of this task remain open.

## The installer (task section 8)

`md-template install` created an environment holding `openmm` and `python` and reported success.
That environment cannot run `sys-gen`: building a system parameterises through OpenFF and
AmberTools, so the failure surfaced part-way through a build rather than at install time.

The solve now carries what `src/` actually imports — derived from the imports, not from a wish
list: `pyyaml`, `numpy`, `openff-toolkit`, `openff-nagl-models`, `openmmforcefields`, `ambertools`,
`parmed`, `rdkit`. AmberTools is there for two distinct reasons that are easy to conflate:
`sqm`/`antechamber` for standard AM1-BCC, and `tleap` for implicit peptide topologies.

Validation replaced the old platform-only probe. It runs inside the target environment, imports
every required package, records versions, locates the AmberTools executables, checks that OpenFF
has a registered AmberTools toolkit — without which `am1bcc` does not resolve at all — and takes
one integration step on Reference, on CPU and on CUDA when present. An environment that cannot run
the advertised workflows fails the command and names what is missing.

Two distinctions the check has to make, or it is worse than none:

- **CUDA absent is not a failure.** A CPU-only machine is supported. A CUDA platform that is
  *listed but cannot take a step* IS a failure, because that one fails at run time instead.
- **Plugins for absent hardware are a note, not a failure.** conda-forge ships HIP plugins; on an
  NVIDIA machine they cannot load, and refusing over that would fail every correct installation
  here. A CUDA plugin failing to load stays fatal.

`--validate PREFIX` runs the same check against an environment this command did not create, and
records it identically. An environment built by hand is not less obliged to work.

### Validation

`md-template install --dry-run` emits the full solve. Against the existing OpenMM 8.6.0
environment, `md-template install --validate` exits 0 with: openmm 8.6.0 / python 3.12.13,
openff.toolkit 0.19.0, openff.nagl_models 2025.9.0, openmmforcefields 0.16.0, parmed 4.3.1, rdkit
2026.03.1, numpy 2.5.2, pyyaml 6.0.3; sqm, antechamber and tleap all present; `am1bcc_ready: true`
with `AmberToolsToolkitWrapper` registered; platforms Reference/CPU/CUDA/OpenCL with all three
one-step checks `ok`; four HIP plugin notes; `problems: []`. All of it in `machine.yaml`.

`pytest tests/` — 73 passed in 62 s.

## Still open

Sections 9 (release-only CI) and 10 (stale surfaces) of the task remain.
