# Focused scientific-runtime correction before the simplified OpenMM release

## Purpose

The simplified six-command CLI is the correct architecture. Do not redesign it again.

This task fixes scientific and execution mistakes introduced when the old runner was replaced by
standalone generated scripts. Preserve the current public commands:

```bash
md-template init
md-template install
md-openmm sys-config
md-openmm show-default
md-openmm sys-gen
md-openmm md-gen
```

Work only on `dev`. Do not merge to `main`, move an existing tag, create a release, introduce a
new engine abstraction, or restore the deleted framework.

## 1. Establish the protocol contract before editing code

Add a short table to the current journal recording these invariants:

| Item | Required meaning |
|---|---|
| REST2 `duration_per_segment_ps` | propagation time **between consecutive exchange rounds** |
| REST2 `number_of_exchanges` | number of exchange rounds in one invocation |
| REST2 total production time | `duration_per_segment_ps * number_of_exchanges` |
| NVT | no active barostat |
| NPT | one active Monte Carlo barostat |
| minimization/equilibration restraint | actually applied to the solute atoms |
| production restraint | zero/off |
| whole-system interval | writes a whole-system trajectory |
| solute interval | writes an atom-subset solute trajectory |
| REST2 temperature | one physical temperature and one physical beta for every replica |
| REST2 ladder | different scaled Hamiltonians, not different thermostat temperatures |
| GPU count | `min(number_of_replicas, number_of_visible_GPUs)` |
| GPU execution | different GPUs propagate concurrently; replicas sharing one GPU propagate sequentially |

Do not proceed by merely making the YAML and README agree. The generated scripts must implement the
contract.

## 2. Correct REST2 duration and exchange bookkeeping

In the generated REST2 runner, replace the current behavior that divides one segment by the number
of exchanges.

The loop must be equivalent to:

```python
segment_steps = steps_for(duration_per_segment_ps)

for exchange_round in range(number_of_exchanges):
    propagate_every_replica(segment_steps)
    attempt_neighbor_exchanges()
```

For the default:

```yaml
duration_per_segment_ps: 10
number_of_exchanges: 1000
```

each replica therefore advances 10 ns per invocation.

Validate that one segment is an integer number of integration steps. Do not require the segment
step count to be divisible by the number of exchanges.

Handle the two-replica ladder correctly: every exchange round should offer pair `(0, 1)`. Do not
allow the alternating schedule to enter a permanent no-pair restart loop.

The exchange history must append on restart, and exchange-round indices and simulation step counts
must continue monotonically.

## 3. Implement the configured minimization and equilibration

Both cMD and every REST2 replica must execute, on a fresh run:

```text
restrained minimization
-> restrained NVT with no active barostat
-> restrained NPT with one active barostat (explicit solvent only)
-> unrestrained production
```

For implicit GBn2:

```text
restrained minimization
-> restrained NVT
-> unrestrained NVT production
```

There must never be a barostat in an implicit-solvent system.

### Positional restraint

Apply the configured `restraint_k_kcal_mol_a2` to the solute atoms listed by `solute.yaml`.

Use a transparent OpenMM `CustomExternalForce` with reference coordinates from
`initial_state.xml`. Document the convention explicitly:

```text
U = 1/2 k |r-r0|^2
1 kcal mol^-1 A^-2 = 418.4 kJ mol^-1 nm^-2
```

Give the force one global strength parameter. Keep the Force in the System so checkpoint structure
remains stable, but set its Context parameter to zero before production.

### NVT to NPT transition

Do not run the NVT stage with an active barostat.

For explicit solvent, one simple valid implementation is:

1. construct the production System with the restraint and a `MonteCarloBarostat`;
2. use barostat frequency 0 for minimization and NVT;
3. enable the production frequency before NPT;
4. reinitialize the Context with `preserveState=True` if required for the Context to see the
   changed Force property;
5. confirm exactly one barostat is present;
6. run restrained NPT;
7. set the restraint strength to zero;
8. reset the production step counter.

A resumed production run must construct the same Force layout before loading its checkpoint.

REST2 must perform this equilibration separately at every tau value before exchange production.
Do not silently skip the `equilibration` block.

For implicit configuration generation, write `npt_duration_ps: 0` and keep the method ensemble NVT.

## 4. Correct REST2 thermodynamics documentation and code

All REST2 replicas use the same thermostat temperature and the same physical beta. They differ by
Hamiltonian scaling.

For an NPT Hamiltonian exchange at common temperature and pressure, the pV contributions cancel
algebraically when the complete configurations are swapped. Do not claim that replicas have
different beta.

Keep the correct part of the recent box fix: positions and periodic box vectors form one
configuration, so the box vectors must travel with the positions during cross-energy evaluation and
an accepted swap.

Use the exchange criterion:

```text
log(alpha) = beta * [U_i(x_i,V_i) + U_j(x_j,V_j)
                     - U_i(x_j,V_j) - U_j(x_i,V_i)]
```

Continue to use the AMBER-compatible tau convention:

```text
solute-solute scaling      = (1-tau)^2
solute-environment scaling = (1-tau)
```

Preserve omega exclusion.

Add one small unit test showing pV cancellation at common beta and pressure, and retain the existing
numerical scaling-equivalence test.

## 5. Correct random seeds

Every REST2 replica must have a distinct deterministic seed for:

- its Langevin integrator;
- initial velocity assignment;
- its barostat.

The current constant integrator seed must be removed. Derive seeds from the configured base seed and
replica index, keep them within OpenMM's accepted integer range, and record them in the startup log.

A restart must load the checkpoint RNG state instead of regenerating velocities.

## 6. Restore the required trajectories

Use OpenMM's normal reporters; do not create a new trajectory framework.

### cMD

Write:

```text
whole_system.dcd    at whole_system_interval_ps (default 100 ps)
solute.dcd          at solute_interval_ps       (default 10 ps)
production.csv      at solute_interval_ps
production.chk      at checkpoint_interval_ps
```

Use `DCDReporter(..., atomSubset=solute_indices)` for `solute.dcd`.

### REST2

For each replica, write:

```text
replica_00_whole.dcd
replica_00_solute.dcd
replica_00.csv
replica_00.chk
```

and corresponding files for every replica.

Add the three output intervals to the REST2 configuration if they are not already available:

```yaml
checkpoint_interval_ps: 100
whole_system_interval_ps: 100
solute_interval_ps: 10
```

Reporters must append during a valid restart.

Because a solute-only DCD needs a matching topology, make `sys-gen` also write `solute.pdb`.
Construct it from the same atom indices recorded in `solute.yaml`.

Do not mistake a StateDataReporter interval for a solute trajectory interval.

## 7. Make multi-GPU REST2 actually concurrent

Keep GPU selection simple:

- respect `CUDA_VISIBLE_DEVICES`;
- use logical OpenMM device indices;
- use at most `min(N_replicas, N_visible_GPUs)`;
- assign replicas round-robin;
- no busy-GPU detection;
- no hardware database or scheduler.

During each propagation segment:

- group replicas by assigned GPU;
- propagate replicas sharing one GPU sequentially;
- propagate different GPU groups concurrently with a small `ThreadPoolExecutor`;
- wait for every group before attempting exchanges.

For CPU or Reference execution, use a simple sequential path.

Add a unit test with mocked stepping functions proving that two GPU groups overlap while two
replicas assigned to the same GPU do not overlap. Do not require CUDA in CI.

## 8. Make the OpenMM installation useful for every advertised command

`md-template install -e openmm -ev 8.6.0` must install the dependencies required by the currently
advertised OpenMM workflows, not only the `openmm` package.

Derive the final list from actual imports, but it should include the needed equivalents of:

- OpenMM 8.6.0
- Python
- PyYAML
- NumPy
- OpenFF Toolkit
- openmmforcefields
- AmberTools
- ParmEd
- RDKit

After installation, validate imports and perform:

- OpenMM platform discovery;
- a one-step Reference/CPU Context test;
- a one-step CUDA Context test when CUDA is present;
- an OpenFF/AmberTools availability check sufficient for standard AM1-BCC;
- a ParmEd import check.

Record the package versions and validation results in `machine.yaml`.

Do not add Amber or GROMACS engines and do not install drivers or a package manager.

The README must make it clear how to activate the created environment and install/use the
`md-templates` CLI inside it. Prefer one explicit command sequence over automatic bootstrap
machinery.

## 9. Replace stale CI instead of fixing the deleted framework

The current GitHub Actions workflows are red because they still import and inspect deleted modules,
profiles, catalogues, bundles and schemas. They also run on every push, contrary to the requested
release-only testing policy.

Delete the obsolete CI scripts and replace the workflows with one small release validation
workflow triggered only by:

```yaml
workflow_dispatch:
push:
  tags:
    - "openmm-v*"
```

It should:

1. create the documented CPU preparation environment;
2. build and install the wheel;
3. invoke the six public commands from outside the checkout;
4. run the current `tests/` suite;
5. run the tiny cMD and REST2 smoke tests on Reference or CPU;
6. fail on skipped scientific smoke tests.

Do not restore tests for the deleted registry, bundle, schema or runtime-export architecture.

## 10. Remove stale active surfaces

Keep Git history and `docs/journal/`, but clean the current working tree.

Remove or update content that tells users to use the retired JSON/framework interface:

- the old root `test/` directory;
- obsolete CI scripts;
- obsolete generated-bundle/configuration documentation;
- outdated support matrices;
- historical Claude instructions that are no longer executable;
- the old installation skill if it still describes Amber/OpenMM 8.5.2 or duplicates the new CLI.

Keep scientifically useful validation reports. If a historical document is necessary to interpret
a retained report, label it clearly as historical and do not link it as current usage.

The root README and `installation/README.md` must describe only the current OpenMM 8.6.0 CLI.

Do not count deletion as evidence that the simulation works.

## 11. Small scientific acceptance suite

Keep the suite small. Add or adjust only tests that protect the public scientific contract:

1. REST2 total steps equal segment steps times exchange count.
2. Two-replica exchanges continue across restarts.
3. Fresh cMD follows minimization -> NVT -> NPT -> production.
4. NVT has no active barostat; explicit NPT/production have exactly one.
5. Implicit systems never contain a barostat.
6. Configured solute restraints are nonzero during minimization/equilibration and zero in production.
7. REST2 equilibrates every replica.
8. Replica integrator, velocity and barostat seeds are distinct.
9. Whole-system and solute trajectories are created at their configured intervals.
10. A solute DCD is readable with `solute.pdb`.
11. Restart appends trajectories and logs.
12. Multi-GPU grouping is concurrent across devices and sequential within a device.
13. Generated scripts contain no absolute checkout path and do not import `md_templates`.
14. REST2 scaling remains numerically identical to the validated scaling implementation.
15. Common-beta/common-pressure pV cancellation is correct.

Use very short ALA simulations on the Reference platform. Do not add hundreds of field-by-field
tests and do not run production-length simulations.

No required scientific test may pass by skipping because a dependency is missing from the release
environment.

## 12. Completion procedure

1. Make the focused corrections on `dev`.
2. Do not change the six-command architecture.
3. Run fast tests during development.
4. Run the complete small suite once the fixes are complete.
5. Manually exercise the explicit ALA path through `sys-config -> sys-gen -> md-gen -> cMD/REST2`
   with shortened durations.
6. Manually exercise implicit GBn2 construction and short cMD/REST2 runs.
7. Confirm the stale push-triggered workflows are gone.
8. Commit and push to `dev`.
9. Do not merge to `main` and do not tag a release yet.

The final report must provide:

- the `dev` commit SHA;
- exact tests and results;
- total simulated steps demonstrating the corrected REST2 duration semantics;
- evidence of NVT/NPT barostat state;
- evidence restraints were active then disabled;
- trajectory filenames and frame counts;
- distinct REST2 seeds;
- CPU/Reference smoke results;
- installer dry-run command plus validation against the existing OpenMM 8.6.0 environment;
- remaining limitations.

Do not return PASS if the public YAML contains settings that the generated scripts ignore, if
either scientific smoke test skips, or if a workflow still tests the deleted architecture.
