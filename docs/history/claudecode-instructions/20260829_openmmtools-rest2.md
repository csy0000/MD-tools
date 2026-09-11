# OpenMMTools-based Amber-like REST2 implementation

Work across:

```text
git@github.com:csy0000/MD-templates.git
git@github.com:csy0000/MD-project.git
```

Expected starting points:

```text
MD-templates
  branch: feat/amber-like-openmm
  commit: 6603c4c198574884b4aa7a61af7340a38e56e20c

MD-project
  branch: dev3-amber-like
  commit: be75ad4fa9645c2a686260252da37f67be1a9c30
```

Create:

```text
MD-templates: feat/openmmtools-rest2
MD-project:   dev4-openmmtools-rest2
```

Create each branch from the exact starting point above unless the remote branch has advanced. If it has advanced, inspect the additional commits and report any conflict before proceeding.

Do not merge into `dev` or `main`. Do not force-push or rewrite history.

## Objective

Implement a simple, Amber-like REST2 interface backed by:

```python
openmmtools.multistate.ReplicaExchangeSampler
```

Use OpenMMTools for:

* replica propagation orchestration;
* reduced-potential evaluation;
* exchange decisions;
* thermodynamic-state assignment;
* multistate NetCDF storage;
* checkpointing;
* restart and extension;
* walker-to-state mappings.

Retain the existing, tested `MD-templates` implementation for constructing REST2-scaled Hamiltonians. OpenMMTools does not itself define this repository’s REST2 scaling convention.

Do not maintain a second production exchange algorithm after parity is established. The current custom exchange loop may temporarily remain as a reference implementation during validation, but the final generated REST2 workflow must use `ReplicaExchangeSampler`.

## Scientific REST2 definition

Use the existing repository convention:

```text
tau = 0: unscaled cold state

s = (1 - tau)^2

solute-solute interactions:       s
solute-environment interactions:  sqrt(s) = 1 - tau
environment-environment:          1
solute torsions:                   s
```

Every replica uses the same physical temperature and, for explicit solvent, the same pressure.

Do not implement REST2 as ordinary temperature REMD. Do not assign a different thermostat temperature to each replica.

Preserve and validate the existing treatment of:

* `NonbondedForce` particles and exceptions;
* solute torsions;
* CMAP;
* explicit solvent;
* whole-system implicit GBn2 scaling;
* enhanced-region atom selection;
* omega-bond exclusions;
* unknown force refusal.

Document clearly that leaving peptide omega torsions unscaled is this repository’s omega-selective REST2 convention. Do not silently describe it as an entirely unmodified standard REST2 Hamiltonian.

## OpenMMTools dependency

Target and test:

```text
openmmtools 0.26.0
```

Do not let pip install a second OpenMM stack.

Add OpenMMTools to the conda/environment specification used for the scientific stack, and make `validate-env` report:

* OpenMM version;
* OpenMMTools version;
* PyMBAR version;
* NetCDF4 version;
* MPI/mpiplus availability;
* selected OpenMM platform;
* CUDA device mapping where applicable.

Record the actual installed versions in run provenance.

Do not add the heavy OpenMM scientific stack to the minimal pure-Python `pyproject.toml` dependencies unless this repository’s existing packaging policy explicitly requires it.

## User-facing design

### Simple generated Python input

The generated REST2 Python file must remain short and scientifically readable. Physical paths must not be embedded in it.

Use a shape comparable to:

```python
from md_openmm_runtime import REST2

def run(files):
    simulation = REST2(
        files,
        tau_min=0.0,
        tau_max=0.5,
        number_of_replicas=6,
        exchange_interval_ps=10.0,
        equilibration_duration_ps=10.0,
        temperature_k=300.0,
        pressure_bar=1.0,
        timestep_fs=4.0,
        whole_output_interval_ps=10.0,
        solute_output_interval_ps=2.0,
    )
    simulation.run(number_of_exchanges=1000)
```

The exact import path may differ after inspecting the current generated-runtime architecture, but the following requirements are mandatory:

* the user-facing file stays concise;
* it contains the scientific protocol;
* it contains no project-specific absolute paths;
* implementation details live in a copied generated runtime module;
* an instantiated project remains runnable without a live `MD-templates` clone;
* the generated runtime records its source commit and content checksums.

Do not put the full exchange implementation into every user-facing `rest2.py`.

### Amber-like launcher

Add an Amber/groupfile-like command, preferably:

```text
openmm-rest2
```

The generated launcher should source `paths.sh` and resemble:

```bash
openmm-rest2 \
    -i "${REST2_DIR}/rest2.py" \
    -p "${INPUT_DIR}/ALA.pdb" \
    -s "${INPUT_DIR}/ALA.xml" \
    -c "${EQ_DIR}/npt_free/npt_free.rst.xml" \
    --solute "${INPUT_DIR}/solute.yaml" \
    -o "${REST2_DIR}/rest2.out" \
    -x "${REST2_DIR}/rest2.nc" \
    -r "${REST2_DIR}/restart.json" \
    --checkpoint "${REST2_DIR}/rest2_checkpoint.nc"
```

Support long option names as well.

The intended meanings are:

| Option         | Meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| `-i`           | concise REST2 Python protocol                                    |
| `-p`           | topology                                                         |
| `-s`           | serialized base OpenMM `System`                                  |
| `-c`           | common equilibrated coordinates/state                            |
| `--solute`     | enhanced-region atom definition                                  |
| `-o`           | human-readable REST2 log                                         |
| `-x`           | authoritative OpenMMTools multistate NetCDF                      |
| `-r`           | portable restart/final-state manifest                            |
| `--checkpoint` | full-coordinate checkpoint NetCDF                                |
| `--resume`     | resume the same run in place                                     |
| `--extend N`   | add `N` exchange attempts                                        |
| `--force`      | replace a new, non-resumed output only when explicitly requested |

Do not pretend that a single XML restart can hold all replicas. `-r` should be a small manifest referencing the authoritative NetCDF/checkpoint storage and any exported per-replica or per-state final files.

On resume, the NetCDF storage is authoritative.

## REST2 initialization

REST2 replicas share:

* one topology;
* one base system;
* one enhanced-region definition;
* one common starting state.

Do not require users to provide six independent topology files.

For each rung:

1. Construct the scaled system using the existing REST2 scaling implementation.
2. Create the corresponding OpenMMTools thermodynamic state at the common physical temperature and pressure.
3. Start from the common equilibrated state.
4. Perform tau-specific equilibration before production exchange.
5. Preserve positions, velocities, and box vectors.
6. Record the rung’s `tau`, `s`, seeds, and input identity.

Tau equilibration must not count toward exchange-production time.

Support explicit initial per-rung states as an advanced resume/import option, but do not make them the normal user interface.

## ReplicaExchangeSampler configuration

Use:

```python
ReplicaExchangeSampler(
    mcmc_moves=...,
    number_of_iterations=...,
    replica_mixing_scheme="swap-neighbors",
)
```

Use `swap-neighbors` for the ordered REST2 ladder.

Create one thermodynamic state per tau value and one sampler state per replica. Ensure that positions and box vectors travel together.

Map physical time exactly:

```text
steps per exchange = exchange_interval_ps / timestep_ps
production per replica = number_of_exchanges × exchange_interval_ps
```

Reject non-integral step conversions.

Do not silently replace Langevin-middle behavior with a different integrator. Inspect OpenMMTools 0.26.0’s available move implementations and select or implement the smallest tested move that reproduces the intended Langevin-middle/BAOAB convention.

Record the exact move class, splitting, collision rate, timestep, constraint tolerance, and number of steps.

## Trajectory and reporting model

Use OpenMMTools `MultiStateReporter` as the authoritative storage.

Store:

* solute coordinates at the requested solute interval;
* full coordinates at the requested whole-system interval;
* reduced-potential matrix;
* thermodynamic-state index for every walker;
* proposed and accepted exchanges;
* iteration and time;
* box vectors;
* checkpoint data;
* metadata and version information.

Use `analysis_particle_indices` for the solute subset where appropriate.

OpenMMTools distinguishes walkers from thermodynamic states. Do not duplicate all coordinates into separate “walker” and “state” trajectories during simulation.

Provide post-processing/export commands that can reconstruct:

```text
walker view: follow one continuous coordinate walker
state view:  collect whichever walker occupied a given tau state
```

Both views must be derived from:

```text
sampler states + replica_thermodynamic_states mapping
```

### Solute output more frequently than exchange

The required example is:

```text
timestep:              4 fs
solute output:         2 ps  = 500 steps
exchange attempt:     10 ps  = 2500 steps
whole-system output:  10 ps  = 2500 steps
```

Do not repeat an exchange-boundary configuration five times and call it a 2 ps trajectory.

First inspect whether OpenMMTools provides a public propagation/reporting hook that can record intermediate solute coordinates inside a 10 ps move.

If a public hook exists, use it.

If it does not, implement the narrowest version-pinned extension needed to obtain real intermediate coordinates while leaving exchange decisions, energy evaluation, state mappings, NetCDF restart, and storage under OpenMMTools control.

Possible approaches must be evaluated rather than assumed:

* a reporting MCMC move that propagates in smaller substeps;
* a strided `ReplicaExchangeSampler` iteration model;
* a narrow sampler subclass that attempts exchange only every configured stride.

Any subclass of an OpenMMTools private method must:

* be isolated in one module;
* explicitly pin OpenMMTools 0.26.0;
* have contract tests against that version;
* record the extension in provenance;
* fail on an untested OpenMMTools version;
* preserve restart determinism and exchange scheduling.

Benchmark the overhead. Do not accept a design that evaluates the full cross-state energy matrix every 2 ps merely to save solute coordinates unless the overhead is measured and explicitly accepted.

## NetCDF and restart contract

Use separate analysis and checkpoint storage if supported by `MultiStateReporter`:

```text
rest2.nc
rest2_checkpoint.nc
```

A completed run must be resumable with:

```python
ReplicaExchangeSampler.from_storage(...)
```

and extendable without creating a sibling run.

Resume must verify:

* topology checksum;
* system checksum;
* enhanced-region checksum;
* tau ladder;
* temperature and pressure;
* timestep and collision rate;
* exchange interval;
* reporting intervals;
* OpenMMTools compatibility;
* replica count;
* seed policy;
* project and template identities.

Refuse a changed scientific configuration before appending.

Do not use existence of a final file alone as proof of completion. Write an atomic completion manifest only after OpenMMTools has committed the final iteration and the expected exchange budget is present.

Use the exact line:

```text
run_status: completed
```

only after successful finalization.

## Parallel GPU execution

Use OpenMMTools/mpiplus/MPI facilities where supported rather than maintaining the existing Python-thread replica scheduler as the production implementation.

Provide an Amber-groupfile-like multi-GPU launcher, for example:

```bash
mpiexec -n 6 openmm-rest2 ...
```

Test and document rank-to-device binding. Every rank must report:

* MPI rank;
* hostname;
* visible CUDA devices;
* selected OpenMM `DeviceIndex`;
* precision mode;
* assigned replica work.

Do not allow all ranks to silently select GPU 0.

Use one rank per replica when enough GPUs are available. When fewer GPUs than replicas are available, require a documented and tested allocation policy. Do not oversubscribe GPUs silently.

CPU and one-GPU execution must remain available for smoke tests.

## Provenance

Record at minimum:

* project repository and commit;
* `MD-templates` commit;
* generated-runtime checksum;
* OpenMM, OpenMMTools, PyMBAR, NetCDF4, Python and MPI versions;
* topology, system, coordinates and solute-definition checksums;
* temperature and pressure;
* tau values and scale factors;
* enhanced-region atom indices;
* omega exclusions;
* timestep and collision rate;
* HMR state;
* exchange and reporting intervals;
* number of attempts requested and completed;
* replica-to-state mappings;
* acceptance attempts and counts by neighboring pair;
* seeds;
* platform, precision, hosts, ranks and devices;
* NetCDF and checkpoint identities;
* execution status and timestamps.

Do not duplicate arrays already stored authoritatively by OpenMMTools. Reference the NetCDF variables and add only project-level provenance.

Provide a readable summary with neighboring-pair acceptance and state round trips, but do not make that summary the authoritative data.

## ALA acceptance example

Add a generated ALA explicit-solvent REST2 example with:

```text
system:                  ACE–ALA–NME
force field:             ff19SB
water:                   OPC
ensemble:                NPT
temperature:             300 K
pressure:                1 bar
timestep:                4 fs with verified HMR
tau:                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
replicas:                6
tau equilibration:       10 ps per replica
exchange interval:       10 ps
solute output interval:   2 ps
whole output interval:   10 ps
```

For automated smoke testing, shorten the durations while preserving at least:

* three replicas;
* two exchange attempts;
* a nontrivial solute subset;
* NetCDF creation;
* at least one accepted or correctly rejected exchange;
* state mapping;
* restart and extension.

Do not claim that a smoke test validates ladder quality.

Add an implicit-solvent ALA dry-generation or short smoke test using the repository’s established GBn2 restrictions. Do not treat partial-region GBn2 REST2 as supported.

## Phenol/IPH acceptance example

Ensure the generator can produce a phenol/IPH explicit-solvent REST2 project using the existing ligand parameterization route.

Requirements:

* committed source input;
* explicit OPC selection;
* correct ligand enhanced-region selection;
* concise `rest2.py`;
* Amber-like launcher;
* valid dry generation;
* no machine-specific paths.

A long phenol REST2 calculation is not required in this milestone. A short smoke test may be used if the environment supports it.

Do not claim that implicit-solvent ligand REST2 is scientifically validated.

## Scaling and exchange validation

Add focused scientific tests:

1. `tau=0` reproduces the unscaled base system.
2. Solute-solute electrostatics and Lennard-Jones terms scale by `s`.
3. Solute-environment terms scale by `sqrt(s)`.
4. Environment-environment terms remain unchanged.
5. Solute torsions scale correctly.
6. Omega exclusions remain unchanged.
7. CMAP handling is verified.
8. Unknown energy-bearing forces are refused.
9. Whole-system GBn2 scaling is correct.
10. Partial-region GBn2 is refused.
11. OpenMMTools reduced potentials agree with direct OpenMM evaluations.
12. Exchange probabilities agree with the analytical Metropolis criterion.
13. The OpenMMTools implementation agrees with the old custom loop on fixed inputs and seeds where the algorithms are equivalent.
14. A two-replica ladder attempts exchanges rather than skipping every second phase.
15. Walker and state trajectory reconstruction is correct.
16. Resume preserves iteration, state assignment, statistics, and output continuity.
17. Extension adds exactly the requested attempts.
18. Corrupt or incompatible storage fails clearly.

Do not keep the custom exchange loop solely because it is easier to make its tests pass.

## `MD-project` integration

After `MD-templates` passes its tests and is pushed:

1. Pin the exact new full commit SHA in `MD-project`.
2. Update every active example/configuration that should use the new generator.
3. Preserve historical dataset provenance.
4. Add ALA and phenol/IPH REST2 setup examples.
5. Generate data beneath `$MD_DATA`.
6. Keep only ignored local links or aliases under `md/`.
7. Do not commit trajectories, NetCDF files, checkpoints, or component clones.
8. Add a Snakemake boundary that invokes the generated REST2 launcher.
9. Treat the atomic REST2 completion manifest as the committed-analysis boundary.
10. Document walker versus state trajectories.
11. Document comparison with Amber groupfiles and GROMACS replica exchange.
12. Explain that OpenMMTools, rather than `MD-project`, owns replica exchange and restart.

The project workflow must not parse a log line as its only completion test. Validate the completion manifest and referenced NetCDF/checkpoint identities.

## Documentation

Update README documentation with:

* the short REST2 Python example;
* the Amber-like `openmm-rest2` command;
* ALA six-replica workflow;
* phenol/IPH generation workflow;
* `$MD_DATA` storage and local symlinks;
* OpenMMTools ownership;
* walker versus thermodynamic-state output;
* NetCDF structure;
* restart and extension;
* CPU, single-GPU and MPI multi-GPU commands;
* explicit limitations.

Include a concise comparison:

| Feature             | Amber                       | GROMACS           | This implementation  |
| ------------------- | --------------------------- | ----------------- | -------------------- |
| single-stage runner | `pmemd.cuda`                | `gmx mdrun`       | `openmm-md`          |
| replica launcher    | `pmemd.cuda.MPI -groupfile` | multidir/REMD     | `openmm-rest2` + MPI |
| protocol input      | `.in`                       | `.mdp/.tpr`       | concise Python       |
| exchange engine     | Amber                       | GROMACS           | OpenMMTools          |
| multistate storage  | Amber outputs               | per-replica files | OpenMMTools NetCDF   |
| physical data root  | user-selected               | user-selected     | `$MD_DATA`           |

Do not claim bitwise equivalence among engines.

## Tests and completion

Run:

* formatting and linting;
* all unit and contract tests;
* complete non-GPU `MD-templates` suite;
* available GPU tests;
* OpenMMTools REST2 CPU smoke;
* available CUDA smoke;
* MPI multi-GPU smoke when at least two GPUs are available;
* ALA and phenol/IPH dry generation;
* complete `MD-project` tests;
* YAML validation;
* ignore-policy checks;
* wheel installation test from outside the checkout.

Record exact results. A skipped GPU/MPI test is not a pass.

Inspect `git status` for accidental `.nc`, `.dcd`, `.chk`, XML state, cache, environment, or generated data files.

Make logical commits and push:

```text
MD-templates → feat/openmmtools-rest2
MD-project   → dev4-openmmtools-rest2
```

Do not merge.

Finish with:

* starting and final SHAs;
* architecture implemented;
* OpenMMTools APIs used;
* any OpenMMTools extension introduced;
* scientific scaling validation;
* output and restart contract;
* ALA and phenol/IPH results;
* CPU/GPU/MPI test results;
* performance measurements;
* known limitations;
* commits and branches pushed.

