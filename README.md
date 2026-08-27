# MD-templates

Generates OpenMM input files and run scripts for conventional MD and REST2. It does not run your
science for you: it produces a small directory of ordinary OpenMM scripts that you can read, edit
and move to another machine.

Currently OpenMM only. Amber and GROMACS are **not** supported and are not in progress.

**What it covers**

| | |
|---|---|
| methods | conventional MD, REST2 (replica exchange with solute tempering), AIS (annealed importance sampling) |
| solutes | peptides (ff14SB) and non-peptides (Sage 2.2.1 + AM1-BCC) |
| explicit solvent | **TIP3P** water, dodecahedral box, **1.5 nm** padding, 0.15 M NaCl, PME at 1.0 nm |
| explicit alternative | ff19SB + OPC, one flag away (`--solvent OPC`) |
| implicit solvent | GBn2 with mbondi3 radii, no surface-area term — peptides and proteins |
| dynamics | Langevin-middle at 300 K, friction 1.0 ps⁻¹, 2 fs, no hydrogen mass repartitioning |
| pressure | Monte Carlo barostat at 1 bar, attempting a move every 25 steps |

Every one of those values is argued, with its evidence classified and its limitations stated, in
**[docs/md-defaults-scientific-rationale.md](docs/md-defaults-scientific-rationale.md)**
([PDF](docs/md-defaults-scientific-rationale.pdf), references in
[docs/md-defaults-references.bib](docs/md-defaults-references.bib)). Read section 12 —
"What these citations do not establish" — before quoting any of it as support for a result.

---

## The six commands

```bash
md-template init            # create an MD stack directory and inspect the machine
md-template install         # install OpenMM into it
md-openmm    sys-config     # write sys.config.yaml and md.config.yaml
md-openmm    show-default   # print the defaults those files start from
md-openmm    sys-gen        # build the OpenMM system
md-openmm    md-gen         # generate the run scripts
```

There is no seventh. AIS is a `--method`, not a command:

```bash
md-openmm sys-config --method cMD AIS --peptide true --solvent TIP3P
md-openmm show-default AIS
```

---

## Example 1 — explicit solvent, ALA peptide

### 1. Set up a stack and install OpenMM

```bash
md-template init --target-dir ~/MD_STACK
export MD_STACK=~/MD_STACK
md-template install -e openmm -ev 8.6.0
```

`init` writes `$MD_STACK/machine.yaml`: cores, memory, GPUs (by UUID, so the record survives
renumbering), and the paths it will use.

`install` creates `$MD_STACK/envs/openmm-8.6.0` containing everything the six commands need, not
only OpenMM — building a system parameterises with OpenFF and AmberTools, so an environment holding
OpenMM alone fails part-way through `sys-gen`:

| package | needed for |
|---|---|
| `openmm` | the simulations themselves |
| `pyyaml`, `numpy` | configuration, box geometry |
| `openff-toolkit`, `openmmforcefields` | ligand parameterisation (SMIRNOFF templates) |
| `openff-nagl-models` | the optional `am1bcc_nagl` charge method |
| `ambertools` | `sqm`/`antechamber` for AM1-BCC, `tleap` for implicit peptide topologies |
| `parmed` | prmtop/rst7 ⇄ OpenMM |
| `rdkit` | SMILES → 3D conformer |

It then validates what it built: imports each package, lists the platforms, takes one integration
step on Reference, on CPU and — if a GPU is present — on CUDA, and checks that OpenFF has a
registered AmberTools toolkit so standard AM1-BCC actually resolves. Package versions, executable
paths and every check go into `machine.yaml` under `installed.openmm`; the full command and output
go to `$MD_STACK/logs/`. If the environment cannot run the advertised workflows, the command fails
and names what is missing.

To check an environment you built yourself, or re-check one later:

```bash
md-template install --validate /path/to/env
```

It uses whichever of `micromamba`, `mamba` or `conda` is already installed. It will not install a
package manager, a driver, or another MD engine — if something is missing it says so and stops.

### 1b. Activate the environment and put the CLI in it

The environment is a normal conda prefix. Activate it, then install this package inside it, so
`md-openmm` and the OpenMM it drives are the same Python:

```bash
conda activate $MD_STACK/envs/openmm-8.6.0     # or: micromamba activate $MD_STACK/envs/openmm-8.6.0
pip install --no-deps md-templates             # or: pip install --no-deps -e /path/to/MD-templates
```

`--no-deps` is deliberate: the scientific stack is already there from conda, and letting pip
re-resolve it would pull a second, pip-built OpenMM alongside the conda one. Check it landed in the
right place with `python -c "import md_templates, openmm; print(md_templates.__file__)"`.

### 2. Write the configuration

```bash
md-openmm sys-config --method cMD REST2 --peptide true            # the default: TIP3P
md-openmm sys-config --method cMD REST2 --peptide true --solvent OPC     # the alternative
```

`--solvent` picks the whole explicit combination, because the protein force field and the water
model are chosen together and not independently:

| `--solvent` | protein | water | why they are paired |
|---|---|---|---|
| `TIP3P` *(default)* | ff14SB (`amber14-all.xml`) | `amber14/tip3p.xml` | ff14SB's backbone correction is an empirical fit made in TIP3P, and Sage's aqueous training data used TIP3P |
| `OPC` | ff19SB (`amber19-all.xml`) | `amber19/opc.xml` | ff19SB's amino-acid-specific CMAPs were trained for a better water model, and its authors recommend OPC |
| `GBn2` | ff14SB (`leaprc.protein.ff14SB`) | none | GBn2 was developed in the ff99SB/ff14SB lineage; see Example 2 |

The two explicit rows are **coupled selections, not two independent keys**, and `sys-gen` enforces
that on the file rather than only on what `sys-config` wrote. Editing `solvent.model` to `OPC` and
leaving `forcefield.protein` at ff14SB is refused before a System exists, naming both fields —
crossing them would run to completion and report a Hamiltonian nobody validated.

Two files, both meant to be edited:

* **`sys.config.yaml`** — the molecular system: force fields, water model, box, salt, cutoff,
  constraints. **Change this when you change the system.** Rebuild with `sys-gen` afterwards.
* **`md.config.yaml`** — the protocol: temperature, timestep, durations, the REST2 ladder.
  **Change this when you change the simulation.** Regenerate with `md-gen`; no rebuild needed.

To see where those values come from:

```bash
md-openmm show-default sys
md-openmm show-default REST2
md-openmm show-default all
```

`show-default` and `sys-config` read the same definitions, so what you see is what you get.

### 3. Build the system

```bash
md-openmm sys-gen -i ./ALA.pdb --config sys.config.yaml -of ./inputs/
```

| file | what it is |
|---|---|
| `system.xml` | the serialized OpenMM `System` — force field, constraints, box |
| `topology.pdb` | the final topology and starting coordinates |
| `initial_state.xml` | starting positions and box vectors, before anything is integrated |
| `solute.yaml` | solute atom indices, the REST2 enhanced region, and the omega bonds excluded from scaling |
| `resolved_sys.config.yaml` | the settings actually used, after resolution |
| `provenance.yaml` | package version, git commit, OpenMM version, input hashes |
| `sys-gen.log` | what preparation did, in order |

`solute.yaml` is the reason the run scripts need nothing from this package: the classification is
done once, here.

### 4. Generate and run

```bash
md-openmm md-gen -if ./inputs/ --config md.config.yaml -of ./MD/
```

```text
MD/
├── md.config.yaml          # the resolved protocol, read by every run.py
├── provenance.yaml
├── md_stages.py            # the shared helper, one copy for the whole project
├── run_all.sh              # convenience wrapper; the stage scripts below are authoritative
├── minimization/           # the common chain: each stage reads its parent's final_state.xml
│   ├── run.py  run.sh  stage.yaml
├── eq/                     # the equilibration stages, grouped
│   ├── nvt_1kcal/          #   restrained NVT
│   ├── npt_1kcal/          #   restrained NPT     (explicit solvent only)
│   └── npt_free/           #   unrestrained NPT   (explicit solvent only)
├── cMD/                    # production, from the last common stage
│   ├── run.py
│   └── run.sh
└── REST2/                  # production, from the SAME last common stage
    ├── equilibrate.py      # per-tau equilibration, one directory per replica
    ├── equilibrate.sh
    ├── run.py              # exchange production
    ├── run.sh
    ├── extend.sh           # extends exchange production only
    ├── rest2_scaling.py    # the tau scaling, beside the script that uses it
    ├── replica_00/{equilibration,production}/
    └── replica_01/{equilibration,production}/
```

**The common chain**

Each stage is its own directory and its own run. They depend on each other through files:

```text
inputs -> minimization -> eq/nvt_1kcal -> eq/npt_1kcal -> eq/npt_free
                                                            ├─> cMD
                                                            └─> REST2
```

```bash
cd MD && ./run_all.sh                      # all of it, in order
cd MD/eq/npt_1kcal && ./run.sh             # or one stage at a time, which is authoritative
```

A stage reads only its parent's `final_state.xml` and writes `stage.log`, `stage.csv`,
`checkpoint.chk`, `final_state.xml`, `final.pdb` and `resolved_stage.yaml`. The checkpoint resumes
*that* stage; the final state is the handoff, and is written only once the stage succeeds — a
downstream stage that consumed a checkpoint would be starting from a partially finished parent
while every file on disk still looked normal. Run a stage whose parent has not finished and it
tells you which file is missing and which command makes it.

The solute is held by the configured `restraint_k_kcal_mol_a2` (`U = 1/2 k |r - r0|^2`,
1 kcal mol⁻¹ Å⁻² = 418.4 kJ mol⁻¹ nm⁻²) through minimisation and the restrained stages, then
released. There is no active barostat during minimisation or NVT and exactly one during NPT — the
Force is present in every explicit stage so the checkpoint layout never changes, and it is its
**frequency**, `common.barostat_frequency_steps`, that decides whether it attempts a move.
25 steps is OpenMM's own default: 0.05 ps at 2 fs, 0.10 ps at the optional 4 fs. Each stage records
both the step count and the interval in ps.
Implicit systems have no box: the chain is `minimization -> eq/nvt_1kcal -> eq/nvt_free`, and no
barostat exists in the System at all. If the restraint is not 1 kcal mol⁻¹ Å⁻², the directory is
named `eq/nvt_restrained` rather than claiming a strength it does not have.

A completed stage will not silently run again — downstream stages may already have consumed its
`final_state.xml`. Re-running says so and changes nothing. To redo it, generate into a new output
directory or remove that stage's runtime outputs (`stage.log`, `stage.csv`, `checkpoint.chk`,
`final_state.xml`, `final.pdb`, `resolved_stage.yaml` — not `run.py`, `run.sh` or `stage.yaml`) and
run it again; anything downstream was built on the old final state and is yours to regenerate.

Completion is identified by a SHA-256 of the whole `stage.yaml`, recorded as `stage_config_sha256`.
Change any field — the input state, the pressure, a seed — and the stage refuses rather than
accepting outputs that came from a different request.

Interrupt a stage and re-run it and it resumes from its own `checkpoint.chk` at the step it
reached, rather than starting the stage over.

**Conventional MD**

```bash
cd MD/cMD && ./run.sh
```

Production only — it does not minimise or equilibrate. It starts from the last common stage's
`final_state.xml`, the same file `REST2/equilibrate.py` starts from, so the two are siblings and
neither has to run before the other.

It writes `whole_system.dcd` and a solute-only `solute.dcd` at their own intervals, `production.csv`
and `production.chk`. Read `solute.dcd` against `inputs/solute.pdb`, which `sys-gen` writes from the
same atom indices.

Run it again and it continues from `production.chk` rather than starting over — reporters append,
and the barostat is restored before the checkpoint is loaded.

**REST2**

```bash
cd MD/REST2 && ./equilibrate.sh      # per-tau equilibration, once
cd MD/REST2 && ./run.sh              # exchange production
```

`equilibrate.py` takes every replica from the same common final state, applies that rung's scaled
Hamiltonian and relaxes under it into `replica_NN/equilibration/`. It repeats none of the common
minimisation or NVT/NPT preparation, and none of its steps count as production. `run.py` then runs
exchange production from each replica's equilibrated state into `replica_NN/production/`.

One replica per rung of a tau ladder. Only the solute Hamiltonian is scaled, so every replica is
the same physical system at a different effective solute temperature:

```text
solute–solute        (1 - tau)^2
solute–environment   (1 - tau)
```

Every rung is thermostatted at the same temperature, so the whole ladder shares one beta; the rungs
differ by Hamiltonian, not by thermostat. Torsions about a peptide bond are left unscaled, so the
hot rungs cannot rotate an amide that the cold rung never rotates.

Each replica is equilibrated separately at its own tau, then the run alternates:

```text
propagate every replica for duration_per_segment_ps -> attempt neighbour exchanges
```

`number_of_exchanges` times, so one invocation advances each replica by
`duration_per_segment_ps × number_of_exchanges`. Exchanges are attempted between alternating
nearest neighbours; a two-rung ladder exchanges on every round. Positions and box vectors travel
together, so the acceptance is

```text
log(alpha) = beta * [U_i(x_i,V_i) + U_j(x_j,V_j) - U_i(x_j,V_j) - U_j(x_i,V_i)]
```

with the pV contributions cancelling at the common beta and pressure. Every replica writes
`replica_NN_whole.dcd`, `replica_NN_solute.dcd`, `replica_NN.csv` and `replica_NN.chk`, and gets its
own integrator, velocity and barostat seeds.

Every attempt is appended to `exchange_attempts.csv`:

```text
attempt_index,phase,step,time_ps,replica_i,replica_j,log_acceptance,accepted
```

**Extending from checkpoints**

```bash
cd MD/REST2 && ./extend.sh 3      # three more exchange-production segments
```

Each replica resumes from its own `production/production.chk` and the exchange history is appended,
never rewritten. Extending does not repeat the common chain or the per-tau equilibration.
The attempt indices continue, so a resumed run is one trajectory rather than several. For cMD,
raise `duration_ns` in `MD/md.config.yaml` and run `./run.sh` again.

**Choosing hardware**

Runs use **CUDA by default**, and there is no quiet fallback: if no CUDA platform is available and
you have not named another one, the run fails rather than spending days on the CPU while looking
healthy.

```bash
./run.sh                              # CUDA
MD_DEVICE=0 ./run.sh                  # cMD: pin one CUDA device
CUDA_VISIBLE_DEVICES=0,1,2 ./run.sh   # REST2: spread the replicas over these three
MD_PLATFORM=CPU ./run.sh              # anything but CUDA must be asked for by name
```

REST2 uses at most `min(number_of_replicas, visible GPUs)`, assigning replicas round-robin.
Different devices propagate concurrently; replicas sharing a device propagate in turn, and every
replica is awaited before exchanges are attempted. There is no busy-device detection: if you want
particular cards, name them.

---

## Example 2 — implicit solvent, non-peptide

```bash
md-openmm sys-config --method cMD --peptide false --solvent GBn2
md-openmm sys-gen -i ./ligand.smi --config sys.config.yaml -of ./inputs/
md-openmm md-gen -if ./inputs/ --config md.config.yaml -of ./MD/
cd MD && ./run_all.sh
```

The protein force field is chosen **with** the solvation model: implicit GBn2 uses
`leaprc.protein.ff14SB`, the force field lineage GBn2 was developed and validated against. ff19SB's
amino-acid-specific CMAPs were fit in explicit OPC water and no GB model has been reparameterised
against them, so `sys-gen` **refuses** that pair rather than running it — it would run, and produce
plausible numbers, which is exactly the problem.

**Scope, and the limit of it.** The implicit route is supported for **peptides and proteins**. For a
Sage-parameterised small molecule it is marked `support_status: "experimental"` in
`forcefield.json`, and no Amber `igb=8` parity is claimed, for two measured reasons:

* `mbondi3`'s adjustments are keyed on GLU/ASP/ARG residue names and the `OXT` atom name, so for a
  one-residue `UNL` ligand mbondi3 is exactly mbondi2;
* GB-Neck2's α/β/γ were fit for H, C, N, O (S copies O). Any other element — a halogen, phosphorus,
  selenium — silently receives ParmEd's generic `α = 1.0, β = 0.8, γ = 4.85, screen = 0.5`.

`sys-gen` reads the built `CustomGBForce` back, counts the atoms outside the fit, prints a warning
naming their atomic numbers, and records the whole audit under
`implicit_solvent.parameter_coverage`. See §6 of the
[scientific rationale](docs/md-defaults-scientific-rationale.md).

The chain here is `minimization -> eq/nvt_1kcal -> eq/nvt_free -> cMD`: no NPT stage, and no
barostat anywhere, because a non-periodic system has no box to control.

The input is a file containing a SMILES string. The ligand is parameterised with Sage 2.2.1
(`openff-2.2.1`) and standard AM1-BCC charges (AmberTools `sqm`).

Implicit solvent has no periodic box, so:

* no barostat is added;
* production ensembles are **NVT**, not NPT;
* `pressure_bar` is written as `null`, with a note saying it does not apply.

GBn2 is built through ParmEd rather than `AmberPrmtopFile` — the two differ by about 16 kJ/mol in
`CustomGBForce` for identical radii, so the construction path is part of the Hamiltonian.

---

## Example 3 — AIS, annealed importance sampling

AIS anneals the REST2 Hamiltonian from `tau = 0.5` to `tau = 0` while the coordinates propagate,
and records the nonequilibrium work that switching costs. The complete editable example is
[`docs/examples/ais-ala.yaml`](docs/examples/ais-ala.yaml).

```bash
md-openmm sys-config --method cMD AIS --peptide true --solvent TIP3P
md-openmm show-default AIS                 # the block, with every required field marked null
# ...edit md.config.yaml: the switching duration, the source trajectory, the time window...
md-openmm sys-gen -i ./ALA.pdb --config sys.config.yaml -of ./inputs/
md-openmm md-gen  -if ./inputs/ --config md.config.yaml   -of ./MD/

cd MD && ./run_all.sh                      # equilibration and the cMD source run
cd MD/AIS && ./run.sh                      # prepares AIS/inputs/, then the switching paths
```

The first `AIS/run.sh` writes `AIS/inputs/sources.dcd` — the starting configurations — before any
path runs. After that the source trajectory is not needed again; see
[the source trajectory and its tau](#ais-the-source-trajectory-and-its-tau).

**AIS is not in `run_all.sh`, on purpose.** It starts from an equilibrium trajectory you have
already produced, so running it "in order" with everything else would run it before its own input
exists.

### The path

The Hamiltonian actually changes, through the same REST2 decomposition a fixed-tau cMD walker or a
REST2 rung uses — it is not interpolated between two endpoint energies:

```text
s        = (1 - tau)^2      solute-solute terms        quadratic along a linear path in tau
sqrt(s)  = 1 - tau          solute-environment terms   linear along a linear path in tau
```

Torsions about an omega bond are left unscaled, from the same `inputs/solute.yaml` list cMD and
REST2 read. Temperature and beta are constant throughout: **this is Hamiltonian switching, not
temperature annealing.** The enhanced region is the solute, as everywhere else.

### Where the configurations come from

AIS does not start from the common equilibration chain. It draws from an existing equilibrium
ensemble **at `tau_start`**:

* point `AIS.source.trajectory` at a **fixed-tau cMD run** (`cMD.tau: 0.5`) or at one **REST2
  rung** (`REST2/replica_NN/production/whole_system.dcd`);
* `start_time_ps` and `end_time_ps` are **inclusive** and select the eligible frames — use them to
  discard the equilibration part of the source run;
* the tau of the source is read from its own `resolved_run.yaml` and must equal `tau_start`. If it
  cannot be established the run **fails** rather than assuming 0.5;
* frame times come from the same record's `frame_time_map`, written where the reporter interval was
  decided. For any other trajectory set both `first_frame_time_ps` and `frame_interval_ps` —
  **a frame index is never treated as a time**, and a DCD header is not the production clock;
* selection is uniform over eligible frames, without replacement, seeded from the recorded master
  seed, and written to `selected_initial_frames.csv` *before* anything is propagated;
* a DCD carries no velocities, so each path draws fresh Maxwell–Boltzmann momenta at the common
  temperature with its own recorded seed.

### 21 observations are not 21 steps

A 100 ps switch at 2 fs is 50,000 integration steps and, at the default
`parameter_update_interval_steps: 1`, 50,000 parameter changes. **21** of those points are
*observed*: both endpoints plus 19 evenly spaced interior points, one coordinate frame each. The
update count must divide exactly by 20, or the 21 points would be rounded onto the update grid
instead of evenly spaced in tau — `md-gen` refuses a duration that does not divide, and says which
multiple would.

### The work convention

```text
delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)
```

The parameters change first, at frozen coordinates; the configuration then propagates under the new
Hamiltonian. `incremental_work_kj_mol` and `cumulative_work_kj_mol` are in **kJ/mol**;
`cumulative_reduced_work` is `beta * W` with the single common beta of `common.temperature_kelvin`.
Row 0 is the source configuration at `tau_start` with zero work; row 20 is `tau_end` with the total.

### What you get

```text
MD/AIS/
├── run.py  run.sh  rest2_scaling.py  path_definition.yaml   # written by md-gen
├── selected_initial_frames.csv        # the deterministic selection, before any dynamics
├── resolved_run.yaml  provenance.yaml # written by the run
├── trajectory_0000/
│   ├── observations.dcd               # exactly 21 whole-system frames
│   ├── observations.csv               # 21 rows, one per frame, with tau/s/sqrt(s) and the work
│   ├── final_state.xml  stdout.log  completed.json
└── trajectory_0001/ ...
```

**One DCD per path, never a shared one.** Each path is an independent realisation; concatenating
them would produce a file that looks like one continuous trajectory and is not.

### Execution

CUDA by default, with no silent CPU fallback. `gpu_devices: auto` uses every visible device;
at most `min(number_of_trajectories, visible GPUs)` paths run at a time, assigned round-robin and
deterministically, each in its own worker **process** — never one multithreaded Context across
devices. Paths sharing a device run in turn. Every path has distinct integrator and velocity seeds.

A completed path is skipped on a rerun, identified by its own completion record and its observation
count. An incomplete one is **replaced**, never appended to.

### What AIS here does not do

* **Forward only.** No reverse path, no bidirectional workflow.
* **No mid-path restart.** An interrupted path reruns from its source frame.
* **Fixed volume.** No barostat is active during switching and **no pressure–volume term is in the
  work**, even when the source ensemble was NPT. Each path keeps the box of the frame it started
  from.
* **No free-energy estimator.** The work columns are the input a Hummer–Szabo or Jarzynski analysis
  would consume. Computing one is not part of this repository, and nothing here claims a free
  energy.

---

## Moving a project to another machine

Copy `inputs/` and `MD/` together, keeping them siblings. Nothing else is needed: the scripts
address `inputs/` relatively, contain no path into this repository, and do not import
`md_templates`. On the target machine you need OpenMM (and, for a non-peptide system you intend to
rebuild, the OpenFF stack) — the same environment `md-template install` creates.

If the recorded interpreter is missing, `run.sh` falls back to whatever `python3` provides.

---

## Contract-managed datasets

By default the generators write wherever you point `-of`, and nothing about MD-data is involved.
Opt in, and the same two commands additionally write a `dataset.yaml` that satisfies the
[MD-data](https://github.com/csy0000/MD-data) **dataset contract v1** — validated by MD-data's own
validator, imported, never reimplemented here.

### The layout

A dataset lives at a three-segment path relative to `$MD_DATA`:

```text
$MD_DATA/{namespace}/{yyyy-mm}/{dataset_name}/
├── dataset.yaml            the manifest, and the only one in the tree
├── common/                 what sys-gen built
├── minimization/
├── eq/                     one component; nvt_1kcal, npt_1kcal, npt_free are STAGES inside it
├── cMD/  REST2/  AIS/      whichever methods you generated
```

`eq/nvt_1kcal` is a stage directory, not a component. Components are the top-level names in
`dataset.yaml`, and a component's `path` always equals its `name`.

### Installing the validator

MD-templates never carries a copy of MD-data's schema, so contract-managed generation needs the
validator itself. It is pinned to an exact commit over **HTTPS**, in one maintained place
(`md_data_contract.MD_DATA_REPOSITORY` / `MD_DATA_COMMIT`):

```bash
pip install 'md-data @ git+https://github.com/csy0000/MD-data.git@48628f9a5d3ace6c6398a63bc3905cd58d542de3'
```

`md-template install` does this for you as part of creating the OpenMM environment, and records the
package version, the pinned commit, whether the import succeeded and whether the validator
functions are present. It is not fatal if it fails — `dataset.enabled` is off by default and an
unregistered project never needs it — but the outcome is recorded either way, so a user who follows
the documented installation is not told one thing and given another.

A branch over SSH is deliberately *not* what is documented. `git+ssh://.../dev` needs a key agent
and moves under your feet: two people running the same documented command on the same day can end
up validating against different contracts, which is the exact drift this arrangement exists to
prevent.

### The two variables

| variable | meaning |
|---|---|
| `MD_DATA` | the storage root. Datasets are addressed relative to it, and the value is never written into any generated file. |
| `MD_DATA_LOCAL` | the one dataset directory this invocation may write into. Must sit under `MD_DATA`, must be a real directory (an alias symlink is refused), and must be `active`. |

Because `MD_DATA` is never recorded, moving the whole tree and re-pointing the variable is enough.

```bash
export MD_DATA=/scratch/md-data
export MD_DATA_LOCAL=$MD_DATA/adenosine/2026-08/a2a-apo-300k

md-openmm sys-gen -i ./A2A.pdb --config sys.config.yaml -of "$MD_DATA_LOCAL/common/"
md-openmm md-gen  -if "$MD_DATA_LOCAL/common/" --config md.config.yaml -of "$MD_DATA_LOCAL/"
```

`sys-gen` writes the manifest with the `common` component; `md-gen` reads the identity back out of
`common/resolved_sys.config.yaml` and adds the method components. The identity is declared **once**,
in `sys.config.yaml`, and never retyped.

### What you must supply, and why it is not guessed

`md-openmm show-default dataset` prints the block with every required field `null`:

`templates.commit` is additionally **checked against the MD-templates that is actually running**,
established from a Git checkout or a PEP 610 `direct_url.json`. A syntactically valid 40-hex string
that names a different commit is refused, and so is generation from an install where no exact commit
can be established — a pin that points at nothing is worse than none. A dirty checkout pins its HEAD
and says so loudly; `provenance.yaml` records `git_dirty`.

| field | why this repository cannot invent it |
|---|---|
| `dataset_id` | permanent and globally unique. MD-data mints it; a guess here would collide or lie. |
| `namespace`, `dataset_name` | they place the dataset in someone's catalogue. |
| `role`, `system` | what this data *is*. Only you know. |
| `created_by.person_id`, `.name` | attribution. Never inferred from a git config or a shell user. |
| `origin.repository`, `origin.commit` | the project this run belongs to. The commit must be an exact 40-hex SHA — never fabricated, never abbreviated, never `HEAD`. |
| `templates.commit` | which MD-templates checkout generated this. Same rule. |

Leave `enabled: false` and none of it applies. Set `enabled: true` and every field above must be
present, or `sys-gen` stops **before building anything**.

### Preflight, and `--check`

Every generated launcher — `run.sh`, `run_all.sh`, the REST2 workers, the AIS paths — runs the same
preflight before an OpenMM `Context`, an integrator, a worker process, a checkpoint or a trajectory
exists. It is not opt-in and there is no flag to skip it.

```bash
./run_all.sh --check     # the whole preflight, zero integration steps, nothing written
cd minimization && ./run.sh --check
```

`--check` runs the identical gate a real run runs and then stops. It creates no trajectory, no
checkpoint, no `final_state.xml`, no `resolved_stage.yaml` — and unlike a real run it does not even
append to the stage's `run.log`, because a check that leaves a trace in the record of what ran is
not the non-destructive thing it claims to be.

What preflight checks: the manifest validates against MD-data; the dataset is `active` and not
read-only; this directory's component is declared, owned and not a link; every record naming a
generator commit agrees; `common/` is present and its `SHA256SUMS` still match; `forcefield.json`
agrees with `resolved_sys.config.yaml`; the CUDA platform exists **and at least one device is
visible**; the parent stage completed the request it still asks for; and this stage has not already
completed a different one.

Two of those recompute rather than look:

**Stage identity.** A stored `stage_config_sha256` being present says only that something was
recorded. `--check` recomputes the fingerprint from the current `stage.yaml`, and the parent's from
the parent's, using the generated project's own `md_stages.stage_config_sha256` — one
implementation, imported, because two subtly different hashes over "the stage request" would have
the run write one and the check compare another. It also compares the parent's `final_state.xml`
against the digest recorded for it, so a handoff edited after the fact is caught.

```text
[FAIL] parent stage  nvt_1kcal/stage.yaml has changed since it ran: it now hashes to
                     91c027debaef but its completion record was written for 07f49acc0587.
```

**Force fields, exactly and by route.** An *absent* expected field fails — that is the case where
what was built is least knowable. A peptide system must claim no ligand force field, a ligand system
no protein one, and an implicit system neither a water model nor a barostat. The two acceptance
routes are ff14SB + TIP3P for a peptide and Sage 2.2.1 with its configured charge method + TIP3P
for a ligand; the generator does not build a combined protein-ligand complex, so no single test
system loads all three.

It is deliberately **bounded**. It reads a fixed list of named files. It never walks `$MD_DATA`,
never enumerates other datasets, never opens a trajectory and never hashes one. A preflight whose
cost grows with the size of the archive is a preflight people learn to disable.

### Lifecycle stays with MD-data

MD-templates writes `status: active` once, at creation. Moving a dataset to `complete` or
`archived`, minting aliases, registering extensions and computing archival checksums are MD-data
operations, performed deliberately. This repository refuses to write into a dataset that is already
`complete` or `archived` rather than quietly reopening it.

---

## AIS: the source trajectory and its tau

AIS starts from an equilibrium ensemble you already produced, so two things about that source have
to be true and provable.

**The tau is evidence, not an assumption.** A source produced by this repository carries a
companion record, and its tau is read from there. A source from anywhere else must say so
explicitly:

```yaml
AIS:
  source:
    trajectory: ../cMD/whole_system.dcd
    source_tau: 0.5          # required when there is no companion record
```

Missing tau is refused. A `source_tau` that contradicts the companion record is refused, naming
both values — it is never silently preferred. A `source_tau` that does not equal `path.tau_start`
is refused: annealing from a state sampled at a different Hamiltonian is not the calculation the
work values would be interpreted as.

**The production source is never hashed.** Not at generation, preflight, preparation or run time.
A full-file digest of a trajectory that may be hundreds of gigabytes costs more than it proves, and
it puts back exactly the cost the bounded survey exists to avoid. `sources.yaml` records bounded
observations instead — path, byte size, frame count, frame timing, selected indices, chunk size and
chunks read — and `trajectory_sha256` is explicitly `null` with a note saying why. MD-data hashes
the trajectory once, at archival. `AIS/inputs/sources.dcd` is a small generated input and is a
different file with a different name.

**The trajectory is streamed, never loaded.** Source frames are read with `mdtraj.iterload` in
bounded chunks (50 frames), in two passes: one to count and locate the frames the time window
selects, one to collect only those frames. Peak memory is set by the chunk size, not by the length
of the trajectory, so a nanosecond source and a microsecond source cost the same. `mdtraj.load` is
never called on a source trajectory, and a test enforces that by making it raise.

**The selected frames become a durable input.** Before any path runs, they are written to
`AIS/inputs/`:

```text
AIS/inputs/sources.dcd     one frame per path, box vectors already reduced
AIS/inputs/sources.yaml    where each frame came from, and what happens to the momenta
```

After that the source trajectory is **never opened again**. You can archive or delete a
200 GB production DCD and still rerun a failed path — the run consumes the prepared inputs, so
what was archived is exactly what ran. This is the AIS counterpart of keeping `inputs/` beside the
trajectories: a set of work values without the configurations they started from cannot say what it
annealed away from.

If the prepared inputs disagree with the configuration — a different window, seed, path count or
`tau_start` — the run **refuses** and names the field. It does not silently re-prepare, because
that would delete the configurations a finished path was started from.

### These are starting configurations, not restart files

`sources.dcd` holds **positions and box vectors only**. There are no velocities in it, and that is
deliberate:

* **DCD cannot carry them.** `mdtraj.Trajectory` has no velocity field at all, so no mdtraj writer
  could store them regardless.
* **AIS does not want them.** Each path draws fresh Maxwell-Boltzmann momenta at the one common
  temperature from its own recorded `velocity_seed`. The canonical distribution factorises,
  `π(x,p) ∝ e^{-βU(x)}·e^{-βK(p)}`, so an equilibrium configuration plus an independent momentum
  draw is a proper sample of the full ensemble — and it makes each path independent by
  construction rather than by how far apart its source frame happened to land.
* **Generated momenta satisfy the constraints.** `setVelocitiesToTemperature` applies the velocity
  constraints. Velocities restored from a file and set raw would not, and the run would not fail —
  it would quietly start with energy in modes that are supposed to be frozen.
* **A seed is better provenance than the data.** Four bytes in the record regenerate the draw
  exactly, so a path can be reproduced without carrying a coordinate-sized blob beside it.

A *restart* in this repository means `final_state.xml`, which does carry velocities. The two are
different things and `sources.yaml` says so in its own `velocities.note`.

`sources.yaml` also records `minimum_frame_gap` and `minimum_time_gap_ps` — how close the two
closest starting frames were. Two paths beginning a few femtoseconds apart are two nearly identical
configurations, and their work values are not the independent samples the CSV makes them look
like. Nothing enforces a spacing; the number is recorded so you can see it.

**A path is complete only if everything agrees**, and the frames are READ to establish it. A DCD
header is not evidence: `NSET` survives an interrupted write intact, so a file can claim 21 frames
and hold 15 and a bit. The small generated files — `AIS/inputs/sources.dcd` and every
`observations.dcd` — are validated by reading every frame with bounded `iterload`, requiring the
exact count, a readable final frame, finite coordinates and a non-degenerate box under explicit
solvent. That is affordable because these files hold one frame per path or the configured
observations; it is never pointed at a production trajectory.

On top of that, the completion record must say so, its trajectory index must match, the CSV must
exist with the right number of rows, and the rows must map one-to-one onto the frames. A path whose
DCD was byte-truncated, or deleted, while its JSON and CSV stayed healthy is reported and rerun,
never skipped — and it is replaced, never appended to.

---

## Archiving finished results

`inputs/` is not scratch. It is the statement of *what was simulated*, and it is the half of a
finished run that is easiest to leave behind:

| file | what it settles |
|---|---|
| `system.xml` | the built System — including whether hydrogen mass was actually repartitioned |
| `solute.yaml` | solute indices, and the omega bonds REST2 leaves unscaled |
| `initial_state.xml` | where the chain started |
| `topology.pdb`, `solute.pdb` | what the trajectories are read against |
| `provenance.yaml`, `resolved_sys.config.yaml` | what produced them, and from what |

So archive `inputs/` alongside the trajectories, not only `MD/`. Results without it cannot answer
which Hamiltonian produced them, and rebuilding one from the configuration afterwards is a weaker
claim than keeping the original: **the configuration records what was requested, `system.xml`
records what was built.** Those two disagreeing is not hypothetical — a profile requesting
`hydrogen_mass_amu: 3.024` once coexisted with a System carrying 1.008 amu hydrogens, and every
check that read only the configuration passed.

If an archive layout is imposed from outside (a project's own data tree, say), publish `inputs/`
into it too. Linking rather than copying keeps the archive readable while the generated restart
tree is still active.

Read a solute trajectory against `inputs/solute.pdb`, which `sys-gen` writes for exactly that
purpose — the whole-system topology has a different atom count.

---

## Faster settings, if you want them

The defaults are conservative: 2 fs with unmodified hydrogen masses. To trade that for speed, set
all of these together — the complete, editable example is
[`docs/examples/hmr-4fs.yaml`](docs/examples/hmr-4fs.yaml):

```yaml
# sys.config.yaml
constraints:
  type: HBonds          # bonds to hydrogen must be constrained
  rigid_water: true     # water is never repartitioned, so it must stay rigid
  hydrogen_mass_amu: 3.024

# md.config.yaml
common:
  timestep_fs: 4.0
```

`md-gen` refuses 4 fs with unrepartitioned hydrogens, with unconstrained hydrogen bonds, or with
flexible water, and names both files: the settings live apart and each looks reasonable on its own.
`sys-gen` then verifies the repartitioning group by group against the same System built without it,
and records the result under `constraints.hmr_group_conservation`.

**What this buys and what it costs.** Roughly a factor of two in wall-clock. Equilibrium
configurational averages are unaffected — the Boltzmann distribution in configuration space does not
depend on masses — but the masses themselves are no longer physical, so **kinetics, time correlation
functions, diffusion and every transport property are altered**. A short stable run is evidence of
stability, not of kinetic validity. See §11 of the
[scientific rationale](docs/md-defaults-scientific-rationale.md).

NAGL charges (`ligand_charge_method: am1bcc_nagl`) are available but never a default. NAGL is a
graph network *trained to predict* AM1-BCC ELF10 charges — close to them, but not that calculation,
so it is a different Hamiltonian.

---

## Running the tests

Molecular dynamics runs on a GPU, and so do the tests that exercise it:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest tests/ -q      # everything, on CUDA
pytest tests/ -q -m "not gpu"                      # packaging and unit tests, no GPU needed
```

Every test that minimises or integrates a molecular system is marked `gpu` and runs on CUDA. A CPU
or Reference run of those would exercise a different code path from the one the work is done on,
so on a machine without a working CUDA platform they are deselected rather than passed. CPU and
Reference are used only for installation and platform probes and for tests that build no system.

The release workflow runs on a GPU-less GitHub runner and therefore validates **packaging only** —
it deselects every `gpu` test and says so in its output. Runtime acceptance comes from the command
above, on the GPU machine.

## Tests

```bash
pytest tests/ -m "not slow"     # configuration, CLI, machine inspection
pytest tests/                   # everything, including MD smoke runs
```

The slow tests build systems and integrate them. Run the whole suite before a release, not on every
commit.

---

## Where this sits

MD-templates is one of three repositories. It makes simulation data **FAIR-ready**; it does not
make it FAIR. It never assigns a dataset identifier. It may write a run's own files and a
contract-valid `dataset.yaml` into the one active dataset directory you explicitly select — see
[Contract-managed datasets](#contract-managed-datasets) — and it never touches any other part of
`$MD_DATA`.

| repository | owns |
|---|---|
| [MD-templates](https://github.com/csy0000/MD-templates) | system construction, force-field record, resolved protocol, seeds, generated scripts, execution provenance |
| [MD-data](https://github.com/csy0000/MD-data) | permanent dataset ID, immutable `$MD_DATA` storage, complete archive checksums, metadata, retention and access |
| [MD-analysis](https://github.com/csy0000/MD-analysis) | analysis configuration, software identity, consumed dataset IDs and checksums, derived-result lineage |
| project-template | project inputs, configs, workflows and component locks — **not yet available** |

See [`docs/FAIR_HANDOFF.md`](docs/FAIR_HANDOFF.md) for what each letter of FAIR requires and which
repository delivers it.

## Provenance the generators write

`sys-gen` keeps your original file and records how the System was parameterised:

```text
inputs/
├── original_inputs/<your file>   byte-for-byte, checksummed
├── preparation/                  tleap/prmtop/ligand artifacts, when the route produces them
├── forcefield.json               how the System was parameterised, recorded where it was decided
├── provenance.yaml               command, implementation identity, environment, box, lineage
└── SHA256SUMS                    every file here, deterministic and reproducible
```

`md-gen` records which prepared system it came from — hashes of the parent `provenance.yaml`,
`forcefield.json` and `SHA256SUMS` — plus the stage plan, every seed, and `generated-files.sha256`
over what it wrote before any dynamics.

At runtime each stage writes `resolved_stage.yaml`; cMD and REST2 additionally write
`resolved_run.yaml` and append one line per invocation to `invocations.jsonl`, so a continued run
keeps its full history rather than overwriting it.

Trajectories are recorded by path, size and frame count, never hashed during a run — MD-data
computes archival checksums once.

### Retrofitting 0.3.x data

```bash
python scripts/retrofit_fair_v030.py --inputs ./inputs --md ./MD --output ./fair-registration \
    [--original-input ./original.pdb] [--environment ./env.yaml]
```

Reads only — it never modifies, renames or adds a file in `inputs/` or `MD/`. Every retrospective
value carries an evidence status (`recorded`, `derived`, `user_supplied`, `unknown`), and the result
is graded **A** rebuildable, **B** prepared-system reproducible, or **C** archival only, with every
reason machine-readable.

## Requirements

Python ≥ 3.11 and OpenMM 8.6.0 to run; the OpenFF toolkit, openmmforcefields, AmberTools, ParmEd
and RDKit to prepare a system. `md-template install` installs all of them and then validates that
they work — see the table in Example 1 for what each is for. Simulations run on CUDA by default and
refuse to fall back to the CPU silently; name another platform with `MD_PLATFORM` if you want one.
