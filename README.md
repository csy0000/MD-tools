# MD-templates

Generates OpenMM input files and run scripts for conventional MD and REST2. It does not run your
science for you: it produces a small directory of ordinary OpenMM scripts that you can read, edit
and move to another machine.

Currently OpenMM only. Amber and GROMACS are **not** supported and are not in progress.

**What it covers**

| | |
|---|---|
| methods | conventional MD, REST2 (replica exchange with solute tempering) |
| solutes | peptides (ff19SB) and non-peptides (Sage 2.2 + AM1-BCC) |
| explicit solvent | OPC water, dodecahedral box, 2.0 nm padding, 0.15 M NaCl, 1.0 nm cutoff |
| implicit solvent | GBn2 with mbondi3 radii |
| defaults | 2 fs timestep, no hydrogen mass repartitioning |

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
md-openmm sys-config --method cMD REST2 --peptide true --solvent OPC
```

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
released. There is no active barostat during minimisation or NVT and exactly one during NPT.
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
`leaprc.protein.ff14SB`, the force field GBn2 was developed and validated against, while explicit
OPC uses ff19SB. ff19SB's amino-acid-specific CMAPs were fit in explicit water and no GB model has
been reparameterised against them, so `sys-gen` refuses that pair rather than running it.

The chain here is `minimization -> eq/nvt_1kcal -> eq/nvt_free -> cMD`: no NPT stage, and no
barostat anywhere, because a non-periodic system has no box to control.

The input is a file containing a SMILES string. The ligand is parameterised with Sage 2.2 and
standard AM1-BCC charges (AmberTools `sqm`).

Implicit solvent has no periodic box, so:

* no barostat is added;
* production ensembles are **NVT**, not NPT;
* `pressure_bar` is written as `null`, with a note saying it does not apply.

GBn2 is built through ParmEd rather than `AmberPrmtopFile` — the two differ by about 16 kJ/mol in
`CustomGBForce` for identical radii, so the construction path is part of the Hamiltonian.

---

## Moving a project to another machine

Copy `inputs/` and `MD/` together, keeping them siblings. Nothing else is needed: the scripts
address `inputs/` relatively, contain no path into this repository, and do not import
`md_templates`. On the target machine you need OpenMM (and, for a non-peptide system you intend to
rebuild, the OpenFF stack) — the same environment `md-template install` creates.

If the recorded interpreter is missing, `run.sh` falls back to whatever `python3` provides.

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
both of these together:

```yaml
# sys.config.yaml
constraints:
  hydrogen_mass_amu: 3.024

# md.config.yaml
common:
  timestep_fs: 4.0
```

`md-gen` refuses 4 fs with unrepartitioned hydrogens and names both files, because the two settings
live apart and each looks reasonable on its own.

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
make it FAIR, and it never assigns a dataset identifier or writes into `$MD_DATA`.

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
