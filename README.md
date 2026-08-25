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
renumbering), and the paths it will use. `install` creates `$MD_STACK/envs/openmm-8.6.0`, prefers
CUDA if a GPU is present, then imports OpenMM and allocates a CUDA context to check the platform
actually works. What it found is recorded under `installed.openmm`, and the full command and output
go to `$MD_STACK/logs/`.

It uses whichever of `micromamba`, `mamba` or `conda` is already installed. It will not install a
package manager, a driver, or another MD engine — if something is missing it says so and stops.

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
├── cMD/
│   ├── run.py              # an ordinary OpenMM script
│   └── run.sh
└── REST2/
    ├── run.py
    ├── rest2_scaling.py    # the tau scaling, beside the script that uses it
    ├── run.sh
    └── extend.sh
```

**Conventional MD**

```bash
cd MD/cMD && ./run.sh
```

Minimises, equilibrates, then runs production, checkpointing as it goes. Run it again and it
continues from `production.chk` rather than starting over.

**REST2**

```bash
cd MD/REST2 && ./run.sh
```

One replica per rung of a tau ladder. Only the solute Hamiltonian is scaled, so every replica is
the same physical system at a different effective solute temperature:

```text
solute–solute        (1 - tau)^2
solute–environment   (1 - tau)
```

Torsions about a peptide bond are left unscaled, so the hot rungs cannot rotate an amide that the
cold rung never rotates. Exchanges are attempted between alternating nearest neighbours, and under
NPT the acceptance carries the pV term explicitly — replicas share a pressure but not a beta, so it
does not cancel.

Every attempt is appended to `exchange_attempts.csv`:

```text
attempt_index,phase,step,time_ps,replica_i,replica_j,log_acceptance,accepted
```

**Extending from checkpoints**

```bash
cd MD/REST2 && ./extend.sh 3      # three more segments
```

Each replica resumes from its own checkpoint and the exchange history is appended, never rewritten.
The attempt indices continue, so a resumed run is one trajectory rather than several. For cMD,
raise `duration_ns` in `MD/md.config.yaml` and run `./run.sh` again.

**Choosing hardware**

```bash
MD_PLATFORM=CPU ./run.sh        # force a platform
MD_DEVICE=0 ./run.sh            # cMD: pin one CUDA device
CUDA_VISIBLE_DEVICES=0,1,2 ./run.sh   # REST2: use these three
```

REST2 uses at most `min(number_of_replicas, visible GPUs)`, assigning replicas round-robin. There
is no busy-device detection: if you want particular cards, name them.

---

## Example 2 — implicit solvent, non-peptide

```bash
md-openmm sys-config --method cMD --peptide false --solvent GBn2
md-openmm sys-gen -i ./ligand.smi --config sys.config.yaml -of ./inputs/
md-openmm md-gen -if ./inputs/ --config md.config.yaml -of ./MD/
cd MD/cMD && ./run.sh
```

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

## Tests

```bash
pytest tests/ -m "not slow"     # configuration, CLI, machine inspection
pytest tests/                   # everything, including MD smoke runs
```

The slow tests build systems and integrate them. Run the whole suite before a release, not on every
commit.

---

## Requirements

Python ≥ 3.11, OpenMM 8.6.0, and for system preparation the OpenFF toolkit, AmberTools and ParmEd.
`md-template install` sets up an environment with OpenMM; the preparation stack is expected in the
environment you run `sys-gen` from.
