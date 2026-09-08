# MD-tools

Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and
registering MD datasets.

One executable, `md-openmm`, and four commands. It builds a solvated, parameterised OpenMM system
from a structure; generates readable entry points and Amber-like inputs for a chosen protocol —
ordinary MD, REST2, rREST2 or annealed importance sampling; runs them on CUDA, under `mpirun` when
the protocol is parallel; and moves a finished run into managed storage as a verified, immutable
dataset.

**Status: unreleased.** Version `0.5.0.dev0`, developed on `dev`. Not on PyPI, not tagged.

## Getting started

Four steps, in order. If you already have a conda environment with OpenMM 8.6, skip to step 3.

### 1. A package manager, if you have none

The scientific stack is conda packages. Any of conda, mamba or micromamba works; micromamba is a
single static binary and needs no base environment:

```bash
mkdir -p ~/software/md-stack && cd ~/software/md-stack
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=$PWD
```

### 2. The environment

```bash
micromamba create -y -p ~/software/md-stack/envs/md-tools -f environment-ci.yml
```

That installs Python 3.12, OpenMM 8.6, OpenFF, AmberTools, ParmEd, RDKit, MDTraj, OpenMMTools and
NetCDF4 — everything the four commands import.

**It is deliberately CPU-only and single-rank.** `environment-ci.yml` is what CI validates
against, and CI has no GPU and no second device to bind a rank to. For real work add both:

```bash
micromamba install -y -p ~/software/md-stack/envs/md-tools -c conda-forge mpi4py openmpi
```

CUDA needs nothing extra on a machine with a working NVIDIA driver — conda-forge's OpenMM carries
the CUDA platform and finds the driver at run time. Verify rather than assume:

```bash
python -c "from openmm import Platform; print([Platform.getPlatform(i).getName()
                                               for i in range(Platform.getNumPlatforms())])"
# -> ['Reference', 'CPU', 'CUDA', 'OpenCL']
```

If `CUDA` is absent, the platform is missing or the driver is not visible; runs will refuse rather
than silently fall back, which is the intended behaviour.

Without `mpi4py`, single-process and single-GPU runs are fully supported. A launch of more than
one rank without it is a **fatal preflight error before any output exists** — N uncoordinated
ranks would each run a whole simulation over one set of paths and the result would look complete.

### 3. The package

```bash
pip install --no-deps .
```

`--no-deps` is deliberate and not optional. The scientific stack comes from conda; this package
adds only pure Python. Without it, pip pulls a second, pip-built OpenMM alongside the conda one
and the two disagree about which native libraries are loaded.

### 4. Sourcing it, every time

**Activate the environment before running anything.** Not a convenience — AM1-BCC goes through
AmberTools' `sqm`, which the OpenFF toolkit discovers by looking on `PATH`. In a bare shell it is
not found even though it is installed, `AmberToolsToolkitWrapper` is silently absent from the
registry, and AM1-BCC becomes unavailable under its own name.

```bash
micromamba activate ~/software/md-stack/envs/md-tools
```

Or put the environment on `PATH` from your shell profile, which is what a shared machine usually
wants:

```bash
# ~/.bashrc
export PATH="$HOME/software/md-stack/envs/md-tools/bin:$PATH"
export MD_DATA="/path/to/your/managed/storage"     # where data-register writes
```

Check it:

```bash
md-openmm --version
command -v sqm mpiexec        # both must resolve inside the environment
```

### 5. Once per machine, before registering data

`md-openmm data-register` needs to know who you are and where managed storage is. This writes
`~/.config/md-tools/user.config` and is asked once:

```bash
md-openmm data-register --init
```

It asks for a stable lowercase `person_id`, your name for provenance, and `$MD_DATA`. Nothing else
in the four commands needs it — building and running work without it.

## The four commands

```text
md-openmm build-top      a structure       -> built.xml + built.pdb + built.log
md-openmm build-md       a protocol config -> run scripts, .in files and run.sh in ./md_script/
md-openmm md-run         an Amber-like .in -> a stage, a ladder, or AIS switching paths
md-openmm data-register  a finished tree   -> a verified dataset under $MD_DATA
```

Nothing else is public. AIS is `protocol: AIS` in a `build-md` configuration, not a command of its
own, and `md-run` is a subcommand rather than a second executable.

## End to end

```bash
# 1. build the system
md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log

# 2. generate the workflow
md-openmm build-md -odir ./md_script/ --config configs/md/cMD.config

# 3. run it
cd md_script && ./run.sh
#   or one stage at a time, the Amber-like way:
md-openmm md-run -i min.in -p ../built.pdb -s ../built.xml \
    -o min.out -x min.dcd -r min.xml -log min.log
#   or as ordinary Python -- the same run, reaching the same installed code:
python min.py -p ../built.pdb -s ../built.xml -o min.out -x min.dcd -r min.xml -log min.log

# 4. register the result
md-openmm data-register -idata ./data/ALA-cMD \
    -project_name ALA -data_name ALA-cMD -year 2026
```

The generated files are compact entry points, not copies of the implementation:

```python
#!/usr/bin/env python
from md_tools.md import run_generated_stage
raise SystemExit(run_generated_stage(__file__, "min"))
```

The behaviour lives in the installed package; `resolved.config` beside the script is the
authoritative resolved declaration of the workflow, found relative to `__file__`. Move the
directory and it still runs.

Flags follow Amber: `-x` is the trajectory (mdcrd) and `-o` the readable output (mdout). `-s` is
the serialised OpenMM System, which Amber has no counterpart for because its prmtop carries the
parameters that live in `built.xml` here. `-o` and `-log` are two files for two readers: `.out` is
what you tail during a run, `.log` is the provenance record a machine parses.

**CUDA is the default and it is mandatory.** The platform is a property of the machine and is
configured once, in `machine.openmm` in the user configuration; it is not part of any protocol.
There is no automatic fall back — a run that quietly moved to the CPU finishes, writes a trajectory
and reports success two orders of magnitude later. `--cpu` overrides the machine default for one
invocation, and the record distinguishes all three provenances.

Parallel protocols are launched the way Amber launches them:

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \
          -o REST2.out -x REST2.nc -r restart.json -log REST2.log
mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in   -p built.pdb -s built.xml \
          -source-traj ../cMD_tau0p5/tau_0p5.dcd -o AIS.out -log AIS.log -odir ./AIS
```

See [Running](docs/md-run.md) for the flags, the input language, the platform policy, the MPI
rules, and what AIS writes.

## Worked examples you can run

Each method page carries a runnable script beside its prose and its configuration, so the
documented commands cannot drift from what actually works:

```text
docs/openmm_methods/cMD/README.md         what the method is, and what the generated directory holds
docs/openmm_methods/cMD/example.config    the configuration those commands use
docs/openmm_methods/cMD/README.sh         the same commands, executable
```

```bash
cd docs/openmm_methods/cMD
./README.sh                     # the documented example, on a GPU
./README.sh --quick --cpu       # a small implicit-solvent version, anywhere
./README.sh -i my_protein.pdb   # your own structure
```

Read `README.sh` as instructions or run it; it builds a system, generates the workflow and runs
the chain, explaining each command as it goes. It never deletes anything: if its output directory
exists it stops and says so.

## Configuration

Six browsable examples at the repository root, one copy each, shipped as wheel data files:

```text
configs/machine/user.config.example        identity, $MD_DATA and the machine's OpenMM defaults
configs/sys/build-top.config               force fields, solvent, box, ions, constraints, HMR
configs/md/{cMD,REST2,rREST2,AIS}.config   protocol, stage lengths, reporting, collective variables
```

They are YAML despite the `.config` suffix, unknown keys are refused with a suggestion, and every
duration is an integer step count. Smaller, task-sized examples live beside each
[method page](docs/openmm_methods/README.md).

## Documentation

| page | what it covers |
|---|---|
| [Running](docs/md-run.md) | `md-run`: the flags, the `.in` language, CUDA policy, MPI ladders, AIS paths and their outputs |
| [Methods](docs/openmm_methods/README.md) | cMD, REST2, rREST2 and AIS: purpose, inputs, generated files, restart, limitations |
| [Collective variables](docs/collective_variables/README.md) | torsion CV reporting: the `cv.yaml` schema, conventions, exact cadences, outputs per protocol |
| [Scientific defaults](docs/scientific-defaults.md) | every consequential default, the evidence for it, and what that evidence does not support |
| [Data registration](docs/data_register/README.md) | `--init`, canonical paths, the transaction, extensions |
| [The dataset contract](docs/data-contract.md) | the schema-level authority for records |
| [Support matrix](docs/support-matrix.md) | supported, experimental, and unsupported combinations |
| [Release notes](docs/release-notes/v0.5.0.md) | what changed in this development cycle |

Working on the code: [`CLAUDE.md`](CLAUDE.md) holds the invariants that must not be broken.

## Tests

```bash
python -m pytest tests -m "not slow and not gpu"    # fast
python -m pytest tests -m "gpu or slow"             # builds systems and integrates them, on CUDA
```

GPU tests run on CUDA and are never satisfied by CPU execution.
