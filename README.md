# MD-tools

Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and
registering MD datasets.

One executable, `md-openmm`, and five commands. It builds a solvated, parameterised OpenMM system
from a structure; generates readable entry points and Amber-like inputs for a chosen protocol —
ordinary MD, REST2, rREST2 or annealed importance sampling; runs them on CUDA, under `mpirun` when
the protocol is parallel; moves a finished run into managed storage as a verified, immutable
dataset; and exports one as a bundle that runs on OpenMM alone, with nothing of this package in
it.

**Status:** `0.5.2`, tagged `openmm-v0.5.2`, on `dev` and `main`. Not on PyPI.

`0.5.2` adds `export-reference`: a finished run becomes a directory someone can run in ten years
with OpenMM and nothing of ours. cMD and REST2 today; umbrella and AIS are not done. A wheel also
knows which commit it was built from now, so `md_tools_commit` stops being `null` in every record
and every manifest. See [`docs/release-notes/v0.5.2.md`](docs/release-notes/v0.5.2.md) — including
the four ways the first version of the exporter was wrong, and why its test agreed with all of
them.

`0.5.1` fixed the one a scheduler-killed campaign needs: an interrupted CV-enabled REST2 ladder can
be resumed. See [`docs/release-notes/v0.5.1.md`](docs/release-notes/v0.5.1.md).

Known limitations are stated
in [`docs/release-notes/v0.5.0.md`](docs/release-notes/v0.5.0.md); open and closed gaps, each with
its reasoning, in [`docs/backlog.md`](docs/backlog.md).

## `md-tools` and `md-openmm` are not the same thing

They are named separately because they are separate, and a project that confuses them will put
code in the wrong place.

| | what it is |
|---|---|
| **`md-tools`** | the **package**. `import md_tools` — modular, inspectable implementations of the basic sampling techniques over OpenMM: Hamiltonian scaling, exchange, switching, restraints, reporting, the dataset contract. Open code, meant to be read and, where a study needs it, modified. |
| **`md-openmm`** | the **executable**. An Amber-like front end that mimics `pmemd.cuda`: it parses input and output flags — `-i`, `-p`, `-c`, `-x`, `-r`, `-o`, `-odir` — and calls those modules. |

`md-openmm` adds **no behaviour of its own**. Every protocol it dispatches is handed to the same
function a generated script calls, which is why the same run can be started three ways — the
executable, a generated `run.sh`, or ordinary Python — and reach identical code. Anyone who has run
`pmemd -i mdin -p prmtop -c inpcrd -o mdout -x mdcrd -r restrt` can read the command line without a
manual; that convenience is the executable's entire job.

So: **the simulation logic lives in `md-tools`, and `md-openmm` is how you type it.** A project
building a new sampling method imports the package. It uses the executable to produce reference
simulations, not to express the method.

## Using `md-tools` from your own project

**Use the functions as built.** A method assembled from the shipped modules — the REST2 scaler, the
torsion restraints, the source-ensemble reader, the platform policy — inherits their tests, their
refusals and their provenance records. Reimplementing one is how two codebases start disagreeing
about the same quantity.

**When a module genuinely needs changing, record it.** If realising your method requires a small
modification to an existing `md-tools` function, write it down in your own repository, in
`<repo>/docs/md_tools_modifications.md`: which module, what changed, and why the shipped behaviour
was not enough. That file is what makes the change reviewable later — and it is what turns "our
results differ from the reference" from a mystery into a lookup.

**What belongs in that file, and what does not.**

* **Yes** — the module is still the right one and still usable, and an existing method needs a
  small change: an extra term, a parameter that was fixed and needs to vary, a hook where there
  was none.
* **No** — a different simulation route. A new way of moving between states, a new work
  convention, a new schedule: that is your method, it belongs in your repository's own code, and
  recording it as a "modification" would misfile the thing you actually built. See
  `CONTRIBUTING.md` in a consuming project for the technique / method / study split.

## Getting started

Step 0 needs root and is done once per machine; steps 1-5 are a normal user install. If you
already have a conda environment with OpenMM 8.6, skip to step 3.

### 0. The NVIDIA driver — the one step conda cannot do

**Needs root, and is done once per machine.** Everything else below is a normal user install.

Conda supplies the CUDA *toolkit* — `libcudart`, `libnvrtc`, `libcufft` — and nothing else. The
**driver** is the kernel module plus `libcuda.so.1`, and it comes from the operating system:

```text
libcuda.so.1   ->  /lib/x86_64-linux-gnu/libcuda.so.1     the DRIVER, from the OS
libnvrtc.so    ->  <env>/lib/libnvrtc.so                  the TOOLKIT, from conda
```

Both are loaded by OpenMM's CUDA plugin, from two different places. No conda package can install
the first: `environment-ci.yml` cannot rebuild a driver, and a machine without one has no CUDA no
matter what the environment contains.

Check what you have:

```bash
nvidia-smi        # driver version, and the highest CUDA version it supports
```

If that prints a table you are done — install the driver through your distribution or NVIDIA's
installer otherwise. The constraint runs one way: a driver is forward-compatible with older
toolkits, so a recent driver serves any CUDA version conda resolves, while a toolkit newer than
the driver supports fails at run time.

**No GPU?** Everything still installs and runs on the CPU and Reference platforms; skip this step
and pass `--cpu` at run time. GPU tests are then unavailable rather than silently passing.

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
NetCDF4 — everything the five commands import.

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
in the five commands needs it — building and running work without it.

## The five commands

```text
md-openmm build-top         a structure       -> built.xml + built.pdb + built.log
md-openmm build-md          a protocol config -> run scripts, .in files and run.sh in ./md_script/
md-openmm md-run            an Amber-like .in -> a stage, a ladder, or AIS switching paths
md-openmm data-register     a finished tree   -> a verified dataset under $MD_DATA
md-openmm export-reference  a finished run    -> a bundle that runs on OpenMM alone
```

Nothing else is public. AIS is `protocol: AIS` in a `build-md` configuration, not a command of its
own, and `md-run` is a subcommand rather than a second executable.

`export-reference` chooses what to write from the RECORD TYPE, not from a flag: `md-stage:*` is a
single Context driven for a fixed number of steps, `md-replica:*` is a ladder, and asking the
caller to say which would let them say the wrong one. A protocol with no exporter yet — umbrella,
AIS — is refused before anything is written.

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
