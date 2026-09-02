# MD-tools

Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and
registering MD datasets.

One executable, `md-openmm`, and four commands. It builds a solvated, parameterised OpenMM system
from a structure; generates readable entry points and Amber-like inputs for a chosen protocol —
ordinary MD, REST2, rREST2 or annealed importance sampling; runs them on CUDA, under `mpirun` when
the protocol is parallel; and moves a finished run into managed storage as a verified, immutable
dataset.

**Status: unreleased.** Version `0.5.0.dev0`, developed on `dev`. Not on PyPI, not tagged.

## Environment

Python 3.12 and OpenMM 8.6, in a conda environment. OpenMM, OpenFF and AmberTools are conda
packages; installing MD-tools must not pull a second, pip-built OpenMM alongside them:

```bash
micromamba create -f environment-ci.yml && micromamba activate md-tools-ci
pip install --no-deps .
```

`--no-deps` is deliberate. The scientific stack comes from conda; this package adds only pure
Python.

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
md-openmm md-run -i min.in -p ../built.pdb -x ../built.xml -r min.xml -log min.log
#   or as ordinary Python -- the same run, reaching the same installed code:
python min.py -p ../built.pdb -s ../built.xml -r min.xml -x min.dcd -log min.log

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

**CUDA is the default and it is mandatory.** There is no automatic fall back — a run that quietly
moved to the CPU finishes, writes a trajectory and reports success two orders of magnitude later.
`--cpu` is the only way to ask for a CPU run, and the record says that you did.

Parallel protocols are launched the way Amber launches them:

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -x built.xml -odir REST2/
mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in   -p built.pdb -x built.xml \
          -source-traj cMD_tau0p5/tau_0p5.nc -odir AIS/
```

See [Running](docs/md-run.md) for the flags, the input language, the platform policy, the MPI
rules, and what AIS writes.

## Configuration

Six browsable examples at the repository root, one copy each, shipped as wheel data files:

```text
configs/machine/user.config.example        identity and $MD_DATA, for `data-register --init`
configs/sys/build-top.config               force fields, solvent, box, ions, constraints, HMR
configs/md/{cMD,REST2,rREST2,AIS}.config   protocol, stage lengths, reporting
```

They are YAML despite the `.config` suffix, unknown keys are refused with a suggestion, and every
duration is an integer step count. Smaller, task-sized examples live beside each
[method page](docs/openmm_methods/README.md).

## Documentation

| page | what it covers |
|---|---|
| [Running](docs/md-run.md) | `md-run`: the flags, the `.in` language, CUDA policy, MPI ladders, AIS paths and their outputs |
| [Methods](docs/openmm_methods/README.md) | cMD, REST2, rREST2 and AIS: purpose, inputs, generated files, restart, limitations |
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
