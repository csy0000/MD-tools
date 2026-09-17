# MD-tools

Standalone, pip-installable OpenMM tools for building systems, generating MD workflows, and
registering MD datasets.

One executable, `md-openmm`, and five commands. It builds a solvated, parameterised OpenMM system
from a structure; generates readable entry points and Amber-like inputs for a chosen protocol —
ordinary MD, REST2, umbrella sampling or annealed importance sampling; runs them on CUDA,
under `mpirun` when the protocol is parallel; moves a finished run into managed storage as a
verified, immutable dataset; and exports one as a bundle that runs on OpenMM alone, with nothing
of this package in it.

**Status:** `0.5.4`, on `dev` and `main`, tagged `v0.5.4`. Not on PyPI.

📖 **[Documentation](https://csy0000.github.io/MD-tools/)** — installation, machine configuration,
the methods, and the reference pages.

## `md-tools` and `md-openmm` are not the same thing

They are named separately because they are separate, and a project that confuses them will put
code in the wrong place.

| | what it is |
|---|---|
| **`md-tools`** | the **package**. `import md_tools` — modular, inspectable implementations of the basic sampling techniques over OpenMM: Hamiltonian scaling, exchange, switching, restraints, reporting, the dataset contract. Open code, meant to be read and, where a study needs it, modified. |
| **`md-openmm`** | the **executable**. An Amber-like front end that mimics `pmemd.cuda`: it parses input and output flags — `-i`, `-p`, `-c`, `-x`, `-r`, `-o`, `-odir` — and calls those modules. |

`md-openmm` adds **no behaviour of its own**. Every protocol it dispatches is handed to the same
function a generated script calls, which is why the same run can be started three ways — the
executable, a generated `run.sh`, or ordinary Python — and reach identical code.

So: **the simulation logic lives in `md-tools`, and `md-openmm` is how you type it.** A project
building a new sampling method imports the package. It uses the executable to produce reference
simulations, not to express the method.

## Install

```bash
micromamba create -y -p ~/software/md-stack/envs/openmm-env -f environment.yml
micromamba activate ~/software/md-stack/envs/openmm-env
pip install --no-deps .
```

`--no-deps` is not optional: the scientific stack comes from conda, and without it pip pulls a
second OpenMM alongside the conda one. The NVIDIA driver, MPI, and the reasons for each step are
in **[Installing](https://csy0000.github.io/MD-tools/install/)**; the platform and `$MD_DATA` are
a machine property, configured once — see
**[Machine configuration](https://csy0000.github.io/MD-tools/machine-configuration/)**.

**Upgrading needs the reinstall, not just the pull.** This is not an editable install, and the
shipped `configs/*.config` are wheel data files read from `<env>/share/md-tools/`, so a `git pull`
leaves both the code and the examples on your `PATH` unchanged:

```bash
git pull && pip install --no-deps --force-reinstall .
md-openmm --version        # must match `version` in pyproject.toml
```

`--force-reinstall` because two `dev` commits usually share a `version`, and pip otherwise no-ops.

## The five commands

```text
md-openmm build-top         a structure       -> built.xml + built.pdb + built.log
md-openmm build-md          a protocol config -> a <method>-run<N>/ run beside build/, min/ and input/
md-openmm md-run            an Amber-like .in -> a stage, a ladder, or AIS switching paths
md-openmm data-register     a finished tree   -> a verified dataset under $MD_DATA
md-openmm export-reference  a finished run    -> a bundle that runs on OpenMM alone
```

Nothing else is public. AIS is `protocol: AIS` in a `build-md` configuration, not a command of its
own, and `md-run` is a subcommand rather than a second executable.

## End to end

```bash
# 1. build the system -- into the system's own build/, whose names are build-top's
md-openmm build-top -i ALA.pdb \
    -os build/built.xml -op build/built.pdb -log build/built.log

# 2. generate a run beside it. build/, min/ and input/ are SHARED by every run here;
#    only cMD-run1/ belongs to this one.
md-openmm build-md -odir ./cMD-run1 --config configs/md/cMD.config

# 3. run it
cd cMD-run1 && ./run.sh
#   or one stage at a time, the Amber-like way:
md-openmm md-run -i ../input/min.in -p ../build/built.pdb -s ../build/built.xml \
    -odir ../min

# 4. register the result. The unit is the RUN: it is registered only once the build/, min/
#    and input/ it ran against validate as the same construct, by digest, not by path.
md-openmm data-register -idata ./cMD-run1 \
    -project_name ALA -data_name ALA-cMD -year 2026
```

`build-top` takes a `.pdb` or a `.seq` (one line of residue names, built extended by tleap) for a
peptide, or a `.smi` or `.sdf` for a single small molecule — a
`.smi` is embedded and MMFF-minimised, a `.sdf` supplies its own coordinates and they are used as
given.

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
`--cpu` overrides the machine default for one invocation.

A REST2 ladder, AIS, the flag table, the `.in` language and the MPI rules are in
**[Running](https://csy0000.github.io/MD-tools/md-run/)**.

## Configuration

```text
configs/machine/user.config.example        identity, $MD_DATA and the machine's OpenMM defaults
configs/sys/build-top.config               force fields, solvent, box, ions, constraints, HMR
configs/md/{cMD,REST2,AIS,umbrella}.config   protocol, stage lengths, reporting, CVs
```

Every file is YAML despite the `.config` suffix, unknown keys are refused with a suggestion, and
every duration is an integer step count. The generated key-by-key reference is
**[Configuration](https://csy0000.github.io/MD-tools/md-configuration/)**; smaller, task-sized
examples live beside each method page.

## Tests

```bash
python -m pytest tests -m "not slow and not gpu"    # fast
python -m pytest tests -m "gpu or slow"             # builds systems and integrates them, on CUDA
```

GPU tests run on CUDA and are never satisfied by CPU execution.

Working on the code: [`CLAUDE.md`](CLAUDE.md) holds the invariants that must not be broken; open
and closed gaps, each with its reasoning, are in [`docs/backlog.md`](docs/backlog.md).
