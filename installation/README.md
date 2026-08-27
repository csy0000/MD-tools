# Installing the OpenMM 8.6.0 stack

This repository simulates with OpenMM. Installing it is one command, and this page is the long
version of what that command does and how to check it worked.

Amber, AmberTools' `pmemd`, and GROMACS are **not** installed or driven by this repository. An
earlier version of this guide covered an Amber26 / `pmemd.cuda.MPI` build; that engine is not
implemented here and describing its installation would advertise something the CLI cannot do.
AmberTools is installed, but only for `sqm`/`antechamber` (AM1-BCC charges) and `tleap` (implicit
peptide topologies) — not as a simulation engine.

## What you need first

- Linux x86-64.
- One of `micromamba`, `mamba` or `conda` already on `PATH`. This repository will not install a
  package manager, a CUDA driver, or a compiler for you: those are the machine's business, and a
  script that installs a driver behind your back is worse than one that stops and says what is
  missing.
- For GPU runs, an NVIDIA driver new enough for the CUDA build conda-forge resolves. The generated
  run scripts default to CUDA and refuse to fall back to the CPU silently. `md-template install`
  now **bounds the solve above** by the CUDA version `nvidia-smi` reports the driver supports:
  unbounded, the solver takes the newest available, and a build newer than the driver installs
  cleanly and then fails every real System with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.

## 1. Create the stack and install

```bash
pip install md-templates                  # or: pip install -e /path/to/MD-templates
md-template init --target-dir ~/MD_STACK
export MD_STACK=~/MD_STACK
md-template install -e openmm -ev 8.6.0
```

`init` writes `$MD_STACK/machine.yaml` — cores, memory, GPUs by UUID so the record survives
renumbering — and creates the layout. `install` creates `$MD_STACK/envs/openmm-8.6.0`.

Add `--dry-run` to see the exact solve without running it.

## 2. What goes into the environment

Not OpenMM alone. Building a system parameterises through OpenFF and AmberTools, so an environment
holding only `openmm` fails part-way through `md-openmm sys-gen`:

| package | needed for |
|---|---|
| `openmm=8.6.0` | the simulations themselves |
| `pyyaml`, `numpy` | configuration files, box geometry |
| `openff-toolkit`, `openmmforcefields` | ligand parameterisation through SMIRNOFF templates |
| `openff-nagl-models` | the optional `am1bcc_nagl` charge method |
| `ambertools` | `sqm`/`antechamber` for AM1-BCC, `tleap` for implicit peptide topologies |
| `parmed` | prmtop/rst7 ⇄ OpenMM |
| `rdkit` | SMILES → 3D conformer |
| `mdtraj` | reads the AIS source trajectory and its periodic box vectors |

`cuda-version>=11.8` is added when `machine.yaml` records a GPU, so conda-forge picks an OpenMM
build matching the driver rather than a version pinned here.

The same list, as a conda environment file, is `environment-ci.yml` at the repository root, with
the CUDA pin removed for CPU-only machines. A test asserts the two agree.

## 3. What the installer checks

After the solve, `install` runs a probe **inside** the new environment and fails if it cannot run
the advertised workflows:

- every required package imports, and its version is recorded;
- `sqm`, `antechamber` and `tleap` are on `PATH`;
- OpenFF has a registered `AmberToolsToolkitWrapper`, without which `am1bcc` does not resolve at
  all — it is not enough for AmberTools to be installed, it must be discoverable when OpenFF builds
  its registry;
- the platforms OpenMM offers, and one real integration step on Reference, on CPU, and on CUDA when
  a CUDA platform is present.

A CUDA platform that is *listed but cannot take a step* is a failure, because that one fails at run
time instead. A machine with no GPU at all is not a failure — it is a supported machine, and you
are told that generated runs will need `MD_PLATFORM` naming another platform.

Plugins that cannot load for absent hardware are reported, not refused: conda-forge ships HIP
plugins that will never load on an NVIDIA box. A **CUDA** plugin failing to load is fatal.

Everything above lands in `machine.yaml` under `installed.openmm`, and the full command and output
in `$MD_STACK/logs/`.

## 4. Activate it and put the CLI inside it

The environment is an ordinary conda prefix:

```bash
conda activate $MD_STACK/envs/openmm-8.6.0
# or: micromamba activate $MD_STACK/envs/openmm-8.6.0

pip install --no-deps md-templates
# or, for development: pip install --no-deps -e /path/to/MD-templates
```

`--no-deps` is deliberate. The scientific stack is already there from conda; letting pip re-resolve
it pulls a second, pip-built OpenMM alongside the conda one, and which of the two you get then
depends on import order.

Confirm the CLI and the OpenMM it drives are the same Python:

```bash
python -c "import md_templates, openmm; print(md_templates.__file__, openmm.version.short_version)"
md-openmm --help
```

**Activate before running anything.** AM1-BCC goes through AmberTools' `sqm`, which OpenFF finds by
looking on `PATH`. In a bare shell it is not found even though it is installed,
`AmberToolsToolkitWrapper` is silently absent from the registry, and AM1-BCC becomes unavailable
under its own name.

## 5. Validate an environment you already have

To check an environment this command did not create — or to re-check one after a conda update:

```bash
md-template install --validate /path/to/env
```

Same probe, same record in `machine.yaml`. An environment built by hand is not less obliged to work.

## Then

```bash
md-openmm sys-config --method cMD REST2 --solvent OPC
```

and follow the root `README.md`.


## Checking that an environment is the exact release

```bash
md-template install --validate /path/to/env --expect-version 8.6.0
```

`--expect-version` is what turns validation into a **release** check. Without it nothing is
refused: an environment you built around a different OpenMM is reported, not judged.

**The obvious check is the wrong one.** OpenMM's own version string does not identify a release:

```text
conda package                : openmm 8.6.0 py312hdfcc665_0 from conda-forge   <- the release
openmm.version.version       : 8.6.0.dev-c6173db
openmm.version.short_version : 8.6.0
```

Upstream stamps the build commit into `version.version`, so `.dev` there does **not** mean a
development build — and `short_version` is `8.6.0` for the release and for any release candidate of
it. The check therefore judges on the **installed package identity** in `conda-meta/openmm-*.json`,
where a prerelease is a different package version. Both Python strings are still reported, and so
is their disagreement:

```text
  openmm        : 8.6.0.dev-c6173db (python 3.12.14)
  openmm package: 8.6.0 build py312hdfcc665_0 from conda-forge  -> release
  version note  : OpenMM reports '8.6.0.dev-c6173db' at runtime (short_version '8.6.0'); the
                  release identity comes from conda package metadata
  release check : 8.6.0 -> exact stable release
```

## What validation actually exercises

The probe imports every package the six commands need, checks that `sqm`, `antechamber` and `tleap`
are on `PATH`, confirms OpenFF has a registered AmberTools toolkit so standard AM1-BCC resolves,
lists the platforms, and takes integration steps on Reference, CPU and — where there is a GPU —
CUDA.

Those steps are taken on a System **with real forces** (a `NonbondedForce` and a
`HarmonicBondForce`, integrated with `LangevinMiddleIntegrator`), not on two free particles. A
mismatched CUDA build creates a context for free particles happily and only fails when a real force
kernel is loaded, which would otherwise be the first molecular system you build.
