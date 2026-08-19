# Installing Amber26, AmberTools26, and OpenMM 8.5.2

This guide prepares one Linux machine for the reproducibility comparison in this repository. It
installs:

- AmberTools26;
- licensed Amber26/`pmemd` components supplied by the user;
- `pmemd.cuda.MPI` for replica exchange and other multi-GPU Amber methods;
- OpenMM 8.5.2 in an isolated Python environment.

CUDA is preferred on a compatible NVIDIA GPU. CPU is the fallback. OpenCL is not selected
automatically; use it only when CUDA is unavailable and you intentionally choose it.

> **License and privacy:** Amber26/`pmemd` files are licensed material. Obtain them through the
> official Amber process. Do not commit, upload, redistribute, or paste their contents into issues
> or agent conversations. This repository does not contain those files.

The repository also contains an agent-readable version of this workflow at
`.claude/skills/install-md-stack/SKILL.md`. People installing manually can follow this README from
start to finish.

## Contents

- [What this guide does—and does not do](#what-this-guide-doesand-does-not-do)
- [1. Collect the inputs](#1-collect-the-inputs)
- [2. Inspect the machine before changing it](#2-inspect-the-machine-before-changing-it)
  - [Choosing a CUDA version](#choosing-a-cuda-version)
- [3. Inspect both archives safely](#3-inspect-both-archives-safely)
- [4. Create a clean directory layout](#4-create-a-clean-directory-layout)
- [5. Install OpenMM 8.5.2](#5-install-openmm-852)
- [6. Install AmberTools26 and licensed Amber26/pmemd](#6-install-ambertools26-and-licensed-amber26pmemd)
- [7. Create an optional activation script](#7-create-an-optional-activation-script)
- [8. Validate the complete stack](#8-validate-the-complete-stack)
- [9. Record installation provenance](#9-record-installation-provenance)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Updating or removing an installation](#11-updating-or-removing-an-installation)
- [12. Next step](#12-next-step)
- [Official OpenMM references](#official-openmm-references)

## What this guide does—and does not do

The guide separates three questions that are often confused:

1. Can OpenMM use an NVIDIA GPU? A working NVIDIA driver is normally the key requirement because
   packaged OpenMM builds can bring a compatible CUDA runtime.
2. Can Amber build `pmemd.cuda` and `pmemd.cuda.MPI`? This additionally needs a CUDA toolkit with
   `nvcc`, a compatible host compiler, a compatible MPI implementation/compiler wrappers, and a
   combination supported by Amber26.
3. Can both programs run on the CPU? This needs the normal C/C++/Fortran build toolchain for Amber;
   OpenMM has a CPU platform.

The exact Amber26 CMake switches and supported CUDA/compiler combinations are version-sensitive.
Therefore, the README/INSTALL files shipped in your Amber26 and AmberTools26 archives are
authoritative. This guide shows the safe workflow around those instructions instead of guessing
release-specific flags.

## 1. Collect the inputs

Before starting, have these paths ready:

- an absolute, user-owned installation prefix, such as `/data/software/md-stack`;
- the AmberTools26 source archive or extracted source directory;
- the licensed Amber26/`pmemd` tar archive;
- a package manager: `micromamba`, `mamba`, `conda`, or Python `venv`;
- for an Amber CUDA build, the intended CUDA toolkit path if several versions are installed.
- for multi-GPU Amber, the intended MPI implementation and launcher (`mpirun`, `mpiexec`, or a
  scheduler launcher), single-node versus multi-node scope, and GPUs per node.

Do not use `/`, `/usr`, `/usr/local`, your home directory itself, or this Git repository as the
installation prefix. Do not reuse a nonempty directory belonging to unrelated software.

In the command examples, substitute your real absolute paths:

```bash
export MD_STACK_PREFIX=/absolute/path/to/md-stack
export AMBERTOOLS26_ARCHIVE=/absolute/path/to/AmberTools26-archive.tar.bz2
export AMBER26_ARCHIVE=/absolute/path/to/licensed-Amber26-or-pmemd-archive.tar.bz2
```

These variables last only for the current shell. Quote them as shown so paths containing spaces do
not split into multiple arguments.

## 2. Inspect the machine before changing it

Run the repository's read-only inspector:

```bash
python3 .claude/skills/install-md-stack/scripts/inspect_machine.py \
  --install-dir "$MD_STACK_PREFIX" \
  --output machine-report.json
```

The report records OS, architecture, logical CPUs, RAM, free disk, compilers, CMake/build tools,
package managers, NVIDIA GPUs, the driver, `nvcc`, and possible CUDA toolkit roots. It does not dump
the environment or create the requested installation directory.

You can also check the most important items directly:

```bash
uname -a
getconf _NPROCESSORS_ONLN
free -h
df -h "$MD_STACK_PREFIX"
gcc --version
g++ --version
gfortran --version
cmake --version
nvidia-smi
nvcc --version
mpicc --version
mpifort --version
mpirun --version
srun --version
```

It is normal for `nvcc --version` to fail on a machine where OpenMM CUDA can run from a packaged
runtime but Amber cannot yet build `pmemd.cuda`. It is also possible to have working single-GPU
`pmemd.cuda` while `pmemd.cuda.MPI` is unavailable because a compatible MPI toolchain was not used.

### Hardware decision

| Machine result | OpenMM | Amber26 |
| --- | --- | --- |
| NVIDIA GPU + working driver + supported toolkit/`nvcc` + compatible MPI | CUDA | Build and test `pmemd.cuda` and `pmemd.cuda.MPI` |
| NVIDIA GPU + toolkit, but MPI missing/incompatible | CUDA | Single-GPU candidate only; CUDA+MPI not ready |
| NVIDIA GPU + working driver, but no supported `nvcc` | CUDA may still work | CPU until a supported toolkit is installed |
| No working NVIDIA GPU | CPU | CPU |
| CUDA unavailable, OpenCL deliberately requested | Test OpenCL | CPU |

### Choosing a CUDA version

A machine that will eventually host Amber, OpenMM, GROMACS, PyTorch, and PyTorch Geometric does not
need one CUDA version. It needs one *rule*.

**Only software you compile needs a CUDA toolkit.** In this stack that is Amber/`pmemd` alone.
OpenMM, GROMACS, PyTorch, and PyTorch Geometric are installed as prebuilt binaries that carry their
own CUDA runtime inside their environment; they need a sufficiently new NVIDIA **driver** and nothing
else. Conda environments are isolated, so one environment can hold CUDA 12.x while another holds
13.x without interfering.

The practical consequence: **pin `cuda-version` explicitly in every environment, and do not install a
CUDA toolkit system-wide.** A single global toolkit forces the oldest consumer to hold back the
newest — with Amber below CUDA 12.9 and current PyTorch builds on CUDA 13, one global choice must
break something.

**The compiled component sets its own limit, and Amber states it in code.** Amber26's
`cmake/CudaConfig.cmake` fails outright outside a supported range:

```
FATAL_ERROR "Error: Untested CUDA version.
             AMBER currently requires CUDA version >= 7.5 and < 12.9."
```

The same file gates the CUDA/host-compiler pairing, including an explicit special case:

```cmake
OR ( CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 13.3
     AND CUDA_VERSION VERSION_EQUAL 12.6 )
     # 13.3 and 12.6 is a special case where stackoverflow and
     # nvidia disagree; allow based on Gerald Monard's testing.
```

So on a host with **gcc 13.3**, **CUDA 12.6** is the combination Amber's authors tested, not a
compromise. Read `CudaConfig.cmake` from the release you actually have rather than reusing this
number: the ceiling moves between releases, and a newer Amber will accept newer toolkits.

A worked example for a mixed machine:

| environment | CUDA | reason |
| --- | --- | --- |
| Amber build | 12.6 **toolkit** (`nvcc`) | the version whitelisted for the host compiler |
| OpenMM | 12.9 or 13.x **runtime** | both build families exist; choosing near Amber keeps caches shared |
| GROMACS | whatever its CUDA build requires | prebuilt; driver is the only shared requirement |
| PyTorch + PyTorch Geometric | 13.x **runtime** | current builds; PyG follows PyTorch exactly |

Do not install PyTorch and OpenMM into the same environment. They constrain `cuda-version`
differently, and the solver will resolve the conflict by silently downgrading one of them. Keep them
in separate environments and join them through files on disk, not a shared runtime.

Do not replace an NVIDIA driver, system CUDA installation, or system MPI stack as part of this guide
unless the machine owner and administrator explicitly approve it. These changes can affect every
GPU or cluster user on the host.

## 3. Inspect both archives safely

The archive inspector hashes each tar file, checks for path traversal, unsafe links, device nodes,
and FIFO entries, and reports only installation-relevant filenames:

```bash
python3 .claude/skills/install-md-stack/scripts/inspect_amber_archive.py \
  "$AMBERTOOLS26_ARCHIVE" \
  --expected ambertools26 \
  --output ambertools26-archive-report.json

python3 .claude/skills/install-md-stack/scripts/inspect_amber_archive.py \
  "$AMBER26_ARCHIVE" \
  --expected amber26-or-pmemd \
  --output amber26-archive-report.json
```

Stop if either command reports `"ok": false`. An unfamiliar archive name can make automatic
classification inconclusive; in that case, compare its top-level layout with the official download
description and bundled install documents. Never extract an archive that reports unsafe members.

Save the SHA-256 values. They identify exactly which private inputs were used without exposing
their contents.

## 4. Create a clean directory layout

After checking that the prefix is correct and unused, create distinct source, build, install,
environment, log, and manifest directories:

```bash
mkdir -p \
  "$MD_STACK_PREFIX/src" \
  "$MD_STACK_PREFIX/build/ambertools26" \
  "$MD_STACK_PREFIX/build/pmemd26" \
  "$MD_STACK_PREFIX/ambertools26" \
  "$MD_STACK_PREFIX/pmemd26" \
  "$MD_STACK_PREFIX/envs" \
  "$MD_STACK_PREFIX/logs" \
  "$MD_STACK_PREFIX/manifests"
```

The intended result is:

```text
<prefix>/
  src/                    private extracted Amber sources
  build/ambertools26/     out-of-source AmberTools build files
  build/pmemd26/          out-of-source licensed PMEMD build files
  ambertools26/           installed AMBERHOME
  pmemd26/                installed PMEMDHOME
  envs/openmm-8.5.2/      isolated OpenMM environment
  logs/                   build and test logs
  manifests/              installation provenance
  activate-md-stack.sh    optional activation helper
```

Keep all build products outside the MD-templates repository.

## 5. Install OpenMM 8.5.2

Use one of the following routes, not both. The conda-compatible route is generally easier to
reproduce. OpenMM's official installation guide documents both conda and pip installation and the
CUDA package choices.

### Option A: micromamba, mamba, or conda

The CPU-capable baseline environment is:

```bash
micromamba create -p "$MD_STACK_PREFIX/envs/openmm-8.5.2" \
  -c conda-forge \
  python=3.11 openmm=8.5.2
```

Replace `micromamba` with `mamba` or `conda` if that is your selected manager. For a CUDA-targeted
environment, add a compatible constraint such as `cuda-version=12` only after checking that the
detected NVIDIA driver supports it:

```bash
micromamba create -p "$MD_STACK_PREFIX/envs/openmm-8.5.2" \
  -c conda-forge \
  python=3.11 openmm=8.5.2 cuda-version=12
```

Treat `cuda-version=12` as an example family, not a promise that every driver supports it. Review
the solver's proposed packages before confirming.

Activate and test:

```bash
micromamba activate "$MD_STACK_PREFIX/envs/openmm-8.5.2"
python -c 'import openmm; print(openmm.__version__)'
python -m openmm.testInstallation
```

If your shell is not initialized for activation, use:

```bash
micromamba run -p "$MD_STACK_PREFIX/envs/openmm-8.5.2" \
  python -m openmm.testInstallation
```

### Option B: Python `venv` and pip

For an NVIDIA/CUDA installation compatible with the CUDA 12 package family:

```bash
python3 -m venv "$MD_STACK_PREFIX/envs/openmm-8.5.2"
"$MD_STACK_PREFIX/envs/openmm-8.5.2/bin/python" -m pip install --upgrade pip
"$MD_STACK_PREFIX/envs/openmm-8.5.2/bin/python" -m pip install \
  'openmm[cuda12]==8.5.2'
```

OpenMM also documents a CUDA 13 extra. Choose it only when the installed driver is compatible. For
a CPU installation use:

```bash
"$MD_STACK_PREFIX/envs/openmm-8.5.2/bin/python" -m pip install 'openmm==8.5.2'
```

Test either pip installation:

```bash
"$MD_STACK_PREFIX/envs/openmm-8.5.2/bin/python" \
  -m openmm.testInstallation
```

For the intended CUDA path, the test output must list the CUDA platform. An import that exposes only
CPU or Reference is not a successful CUDA installation. Do not silently substitute OpenCL.

## 6. Install AmberTools26 and licensed Amber26/pmemd

### 6.1 Read the bundled release instructions

Before extraction or overlay, locate the README/INSTALL filenames listed by the archive reports.
Read those documents and confirm:

- the required extraction order;
- whether the licensed Amber26/`pmemd` archive overlays the AmberTools26 source tree;
- the supported operating systems, compilers, CMake version, Python, CUDA toolkits, and GPU compute
  capabilities, plus supported MPI implementations and compiler wrappers;
- the release's install-prefix, CUDA, and MPI enable/disable settings;
- the official build and test commands.

Archive layouts and option names can change. The bundled Amber26 instructions override every
generic example below.

### 6.2 Extract only after the safety check

Extract into a new subdirectory under `$MD_STACK_PREFIX/src`, following the order in the bundled
instructions. A typical single-archive extraction looks like:

```bash
mkdir -p "$MD_STACK_PREFIX/src/ambertools26-staging"
tar -xf "$AMBERTOOLS26_ARCHIVE" -C "$MD_STACK_PREFIX/src/ambertools26-staging"
```

Do not blindly unpack the licensed archive on top of that directory. First confirm the required
overlay path in its documentation. After extraction, define the actual source root:

```bash
export AMBERTOOLS_SOURCE_ROOT=/absolute/path/to/the/extracted/ambertools26_source_root
export PMEMD_SOURCE_ROOT=/absolute/path/to/the/extracted/pmemd26_source_root
```

### 6.3 Configure an out-of-source build

Amber releases commonly provide a `build/run_cmake` helper. Read it before running it. Configure it
so that:

- the AmberTools install prefix is exactly `$MD_STACK_PREFIX/ambertools26` and the licensed PMEMD
  prefix is exactly `$MD_STACK_PREFIX/pmemd26`, unless the release explicitly documents a combined
  install;
- the build directory is separate from the source directory;
- CUDA and MPI are enabled only for a toolkit/compiler/MPI combination supported by Amber26;
- the intended MPI compiler wrappers are selected so both `pmemd.cuda` and `pmemd.cuda.MPI` are
  produced without mixing incompatible MPI libraries;
- CUDA is disabled explicitly for the accepted CPU fallback;
- optional components are chosen intentionally rather than inherited from another installation.

If the release documentation uses generated `run_cmake` scripts, configure and review AmberTools
and licensed PMEMD separately, saving both complete configure outputs:

```bash
cd "$MD_STACK_PREFIX/build/ambertools26"
./run_cmake 2>&1 | tee "$MD_STACK_PREFIX/logs/ambertools26-configure.log"

cd "$MD_STACK_PREFIX/build/pmemd26"
./run_cmake 2>&1 | tee "$MD_STACK_PREFIX/logs/pmemd26-configure.log"
```

Do not run this example until `run_cmake` exists in the documented location and its
prefix/CUDA/MPI settings have been reviewed.

### 6.4 Compile, install, and run Amber tests

Choose parallelism conservatively. More CPU cores are not helpful if each compiler process exhausts
RAM. Substitute a suitable job count:

```bash
cmake --build "$MD_STACK_PREFIX/build/ambertools26" --parallel 4 \
  2>&1 | tee "$MD_STACK_PREFIX/logs/ambertools26-build.log"
cmake --install "$MD_STACK_PREFIX/build/ambertools26" \
  2>&1 | tee "$MD_STACK_PREFIX/logs/ambertools26-install.log"

cmake --build "$MD_STACK_PREFIX/build/pmemd26" --parallel 4 \
  2>&1 | tee "$MD_STACK_PREFIX/logs/pmemd26-build.log"
cmake --install "$MD_STACK_PREFIX/build/pmemd26" \
  2>&1 | tee "$MD_STACK_PREFIX/logs/pmemd26-install.log"
```

If Amber26's bundled instructions specify `make install` or another command instead, use that exact
command. Then source the installed activation file in the current shell:

```bash
export AMBERHOME="$MD_STACK_PREFIX/ambertools26"
source "$AMBERHOME/amber.sh"
export PMEMDHOME="$MD_STACK_PREFIX/pmemd26"
source "$PMEMDHOME/amber.sh"
```

The Amber26 manual uses `DO_PARALLEL` to select the launcher and rank count. Run `test.parallel` in
each MPI-enabled installed tree. It recommends repeating with four or eight ranks because some
replica-exchange tests need more than two:

```bash
cd "$AMBERHOME"
export DO_PARALLEL="mpirun -np 2"
make test.parallel 2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-test-parallel-np2.log"

export DO_PARALLEL="mpirun -np 4"
make test.parallel 2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-test-parallel-np4.log"
```

If the licensed PMEMD installation has a separate `$PMEMDHOME`, repeat the documented MPI target
there. To test `pmemd.cuda.MPI` specifically, expose one GPU per MPI rank and run the dedicated CUDA
parallel targets from the PMEMD installation/test tree:

```bash
cd "${PMEMDHOME:-$AMBERHOME}"
export CUDA_VISIBLE_DEVICES=0,1
export DO_PARALLEL="mpirun -np 2"
make test.cuda.parallel \
  2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-test-cuda-parallel.log"
make test.cuda.parallel.SPFP \
  2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-test-cuda-parallel-spfp.log"
```

Substitute the documented launcher or scheduler command and visible GPU IDs for the actual machine.
Do not request more CUDA ranks than visible GPUs. Inspect the diff files identified by the Amber26
manual: small floating-point differences can occur, whereas large or unexplained differences need
investigation. A compiler completing successfully—or a one-rank MPI launch—is not enough.

## 7. Create an optional activation script

It is safer to create a standalone script than to edit `.bashrc` or `.zshrc`. Save the following as
`$MD_STACK_PREFIX/activate-md-stack.sh`, replacing the prefix with its real absolute value:

```bash
#!/usr/bin/env bash
export AMBERHOME=/absolute/path/to/md-stack/ambertools26
source "$AMBERHOME/amber.sh"
export PMEMDHOME=/absolute/path/to/md-stack/pmemd26
source "$PMEMDHOME/amber.sh"

# Keep the two Python environments apart. AmberTools' amber.sh exports PYTHONPATH
# to its own site-packages. Left set, that PYTHONPATH is inherited by ANY python
# later on PATH -- including OpenMM's, which is commonly the same Python version.
# AmberTools' own Python tools use absolute shebangs and keep working without it.
unset PYTHONPATH

# Expose OpenMM explicitly rather than putting its bin on PATH, so exactly one
# `python` is in scope instead of two whose precedence depends on source order.
export MD_OPENMM_PYTHON=/absolute/path/to/md-stack/envs/openmm-8.5.2/bin/python

# Stable GPU numbering. Without this, CUDA device indices are ordered by compute
# capability and can move between reboots -- which matters on mixed-GPU machines.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

If you would rather have `import parmed` work from a bare `python`, drop the `unset PYTHONPATH`
and accept that OpenMM's interpreter inherits AmberTools' site-packages.

Activate it only when needed:

```bash
source "$MD_STACK_PREFIX/activate-md-stack.sh"
```

Do not add this to a shell startup file unless you intentionally want the MD stack active in every
new shell.

## 8. Validate the complete stack

The validator checks the required Amber executables, imports exactly OpenMM 8.5.2, lists OpenMM
platforms, runs OpenMM's self-test, and—when requested—requires `pmemd.cuda`, `pmemd.cuda.MPI`, an
MPI/scheduler launcher, and OpenMM CUDA:

```bash
python3 .claude/skills/install-md-stack/scripts/verify_install.py \
  --amberhome "$MD_STACK_PREFIX/ambertools26" \
  --pmemdhome "$MD_STACK_PREFIX/pmemd26" \
  --python "$MD_STACK_PREFIX/envs/openmm-8.5.2/bin/python" \
  --require-cuda-mpi \
  --mpi-launcher mpirun \
  --output "$MD_STACK_PREFIX/manifests/validation.json"
```

`--require-cuda-mpi` implies `--require-cuda`. Substitute `mpiexec` or the documented scheduler
launcher where appropriate. For an explicitly accepted CPU installation, omit both requirements.
For OpenCL, separately verify that `openmm.testInstallation` reports the OpenCL platform; Amber
remains a CPU build unless its own documentation says otherwise.

Success requires all of the following:

- `tleap`, `sander`, `pmemd`, and `cpptraj` are installed and executable;
- `pmemd.cuda` exists when CUDA was selected for Amber;
- `pmemd.cuda.MPI` and the intended launcher exist for replica exchange/multi-GPU Amber;
- `make test.parallel` passes with the rank counts needed for replica-exchange coverage;
- `make test.cuda.parallel` and `make test.cuda.parallel.SPFP` pass with one rank per visible GPU;
- OpenMM reports version `8.5.2`;
- OpenMM reports the CUDA platform when CUDA was selected;
- `python -m openmm.testInstallation` passes;
- the appropriate Amber release tests pass.

## 9. Record installation provenance

Keep a small machine-readable manifest under `$MD_STACK_PREFIX/manifests`. Record:

- component versions and absolute installation paths;
- source archive filenames, sizes, and SHA-256 hashes;
- CUDA/CPU/OpenCL choice for each engine;
- GPU model, NVIDIA driver, toolkit/`nvcc`, compilers, CMake, MPI implementation/compiler wrappers,
  launcher/scheduler, and package-manager versions;
- OpenMM environment export or lock file;
- Amber configure command and selected build options;
- validation commands, timestamps, return codes, and log paths;
- `DO_PARALLEL`, rank counts, `CUDA_VISIBLE_DEVICES`, and results/diff-log paths for
  `test.parallel`, `test.cuda.parallel`, and `test.cuda.parallel.SPFP`;
- warnings and any accepted CPU fallback.

Do not record usernames, tokens, license data, full archive listings, licensed contents, or a dump
of the entire process environment.

For a conda-compatible environment, also export an explicit package record:

```bash
micromamba list -p "$MD_STACK_PREFIX/envs/openmm-8.5.2" --explicit \
  > "$MD_STACK_PREFIX/manifests/openmm-explicit.txt"
```

## 10. Troubleshooting

### `nvidia-smi` does not work

The NVIDIA driver is missing, inaccessible, or unhealthy. Use CPU temporarily or ask the machine
administrator to repair the driver. Do not install a driver blindly from this repository.

### OpenMM imports but CUDA is missing

Check the installed package variant, driver/runtime compatibility, and the diagnostics printed by
`python -m openmm.testInstallation`. A working local `nvcc` is not proof that OpenMM loaded its CUDA
platform. Do not report success based only on `import openmm`.

### OpenMM reports unsupported PTX or a driver/runtime error

The selected CUDA runtime family may be newer than the NVIDIA driver supports, or the GPU may not
be supported by that package. Select a compatible package/runtime or update the driver through the
machine's normal administrative process.

### `pmemd.cuda` was not built

Confirm that `nvcc` was available during Amber configuration, the toolkit and host compiler are
supported by Amber26, CUDA was enabled in the recorded CMake command, and the licensed `pmemd`
source was overlaid exactly as documented. OpenMM CUDA working does not prove that Amber CUDA was
buildable.

### `pmemd.cuda.MPI` is missing or its tests fail

Confirm that Amber26 was configured with both MPI and CUDA enabled, that the intended MPI compiler
wrappers were selected, and that the run-time launcher uses the same compatible MPI implementation.
Run `make test.parallel` first, then `make test.cuda.parallel` and
`make test.cuda.parallel.SPFP`. Check `CUDA_VISIBLE_DEVICES`, use one rank per GPU, and inspect the
manual's saved diff files before deciding whether a numerical difference is significant.

### Amber rejects the CUDA or compiler version

Use a combination listed by Amber26's bundled installation notes. A CPU build is safer than
disabling compatibility checks without evidence. Record any fallback explicitly.

### `gfortran`, CMake, or another compiler is missing

Install the prerequisite using the operating system's normal package-management process. On a
shared machine, ask the administrator; do not use `sudo` without authorization.

### Archive classification is inconclusive

Compare the archive's top-level layout and installation-document filenames with the official Amber
download description. A filename alone is not proof. Stop if the release or overlay relationship is
still ambiguous.

### Amber tests fail

Preserve the configure, build, and test logs. Identify the first real failure rather than only the
final summary. Do not begin the alanine-dipeptide comparison until the relevant test suites pass.

### Disk or memory is insufficient

Choose a different prefix with more free space or reduce build parallelism. Do not place temporary
build products in the Git repository.

## 11. Updating or removing an installation

Install a new version into a new versioned prefix or environment. Do not overwrite a working
Amber/OpenMM installation in place. To remove an installation, first review the exact directories
under the selected prefix and archive the manifests/logs you need; remove only those confirmed
version-specific directories. Never use a recursive deletion command with an unset variable, a
home directory, `/`, or a broad shared prefix.

## 12. Next step

Only after the full validator and the release-provided Amber tests pass should this repository move
to the alanine-dipeptide equivalence test:

- ff19SB + OPC;
- truncated-octahedral box with at least 12 Å solute padding;
- 1000 minimization steps with 1 kcal mol⁻¹ Å⁻² solute positional restraints;
- 10 ps NVT with the same restraints;
- 10 ps NPT with the same restraints;
- 1 ns NPT production;
- matched Amber26 and OpenMM 8.5.2 settings and recorded provenance.

Installation validation proves that both engines run; the alanine test then checks whether the two
templates describe the same scientific protocol.

## Official OpenMM references

- [OpenMM installation guide](https://docs.openmm.org/development/userguide/application/01_getting_started.html)
- [OpenMM releases](https://github.com/openmm/openmm/releases)

For Amber26, use the official release documentation supplied with your licensed archives and the
official Amber website/download portal associated with those files.
