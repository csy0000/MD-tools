---
name: install-md-stack
description: Inspect a Linux workstation and safely install or validate Amber26, AmberTools26, and OpenMM 8.5.2, including pmemd.cuda.MPI for replica exchange and multi-GPU Amber workflows. Use when a user asks to prepare an MD machine, install Amber/OpenMM, inspect CPU/RAM/disk/NVIDIA/CUDA/MPI resources, choose an installation prefix, validate user-supplied Amber or pmemd archives, prefer CUDA over OpenCL, or diagnose an incomplete MD software installation.
---

# Install the MD stack

Install only after producing a read-only machine report and obtaining the user's paths and approval.

## Required interaction

1. Run `scripts/inspect_machine.py --install-dir <candidate>` before proposing commands.
2. Summarize CPU, RAM, free disk, OS, compilers, NVIDIA GPUs, driver, `nvcc`, MPI compilers,
   launchers/scheduler commands, and package managers.
3. Ask for:
   - an absolute installation prefix owned by the user;
   - the AmberTools26 source archive or extracted source directory;
   - the user-provided licensed Amber26/pmemd archive;
   - whether the release documentation specifies separate AmberTools and PMEMD install prefixes;
   - preferred environment manager (`micromamba`, `mamba`, `conda`, or `venv`);
   - permission to modify shell startup files, defaulting to **no**;
   - a CUDA toolkit path when more than one is present;
   - the intended MPI implementation and launcher (`mpirun`, `mpiexec`, or scheduler launcher),
     whether runs are single-node or multi-node, and the number of GPUs per node.
4. Inspect each archive with `scripts/inspect_amber_archive.py`. Never upload, redistribute, commit, or reveal archive contents beyond installation-relevant filenames and the archive hash.
5. Present the exact installation plan, target directories, expected downloads, and validation commands. Obtain confirmation before extraction, environment creation, compilation, or shell changes.

## Backend selection

Use this order:

1. **CUDA** when a compatible NVIDIA GPU and driver are present.
2. **CPU** when CUDA is unavailable or Amber cannot be built against the installed toolkit.
3. **OpenCL** for OpenMM only when CUDA is unavailable and the user explicitly selects it. Never silently choose OpenCL over working CUDA.

### Choosing a CUDA version

Do not pick one CUDA version for the whole machine. Only software you **compile** needs a CUDA
toolkit; in this stack that is Amber/`pmemd` alone. OpenMM, GROMACS, PyTorch, and PyTorch Geometric
ship prebuilt binaries carrying their own CUDA runtime and need only a sufficiently new driver.
Conda environments are isolated, so pin `cuda-version` per environment and never install a toolkit
system-wide — one global toolkit forces the oldest consumer to hold back the newest.

Take the compiled component's limit from its own source, not from a general recommendation. Amber26
declares it in `cmake/CudaConfig.cmake`, which fails with `FATAL_ERROR "Untested CUDA version. AMBER
currently requires CUDA version >= 7.5 and < 12.9."` and separately gates the CUDA/host-compiler
pairing — including an explicit special case allowing **gcc 13.3 with CUDA 12.6**. Read that file
from the release actually being installed; the ceiling moves between releases.

Never place PyTorch and OpenMM in one environment. They constrain `cuda-version` differently and the
solver resolves the conflict by silently downgrading one of them.

An NVIDIA driver can be sufficient for a packaged OpenMM CUDA runtime, but building `pmemd.cuda`
also requires a compatible CUDA toolkit and compiler. Building and running `pmemd.cuda.MPI` additionally
requires the MPI compiler wrappers, runtime/launcher, and scheduler integration documented for Amber26.
Report OpenMM CUDA, Amber single-GPU CUDA, and Amber CUDA+MPI readiness separately. Never install or
replace a system GPU driver or system MPI stack without explicit user and administrator approval.

## Safety rules

- Treat Amber/pmemd archives as licensed private inputs. Do not download them on the user's behalf unless the official authenticated process is explicitly requested and supported; never place them in Git.
- Never use `sudo`, overwrite an existing Amber installation, edit `/usr/local`, or modify `.bashrc`/`.zshrc` without explicit confirmation.
- Use a dedicated source directory, build directory, installation prefix, and OpenMM environment. Do not build inside the repository.
- Refuse an installation prefix that is `/`, a home directory itself, the repository root, or a nonempty unrelated directory.
- Validate archive paths before extraction; reject absolute paths, `..` traversal, device nodes, and archive layouts that do not match the bundled installation documentation.
- Prefer reversible activation scripts over persistent shell changes.
- Never delete an existing installation. On failure, preserve logs and report the exact failed command.
- Read the installation documents shipped inside the Amber26/AmberTools26 archives. Those version-specific documents override generic command examples in `references/installation-reference.md`.

## Installation workflow

### 1. Inspect and decide

Run the machine and archive inspectors. Read `references/installation-reference.md` for the decision matrix and command patterns. Distinguish:

- OpenMM CUDA readiness: NVIDIA driver plus a compatible packaged CUDA runtime;
- Amber CUDA readiness: NVIDIA driver plus supported `nvcc`/toolkit and Amber build support;
- Amber CUDA+MPI readiness: Amber CUDA readiness plus a compatible MPI implementation, compiler
  wrappers, launcher/scheduler environment, and enough visible GPUs for the intended rank layout;
- CPU-only readiness: C/C++/Fortran compilers, CMake, build tool, Python, and sufficient disk.

### 2. Install OpenMM 8.5.2

Create an isolated environment under the chosen prefix. Pin `openmm==8.5.2`. Prefer the official CUDA package path compatible with the detected driver; do not install a newer CUDA runtime than the driver can support. Install the CPU/OpenCL-only package only when CUDA is not selected.

Verify with:

```bash
python -c 'import openmm; print(openmm.__version__)'
python -m openmm.testInstallation
```

Require the CUDA platform when CUDA was selected. Record platform names and versions.

### 3. Install AmberTools26 and Amber26/pmemd

Validate and hash both user-provided archives. Extract into a new staging directory using the archive layout and instructions shipped with that release. Confirm whether the Amber26/pmemd archive overlays the AmberTools source tree before doing so; never guess.

Configure separate build directories and use the Amber26-documented install layout. Support distinct
`AMBERHOME` and `PMEMDHOME` prefixes as well as a documented combined installation. Enable CUDA and MPI
using the exact Amber26-documented options so the installation produces both `pmemd.cuda` and
`pmemd.cuda.MPI`. Confirm that Amber uses the intended MPI compiler wrappers and implementation;
do not mix incompatible build-time and run-time MPI libraries. Build with a conservative parallel
job count based on RAM and CPU. Capture configure, build, install, and test logs.

Source the generated `amber.sh` only in the current shell during validation. Create a standalone activation script after success; edit shell startup files only if the user opted in.

### 4. Validate

Run the release-provided Amber tests appropriate to the installed components. Confirm at least
`tleap`, `sander`, `pmemd`, and `cpptraj`; require both `pmemd.cuda` and `pmemd.cuda.MPI` when
CUDA/multi-GPU support was selected. Following the Amber26 manual, run `make test.parallel` with
`DO_PARALLEL` first at two ranks and, when resources permit, at four or eight ranks for the
replica-exchange tests. Test `pmemd.cuda.MPI` specifically with `make test.cuda.parallel` and the
production SPFP build with `make test.cuda.parallel.SPFP`, using one MPI rank per visible GPU.
Preserve the logs and inspect the documented diff files rather than treating every small numerical
difference as a build failure. Then run `scripts/verify_install.py` with `--require-cuda-mpi`, the
exact AmberTools and PMEMD prefixes, and the OpenMM Python.

Write a machine-readable installation manifest containing versions, prefix paths, archive SHA-256 values, selected backend, compiler/CUDA versions, validation commands, and results. Do not include usernames, secrets, licensed content, or broad environment dumps.

## Completion report

Report:

- installed and missing components;
- exact activation commands;
- CUDA/CPU/OpenCL status for each engine, plus Amber CUDA+MPI status and launcher;
- validation results and log locations;
- unresolved warnings;
- the installation-manifest path.

Do not begin a scientific simulation until both engines pass their installation checks.
