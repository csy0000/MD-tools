# Installing Amber26, AmberTools26, and OpenMM 8.5.2

This guide prepares one Linux machine for the reproducibility comparison in this repository. It
installs:

- AmberTools26;
- licensed Amber26/`pmemd` components supplied by the user;
- OpenMM 8.5.2 in an isolated Python environment.

CUDA is preferred on a compatible NVIDIA GPU. CPU is the fallback. OpenCL is not selected
automatically; use it only when CUDA is unavailable and you intentionally choose it.

> **License and privacy:** Amber26/`pmemd` files are licensed material. Obtain them through the
> official Amber process. Do not commit, upload, redistribute, or paste their contents into issues
> or agent conversations. This repository does not contain those files.

The repository also contains an agent-readable version of this workflow at
`.claude/skills/install-md-stack/SKILL.md`. People installing manually can follow this README from
start to finish.

## What this guide does—and does not do

The guide separates three questions that are often confused:

1. Can OpenMM use an NVIDIA GPU? A working NVIDIA driver is normally the key requirement because
   packaged OpenMM builds can bring a compatible CUDA runtime.
2. Can Amber build `pmemd.cuda`? This additionally needs a CUDA toolkit with `nvcc`, a compatible
   host compiler, and a combination supported by Amber26.
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
```

It is normal for `nvcc --version` to fail on a machine where OpenMM CUDA can run from a packaged
runtime but Amber cannot yet build `pmemd.cuda`.

### Hardware decision

| Machine result | OpenMM | Amber26 |
| --- | --- | --- |
| NVIDIA GPU + working driver + supported toolkit/`nvcc` | CUDA | CUDA candidate; verify Amber26 compatibility |
| NVIDIA GPU + working driver, but no supported `nvcc` | CUDA may still work | CPU until a supported toolkit is installed |
| No working NVIDIA GPU | CPU | CPU |
| CUDA unavailable, OpenCL deliberately requested | Test OpenCL | CPU |

Do not replace an NVIDIA driver or system CUDA installation as part of this guide unless the
machine owner and administrator explicitly approve it. Driver changes can affect every GPU user on
the host.

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
  "$MD_STACK_PREFIX/build/amber26" \
  "$MD_STACK_PREFIX/amber26" \
  "$MD_STACK_PREFIX/envs" \
  "$MD_STACK_PREFIX/logs" \
  "$MD_STACK_PREFIX/manifests"
```

The intended result is:

```text
<prefix>/
  src/                    private extracted Amber sources
  build/amber26/          out-of-source build files
  amber26/                installed AMBERHOME
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
  capabilities;
- the release's install-prefix setting and CUDA enable/disable setting;
- the official build and test commands.

Archive layouts and option names can change. The bundled Amber26 instructions override every
generic example below.

### 6.2 Extract only after the safety check

Extract into a new subdirectory under `$MD_STACK_PREFIX/src`, following the order in the bundled
instructions. A typical single-archive extraction looks like:

```bash
mkdir -p "$MD_STACK_PREFIX/src/amber26-staging"
tar -xf "$AMBERTOOLS26_ARCHIVE" -C "$MD_STACK_PREFIX/src/amber26-staging"
```

Do not blindly unpack the licensed archive on top of that directory. First confirm the required
overlay path in its documentation. After extraction, define the actual source root:

```bash
export AMBER_SOURCE_ROOT=/absolute/path/to/the/extracted/amber26_source_root
```

### 6.3 Configure an out-of-source build

Amber releases commonly provide a `build/run_cmake` helper. Read it before running it. Configure it
so that:

- the install prefix is exactly `$MD_STACK_PREFIX/amber26`;
- the build directory is separate from the source directory;
- CUDA is enabled only for a toolkit/compiler pair supported by Amber26;
- CUDA is disabled explicitly for the accepted CPU fallback;
- optional components are chosen intentionally rather than inherited from another installation.

If the release documentation uses a generated `run_cmake` script, copy or generate it in
`$MD_STACK_PREFIX/build/amber26`, review the resulting CMake command, and save the full configure
output:

```bash
cd "$MD_STACK_PREFIX/build/amber26"
./run_cmake 2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-configure.log"
```

Do not run this example until `run_cmake` exists in the documented location and its prefix/CUDA
settings have been reviewed.

### 6.4 Compile, install, and run Amber tests

Choose parallelism conservatively. More CPU cores are not helpful if each compiler process exhausts
RAM. Substitute a suitable job count:

```bash
cmake --build "$MD_STACK_PREFIX/build/amber26" --parallel 4 \
  2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-build.log"

cmake --install "$MD_STACK_PREFIX/build/amber26" \
  2>&1 | tee "$MD_STACK_PREFIX/logs/amber26-install.log"
```

If Amber26's bundled instructions specify `make install` or another command instead, use that exact
command. Then source the installed activation file in the current shell:

```bash
export AMBERHOME="$MD_STACK_PREFIX/amber26"
source "$AMBERHOME/amber.sh"
```

Run the release-provided AmberTools, `pmemd`, and CUDA test targets. Capture each result in
`$MD_STACK_PREFIX/logs`. A compiler completing successfully is not enough: failed release tests must
be investigated before any scientific simulation.

## 7. Create an optional activation script

It is safer to create a standalone script than to edit `.bashrc` or `.zshrc`. Save the following as
`$MD_STACK_PREFIX/activate-md-stack.sh`, replacing the prefix with its real absolute value:

```bash
#!/usr/bin/env bash
export AMBERHOME=/absolute/path/to/md-stack/amber26
source "$AMBERHOME/amber.sh"
export PATH=/absolute/path/to/md-stack/envs/openmm-8.5.2/bin:"$PATH"
```

Activate it only when needed:

```bash
source "$MD_STACK_PREFIX/activate-md-stack.sh"
```

Do not add this to a shell startup file unless you intentionally want the MD stack active in every
new shell.

## 8. Validate the complete stack

The validator checks the required Amber executables, imports exactly OpenMM 8.5.2, lists OpenMM
platforms, runs OpenMM's self-test, and—when requested—requires both `pmemd.cuda` and OpenMM CUDA:

```bash
python3 .claude/skills/install-md-stack/scripts/verify_install.py \
  --amberhome "$MD_STACK_PREFIX/amber26" \
  --python "$MD_STACK_PREFIX/envs/openmm-8.5.2/bin/python" \
  --require-cuda \
  --output "$MD_STACK_PREFIX/manifests/validation.json"
```

For an explicitly accepted CPU installation, omit `--require-cuda`. For OpenCL, omit that flag and
separately verify that `openmm.testInstallation` reports the OpenCL platform; Amber remains a CPU
build unless its own documentation says otherwise.

Success requires all of the following:

- `tleap`, `sander`, `pmemd`, and `cpptraj` are installed and executable;
- `pmemd.cuda` exists when CUDA was selected for Amber;
- OpenMM reports version `8.5.2`;
- OpenMM reports the CUDA platform when CUDA was selected;
- `python -m openmm.testInstallation` passes;
- the appropriate Amber release tests pass.

## 9. Record installation provenance

Keep a small machine-readable manifest under `$MD_STACK_PREFIX/manifests`. Record:

- component versions and absolute installation paths;
- source archive filenames, sizes, and SHA-256 hashes;
- CUDA/CPU/OpenCL choice for each engine;
- GPU model, NVIDIA driver, toolkit/`nvcc`, compilers, CMake, and package-manager versions;
- OpenMM environment export or lock file;
- Amber configure command and selected build options;
- validation commands, timestamps, return codes, and log paths;
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
