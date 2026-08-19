# Installation reference

Use this reference for planning and command patterns. The installation documents shipped in the
user's Amber26 and AmberTools26 archives are authoritative for release-specific prerequisites,
archive layout, CMake options, CUDA compatibility, and test targets.

## Contents

- [Information to collect](#information-to-collect)
- [Suggested layout](#suggested-layout)
- [Backend decision matrix](#backend-decision-matrix)
- [OpenMM 8.5.2 patterns](#openmm-852-patterns)
- [Amber archive and build pattern](#amber-archive-and-build-pattern)
- [Activation pattern](#activation-pattern)
- [Validation and manifest](#validation-and-manifest)
- [Common failure interpretation](#common-failure-interpretation)

## Information to collect

- Absolute, user-owned installation prefix.
- AmberTools26 source archive or already-extracted source directory.
- Licensed Amber26/pmemd archive path.
- Preferred environment manager: micromamba, mamba, conda, or Python `venv`.
- Whether shell startup files may be edited. Default: no.
- Preferred CUDA toolkit path if multiple `nvcc` installations are present.
- Intended MPI implementation and launcher, scheduler (if any), single-node versus multi-node
  scope, GPUs per node, and the expected MPI-rank-to-GPU layout.
- Whether a CPU fallback is acceptable if Amber26 cannot use the detected CUDA toolkit.

Do not ask the user to paste archive contents, license data, credentials, or environment dumps.

## Suggested layout

For a selected prefix named `<PREFIX>`, propose distinct directories:

```text
<PREFIX>/
  src/                  # private extracted Amber sources
  build/ambertools26/   # out-of-source AmberTools build
  build/pmemd26/        # out-of-source licensed PMEMD build
  ambertools26/         # installed AMBERHOME
  pmemd26/              # installed PMEMDHOME; omit only for a documented combined install
  envs/openmm-8.5.2/    # isolated OpenMM environment
  logs/                 # configure, build, test, and validation logs
  manifests/            # redacted machine-readable provenance
  activate-md-stack.sh  # optional activation script
```

Reject `/`, `/usr`, `/usr/local`, a home directory itself, the repository root, and unrelated
nonempty directories. Existing installations must receive a new prefix unless the user explicitly
asks only for validation.

## Backend decision matrix

| Observation | OpenMM recommendation | Amber recommendation |
| --- | --- | --- |
| NVIDIA GPU and working driver; compatible toolkit and MPI toolchain present | CUDA | Build both `pmemd.cuda` and `pmemd.cuda.MPI`; run Amber26's parallel CUDA tests |
| NVIDIA GPU and working driver; compatible toolkit present but MPI missing | CUDA | Build `pmemd.cuda`; install/choose compatible MPI before claiming multi-GPU readiness |
| NVIDIA GPU and working driver; no `nvcc` | CUDA package may still work because it can supply its own runtime | CPU build until a supported toolkit is provided |
| No usable NVIDIA GPU | CPU | CPU |
| CUDA unavailable and user explicitly requests OpenCL | OpenCL after testing | CPU; do not imply that OpenCL provides `pmemd.cuda` |

Do not infer Amber CUDA build readiness merely from the presence of `nvidia-smi`. Packaged OpenMM
and source-built Amber have different CUDA prerequisites. Likewise, the existence of `mpirun` alone
does not prove that `pmemd.cuda.MPI` was built against a compatible MPI library.

## OpenMM 8.5.2 patterns

Prefer an isolated prefix environment and pin the version exactly. Choose only one package-manager
route.

### Conda-compatible environment

Use micromamba, mamba, or conda according to the user's choice. For example:

```bash
micromamba create -p <PREFIX>/envs/openmm-8.5.2 -c conda-forge \
  python=3.11 openmm=8.5.2
micromamba run -p <PREFIX>/envs/openmm-8.5.2 \
  python -m openmm.testInstallation
```

If a CUDA constraint is needed, select a `cuda-version` supported by both the chosen package and
the detected NVIDIA driver. Resolve the environment before installing and show the package plan.
Do not assume that the locally installed `nvcc` version must equal the environment runtime.

### Python virtual environment

Use this only when the user chooses pip/venv:

```bash
python3 -m venv <PREFIX>/envs/openmm-8.5.2
<PREFIX>/envs/openmm-8.5.2/bin/python -m pip install --upgrade pip
<PREFIX>/envs/openmm-8.5.2/bin/python -m pip install 'openmm[cuda12]==8.5.2'
<PREFIX>/envs/openmm-8.5.2/bin/python -m openmm.testInstallation
```

The `cuda12` extra is an example, not a universal choice. Use the compatible CUDA extra available
for OpenMM 8.5.2 and the detected driver. For a CPU installation use `openmm==8.5.2`; do not claim
CUDA success unless the test reports the CUDA platform.

## Amber archive and build pattern

1. Run the archive inspector and record only the path, size, SHA-256, classification, top-level
   entries, relevant documentation filenames, and safety result.
2. Read the archive's README/INSTALL documents before extraction.
3. Extract to a new staging directory only after the user approves the plan.
4. Determine from the release documents whether Amber26/pmemd must be overlaid onto an AmberTools26
   source tree. Never guess or perform a blind merge.
5. Use out-of-source builds and the install layout documented by Amber26. Default to distinct
   `<PREFIX>/ambertools26` (AMBERHOME) and `<PREFIX>/pmemd26` (PMEMDHOME) prefixes unless the
   supplied release explicitly documents a combined install.
6. Enable CUDA and MPI only when Amber26 documents support for the detected toolkit, host compiler,
   MPI implementation, and compiler wrappers. Require the build to produce both `pmemd.cuda` and
   `pmemd.cuda.MPI`. Preserve the complete CMake configure output.
7. Choose parallel jobs conservatively. Reduce jobs on low-memory machines rather than allowing
   the compiler to exhaust RAM.
8. Run the Amber26 release targets through the intended launcher or scheduler environment and
   capture their logs: `make test.parallel` for the MPI installation (two ranks, then four or eight
   when resources permit for replica-exchange coverage), `make test.cuda.parallel` for
   `pmemd.cuda.MPI`, and `make test.cuda.parallel.SPFP` for the production SPFP build. Use one MPI
   rank per visible GPU for CUDA-parallel tests and inspect the manual's documented diff files.

Do not publish generic CMake flags as if they were guaranteed for Amber26. Quote the exact options
from the bundled release documentation in the proposed plan.

## Activation pattern

Prefer a standalone script rather than changing a shell startup file:

```bash
#!/usr/bin/env bash
set -e
export AMBERHOME=<PREFIX>/ambertools26
source "$AMBERHOME/amber.sh"
export PMEMDHOME=<PREFIX>/pmemd26
source "$PMEMDHOME/amber.sh"
export PATH=<PREFIX>/envs/openmm-8.5.2/bin:"$PATH"
```

Use shell-safe quoting for the real prefix. Do not put credentials, archive locations, or host
secrets in the activation script.

## Validation and manifest

Validate the exact executables and environment that the activation script selects:

```bash
python scripts/verify_install.py \
  --amberhome <PREFIX>/ambertools26 \
  --pmemdhome <PREFIX>/pmemd26 \
  --python <PREFIX>/envs/openmm-8.5.2/bin/python \
  --require-cuda-mpi \
  --mpi-launcher mpirun \
  --output <PREFIX>/manifests/validation.json
```

`--require-cuda-mpi` implies `--require-cuda`. Substitute `mpiexec` or the documented scheduler
launcher when appropriate. Omit the MPI requirement only when multi-GPU Amber support is explicitly
out of scope. This structural probe does not replace `make test.parallel`,
`make test.cuda.parallel`, or `make test.cuda.parallel.SPFP`; run those targets with the intended
rank/GPU layout and preserve their logs and diff summaries.

The final manifest should include:

- component versions and absolute installation paths;
- archive filenames, sizes, and SHA-256 values, without licensed contents;
- selected OpenMM and Amber backends;
- compiler, CMake, driver, CUDA toolkit, MPI implementation/compiler-wrapper, launcher, and
  scheduler versions used;
- environment package list or lock/export file;
- configure command and build options;
- validation commands, timestamps, return codes, and log paths;
- rank counts, `DO_PARALLEL`, `CUDA_VISIBLE_DEVICES`, and results/diff-log paths for
  `test.parallel`, `test.cuda.parallel`, and `test.cuda.parallel.SPFP`;
- warnings and any explicitly accepted CPU fallback.

Do not record usernames, tokens, license data, the full environment, or arbitrary host files.

## Common failure interpretation

- `nvidia-smi` fails: treat CUDA as unavailable until the driver is repaired by the machine owner.
- OpenMM imports but lists no CUDA platform: inspect its package variant, driver compatibility, and
  plugin load diagnostics; do not silently switch to OpenCL.
- `nvcc` is absent: OpenMM's packaged CUDA runtime may still work, but `pmemd.cuda` cannot be built
  from source without a supported toolkit.
- `pmemd.cuda` exists but `pmemd.cuda.MPI` does not: reconfigure with the exact Amber26-documented
  MPI and CUDA options and confirm that the intended MPI compiler wrappers were selected.
- `pmemd.cuda.MPI` starts with one rank but fails with multiple ranks: check MPI ABI consistency,
  launcher/scheduler environment, visible devices, GPU affinity, and Amber's parallel CUDA tests.
- CUDA compiler is rejected by Amber CMake: use a supported toolkit/compiler pair or obtain consent
  for a CPU build; do not patch away a compatibility check without evidence.
- Archive inspector reports unsafe members: stop. Do not extract the archive.
- AmberTools and Amber/pmemd layouts do not match the bundled overlay instructions: stop and ask for
  the correct release archives.
- Amber tests fail: keep the logs and report the failing test; do not proceed to scientific work.
