# Installing

Step 0 needs root and is done once per machine; steps 1–4 are a normal user install. If you
already have a conda environment with OpenMM 8.6, skip to step 3.

Once this is done, see [machine configuration](machine-configuration.md) for the one file that
says which platform this machine runs on and where registered data goes.

## 0. The NVIDIA driver — the one step conda cannot do

**Needs root, and is done once per machine.** Everything else below is a normal user install.

Conda supplies the CUDA *toolkit* — `libcudart`, `libnvrtc`, `libcufft` — and nothing else. The
**driver** is the kernel module plus `libcuda.so.1`, and it comes from the operating system:

```text
libcuda.so.1   ->  /lib/x86_64-linux-gnu/libcuda.so.1     the DRIVER, from the OS
libnvrtc.so    ->  <env>/lib/libnvrtc.so                  the TOOLKIT, from conda
```

Both are loaded by OpenMM's CUDA plugin, from two different places. No conda package can install
the first: `environment.yml` cannot rebuild a driver, and a machine without one has no CUDA no
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

## 1. A package manager, if you have none

The scientific stack is conda packages. Any of conda, mamba or micromamba works; micromamba is a
single static binary and needs no base environment:

```bash
mkdir -p ~/software/md-stack && cd ~/software/md-stack
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=$PWD
```

## 2. The environment

```bash
micromamba create -y -p ~/software/md-stack/envs/openmm-env -f environment.yml
```

That installs Python 3.12, OpenMM 8.6, OpenFF, AmberTools, ParmEd, RDKit, MDTraj, OpenMMTools and
NetCDF4 — everything the five commands import — plus `mpi4py` and `openmpi` for multi-rank REST2,
rREST2 and AIS, and a `cuda-version` pin.

**One file, for every machine.** There used to be a second, CPU-only one for CI; they were merged
because two descriptions of "the environment MD-tools is tested against" drifted apart. CI
installs the CUDA stack and cannot use it — the runners have no driver — which costs a larger
solve and nothing else, since no test there claims CUDA evidence.

**The `cuda-version` pin is not a preference.** OpenMM compiles kernels at run time and the driver
assembles the PTX, which is backward-compatible only, so a newer CUDA runtime against an older
driver installs cleanly and then fails every kernel it must compile. The file's header explains
how to reproduce that deliberately; raise the pin only alongside a stated minimum driver.

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

## 3. The package

```bash
pip install --no-deps .
```

`--no-deps` is deliberate and not optional. The scientific stack comes from conda; this package
adds only pure Python. Without it, pip pulls a second, pip-built OpenMM alongside the conda one
and the two disagree about which native libraries are loaded.

## 4. Sourcing it, every time

**Activate the environment before running anything.** Not a convenience — AM1-BCC goes through
AmberTools' `sqm`, which the OpenFF toolkit discovers by looking on `PATH`. In a bare shell it is
not found even though it is installed, `AmberToolsToolkitWrapper` is silently absent from the
registry, and AM1-BCC becomes unavailable under its own name.

```bash
micromamba activate ~/software/md-stack/envs/openmm-env
```

Or put the environment on `PATH` from your shell profile, which is what a shared machine usually
wants:

```bash
# ~/.bashrc
export PATH="$HOME/software/md-stack/envs/openmm-env/bin:$PATH"
export MD_DATA="/path/to/your/managed/storage"     # where data-register writes
```

Check it:

```bash
md-openmm --version
command -v sqm mpiexec        # both must resolve inside the environment
```

## Upgrading

**A `git pull` alone changes nothing you run.** `pip install --no-deps .` copies the package into
`site-packages`; it is not an editable install, so the checkout and the thing on your `PATH` are
two different copies of the code:

```bash
git pull
pip install --no-deps --force-reinstall .
```

`--force-reinstall` is there because pip compares versions and does nothing when they match — and
two different `dev` commits usually carry the same `version`. Without it, an upgrade silently
no-ops and `md-openmm --version` keeps reporting what it reported before.

Check that it took, rather than assuming:

```bash
md-openmm --version                     # must match `version` in pyproject.toml
```

**This applies to the shipped configurations too, and that part surprises people.**
`configs/*.config` are installed as **wheel data files**, and `md_tools.configs.example()` reads
them from `<env>/share/md-tools/configs/` — never from your checkout. So a stale install serves
stale examples: on the machine this was written on, the installed `cMD.config` was 15,756 bytes of
documentation that the checkout had reduced to 1,938 four commits earlier, and every local test
run had been reading the old one for weeks without anyone noticing.

**If you run the test suite**, note that `environment.yml` gained `pytest-xdist` in 0.5.3.
`pyproject.toml` sets `addopts = "--dist loadgroup"`, which every `pytest` invocation passes, so an
environment without the plugin exits 4 with `unrecognized arguments: --dist` and collects nothing.
An existing environment needs it added once:

```bash
micromamba install -y -p ~/software/md-stack/envs/openmm-env -c conda-forge pytest-xdist
```

Running simulations needs none of this — it matters only for the suite.

**The environment files were merged in 0.5.3.** There is one `environment.yml` now;
`environment-ci.yml` and `environment-cuda.yml` are gone, so any script naming them needs updating.

## Next

* [Machine configuration](machine-configuration.md) — the platform this machine uses, and `$MD_DATA`.
* [Methods](openmm_methods/README.md) — pick a protocol and run it.
* [Running](md-run.md) — the flags, the `.in` language, the MPI rules.
