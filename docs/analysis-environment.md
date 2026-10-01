# The analysis environment

Analysis and plotting live in their own environment, **not** in `openmm-env`.

```bash
mamba create -p /data3/data/chen/software/md-stack/envs/analysis-env -c conda-forge \
    python=3.12 numpy scipy pandas matplotlib seaborn scikit-learn pymbar mdtraj \
    netcdf4 pyyaml

# jaxlib resolves to a CUDA build; replace it with the CPU one, and PIN it on every later
# install so the solver cannot pull the CUDA build back in. See below for why this matters.
mamba install -p /data3/data/chen/software/md-stack/envs/analysis-env -c conda-forge \
    "jaxlib=*=*cpu*"
```

| package | version as installed |
|---|---|
| python | 3.12.14 |
| numpy | 2.4.6 |
| scipy | 1.18.1 |
| pandas | 2.3.3 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| scikit-learn | 1.9.1 |
| statsmodels | 0.15.0 (via seaborn) |
| pymbar | 4.2.0 |
| mdtraj | 1.11.1 |
| netCDF4 | 1.7.4 |
| jax / jaxlib | 0.10.2, **cpu build** |

Verified after installation:

```text
$ analysis-env/bin/python -c "import jax; print(jax.devices())"
[CpuDevice(id=0)]
```

## Why it is separate

`openmm-env` is the environment every session runs simulations in, and `md-tools` is installed
there non-editable. Three reasons not to put an estimator or a plotting stack beside it:

**An analysis package can take the GPUs out from under a running campaign.** `pymbar` 4 imports
JAX, and a CUDA-enabled JAX preallocates memory on *every visible device* at import. On a host
where other sessions hold cards, importing pymbar in the simulation environment is enough to
disturb their runs — and it happens on `import`, before any analysis is asked for.

**The engine's dependency surface is a deliberate 3.3 MB.** `md_tools` declares `pyyaml`, `numpy`
and `pydantic`, and imports `openmm`, `mdtraj`, `netCDF4`, `rdkit` and `openff-toolkit`. Estimators
bring dependencies the sampling does not need, which is exactly why WHAM and MBAR are not in the
package. An environment that mixes them makes that boundary unobservable: a script that quietly
imports pymbar still runs, so nothing says the line was crossed.

**Analysis versions have to be quotable.** A free energy depends on which pymbar computed it —
`4.0.x` and `4.2.x` differ in API and solver defaults, and the `hpREST2` environment on this host
carries 4.0.3 while `openmm-env` carries 4.2.0. Pinning the analysis stack in its own environment
makes "which estimator produced this number" answerable.

## The CPU-only jaxlib is the point, not a detail

conda-forge resolves `pymbar` to a CUDA jaxlib by default (`jaxlib-0.10.2-cuda130_*`). Installed
that way, the analysis environment both bloats and becomes capable of seizing a card. The CPU
build removes the capability rather than relying on remembering an environment variable:

```text
$ analysis-env/bin/python -c "import jax; print(jax.devices())"
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.
Falling back to cpu.
```

With the CUDA build, the equivalent of that line allocates on every visible GPU.
`PYMBAR_DISABLE_JAX=1` and `CUDA_VISIBLE_DEVICES=""` still work, and are worth setting in a
script that might run in a different environment, but they are a second line of defence here
rather than the first.

## What the analysis scripts need

Nothing from `md_tools`. `docs/tutorial/ALA/umbrella_analysis.py` and `umbrella_vs_unbiased.py`
import only `numpy`, `yaml` and the standard library, and read a run's *outputs* — the CV series
and the content-addressed restraint definition the run recorded. That is what lets the engine and
the estimator sit in different environments, and it is the same boundary the package draws by
refusing to contain an estimator at all.

## Removing the analysis packages from `openmm-env`

Not done unilaterally: `openmm-env` is shared by every session on this host. The audit, for
whoever does it:

| package | in `openmm-env` | needed there? |
|---|---|---|
| `pymbar`, `jax`, `jaxlib` | yes | **no** — appear nowhere in `src/` or `tests/` |
| `matplotlib` | yes | **no** — appears nowhere in `src/` or `tests/`; the scaler's `<RESNAME>-unscaled.png` is drawn by rdkit |
| `pandas`, `scikit-learn` | yes | **no** — named only in comments |
| `mdtraj` | yes | **YES** — imported by `md_tools` itself (`openmm/trajectory.py`, `ais/source_ensemble.py`, `ais/run.py`) and by 20 test files |
| `netCDF4` | yes | **YES** — imported by 6 modules and 7 test files |
| `scipy` | yes | **YES, indirectly** — `mdtraj` requires it |
| `rdkit`, `openff-toolkit` | yes | **YES** — ligand parameterisation, 10 and 4 modules |

So `pymbar`, `jax`, `jaxlib`, `matplotlib`, `pandas` and `scikit-learn` are the removable set, and
`mdtraj`, `netCDF4`, `scipy`, `rdkit` and `openff-toolkit` must stay. Removing anything from a
shared environment should be agreed with the sessions using it first.
