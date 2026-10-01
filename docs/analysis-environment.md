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

Not done unilaterally: `openmm-env` is shared by every session on this host.

!!! warning "This audit is a measurement with a date, not a property of the package"
    Taken against **`58e4649c`**, by grepping `src/` and `tests/` for each import. It is true of that
    tree and of no other, and the way it goes false is by someone adding an import — which is
    silent from the audit's point of view.

    **It has already happened once, exactly as predicted — and this is the mechanism working.**
    A torsional-HDBSCAN classifier landed in **v0.6.4** (`8516c12a`) as `md_tools.analysis`, and
    it needs `sklearn.cluster.HDBSCAN`. So the `scikit-learn` row below has moved: it IS now a
    declared dependency of the package, but **only of the `analysis` extra**, and every `sklearn`
    import in it is function-level (`_torsions.py`, `_t_symmetry.py`), so nothing pulls it in at
    module load. An ordinary simulation install acquires neither scikit-learn nor rdkit through it.

    What that means for the table: `scikit-learn` is still absent from the import path of a plain
    `import md_tools`, so removing it from a SIMULATION environment remains safe — but it is no
    longer true that it "appears nowhere", and anyone running the analysis extra needs it present.
    Re-grep, and note that the row's wording changed while its practical answer did not. That
    distinction is the whole reason this box exists. The shape to use is a
    lazily-imported **extra** (`analysis = ["scikit-learn"]`), for the same reason as the CPU-only
    jaxlib above: a dependency that drags a stack in for a user who wanted none of it belongs
    behind an extra and a lazy import, so that `import md_tools` costs nothing to someone who
    only wants to run a simulation.

    **There is a working precedent, verified at `0349c9e4`:** `md_tools.alchemy.estimators` is the
    only module mentioning pymbar, and `_pymbar()` is a function-level import raising a named
    `MissingAnalysisDependency` when the extra is absent. Two details in it are worth copying:

    * it sets `PYMBAR_DISABLE_JAX` before importing pymbar, and deliberately does **not** touch
      `JAX_PLATFORMS`, which would be a process-wide device decision imposed on every other JAX
      user in the process;
    * it RECORDS which solver ran, as `pymbar_backend`, because an earlier import elsewhere in
      the same process may already have chosen JAX. A guard that cannot promise a backend should
      report the one it got rather than assert the one it wanted.

    So: re-run the grep before acting on any row. Do not act on this table's age.

!!! danger "Do not check an import's ABSENCE through `openmm-env`'s interpreter"
    `openmm-env`'s python imports the **installed** `md_tools` from site-packages, not the tree you
    are standing in, and the installed copy lags. Asking it whether a module imports `sklearn`
    answers a question about site-packages.

    The asymmetry is the trap. Checking for an import's PRESENCE fails loudly when you are pointed
    at the wrong copy; checking for its ABSENCE **succeeds** — the module really is not there, in
    that copy — and reads as confirmation. A sibling session hit exactly this while verifying the
    guard above, getting `ModuleNotFoundError` for a module that exists on its own branch.

    The rows in the table below were grepped over `src/` and `tests/` as FILES, so they are not
    exposed. Any import-time check is. Use `PYTHONPATH=src` and print `md_tools.__file__` in the
    same breath, so the answer says which copy it came from.

The audit, for whoever does it:

| package | in `openmm-env` | needed there? |
|---|---|---|
| `pymbar`, `jax`, `jaxlib` | yes | **no** — appear nowhere in `src/` or `tests/` |
| `matplotlib` | yes | **no** — appears nowhere in `src/` or `tests/`; the scaler's `<RESNAME>-unscaled.png` is drawn by rdkit |
| `pandas` | yes | **no** — named only in comments |
| `scikit-learn` | yes | **not at module load** — since v0.6.4 it is a dependency of the `analysis` EXTRA, imported inside functions in `md_tools.analysis`. Safe to remove from a simulation environment; required wherever the extra is used |
| `mdtraj` | yes | **YES** — imported by `md_tools` itself (`openmm/trajectory.py`, `ais/source_ensemble.py`, `ais/run.py`) and by 20 test files |
| `netCDF4` | yes | **YES** — imported by 6 modules and 7 test files |
| `scipy` | yes | **YES, indirectly** — `mdtraj` requires it |
| `rdkit`, `openff-toolkit` | yes | **YES** — ligand parameterisation, 10 and 4 modules |

So `pymbar`, `jax`, `jaxlib`, `matplotlib`, `pandas` and `scikit-learn` are the removable set as
of the commit above, and `mdtraj`, `netCDF4`, `scipy`, `rdkit` and `openff-toolkit` must stay.

`scikit-learn` is the row most likely to move, for the reason in the warning. `pandas` is listed as
removable on the strength of its two hits being **comments** rather than imports — a distinction a
coarser grep would miss, and one worth re-checking the same way.

**Who should do it, and when.** Not three peer sessions agreeing among themselves. Removing a
package from a shared, non-editable environment changes what every session's `import md_tools`
resolves to, so it waits until 0.6.4 is tagged, needs the sign-off of the sessions using the
environment, and should be run by whoever owns it. Both other sessions on this host independently
reached the same answer, on partly different grounds: one that an installed-config failure reads
as a defect when it is really install lag, the other that a reallocation nobody's user asked for
is not a peer's to make.

**The environment's name lives in documentation only.** No committed file names an environment or
a machine path — that is a project rule, and it also prevents a near-miss from becoming load
bearing: this environment was referred to elsewhere as `openmm-analysis`, which does not exist on
this host. The real path is at the top of this page.
