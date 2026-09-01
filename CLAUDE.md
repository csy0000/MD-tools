# MD-tools — working guidance

Read this before acting. It is short on purpose.

## The package

One installed executable, `md-openmm`. Exactly three public work commands:

```text
md-openmm build-top      a structure       -> built.xml + built.pdb + built.log
md-openmm build-md       a protocol config -> readable run scripts in ./md_script/
md-openmm data-register  a finished tree   -> a verified dataset under $MD_DATA
```

AIS is `protocol: AIS` in a `build-md` configuration. **Do not add a fourth command.**

These do not exist and must never be suggested: `sys-config`, `sys-gen`, `md-gen`, `setup`,
`show-default`, `openmm-md`, `md-template`, `md-data-finish`, `md-data-register`. If you find one
named anywhere outside a release note explaining the change, it is stale text, not an interface.

## Configuration

```text
configs/machine/user.config.example   identity and $MD_DATA, for `data-register --init`
configs/sys/build-top.config          force fields, solvent, box, ions, constraints, HMR
configs/md/{cMD,REST2,rREST2,AIS}.config   protocol, stage lengths, reporting
```

Ordinary browsable files at the repository root, one copy each, shipped as **wheel data files**
and found through `md_tools.configs.example_root()`. Never add a symlink or a second hand-edited
copy. Every file is YAML despite the `.config` suffix, unknown keys are refused, and the examples
must resolve — through the real resolver — to the model's own defaults.

User configuration lives at `${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config`. Never write
it into the package, the repository, or `site-packages`.

## The contract

One contract, version 2, in `md_tools.data_contract`. Canonical paths:

```text
$MD_DATA/{year}/{project_name}/{data_name}/
$MD_DATA/{year}/common/{project_name}/{data_name}/
```

No month segment. No `baseline/` namespace. **`md_data` is never imported at runtime.** Schemas are
generated from the models and drift-checked; never hand-edit one.

## Scientific invariants

Do not change these without a failing test that demonstrates a defect.

* **REST2/rREST2**: every replica at the same physical temperature — Hamiltonian scaling, not
  temperature REMD. Bonds and angles unscaled; ordinary amide omega torsions unscaled; eligible
  solute torsions and CMAP by `(1-tau)²`; solute–solute nonbonded and 1-4 by `(1-tau)²`;
  solute–environment by `(1-tau)`; generalized-Born by `(1-tau)`. Exchanges never rescale
  velocities. The runtime is NVT. One trajectory per fixed thermodynamic **state**
  (`remd0.nc`…), never per walker.
* **AIS**: `tau` is the only public, persisted coordinate — never persist `s` or `sqrt(s)`.
  Work is `ΔW_j = U(τ_{j+1}, x_j) − U(τ_j, x_j)`: parameters move at frozen coordinates, then the
  configuration propagates. Observation 0 precedes all work and has exactly zero. Switching is at
  fixed volume; a barostat in the System is refused.
* **Lengths are integer step counts**, everywhere. Logs derive ps/ns for the reader. A schedule
  that would have to be rounded is refused with the arithmetic that would fix it.
* **Implicit solvent (GBn2)** has no box, no barostat, no salt and no NPT stage. Implicit stages
  are *renamed* (`eq_nvt_posres_2.py`, `eq_nvt_free.py`), never NPT with the pressure ignored.
* **HMR defaults off.** A 4 fs timestep is refused unless the masses serialised in the System prove
  repartitioning — a fact at run time, not a claim in a configuration.
* **Completion is read from a machine record**, never from prose in a log.

## Test lanes

```bash
python -m pytest tests -m "not slow and not gpu"    # fast
python -m pytest tests -m "gpu or slow"             # builds systems and integrates them
python -m build                                     # then install the wheel and test outside the checkout
```

GPU tests run on **CUDA**. Never substitute CPU execution for CUDA evidence, and never mark a
GPU test as passing on a machine that has no GPU — deselect it honestly.

## Rules

* No project-specific path, machine path, `$MD_DATA` value, GPU index or username in a committed
  file. A generated script must contain no absolute path and must not import a source checkout.
* Do not remove a test because it fails. Classify it: migrated, obsolete with a named deleted
  feature, or still blocking.
* `.gitignore` patterns must be **root-anchored** (`/build/`, not `build/`). A bare pattern matches
  at any depth and once silently excluded `src/md_tools/build/` from every commit.
* Do not search Git history for how the package works today; see `docs/README.md`.
