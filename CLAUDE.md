# MD-tools — working guidance

Read this before acting. It is short on purpose.

## The package

One installed executable, `md-openmm`. Exactly four public work commands:

```text
md-openmm build-top      a structure       -> built.xml + built.pdb + built.log
md-openmm build-md       a protocol config -> run scripts, .in files and run.sh in ./md_script/
md-openmm md-run         an Amber-like .in -> a stage, a ladder, or AIS switching paths
md-openmm data-register  a finished tree   -> a verified dataset under $MD_DATA
```

AIS is `protocol: AIS` in a `build-md` configuration and `protocol = AIS` in an `.in` file. **Do
not add a fifth command**, and do not add a second executable: `md-run` is a SUBCOMMAND.

`md-run` is a surface, not an implementation. It parses the Amber-like input, resolves it through
`md_tools.build.md`, writes the resulting `resolved.config` into `-odir` with the input's sha256,
and hands the work to the same function a generated script calls — `stage_main`, `replica_main`,
`ais_main`. If you find run logic in `md_tools.run`, it is in the wrong package.

**The short flags are Amber's**, and this is contractual:

```text
-i mdin   -o mdout   -p topology   -c input restart   -r output restart   -x trajectory
-s the serialised OpenMM System (no Amber counterpart)   -log the provenance record
```

`-x` is never the System. `-s` is required, `-x` is optional with a protocol default, and both
mistakes are refused by name. Flag abbreviation is off.

**`-o` and `-log` are two files**, for two readers: `.out` is what a person tails during a run,
`.log` is the machine-readable provenance record. Never merge them; only a genuine collision is
refused.

**`resolved.config` is authoritative** and md-run writes it on every invocation, from the `.in`
file it re-reads each time. Every `.in` that `build-md` writes resolves back to exactly the
`resolved.config` beside it; that round-trip is a test, not a convention.

These do not exist and must never be suggested: `sys-config`, `sys-gen`, `md-gen`, `setup`,
`show-default`, `openmm-md`, `md-template`, `md-data-finish`, `md-data-register`.

They may still be NAMED, in exactly two places: a test that asserts one is refused, and
release or migration history that says it is retired. Both are how the guarantee is kept. Anywhere
else — a docstring, a comment, a module name, an example — naming one as if it works is stale text,
not an interface. An internal module may not be named after a retired executable either: the
executor lives at `md_tools/remd/executor.py` because `openmm_md.py` read as the command.

## Configuration

```text
configs/machine/user.config.example   identity, $MD_DATA, and `machine.openmm`
configs/sys/build-top.config          force fields, solvent, box, ions, constraints, HMR
configs/md/{cMD,REST2,rREST2,AIS}.config   protocol, stage lengths, reporting
```

`md_tools.build.md` is the ONE authority for MD workflow configuration — protocol, stage lengths,
reporting. `md_tools.openmm.system_config` and `system_defaults` cover the `build-top` system half
and nothing else. If you find a second function resolving an MD configuration, it is residue.

## The import API

```text
md_tools.md      PositionalRestraint, ReportingConfig, run_stage,
                 run_generated_stage, run_generated_workflow
md_tools.rest2   REST2Scaler, ScalingSelection
md_tools.remd    REMDRunner, NeighborExchangeRule, run_remd, run_generated_remd
md_tools.remd.reservoir   ReservoirRefreshRule
md_tools.ais     run_generated_ais, path_trajectory_name, paths_for_rank
md_tools.run     parse_run_input, md_run_main
```

**Generated Python files are entry points, not copies of the implementation.** A stage script is:

```python
#!/usr/bin/env python
from md_tools.md import run_generated_stage
raise SystemExit(run_generated_stage(__file__, "min"))
```

No function or class definition, no `argparse`, no OpenMM import, no reporter, restraint, barostat,
scaler or exchange rule built there. `resolved.config` beside the script is the single resolved
declaration: located from `__file__` so the directory is movable, re-validated by the strict
resolver at execution, and bound into the checkpoint fingerprint.

ONE REST2 scaler serves fixed-τ cMD, REST2, rREST2 and AIS. `md_tools/openmm/templates/` is gone —
it held installed runtime code, not templates. `md_tools.runtime` is compatibility-only: it
re-exports and defines nothing, and new work must not import from it.

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
* **CUDA is the default and it is mandatory**, and the platform is a MACHINE property:
  `machine.openmm.{platform,precision,device_policy}` in the user configuration, never in a
  protocol. `dynamics.platform` is retired and refused with the migration. There is no automatic
  fall back in either direction: a run that quietly moved to the CPU finishes, writes a trajectory
  and reports success two orders of magnitude later. `--cpu` is the one per-run override and the
  record distinguishes `built-in default`, `machine.openmm` and `--cpu (command line)`. One
  resolver — `md_tools.openmm.platform_policy` — serves stages, ladders and AIS alike.
* **Ordinary MD writes genuine DCD; AIS and REMD write genuine NetCDF.** OpenMM has a native DCD
  writer and no native NetCDF writer, so a stage refuses a `.nc` trajectory name rather than
  putting DCD bytes in it. What a file IS is decided from its leading bytes
  (`md_tools.openmm.trajectory`), never from its suffix, and a source whose suffix and contents
  disagree is refused with both named.
* **A multi-rank launch either coordinates or stops.** `md_tools.remd.mpi` is the ONLY MPI
  authority — the only module that imports `mpi4py`, and the only place a barrier, gather or abort
  is decided. It cross-checks the launcher's rank and size against the communicator's and both
  against `-ng` and the replica count, all before any output exists. No collective may silently
  become a no-op while the world is plural: N ranks with no coordination are N simulations writing
  over one set of paths, and the result looks complete. A second `except ImportError: return`
  anywhere is a second policy, and it will be the one that runs.
* **The guard belongs in the runtime, not in `md-run`.** `replica_main`, `ais_main` and
  `stage_main` validate the launch themselves, because the generated wrappers call them directly.
  A safe outer command wrapping an unsafe runtime is worse than no wrapper: it makes the unsafe
  path look tested.
* **Nothing is written until the whole preflight passes.** `md_tools.run.preflight` checks flag
  roles, resolved-path collisions, input existence and format, the MPI launch, the machine
  configuration, `--cpu`/`--device`, the device policy, platform availability and a real CUDA
  Context — before `-odir`, `resolved.config`, the `.out`, the `.log`, a group file, a trajectory
  or a checkpoint exists. A `-odir` holding a `resolved.config` is indistinguishable from a run
  that happened.
* **An absent configuration is not an invalid one.** No user configuration resolves to the
  built-in CUDA/mixed/local_rank. An existing one that is malformed, has a duplicate key, an
  unknown field or an invalid value is FATAL and is never replaced by those defaults — the machine
  would then run on settings nobody chose. `MD_TOOLS_CONFIG` naming a missing file is a broken
  reference, not an absence.
* **An AIS checkpoint commit is a generation transaction.** New generation, fsync, digest,
  sidecar, fsync, then the `current_checkpoint.json` pointer replaced atomically last. Resume
  follows only the pointer and verifies the digest; never the newest generation on disk, which is
  exactly what a crash leaves behind. Overwrite-then-replace can pair a new Context with old
  accumulated work, and nothing fails.
* **Every accepted reporting option does something.** AIS has four independent cadences — work
  observations, trajectory frames, the `system.csv` state table, and checkpoints — each dividing
  `switching_steps` on its own. A setting that is accepted and inert is worse than one refused.
* **AIS path identity does not depend on the worker count.** `paths_for_rank(rank, size, total)`
  is a pure function, and global path *n* always writes `AIS_traj000n.nc`. A campaign resumed on a
  different number of GPUs must land on the same files.
* **A scaled run (`dynamics.tau > 0`) is fixed-volume throughout**, equilibration included, and
  its pressure-coupled stages are RENAMED exactly as the implicit ones are — never NPT with the
  pressure ignored.

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
