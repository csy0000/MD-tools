# MD-tools — working guidance

Read this before acting. It is short on purpose.

## The package

One installed executable, `md-openmm`. Exactly four public work commands:

```text
md-openmm build-top      a structure       -> built.xml + built.pdb + built.log
                         --rest2-scaler: a built System -> build/<method>/system_state<i>.xml + scaler.yaml
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
-s2 / -p2  AIS only: the second end state V1 (sander's second group; refused by name elsewhere)
```

`-x` is never the System. `-s` is required, `-x` is optional with a protocol default, and both
mistakes are refused by name. Flag abbreviation is off. A REST2 ladder is the one exception to
where `-s` goes: it reads `-s` ONLY from its group file, one saved state per line, and refuses it on
the command line.

**`-o` and `-log` are two files**, for two readers: `.out` is what a person tails during a run,
`.log` is the machine-readable provenance record. Never merge them; only a genuine collision is
refused.

**`resolved.config` is authoritative** and md-run writes it on every ACCEPTED invocation, from
the `.in` file it re-reads each time. Not on a refused one: an invocation that is rejected while
validating the records it intended to continue writes nothing into the output tree at all --
no `resolved.config`, no definition copy, no `.out`, no `.log`, no run-state record. The
rejected attempt must not replace the prior run's authoritative account of itself, which is the
same reason completion is read from a machine record rather than from prose. Every `.in` that `build-md` writes resolves back to exactly the
`resolved.config` beside it; that round-trip is a test, not a convention.

## Configuration

```text
configs/machine/user.config.example   identity, $MD_DATA, and `machine.openmm`
configs/sys/build-top.config          force fields, solvent, box, ions, constraints, HMR
configs/md/{cMD,REST2,AIS}.config   protocol, stage lengths, reporting
```

`md_tools.build.md` is the ONE authority for MD workflow configuration — protocol, stage lengths,
reporting. `md_tools.openmm.system_config` and `system_defaults` cover the `build-top` system half
and nothing else. If you find a second function resolving an MD configuration, it is residue.

## The import API

```text
md_tools.md      PositionalRestraint, ReportingConfig, run_stage, run_generated_stage
md_tools.rest2   REST2Scaler, ScalingSelection
md_tools.remd    REMDRunner, NeighborExchangeRule, run_remd, run_generated_remd
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

ONE REST2 scaler, and it runs in ONE place: `md-openmm build-top --rest2-scaler`, which writes
every scaled Hamiltonian a run will integrate as a file (see the invariants below). A stage and a
ladder never scale, and AIS scales nothing: it mixes two end-state Systems it is given.
`md_tools/openmm/templates/` is gone —
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
$MD_DATA/common/{year}/{project_name}/{data_name}/
```

No month segment. No `baseline/` namespace. **`md_data` is never imported at runtime.** Schemas are
generated from the models and drift-checked; never hand-edit one.

## Scientific invariants

Do not change these without a failing test that demonstrates a defect.

* **REST2**: every replica at the same physical temperature — Hamiltonian scaling, not
  temperature REMD. Convention v3, `rest2-unscaled-torsions`: bonds and angles unscaled; every
  UNSCALED TORSION unscaled — each proper torsion across an ordinary amide C–N (omega), an aromatic
  ring bond or another double bond (the ARG guanidinium included), and every improper; eligible
  solute torsions and CMAP by `(1-tau)²`; solute–solute nonbonded and 1-4 by `(1-tau)²`;
  solute–environment by `(1-tau)`; generalized-Born by `(1-tau)`. Exchanges never rescale
  velocities. The runtime is NVT. One trajectory per fixed thermodynamic **state**
  (`solute_state<i>_prod<N>.nc` and, when a whole-system cadence is set,
  `whole_state<i>_prod<N>.nc` — `state_trajectory_name` is the one authority), never per walker.
  The index is the STATE's, not the walker's: Amber and GROMACS write walker-following files and
  sort them afterwards, which is why `cpptraj` needs `remdtrajtemp` and GROMACS ships `demux.pl`.
* **A scaled Hamiltonian is built once, as a file, and never re-derived at run time.**
  `md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config
  build/scaler.config` writes `build/<method>/system_state<i>.xml` (method REST2, cMD or AIS) and a
  `scaler.yaml` recording the source System's sha256, every state's tau and sha256, the solute and
  every unscaled central bond and improper, plus `<RESNAME>-unscaled.png` for each small molecule.
  The directory is staged and renamed into place; `--overwrite` moves the old one aside, never
  deletes it. `md_tools.build.scaler` is the one writer and `md_tools.rest2.states.
  scaled_state_identity` the one reader.
* **Unscaled torsions are CLASSIFIED, and a torsion nobody can classify is refused.** Proteins by
  the residue table (`PROTEIN_UNSCALED_BONDS`), small molecules by bond orders from an SDF:
  `sdf_filelist` in the scaler config, else `<RESNAME>.sdf` beside the System, else `built.sdf`
  when it is the only non-standard residue; with several and no mapping it refuses rather than
  guesses. Proper versus improper is
  decided from the bond graph (bonds ∪ constraints): a bonded chain is proper, one atom bonded to
  the other three is improper, anything else is refused. `unscaled_torsions` (the old
  `omega_*` keys) is the one enforcing entry point; records written under the old names are still
  read, and continuing a v2 run is refused.
* **A stage never scales; `dynamics.tau` is a CLAIM checked against the System it is given.** A
  saved state is integrated as it is, and a tau that disagrees with its `scaler.yaml` -- including
  tau 0 on a hot state, which is an NPT stage on a scaled System -- is refused. tau > 0 on a
  System that is not a saved state is refused, naming the command that builds one. Minimisation
  claims tau 0 and minimises the unscaled built System, because `min/` is shared by every method.
  `build-md` refuses a hot run until `build/<method>/system_state0.xml` exists, and `run.sh` passes
  it as `SCALED_SYSTEM`.
* **A REST2 ladder integrates saved states, and reads them only from its group file.** Every line
  of `remd_groupfile.<segment>` names `-s ../build/REST2/system_state<i>.xml`, state i on line i,
  all from ONE `scaler.yaml` whose taus equal the ladder's; `build-md` checks that record against
  `built.xml` before writing a line, and the ladder preflight checks it again. The runtime writes no
  group file; `solute.yaml` records the `scaler.yaml` (relative to the run) and every state's
  sha256. Each line's `-i` must be the `_protocol.py` the ladder writes into `-odir` -- `run.sh`
  launches from the run directory with `-odir .` -- and any other `-odir` is refused before
  output. A REST2 reference bundle carries the saved states byte for byte, with `scaler.yaml` and
  the built System that `verify_rungs.py` re-derives them from.
* **AIS**: a transformation between two end-state Systems with identical particles, masses and
  constraints, `V(λ) = (1−λ)V0 + λV1`. V0 is `-s`/`-p`, the state the source ensemble was sampled
  from; V1 is `-s2`/`-p2`. `λ` is the only public, persisted coordinate and runs 0 → 1 — a reverse
  switch exchanges the files, never runs the schedule downhill. Work is
  `ΔW_j = V(λ_{j+1}, x_j) − V(λ_j, x_j)`: parameters move at frozen coordinates, then the
  configuration propagates. Observation 0 precedes all work and has exactly zero. Switching is at
  fixed volume; a barostat in either System is refused. A pair that differs in anything but
  parameters — particles, masses, constraints, force layout, long-range treatment — is refused;
  atom mapping and softcore are not implemented. A source trajectory recording a System digest
  must record V0's. `ais.tau_start`, `tau_end`, `work_measurement` and `verify_every_updates` are
  retired and refused with the migration.
* **The potential is exactly linear in λ, and there are TWO probes of it.**
  `V(λ, x) = (1−λ)V0(x) + λV1(x)` is an IDENTITY at every coordinate, PME and dispersion correction
  included, because each end state is evaluated exactly as its own System would evaluate it
  (`md_tools.ais.two_state`: shared forces once, differing force pairs as the collective variables
  of one `CustomCVForce`, `λ` a Context parameter). `dV/dλ = V1 − V0`, so
  `ΔW_j = (λ_{j+1} − λ_j)·(V1 − V0)(x_j)` exactly — one evaluation per switch. The finite
  difference stays the DEFINITION, so a schedule changes which λ a path visits and never what
  work means. `ais.lambda_schedule` is `linear` (λ = t) or `tau-linear`
  (λ = [(1−τ₀+τ₀t)² − (1−τ₀)²]/[1 − (1−τ₀)²], matching REST2's solute–solute `(1−τ)²` along a τ
  linear in t, NOT its solute–environment `(1−τ)`); τ₀ is a number in `resolved.config`, never
  read from a REST2 record, and tau-linear is refused unless `-s` is the saved state at τ₀ and
  `-s2` its recorded source. The schedule, τ₀ and a digest of the λ table are in `AIS_run.json`
  and the fingerprint.
  The **work-basis probe** runs at the frozen pre-switch `x_j`, which is where the work convention
  defines work. The **observation-potential probe** runs at the coordinate a row SAVES, under that
  row's λ, which is what Hummer–Szabo reweighting consumes. They are different coordinates and
  must never be confused: writing the first under names that read as the second pairs the work of
  one configuration with the energy of another, silently. A row naming a `coordinate_frame_index`
  carries `potential_v0_kj_mol`, `potential_v1_kj_mol` and `potential_direct_kj_mol` measured at
  that frame, with `V1 − V0` checked against an independent evaluation; a row with no saved
  coordinate leaves them EMPTY rather than borrowing a neighbour's. `AIS_hs.csv` is the
  frame-aligned subset.
* **One cMD resume contract, and `--resume` is not required.** An INTERRUPTED stage — one whose
  committed checkpoint is short of its step count — continues automatically, with every appendable
  stream (DCD, state CSV, phase-space NetCDF) truncated to the counts that generation vouches for.
  A COMPLETED stage that verifies is skipped. Anything else is refused until `--overwrite`, which
  starts CLEAN and does not load the generations it was asked to replace. Requiring `--resume`
  would mean a re-run silently discards committed work, which is what the checkpoint prevents.
* **A serial protocol under a plural launch is refused.** `mpirun -n 8` on a cMD stage is eight
  simulations over one set of paths, interleaved, with nothing saying so.
* **An interrupted AIS path resumes mid-path**, from its last committed generation. The older
  claim that a switching path has no meaningful mid-path restart had the premise right (the work
  integral is defined along a whole path) and the conclusion wrong: a resume continues *that*
  path. Resume is exact in COMMITTED STATE — λ, accumulated work, counters, positions,
  velocities, stream counts — and not in trajectory on CUDA: the mixing force's inner Contexts
  keep atom-ordering state no checkpoint captures, so the continuation is a new realisation of the
  same switching process (decided 2026-09-16). On CPU and Reference it reproduces exactly. A completed path is skipped only after its manifest and the sha256 of every output it
  claims verify.
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
* **The guard belongs in the runtime, not in `md-run`.** `stage_main`, `replica_main` and
  `ais_main` run the shared preflight themselves and CONSUME its result, because the generated
  wrappers call them directly. A safe outer command wrapping an unsafe runtime is worse than no
  wrapper: it makes the unsafe path look tested. "Calls the preflight and ignores what it returns"
  is the same defect wearing a better name — if a runtime re-resolves the platform, the machine
  settings or the MPI world, there are two policies again.
* **An output directory has ONE identity.** `AIS_run.json` records the source digest, both
  end-state digests, the λ schedule, the seed policy, the selected frames, the reporting schema, the path count and the
  column schema. A second invocation into the same `-odir` with any of those changed is refused by
  name: the completed paths would be skipped, the rest run under the new settings, and the work
  table assembled out of two different experiments — readable, and describing neither.
* **A checkpoint commit is a generation transaction, for cMD as well as AIS.**
  `md_tools.openmm.checkpoint` is the one implementation. The cMD stage used to
  `saveCheckpoint(path)` and then write a sidecar beside it: a crash between them paired a new
  Context with an old `steps_done`, and the resume continued from step 3000 believing it was at
  2000. The committed state binds the fingerprint, the step, the stage, the seeds, the platform
  and the committed row and frame counts of every appendable stream; recovery truncates those
  streams to the committed counts and never infers progress from whichever file is longest.
* **`--check` creates nothing**, not even `-odir`. A `--check` that leaves a directory behind has
  already produced the thing whose absence the caller was asking about.
* **Every accepted flag does its job or is refused before any output.** Flags of another protocol
  are accepted by the parser and refused BY NAME in the preflight — leaving them off makes argparse
  say "unrecognized arguments", which explains nothing and puts the rule where `md-run` and a
  generated script can disagree about it. They did.
* **Under MPI, rank 0 writes the shared files** — `solute.yaml`, `_protocol.py` —
  atomically, and every rank then verifies the digest. The helpers are content-addressed: a stale
  `_protocol.py` from a ladder with a different state count is executable, runs perfectly and
  simulates something else.
* **A rank-local failure is made collective.** The preflight agrees across ranks before any output
  exists, and a runtime failure aborts the communicator. A rank that raises alone leaves the
  others waiting at the next collective for a participant that has already exited.
* **A refused continuation is read-only, and that includes its own logs.** An invocation that
  continues or verifies existing data validates every authoritative record it needs FIRST --
  committed CV prefixes, per-state and aggregate costs, completion manifests, and for AIS every
  selected path before work begins on any of them. That phase creates, rewrites, truncates and
  replaces nothing; the refusal goes to stderr. Only once it passes do logs, run-state records,
  `resolved.config`, reporters, truncation and publication happen, and ordinary runtime failure
  logging resumes from that point. Under MPI the ranks agree they all passed before any rank
  writes. `md_tools.run.continuation` is the one implementation, used by the public `md-run` and
  by the generated entry points alike.
* **Nothing is written until the whole preflight passes.** `md_tools.run.preflight` checks flag
  roles, resolved-path collisions (outputs against each other AND against the inputs), input
  existence and format, topology/System agreement, the MPI launch, the machine configuration,
  `--cpu`/`--device`, the device policy, platform availability and a real CUDA Context — before
  `mkdir`, `resolved.config`, the `.out`, the `.log`, `solute.yaml`, `_protocol.py`,
  a trajectory or a checkpoint exists. A `-odir` holding a `resolved.config` is indistinguishable
  from a run that happened, and a refusal must not touch an existing directory either.
* **`explicit_cpu` means the user typed `--cpu` on this invocation.** Nothing else. A machine
  configured for CPU is `platform_selection: machine-config`, and conflating the two makes a
  file somebody wrote months ago look like something they just typed.
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
* **Collective-variable reporting is OBSERVATION, and adds no Force.** v1 is torsions only, read
  from a strict `cv.yaml`. Nothing in `md_tools.cv` imports OpenMM — a test asserts it through the
  AST — and the evaluation is arithmetic over positions the reporting point already had. Adding a
  `CustomTorsionForce` would change the System's serialisation, its force groups and the
  checkpoint that records them, so a CV-enabled run would stop being the same experiment as a
  CV-disabled one and every comparison between them would be invalid. The suite asserts the
  System, force inventory, force groups and a single-point energy are *identical* with reporting
  on and off.
* **A CV cadence is independent of every other cadence, and must be exact.** It may be finer than
  the trajectory — that is the point of a separate series. `interval_steps` divides the cMD stage
  length, the REST2 `exchange_interval_steps`, and for AIS is a multiple of
  `parameter_update_interval_steps` *and* divides `switching_steps`. Step 0 and the final step
  appear exactly once **in every protocol, ladders included** — a ladder observes step 0 before a
  single step is propagated, with `exchange_attempt = -1`; minimisation produces no series, because its iterations have no timestep
  and a `time_ps` for them would be a fiction. An interval that does not divide is refused, never
  rounded: a final partial gap breaks the uniform spacing every downstream time-series analysis
  assumes and none can detect.
* **A CV series follows a thermodynamic STATE, as trajectories do.** `cv_state2.csv` holds
  whatever configuration occupied state 2, and `walker_index` says which walker supplied it. Rows
  on an exchange boundary are **pre-exchange** — the configuration as propagated, before any swap
  — so a state's series never contains values from a trajectory that never visited it. The
  convention is enforced structurally (`cv` precedes `exchange` in `EVENT_ORDER`) and written
  into every row and sidecar rather than left to be inferred.
* **An AIS CV row aligned to an observation was measured on exactly that saved coordinate.**
  `observation_index` and `coordinate_frame_index` are empty when the cadence does not coincide
  with one, never filled with a nearest neighbour. Attaching a CV measured at `x_j` to a
  different observation's coordinate is the same class of error the two-probe separation exists
  to prevent, and it is invisible in the output.
* **A REMD `trajectory_frame_index` names a frame holding exactly that row's configuration, and
  is otherwise EMPTY.** The state frame is written from the post-exchange occupant and the CV row
  describes the pre-exchange one, so at an accepted swap those are different configurations and
  no frame in that file holds what the row measured. The decision is therefore PER STATE at the
  same step. Never `-1`: that is not a frame index, and writing it invites a reader to index from
  the end of the file.
* **`simulation.currentStep` is the only absolute-step authority after a checkpoint restore.**
  OpenMM's `loadCheckpoint` restores the Context's step count, so nothing may add the
  already-completed count to it. Two step conventions in one runtime is the defect; an inert
  offset argument left behind is the next person's bug, so such parameters are removed rather
  than defaulted to zero.
* **Every appendable CV stream has a committed PREFIX in the checkpoint that vouches for it** --
  a row count AND a digest of exactly the header plus those rows, in the same generation
  transaction as the Context state. A count catches a truncation; it cannot catch a committed row
  edited in place, and a continuation appends onto exactly those rows. Never a digest of the whole
  file: after a crash the file is legitimately longer, and hashing the uncommitted tail would make
  every ordinary crash look like corruption. Ladder prefixes are per state, because a combined
  hash cannot say which file changed and would not notice two states' files being swapped. The
  prefix is validated -- digest, columns, finite values, grid, identifiers -- BEFORE a byte is
  truncated, and a checkpoint with no prefix record refuses with a compatibility message.
* **rREST2 is archived.** `protocol: rREST2`, `protocol = rREST2` and the `reservoir` section are
  refused by name, pointing at `archive/rREST2/` and the tag that holds the last working version.
  Nothing under `archive/` is packaged, imported or collected by pytest. Do not restore a piece of
  it into `src/` on its own: the reservoir refresh, its CV pre-refresh snapshot and its NetCDF
  variables were one design, and a partial return is a reservoir that opens and never draws.
* **A ladder's CV series are authoritative completed outputs.** `restart.json` records every
  state's digests, sizes, row count, header, grid, definition and conventions; completion is
  refused if any fails verification; `validate_replica_output` and extension-parent validation
  re-read them. A CV-disabled run records `collective_variables: null` explicitly, because
  omission is indistinguishable from a manifest predating the field.
* **A CPU test is never CUDA evidence.** The coverage matrix must name a test that runs the
  function on a device with the feature ENABLED. It once cited a `--cpu` file for the AIS CV
  path, which is worse than an empty cell: a gap invites work, a false entry closes the question.
* **Every CV series has a sidecar, and each shape is named in one place.** A stage's is
  `<key>.cv.json` beside `<key>.cv.csv` (`md_tools.cv.reporter.SIDECAR_SUFFIX`), where `<key>` is
  the name the stage is FILED under -- `eq_1`, not `eq_nvt_posres`, so the series sits beside the
  `eq_1.xml`, `solute_eq_1.nc` and `mdout_eq_1.csv` a reader has to pair it with; a ladder's is
  `cv_state<i>.json` beside `cv_state<i>.csv` (`md_tools.remd.cv_states`). Neither is the input
  `cv.yaml`, and neither is the content-addressed copy in the generated directory. The inventory
  once named a `.cv.yaml` that never existed, so the real sidecar was governed by no collision or
  overwrite policy at all — and it later named `remd<N>.cv.csv` for a ladder, which never existed
  either. A stale name in this file is not a typo: it is a contract nobody can check.
* **CV output failure is simulation failure.** Reporting is never silently disabled, and CV
  evaluation count and wall time are recorded under their own `cv_*` names — a position-only
  torsion is not an energy evaluation, and folding it into that total would corrupt the one
  number that says how expensive the Hamiltonian is.

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
