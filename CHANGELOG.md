# Changelog

## 0.5.3 — 2026-09-13

**An extension can run the group file an extension writes.** `--extend-from` takes the physical
state from the parent's checkpoint and so writes no `-c`; the parser required `coordinates` on
every group line regardless, and every rank aborted with `no coordinates given`. That stopped an
explicit-solvent ladder at 100 ns/state of the 500 it was asked for. Coordinates are now required
only when an extension is not in force, the refusal says so, and a test parses the exact shape
`--extend-from` emits — nothing had exercised the generated file against the parser that reads it.

**An interrupted extension is redone, and now says so.** An extension segment is atomic, and
atomic *by omission*: no resume path knows `extends` exists, so `--resume` on a segment's own
directory would have continued it and written a `restart.json` with no `extends` block — parent
pinning, segment-local counts and chain accounting silently gone — while `--resume` with
`--extend-from` is refused as two different operations, which reads as a flag complaint rather
than an answer. Nothing refused the dangerous one and nothing documented either, so the obvious
move after an interruption was the wrong one; a campaign lost two hours of six-GPU time to it.
The `interrupted` run-state note, the driver's printed line, the `DriverError` on a short run and
the executor's stderr now all distinguish an in-place run from a segment and name re-running the
segment with `--extend-from` into a **fresh** directory. `tests/test_extension_directory.py`
interrupts a real extension after a committed checkpoint and asserts the advice, and the two dead
ends are tests rather than folklore. The behaviour is unchanged — this is what it always did, now
stated.

**`-o`, `-log`, `-r` and `-chk` default by segment.** They come from the command line, and `SimOut`
opens its file `"w"`, so a second in-place segment destroyed the first segment's output and
provenance record while `mdout_prod2.csv` and `solute_prod2.nc` sat beside them. `stage_artifact_name`
applies `info_csv_name`'s rule to all four; an explicitly passed path still wins.

**A remedy message named flags `md-run` does not define.** The 0.5.1 test that polices this omitted
`remd/executor.py` from its file list and captured only the first flag after "pass". Widening both
found a live instance offering `--extend` and `--force` to `md-run` users.

**A ladder can carry torsion restraints.** `umbrella.file` is accepted with `protocol: REST2` and
`rREST2`, meaning the SAME restraints on every rung, resolved against `collective_variables.file`.
They are added after the REST2 scaling and never scaled by tau, which is what makes the bias cancel
from the exchange criterion exactly: `u_i(x)` and `u_j(x)` both contain `W(x)`, so `log alpha` is
the number it would have been unbiased. Per-rung variation is not offered. `restart.json` records
the definition and every resolved restraint under `scientific_identity.torsion_restraints`, and
`export-reference` refuses such a ladder by name, because a bundle's `verify_rungs.py` rebuilds
each rung by scaling rung 0 and a restraint is not part of what that reconstructs. Tested by
arithmetic: `log alpha` with and without the bias at the same configurations, equal to 1e-6 kJ/mol,
with a counter-example proving a bias that differs between rungs does not cancel.

**A stage's `.out` now says what ran.** Amber's `mdout` opens with a topology census and a full
echo of every resolved control variable, which is why an Amber run can nearly be reconstructed from
its own output. This wrote the inputs, the step count and the platform: the cutoff, the PME
treatment, the constraint count, the net charge and the size of the restrained selection all
existed — in `resolved.config`, in the machine record, in `solute.yaml` — and a reader had to open
three files to assemble them. A stage `.out` now carries `System` (atoms, residues by name, net
charge, degrees of freedom, box and volume), `Method` (nonbonded treatment and cutoff, Ewald
tolerance, dispersion correction, switching, 1-4 exception count, constraints, barostat present,
force inventory) and `Selections` (the solute count, and the omega bonds left unscaled). All
derived from the serialised System, never from the configuration that asked for it: a config
claiming a 1.0 nm cutoff and a System carrying 0.8 nm are different runs and only one integrates.
The same facts go into the `.log` as structured fields, because prose is not a database.

A box is reported only when the System is actually PERIODIC. `getDefaultPeriodicBoxVectors` cannot
answer "is there a box" — an OpenMM System defaults to a 2 nm cube — so an implicit-solvent stage
would have printed `2.000 x 2.000 x 2.000 nm` and a volume of 8 nm³, invented and stated in the
same voice as the measurement above it.

**An rms fluctuation over one sample is not zero, it is absent.** `sqrt(<x²> - <x>²)` over a single
report is exactly 0, which reads as "this quantity did not move" when it means "there was nothing
to compare it against" — and a stage shorter than `state_interval_steps` produces exactly one row,
so this was the common case. It now says `n/a (single sample)`.

**A ladder's `.out` names the Hamiltonian it integrated.** The grouped summary now prints the τ
ladder, the scaling laws in force, which terms were left unscaled, the solute region with its
omega-bond exclusions, and the rung's `system_sha256` — plus a `TIMINGS` block with elapsed time,
ns/day per replica and aggregate, and ms/step. Amber prints its REAF equivalent but cannot name the
resulting Hamiltonian, because `gti_add_re=6` is an index into a table in the manual and the scaled
potential exists only inside the binary. A rung here IS a serialised System, so it can be named and
digested.

**`energy_components.csv` is a decomposition on every build route, not just one.** OpenMM can only
separate energies by force group, and the two routes disagree: ParmEd's `createSystem` (implicit)
assigns bonds 0, angles 1, torsions 2, nonbonded and GB 11, while OpenMM's `ForceField.createSystem`
(explicit) leaves every force in group 0. So an implicit run decomposed and an explicit one produced
a single column holding the total potential energy under a joined name that promised a breakdown —
with nothing in the output distinguishing the two. 31 such files exist under `$MD_DATA`. Where the
System carries no groups, a group-separated COPY is now probed instead; `built.xml`, the production
Context and every digest taken from them are untouched, because a force group is part of the
serialised System and regrouping the run's own would change `system_sha256` and invalidate every
checkpoint fingerprint in flight. Bonds, angles, torsions and nonbonded direct space come apart
cleanly and PME reciprocal splits out for free. `EELEC` from `VDWAALS`, and the 1-4 terms from
either, do NOT: all 7988 1-4 pairs are exceptions inside the single `NonbondedForce`, which
evaluates charge and dispersion in one kernel. That split needs duplicated forces and belongs in
post-hoc analysis, where nothing integrates.

**A ladder no longer recomputes the energy it already had.** `u[i][state_to_walker[i]]` is the
energy of the configuration state *i*'s Context is already holding — `_gather_configurations` read
it out of that very Context — so installing it again to measure it was a round trip to the device
for a number in hand. It is now read directly, through the same `reduced_potential` conversion, so
the value is identical rather than close, and read before any cross energy so it is measured on the
Context as propagation left it. This is the value Amber takes for free as
`my_ene_temp%energy_1`, paying one force call only for the cross term it calls "my pot ene with
THEIR coordinates".

Rules may now declare which reduced potentials they will read, through `required_entries` —
honouring a promise `ExchangeContext` already made ("a callable rather than a precomputed matrix so
a rule that needs only four numbers pays for four, not for N squared"). The driver records the
declaration and deliberately still evaluates every entry: `rem_log.block_rows` indexes
`u[state, state]` and `u[state, mate]`, `exchange_free_energies` needs both cross terms of every
proposed pair, and `exchange.csv` indexes `u[state][walker]` — two conventions over three
renderers, so a sparse matrix would leave holes that some of them index and a hole reaches the
reader as `nan` in `rem.log`.

**`u_evaluated` is observed rather than asserted.** It documents "1 where u was actually computed"
and was written as `np.ones(...)`: true, but a claim that cannot become false cannot catch the day
it stops being true. It is now derived from the matrix.

**Output names in the documentation matched no file.** `remd0.nc .. remdN.nc` and
`remd<N>.cv.csv` appeared in CLAUDE.md, all four shipped `configs/md/*.config`, `build/md.py`'s
`state_trajectory` help, `storage.py`, `preflight.py`, two published docs and two `example.in`
files — and nothing has ever written either. A ladder writes `solute_state<i>_prod<N>.nc` and
`whole_state<i>_prod<N>.nc` (`state_trajectory_name` is the one authority) and its CV series are
`cv_state<i>.csv` with `cv_state<i>.json` sidecars; `<stage>.cv.json` is the STAGE sidecar and was
being quoted as though it covered both shapes. Corrected everywhere it is a live instruction, and
left alone in `docs/history/**`, `docs/release-notes/**` and `docs/reports/**`, which record what
was true when written. One test asserted that `remd0.nc.1` does not exist — vacuously true of a
name nothing writes — and now guards against a second SEGMENT (`..._prod2.nc`) appearing, which is
how an in-place `--extend` could actually break its contract.

**One docstring overstated a refusal.** `reduced_potential_of` said a cross energy is never
inferred from another because the Hamiltonians differ by more than a single factor. True of
rescaling a total, and the method still evaluates directly — but as written it denies the exact
three-term identity `md_tools.ais.decomposition` is built on, and would talk the next reader out of
a sound optimisation. It now says which shortcut is wrong and which is arithmetic, and names the
real constraint: under PME with the long-range dispersion correction the scaler declines global
switching altogether, because the tail term is computed from stored epsilons and does not follow a
parameter offset.

## 0.5.2 — 2026-09-11

**A finished run can leave this package behind.** `md-openmm export-reference -idata <run> -odir
<out>` writes a directory that runs on OpenMM alone — the System that was integrated, the topology,
the state the stage continued from, a standalone runner, provenance and a `SHA256SUMS` inventory.
Nothing in it imports `md_tools`, enforced by running the bundle under an import hook rather than
by reading the source. cMD and REST2; umbrella and AIS are not done.

A REST2 bundle does not reimplement the ladder. The modules that decide what happens — the
acceptance criterion, the sweep schedule, the reduced potential, the seed derivation — are copied
byte for byte, which took moving `BAR_NM3_TO_KJ_PER_MOL` and `driver._stream_seed` into
`remd/core.py`. Checked against the engine's own run: 10 of 10 exchanges with an identical
state-to-walker mapping, not merely a similar acceptance rate. Its `md_tools_commit` is the engine
that ran, as recorded; the commit the modules were copied from is `ladder_modules_from`. The rung
construction travels too: `rest2/hamiltonian.py` (OpenMM only, the one implementation) is copied
into the bundle, the solute atoms and omega bonds are recorded, and `verify_rungs.py` rebuilds
every rung from rung 0 and requires an identical System. The runner also performs the ladder's
per-state `equilibration_steps`, which it used to skip; a ladder with that setting above zero is
now reproduced exchange for exchange, and tested so.

**Every bundle carries `input/`**: the structure, the build-top configuration, the run's
`resolved.config`, the built System and topology every stage ran on, and each stage's `.in` -- each
proven against the run's own records before it is copied. `input/build_system.py` rebuilds the
System from the structure with OpenMM and its chemistry libraries alone, no md-tools, and checks it
is the same bytes; a test holds that for ALA and phenol in implicit and explicit solvent. A
flexible molecule charged with AM1-BCC may not rebuild identically (backlog entry 6: OpenFF
generates the charge conformer unseeded), and the script then reports the difference; such a
bundle's README says why. `export-reference -idata .` works: both paths are resolved before
anything is searched. The README gives four ways to reproduce a run, starting with OpenMM alone, and says where
the structure came from -- for a capped peptide, the tleap `sequence` that writes it, checked by
running tleap at export.

**A ladder can equilibrate every rung under its own tau.** `rest2.equilibration_per_tau: true`
(REST2/rREST2, off by default) stops the tau = 0 chain at minimisation under implicit solvent, or
after its NPT stages under explicit solvent, which fix the box every rung shares; every rung, tau = 0
included, then runs `eq_nvt_posres`, `eq_nvt_posres_2` and `eq_nvt_free` under its own tau at fixed
volume, then `equilibration_steps`, then exchanges. The stages are done as the stage chain does them
-- a fresh integrator per stage seeded per rung, velocities carried, the chain's restraint set after
the configuration -- on a restrained copy of each rung, so the propagated rungs are unchanged. Each
rung's end state is kept and verified into `restart.json`; an interruption during it is refused by
`--resume` by name. A reference bundle carries the same code (`ladder/rung_equilibration.py`) and
reproduces such a ladder exchange for exchange. Off, every generated script, `run.sh`,
`_protocol.py` and the resume identity are byte-identical to before; each `.in` gains one line.

**Test datasets.** `docs/campaigns/test-systems-2026-09/` builds, runs, exports and registers ALA
and phenol, implicit and explicit, cold and hot cMD and REST2, 1 ns each, under
`$MD_DATA/2026/md-tools/<system>-test/`.

The first version of the cMD exporter carried the **build** System rather than the integrated one
— a different Hamiltonian at non-zero tau — plus the config seed instead of the derived one, the
built coordinates instead of the continuation state, and a restraint left on by `setState`. All
four ran perfectly and none was visible from reading the script. The test did not catch them
because it compared the export against a Context built in the test file, sharing all of its
assumptions. It compares against the engine now.

**A wheel knows its own commit.** `md_tools_commit` was `null` everywhere, because a wheel is built
from a directory and nothing in the build consults git. An in-tree PEP 517 backend bakes it at
build time; a dirty tree bakes nothing, since a commit that does not describe the wheel is worse
than no commit.

**Shared datasets leave the year.** `--common-data` registers to `common/{project}/{data}` rather
than `{year}/common/{project}/{data}`: a reference is used for as long as it is the best one
available, and a year segment means finding it requires knowing when it was made. `data_name` may
be several segments deep, each validated separately.

**`origin` named the wrong repository.** It ran git with no `-C`, so it recorded whichever
repository the person was standing in — three datasets claimed MD-tools' HEAD for another project's
campaign, passing both guards on the way. It resolves from `-idata` now and refuses when the data
are not in a repository, naming `--project-repo`. New `--notes TEXT`.

**The environment is `openmm-env`.** `environment-cuda.yml` creates `openmm-env` and the CPU
file `openmm-env-ci`; the README installs to `envs/openmm-env`. `md-openmm` is the command the
environment provides, and naming the environment after it made "install md-openmm" and "run
md-openmm" sentences about different objects.

**The development history and the ALA reference campaign live here now**, under `docs/history/`
and `docs/campaigns/ala-2026-09/`, copied from the project repository where that work was done.
Four tests cited "the MD-project journal" as their real-run evidence; they cite a file in this
repository instead, and nothing here refers to another repository's working tree.

**Three more tests that failed for reasons outside the code.** Two counted the machine's GPUs
with `nvidia-smi`, which ignores `CUDA_VISIBLE_DEVICES`: on a suite confined to five of nine cards
one asked for device 8, and a six-state ladder passed its own "needs six devices" guard and failed
later with two ranks on one card. They count the devices the process can use now. The third read a
directory another test created, and its module built the wheel inside the checkout, where parallel
workers collided and the collision was reported as a skip. Behind all of them, `conftest.py` wrote
each test's GPU assignment into the worker's environment and never took it back, so a test given
no assignment inherited the previous test's card; it now resets before every test.

**Three tests were not running and reported it as a fact about the software** — a stale filename
(`cMD.csv` for what is now `mdout.csv`) that had never once been satisfied, two tests reading
another test's output, and an interrupt driven by a 45-second timer. One skip remains, an opt-in
evidence writer. `--dist loadgroup` is now the default so an expensive module-scoped fixture is
built once rather than once per worker.

## 0.5.1 — 2026-09-10

**An interrupted CV-enabled REST2 ladder can be resumed.** It could not be, on any ladder whose tau
was not exactly representable at six decimal places — four, eight or twelve rungs. Three
implementations of one linear tau ladder disagreed in the seventh decimal: the ladder that RUNS
rounds to six places and its values are what a generated `_protocol.py` executes and what every
`cv_stateN.json` records, while the resume check recomputed an unrounded one and compared with a
1e-12 tolerance. The resume was refused for a serialisation artefact, with the data intact.

There is one implementation now. `run/continuation.py`'s inline copy is gone and
`rest2.linear_tau_ladder` delegates rather than rounding to match — two implementations that agree
are what produced this.

Found by interrupting a real four-rung ladder at step 2,610,250 of 5,000,000 and resuming it: the
deferred integration experiment from `docs/release-notes/20260907-readiness.md`. After the fix the
same run completed with 20001 of 20001 rows on every state, no gap and no duplicate.

Ladders of two, three, five or six rungs were unaffected, as was any ladder with CV reporting off.


## 0.5.0 — 2026-09-10

**Breaking.** MD-templates became MD-tools: a standalone, pip-installable package with one
executable and three commands. Every detail, with the tests that verify it, is in
[`docs/release-notes/v0.5.0.md`](docs/release-notes/v0.5.0.md).

### The interface

- one installed executable, `md-openmm`, with exactly `build-top`, `build-md` and `data-register`
- AIS is a `build-md` protocol, not a fourth command
- **removed**: `sys-config`, `sys-gen`, `md-gen`, `setup`, `show-default`, the `openmm-md`
  executable, and the `md-template` environment installer. `pip install md-tools` replaces the
  installer; nothing replaces the rest, because the three commands cover what they did.

### Torsion collective variables

- new `collective_variables: {file, interval_steps}` section on every MD protocol, off by
  default; supplying only one of the two keys is an error rather than a guess
- a strict `cv.yaml` (v1: torsions only) naming atoms by zero-based index or by
  chain/residue/atom selector; ambiguous selectors, duplicate names, unknown fields, duplicate
  YAML keys and unsupported types are all refused rather than resolved
- **observation only**: no `Force` is added, and `md_tools.cv` imports no OpenMM at all. The suite
  asserts that enabling reporting leaves the System serialisation, force inventory, force groups
  and single-point energy identical
- degrees, wrapped to `[-180, 180)`, IUPAC/MDTraj sign, triclinic minimum image applied to the
  three sequential bond vectors
- a cadence independent of the trajectory and state-data intervals and possibly finer, required
  to divide exactly: the cMD stage length, the REST2/rREST2 exchange interval, and for AIS to sit
  on the parameter-update grid *and* divide `switching_steps`
- one CSV per cMD stage, per REST2/rREST2 **thermodynamic state** (with `walker_index` and a
  documented **pre-exchange** boundary convention), and per AIS path plus a verified-manifest-only
  `AIS_cv.csv` aggregate; each with a JSON sidecar carrying units, conventions, digest and
  resolved indices
- the definition is resolved against the MD configuration file and copied content-addressed into
  the generated directory, so a generated tree stays movable; on resume the series is truncated to
  the checkpoint's committed count before appending
- CV evaluation count and wall time are recorded under `cv_*`, never merged into energy evaluations

See [`docs/collective_variables/README.md`](docs/collective_variables/README.md).

### Configuration

- `configs/{machine,sys,md}/` are ordinary browsable files at the repository root, one copy each,
  shipped as wheel data files and located through distribution metadata
- every length is an exact integer step count; logs derive ps/ns
- examples must resolve, through the real resolver, to the model's own defaults

### The dataset contract

- contract **v2**, owned by MD-tools: `$MD_DATA/{year}/{project}/{data}/`, no month segment
- `extension.yaml` ported with a newly required checkpoint digest
- `dataset-v2.0` and `extension-v2.0` schemas generated from the models and drift-checked
- **`md_data` is no longer imported at runtime**, and contract v1 is gone

### A stable import API, and generated files that are entry points

- `md_tools.md` (`PositionalRestraint`, `ReportingConfig`, `run_stage`, `run_generated_stage`,
  `run_generated_workflow`), `md_tools.rest2` (`REST2Scaler`, `ScalingSelection`), `md_tools.remd`
  (`REMDRunner`, `NeighborExchangeRule`, `run_remd`, `run_generated_remd`),
  `md_tools.remd.reservoir` (`ReservoirRefreshRule`) and `md_tools.ais` (`run_generated_ais`)
- generated Python files are now compact entry points that call those APIs; `resolved.config`
  beside them is the single resolved declaration, found from `__file__`, re-validated at execution
  and bound into the checkpoint fingerprint
- **one REST2 scaler** for fixed-τ cMD, REST2, rREST2 and AIS
- `openmm/templates/` removed: it held installed runtime code, not templates. `md_tools.runtime`
  remains as compatibility-only facades so pre-v0.5 generated scripts still run

### Scientific options

- box shape: `cube` and `octahedron` documented alongside the `dodecahedron` default
- ligand force field: GAFF available and **resolved to an exact installed version**
- crossed protein/water pairs warn instead of refusing, and the warning is recorded
- `hydrogen_mass_repartitioning: {enabled, hydrogen_mass_amu}` replaces the null-means-off scalar
- `dynamics.timestep_fs: auto` resolves from the masses in the built System

### One current MD configuration model

- `md_tools.build.md` is the only authority for MD workflow configuration. The retired `methods:`
  model — `md_defaults`, `ais_defaults`, a second `resolve_md_config` and a second `stage_plan`,
  in `duration_ns` and `switching_duration_ps` — is gone from `openmm/`
- `openmm/config.py` → `openmm/system_config.py` (the `build-top` system resolver),
  `openmm/defaults.py` → `openmm/system_defaults.py`, `write_yaml` → `openmm/yaml_io.py`;
  `openmm/stages.py` deleted, having had no importer
- `templates/openmm_md.py` → `templates/replica_executor.py`: an internal module must not be named
  after a retired executable
- one `ConfigError`, not two, so a refusal from the builders is the one the CLI handles

### Fixed

- `src/md_tools/build/` — the package implementing `build-top` and `build-md` — had never been
  committed, silently excluded by an unanchored `build/` ignore rule. A clone of this repository
  did not contain two of its three commands.
- a stage now records the parent state it continued from **with its digest**, so a parent rewritten
  after a child consumed it is caught rather than becoming false ancestry
- a 4 fs timestep is refused unless the masses in the System prove hydrogen mass repartitioning

---

Earlier releases are in Git history. The state immediately before the v0.5.0 cleanup is preserved
at the tag `pre-v0.5-doc-cleanup`; this file no longer carries hundreds of lines about commands and
an installer that no longer exist.
