# Changelog

## 0.6.0 — 2026-09-18

Reusable ligand parameter packages, protonation you choose (PROPKA3 or OpenMM), mmCIF biological
assemblies and built missing atoms, and concurrent CUDA placed by measured throughput with MPS
verified rather than assumed. One key is RETIRED — `input.remove`, refused by name with the one-line
shell that replaces it, because deleting a crystallisation additive is an edit to your own file —
and one default moves (`solute.parameters` is now `search`). Full notes:
[docs/release-notes/v0.6.0.md](docs/release-notes/v0.6.0.md).

**`build-top --parameterize` writes a reusable ligand parameter package without building a
System.** `md-openmm build-top --parameterize -i MOL.{sdf,mol2,smi} --resname NAME -op DIR/NAME.pdb
-os DIR/NAME.xml -log LOG [--register]` is the command's third mode, beside the build and
`--rest2-scaler`. The input must carry bond orders — a `.smi` states the chemistry and the
conformer is embedded, a `.pdb` is refused because bond orders cannot be recovered from
coordinates. `--register` adds the package to `$MD_DATA/parameters/ligands`, which is what lets a
later configuration name it with no path; without it, a configuration points `parameter` straight
at the directory. Reuse never requires registration.

**AIS has a second switching schedule, `tau-linear`.** `ais.lambda_schedule: tau-linear` moves λ as
`[(1 − τ₀ + τ₀t)² − (1 − τ₀)²] / [1 − (1 − τ₀)²]`, so that with V0 a saved REST2 state at τ₀ and V1
its unscaled source the solute–solute scaling of the mixture follows `(1 − τ)²` along a τ linear in
time. The solute–environment scaling does not follow `(1 − τ)`, and the documentation says so.
`linear` stays the default and its λ values are unchanged. τ₀ is `ais.lambda_schedule_tau0`, filled
from `dynamics.tau` when the run generates its source; the run refuses tau-linear unless `-s` is the
saved state at τ₀ and `-s2` the System it was scaled from.

**`evenly_spaced` source frames cover the whole window.** The stride was rounded down, so 64 paths
from 95 eligible frames started from the first 64 and never the rest. The picks are now spread from
the first eligible frame to the last.

**AIS run identity is v3.** It records the schedule, τ₀ and a digest of the λ table. A v2 run
directory — any AIS run from 0.5.4 — is refused on continuation, because neither its λ values nor
its selected frames can be confirmed unchanged. Start a new `-odir`.

## 0.5.4 — 2026-09-17

**`build-top` takes a peptide as a sequence.** `-i ALA.seq`, where the file holds one line of
residue names (`ACE ALA NME`), builds the chain with tleap's `sequence` from the residue library
matching `forcefield.protein`, and then continues exactly as the `.pdb` peptide route, explicit or
implicit. The conformation is tleap's extended one, and the log says so. Only `solute.kind:
peptide` accepts it; a malformed file or a residue tleap does not know is refused before any
output exists. The record keeps the residues, the tleap commands and the digests of its log and
of the PDB it wrote.

**`solute.residue_name` is applied, and the prepared molecule is `<RESNAME>.sdf`.** A `.smi` or
`.sdf` molecule's residue used to be RDKit's `UNL` whatever the record said (backlog 18). The
stated or assigned name is now the residue name in `built.pdb`, `built.solute.pdb` and the
topology, and the molecule beside the System is written as `<RESNAME>.sdf` instead of `built.sdf`.
The run-time preflight reads either layout, so existing build directories keep working; build-top
refuses to write a new System beside an old `built.sdf`. A stated name that is not three letters or
digits, that already names water, an ion or a protein residue, or that is given for a peptide, is
refused.

**`--all-in-one` is retired, and its whole-chain preflight moved to generation.** The flag emitted
one `md.py` running every stage in one process instead of one script per stage, under identical
resolved settings, seeds, logs, checkpoints and restart semantics — so it bought a reader nothing
while every generated-run behaviour had to be built twice, and the split form was the one that got
tested. It was not a bundle that runs without MD-tools: its `md.py` imported `md_tools.md` like any
other generated script, and `export-reference` remains the command that produces something runnable
without this package. What it did carry alone was a preflight over the whole chain, and that is now
`build-md`'s: a chain whose last stage is invalid is refused **before a script exists**, rather than
after every earlier stage has run to completion. `md_tools.md.run_generated_workflow` is removed
with it. Use `./run.sh`, which drives the same stages in order.

**`build-md` requires the built System for every protocol.** A ladder already did, because its rungs
are scaled at build time; now every protocol does, because every chain is validated against that
System at generation. A cMD run used to generate in a bare directory and leave the masses behind a
timestep, the box behind an ensemble and the forces the scaling convention must place to whichever
stage first opened the System. A dataset root has one System and it must match the projects
generated in it, so an explicit-solvent project on a boxless one is refused rather than discovered
later. Run `build-top` into `build/` first.

**The generation-time validator is device-free, deliberately.** It resolves no platform, places no
device, opens no CUDA Context and consults no MPI world: generation happens on login nodes, in CI,
and on different machines from the run, so a device answered there would describe the wrong
computer. Those checks stay in the run-time preflight.

**The retired-command guard is retired.** `sys-config`, `sys-gen`, `md-gen`, `setup` and
`show-default` have been gone long enough that a test and a CI loop asserting they still fail were
upkeep with nothing behind them. The rule that no second executable is installed is separate, still
live, and still checked.

**AIS is a linear transformation between two topologies.** Until now AIS switched ONE System along
a REST2 `tau`, which made the potential quadratic in `1 − tau`, needed a three-point basis probe to
decompose the work, and could not switch explicit solvent without re-uploading every solute
parameter at every update. It now mixes two end-state Systems given as files,
`V(λ) = (1 − λ)·V0 + λ·V1` with λ running 0 → 1: V0 is `-s`/`-p`, the state the source ensemble was
sampled from, and V1 is the new `-s2`/`-p2`. The two must hold identical particles, masses,
constraints and force layout — parameters only, like sander's no-softcore mixing — and any other
difference is refused by name, together with a barostat in either. The work is
`ΔW_j = (λ_{j+1} − λ_j)·(V1 − V0)(x_j)`, exact for a linear path and one evaluation per switch;
every saved observation records `V0`, `V1` and `V(λ)` with the identity checked. Mechanism: shared
forces are added once and each differing force pair becomes a collective variable of one
`CustomCVForce`, so λ is a Context parameter for every System, explicit solvent included. Measured
on 6232 explicit-solvent particles on an RTX 3080: 1.40 ms per switch-and-step against 7.77 ms for
the tau switch.

A REST2 switch is now a pair of files, the scaled state and `build/built.xml`. `TauSwitcher`, the
global-parameter switching code, `md_tools.ais.decomposition` and `build_scaled_system`'s
`prepare_for_switching` are deleted; `ais.tau_start`, `ais.tau_end`, `ais.work_measurement` and
`ais.verify_every_updates` are refused with the migration, in a configuration and in an `.in`.
An `-odir`, checkpoint or manifest written by the single-topology AIS is refused rather than
continued. A stage's whole-system NetCDF now records `system_sha256` when the Hamiltonian it
integrated is `-s` unmodified, and AIS refuses a source whose recorded digest is not V0's.

**One promise changes.** A `CustomCVForce` cannot resume bit-for-bit on CUDA: its inner Contexts
keep atom-ordering state no checkpoint captures. A resumed path restores the committed generation
exactly and continues the same switching process as a new realisation; on CPU it still reproduces
an uninterrupted path exactly. Path-by-path results from 0.5.3 AIS are not comparable with 0.5.4:
the endpoints and ΔF are the same, the path — and so the work distribution — is not. Design and
measurements: `docs/amber-like-fix/AIS-two-topology.md`.

## 0.5.3 — 2026-09-16

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
the value is the same arithmetic rather than an approximation -- verified to 5e-11 reduced units at
double precision; at mixed precision neither path is bitwise stable, and the old save/restore round
trip perturbed the Context by ~100x the read-to-read noise floor, so the reuse disturbs the run LESS
than evaluating did -- and read before any cross energy so it is measured on the
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

**A generated tree is now a SYSTEM with RUNS beside it, not one flat `md_script/`.** `build/` holds
build-top's own output, `min/` the minimisation and `input/` the `.in` files — all three SHARED,
because every run on a system starts from the same built System, the same minimised coordinates and
the same instructions. Only what is genuinely per run goes into `<method>-run<N>/`: its `eq/`,
`remd<n>/`, `remd_records/`, `rank/`, `bundles/`, and a `run.config` carrying the seed and nothing
else. The sharing is enforced rather than assumed — a second configuration that resolves
`input/min.in` or `input/eq_<k>.in` differently is refused by name, since the runs already beside
that file read it, and two methods needing different equilibration are not comparable. `min/` is
first-writer-wins with the seed exempt, because minimisation draws no velocities. `md_tools.layout`
is the single authority for where every file goes, so the generator and the runtime read one answer
instead of two. `build-md` refuses a run directory that already exists rather than generating into
it, and `docs/run-layout.md` section 4 carries the migration for finished reference runs.

**A stage has two names, and no `run.sh` chain could complete because they disagreed.** A stage is
GENERATED as `eq_nvt_posres` — renamed to NVT spellings under implicit solvent or a scaled run, so a
pressure-coupled name never appears on a boxless one — and FILED as `eq_1`, by position. The stage
name decides the physics; the filing key decides the filename. `run.sh` chained `-c eq/eq_1.xml`
while the stage wrote `eq/eq_nvt_posres.xml`, so **no cMD or REST2 chain could complete through
`run.sh`**, the documented way to run one. `stage_plan` now stamps `file_key` on every plan entry
and is the one function every surface goes through; the plan is recomputed from `resolved.config` at
each execution rather than persisted, so an older generated tree gets the key too. The CV series
moved to the filing key (`eq_1.cv.csv`) as a contract edit, and `CLAUDE.md` and
`docs/collective_variables/README.md` moved with it.

**A ladder's rungs are scaled at build time.** Each rung is serialised to
`remd<n>/build_state<n>.xml` by `build-md` and recorded in `build_states.log`, so the Hamiltonian a
state ran under is a file that can be read rather than a derivation that has to be trusted.
`build-md` therefore needs `build/built.xml` to generate a ladder at all and refuses before writing
anything if it is absent. A group file's `-s` is consequently PER LINE, and `system` left
`HOMOGENEOUS_GROUP_FIELDS`; a stronger check replaced it — line *i* must name rung *i* and no two
lines may name one file, so a line carrying `remd2/build_state2.xml` against `--group-index 3` is
refused rather than run.

**`-s` or `-groupfile`, exactly one — a grouped launch was impossible through every public
surface.** A group file is the Amber-style description of a coordinated run (`-rem 3`, the only REMD
type this build supports): the per-replica INPUT commands live on its lines, and `-i`, `-p`, `-s`
and `-c` are validated THERE, not on the run-level command line, which carries only the outputs.
`-s` was `required=True` in four parsers sitting above the runtime's own grouped exemption. Eight
gates refused it in turn: argparse in `md-run` and in `replica_parser`; `_forward` splicing
`["-s", None]` into the delegated runner; `_common`'s particle check; the XOR itself, now refused BY
NAME and read-only in both surfaces; `preflight_ladder`, which resolves a System — and the starting
state — from the group file's first line rather than skipping the check, because skipping would have
dropped the HMR timestep refusal on every ladder; the provenance record, which now names
`group_file` and each `system_state<i>`; and the group file's own `-i`, which must be `_protocol.py`,
the field the executor imports as Python. `preflight_ladder` reading the group file's `-c` is what
makes `rest2.equilibration_per_tau` usable under `run.sh`: it refused a fully described launch with
"no -c was given" while every line named that state.

**A completed ladder recorded itself as failed.** `replica_main` built the executor's `-r` as
`args.restart or out / "restart.json"`, honouring what `run.sh` passes, but decided completion from
the hardcoded default. With `-r` given the two disagreed, so a ladder that had just finished wrote
`status: failed` with `failure_reason: "the executor returned 0"` while its own `.out` ended
`run_status: completed`. Registration reads the record, so every `run.sh`-driven ladder was
unregistrable, and two records of one run contradicted each other — which is what "completion is
read from a machine record, never from prose" exists to prevent.

**A refused `md-run` wrote into the tree it was declining to touch.** `resolved.config` records the
content-addressed CV and umbrella copies by BARE NAME so a directory stays movable, and the copy is
now written beside EACH declaration — the run root, the shared `input/`, `min/` and `eq/` — because
the name resolves beside whichever declaration a reader started from. Written to the run root alone,
`input/cMD.in` named a `cv.<digest>.yaml` that `input/` did not hold, so `md_tools.run.continuation`
could not load the definition, returned `None`, and the read-only boundary silently did nothing: a
REFUSED invocation wrote `resolved.config`, `<stage>.out` and `<stage>.log` into a protected tree.
Sharing one copy is safe by construction, because the name carries the digest.

**`run.sh` named two inputs that never existed.** The AIS branch typed `-i AIS.in` and the
all-in-one branch `-i cMD.in`, both bare basenames, while the inputs live in `../input/`. Both died
with `md-run: -i …: no such run input file`, and the all-in-one line named `cMD` literally, so an
all-in-one umbrella run asked for a file no generation has ever written. `test_md_run_inputs` pinned
only a SPLIT REST2 tree, which is why both survived two green fast lanes; it now pins the input line
by protocol and by shape.

**The sdist could not build a wheel from itself.** `pyproject.toml` declares
`build-backend = "_build_backend"` with `backend-path = ["."]`, but `MANIFEST.in` never shipped
`_build_backend.py`, so `python -m build` wrote the tarball and then failed with
`BackendUnavailable: Cannot find module '_build_backend'`. Pre-existing since at least 0.5.2. The
suite could not see it because `tests/wheel_build.py` builds from a copy of the working tree, where
the file is present — nothing anywhere built from the sdist, which is the artefact a release
publishes. A test now runs `python -m build` itself.

**Also in this work.** `min.in` carries no collective-variable cadence, since minimisation produces
no series and leaving it in made two runs differing only in CV reporting collide on a shared input.
The seed left the shared input and became the per-run `run.config`. `remd_records/` holds the
ladder's per-segment `.out`, `.log` and `restart_prod<N>.json` — a ledger, not a trajectory — so an
extension writes a `_prod2` set beside the first rather than over it. A configuration manual is
rendered from the schemas (`md_tools.build.manual`), with the shipped `configs/` reduced to minimal
examples. The README gained an end-to-end REST2 walkthrough, and `md-openmm --help` teaches the
layout.

**`build-top` takes a supplied 3D structure.** `-i` accepts a `.sdf` beside `.pdb` and `.smi`, for
a `ligand` or `peptide-like` solute. The difference from the SMILES route is where the coordinates
come from and nothing else: a `.smi` is embedded with ETKDGv3 and MMFF-minimised, while a `.sdf`
is used **as given**, so a docked or crystallographic pose survives instead of being silently
replaced by an MMFF minimum. `initial_structure_from_sdf` writes the same two files as
`initial_structure` and returns the same keys, so every step after preparation is identical; there
is no seed to record, and the input's sha256 takes its place in the provenance. Refused by name,
before anything is created: a two-dimensional conformer, missing explicit hydrogens, more than one
molecule record, and a file RDKit cannot open at all.

**An explicit-solvent ligand build wrote no `built.sdf`, and reported success.** Found while
adding the above. `initial_structure` was called with the staging root on the explicit route and
with `staging/structure` on the implicit one, while `build-top` copies `built.sdf` out of the
latter — so on the explicit route the file was written where nothing reads it and none was
published. Bond orders are not recoverable from a topology and three consumers need them
(`classify_omega_bonds`'s ligand route, `map_from_sdf`, `preflight._ligand_sdf_beside`), so an
explicit-solvent ladder over such a solute had nothing to perceive amides from. It survived
because every test asserting `built.sdf` ran under GBn2 while the suite's explicit ligand builds
asserted radii and records instead — disjoint sets. This is a candidate cause of the omega item
below on the explicit path. Also corrected: the kind × suffix rules ran *below* the output
`mkdir`, so every refusal of them created the output directory first; they now run above it and a
refused build leaves nothing behind. `tests/test_build_top_input_formats.py` pins the nine-cell
matrix, the SDF refusals and `built.sdf` under both solvents, none of which had a test before.
`docs/backlog.md` entry 16.

**Deferred to 0.5.4, by decision.** Three items are known and deliberately not addressed here.
*The omega exclusion*, where the case that matters is a solute whose `kind` is `peptide` or
`peptide-like`: `classify_omega_bonds` offers a residue-aware `peptide` route and a bond-order
`ligand` route, while `peptide-like` is built through the whole-molecule ligand route and fits
neither cleanly. Its own docstring already states the rule to enforce — "a non-empty unclassified
list must block production" — and nothing consumes `omega_unclassified_candidates`. *The `.in`
file's consistency with Amber's own input conventions*, which has not been audited term by term;
the surface is Amber-*like* by design, and which divergences are deliberate is not written down.
*The AIS implementation*, unchanged here and analysed but explicitly unimplemented in
`docs/amber-like-fix/AIS.md`; the intended direction is an Amber-style alchemical transformation
between two topologies rather than the present single-topology tau scaling, which is a design
change rather than a fix. `docs/backlog.md` entries 13–15. Two smaller items are filed rather than
changed — `--all-in-one` writes its artefacts flat in the run root while the
declarations sit in `eq/` (`docs/backlog.md` entry 12), and two runs on one system that differ only
in CV reporting cannot share a root, because the cadence lives in `eq_*.in`.

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
