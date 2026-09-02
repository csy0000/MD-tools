# `md-openmm md-run` — running what `build-md` resolved

`md-run` executes a stage, a REST2/rREST2 ladder, or a set of AIS switching paths from a short
Amber-like input file. If you have run `pmemd -i mdin -p prmtop -c inpcrd -o mdout -r restrt`, you
can read every command on this page without a manual, which is the whole reason it exists.

```bash
md-openmm md-run -i min.in -p built.pdb -s built.xml -c prev.xml \
          -o min.out -x min.dcd -r min.xml -log min.log

mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \
          -c eq_npt_free.xml -o REST2.out -x REST2.nc -r restart.json -log REST2.log

mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \
          -source-traj ../cMD_tau0p5/tau_0p5.dcd -o AIS.out -log AIS.log -odir ./AIS
```

It is a **surface**, not a second implementation. Every protocol is handed to the same function a
generated script calls — `stage_main`, `replica_main`, `ais_main` — so `python min.py` and
`md-openmm md-run -i min.in` are the same run, and a correction in the installed package reaches
both. There is no behaviour reachable from one and not the other.

## The flags

| flag | meaning | Amber |
|---|---|---|
| `-i` | the run input: `&cntrl` / `&remd` / `&AIS` sections | `-i mdin` |
| `-o` | human-readable simulation output | `-o mdout` |
| `-p` | topology and reference coordinates, `built.pdb` | `-p prmtop` |
| `-c` | starting state — the previous stage's final state | `-c inpcrd` / `restrt` |
| `-r` | output final state, the handoff to the next stage | `-r restrt` |
| `-x` | output trajectory | `-x mdcrd` |
| `-s` | the serialised OpenMM System, `built.xml` | — |
| `-log` | the provenance record | — |
| `-chk` | output checkpoint, written periodically for resume | — |
| `-odir` | where outputs and `resolved.config` are written | — |
| `-ng` | how many replicas or workers this launch coordinates | `-ng` |
| `-groupfile` | an Amber-style group file, for a heterogeneous ladder | `-groupfile` |
| `-source-traj` | AIS only: the equilibrium ensemble the paths start from | — |
| `--cpu` | run this invocation on the CPU — see the platform policy below | — |
| `--device` | which CUDA device: placement, never platform | — |

**`-x` is the trajectory and `-s` is the System.** Amber's `prmtop` carries the topology *and* the
parameters; here they are two files, and `-s` is the one that holds the physics. `-s` has no Amber
counterpart, which is exactly why it gets the flag Amber does not use — `-x` has meant mdcrd for
decades, and briefly using it for `built.xml` here trained people to type the one command that
would write a trajectory over the System their run needs. Both mistakes are refused by name.

`-s` is required. `-x` is optional: a stage defaults to `<stage>.dcd`, and AIS writes
`AIS_trajNNNN.nc`, one per path, which no single path could name — pass `-odir` instead.

Flag abbreviation is off. `--traj` fails rather than quietly becoming `--trajectory`: a misspelling
argparse resolves is worse than one that errors, because it runs, with a setting nobody wrote.

`-h` works on a machine with no GPU, no driver and no OpenMM Context, because that is where people
read it before submitting a job.

## Two outputs, two readers

`-o` and `-log` are different files and are never merged.

| | `-o` (`<name>.out`) | `-log` (`<name>.log`) |
|---|---|---|
| for | a person, during the run | a machine, afterwards |
| holds | stage, settings, progress, energies, temperature, how it ended | resolved configuration identity, input/output hashes, software and hardware, warnings, status, MD-data-contract fields |
| read by | `tail -f` | `md_tools.build.record.read_record` |

They were briefly one file, on the argument that this package writes a single readable record.
That made a person open a machine record and scroll past a hundred lines of hashes to find out
whether a simulation was moving. Amber has kept `mdout` and its logfile apart for the same reason.

Each names the other, so whichever you open first tells you where the rest is. Only an actual
collision — the two resolving to the same path — is refused.

## Which file drives the run

**`resolved.config`. Always — and md-run writes it on every invocation.**

The `.in` file is an *input*: a short, partial, human-written statement of intent. Every
invocation parses it and resolves it through `md_tools.build.md`, the one authority for MD
workflow configuration, producing a document with every default written out and every cross-field
rule applied. That resolved document is what the run reads, what the checkpoint fingerprint binds,
and what the log records; it is written into `-odir` with the `.in` file's sha256 in its header,
before anything integrates:

```text
# The configuration this run resolved to, in full. THIS FILE IS AUTHORITATIVE:
#
#   input      : REST2.in
#   sha256     : 3f0c…
#   sections   : &cntrl, &remd
```

Two consequences worth stating plainly:

* editing the `.in` file changes the NEXT invocation, because the next invocation reads it again.
  It does not change a run that already happened;
* `resolved.config` in `-odir` is the record of what a particular run resolved to, and a
  continuation whose resolution differs is refused. Editing it by hand changes the fingerprint and
  the continuation stops, which is the intent.

If `-odir` already holds a `resolved.config` describing a *different* run, md-run refuses rather
than overwriting: outputs from one resolution beside the configuration of another cannot be read
correctly afterwards. An identical one is left alone, so rerunning a command is safe.

**`--overwrite` governs the COMPLETE output inventory**, not `resolved.config` alone. That was the
old behaviour and it meant every other file — the reports, the trajectories, the phase-space
stream, the state tables, the restarts, the per-state `remdN.nc`, the generated helpers — was
replaced silently whether it was asked for or not. The preflight now names every artefact a run
will create, before any of it exists, and refuses a `-odir` that already holds a run unless one of
three things is true:

* the run there **completed** under this exact configuration, and its outputs verify — then it is
  skipped, and rerunning a command is still safe;
* the run there was **interrupted** — a committed checkpoint short of the stage's step count —
  then it continues, which is what `--resume` means and what happens by default;
* you passed **`--overwrite`**, which replaces all of it.

Anything else is half a run nobody claimed, and it has to be answered for rather than written
into. `--resume` and `--overwrite` are forwarded to the runtimes, so a generated script and
`md-openmm md-run` mean the same thing by them.

**`--check` creates nothing at all**, not even `-odir`. It used to make the directory, both
reports and (for AIS) `selected_source_frames.csv`, then say nothing had been run — leaving
exactly the directory whose absence the caller was asking about.

Every `.in` that `build-md` writes resolves back to exactly the `resolved.config` beside it. That
is a test over all four protocols and every generated input, not a convention.

## The input files

Namelist-*shaped*, deliberately not an Amber parser:

```text
&cntrl
  ! minimisation of the built system
  stage                   = min,
  protocol                = cMD,
  minimization_iterations = 1000,
  timestep_fs             = auto,
  temperature_K           = 300.0,
/
```

Strict, on purpose. A run input is read once, by a machine, and every mistake it can contain is
cheaper to refuse than to discover in the output — none of them fail at run time, they produce a
run with a setting that did nothing:

* an **unknown key** is refused by name, with the closest known key suggested;
* a **duplicate key** or a **repeated section** is refused; the last value never silently wins;
* a **key in the wrong units** — `timestep = 2` where `timestep_fs` was meant — is refused with
  the right key named. The unit is part of the name here precisely so that a bare `timestep`
  cannot be written;
* an **Amber control with a counterpart here** — `nstlim`, `ntpr`, `ntwx`, `gamma_ln`, `ig` — is
  refused with that counterpart named, because people arrive from `pmemd` and refusing without
  saying what to write instead wastes their time;
* **booleans have one spelling set**, `true` and `false`. `.true.`, `yes` and `1` are refused: a
  configuration language with three spellings for true has three ways to typo it;
* **every length is an integer step count**. Durations in picoseconds are derived for the reader by
  the logs, never stored.

Parsing and every semantic check finish before an OpenMM Context or an output file is created.

The parser defines **no defaults**. It projects onto the schema that owns them and resolves there;
two files that each decided what "the default timestep" is would eventually disagree, and the one
that lost would be invisible.

## The platform is a property of the machine

```text
md-openmm md-run ≈ pmemd.cuda
```

It is configured once per machine, in the user configuration, and not in any protocol:

```yaml
machine:
  md_data: /absolute/path/to/MD_DATA
  openmm:
    platform: CUDA        # or CPU, for a deliberate machine-wide CPU default
    precision: mixed
    device_policy: local_rank
```

found through the order that already existed — `--user-config`, `$MD_TOOLS_CONFIG`,
`${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config` — and defaulting to CUDA, mixed precision
and local-rank device placement when the file or the `openmm:` block is absent. A configuration
written before this block existed is still valid. `md-openmm data-register --init` writes it out.

`dynamics.platform` in a protocol configuration is **retired** and refused with this migration. A
protocol is the same experiment wherever it runs; a workflow that carried `platform: CUDA` carried
one machine's hardware into every repository it was shared through.

**There is no automatic fallback, in either direction.** A run that quietly moved to the CPU still
finishes, still writes a trajectory and still reports success — two orders of magnitude later, on a
machine whose GPU was simply not visible. The result is not obviously wrong, which is what makes it
expensive: it is found weeks later, if at all. CUDA that cannot be initialised is an error
**before** dynamics; listing the platform is not the same as having a usable device, since
conda-forge ships the plugin unconditionally, so the resolver opens and discards a one-particle
Context to prove it.

The CPU is reachable two ways, and both are choices somebody made: `machine.openmm.platform: CPU`
for a machine with no GPU, and `--cpu` for one invocation. There is no `--platform` — a per-run
platform flag would be a second authority for a machine property, and the two would disagree the
first time somebody scripted one and configured the other. `--device` says *which* GPU, never
*whether*, and is refused together with `--cpu`.

`device_policy` decides where a rank's Context goes, and both values do something:

| value | behaviour |
|---|---|
| `local_rank` | one rank per visible device. The only policy that keeps a ladder off a single GPU |
| `openmm` | set no `DeviceIndex` and let OpenMM choose — right when a scheduler or MPS has already partitioned the GPUs |

**An absent configuration is not an invalid one.** No file at all resolves to the built-in
defaults. A file that exists and is malformed — bad YAML, a duplicate key, an unknown field, an
invalid value — is fatal, and is never replaced by those defaults: the machine would then run on
settings nobody chose, and the file that said otherwise would never be mentioned again.
`MD_TOOLS_CONFIG` naming a file that does not exist is a broken reference, because somebody meant
that path.

Every run record distinguishes the three ways a platform can be chosen, because a CPU result has
three possible causes and only one of them is nobody's decision:

```yaml
acceleration:
  platform_selection: machine-config     # or built-in-default, or cli-override
  cli_cpu_override: false                # true ONLY for --cpu
  platform_origin: machine.openmm
  requested_policy: machine-cpu          # or default-cuda, or explicit-cpu
  resolved_platform: CUDA
  cuda_device_index: 3
  cuda_precision: mixed
  device_policy: local_rank
  gpus_on_host: [NVIDIA RTX A5000, ...]
  cuda_visible_devices: null
  cuda_driver_version: "580.95.05"
  openmm_version: "8.6"
  mpi: {rank: 3, size: 8, local_rank: 3}
```

One resolver, `md_tools.openmm.platform_policy`, serves stages, ladders and AIS alike, so they
cannot drift apart. This applies to OpenMM Context work; `build-top` assigns parameters with
OpenFF and AmberTools, which is CPU work and is not claimed to be anything else.

## REST2 and rREST2 under MPI

One rank per thermodynamic state:

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \
          -c eq_npt_free.xml -o REST2.out -x REST2.nc -r restart.json -log REST2.log
```

Three numbers must agree — `rest2.number_of_replicas`, `-ng`, and the MPI world size — and a
mismatch is refused before a Context is opened, naming all three:

```text
a REST2 ladder runs one process per thermodynamic state, and these three numbers must be the same:
  replicas in the configuration : 4  (rest2.number_of_replicas)
  -ng on the command line       : 2
  MPI world size                : 2
```

`-ng` never resizes the ladder. A world size that is not the state count leaves states either
unowned or shared, and the exchange record would describe neither. `-ng > 1` outside an MPI launch
is refused too: it says how many processes coordinate, it does not create them.

**Device placement is deterministic and recorded.** Nothing binds ranks to devices automatically;
without placement every rank builds its Context on the default device and the whole ladder runs on
one GPU, silently and slowly. `CUDA_VISIBLE_DEVICES` renumbers devices from 0 for the process, so
the ordinals are `0..n-1` for that process rather than the driver values in the variable. If a rank
cannot initialise its device, the coordinated run aborts; it never continues with that rank on CPU.

Ranks do not share output files. Rank 0 prepares the ladder's inputs behind a barrier, and each
rank keeps its own `.out` and log beside rank 0's — a rank that failed to bind its device is
exactly what a multi-GPU run needs to be able to show.

What the ladder writes is unchanged: one trajectory per fixed thermodynamic **state**
(`remd0.nc` … `remd7.nc`, never walker- or tau-named), an Amber-compatible `rem.log`, the
neighbouring-pair acceptance report, and `restart.json`.

A group file is still supported through `-groupfile`, but an ordinary homogeneous ladder shares one
topology and one System, and restating the same two paths eight times is a way to get one of them
wrong.

## AIS

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \
          -source-traj ../cMD_tau0p5/tau_0p5.dcd -o AIS.out -log AIS.log -odir ./AIS
```

Paths are independent and never exchange, so `-ng` here is simply how many workers share them.
`number_of_paths` is the **global** total, not a count per rank.

**Path identity does not depend on the worker count.** `paths_for_rank(rank, size, total)` is a
pure function, so global path *n* starts from the same source frame with the same seeds and writes
the same file whatever the world size. A campaign resumed on a different number of GPUs lands on
the same files rather than reshuffling which trajectory is which.

```text
AIS_traj0000.nc … AIS_traj0099.nc     one trajectory per path, zero-based, ≥ 4 digits
AIS_work.csv                          one row per (path, switching step), sorted by that pair
AIS_paths.csv                         one row per path: the work distribution
selected_source_frames.csv            which frame each path started from, and with which seeds
path_NNNN/observations.csv            the work rows for one path
path_NNNN/completed.json              the machine record that says the path finished
```

The work table is assembled by rank 0 from the per-path records on disk rather than gathered over
MPI, so an interrupted campaign still produces a table describing exactly what was measured. A path
with no completion record is simply absent; the table never invents a row for work that was not
measured.

A completed path is **skipped, never appended to** — and only after its completion manifest and
the sha256 of every output it claims have been verified, because "status: completed" is a field in
a file and the files it describes are what a reader will actually load.

An interrupted path **resumes mid-path**, from its last committed checkpoint generation. This page
used to say the opposite: that a switching path has no meaningful mid-path restart, because the
work integral is only defined along a whole path. The premise is right and the conclusion was
wrong — the integral is defined along the whole path, and a resume continues *that* path, with the
accumulated work, the three component accumulators, the Context and every stream counter restored
to one committed instant.

### The source ensemble

`-source-traj` names a file you already produced, typically a fixed-τ cMD run at `tau_start`. AIS
anneals away from an equilibrium ensemble; it cannot generate one.

* a path with a trailing slash fails as *"is a directory, not a file"*, here, rather than obscurely
  inside the trajectory reader;
* the atom count is read from the file's own header and compared with **`-p` and `-s`** — the
  topology and the serialised System the paths actually run in. This page said `-p` and `-x`,
  which is wrong twice over: `-x` is an *output* trajectory, and AIS does not take one at all
  (it writes `AIS_trajNNNN.nc`, one per path, and refuses `-x` by name). A source trajectory of a
  different system reads without error and produces work values that mean nothing;
* the file is hashed into the record;
* frames are chosen deterministically from the global seed, and every path's source frame is
  written down *before* any dynamics;
* `ais_source.frame_stride` takes every Nth frame of the window. Consecutive MD frames are
  correlated; a stride is the honest way to say how far apart samples must be. It does not make
  them independent, it stops them being obviously dependent;
* **`tau_start` is asserted, not verified.** Nothing in a coordinate trajectory records the
  Hamiltonian it was sampled under. The log says so in those words, so a reader checks it against
  the run that produced the file rather than assuming it was checked here;
* a coordinate trajectory carries no velocities, so each path draws fresh Maxwell–Boltzmann momenta
  at the configured temperature with its own recorded seed. That policy is in the record. Nothing
  pretends an ordinary trajectory contains stored phase space.

### Divisibility

Every enabled step interval must divide the switching path exactly:

```python
switching_steps % interval == 0
```

for `ais.observation_interval_steps`, `reporting.solute_printout`, `reporting.system_printout` and
`reporting.checkpoint_printout`. A path is a complete object: it starts at `tau_start` and ends at
`tau_end`, and an interval that does not divide `switching_steps` cannot place a frame on the final
step. The last frame would fall at some interior τ, and "the end of path A" would not be comparable
with "the end of path B". That is not a rounding inconvenience, it is a different measurement.

Zero means disabled and is exempt. The refusal names both values and the divisors near the one you
wrote:

```text
reporting.system_printout is 100, which does not divide ais.switching_steps = 250
(250 % 100 = 50).
  Divisors of 250 near 100: 50, 125, 250.
```

The four cadences are **independent**. They answer four different questions, and tying two of
them together answers one of them with the other's answer:

| setting | controls | disabled by 0 |
|---|---|---|
| `ais.observation_interval_steps` | how often the WORK is measured — this one is the method | no |
| `reporting.solute_printout` | frames written to `AIS_trajNNNN.nc` | no |
| `reporting.system_printout` | rows in `path_NNNN/system.csv` | yes |
| `reporting.checkpoint_printout` | how often the path becomes resumable | yes |

Work every 10 steps with frames every 50 is a perfectly ordinary thing to want — it is a smaller
file — and it used to be refused. Leave `solute_printout` unstated and it follows the observation
cadence, which is what every existing project was generated with.

### `system.csv`: what the path is doing

At `reporting.system_printout`, each path writes a row of `path_NNNN/system.csv`:

```text
path_index, protocol_step, switching_time_ps, tau,
potential_energy_kj_mol, kinetic_energy_kj_mol, total_energy_kj_mol,
temperature_kelvin, volume_nm3, density_g_per_ml
```

Volume and density are empty under implicit solvent, where there is no box to have either. This is
a different question from the work: it is how the system is *behaving* while the Hamiltonian
moves, which tells you a switch is too fast long before the work distribution does.

### Mid-path resume

At `reporting.checkpoint_printout` a path writes an OpenMM binary checkpoint and a sidecar holding
everything outside the Context — accumulated work, work since the last observation, and how many
work rows, frames and state rows are on disk. Restoring one without the other resumes a simulation
into somebody else's bookkeeping, so they are written and validated together.

```bash
md-openmm md-run -i AIS.in -p built.pdb -s built.xml \
          -source-traj ../cMD_tau0p5/tau_0p5.dcd -odir ./AIS --resume
```

On `--resume`, and in this order:

1. **every fingerprint is checked before anything loads** — the System, the topology, the source
   ensemble, the whole schedule, the path id, its source frame and its seeds. A checkpoint from a
   different run would continue with right-looking numbers for a different measurement;
2. the Context is restored, and τ is put back to the checkpoint's rung;
3. the streams are cut back to the counts the sidecar vouches for, so a record interrupted
   mid-write is dropped rather than appended to;
4. the path continues, and finishes with exactly the number of observations, frames and state rows
   an uninterrupted run produces.

Frames are staged inside the path directory and published to `AIS_trajNNNN.nc` in one atomic move
only after the path is complete and validated. Until then there is nothing at the run root that
could be mistaken for a finished path. A completed path is skipped, never overwritten, and its
checkpoint is removed so nothing invites a resume of finished work.

With `checkpoint_printout: 0` there are no checkpoints, and an interrupted path restarts from its
source frame — there is nothing committed to resume from. That is the setting, not a property of
the method: with checkpoints on, an interrupted path continues from the last committed generation.

### The work convention, unchanged

```text
ΔW_j = U(τ_{j+1}, x_j) − U(τ_j, x_j)
```

Parameters move first, at frozen coordinates; the configuration then propagates under the new
Hamiltonian. Observation 0 is the source configuration under the source Hamiltonian, before any
parameter change and before any propagation, so its work is exactly zero by definition rather than
nearly zero. Switching is at **fixed volume**; a barostat in the prepared System is refused.

## What `build-md` generates

```text
md_script/
  min.py  eq_nvt_posres.py  …      compact Python entry points, one import and one call
  min.in  eq_nvt_posres.in  …      the Amber-like inputs, one per stage
  REST2.in                         the ladder, or AIS.in for switching paths
  run.sh                           drives `md-openmm md-run`
  resolved.config                  authoritative
  build-md.log                     the generation record
```

`run.sh` calls the installed command rather than `python <stage>.py`, because that is the interface
a person types by hand, and a script using a different one would be a second way to run the same
thing that can drift from the documented way. A ladder line spells the state count out —
`mpirun -n 4 md-openmm md-run -ng 4` — since the ladder's size is a property of the configuration,
not of the machine it lands on. An AIS `run.sh` honours `NPROC`, because how many workers run is a
machine choice while which paths exist is not.
