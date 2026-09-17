# `md-openmm md-run` — running what `build-md` resolved

`md-run` executes a stage, a REST2 ladder, or a set of AIS switching paths from a short
Amber-like input file. If you have run `pmemd -i mdin -p prmtop -c inpcrd -o mdout -r restrt`, you
can read every command on this page without a manual, which is the whole reason it exists.

```bash
md-openmm md-run -i min.in -p built.pdb -s built.xml -c prev.xml \
          -o min.out -x min.dcd -r min.xml -log min.log

mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \
          -c eq_npt_free.xml -o REST2.out -x REST2.nc -r restart.json -log REST2.log

mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s V0.xml -p2 built.pdb -s2 V1.xml \
          -source-traj ../hot/whole_prod1.nc -o AIS.out -log AIS.log -odir ./AIS
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
| `-s2` / `-p2` | AIS only: V1, the second end state's System and topology | a second group's `-p` |
| `-source-traj` | AIS only: the equilibrium ensemble of V0 the paths start from | — |
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

### If you are coming from Amber, `mdout.csv` is not `mdout`

The names invite a wrong mapping. Amber writes two energy files and MD-tools writes three, and
they do not line up one-to-one:

| Amber | controlled by | MD-tools | controlled by |
|---|---|---|---|
| `mdout` — human-readable, with the per-term breakdown | `ntpr` | **`<name>.out`** | `info_printout` |
| `mden` — columnar energies, for parsing | `ntwe` | **`mdout.csv`** | `info_printout` |
| the `BOND`/`ANGLE`/`EEL`/`EGB` decomposition inside `mdout` | `ntpr` | **`energy_components.csv`** | `info_printout` |

So the file called `mdout.csv` is the analogue of Amber's `mden`, and the analogue of Amber's
`mdout` is the `.out`. One `info_printout` drives all three, which is why there is no separate
`ntwe`.

Two differences worth knowing before you go looking for something that is not there:

- **The decomposition is by FORCE GROUP, not by Amber energy term.** A column is labelled with
  the forces in its group, so `CustomGBForce+NonbondedForce` means those two share a group and
  their energies are reported as one number. That is stated rather than hidden behind a prettier
  name; splitting them would need a different force-group assignment, which changes the
  serialised System and so its digest, and would make every run in flight unresumable.
- **Averages and RMS fluctuations** are printed at the foot of the `.out`, as Amber prints them
  at the foot of an `mdout`. They are computed from the rows `mdout.csv` actually holds, so a
  resumed run summarises what it kept rather than steps whose rows were truncated.

On implicit solvent no box columns are written to either file: a system with no periodic box has
no volume and no density, and reporting the nominal unit cell for one would be a number
describing something that does not exist.

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
stream, the state tables, the restarts, the per-state `whole_state<i>_prod<N>.nc`, the generated
helpers — was
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
| `local_rank` | this package places every worker: CPUs, then devices by measured throughput (see *CPUs, devices and MPS*) |
| `openmm` | set no `DeviceIndex` and let OpenMM choose — right when a scheduler has already given each rank one device with `CUDA_VISIBLE_DEVICES`. Otherwise the workers are treated as sharing, and MPS is required |

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
  placement: {workers: 8, placement: measured throughput (balanced), ...}
```

One resolver, `md_tools.openmm.platform_policy`, serves stages, ladders and AIS alike, so they
cannot drift apart. This applies to OpenMM Context work; `build-top` assigns parameters with
OpenFF and AmberTools, which is CPU work and is not claimed to be anything else.

## REST2 under MPI

One rank per thermodynamic state, from the run directory, as `run.sh` launches it:

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i ../input/REST2.in -p ../build/built.pdb \
          --groupfile remd_groupfile.1 -odir . \
          -o remd_records/REST2_prod1.out -log remd_records/REST2_prod1.log \
          -r remd_records/restart_prod1.json
```

There is no `-s` and no `-c`: every line of the group file names its state's saved System
(`-s ../build/REST2/system_state<i>.xml`, from `md-openmm build-top --rest2-scaler`) and the
starting state, and `-s` on the command line is refused. The group file's `-i _protocol.py` is the
one the ladder writes into `-odir`, so any `-odir` other than the group file's directory is refused
too.

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

**Placement is decided once, in the preflight, and recorded** — see *CPUs, devices and MPS* below.
`CUDA_VISIBLE_DEVICES` renumbers devices from 0 for the process, so the ordinals are `0..n-1` for
that process rather than the driver values in the variable. If a rank cannot initialise its device,
the coordinated run aborts; it never continues with that rank on CPU.

Ranks do not share output files. Rank 0 prepares the ladder's inputs behind a barrier, and each
rank keeps its own `.out` and log beside rank 0's — a rank that failed to bind its device is
exactly what a multi-GPU run needs to be able to show.

What the ladder writes is unchanged: one trajectory per fixed thermodynamic **state**
(`solute_state0_prod1.nc` … `solute_state7_prod1.nc`, never walker- or tau-named), an
Amber-compatible `rem.log`, the neighbouring-pair acceptance report, and `restart.json`.

A group file is still supported through `-groupfile`, but an ordinary homogeneous ladder shares one
topology and one System, and restating the same two paths eight times is a way to get one of them
wrong.

## CPUs, devices and MPS

A REST2 ladder runs one worker per state and an AIS launch one worker per rank. Where each worker
runs is decided by `md_tools.openmm.placement`, for every protocol, in the preflight — before any
output exists — in this order.

**1. CPUs.** The CPUs the launch may use on a node must be an integer multiple of the workers on
that node, and each worker is bound to its own equal block (whole cores together, one socket
before the next). The count is what the kernel allows — the scheduler affinity, narrowed by a
cgroup v2 `cpu.max` quota — never a number a configuration claims. Four workers accept 4, 8 or 12
CPUs; five is refused, naming the arithmetic:

```text
REST2: 5 CPU(s) are available to 4 worker(s), and 5 is not an integer multiple of 4 (5 = 4 x 1 + 1).
  Every worker gets an equal block of CPUs, so the count must divide exactly. Make 4 CPUs (1 per
  worker) or 8 CPUs (2 per worker) available ...
```

Open MPI binds each rank to one core by default, and those cores are then all the launch may use.
To give the workers a larger share, hand the launch its CPUs and let placement divide them:

```bash
taskset -c 0-23 mpirun --bind-to none -n 4 md-openmm md-run -ng 4 ...   # 6 CPUs per worker
```

On the CPU platform a bound worker's OpenMM thread pool is its block size, unless
`OPENMM_CPU_THREADS` says otherwise.

**2. Devices, by measured throughput.** With more than one worker on a node, the first rank on
that node runs *this run's* System for about a second on each visible device, one device at a time,
and every worker is placed from that measurement: each takes the device whose per-worker share
(steps per second divided by the workers on it) stays highest, so a slower device carries fewer
workers or none. With at least as many devices as workers, each worker gets its own — the fastest
ones. Fewer devices than workers is allowed, including uneven splits: 4 workers on 3 equal devices
are placed 2, 1, 1. Whatever else the GPUs are running at that moment is part of the measurement.
A single worker takes the first visible device, as before, and measures nothing.

**3. MPS, whenever a GPU hosts more than one worker.** Without NVIDIA MPS, processes on one GPU
are time-sliced, and a synchronous ladder runs at the pace of that shared GPU. A shared device is
therefore refused unless MPS is **verified** for each worker. Three states are told apart and
recorded:

| status | meaning |
|---|---|
| `requested-not-detected` | `CUDA_MPS_PIPE_DIRECTORY` is set, and no control daemon serves it |
| `detected-unverified` | a daemon is running with its `control` pipe in this process's pipe directory; nothing yet says this process is its client |
| `verified` | while this worker holds a Context, `nvidia-smi -q -x` lists its PID as an MPS client (`M+C`) |
| `not-a-client` | the driver lists the worker as an ordinary CUDA process (`C`): the daemon exists, but this process is not using it |
| `absent` / `unknown` | no daemon; or the process table could not be read |

MD-tools never starts, stops or talks to an MPS daemon: it is a shared service that someone else
may own. To run with one you started yourself:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe      # directories you own
export CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d                          # the control daemon; servers start on demand

mpirun --bind-to none -n 4 md-openmm md-run -ng 4 ...   # launched from the SAME environment

echo quit | nvidia-cuda-mps-control                 # stop it: only a daemon you started
```

A client finds the daemon only through `CUDA_MPS_PIPE_DIRECTORY` (default `/tmp/nvidia-mps`); a
worker launched without it runs as an ordinary CUDA process and says nothing, which is why
detection is not verification. The daemon's own `CUDA_VISIBLE_DEVICES` limits which GPUs its
clients can use.

The run record carries the whole plan under `acceleration.placement`: every worker's host, local
rank, CPU block and the CPUs the launcher had bound it to, its device and how many workers share
it, the measurement, and the MPS status with the MPS environment. `--check` prints this rank's
share of it. Placement never enters a scientific fingerprint: a ladder's state index is the state's
and an AIS path id is the path's, whatever device ran them.

## AIS

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in \
          -p built.pdb -s ../build/REST2/system_state3.xml \
          -p2 built.pdb -s2 ../build/built.xml \
          -source-traj ../hot/whole_prod1.nc -o AIS.out -log AIS.log -odir ./AIS
```

AIS transforms one end state into another,

```text
V(λ) = (1 − λ)·V0 + λ·V1,        λ: 0 → 1, linear
```

**V0** is `-s`/`-p`, the state the source ensemble was sampled from. **V1** is `-s2`/`-p2`. They must
hold the same particles in the same order, with the same masses, constraints, virtual sites, force
layout and long-range treatment — they differ in parameters only, as in sander's no-softcore
mixing — and a pair that differs in anything else is refused with every difference named, as is a
barostat in either. For a REST2 switch, V0 is the scaled state the hot run integrated and V1 is the
unmodified `build/built.xml`. A reverse switch is the same command with the files exchanged. `-s2`
and `-p2` are refused by name on every other protocol.

`ais.tau_start`, `ais.tau_end`, `ais.work_measurement` and `ais.verify_every_updates` belonged to the
single-topology AIS, which switched one System along τ, and are refused with the migration.

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
path_NNNN/observations.csv            the work rows for one path, with V0, V1 and V(λ) at saved frames
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
accumulated work, the Context and every stream counter restored to one committed instant. On CUDA
the continuation is not bit-for-bit the trajectory an uninterrupted run would have taken: the
mixing force's inner Contexts keep state no checkpoint captures. The committed state is restored
exactly and the path continues as a new realisation of the same switching process; on CPU it
reproduces exactly.

### The source ensemble

`-source-traj` names a file you already produced, an equilibrium run of V0 — typically a fixed-τ cMD
run on the scaled state that is `-s`. AIS switches away from an equilibrium ensemble; it cannot
generate one.

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
* **the source is VERIFIED to be V0's when it can be, and asserted when it cannot.** A stage's
  whole-system AMBER NetCDF records `system_sha256` when the Hamiltonian it integrated is its `-s`
  unmodified, and AIS refuses a source whose recorded digest is not `sha256(-s)`. A DCD or a
  foreign file records nothing, and the log says the ensemble was asserted, in those words;
* a coordinate trajectory carries no velocities, so each path draws fresh Maxwell–Boltzmann momenta
  at the configured temperature with its own recorded seed. That policy is in the record. Nothing
  pretends an ordinary trajectory contains stored phase space.

### Divisibility

Every enabled step interval must divide the switching path exactly:

```python
switching_steps % interval == 0
```

for `ais.observation_interval_steps`, `reporting.crd_printout_solute`, `reporting.info_printout` and
`reporting.checkpoint_printout`. A path is a complete object: it starts at λ = 0 and ends at λ = 1,
and an interval that does not divide `switching_steps` cannot place a frame on the final step. The
last frame would fall at some interior λ, and "the end of path A" would not be comparable
with "the end of path B". That is not a rounding inconvenience, it is a different measurement.

Zero means disabled and is exempt. The refusal names both values and the divisors near the one you
wrote:

```text
reporting.info_printout is 100, which does not divide ais.switching_steps = 250
(250 % 100 = 50).
  Divisors of 250 near 100: 50, 125, 250.
```

The four cadences are **independent**. They answer four different questions, and tying two of
them together answers one of them with the other's answer:

| setting | controls | disabled by 0 |
|---|---|---|
| `ais.observation_interval_steps` | how often the WORK is measured — this one is the method | no |
| `reporting.crd_printout_solute` | frames written to `AIS_trajNNNN.nc` | no |
| `reporting.info_printout` | rows in `path_NNNN/system.csv` | yes |
| `reporting.checkpoint_printout` | how often the path becomes resumable | yes |

Work every 10 steps with frames every 50 is a perfectly ordinary thing to want — it is a smaller
file — and it used to be refused. Leave `crd_printout_solute` unstated and it follows the observation
cadence, which is what every existing project was generated with.

### `system.csv`: what the path is doing

At `reporting.info_printout`, each path writes a row of `path_NNNN/system.csv`:

```text
path_index, protocol_step, switching_time_ps, lambda,
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
md-openmm md-run -i AIS.in -p built.pdb -s ../build/REST2/system_state3.xml \
          -p2 built.pdb -s2 ../build/built.xml \
          -source-traj ../hot/whole_prod1.nc -odir ./AIS --resume
```

On `--resume`, and in this order:

1. **every fingerprint is checked before anything loads** — both end states, both topologies, the
   source ensemble, the whole schedule, the path id, its source frame and its seeds. A checkpoint from a
   different run would continue with right-looking numbers for a different measurement;
2. the Context is restored, and λ is put back to the committed update's value;
3. the streams are cut back to the counts the sidecar vouches for, so a record interrupted
   mid-write is dropped rather than appended to;
4. the path continues, and finishes with exactly the number of observations, frames and state rows
   an uninterrupted run produces.

A checkpoint, manifest or `AIS_run.json` written by the single-topology AIS is refused by name
rather than continued: its work was measured along a different path.

Frames are staged inside the path directory and published to `AIS_trajNNNN.nc` in one atomic move
only after the path is complete and validated. Until then there is nothing at the run root that
could be mistaken for a finished path. A completed path is skipped, never overwritten, and its
checkpoint is removed so nothing invites a resume of finished work.

With `checkpoint_printout: 0` there are no checkpoints, and an interrupted path restarts from its
source frame — there is nothing committed to resume from. That is the setting, not a property of
the method: with checkpoints on, an interrupted path continues from the last committed generation.

### The work convention

```text
ΔW_j = V(λ_{j+1}, x_j) − V(λ_j, x_j) = (λ_{j+1} − λ_j) · (V1 − V0)(x_j)
```

Parameters move first, at frozen coordinates; the configuration then propagates under the new
Hamiltonian. The second equality is exact because V is linear in λ, and it is how the work is
measured: one evaluation of dV/dλ per switch. It is also exactly Amber's Jarzynski increment
(§27.8, `(∂U/∂λ)·Δλ`). Observation 0 is the source configuration under V0, before any parameter
change and before any propagation, so its work is exactly zero by definition rather than nearly
zero. Switching is at **fixed volume**; a barostat in either end state is refused.

At every observation that saves a frame the row carries `potential_v0_kj_mol`,
`potential_v1_kj_mol` and `potential_direct_kj_mol`, measured at that frame, with `V1 − V0` checked
against an independent evaluation before the row is written.

## What `build-md` generates

```text
<system>/                          THE DATASET ROOT -- one system and every run on it
  build/   built.xml built.pdb     from build-top, shared by every run
  min/     min.py resolved.config run.config    the SHARED minimisation
  input/   min.in eq_1.in eq_2.in eq_3.in       shared: an input is not per-repeat
           REST2.in cMD.in AIS.in               one production input per method
  <method>-run<N>/                 ONE RUN, named by `-odir`; must not already exist
    eq/      eq_1.py eq_2.py eq_3.py resolved.config run.config
    remd<n>/ build_state<n>.xml    the rung Hamiltonian, pre-scaled at BUILD time
    remd_groupfile.<x>             one per segment, naming one rung per line
    REST2.py                       the ladder entry point, or AIS.py / cMD.py
    run.sh                         drives `md-openmm md-run`
    resolved.config                authoritative for this run
    run.config                     the seed -- the only per-run declaration
    build_states.log               which factors scaled which terms
    remd_records/  rank/  bundles/
```

`min/` and `eq/` hold METHOD-NEUTRAL declarations, because the scripts in them read the shared
`input/*.in`, which carry no protocol. See `docs/run-layout.md`.

`run.sh` calls the installed command rather than `python <stage>.py`, because that is the interface
a person types by hand, and a script using a different one would be a second way to run the same
thing that can drift from the documented way. A ladder line spells the state count out —
`mpirun -n 4 md-openmm md-run -ng 4` — since the ladder's size is a property of the configuration,
not of the machine it lands on. An AIS `run.sh` honours `NPROC`, because how many workers run is a
machine choice while which paths exist is not.
