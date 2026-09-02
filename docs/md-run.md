# `md-openmm md-run` — running what `build-md` resolved

`md-run` executes a stage, a REST2/rREST2 ladder, or a set of AIS switching paths from a short
Amber-like input file. If you have run `pmemd -i mdin -p prmtop -c inpcrd -o mdout -r restrt`, you
can read every command on this page without a manual, which is the whole reason it exists.

```bash
md-openmm md-run -i min.in -p built.pdb -x built.xml -r min.xml -log min.log

mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -x built.xml \
          -c eq_npt_free.xml -odir REST2/

mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -x built.xml \
          -source-traj cMD_tau0p5/tau_0p5.nc -odir AIS/
```

It is a **surface**, not a second implementation. Every protocol is handed to the same function a
generated script calls — `stage_main`, `replica_main`, `ais_main` — so `python min.py` and
`md-openmm md-run -i min.in` are the same run, and a correction in the installed package reaches
both. There is no behaviour reachable from one and not the other.

## The flags

| flag | meaning | Amber |
|---|---|---|
| `-i` | the run input: `&cntrl` / `&remd` / `&AIS` sections | `-i mdin` |
| `-p` | topology and reference coordinates, `built.pdb` | `-p prmtop` |
| `-x` | the serialised OpenMM System, `built.xml` | (part of `prmtop`) |
| `-c` | starting state — the previous stage's final state | `-c inpcrd` / `restrt` |
| `-r` | output final state, the handoff to the next stage | `-r restrt` |
| `-o` | human-readable run output | `-o mdout` |
| `-log` | readable log carrying the machine record | |
| `-chk` | output checkpoint, written periodically for resume | |
| `--trajectory` | output trajectory | `-x mdcrd` |
| `-odir` | where outputs and `resolved.config` are written | |
| `-ng` | how many replicas or workers this launch coordinates | `-ng` |
| `-groupfile` | an Amber-style group file, for a heterogeneous ladder | `-groupfile` |
| `-source-traj` | AIS only: the equilibrium ensemble the paths start from | |
| `--cpu` | explicit CPU execution — see the platform policy below | |

Amber's `prmtop` carries the topology *and* the parameters. Here they are two files, and `-x` is
the one that holds the physics. That is the single place this surface differs from `pmemd`, and it
is why `-x` is not the trajectory: the output trajectory keeps only its long `--trajectory` form
rather than borrowing a short flag that means something else here.

`-h` works on a machine with no GPU, no driver and no OpenMM Context, because that is where people
read it before submitting a job.

## Which file is authoritative

**`resolved.config`. Always.**

The `.in` file is an *input*: a short, partial, human-written statement of intent. Resolving it —
through `md_tools.build.md`, the one authority for MD workflow configuration — produces
`resolved.config`, with every default written out and every cross-field rule applied. That resolved
document is what the run reads, what the checkpoint fingerprint binds, and what the log records.

`md-run` writes it into `-odir` before anything integrates, with the `.in` file's sha256 in its
header, so the chain from what a person wrote to what actually ran is a link a reader can follow
rather than a claim:

```text
# The configuration this run resolved to, in full. THIS FILE IS AUTHORITATIVE:
#
#   input      : REST2.in
#   sha256     : 3f0c…
#   sections   : &cntrl, &remd
```

Two consequences worth stating plainly:

* editing `resolved.config` between a run and its continuation changes the fingerprint and the
  continuation is refused — which is the intent;
* editing the `.in` afterwards changes nothing at all. Its digest in the record simply stops
  matching, which is how you find out.

If `-odir` already holds a `resolved.config` describing a *different* run, `md-run` refuses rather
than overwriting: a directory holding outputs from one resolution and the configuration of another
cannot be read correctly afterwards. An identical one is left alone, so rerunning a command is
safe. `--overwrite` is the explicit way through.

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

## CUDA is the default, and it is mandatory

```text
md-openmm md-run ≈ pmemd.cuda
```

There is no automatic fall back to CPU, OpenCL or Reference. A run that quietly moved to the CPU
still finishes, still writes a trajectory and still reports success — two orders of magnitude
later, on a machine whose GPU was simply not visible to the process. The result is not obviously
wrong, which is what makes it expensive: it is found weeks later, if at all.

* CUDA that cannot be initialised is an error **before** dynamics. Listing the platform is not the
  same as having a usable device — conda-forge ships the plugin unconditionally — so the resolver
  opens and discards a one-particle Context to prove it.
* `--cpu` is the only public way to ask for a CPU run.
* `--cpu --platform CUDA` is refused as a contradiction rather than resolved by precedence: either
  choice would silently discard half of what was asked for.

One resolver, `md_tools.openmm.platform_policy`, serves stages, ladders and AIS alike, so they
cannot drift apart again. Every run record carries the requested policy (`default-cuda` or
`explicit-cpu`), the resolved platform, the CUDA device index and precision, the GPUs on the host,
`CUDA_VISIBLE_DEVICES`, the OpenMM version, and the MPI rank and world size. **A CPU result can
never be mistaken for an unnoticed CUDA fallback.**

This applies to OpenMM Context work. `build-top` assigns parameters with OpenFF and AmberTools,
which is CPU work and is not claimed to be anything else.

## REST2 and rREST2 under MPI

One rank per thermodynamic state:

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -x built.xml \
          -c eq_npt_free.xml -odir REST2/ -log REST2.log
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
mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -x built.xml \
          -source-traj cMD_tau0p5/tau_0p5.nc -odir AIS/ -log AIS.log
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

A completed path is **skipped, never appended to**. An interrupted one is rerun from its source
frame, because a switching path has no meaningful mid-path restart — the work integral is only
defined along a whole path.

### The source ensemble

`-source-traj` names a file you already produced, typically a fixed-τ cMD run at `tau_start`. AIS
anneals away from an equilibrium ensemble; it cannot generate one.

* a path with a trailing slash fails as *"is a directory, not a file"*, here, rather than obscurely
  inside the trajectory reader;
* the atom count is read from the file's own header and compared with `-p` and `-x`. A trajectory
  of a different system reads without error and produces work values that mean nothing;
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

An AIS run writes one frame per observation, so `reporting.solute_printout` and
`ais.observation_interval_steps` must be the same number: each observation writes one frame, and
its row indexes that frame. Leave `solute_printout` unstated and it follows the observation
cadence.

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
