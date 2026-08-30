# REST2 on OpenMMTools

REST2 replica exchange runs on
[`openmmtools.multistate.ReplicaExchangeSampler`](https://openmmtools.readthedocs.io/), with
`MultiStateReporter` NetCDF as the authoritative storage. This repository keeps the REST2
Hamiltonian; OpenMMTools keeps everything about running a ladder.

| owned by | what |
|---|---|
| **MD-templates** | the REST2 scaling convention, the tau ladder, physical-time arithmetic, the enhanced region and omega exclusions, the run's scientific identity, the completion manifest, and **which iterations attempt an exchange** |
| **OpenMMTools** | propagation, reduced potentials, **the accept/reject decision under the default `swap-all`**, thermodynamic-state assignment, multistate NetCDF storage, checkpointing, restart, extension, walker-to-state mapping |

### Who decides a swap

This is stated precisely because it was previously overstated. Under the default scheme,
`swap-all`, every proposal and every Metropolis decision happens inside
`ReplicaExchangeSampler._mix_all_replicas_numba` and this repository touches none of it.

What this repository owns about mixing is **when** it is attempted: `_mix_replicas` is gated so
exchange happens every `exchange_stride` iterations. That is a schedule, not a decision.

`swap-neighbors` is different and is not the default. OpenMMTools 0.26.0's neighbour path is broken
on NumPy ≥ 1.25, so selecting it activates `ProjectNeighbourExchangeMixin`, in which **this
repository performs the Metropolis call**. A run using it records
`exchange_decision_owner: md-templates`, and every run records which of the two applied.

`MD-project` owns none of it. It pins a generator commit, states a request and consumes results.

## The scaling, and the omega convention

`tau` is the source parameter. `s` and any "effective temperature" are derived, labelled as such,
and never accepted back as input.

```
s        = (1 - tau)^2      solute-solute terms, and solute torsions
sqrt(s)  = 1 - tau          solute-environment terms
1                           environment-environment terms
```

**Every replica is thermostatted at the same temperature and, under NPT, the same pressure.** The
rungs differ by Hamiltonian only. This is not temperature REMD, and the pV terms cancel in the
acceptance criterion because a swap moves each configuration — positions *and* its box — to the
other rung.

**This is omega-selective REST2.** Torsions about a peptide omega bond are left **unscaled**:
scaling them lets a peptide bond rotate at the hot rungs, so the ladder samples cis/trans
interconversion the cold rung never sees and the exchange stops connecting two states of the same
system. That is a deliberate departure from a plain textbook REST2 Hamiltonian. It is recorded in
`restart.json` under `omega_convention`, printed in the `.out`, and should not be described as an
unmodified standard REST2.

Implicit solvent scales the **whole** generalised-Born energy by `s`, and the whole system must be
the enhanced region. A partial GBn2 region is refused, not approximated: every Born radius depends
on every other atom's position, so a partial selection would need a validated treatment of the
solute-environment cross terms and there is none here.

## Time: the iteration is the solute output interval

This is the part worth reading twice, because it is what makes the trajectory honest.

`MultiStateReporter` writes `analysis_particle_indices` positions **every iteration** and full
coordinates every `checkpoint_interval` iterations. So to store solute frames more often than
exchanges are attempted, the iteration is made the *solute* interval and exchange is attempted
every `exchange_stride` iterations:

```
timestep                4 fs
iteration = solute out  2 ps  =   500 steps          -> one stored solute frame, every iteration
exchange attempt       10 ps  =  2500 steps          -> every 5th iteration
whole-system output    10 ps  =  2500 steps          -> checkpoint_interval = 5 iterations
```

Every solute frame is a configuration the integrator actually visited. The alternative — one 10 ps
iteration with the boundary configuration written five times — would be a fabricated trajectory and
is not done. A test asserts that consecutive solute frames are never identical.

Every conversion must be exact. A duration that does not divide is **refused, never rounded**: a
rounded interval means the physical time the records claim is not the one simulated.

### The cost, measured

Computing the reduced-potential matrix once per iteration rather than once per exchange is a real
cost, so it was measured rather than assumed.

| configuration | solute frames | wall time |
|---|---|---|
| iteration 2 ps, exchange 10 ps (stride 5) | one per 2 ps | 30.7 s |
| iteration 10 ps, exchange 10 ps (stride 1) | one per 10 ps | 25.6 s |

**+19.9%** for 5× finer solute sampling — ALA in OPC, 2418 particles, 6 replicas, 100 ps per
replica, one RTX 3080. The overhead is per *iteration* and largely independent of system size,
while propagation scales with it, so the relative cost falls as the system grows. Set
`output_interval` equal to the exchange interval and the stride becomes 1 and the overhead goes
away entirely; that is the trade the number above is there to let you make.

## The two upstream defects, and what is still pinned

`templates/rest2_openmmtools.py` holds the whole OpenMMTools extension. Under the default
`swap-all` it overrides **no** private method that decides anything: only `_mix_replicas`, to skip
non-exchange iterations, and `equilibrate`, to hold each replica at its own tau.

Two defects in 0.26.0's neighbour path, both established by reading the installed source and
running it:

**1. `swap-neighbors` is unusable on NumPy ≥ 1.25.** `_mix_neighboring_replicas` locates replicas
with `np.where(...)`, which returns a *tuple* of arrays. Indexing the energy matrix with that tuple
yields a 2-d array, and the acceptance test then calls `math.exp` on it:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

`swap-all` is unaffected because `_mix_all_replicas` draws plain integers.

**2. A two-replica ladder exchanged at half its configured rate.** Upstream draws an offset from
`{0, 1}` and iterates `range(offset, n-1, 2)`; with two replicas the odd phase covers no pair, so
half the exchange iterations proposed nothing, with a healthy-looking log.

Both are corrected in `ProjectNeighbourExchangeMixin`, used **only** when `swap-neighbors` is
explicitly selected. Because it subclasses private methods, selecting it requires OpenMMTools
exactly 0.26.0. Choosing `swap-all` needs no version pin for the decision path at all.

### A third defect, in this repository's own scheduling

The stride gate returned early without zeroing OpenMMTools' proposal matrices, which upstream
resets at the top of its own `_mix_replicas`. Every skipped iteration therefore re-wrote the
previous mixing event's counts, and a stride-2 run recorded **23 mixing events where 12 occurred**.
Skipped iterations now record genuine zeros — which is also what makes a mixing event countable
from the stored history at all.

## Running it

```bash
openmm-rest2 \
  -i "${REST2_DIR}/rest2.py" \
  -p "${INPUT_DIR}/topology.pdb" \
  -s "${INPUT_DIR}/system.xml" \
  -c "${NPT_FREE_DIR}/npt_free.state.xml" \
  --solute "${INPUT_DIR}/solute.yaml" \
  -o "${REST2_DIR}/rest2.out" \
  -x "${REST2_DIR}/rest2.nc" \
  -r "${REST2_DIR}/restart.json" \
  --checkpoint "${REST2_DIR}/rest2_checkpoint.nc"
```

which is what the generated `REST2/rest2.sh` runs, so in practice:

```bash
REST2/rest2.sh                        # one process, one device
mpiexec -n 6 REST2/rest2.sh           # one rank per replica
REST2/rest2.sh --resume               # continue in place
REST2/rest2.sh --extend 200           # add 200 exchange attempts
```

| option | meaning |
|---|---|
| `-i` | the concise REST2 Python protocol |
| `-p` | topology |
| `-s` | serialized base OpenMM `System` |
| `-c` | the common equilibrated state every replica starts from |
| `--solute` | the enhanced-region definition (`solute.yaml`) |
| `-o` | human-readable log |
| `-x` | the authoritative OpenMMTools multistate NetCDF |
| `-r` | a small JSON manifest that *references* the NetCDF |
| `--checkpoint` | full-coordinate checkpoint NetCDF |
| `--resume` | continue the same run in place |
| `--extend N` | add N exchange attempts |
| `--force` | replace a new, non-resumed output deliberately |

`-r` is **not** a restart state. Six replicas cannot be held in one XML, and pretending otherwise is
how a resume comes to start from one replica's coordinates. On resume the NetCDF is authoritative
for where the run stopped; the manifest is what the scientific configuration is checked against.

### The user-facing protocol

The generated `rest2.py` is 34 lines, most of them the docstring, and contains no path:

```python
from rest2_runtime import REST2

def run(files):
    REST2(
        files,
        tau_min=0.0, tau_max=0.5, number_of_replicas=6,
        exchange_interval_ps=10.0,
        solute_output_interval_ps=2.0,
        whole_output_interval_ps=10.0,
        equilibration_duration_ps=10.0,
        temperature_k=300.0, pressure_bar=1.0,
        timestep_fs=4.0, friction_per_ps=1.0,
        random_seed=1496485512, hydrogen_mass_amu=3.024,
        platform='CUDA',
    ).run(number_of_exchanges=1000)
```

The engine lives in `rest2_runtime.py`, copied into the project beside it. The project imports no
`md_templates` and runs with no clone of this repository present.

## Multi-GPU

**`mpiplus` does no device binding at all.** It distributes replicas across ranks and leaves the
platform entirely to the caller, so without an explicit map every rank creates its Context on the
default device and the whole ladder runs on GPU 0 — silently, at a fraction of the speed. Ranks
bind explicitly, and every rank reports its rank, host, visible devices, selected `DeviceIndex`,
precision and assigned work in its own `.out`:

```
# platform         : CUDA device=3 (one rank per device (6 ranks, 9 devices))
# mpi              : rank 3/6 on <host>
```

One rank per replica when there are enough GPUs. With fewer GPUs than ranks the policy is
round-robin and is *stated in the record*, never stumbled into. `CUDA_VISIBLE_DEVICES` renumbers
devices from 0 for the process, so the ordinals passed to OpenMM are local ones — passing the
driver's values through would address the wrong device whenever the variable does not start at 0.

Under MPI the shared outputs belong to the run, not to a rank: rank 0 owns the NetCDF and the
manifest, and other ranks write `rest2.out.rankNN` beside it.

CPU and single-GPU execution remain available and are what the smoke tests use.

## Three records, three jobs

Confusing these is what made an interrupted run unrecoverable, so they are named separately.

| record | written | job |
|---|---|---|
| **reporter metadata** | inside the analysis NetCDF, **before propagation begins** | the run's scientific identity. This is what makes an interrupted run resumable, because it survives in the authoritative storage whatever happens to the process. |
| **`<stem>.runstate.json`** | beside the NetCDF, atomically, rank 0 only | `initialized` → `running` → `completed` / `interrupted` / `failed`. Never claims completion. An independent fallback identity source. |
| **checkpoint NetCDF** | every `checkpoint_interval` iterations | full coordinates; what a restart propagates from. |
| **`restart.json`** | atomically, after the final iteration is committed | evidence that the run **completed**. It is *not* the source of resume identity. |

## Interrupted resume, and completed extension

These are different operations and are kept distinct.

```bash
REST2/rest2.sh --resume        # finish a run that stopped short of its budget
REST2/rest2.sh --extend 200    # add 200 mixing events to a run that reached its budget
```

**`--resume` does not require `restart.json`.** Requiring it was the bug: a run interrupted before
finishing never writes one, and that is exactly the run that needs resuming. The identity a
continuation is checked against comes from reporter metadata, with the run-state sidecar as an
independent fallback; a continuation that can establish neither is **refused** rather than trusted.

A resume continues to the run's *original* budget without resetting iteration, mapping, seeds or
statistics. `--extend N` requires a run that already reached its budget and adds exactly `N` mixing
events; extending an unfinished run is refused, because it would append iterations onto the end
while silently abandoning the ones never run.

SIGINT and SIGTERM are recorded: the sidecar becomes `interrupted`, no manifest is written, and the
OpenMMTools storage is left usable. Reporting is wrapped in `mpiplus.delayed_termination` upstream,
so a signal arriving mid-write is deferred until that iteration's records are complete.

## Authoritative validation

```bash
openmm-rest2 --verify-only -x REST2/rest2.nc \
    --checkpoint REST2/rest2_checkpoint.nc -r REST2/restart.json
```

There is **one** validator, `rest2_validate.py`, copied into every project; `MD-project` calls it
through this command rather than reimplementing the NetCDF schema. It **opens and reads** the
storage through the public `MultiStateReporter` API — `Path.is_file()` proves only that a path was
created, which is what a crashed run leaves behind.

It checks that the analysis and checkpoint NetCDF open; the recorded OpenMMTools version is
supported; the storage carries a scientific identity; the last committed iteration matches the
budget; the final checkpoint and its sampler states are readable and finite; the mapping has the
right shape and **every row is a permutation**; reduced potentials are the right shape and finite;
mixing statistics exist for **every** scheduled mixing event and none outside the stride; the
manifest names only files beside itself; and the manifest agrees with the storage's own metadata.

It rejects truncated files, valid NetCDF missing required variables, a manifest copied from a
different run, a missing final checkpoint, an incomplete budget, and a changed identity. Exit
status 0 means valid, 1 means not.

The budget's authority is chosen deliberately: the manifest and the sidecar are exact, while
reporter metadata records the **original** request and is never rewritten — so an extended run
legitimately exceeds it, and demanding equality there reported a correctly extended run as
incomplete.

## Lifetime mixing statistics

`sampler._n_proposed_matrix` is reset at the start of every mixing call, so reading it after a run
gives the **last event only**. A summary built from it described 1000 exchange attempts using the
statistics of one.

`MultiStateReporter.write_mixing_statistics` runs on *every* iteration, so
`read_mixing_statistics(slice(None))` returns the complete non-cumulative history — including
across an interrupted resume and an extension, because the storage is authoritative and not the
process that wrote it. Summing it is the only honest lifetime figure, and it is a public method, so
nothing parses a raw NetCDF variable.

Reported per state pair: total proposals, total acceptances, the lifetime fraction, and the
iteration range covered. Mixing events are **counted** from the stored history and cross-checked
against the configured stride, never inferred as `iteration // stride`.

**`swap-all` proposals are not a neighbour sweep** and are not labelled as one: they land on all
state pairs, and a pair may be drawn with `i == j`. A self-swap has `log_p = 0`, is always
accepted, and is counted on the **diagonal** — so every figure above is computed from off-diagonal
entries and the diagonal is reported separately.

## Round trips

A complete round trip is:

```
cold state visited  →  hot state visited later  →  cold state visited later still
```

The starting position earns nothing. A walker that **begins at the hot state** has not completed a
round trip when it first reaches cold — it has completed half of one, and must then return to hot
and come back. The previous counter credited that walker immediately, inflating the count for
exactly the walkers that had travelled least. It is now a three-state machine, tested for walkers
starting cold, hot and intermediate, for repeated endpoint residence, half trips and multiple
trips.

## Walker view and state view

OpenMMTools stores **walkers** plus a per-iteration `replica_thermodynamic_states` mapping. Both
views come from that mapping; neither is a second copy of the coordinates.

```
walker view : follow one continuous coordinate walker  -> column r of the mapping
state view  : whichever walker occupied tau state s    -> where each row equals s
```

Every row of the mapping is a permutation: each thermodynamic state is occupied exactly once per
iteration. Coordinates are **not** duplicated into separate walker and state trajectories during
the run — that is what makes the NetCDF the authoritative store rather than one of three.

## Completion

A finished run writes `restart.json` **atomically**, and only after OpenMMTools has committed the
final iteration and the expected exchange budget is present. Only then does the protocol print the
exact line

```
run_status: completed
```

`openmm-rest2` refuses to accept that line standing over a missing NetCDF, and refuses a manifest
that does not itself record `run_status: completed`. The existence of an output file is never
treated as proof of completion.

A resume recomputes and compares the whole scientific identity — tau ladder, replica count,
temperature, pressure, timestep, collision rate, exchange and reporting intervals, the enhanced
region and omega exclusions, and digests of the topology, system and solute definition — and
refuses to append when any of it changed. Running longer is a legitimate extension; changing the
physics is not.

## Comparison with Amber and GROMACS

| Feature | Amber | GROMACS | This implementation |
|---|---|---|---|
| single-stage runner | `pmemd.cuda` | `gmx mdrun` | `openmm-md` |
| replica launcher | `pmemd.cuda.MPI -groupfile` | multidir / `-replex` | `openmm-rest2` + MPI |
| protocol input | `.in` | `.mdp` / `.tpr` | concise Python |
| exchange engine | Amber | GROMACS | OpenMMTools |
| multistate storage | per-replica Amber outputs | per-replica files | one OpenMMTools NetCDF |
| physical data root | user-selected | user-selected | `$MD_DATA` |

There is **no groupfile**, because there is nothing per-replica to name: the replicas share one
topology, one base System and one starting state, and differ only by tau. This is a comparison of
user-facing execution models. **No claim of bitwise equivalence among engines is made or implied.**

## The ALA acceptance run

ACE-ALA-NME, ff19SB + OPC, NPT 300 K / 1 bar, 4 fs with HMR at 3.024 amu, six replicas over
tau 0 -> 0.5, 10 ps per-tau equilibration, 10 ps exchange interval, 2 ps solute output, 10 ps
whole-system output. **10 ns per replica, 1000 exchange attempts**, one MPI rank per GPU on six
RTX 3080s.

```
iterations          5000 of 5000        exchange attempts   1000 of 1000
production          10 000 ps/replica   wall clock          9.2 min (60 ns aggregate)
aggregate rate      9381 ns/day         storage             23.6 MB analysis + 270 MB checkpoint
```

Acceptance by neighbouring pair, over the whole run — computed from the stored mapping. (That run
predates the lifetime-statistics correction, so its `.out` summary showed only the last attempt;
the figures below were recomputed from the storage. A run made today reports these directly.)

| states | tau | accepted / attempted | rate |
|---|---|---|---|
| 0-1 | 0.0 - 0.1 | 232 / ~500 | 46.4% |
| 1-2 | 0.1 - 0.2 | 229 / ~500 | 45.8% |
| 2-3 | 0.2 - 0.3 | 237 / ~500 | 47.4% |
| 3-4 | 0.3 - 0.4 | 232 / ~500 | 46.4% |
| 4-5 | 0.4 - 0.5 | 261 / ~500 | 52.2% |

Every walker visited all six states, with 5 to 10 round trips from tau = 0 to tau = 0.5 and back.
Every one of the 5001 stored mapping rows is a permutation, so no state was ever unoccupied or
doubly occupied.

This says the ladder mixes and the machinery works. **It is one 10 ns run of a dipeptide and is not
a convergence claim** about any observable.

Implicit ALA (ff14SB + GBn2/mbondi3, NVT, four replicas, 100 ps/replica) and phenol/IPH
(Sage 2.2.1 + AM1-BCC, ff19SB/OPC, 100 ps/replica) both complete the same contract: NetCDF written,
mapping recorded, manifest verified. The phenol enhanced region is the 13-atom ligand and its omega
exclusion list is empty, which is correct — a ligand has no peptide bond.

## The legacy exchange loop

`templates/rest2_run.py` -- the custom Python exchange loop -- is **not** the production engine and
is not maintained as an alternative to OpenMMTools. It remains only for the older contract-managed
`md-gen --method REST2` route, whose already-generated datasets depend on its on-disk layout: the
per-replica directories, its own `exchange_attempts.csv`, and the stage fingerprints the shared
preflight checks. Deleting it would invalidate existing dataset provenance, which is worse than
keeping a legacy path that states plainly what it is -- and it does, in its module docstring and in
a line it prints at run time.

The two are kept in agreement rather than developed in parallel. Both compute acceptance with the
same `rest2_scaling.exchange_log_acceptance` and pair replicas with the same
`rest2_scaling.exchange_pairs`, and a test asserts they produce identical accept/reject sequences
from identical inputs and seeds. **A scientific capability added to one must be added to the
OpenMMTools path, not to the loop.**

New REST2 work uses `md-openmm setup` with `protocol: REST2`. Retiring
`md-gen --method REST2` entirely is a separate change: it needs a migration for the existing
contract-managed datasets and for AIS, which sources equilibrium frames from those runs.

## Limitations

- Pinned to OpenMMTools **0.26.0**. Any other version is refused until the contract tests are
  extended to it.
- The strided iteration model costs the measured overhead above. It is a deliberate trade for real
  intermediate solute coordinates.
- MPI multi-GPU needs `mpi4py` and a working MPI, which are **not** in `environment-ci.yml`.
  Single-process and single-GPU REST2 need neither.
- Smoke tests are picoseconds. **A smoke test is not validation, convergence, or evidence of ladder
  quality**, and the acceptance runs here are not either.
- Implicit-solvent **ligand** REST2 is not scientifically validated and is not claimed to be.
- Acceptance statistics in the `.out` and the manifest are **lifetime** figures summed over the
  whole stored history, including across resumes and extensions. The NetCDF remains authoritative.
