# REST2 on OpenMMTools

REST2 replica exchange runs on
[`openmmtools.multistate.ReplicaExchangeSampler`](https://openmmtools.readthedocs.io/), with
`MultiStateReporter` NetCDF as the authoritative storage. This repository keeps the REST2
Hamiltonian; OpenMMTools keeps everything about running a ladder.

| owned by | what |
|---|---|
| **MD-templates** | the REST2 scaling convention, the tau ladder, physical-time arithmetic, the enhanced region and omega exclusions, input identity checks, the completion manifest |
| **OpenMMTools** | propagation, reduced potentials, exchange decisions, thermodynamic-state assignment, multistate NetCDF storage, checkpointing, restart, extension, walker-to-state mapping |

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

## The extension, and why it is pinned

`templates/rest2_openmmtools.py` is the whole of the OpenMMTools extension. It subclasses three
private methods, so it **refuses any OpenMMTools other than 0.26.0** rather than silently following
a changed parent. Two corrections, both established by reading the installed source and running it:

**1. `swap-neighbors` is unusable on NumPy >= 1.25.** In 0.26.0,
`ReplicaExchangeSampler._mix_neighboring_replicas` locates replicas with `np.where(...)`, which
returns a *tuple* of arrays. Indexing the energy matrix with that tuple yields a 2-d array, and the
acceptance test then calls `math.exp` on it:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

The ordered REST2 ladder wants neighbour swaps, so the scheme cannot simply be avoided. The
override is the same algorithm with scalar indices, and it computes the criterion by calling this
repository's own `exchange_log_acceptance` — so the ladder and its tests share one definition of
the Metropolis criterion rather than agreeing by inspection. `swap-all` is unaffected and remains
available.

**2. A two-replica ladder skipped every second exchange iteration.** Upstream draws an offset from
`{0, 1}` and iterates `range(offset, n-1, 2)`; with two replicas the odd phase covers no pair at
all, so half the exchange iterations proposed nothing and the ladder exchanged at half its
configured rate, with a healthy-looking log. The offset is now drawn only from phases that contain
a pair. The alternating scheme is unchanged for three or more replicas, where both phases do.

`equilibrate` is overridden for a third reason: upstream calls `_mix_replicas` on every
equilibration iteration with the counter still at 0, which the stride test reads as an exchange
iteration — so every replica would swap during the per-tau relaxation that exists to hold it at its
own tau. Tau equilibration does not count toward production; upstream does not increment the
iteration counter for it.

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
- Acceptance statistics printed in the `.out` are for the last exchange attempt only, because
  OpenMMTools resets its proposal matrices each mixing call. The NetCDF holds the full history.
