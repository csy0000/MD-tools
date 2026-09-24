# Placement, MPS and the concurrency comparison — measured evidence

Date: 2026-09-17. Branch `feature/mps-placement`, on top of the 0.6.0 development branch.
Hardware: a nine-GPU workstation, 48 logical CPUs (2 sockets x 12 cores x 2 threads), driver
580.173.02 / CUDA 13.0, all cards compute capability 8.6. Four identical RTX 3080s were reserved
for this work; the other cards belonged to other runs and were hidden from every launch with
`CUDA_VISIBLE_DEVICES`, because the throughput measurement opens a Context on every VISIBLE device.

Everything below was produced by running the shipped command. Nothing is inferred from source.

## 1. What the placement does, on hardware

Four workers, four cards, `md-run --check` (which creates nothing):

```text
REST2: --check passed. Nothing was created.
  platform          CUDA (machine-config)
  device            2  [measured throughput (balanced) (rank 2 of 4, local rank 2, 1 worker(s) on this device)]
  cpus              12 bound (12-17,36-41) of 48 usable, 4 worker(s)
  throughput        18027, 17289, 17096, 16792 steps/s by device
  mps               absent
```

The four rates are the four cards, measured one at a time with this run's own System. They agree
to within 7%, as four identical cards should. The CPU blocks were `0-5,24-29`, `6-11,30-35`,
`12-17,36-41` and `18-23,42-47`: twelve logical CPUs each, a core's two hardware threads kept
together, and no CPU in two blocks.

## 2. The MPS verdict, both ways, same code and same card

The pair is the evidence, not either line alone. The only difference is whether the process is
genuinely an MPS client; the verdict is the driver's classification, never an inference from where
the processes were placed.

| | what the run did |
|---|---|
| no daemon reachable | **refused**, nothing written: *"device 0 hosts 4 worker(s), and NVIDIA MPS is not-a-client for this process (the driver lists process 2661979 as an ordinary CUDA process (C), not an MPS client)"* |
| daemon reachable | **accepted and completed**: `restart.json` records `placement.mps.status: verified`, `shared_devices: true`, every worker on one device with `co_tenants: 4`, and four state trajectories |

Twelve workers on four cards, under MPS: three workers per card, `co_tenants: 3` for every worker,
twelve state trajectories, four CPUs per worker, 48 distinct CPUs across the twelve blocks.

## 3. The concurrency comparison

Identical workload in every arm: alanine dipeptide in explicit TIP3P, 1760 particles, four
replicas, 20 exchanges x 1000 steps = 80,000 integration steps, one seed, one equilibrated
starting state, each arm a fresh copy of the same generated run tree. Three repeats per arm;
the table gives the median with the range in brackets.

| arm | placement | wall time (s) | aggregate steps/s | peak GPU memory |
|---|---|---|---|---|
| sequential | 1 worker, 4 replicas in turn, 1 card | **12.58** [12.52–12.60] | 6358 | 980 MiB |
| concurrent, one card each | 4 workers, 4 cards | 17.89 [16.84–18.54] | 4473 | 924 MiB on the busiest card, ~240 MiB on the others |
| concurrent, shared card, MPS | 4 workers, 1 card, MPS verified | 21.33 [19.81–21.57] | 3750 | 1169–1445 MiB |
| *(unsupported)* concurrent, shared card, no MPS | 4 workers, 1 card, verdict forced | *19.22* [19.08–19.55] | *4162* | *1201–1486 MiB* |

**At this size concurrency costs time, and that is reported as measured.** (The larger workload
below reverses it.) A 1000-step segment of a
1760-particle system is a few milliseconds of GPU work per replica, while each of the 20 exchange
boundaries pays an MPI barrier and a gather, and the launch pays four Context creations and four
sets of reporters. That overhead is larger than the work it overlaps, so the serial ladder wins.
The plan anticipated this regime for a 22-atom system; it still holds at 1760 particles.

This is a FLOOR measurement, not a verdict on concurrency. The crossover is a property of the
per-segment GPU work against the barrier cost, so it moves with system size and with the number of
steps between exchanges. A ladder with tens of thousands of particles and longer segments is a
different regime, and nothing here measures it. **No speedup is claimed for any system on the basis
of this table.**

MPS did not beat time-slicing here either (21.3 s against 19.2 s). With kernels this small the MPS
server's own scheduling is a cost rather than a saving. What MPS buys at this size is not speed: it
is that four workers on one card is a configuration the tool will run at all.

### The same comparison at 23,659 particles, where the answer changes

The 1760-particle table is a floor, so the comparison was repeated on a larger system: the same
peptide in a 3.5 nm water box, 23,659 particles, four replicas, 6 exchanges x 5000 steps = 120,000
integration steps, three repeats, everything else identical.

| arm | placement | wall time (s) | aggregate steps/s | vs sequential |
|---|---|---|---|---|
| sequential | 1 worker, 4 replicas in turn, 1 card | 45.49 [42.83–46.51] | 2638 | — |
| concurrent, one card each | 4 workers, 4 cards | **34.56** [32.91–35.60] | 3472 | **1.32x faster** |
| concurrent, shared card, MPS | 4 workers, 1 card, MPS verified | 46.63 [44.38–52.19] | 2574 | 0.98x |
| *(unsupported)* concurrent, shared card, no MPS | 4 workers, 1 card, verdict forced | *58.87* [54.14–60.92] | *2038* | *0.77x* |

Three findings, and they are different claims:

1. **Concurrency pays once the kernels are large enough.** Four workers on four cards are 1.32x
   faster than the serial ladder here, against 0.70x at 1760 particles. The crossover is real and
   it is a property of per-segment GPU work against the fixed per-boundary cost, not of the code.
2. **MPS makes a shared card faster: 1.26x** (46.63 s against 58.87 s). At 1760 particles it was
   the other way round (21.3 s against 19.2 s), because the server's scheduling cost more than the
   kernels it overlapped. This is the measurement that answers "does MPS make REST2 faster": yes,
   for sharing, at a size where the work per segment is real.
3. **Sharing one card is still not faster than using it serially** (46.63 s against 45.49 s, within
   the repeat spread). Four workers on one GPU do the same total work on the same silicon and pay
   four Contexts for it. What sharing buys is that a ladder runs at all when there are fewer cards
   than replicas — and MPS is what makes that sharing cost 1.26x less than it otherwise would.

The speedup to quote for concurrency is therefore the multi-card one, and the speedup to quote for
MPS is against unmanaged sharing, never against the serial ladder.

### The fourth arm, and why it is set apart

Four workers sharing one card WITHOUT MPS is a configuration the tool refuses by design, and taking
the daemon down does not make it runnable: the rule requires `verified`, and `absent` is not
`verified` any more than `not-a-client` is. The only way to time it is `MD_TOOLS_FORCE_MPS_VERDICT`,
a test seam that makes the preflight accept what it would otherwise refuse. That row therefore
measures something no user can run, and it is marked unsupported for that reason. The seam prints a
warning to stderr on every use and the run record carries `placement.mps.forced: true` beside the
forced `verified`, so a record produced this way can be told from a real one by a field rather than
by reading prose.

## 4. Defects this work found, all fixed here

| found by | defect |
|---|---|
| running the refusal | the message pointed at a documentation section that does not exist; a test now asserts the quoted heading is a real heading |
| moving the daemon | a STOPPED daemon leaves `control`, `control_lock` and `control_privileged` behind and removes only its `.pid` file, so "a control process exists and the socket exists" reported a daemon dead for half an hour. The status is read from the pid file and that pid in `/proc`; sockets with no pid file are `unknown`, never running |
| the same | `unknown` was reported as `requested-not-detected`: being-asked outranked not-being-able-to-tell |
| the forced arm | a forced verdict recorded a bare `verified` beside a daemon that was not running; `forced` is now its own field and the seam warns on stderr |

Two properties of MPS itself were learned the same way and are now in `docs/basics/md-run.md`: a daemon on
the DEFAULT pipe directory captures every new CUDA process on the host, and a client addresses the
daemon's devices as `0..n-1` of the set the daemon was started with — asking for the physical
numbers gives `CUDA_ERROR_NO_DEVICE (100)` on a card `nvidia-smi` shows idle and healthy.

## 5. What is still unproven

* The crossover point itself. Two workloads bracket it -- 1760 particles where concurrency costs
  30% and 23,659 where it pays 1.32x -- but nothing here locates where it falls, and it moves with
  the number of steps between exchanges as well as with system size.
* AIS paths sharing GPUs under MPS. AIS reaches placement through the same shared preflight and the
  same record, and `tests/test_placement_cuda.py` covers REST2 only.
* Multi-node placement. The CPU rule and the local-rank derivation are written per node and unit
  tested with two hosts, but every measurement here is single-node.
* Physical device identity in the record. With `CUDA_DEVICE_ORDER=PCI_BUS_ID` the ordinals are
  stable, but the record does not resolve them to UUIDs, and under MPS the ordinals are the
  daemon's, not the host's.

## 6. Reproducing it

```bash
# one worker per card, nothing shared
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<your cards> \
  mpirun --bind-to none -n 4 md-openmm md-run -ng 4 -i ../input/REST2.in \
         -p ../build/built.pdb --groupfile remd_groupfile.1 -odir . --check

# four workers on one card, as an MPS client: the pipe directory AND the daemon's own numbering
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=<the daemon's private pipe directory> \
  mpirun --bind-to none -n 4 md-openmm md-run -ng 4 ...
```

The suite's own CUDA coverage is `tests/test_placement_cuda.py`, which takes the physical cards from
`MD_TOOLS_TEST_CARDS` and the client-side numbering from `MD_TOOLS_TEST_MPS_CARDS`, and skips rather
than guesses when the latter is unset.
