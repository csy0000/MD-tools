# REST2 on OpenMMTools

| | |
|---|---|
| Date | 2026-08-29 |
| MD-templates branch | `feat/openmmtools-rest2`, from `feat/amber-like-openmm` at `6603c4c198574884b4aa7a61af7340a38e56e20c` |
| MD-project branch | `dev4-openmmtools-rest2`, from `dev3-amber-like` at `be75ad4fa9645c2a686260252da37f67be1a9c30` |
| Goal | retire the custom Python exchange loop as the production engine and put REST2 on `openmmtools.multistate.ReplicaExchangeSampler` with `MultiStateReporter` NetCDF |

Both remote heads matched the instruction exactly before any edit.

The architecture, unchanged from the instruction:

```text
existing MD-templates REST2 Hamiltonian scaling
+ openmmtools.multistate.ReplicaExchangeSampler
+ MultiStateReporter NetCDF storage/restart
```

---

## Phase 0 — the dependency, reported before any code

OpenMMTools 0.26.0, the specified tested target, was installed **nowhere**. `netCDF4` was absent
from the MD-templates environment and `mpi4py` from every environment on the machine. The
instruction says to inspect OpenMMTools' actual source before designing adapters, and that cannot be
done against a version that is not installed; designing from memory and labelling the result
"tested against 0.26.0" is the fabrication the instruction forbids.

Installing into the shared `openmm-8.6.0` environment was tried as a **dry run first**, and the
result is why it was not done:

```text
DOWNGRADED:
  pytorch    2.13.0-cuda130_mkl_py312  -->  2.12.0-cpu_mkl_py312
  libtorch   2.13.0-cuda130_mkl        -->  2.12.0-cpu_mkl
```

That strips CUDA out of PyTorch in the environment carrying the green suites. A dedicated
environment was built instead — additive, reversible, and pinned to `environment-ci.yml` plus
`openmmtools=0.26.0`, `netcdf4` and `cuda-version=13.0`. `mpi4py` and OpenMPI installed cleanly
afterwards, so the MPI tests the earlier report called impossible were in fact run.

## Phase 1 — what reading the source actually found

Two things that changed the design, neither of which could have been assumed:

**`MultiStateReporter` stores analysis-particle positions every iteration** and full coordinates
every `checkpoint_interval` iterations. That is the public hook the instruction asked to look for,
and it exists — so no private reporting extension was needed. It also fixes the time model: the
ITERATION becomes the solute output interval, and exchange is attempted every *n*th iteration.

**`LangevinDynamicsMove` uses `openmm.LangevinMiddleIntegrator`** — the identical integrator this
repository already uses everywhere. No integrator substitution, no splitting to reproduce, no
extension. `LangevinSplittingDynamicsMove` would have been the wrong choice: it uses OpenMMTools'
own integrator implementation.

**`mpiplus` does no device binding at all.** It distributes replicas across ranks and leaves the
platform entirely to the caller, so without an explicit rank→device map every rank builds its
Context on the default device and a six-replica ladder runs on GPU 0.

## Phase 2 — two upstream defects, found by running it

**`swap-neighbors` is unusable on NumPy ≥ 1.25**, and it is the scheme an ordered REST2 ladder
wants. `_mix_neighboring_replicas` locates replicas with `np.where(...)`, which returns a *tuple*;
indexing the energy matrix with it yields a 2-d array, and the acceptance test then calls
`math.exp` on it:

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

**A two-replica ladder silently exchanged at half its configured rate.** Upstream draws an offset
from `{0, 1}` and iterates `range(offset, n-1, 2)`; with two replicas the odd phase covers no pair,
so half the exchange iterations proposed nothing — with a healthy-looking log. This one was found
by writing the test the instruction asked for (#14) and watching it fail: 24 attempts where 40 were
due.

Both are corrected in one isolated module, `templates/rest2_openmmtools.py`, which **refuses any
OpenMMTools other than 0.26.0** because it subclasses private methods. The acceptance criterion is
not restated there: it calls this repository's own `exchange_log_acceptance`, so the ladder and its
tests share one definition rather than agreeing by inspection.

## Phase 3 — the honest trajectory, and what it costs

The required example wants solute coordinates every 2 ps while exchanging every 10 ps. Writing one
exchange-boundary configuration five times would satisfy a frame count and be a fabricated
trajectory. Instead the iteration is 2 ps and exchange is attempted every fifth one, so every
stored solute frame is a configuration the integrator actually visited — asserted by a test that
consecutive frames are never identical.

That costs a full reduced-potential matrix per iteration rather than per exchange, so it was
**measured rather than assumed**: 30.7 s against 25.6 s for the same 100 ps per replica,
**+19.9%** for 5× finer sampling on ALA in OPC. Setting `output_interval` equal to the exchange
interval makes the stride 1 and removes it entirely.

## Phase 4 — acceptance

ACE-ALA-NME, ff19SB/OPC, NPT 300 K/1 bar, 4 fs with HMR, six replicas over tau 0 → 0.5, **10 ns per
replica and 1000 exchange attempts**, one MPI rank per GPU on six RTX 3080s:

```text
iterations   5000 / 5000        exchange attempts   1000 / 1000
wall clock   9.2 min            aggregate rate      9381 ns/day
acceptance   46-52% on every neighbouring pair
round trips  5-10 per walker; every walker visited all six states
```

All 5001 stored mapping rows are permutations — no state ever unoccupied or doubly occupied. This
says the ladder mixes and the machinery works. It is one 10 ns dipeptide run and is **not** a
convergence claim about any observable.

Implicit ALA (GBn2, whole-system scaling, four replicas) and phenol/IPH (Sage 2.2.1 + AM1-BCC,
13-atom ligand enhanced region, empty omega list — correct, a ligand has no peptide bond) both
complete the same contract.

## Phase 5 — this repository

The pin moves to `f2c3c48b06f3f8492f7ce0776a05b75c4fc15757`, and every example configuration with
it. `workflow/rest2.smk` is new: `docs/architecture.md` records why the old `workflow/` was deleted
and names the case that justifies one — "a campaign that needs scheduling, GPU assignment or
staging" — which a six-replica ladder with a hard equilibration barrier is.

Its `verify` rule is the committed-analysis boundary, and it does **not** parse a log line. It
requires the atomic manifest to record `run_status: completed`, the promised exchange budget to
equal the recorded one, and every referenced NetCDF to exist. Both failure modes were tested
against a real finished run with its manifest falsified, and both refused without creating the
marker.

## What was corrected in the checks themselves

Three guards were changed, and in each case the change makes them *stronger*:

- The platform-policy audit required CUDA of any file naming `Reference`. The rule exists for
  **propagation**; the REST2 scaling tests propagate nothing and compare energies at 1e-12, which
  single precision cannot carry. It now asks whether a file propagates first — with two new tests
  proving it did not become vacuous.
- `test_example_coherence.py` hard-coded the expected MD-templates commit. A literal means every
  repin edits the test too, and the test then proves only that someone edited both. It now reads
  `components.lock.yaml` and asserts every configuration agrees with it.
- The component-pin reachability guard applied only when the lock said `dev`, leaving every feature
  branch — which is what this project actually pins — unchecked. It now holds for **whatever** ref
  the lock names, and it verified the new pin against the real remote rather than skipping.

## Results

| suite | result |
|---|---|
| MD-templates, non-GPU | **483 passed**, 0 failed |
| MD-templates, GPU (CUDA) | **89 passed**, 0 failed |
| MD-project | **336 passed**, 0 failed, 0 skipped |
| MPI multi-GPU REST2 | 6 ranks on devices 0-5, one rank per device, completed |
| CPU REST2 smoke | completed, manifest verified |

## Known limitations

- Pinned to OpenMMTools **0.26.0**; any other version is refused until the contract tests extend
  to it.
- `md-gen --method REST2` still generates the legacy exchange loop. It is marked as legacy in its
  docstring, at run time, at the call site and in the documentation, and is kept in agreement with
  the OpenMMTools engine by a shared criterion and a test — but retiring it outright needs a
  migration for the contract-managed datasets and for AIS, which sources frames from those runs.
  **Flagged, not silently done.**
- Smoke runs are picoseconds and are not validation or convergence.
- Implicit-solvent **ligand** REST2 remains unvalidated and is not claimed otherwise.
