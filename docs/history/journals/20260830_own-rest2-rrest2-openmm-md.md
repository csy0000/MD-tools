# Own REST2 and rREST2 through `openmm-md`

| | |
|---|---|
| Date | 2026-08-30 |
| MD-templates | `feat/openmm-md-rest2-rrest2`, from `068ef2f0538a20260e2d40b566ca862e50c6b2c8` → `5c4edfba40a9724e5737c60ca3da9cee55dbf5b0` |
| MD-project | `dev5-openmm-md-rest2-rrest2`, from `ae6ca8e13284d0e6c4e2c5ff2cbd0ca2440998e9` → `a5fefd3` (+ this SHA-recording commit) |
| Instruction | [`../claudecode-instructions/20260830_own-rest2-rrest2-openmm-md.md`](../claudecode-instructions/20260830_own-rest2-rrest2-openmm-md.md) |

Both remote heads were fetched and verified before any edit. This supersedes the earlier decision
that `ReplicaExchangeSampler` must be the production controller.

---

## Environment

```text
python 3.12.14   openmm 8.6.0 (8.6.0.dev-c6173db)   netCDF4 1.7.4   numpy 2.4.6
mdtraj 1.11.1    mpi4py 4.1.2 + OpenMPI             md-data 0.2.0 @ 48628f9a
openmmtools 0.26.0 -- INSTALLED BUT UNUSED by generated production code
snakemake 9.26.1 (separate venv, for dry runs only)
```

Environment `$DATA_ROOT/software/md-stack/envs/openmm-rest2`. Hardware: 9 CUDA devices
(1 × RTX A5000, 8 × RTX 3080), driver 580.173.02.

## What changed

`openmm-rest2` is gone -- the executable, its template, the OpenMMTools runtime it drove and the
tests written against them. There is **one simulation executor**:

```text
openmm-md -i ... -p ... -s ... -c ... -o ... -x ... -r ...
openmm-md -ng N --groupfile FILE [--exchange-rule RULE.py] [--reservoir RES.yaml] -o ... -x ...
```

Nine new template modules, each with a tested boundary: `replica_protocol`, `replica_engine`,
`exchange_rules`, `replica_storage`, `replica_statistics`, `replica_validate`, `replica_driver`,
`replica_runtime` (the facade a generated input imports), and `rrest2_reservoir` plus the generated
`rrest2_exchange.py`. `source_ensemble.py` is the shared reader AIS and rREST2 both use.

### Decisions, stated once

- **Representation.** Contexts are fixed to thermodynamic states; complete configurations
  (positions, velocities, box) move between them. Both views are stored and tested to invert.
- **Ownership.** OpenMM owns System/Context/Integrator/State and every energy evaluation.
  MD-templates owns scheduling and rules. OpenMMTools is not imported by generated code.
- **Refresh order.** The neighbouring sweep runs first, the reservoir refresh second, recorded in
  the rule's `describe()` and the manifest.
- **Parallel policy.** World size is 1 or exactly the number of states. Nothing in between.
- **Energy matrix.** The full n×n is evaluated at each exchange, rank *r* computing row *r*. More
  work than a neighbour sweep needs; it makes the rule contract a pure lookup with no
  communication.

## Defects found by running, not by reading

1. **The odd/even phase alternated on the iteration number.** Exchanges land every
   `exchange_stride` iterations, so an even stride froze the parity: a three-state ladder proposed
   (1,2) six times and (0,1) **not once**. The phase now alternates on the exchange-attempt count.
2. **Resume continued from the last committed row.** The reporter commits every iteration while
   checkpoints are written less often, so an interruption left committed rows with no
   configurations behind them. Resume now continues from the last checkpoint and rewinds the rest;
   `segment == iteration + 1` is asserted.
3. **SIGTERM ended the process outright**, leaving the sidecar saying `running`.
4. **The stride skip did not zero OpenMMTools' proposal matrices** (found in the previous pass and
   carried into the owned runtime's design: skipped iterations record genuine zeros).
5. **Every OpenMM System carries default periodic box vectors**, so "vectors are non-zero" is true
   even for an implicit system. The implicit ladder carried a phantom box and rejected its own
   boxless reservoir frames. Periodicity is asked of the System.
6. **The reservoir read `pressure_bar is None` as "implicit".** This runtime is NVT, so pressure is
   None for an explicit fixed-volume ladder too, and an explicit reservoir was prepared with no
   boxes.
7. **`check_box_matches` existed but was never called.** An explicit refresh silently changed the
   ladder's density: two independently equilibrated systems do not share a box.
8. **Comparing box representations was itself wrong.** Reduction has a boundary case at
   `|c_x| = a_x/2` where `+a_x/2` and `-a_x/2` describe one rhombic dodecahedron. The check is now
   an integer change of basis.
9. **The companion-record block landed in the equilibration protocols too**, referencing a `frames`
   variable they do not have.

## The shared source reader

`source_ensemble.py` holds the rules AIS established; `ais_run.py` now delegates to it rather than
carrying a second copy. Its seed derivation is deliberately the repository's own multiplicative
hash, not a SHA-256 of the same inputs, and the selection stream is still
`("AIS", "source-selection")` -- so **every existing AIS project selects the configurations it
always did**. All 42 AIS tests pass unchanged apart from the generated-layout list, which gains
`source_ensemble.py`.

## The four ALA acceptance workflows

Six states, tau 0 → 0.5, 20 segments and 10 exchange attempts each, on CUDA. Picosecond smoke
sizes: **correctness workflows, not convergence evidence.**

| workflow | result |
|---|---|
| ALA implicit GBn2 REST2 | 20/20 segments, 10 exchanges, walkers mixed to `[3,1,4,5,0,2]`, adjacent-pair acceptance 40–100% |
| ALA implicit rREST2 | as above; **5/5 refreshes at state 5 only**, 5 distinct frames |
| ALA explicit ff19SB/OPC NVT REST2 | 20/20 segments; box **constant across every frame and walker** |
| ALA explicit NVT rREST2 | 5/5 refreshes at state 5, 5 distinct frames; box constant |

The reservoir manifest records `tau: 0.5` from
`ALA_imp_source/cMD_tau0p5/resolved_run.yaml`, with `tau_from_directory_name: false`; frame timing
from the same record; the window inclusive `[1.0, 20.0]` with 20 eligible frames and 10 chosen;
`hashed: false` with `mdtraj.iterload` chunking; and the Boltzmann/NVT contract, acceptance rule,
finite-reservoir approximation and both citations.

## MPI

Six ranks on CUDA, one state each:

```text
rank 0/6  CUDA device=0  drives state(s) [0]      rank 3/6  CUDA device=3  drives state(s) [3]
rank 1/6  CUDA device=1  drives state(s) [1]      rank 4/6  CUDA device=4  drives state(s) [4]
rank 2/6  CUDA device=2  drives state(s) [2]      rank 5/6  CUDA device=5  drives state(s) [5]
```

All six completed; rank 0 owns the manifest and the storage.

## Interruption, resume and extension

Verified on CPU with a real `SIGTERM`: no `restart.json`, sidecar `interrupted` with
`KeyboardInterrupt: signal 15`, storage resumable. `--resume` then completed the original budget
with `schedule ok: True` and balanced pair statistics (150/150). `--extend 3` took 12 → 18 segments
and 6 → 9 exchange iterations.

## Tests

| suite | result |
|---|---|
| MD-templates non-GPU | **536 passed**, 0 failed |
| MD-templates GPU (CUDA) | **77 passed**, 0 failed |
| MD-project | **355 passed**, 0 failed, 0 skipped |
| Snakemake dry run | 25 jobs across 6 systems |
| Workflow verify delegation | valid → `.verified`; truncated analysis and truncated checkpoint both refused |

79 new tests in `test_own_replica_exchange.py` plus 20 workflow tests. Two guards moved rather than
relaxed: the atom-identity guard follows `atom_identity` into the shared reader, and the
`openmm-md` import guard now separates module-level imports (standard library and OpenMM only, so a
single stage still runs once `md_templates` is gone) from grouped-mode imports deferred into the
path that needs them.

## Commits pushed

```text
MD-templates  feat/openmm-md-rest2-rrest2   (from 068ef2f0)
  3bbf896  Own the replica-exchange runtime, and make openmm-md the one executor
  5bf09f5  Generate REST2 and rREST2 through openmm-md, and retire openmm-rest2
  ec1b008  Add rREST2: a Boltzmann reservoir refresh of the hottest rung
  5c4edfb  Test the owned runtime, and document one executor rather than two

MD-project    dev5-openmm-md-rest2-rrest2   (from ae6ca8e1)
  60f9523  Pin MD-templates 5c4edfba: one executor, owned replica runtime
  ed02b6b  Run the four ALA workflows through openmm-md --groupfile
  a5fefd3  Document one executor, the reservoir contract, and journal the milestone
  <this>   Record the final SHAs, which a commit cannot contain for itself
```

Neither branch was merged, force-pushed or rewritten.

## Not run, and why

- **`ruff`** is not installed in this environment; `compileall` was used instead. Recorded as
  unavailable, not as a pass.
- **Long production runs.** Out of scope; the instruction asks for picosecond smoke sizes.
- **Multi-node MPI.** One host only.
- **OpenMMTools as a test oracle.** The optional cross-check against `ReplicaExchangeSampler` was
  not implemented; the production path does not depend on it, and the acceptance criterion is
  instead checked against its analytical form directly.

## Limitations

- **NVT only.** NPT exchange is refused rather than approximated.
- **v1 reservoirs are Boltzmann-weighted, complete-configuration and fixed-volume.** Non-Boltzmann,
  clustered and kinetic reservoirs are refused; each needs its own derived acceptance rule.
- An explicit rREST2 ladder must share a box with its reservoir source, so it starts from that
  source's equilibrated state. Two independently equilibrated systems are refused by the lattice
  check.
- The legacy `md-gen --method REST2` loop remains for the contract-managed route and its existing
  datasets; it is marked legacy and points at `openmm-md`.
- Smoke runs are picoseconds and validate neither ladder quality nor convergence.
- Implicit-solvent **ligand** REST2 remains scientifically unvalidated.
