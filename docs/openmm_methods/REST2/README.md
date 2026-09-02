# REST2 — replica exchange with solute tempering

Hamiltonian replica exchange at **one physical temperature**. The rungs of the ladder differ by how
much of the solute's Hamiltonian is scaled down, not by how hot the thermostat is
[@wang2011rest2]. That is the whole point: only the solute's internal and solute–environment terms
are softened, so the cost of the ladder does not grow with the number of solvent molecules the way
temperature REMD's does.

## Ensemble

**NVT.** The runtime installs no barostat and a requested pressure is refused. Exchanging complete
configurations under NPT would also have to exchange volumes and carry the *pV* work; that is
deliberately not implemented rather than approximated.

Every replica runs at the same thermostat temperature and the same β, so **an exchange never
rescales velocities**. Velocity rescaling belongs to temperature REMD, where the rungs differ in β;
here it would inject or remove energy at every accepted swap. A test asserts that no runtime module
does it.

## What is scaled, and what is not

τ is the only persisted scaling coordinate. Everything else is derived from it:

| term | factor |
|---|---|
| solute–solute nonbonded, 1-4, solute torsions, CMAP | `(1 − τ)²` |
| solute–environment nonbonded | `(1 − τ)` |
| generalized Born (implicit) | `(1 − τ)` |
| environment–environment | `1` — untouched |
| bonds and angles | `1` — untouched |
| ordinary amide ω torsions | `1` — **unscaled by convention** |

Leaving ω unscaled is *this repository's* convention, not textbook REST2. A softened ω barrier lets
the backbone sample *cis* amides that the force field was never fit to describe, so the hot rungs
would explore a region the parameters do not cover. The exclusion is expressed as **central bonds**:
every torsion term that shares a peptide C–N bond is excluded, so no term is missed because it was
enumerated differently.

The resolved exclusions are written to `solute.yaml` beside the run, with the topology digest they
were derived against. See [`ScalingSelection`](#the-selection-file).

The scaling is applied by `md_tools.rest2.REST2Scaler`. Fixed-τ cMD, REST2, rREST2 and AIS all use
that one implementation.

## What MD-tools implements

* a linear τ ladder from 0 to `tau_max`, one Context per **thermodynamic state**;
* neighbouring-pair exchange with a Metropolis criterion evaluated by recomputing both
  Hamiltonians, never by comparing two stored claims;
* one trajectory per fixed state (`remd0.nc`, `remd1.nc`, …), never per walker;
* an Amber-format `rem.log`, parseable by `cpptraj` as type Hamiltonian;
* four independent schedules — exchange, whole-system output, solute output, checkpointing;
* checkpoint, restart and out-of-place extension.

## What it does not implement

* no NPT ladder (see *Ensemble* above);
* no temperature REMD, and no velocity rescaling on exchange;
* no automatic ladder optimisation — `number_of_replicas` and `tau_max` are yours;
* OpenMMTools is **not** imported by generated production code. It remains an optional oracle in
  tests, and the package works without it.

## Required inputs

`built.pdb`, `built.xml`, and an equilibrated state to start every rung from — normally
`eq_npt_free.xml` from the cMD chain that `build-md` generates alongside the ladder.

## Minimal sequence

```bash
md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log
md-openmm build-md  -odir ./md_script/ --config example.config
cd md_script && ./run.sh
```

The ladder itself, driven directly:

```bash
python REST2.py -p ../built.pdb -s ../built.xml -c eq_npt_free.xml -log REST2.log
```

One process holds every state and lets OpenMM choose the device. Under MPI, one rank per state:

```bash
mpiexec -n 4 python REST2.py -p ../built.pdb -s ../built.xml -c eq_npt_free.xml -log REST2.log
```

A world size that is neither 1 nor exactly the number of states is refused.

## Generated files

```text
md_script/
├── resolved.config      the single resolved declaration
├── min.py … eq_npt_free.py   the equilibration chain the ladder starts from
├── REST2.py             the compact ladder entry point
└── run.sh
```

Running produces `solute.yaml` (the resolved scaling selection), `remd0.nc` … `remdN.nc` (one per
state), `rem.log`, `REST2.nc` (the analysis file), `REST2_checkpoint.nc`, `REST2.out` and
`REST2.log` (the machine record).

## Restart and continuation

`--resume` continues from the checkpoint; the run state records `interrupted` or `failed` with a
reason when it does not finish, and only a run that did finish writes a completion manifest.

**Extension is out of place.** `--extend-from` writes a new directory holding only the new segment
and pins the parent by content. The parent is left byte-for-byte unchanged, so a chain is a
sequence of immutable segments rather than a file that grows and loses its own history.

## Storage, restart and validation

Four records, four jobs:

| record | written | job |
|---|---|---|
| analysis NetCDF (`-x`) | every committed iteration | the authoritative history, plus the scientific identity written **before** propagation begins |
| checkpoint NetCDF | every whole-output interval | configurations, mapping, RNG states, rule state, iteration and budget |
| `<stem>.runstate.json` | atomically, on transition | `initialized` → `running` → `completed`/`interrupted`/`failed`; never claims completion |
| `restart.json` (`-r`) | atomically, at the end | evidence of completion; **never** a precondition for resuming |

Each stream's marker is written **after** every array for that row, so an interrupted write leaves
the counter on the previous complete row. A marker may lag its data but never lead it, and one that
leads is refused as a file disagreeing with itself.

A resume continues from **the last checkpoint**, not the last committed row: the exchange stream
commits every attempt while checkpoints are written less often, so an interruption leaves committed
rows with no configurations behind them. Those rows are rewound. Before anything is opened for
writing the stored output is read and validated — markers within their rows, steps strictly
increasing in every stream, and the stored exchange steps equal to the *scheduled* ones, because a
dropped attempt in the middle still leaves steps increasing and can still reach the budget.

`--resume` needs no `restart.json`. `--extend N` requires a run that reached its budget and adds N
attempts to **the budget that run reached**, not to the one in the configuration — a run already
extended stores a larger budget than the configuration describes.

An interruption is neither success nor failure and exits with its own status (130). Verification
**opens and reads** the files: a file that exists is what a crashed run leaves behind, so existence
is never treated as completion.

## The selection file

`solute.yaml` carries a schema version, the topology digest it was derived against, the solute atom
indices, and the unscaled torsion central bonds with readable residue and atom labels. Supplying
your own is supported; it is validated rather than trusted:

* the topology digest must match the topology actually loaded;
* every atom index must be in range and every named bond must exist;
* duplicates are refused;
* an exclusion that matches **no** torsion is refused rather than silently ignored.

If no file is supplied, the classifier derives the selection and writes the resolved file.

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `rest2.number_of_replicas` | 4 | the number of states. Too few and neighbouring rungs never exchange |
| `rest2.tau_max` | 0.5 | the top rung. τ = 0.5 scales solute–solute terms by 0.25 |
| `rest2.exchange_interval_steps` | 5000 | steps between exchange attempts |
| `rest2.number_of_exchanges` | 500 | the length of the run, in exchanges |
| `rest2.state_trajectory` | true | one trajectory per state |
| `rest2.rem_log` | true | the Amber-format exchange log |

## Reporting and registration

`REST2.log` is the machine record. The per-state trajectories and `rem.log` are what an analysis
reads; `cpptraj` parses both. Register the finished tree with `md-openmm data-register` — see
[data registration](../../data_register/README.md).

## Limitations

* The ladder is linear in τ. Nothing here optimises the spacing for a target acceptance ratio.
* Acceptance is reported, not enforced. A ladder that never exchanges completes and says so; it is
  not automatically repaired.
* NVT only.
* The ω convention above is a deliberate deviation from textbook REST2 and must be stated in any
  write-up that uses this implementation.
* Implicit-solvent **ligand** REST2 remains scientifically unvalidated.
* `--extend` and `--resume` are coordinated across ranks at an event boundary; only rank 0 opens
  the analysis storage for writing, in any mode.
* Smoke runs are picoseconds and validate neither ladder quality nor convergence.
* Datasets registered before v0.5.0 were produced by a retired generator. Their provenance records
  that and is not rewritten: only the way a ladder is generated changed, not the Hamiltonian it
  runs.

## References

REST2 [@wang2011rest2]; the force fields and integrator settings in
[scientific defaults](../../scientific-defaults.md).
