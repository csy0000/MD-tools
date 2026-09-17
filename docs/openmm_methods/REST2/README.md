# REST2 — replica exchange with solute tempering

Hamiltonian replica exchange at **one physical temperature**. The rungs of the ladder differ by how
much of the solute's Hamiltonian is scaled down, not by how hot the thermostat is
[@wang2011rest2]. That is the whole point: only the solute's internal and solute–environment terms
are softened, so the cost of the ladder does not grow with the number of solvent molecules the way
temperature REMD's does.


## The two example files beside this README

| file | what it is |
|---|---|
| [`example.config`](example.config) | what you hand to `md-openmm build-md` |
| [`example.in`](example.in) | what `md-run` then reads — the ladder's production stage |

`build-md` generates the `.in` from the `.config`; you do not normally write one by
hand. It is shipped here because it is the file the run actually reads, and because
every setting in it carries the schema's own description as a comment — so the meaning
of a key can be looked up where it is used rather than in the source.

`resolved.config`, written beside the `.in` at run time, stays AUTHORITATIVE: the `.in`
is resolved into it, and that resolved document is what the run reads.

Both are checked by `tests/test_method_example_inputs.py`, which regenerates the `.in`
from the `.config` and fails if they have drifted — an example that no longer matches
the engine is worse than none.

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
| **unscaled torsions**: every proper torsion across an ordinary amide C–N (ω), an aromatic ring bond or another double bond; every improper | `1` — **unscaled by convention** |

Leaving these unscaled is *this repository's* convention (v3, `rest2-unscaled-torsions`), not
textbook REST2. A softened ω barrier lets the backbone sample *cis* amides the force field was never
fit to describe; a softened ring or double-bond torsion lets a hot state bend a planar group, and a
softened improper lets it invert a centre. In every case the hot states would explore geometry
state 0 never visits, and exchanges would stop being useful. The exclusion is expressed as
**central bonds**: every proper torsion term across such a bond is left unscaled, so no term is
missed because it was enumerated differently. The ARG guanidinium counts as double bonds.

**How a bond is classified.** Protein residues from a residue table (PHE, TYR, TRP, HIS rings; ARG
NE–CZ, CZ–NH1, CZ–NH2). A small molecule from its bond orders, read from an SDF (see
[the scaler](#the-scaled-states-build-top-rest2-scaler)). A torsion is proper or improper by the
bond graph: a bonded chain is proper; one atom bonded to the other three is improper. **A torsion
that cannot be classified is refused**, never silently scaled. A ring-locked amide nitrogen
(proline, or a ring of at most `max_proline_ring_size` atoms) keeps its ω eligible for scaling.

## The scaled states: `build-top --rest2-scaler`

**The ladder scales nothing.** Its states are built ONCE, as files, before any run:

```bash
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb --config build/scaler.config
```

```yaml
# build/scaler.config
method: REST2              # REST2 (a ladder), cMD (one hot state) or AIS (the V0 end state)
schedule:
  kind: linear
  n_states: 4
  tau_min: 0.0             # default
  tau_max: 0.5
# unscaled_torsions: true  # default; false scales every solute torsion
# sdf_filelist: {MO1: MO1.sdf, MO2: MO2.sdf}
```

It writes `build/REST2/`:

```text
build/REST2/
├── system_state0.xml … system_state3.xml   one scaled System per state, tau 0 … 0.5
├── scaler.yaml       the record: source System sha256, every state's tau and sha256, the solute,
│                     every unscaled central bond and improper, the SDFs used
├── scaler.log
└── <RESNAME>-unscaled.png   each small molecule, its unscaled torsions' bonds in red
```

A small molecule's bond orders come from `sdf_filelist`, else `<RESNAME>.sdf` beside the System,
else `built.sdf` when it is the only non-standard residue; with several and no mapping, the scaler
refuses rather than guesses. The directory appears complete or not at all, and `--overwrite` moves
the previous set aside rather than deleting it.

`build-md` refuses a REST2 configuration until `build/REST2/scaler.yaml` exists, was built from
this `built.xml`, and holds exactly the configured taus. Every run integrates those files as they
are; a state is never scaled a second time.

## What MD-tools implements

* a linear τ ladder from 0 to `tau_max`, one Context per **thermodynamic state**;
* neighbouring-pair exchange with a Metropolis criterion evaluated by recomputing both
  Hamiltonians, never by comparing two stored claims;
* one trajectory per fixed state (`solute_state0_prod1.nc`, `solute_state1_prod1.nc`, …, plus
  `whole_state<i>_prod<N>.nc` when a whole-system cadence is set), never per walker;
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

`built.pdb`, `built.xml`, the saved states in `build/REST2/`, and an equilibrated state to start
every rung from — normally `eq_npt_free.xml` from the cMD chain that `build-md` generates alongside
the ladder.

## Minimal sequence

```bash
md-openmm build-top -i ALA.pdb \
    -os build/built.xml -op build/built.pdb -log build/built.log
md-openmm build-top --rest2-scaler -s build/built.xml -p build/built.pdb \
    --config build/scaler.config
md-openmm build-md  -odir ./REST2-run1 --config example.config
cd REST2-run1 && ./run.sh
```

The ladder itself, driven directly:

```bash
python REST2.py -p ../build/built.pdb -ng 4 --groupfile remd_groupfile.1 -log REST2.log
```

One process holds every state and lets OpenMM choose the device. Under MPI, one rank per state:

```bash
mpiexec -n 4 python REST2.py -p ../build/built.pdb -ng 4 \
    --groupfile remd_groupfile.1 -log REST2.log

# or, the Amber-like way -- one rank per thermodynamic state, `-ng` checked against both the
# configured replica count and the MPI world size:
mpirun -n 4 md-openmm md-run -ng 4 -i ../input/REST2.in -p ../build/built.pdb \
    --groupfile remd_groupfile.1 -odir . \
    -o remd_records/REST2_prod1.out -log remd_records/REST2_prod1.log \
    -r remd_records/restart_prod1.json
```

**A ladder reads `-s` only from its group file.** Every line names one saved state,
`-s ../build/REST2/system_state<i>.xml`, state i on line i, all from one `scaler.yaml` whose taus
are the ladder's. There is no single `-s` for the launch to carry — one would claim one Hamiltonian
for every rung — so `-s` on the command line is refused by name, and so is a group file naming
anything other than saved states.

**`--groupfile` always comes with `-ng`**, and the group file also carries each rung's own `-c`, so
none is passed on the command line. `-ng` states how many groups the file is expected to hold: a
truncated group file would otherwise run a shorter ladder silently. A run with no group file at all
is refused — there is no second way to describe a coordinated run.

A world size that is neither 1 nor exactly the number of states is refused.

## Equilibrating every rung under its own tau

By default the equilibration chain runs once, at τ = 0, and every rung starts from its end state.
`rest2.equilibration_per_tau: true` (off by default, REST2 and rREST2 only) runs the
equilibration **on every rung, under that rung's own τ**, instead:

| solvent | the τ = 0 chain (`run.sh`) | the ladder's `-c` | then, on every rung including τ = 0 |
|---|---|---|---|
| implicit | `min` only | `min.xml` | `eq_nvt_posres`, `eq_nvt_posres_2`, `eq_nvt_free` |
| explicit | `min`, `eq_nvt_posres`, `eq_npt_posres`, `eq_npt_free` — the NPT stages fix the box | `eq_npt_free.xml` | the same three, at that box |

The per-rung stages are the fixed-volume stages a scaled run gets — a scaled Hamiltonian is NVT
throughout, and a ladder's rungs share one volume — with the lengths `stages.restrained_nvt_steps`,
`stages.restrained_npt_steps` and `stages.unrestrained_npt_steps`; a stage of 0 steps is skipped
and all three at 0 is refused. Each is done as the stage chain does it: a fresh integrator seeded
`derive_seed(seed, stage, "state<i>")`, the previous stage's positions, velocities and box
installed (velocities are carried, never redrawn), then the restraint strength set; the restraint
is the chain's own Force, on the solute, towards the topology's coordinates, on a **copy** of the
rung — the rung Systems the ladder propagates never carry it.

Order: these stages, then `rest2.equilibration_steps` (if any), then the step-0 observation and
the first exchange. None of it is production. Each rung's end state is kept as
`per_tau_state<i>.xml`, described (seeds, stages, digests) by `per_tau_equilibration.json` and
recorded in `restart.json` under `per_tau_equilibration`; no trajectory or CV series is written
for it. A ladder interrupted during it has taken no exchange step and has no checkpoint, so
`--resume` refuses it by name — rerun with `--overwrite`; the equilibration is deterministic from
the recorded seeds. An exported reference bundle performs it too, with the same code.

## Restraining every rung

A ladder may carry torsion restraints: `umbrella.file` names a restraint definition, exactly as
`protocol: umbrella` does, and `collective_variables.file` must name the definition those
restraints resolve their torsions against.

```yaml
collective_variables: {file: cv.yaml, interval_steps: 500}
umbrella: {file: umbrella.yaml}      # the SAME restraints on every rung
```

**The same restraints on every rung, and that is the point.** The exchange criterion compares
`u_i(x)` and `u_j(x)`, each evaluated by installing a configuration in that rung's own Context, so
a bias `W(x)` present identically on both rungs enters both reduced potentials and cancels from
`log alpha` exactly. The ladder therefore samples the restrained ensemble at every rung with an
acceptance probability that is the unrestrained one. Per-rung variation is not offered: a bias that
differed between rungs would enter the acceptance probability and tilt the ladder towards whichever
rung restrains least, silently and with every output still well formed.

The restraint is added **after** the REST2 scaling and is never scaled by τ — it is not part of the
molecular Hamiltonian REST2 weakens, and a τ-dependent bias would not cancel. Each restraint's
force constant rides on its own per-torsion `scale`, so one definition may mix strengths and forms
(`harmonic`, `flat_bottom`) freely; the force's global parameter defaults to 1.0, so a rung is
biased from its first step without a runtime call. `restart.json` records the definition file and
every resolved restraint — which four atoms, which centre, which constant — under
`scientific_identity.torsion_restraints`, and the rank log prints them under "Torsion restraints".

Tested by arithmetic rather than by sampling: `tests/test_ladder_torsion_restraints.py` computes
`log alpha` with and without the bias at the same configurations and requires equality to 1e-6
kJ/mol, with a counter-example showing a bias that differs between rungs does **not** cancel;
`tests/test_hprest2_gpu_evidence.py` repeats the identity on a real CUDA ladder's own states.

Two consequences worth stating. A restrained and an unrestrained ladder from the same seed do not
accept the same exchanges — the bias changes the forces, so their trajectories diverge from the
first propagation; what is identical is the criterion at the same configurations. And a restrained
ladder cannot be exported as a reference bundle: `verify_rungs.py` rebuilds each rung from rung 0
by scaling, which is not how a restraint is applied, so `export-reference` refuses it by name.

## Generated files

`build/`, `min/` and `input/` belong to the SYSTEM and are shared by every run beside them; only
`REST2-run1/` belongs to this run. See [the layout](../../run-layout.md).

```text
ALA/                          the dataset root -- this is what you register
├── build/                    built.xml  built.pdb  built.log
│   └── REST2/                system_state<n>.xml  scaler.yaml  <RESNAME>-unscaled.png
├── min/                      the minimised structure, shared
├── input/                    min.in  eq_1.in  eq_2.in  eq_3.in  REST2.in
└── REST2-run1/
    ├── resolved.config       authoritative      run.config   the seed
    ├── REST2.py              the compact ladder entry point
    ├── run.sh                build-md.log
    ├── eq/                   this run's equilibration: eq_<k>.{py,xml,out,log}
    │                         solute_eq_<k>.nc  mdout_eq_<k>.csv
    ├── remd0/  remd1/ …      ONE DIRECTORY PER THERMODYNAMIC STATE
    │                         remd_state<n>_prod<x>.nc    whole-system trajectory, segment x
    │                         solute_state<n>_prod<x>.nc  solute trajectory, segment x
    ├── remd_records/         the ladder's own records, ONE SET PER SEGMENT:
    │                         REST2_prod<x>.{out,log}  restart_prod<x>.json  rem_prod<x>.log
    │                         exchange_prod<x>.csv  ledger_prod<x>.nc
    ├── cv_state<i>.csv       the CV series, one per STATE, with its cv_state<i>.json sidecar
    ├── remd_groupfile.1      one line per state, naming that state's saved System
    ├── rank/                 per-PROCESS reports
    └── solute.yaml  _protocol.py    written by rank 0, verified by every rank
```

**The index is the STATE's, never the walker's.** After an accepted exchange the configuration in
`remd2/` belongs to a different walker, and that is the point: MD-tools writes the state-sorted
file directly, which is why `cpptraj` needs `remdtrajtemp` and GROMACS ships `demux.pl` and this
does not. Files are grouped by state so that `remd0/*.nc` concatenates one state across every
segment in order; the per-segment records are grouped the other way, in `remd_records/`.

## Restart and continuation

`--resume` continues from the checkpoint; the run state records `interrupted` or `failed` with a
reason when it does not finish, and only a run that did finish writes a completion manifest.

**A resume reproduces the trajectory, not merely a valid one.** Positions, velocities and box are
the complete *physical* state of a Langevin walker and are stored unconditionally, which is what
keeps a checkpoint readable on any device. They are not sufficient to continue a trajectory: the
integrator's pseudo-random stream has a position within it that coordinates do not carry, so a
walker resumed from them alone draws different noise from that point on — correct, statistically
exact, and a *different* trajectory. So each rung's OpenMM context checkpoint is stored alongside
the coordinates, with the platform and precision that produced it. A continuation that finds them
and is running on that platform restores them and reproduces an uninterrupted run exactly; one on
a different device ignores them and falls back to coordinates, as before. Which path was taken is
printed as `# continuation :` and recorded, never inferred. The cost is checkpoint size: one
context checkpoint per rung beside the coordinates, on a file that is rewritten whole rather than
accumulated.

One caveat belongs with this: OpenMM's **CPU platform** sums force reductions in
thread-completion order and so is reproducible only at a fixed thread count. Two replicate CPU
ladder runs with the same pinned seed diverge by the first observation when the pool differs.
That is a property of the platform, not of the ladder, and the run record carries `cpu_threads`
so two runs can be told apart from two thread counts.

**Extension is out of place.** `--extend-from` writes a new directory holding only the new segment
and pins the parent by content. The parent is left byte-for-byte unchanged, so a chain is a
sequence of immutable segments rather than a file that grows and loses its own history.

**An extension segment is atomic: an interrupted one is REDONE, not resumed.** There is no
mid-extension restart. `--resume` with `--extend-from` is refused as two different operations, and
`--resume` alone on the segment's own directory is worse than refused — it would physically
continue the run and write a `restart.json` with no `extends` block at all, so the segment would
finish looking like an ordinary run with its parent pinning, its segment-local counts and its
chain accounting silently gone. Nothing refuses that: an extension is atomic *by omission*,
because no resume path knows `extends` exists. Re-running into the partial directory is refused
three separate ways (`--force` is refused with `--extend-from`, `--overwrite` does not apply, and
the existing `-x`/`-r`/checkpoint collide), so the answer is a **fresh** directory and the whole
segment re-run. A partial segment costs what it cost; chunk a long chain into several smaller
`--extend` segments, each extending the previous completed one, so an interruption loses one
small segment rather than a long one. The chunk's own generated script is never the right entry
point either — the helpers in an extension directory are written and digest-verified by whichever
generated script runs with `-odir`, which is the **parent's**.

## Storage, restart and validation

Four records, four jobs:

| record | written | job |
|---|---|---|
| analysis NetCDF (`-x`) | every committed iteration | the authoritative history, plus the scientific identity written **before** propagation begins |
| checkpoint NetCDF | every whole-output interval | configurations, mapping, RNG states, rule state, iteration and budget |
| `<stem>.runstate.json` | atomically, on transition | `initialized` → `running` → `completed`/`interrupted`/`failed`; never claims completion |
| `restart.json` (`-r`) | atomically, at the end | evidence of completion; **never** a precondition for resuming an in-place run — but an out-of-place extension has no resume at all, so for a segment its absence means the segment is redone |

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

Both of those are about a run continuing **in place**. The auto-continue contract and every
sentence above about resuming cover in-place runs only: an interrupted out-of-place extension
(`--extend-from`) is re-run as a whole segment into a fresh directory, as described under
*Extension is out of place*. The `interrupted` run-state note and the executor's message say
which of the two you are holding, so the file itself answers the question.

An interruption is neither success nor failure and exits with its own status (130). Verification
**opens and reads** the files: a file that exists is what a crashed run leaves behind, so existence
is never treated as completion.

## The selection file

`solute.yaml`, written beside the run by rank 0, records what the ladder integrated: the solute
atom indices, the unscaled central bonds and impropers with readable residue and atom labels, the
topology digest, and `rest2.scaled_states` — the `scaler.yaml` it came from (relative to the run)
and every state's sha256. It is a **record, not an input**: the selection is made once, by
`build-top --rest2-scaler`, and the ladder classifies nothing itself.

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `rest2.number_of_replicas` | 4 | the number of states. Too few and neighbouring rungs never exchange |
| `rest2.tau_max` | 0.5 | the top rung. τ = 0.5 scales solute–solute terms by 0.25 |
| `rest2.exchange_interval_steps` | 5000 | steps between exchange attempts |
| `rest2.number_of_exchanges` | 500 | the length of the run, in exchanges |
| `rest2.equilibration_steps` | 0 | a free relaxation of every rung under its own τ before the first exchange |
| `rest2.equilibration_per_tau` | false | the whole equilibration chain on every rung under its own τ; see above |
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
