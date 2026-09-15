# The run directory layout

**Status: IMPLEMENTED. `build-md` writes this layout.** `md-openmm build-md -odir REST2-run1`
produces it directly, and the nine migrated reference runs under
`data/reference/{ALA-explicit-HMR,ALA-implicit-HMR,RGDfV-implicit-HMR}/` already have this shape
-- which is how the errors in the first three drafts of this document were found: the missing
exchange ledger, `build/` being a sibling rather than absent, and the equilibration streams being
solvent-dependent.

WHAT THE IMPLEMENTATION ADDED that this document did not originally anticipate, each because a
test found it:

* **`min/run.config` and `eq/run.config`.** A generated script reads the `resolved.config`
  strictly beside itself, so every directory holding one needs the declaration AND the per-run
  seed it was resolved with. `min/` is shared, so it carries its OWN seed, which may differ from
  every run on the system: minimisation draws no velocities and has no seeded stochastic element.
  Without `eq/run.config`, every equilibration stage refused with
  `eq/resolved.config describes a different run: dynamics.seed: was <run>, now 1`.
* **`min/resolved.config` AND `eq/resolved.config` are METHOD-NEUTRAL**, through one shared
  projection so the two cannot drift apart. Both directories hold scripts driven by the shared
  `input/*.in`, which carry no `protocol` -- that absence is what lets one `min.in` and one
  `eq_<k>.in` serve every method -- so a per-run document here recorded `protocol: REST2` while
  the input it is compared against resolves to the default, and every preparation stage of a
  ladder refused with `protocol: was 'REST2', now 'cMD'`. The stored document matches the input
  rather than the run. THE COST: nothing inside `min/` or `eq/` says which method the run was.
  That is recorded at the run root and in `build-md.log`.
* **A recorded path is resolved against the run directory, never reduced to its basename.** Five
  places assumed a run's artefacts sit flat in one directory -- `_stage_inputs`, `_locate` and
  the cMD `-c` parent in `reference/export.py`, the ladder's starting state in
  `reference/rest2_export.py`, and `_stage_inputs`' record glob, which searched the run root only
  and so bundled ONE stage where two had run, silently omitting the equilibration input a
  reproduction needs. Each now resolves the recorded path first and falls back to the bare name,
  which is where it sits for a run generated before the split.
* **`min.in` carries no collective-variable cadence.** Minimisation produces no series -- its
  iterations have no timestep -- so the cadence cannot change what it does, and leaving it in
  made two runs differing only in CV reporting collide on a shared input.
* **Every declaration travels with the definition it names.** `resolved.config` records the CV and
  umbrella copies by BARE NAME, so that a generated directory stays movable and the name resolves
  beside whichever declaration a reader started from. The content-addressed copy is therefore
  written into the run root, the SHARED `input/`, `min/` and `eq/` alike.

  It was written into the run root ALONE, and the layout then put declarations in four places. So
  `input/cMD.in` named `cv.<digest>.yaml` while `input/` held no such file — `md-run -i
  ../input/cMD.in` could not resolve its own definition from any directory, and an `eq/` stage
  resolving beside its own declaration found nothing either. `md_tools.run.continuation` has to
  LOAD the definition to know what an invocation intends to continue, so it got `None` and
  returned early: the read-only boundary silently did nothing and let a REFUSED `md-run` write
  `resolved.config`, `<stage>.out` and `<stage>.log` into a tree it was declining to touch.

  Sharing one copy is safe by construction rather than by convention — the name carries the
  digest, so a definition that changed gets a different name and cannot quietly replace the one a
  previous run read. An existing copy of the same name whose bytes differ is refused, because then
  one of the two digests does not describe its own file — and `--overwrite` replaces it, as it does
  every other file in the shared tree, so one corrupt copy cannot block regenerating the run.

STILL OPEN, and deliberately not decided here: two runs on one system that differ only in CV
reporting have different `eq_*.in` bytes and the second is refused. Reporting is observation and
changes no Hamiltonian -- it adds no Force, and the System, force inventory, force groups and a
single-point energy are asserted identical with it on and off -- so those two runs are the same
experiment in every respect except what they record.

`--overwrite` is NOT the remedy, and this was established by measurement rather than argument.
It rewrites `input/eq_*.in` to the second run's form, and every later generation with the first
run's setting then refuses against that: within one root the collision moves rather than
resolving, whichever order the runs go in. So the choice is real and narrow:

* **the cadence becomes a per-run value beside the seed**, which widens `RUN_CONFIG_ALLOWED` --
  a deliberately one-key schema, on the reasoning that an override able to set everything is a
  second configuration authority; or
* **such runs take separate `<system>` roots**, which is what the tests do today, at the cost
  that a CV-on and a CV-off run on the same molecule are no longer one comparable dataset.

The migration scripts under `data/reference/` are deliberately THROWAWAY and untracked: the point
is for the code to write this layout natively, so there is nothing to migrate in future. They are
not part of the package and will not be maintained.

Read the tree, correct what is wrong, and the implementation follows the corrected version.

---

## 1. The target

```text
<system>/                     THE DATASET ROOT. One dataset is a system and every run on it.
  build/                     the built system: shared by every method and every repeat
  min/                        the minimised structure: shared for the same reason
  input/                      every .in: shared, because an input is not per-repeat
  REST2-run1/                 one run
    run.config                the ONLY per-run declaration: the seed
    resolved.config           authoritative, written per run from input/ + run.config
    eq/                       this run's equilibration
    remd0/  remd1/  ...       one directory per thermodynamic STATE
    remd_records/             the ladder's own records, one set per segment
    rank/                     per-PROCESS reports
    bundles/                  reproducibility, with or without MD-tools
  REST2-run2/
  cMD-run1/
  AIS-run1/
```

**What is shared and what is not follows from the physics, not from tidiness.**

`build/`, `min/` and `input/` are shared because every run on this system starts from the same
built System, the same minimised coordinates and the same instructions. Minimisation draws no
velocities and has no seeded stochastic element, so two runs minimising the same System produce
the same structure, and a second copy could only drift from the first. `eq/` is per-run because
equilibration draws Maxwell velocities from that run's own seed: two repeats are *supposed* to
diverge there, and that divergence is the point of a repeat.

### `build/` — at the dataset root

```text
build/  built.xml           the serialised OpenMM System
        built.pdb           the topology
        built.solute.pdb    the solute alone
        build-top.config    what produced them
        built.log           the provenance record
        build-top.out       the human-readable report
        solute.yaml         the resolved scaling selection
```

**The names are `build-top`'s own.** `-os` defaults to `built.xml`, `-op` to `built.pdb` and
`-log` to `built.log`, so the layout does not rename the tool's output -- an earlier draft called
them `system.{xml,pdb}`, which invented a second name for a file that already had one.

This is the existing convention made explicit rather than a new one, and the reference run proves
it: its group file names `-p ../../../build/built.pdb -s ../../../build/built.xml`, i.e. a
directory OUTSIDE the run, already shared between that run and its siblings. The layout keeps that
directory and moves it inside the dataset -- one level up from a run instead of three, so the same
relative reference becomes `../build/built.xml`.

### `min/` — at the dataset root

```text
min/  min.xml            the minimised coordinates: what every run's eq_1 starts from
      min.out  min.log   human-readable, and the provenance record
      min.checkpoints/
```

A run's first equilibration stage installs `../min/min.xml`. Nothing in a run writes here.

`min.config` is listed in neither this block nor the source data: a minimisation's resolved
declaration lives in the shared `resolved.config`, and inventing a per-stage copy would be a
second authority.

### `input/` — at the dataset root

Every `.in`, shared. An input says what a method was *asked* to do, and two repeats of one method
on one system are asked the same thing — so a second copy could only drift from the first.

```text
input/  min.in
        eq_1.in  eq_2.in  eq_3.in
        REST2.in  cMD.in  AIS.in       one production input per method
```

`min.in` and `eq_*.in` are shared across methods as well as repeats. **That sharing is a claim,
and it is enforced rather than assumed**: a run registers only if the inputs it resolved against
are byte-identical to these (§3). If two methods on one system would need different
equilibration, they are not comparable and belong under a different `<system>` — a refusal, not a
subdirectory.

AIS reads neither `min.in` nor `eq_*.in`: a switching campaign starts from a source ensemble that
already exists, so it has no minimisation or equilibration chain of its own.

**Every segment shares one `<method>.in`.** The runtime loops it, so a segment is not a stage and
one `.in` no longer corresponds to one stage — a change to a currently tested contract
(`test_method_example_inputs.py`). The segment number is a runtime fact, not an input fact.

### `run.config` — the only thing that is per-run

```text
<method>-run<N>/  run.config        dynamics.seed, and nothing that is not genuinely per-run
                  resolved.config   AUTHORITATIVE: input/ + run.config, resolved and written here
```

**The seed is the whole reason two repeats differ.** `derive_seed(base, *purpose)` hashes the base
seed with each stage and replica name, so every stream in a run descends from that one number:
two runs sharing an identical input and an identical seed would be bit-identical, not repeats.
The migrated reference run carries `dynamics.seed: 700501` here.

It is a file rather than a command-line flag (`md-run` has no seed override today, and adding one
would leave the seed living only in a shell history until `resolved.config` was written — a re-run
typed without it would silently repeat run 1), and rather than a number parsed out of the
directory name, which would make a filename load-bearing data.

**This needs config layering, which does not exist yet.** `resolve_md_config` takes one path, and
`run.config` must be restricted to the seed rather than able to set anything: an override that can
set everything is a second configuration authority.

### `eq/` — per run

```text
eq/  eq_1.{xml,out,log,chk}  eq_1.py  eq_1.checkpoints/
     whole_eq_1.nc  solute_eq_1.nc
     mdout_eq_1.csv  energy_components_eq_1.csv
     eq_2.*  eq_3.*
```

Every name here is the stage's FILING KEY, not the stage name: `eq_nvt_posres` is filed as `eq_1`,
because the ensemble moved out of the filename so that a renamed stage cannot make a filename
claim something false. `stage_plan` stamps the key onto every plan entry, so the layout that writes
`run.sh` and the runtime that writes the files read it from one place. They did not, once: the
runtime named outputs from the stage, `run.sh` chained `-c eq/eq_1.xml`, and no cMD or REST2 chain
could complete through `run.sh` — the documented way to run one.

The stream and table names keep the affix style the runtime uses everywhere else
(`solute_<key>.nc`, `mdout_<key>.csv`), rather than a `eq_1.solute.nc` form that would apply to
these files alone:

```text
```

**Two trajectory streams per stage**, as the reference data has: a single `eq_N.nc` cannot hold
both the whole-system and the solute stream.

**The renaming loses the ensemble from the filename.** Today the stages are `eq_nvt_posres`,
`eq_npt_posres`, `eq_npt_free`, and implicit solvent *renames* them (`eq_nvt_posres_2`,
`eq_nvt_free`) precisely so a pressure-coupled name never appears on a boxless run — an invariant
with its own tests. Under `eq_1/eq_2/eq_3` that cannot live in the name, so it moves into each
stage's `.out` header and the resolved configuration, which state the ensemble explicitly. The
invariant survives; the evidence for it changes location. Stage order comes from `build-md.log`:
`eq_nvt_posres` → 1, `eq_npt_posres` → 2, `eq_npt_free` → 3.

### `remd<n>/` — one directory per thermodynamic state

```text
remd<n>/  build_state<n>.xml                   the rung Hamiltonian, serialised
          remd_state<n>_prod<x>.nc              whole-system trajectory, segment x
          solute_state<n>_prod<x>.nc            solute trajectory, segment x
          cv_state<n>_prod<x>.dat               CV series
          cv_state<n>_prod<x>.json              its sidecar
          restart_state<n>_prod<x>.xml          final positions/velocities/box
          restart_state<n>_prod<x>.json         this state's record for this segment
```

**The index is the STATE's, never the walker's.** After an accepted exchange the configuration in
`remd2/` is a different walker's, and that is the point: MD-tools writes the state-sorted thing
directly, which is why Amber needs `remdtrajtemp` and GROMACS ships `demux.pl` and we do not.

**Grouped by state rather than by segment**, so `remd0/*.nc` concatenates one state across every
segment in order — which is the common analysis operation. The segment is a suffix because these
files are indexed by state *and* segment; the records in `remd_records/` are indexed by segment
alone, which is why they are grouped the other way.

**`build_state<n>.xml` is new**, and cannot be backfilled. The preflight builds every rung
(`rung_systems`) and never serialises one; no `XmlSerializer.serialize` call for a rung exists in
`md_tools/remd/`. Writing one for the migrated run would mean re-scaling `built.xml` to each tau
*now* and asserting the result matches a run from September. Migrated runs therefore legitimately
lack it, and the migration records its absence rather than synthesising it.

### `remd_records/` — the ladder's own records, per segment

```text
remd_records/  ledger_prod<x>.nc            the exchange ledger  (today: REST2.nc)
               ledger_prod<x>.solute.nc
               checkpoint_prod<x>.nc        configurations, mapping, RNG states, rule state
               rem_prod<x>.log              Amber-format, cpptraj-readable as type Hamiltonian
               exchange_prod<x>.csv         the flat per-(exchange, state) view
               restart_prod<x>.json         this segment's completion manifest
               runstate_prod<x>.json        initialized -> running -> completed/interrupted/failed
               REST2_prod<x>.out            the ladder's human-readable report
               REST2_prod<x>.log            its provenance record
               REST2_prod<x>.group          the group file (per segment: the -c path differs)
```

One directory for every segment, with the segment kept in each filename — these records are
indexed by segment only, so a `prod<x>/` directory per segment would scatter one subject across
five places for no gain.

**`ledger_prod<x>.nc` is NOT a trajectory, and its old name said it was.** It carries no
coordinates and no `Conventions` attribute, so `cpptraj` will not read it. Its variables are the
exchange history: `u`, `u_evaluated`, `proposed` and `accepted`, each `(exchange, state, state)`
— 10000×6×6 in the reference run — plus `state_to_walker` at `(exchange, state)`, the tau ladder,
and the frame bookkeeping. `restart.json` calls it authoritative because the run's whole history
is reconstructable from it.

It stays NetCDF rather than becoming `.csv` or `.dat` because four of its variables are 3-D: a
flat table cannot hold a 10000×6×6 array without exploding to 360,000 rows or inventing a column
encoding. `exchange_prod<x>.csv` beside it already IS the flat view —
`exchange,step,time_ps,state,tau,walker,reduced_potential,potential_energy_kj_per_mol,`
`proposed_with_next,accepted_with_next`, one row per (exchange, state), carrying the
neighbour-pair subset. NetCDF for the full matrix, CSV for the flat view.

### `rank/`

```text
rank/  REST2_prod<x>.out.rank<r>
       REST2_prod<x>.log.rank<r>
```

**Not under `remd<n>/`, and this is correctness rather than tidiness.** A rank is a process and a
state is a thermodynamic state; they are not in bijection. The reference ladder ran **6 states on
5 ranks**, so `REST2.out.rank01` is the report of a process that owned more than one state and
belongs to no single one of them.

### `bundles/`

Reproducibility, with or without MD-tools installed: the exported reference bundle
(`engine.py`, `core.py`, `rung_equilibration.py`, `verify_rungs.py`) and its manifest. It keeps
its **own copy** of what it needs from `build/` and `min/`, because standing alone is its job.

---

## 2. What moves, from today

| today | target |
|---|---|
| `build/` beside the run (three levels up from `md_script/`) | `<system>/build/` — same names, one level up from a run |
| `min.{xml,out,log}`, `min.checkpoints/` (per run) | `<system>/min/` — shared |
| `min.in`, `eq_*.in`, `REST2.in` (run root) | `<system>/input/` — shared |
| `resolved.config` (run root) | `<run>/resolved.config` — stays per run |
| `REST2.config` (the per-run declaration) | `<run>/run.config` |
| `eq_nvt_posres.*`, `eq_npt_posres.*`, `eq_npt_free.*` | `<run>/eq/eq_1.*`, `eq_2.*`, `eq_3.*` |
| `whole_state<n>_prod<x>.nc` | `<run>/remd<n>/remd_state<n>_prod<x>.nc` |
| `solute_state<n>_prod<x>.nc` | `<run>/remd<n>/solute_state<n>_prod<x>.nc` |
| `cv_state<n>.csv` / `.json` | `<run>/remd<n>/cv_state<n>_prod<x>.dat` / `.json` |
| `REST2.nc`, `REST2.solute.nc` | `<run>/remd_records/ledger_prod<x>.nc`, `.solute.nc` |
| `REST2_checkpoint.nc` | `<run>/remd_records/checkpoint_prod<x>.nc` |
| `rem.log`, `exchange.csv` | `<run>/remd_records/rem_prod<x>.log`, `exchange_prod<x>.csv` |
| `restart.json` (ladder-wide) | `<run>/remd_records/restart_prod<x>.json` |
| `restart.json` (per-state blocks) | `<run>/remd<n>/restart_state<n>_prod<x>.json` |
| `REST2.runstate.json` | `<run>/remd_records/runstate_prod<x>.json` |
| `REST2.out`, `REST2.log`, `REST2.group` | `<run>/remd_records/REST2_prod<x>.{out,log,group}` |
| `REST2.out.rank<r>`, `REST2.log.rank<r>` | `<run>/rank/` |
| *(nothing)* | `<run>/remd<n>/build_state<n>.xml` — new, not backfillable |
| *(nothing)* | `<run>/run.config` — the seed |

## 3. Registration

**A `<method>-run<N>/` is registered only after the `build/`, `min/` and `input/` it was run
against validate as the same construct.** Not by path and not by directory name: each run records
the sha256 of `build/built.xml`, `min/min.xml` and every `.in` it read, in its own `.log` and
restart record, and registration recomputes them and compares. A run whose digests do not match
is refused rather than filed beside them, because "these runs are on the same system, from the
same structure, under the same instructions" is the claim every comparison between them rests on.

The seed is deliberately outside that check: `run.config` is *expected* to differ per run, and it
is the only thing that may.

**The dataset root is `<system>/`**, so one dataset is the system, its minimised structure, and
every run on it — the natural scientific unit. `dataset.yaml` sits there, and every path in it
stays relative to it, as the contract already requires.

## 4. Migrating the finished reference runs

`REST2-run1/` was five `chunk0N/md_script/` directories. **All five are complete segments**: each
reports `run_status: completed` with 10,000 committed exchanges, at cumulative steps 25M, 50M,
75M, 100M and 125M. What distinguishes `chunk01` is that it carries the ONE-TIME SETUP — `min.*`,
`eq_*`, `build-md.log`, `run.sh`, the checkpoint trees — because 02–05 are continuations that
correctly did not re-minimise or re-equilibrate. Nothing is missing from any of them.

Every chunk wrote its per-state trajectories as `_prod1`, which is the defect being corrected.

The migration:

1. **assigns segment numbers from the chain, verified not assumed** — `chunk01` has no `extends`;
   02→01, 03→02, 04→03, 05→04 by parent path and parent checkpoint sha256. So `chunk0N` → `prodN`.
2. **checks by digest that the shared files really are shared** before hoisting one copy.
   `REST2.in`, `resolved.config`, `solute.yaml`, `_protocol.py`, `REST2.py` are byte-identical in
   all five. `REST2.group` is NOT — the `-c` path differs — so it is per segment.
3. **copies the sibling `build/` into the dataset root.** It sits beside the run rather than
   inside it, which is why
   the first draft of this document wrongly concluded it did not exist.
4. **moves only production artefacts**, leaving each `chunk0N/md_script/` intact beside the new
   layout. Nothing is lost and the step is reversible.
5. **dry-runs first**, printing every planned move; **verifies sha256** after copying; and
   **removes a source only after** its copy verifies — and only when explicitly told to.
6. **records what does not exist rather than synthesising it.** Absent from the source and
   therefore from the migrated tree: `build_state<n>.xml`, per-state restart records, CV series,
   per-state `.out`/`.log`, `bundles/`. `restart.json`'s `states` entries carry only `index`,
   `trajectory`, `tau` and `effective_temperature_k`, and the per-state digests inside `extends`
   describe the PARENT segment's outputs — deriving this segment's per-state record from them
   would be fabrication.
7. **refuses `run.sh`.** Every path in it is relative to `chunk01/md_script`, so a copy at the run
   root would be an executable resolving to the wrong place.

First pass: 219 files, 964,582,362 bytes, every destination sha256-verified, idempotent on a
second `--apply`, sources intact at 251 files.

## 5. Consequences to accept before implementing

* **A run directory is not self-contained**, by design. It reads `../build/`, `../min/` and
  `../input/`. A run moved out of its dataset is incomplete, and detectably so: the digests it
  recorded will not resolve.
* **One `.in` stops mapping to one stage**, so `test_method_example_inputs.py`'s production-stage
  assertion needs rethinking.
* **`validate.py:65` refuses any manifest entry containing a path separator**, on purpose. Entries
  therefore need an implicit `remd<index>/` or `remd_records/` prefix resolved from the entry's own
  fields, which keeps that no-traversal guard intact rather than weakening it.
* **Existing datasets stop validating in place.** Migration is the answer rather than a
  compatibility shim.
* **The extension-segment vocabulary collides.** `driver.py` already says "segment" for an
  *extension* — a new output directory chained to a parent — which is a different thing from
  `stages.number_of_segments` inside one run. Both cannot keep the word.

## 6. Implementation order

1. A path authority: one module returning every path above from
   `(dataset_root, run, state, segment)`. Partly done — `md_tools.remd.amber_trajectory` has the
   per-state names.
2. `number_of_segments` reaching the runtime. The field exists and validates; nothing reads it.
3. Config layering: shared `input/*.in` plus a narrow per-run `run.config`.
4. `build/`, `min/` and `input/` hoisted, with the digests a run records.
5. The per-run reorganisation, one directory at a time, with the suite green between each.
6. `validate.py` and the manifest, including the implicit directory prefix.
7. Serialising `build_state<n>.xml` per rung, which is new behaviour rather than a move.
8. The registration gate.
9. ~~A migration tool~~ -- not needed. The nine existing reference runs are already
   migrated, and future runs are written into this layout directly, so no tool has to
   exist for it.
10. `data-register` and the data contract.
