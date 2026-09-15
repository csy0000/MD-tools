# The run directory layout

**Status: proposed. Nothing below is implemented yet.** This is the target for 0.5.3's data
structure work, written down before the refactor because it moves essentially every path in
`md_tools.remd` and `md_tools.md`, and a refactor toward the wrong target is expensive to undo.

Read the tree, correct what is wrong, and the implementation follows the corrected version.

---

## 1. The target

```text
<system>/                     THE DATASET ROOT. One dataset is a system and every run on it.
  system/                     the built system: shared by every method and every repeat
  min/                        the minimised structure: shared for the same reason
  input/                      every .in: shared, because an input is not per-repeat
  REST2-run1/                 one run
    run.config                the ONLY per-run declaration: the seed
    resolved.config           authoritative, written per run from input/ + run.config
    eq/                       this run's equilibration
    remd0/  remd1/  ...       one directory per thermodynamic STATE
    rank/                     per-PROCESS reports
    bundles/                  reproducibility, with or without MD-tools
    <run root>                the ladder's own records
  REST2-run2/
  cMD-run1/
  AIS-run1/
```

**What is shared and what is not follows from the physics, not from tidiness.**

`system/` and `min/` are shared because every run on this system starts from the same built
System and the same minimised coordinates. Minimisation draws no velocities and has no seeded
stochastic element, so two runs minimising the same System produce the same structure — and a
second copy of it could only ever drift from the first. `eq/` is per-run because equilibration
draws Maxwell velocities from that run's own seed: two repeats are *supposed* to diverge there,
and that divergence is the point of a repeat.

### `system/` — at the dataset root

```text
system/  system.xml          the serialised OpenMM System            (today: built.xml)
         system.pdb          the topology                            (today: built.pdb)
         system.solute.pdb   the solute alone                        (today: built.solute.pdb)
         build-top.config    what produced them
         build-top.log       the provenance record
         solute.yaml         the resolved scaling selection
```

This is the existing convention made explicit rather than a new one. Every generated script
already reaches outside its own directory for the System — `md-openmm md-run -i min.in
-p ../built.pdb -s ../built.xml`, and `run.sh` is documented as "run from the built system in the
parent directory". One copy at the dataset root is where those relative paths were always
pointing.

### `min/` — at the dataset root

```text
min/  min.xml            the minimised coordinates: what every run's eq_1 starts from
      min.config         the resolved declaration that produced them
      min.out  min.log   human-readable, and the provenance record
      min.checkpoints/
```

A run's first equilibration stage installs `../min/min.xml`. Nothing in a run writes here.

### `input/` — at the dataset root

Every `.in`, shared. An input says what a method was *asked* to do, and two repeats of one method
on one system are asked to do the same thing — so a second copy could only drift from the first,
exactly as with `system/` and `min/`.

```text
input/  min.in                  the minimisation, shared with min/
        eq_1.in  eq_2.in  eq_3.in
        REST2.in  cMD.in  AIS.in       one production input per method
```

`min.in` and `eq_*.in` are shared across methods as well as across repeats. **That sharing is a
claim, and it is enforced rather than assumed**: a run registers only if the inputs it resolved
against are byte-identical to these (§2). If two methods on one system would need different
equilibration, they are not comparable and belong under a different `<system>` — which is a
refusal, not a subdirectory.

AIS reads neither `min.in` nor `eq_*.in`: a switching campaign starts from a source ensemble that
already exists, so it has no minimisation or equilibration chain of its own. The files being
present and unused by one method is not a problem; the files differing between methods would be.

**Every segment shares one `<method>.in`.** The runtime loops it, so a segment is not a stage and
one `.in` no longer corresponds to one stage — a change to a currently tested contract
(`test_method_example_inputs.py` asserts `example.in` matches what `build-md` generates for the
production stage). The segment number is a runtime fact, not an input fact.

### `run.config` — the only thing that is per-run

```text
<method>-run<N>/  run.config        dynamics.seed, and nothing that is not genuinely per-run
                  resolved.config   AUTHORITATIVE: input/ + run.config, resolved and written here
```

**The seed is the whole reason two repeats differ.** `derive_seed(base, *purpose)` hashes the base
seed with each stage and replica name, so every stream in a run descends from that one number:
two runs sharing an identical input and an identical seed would be bit-identical, not repeats.

It is a file rather than a command-line flag (`md-run` has no seed override today, and adding one
would leave the seed living only in a shell history until `resolved.config` was written — a re-run
typed without it would silently repeat run 1), and rather than a number parsed out of the
directory name (`REST2-run2` → 2), which would make a filename load-bearing data. The same
objection already keeps tau out of a trajectory filename.

`resolved.config` stays per-run and authoritative, as it is today: it is what the run actually
read, merged from the shared input and this run's seed.

**This needs layering, which does not exist yet.** `resolve_md_config` takes one path. A shared
input plus a per-run override is a second document, and the strict resolver will need to accept
both while keeping unknown keys refused and `run.config` narrow — an override file that can set
anything is a second configuration authority, which is the thing `md_tools.build.md` exists to
prevent.

### `eq/` — per run

```text
eq/  eq_1.{xml,config,nc,out,log}   eq_1.checkpoints/
     eq_2.{xml,config,nc,out,log}   eq_2.checkpoints/
     eq_3.{xml,config,nc,out,log}   eq_3.checkpoints/
```

**The renaming loses the ensemble from the filename**, and that has a consequence worth stating.
Today the stages are `eq_nvt_posres`, `eq_npt_posres`, `eq_npt_free`, and implicit solvent
*renames* them (`eq_nvt_posres_2`, `eq_nvt_free`) precisely so a pressure-coupled name never
appears on a boxless run — an invariant with its own tests. Under `eq_1/eq_2/eq_3` that cannot
live in the name, so it moves into each stage's `.out` header and `.config`, which state the
ensemble explicitly. The invariant survives; the evidence for it changes location.

### `remd<n>/` — one directory per thermodynamic state

```text
remd<n>/  system_state<n>.xml                   the rung Hamiltonian, serialised
          remd_state<n>_prod<x>.nc              whole-system trajectory, segment x
          solute_state<n>_prod<x>.nc            solute trajectory, segment x
          cv_state<n>_prod<x>.dat               CV series
          cv_state<n>_prod<x>.json              its sidecar
          restart_state<n>_prod<x>.xml          final positions/velocities/box
          restart_state<n>_prod<x>.json         this state's record for this segment
          remd_state<n>_prod<x>.out             human-readable
          remd_state<n>_prod<x>.log             machine-readable provenance
```

**The index is the STATE's, never the walker's.** After an accepted exchange the configuration in
`remd2/` is a different walker's, and that is the point: MD-tools writes the state-sorted thing
directly, which is why Amber needs `remdtrajtemp` and GROMACS ships `demux.pl` and we do not.

**Two trajectory streams, not one.** They have independent cadences, the whole-system one is off
by default, and rREST2's reservoir is drawn from the solute one — so they cannot be merged.

**`system_state<n>.xml` is new.** The preflight already builds every rung (`rung_systems`); this
writes it down. It carries no segment: the Hamiltonian does not change between segments, so a
segment in that name would be a lie.

### `rank/`

```text
rank/  rest2.out.rank<r>
       rest2.log.rank<r>
```

**Not under `remd<n>/`, and this is correctness rather than tidiness.** A rank is a process and a
state is a thermodynamic state; they are not in bijection. The reference ladder being migrated ran
**6 states on 5 ranks**, so `REST2.out.rank01` is the report of a process that owned more than one
state and belongs to no single one of them.

### `bundles/`

Reproducibility, with or without MD-tools installed: the exported reference bundle
(`engine.py`, `core.py`, `rung_equilibration.py`, `verify_rungs.py`) and its manifest. It keeps
its **own copy** of what it needs from `system/` and `min/`, because standing alone is its job.

### The run root

Ladder-wide facts, which no per-state file can hold — an exchange happens *between* states.

```text
rem.log                Amber-format, cpptraj-readable as type Hamiltonian
exchange.csv           every attempt, its energies and its outcome
REST2.restart.json     run_status, exchange statistics, mapping integrity, scientific identity,
                       schedule, versions, execution  (renamed from restart.json)
REST2.runstate.json    initialized -> running -> completed/interrupted/failed
REST2.out  REST2.log   the ladder's own reports
REST2_checkpoint.nc    configurations, mapping, RNG states, rule state, iteration, budget
```

---

## 2. Registration

**A `<method>-run<N>/` is registered only after the `system/`, `min/` and `input/` it was run
against validate as the same construct.** Not by path and not by directory name: each run records
the sha256 of `system/system.xml`, `min/min.xml` and every `.in` it read, in its own `.log` and
restart record, and registration recomputes them and compares. A run whose digests do not match
the dataset's is refused rather than filed beside them, because "these runs are on the same
system, from the same structure, under the same instructions" is the claim every comparison
between them rests on.

The seed is deliberately outside that check: `run.config` is *expected* to differ per run, and it
is the only thing that may.

This is what makes sharing safe. One copy cannot drift from itself, and a run that was somehow
produced against a different System is detectable instead of silently comparable.

**The dataset root is `<system>/`**, so one dataset is the system, its minimised structure, and
every run on it — which is the natural scientific unit. "ALA-explicit-HMR with its three REST2
repeats" is one thing, not four. `dataset.yaml` sits there, and every path in it stays relative to
it, as the contract already requires.

## 3. What moves, from today

| today | target |
|---|---|
| `built.xml`, `built.pdb`, `built.solute.pdb` | `<system>/system/system.{xml,pdb}`, `system.solute.pdb` |
| `min.{xml,out,log}`, `min.checkpoints/` (per run) | `<system>/min/` — shared |
| `min.in`, `eq_*.in`, `REST2.in` (run root) | `<system>/input/` — shared |
| `resolved.config` (run root) | `<run>/resolved.config` — stays per run |
| *(nothing)* | `<run>/run.config` — the seed, the only per-run declaration |
| `eq_nvt_posres.*`, `eq_npt_posres.*`, `eq_npt_free.*` | `<run>/eq/eq_1.*`, `eq_2.*`, `eq_3.*` |
| `whole_state<n>_prod<x>.nc` (run root) | `<run>/remd<n>/remd_state<n>_prod<x>.nc` |
| `solute_state<n>_prod<x>.nc` (run root) | `<run>/remd<n>/solute_state<n>_prod<x>.nc` |
| `cv_state<n>.csv` / `.json` | `<run>/remd<n>/cv_state<n>_prod<x>.dat` / `.json` |
| `restart.json` (per-state blocks) | `<run>/remd<n>/restart_state<n>_prod<x>.json` |
| `restart.json` (ladder-wide facts) | `<run>/REST2.restart.json` |
| `REST2.out.rank<r>`, `REST2.log.rank<r>` | `<run>/rank/` |
| *(nothing)* | `<run>/remd<n>/system_state<n>.xml` |

## 4. Migrating the finished reference runs

`hpREST2/data/reference/ALA-explicit-HMR/REST2-run1/` is five `chunk0N/md_script/` directories,
each a **complete independent run** — its own `build-md.log`, `min.in`, `eq_*` stages,
`resolved.config`. They are not five segments inside one run; they are five runs chained by
restart, and every one of them wrote `_prod1`.

The migration therefore:

1. **assigns segment numbers from the chunk chain** — `chunk01` → `prod1`, … `chunk05` → `prod5` —
   taken from the extension lineage in each chunk's manifest, not from the directory name, so a
   renamed directory cannot silently renumber a trajectory;
2. **hoists one `system/` and one `min/`** to `ALA-explicit-HMR/`, after verifying every chunk
   used the same ones by digest. If they differ, that is a finding and the migration stops rather
   than picking one;
3. **moves only production artefacts** into `REST2-run1/remd<n>/`, leaving each
   `chunk0N/md_script/` intact beside the new layout. Nothing is lost and the step is reversible;
4. **dry-runs first**, printing every planned move; **verifies sha256** after copying; and
   **removes a source only after** its copy verifies — and only when explicitly told to.

A 921 MB copy of `REST2-run1` is already in `data/reference/` (gitignored, 251 files, checksums
spot-verified) to develop this against. The originals in `hpREST2` have not been touched.

## 5. Consequences to accept before implementing

* **A run directory is not self-contained**, by design. It reads `../system/` and `../min/`, and
  those relative paths are what the generated scripts already use. A run moved out of its dataset
  is incomplete, and detectably so: the digests it recorded will not resolve.
* **One `.in` stops mapping to one stage.** `test_method_example_inputs.py` asserts the shipped
  `example.in` matches what `build-md` generates for the production stage; with segments looping
  one input, "the production stage" is no longer a single generated file.
* **`--overwrite`'s inventory, the completion manifest and `validate.py` all index by bare
  filename.** `validate.py:65` refuses any manifest entry containing a path separator, on purpose.
  Per-state entries therefore need an implicit `remd<index>/` prefix resolved from the entry's own
  `index` field — which keeps that no-traversal guard intact rather than weakening it.
* **Existing datasets stop validating in place.** Migration is the answer rather than a
  compatibility shim, so `data-register` and the contract's path rules need the new shape, and
  every registered REST2 dataset needs migrating before it validates again.
* **The extension-segment vocabulary collides.** `driver.py` already says "segment" for an
  *extension* — a new output directory chained to a parent — which is a different thing from
  `number_of_segments` inside one run. Both cannot keep the word.

## 6. Implementation order

1. A path authority: one module that returns every path above from
   `(dataset_root, run, state, segment)`. Partly done — `md_tools.remd.amber_trajectory` has the
   per-state names.
2. `number_of_segments` reaching the runtime. The field exists and validates; nothing reads it.
3. Config layering: a shared `input/*.in` plus a narrow per-run `run.config`, resolving to the
   per-run `resolved.config`. `resolve_md_config` takes one path today, and `run.config` must be
   restricted to the seed rather than able to set anything — an override that can set everything
   is a second configuration authority.
4. `system/`, `min/` and `input/` hoisted to the dataset root, with the digests a run records.
5. The per-run reorganisation, one directory at a time, with the suite green between each.
6. `validate.py` and the manifest, including the implicit `remd<index>/` prefix.
7. The registration gate: a run registers only when `system/`, `min/` and every `.in` it read
   verify by digest. The seed in `run.config` is deliberately exempt.
8. The migration tool, dry-run first.
9. `data-register` and the data contract.
