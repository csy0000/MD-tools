# Backlog

Known gaps, and what became of them. Each entry says what it is, what it costs, and the concrete
thing that should cause it to be picked up. **Read the first line of an entry before anything
else** — an entry that opens `RESOLVED` or `RECLASSIFIED` is a record of what was wrong and how it
was closed, kept because the reasoning is worth more than the deletion would be.

Nothing here is a release gate: none of it has been shown to corrupt, discard or misreport
scientific output, or to prevent recovery of an interrupted run. An open entry is not "handled" —
if one turns out to affect a result it stops being backlog and becomes a defect.

**Where it stands.** Entries 1, 2, 5, 7, 8, 9 and 10 are fixed, each with the test that would fail
if it came back. Entry 3 is reclassified: it described deliberate behaviour, which does not belong
in a list of debt. Entry 4 is neither fixed nor accepted but UNDIAGNOSED, and cannot be closed by
work. Entry 6 is open in code and avoided in practice. Entries 13, 14 and 15 are the **0.5.4 scope**,
deferred by decision on 2026-09-16 rather than by oversight: the omega exclusion for a `peptide` or
`peptide-like` solute (**resolved** — it was the enforcement, not the classifier), the `.in` file's
undocumented divergences from Amber's input conventions, and rebuilding AIS as a transformation
between two topologies. Entry 16 is **fixed in 0.5.3** — `.sdf`
input, and the missing `built.sdf` on the explicit ligand route that adding it uncovered, which
entry 13 should be re-checked against. Entries 17, 18 and 19 are open and were filed the same day: the
documentation reorganisation, a `solute.residue_name` that is recorded but never applied, and a
chemistry mismatch that exits 1 and leaves a directory where a shape mismatch exits 2 and leaves
nothing.

**Three of these entries described code that had already moved on** — 8 said a flag was dropped
that was being forwarded, 9 described a reader whose sidecar nothing wrote, 1 said an aggregate
went unaudited that the ladder had been auditing since `916feca`. The lesson is in the file rather
than in any one entry: a backlog is written once and read many times, and an entry is a claim
about the code that needs re-checking against it before being acted on.

---

## 1. The AIS aggregate CV-cost record was not cross-checked against the entries it sums

> **RESOLVED 2026-09-10, and the entry was half stale.** `cv/cost.py::parse_aggregate_record` has
> audited an aggregate against its own entries since `916feca` — observations and evaluations
> exactly, `wall_seconds` within an epsilon that scales with the entry count, every identity
> present and unique. The LADDER calls it, in two places. **AIS never did.** It assembled the
> aggregate by accumulating in a loop, wrote it, and published it unaudited, so a record whose
> parts were each valid and whose total was wrong was accepted on that path only.
>
> `aggregate_cv_cost` now runs the same parser over the record it just built, against
> `expected_identities`, before returning it.
>
> **It was never only a file.** `ais_main` reads `cumulative` straight back out and writes
> "N scalar evaluation(s) over M observation(s), summed over K completed path(s)" into the run
> log, so a slip in that arithmetic is read by a person as a measured fact about the campaign.
>
> **Why the existing test did not catch it**, which is the part worth keeping: it called the
> parser on the returned record, proving the record WOULD satisfy the parser rather than that the
> writer runs it. Those differ exactly when the writer is wrong, which is the only case that
> matters. The same shape as entry 8, where a source check proved a flag travelled to a runtime
> that ignored it. Pinned now by
> `test_the_writer_refuses_an_aggregate_whose_total_is_not_the_sum`, which corrupts the aggregate
> the writer builds and requires it to notice; it reports "DID NOT RAISE" against the parent
> commit.

---

## 2. A completion manifest with no cost record at all was accepted

> **RESOLVED 2026-09-10.** The first half of this entry. A committed PREFIX whose cost record is
> missing is refused by name; a completion MANIFEST with no `cost` block passed every check,
> because `remd/cv_states.py::_cost_problems` returned early — `if block is None: return []` —
> before the strict parsers it delegates to could see the absence. Both parsers refuse `None`
> when reached, answering "no collective-variable cost record". Nothing reached them.
>
> The asymmetry was the defect, more than the absence: the same omission was reported loudly on a
> resume and silently on a completed run, so which answer you got depended on which operation
> happened to read the campaign.
>
> Refused now WHEN THERE IS SOMETHING IT COULD DESCRIBE — a manifest recording CV series for N
> states and carrying no cost. A run with reporting switched off has no series and no cost, and
> that is a complete record rather than a damaged one; refusing every absent cost would have made
> every non-CV ladder unverifiable. Both directions pinned in
> `tests/test_cv_validation_gaps.py`, section C.
>
> **The second half stands, and is not a defect.** Malformed-metadata coverage is representative
> rather than exhaustive: it covers the field shapes that decide a truncation or a restoration,
> not every field of every record under every mutation. The unexercised mutations are in fields
> no continuation reads. Exhaustive mutation coverage of records nothing consumes is a cost
> without a benefit; it becomes worth doing if manifest metadata starts being consumed by
> something other than a human reader — a dashboard, an analysis package, an automated pin.

---

## 3. Diagnostics are not transactional, and that is the design

> **RECLASSIFIED 2026-09-10 — not a gap.** This entry described its own subject as deliberate:
> "This is deliberate under the current preservation rules — a rejected attempt is allowed to say
> why it refused". A backlog is a list of things that are wrong and unfixed. An entry that says
> the behaviour is intended does not belong in one, and leaving it here made the list longer than
> the actual debt — which matters, because the list is read to decide whether the software is fit
> to use.
>
> The behaviour, stated as behaviour: the read-only continuation boundary protects trajectories,
> CV/work/HS tables, committed checkpoint generations and pointers, completion manifests and
> authoritative configuration. It does NOT protect `.out` and `.log` files. A refused attempt may
> append to one, and a crash mid-write may leave a partial line. That is the point — a refusal
> that could not write down why it refused would be a worse failure than an untidy log. No
> authoritative record is involved, and nothing reads a diagnostic back to decide anything.
>
> Schema-version handling is likewise strict where a record decides scientific behaviour and
> permissive where it does not, which is a deliberate boundary rather than an inconsistent one.
>
> **It becomes a defect** the moment either assumption stops holding: if a diagnostic file is made
> authoritative for anything, or if partial diagnostic lines start breaking a downstream parser.
> Reopen it then, and it will be a real entry.

---

## 4. One unexplained AIS MPI failure, with no retained diagnostic

**What.** A single run of `tests/test_cv_mpi_cuda_ais.py` reported `1 failed, 10 passed`. The
diagnostic was lost to an output filter before it could be read, and the file has passed every
run since — thirteen consecutive clean full-file runs at the time it was recorded, plus the runs
in this pass. The cause is unknown. It is recorded as unresolved historical evidence, not as a
diagnosed and fixed defect and not as an accepted limitation.

**Impact.** Unknown, which is the problem. It has never recurred and has never been reproduced,
so there is nothing to characterise.

**Evidence since, 2026-09-10.** Seven further full-suite runs on the nine-GPU machine, including
the two release gates for `openmm-v0.5.0`, with zero failures in `tests/test_cv_mpi_cuda_ais.py`
in any of them. That is added as evidence, NOT as a resolution: a fault that has not recurred is
not a fault that has been found, and the count of clean runs can never reach a proof. It is
recorded so that a future reader can see how much clean evidence has accumulated against how
little of the original.

**This entry cannot be closed by work.** There is nothing to reproduce and nothing to diagnose;
the diagnostic was lost when it happened. It closes if it recurs — and is then a real defect with
evidence — or it is eventually retired as stale. Neither is something a change to this repository
can bring about, which is why it sits here differently from every other entry in this file.

**Not fixed, and not accepted either.** Those are the two states a backlog entry usually has, and
this is a third: undiagnosed. "Fixed" would mean the cause was found and removed, and nobody knows
the cause. "Accepted limitation" would mean the behaviour is intended, as entry 3 is — and if the
cause is real, this is a bug. What exists is MITIGATION: the lane is instrumented so a RECURRENCE
is diagnosable. That does not explain the original.

**Trigger.** Pick this up the moment it recurs. Every launcher call in that lane now carries a
subprocess timeout and retains complete stdout and stderr, so a recurrence will arrive with the
diagnostic this one lacked — verified 2026-09-10 rather than assumed: all of them route through
`_launch`, which passes `timeout=LAUNCH_TIMEOUT` and `capture_output=True` and asserts with the
last 4000 characters of both streams. A reproducible scientific, restart or MPI failure is a release
blocker; this one is not, because it is not reproducible.

---

## 5. `constraints.type` was ignored under implicit solvent, and the log said otherwise

> **RESOLVED 2026-09-10.** The implicit route passed `app.HBonds` to `createSystem`
> unconditionally, so `AllBonds` + GBn2 built a System identical to `HBonds` + GBn2 while the
> build log recorded `constraints.type  AllBonds  (set)` and the bundle record wrote the literal
> string `"HBonds"`. The setting was accepted, echoed back, and never applied.
>
> Resolved by HONOURING it rather than refusing it, which was the policy call this entry left
> open. Refusing anything but `HBonds` under implicit solvent would have been defensible, but the
> explicit route already honours the setting and OpenMM applies it identically with or without a
> solvent — so refusing would have made the two routes differ in what they accept as well as in
> what they do. Both now resolve the string through one function,
> `md_tools.openmm.system.constraint_option`.
>
> Measured on ALA + GBn2: `HBonds` gives 12 constraints, `AllBonds` gives **21** — the same +9
> heavy-heavy bonds `AllBonds` adds under TIP3P.
>
> **The two smaller disagreements are closed too.** `_check_constraints` compared against Python
> `None` while the schema enum offers the string `"None"`, so the advertised value was refused;
> and it accepted `HAngles`, which the enum does not offer and which
> `docs/scientific-defaults.md` §11.2 says is deliberately unavailable because no angle is ever
> constrained by this option. The enum was the intended policy both times, and the validator now
> matches it — `HAngles` is refused by name, with the reason.
>
> Pinned by `tests/test_constraint_scope.py`: the implicit `AllBonds` counterpart to the explicit
> test that already existed, a test that the record and the System agree, and one asserting the
> schema and the validator accept exactly the same three values.

---

## 6. AM1-BCC charges are not reproducible between builds of the same input

**What.** Two `build-top` runs of the same `.smi`, same configuration, same machine, can produce
different partial charges. Measured on a 43-atom cyclo(Gly-Asp-Arg): **41 of 43 atoms differed**,
one example atom moving from -0.63047675 to -0.71247675 e. Reproduced with the two builds run
SEQUENTIALLY, so it is not a parallel-execution artefact.

**It is molecule-dependent, not universal.** The 79-atom c(RGDfV) reproduced its charges
**exactly** -- 0 of 79 atoms differed -- across builds separated by a day and by a change to the
engine. So some molecules land in the same conformer every time and some do not; the absence of a
difference is luck rather than a guarantee, and cannot be relied on for any particular input.

**Cause, from the installed source.** The ETKDG embedding IS seeded, and `build-top` refuses an
unseeded one precisely so "this prepared system would not be reproducible" cannot happen. But the
OpenFF toolkit then discards that conformer: `AmberToolsToolkitWrapper.assign_partial_charges`
does `if use_conformers is None: mol_copy._conformers = None; mol_copy.generate_conformers(...)`,
and `md_tools.openmm.system` calls `offmol.assign_partial_charges(scheme)` without
`use_conformers`. The charges therefore come from an unseeded conformer that is neither the one
recorded in the build provenance nor the same one twice.

**Impact.** Two: no ligand or peptide-like build is bitwise reproducible, and the recorded ETKDG
conformer provenance does not describe the geometry the charges were computed on. Nothing is
wrong within a single build -- the charges are real AM1-BCC for the conformer that was used -- and
every published dataset remains internally consistent and self-describing. What cannot be done is
reproduce one from its inputs, or compare two builds expecting only an intended difference.

**How method development avoids it entirely — and the reason this is not being fixed in code.**
A project comparing sampling or analysis efficiency against reference data takes the SAME topology
and `system.xml` the reference simulations used, from `$MD_DATA/common/`, rather than building its
own. Then no two arms of a comparison can differ in their charges, because there is only one build.
The non-reproducibility is real and stays open, but it never enters a study run this way, and
`docs/scientific-defaults.md` §13 now states the rule where a reader planning a comparison will
meet it.

That also settles the question this entry left hanging. The fix is one keyword — passing
`use_conformers=` to `assign_partial_charges` — and it would change the charges of every future
ligand and peptide-like build, including any comparison against a dataset built before it. Paying
that to remove a problem the practice already avoids is the wrong trade while the practice holds.

**Since 0.5.2 it is visible in every bundle.** `input/build_system.py` rebuilds a System from its
structure and compares bytes. It rebuilds phenol identically, and for a molecule like
cyclo(Gly-Asp-Arg) it would report DIFFERS and exit 1 -- correctly -- and the bundle's
`input/README.md` says why for any ligand charged with AM1-BCC. Revise that sentence when this
entry is settled.

**Not fixed here, deliberately.** **Trigger.** Pick this up before any study that needs to reproduce a build from its inputs, before
comparing two builds for anything but their radii, or when deciding whether recorded conformer
provenance should be binding. The fix is small; its consequences are not.

---

## 7. An interrupted cMD chain cannot be resumed, and three messages disagree about it

> **RESOLVED 2026-09-10** (`7d76b05`). `md-run` ran the output-collision check before
> `stage_main` could consult the completion record, so a completed stage was never skipped.
> `stage_main` runs the identical check itself and does it LATER -- after the
> completed-and-verified short-circuit and after reading the committed checkpoint that decides
> whether a stage is merely interrupted -- so the early call is gone on that path. Re-running
> the same command now continues the chain, which is what `run.sh`'s header always claimed.
>
> `--resume` is still refused by name on a cMD chain, deliberately: a stage continues on the
> strength of a committed checkpoint, a fact about the directory, not a claim on the command
> line. The collision message no longer advertises it on that path.
>
> Pinned by `tests/test_examples_getting_started.py::
> test_example_3_an_interrupted_cmd_chain_resumes_by_rerunning_the_same_command`, which is the
> same example rewritten to assert the working behaviour.

**What (as it was).** Interrupt a cMD chain mid-production and there is no way to continue it. Every
documented route is refused:

| you do | you get |
|---|---|
| `./run.sh ...` again | `3 output(s) already exist ... Pass --overwrite to replace them, --resume to continue that run, or choose another -odir.` |
| `./run.sh ... --resume` | `--resume is not a cMD flag. An interrupted chain continues from each stage's committed checkpoint automatically; re-run the same command.` |
| `./run.sh ...` again | the first message again |

`run.sh`'s own header promises a third behaviour that does not happen: *"A stage that already
reports completion is skipped rather than silently rerun."* In the code,
`preflight._refuse_existing_outputs` returns early only for `--overwrite` or `--resume`, and
there is no completed-stage skip; the cMD workflow layer then rejects `--resume`.

**Impact.** The only way forward is `--overwrite`, which discards every finished stage, or a
fresh `-odir`. For a long cMD chain on a scheduler with a wall-clock limit that is the difference
between resuming and starting again. Committed checkpoints ARE written, so the data to resume
from exists; nothing can reach it.

**Not affected:** REST2/rREST2 ladders, which take `--resume` properly -- though not through
`run.sh`, which forwards its arguments to the cMD-style `min`/`eq` stages first and is rejected
there. A ladder is resumed by invoking its `mpirun` line directly.

**Trigger.** Pick this up before running any cMD chain long enough to be interrupted. Two
separate fixes: make the cMD layer accept `--resume` (or make re-running skip completed stages,
which is what `run.sh` already claims), and stop the output-collision message suggesting a flag
that the next layer always rejects.

That test did its job: it failed on the day the defect was fixed, which is how the fix was
noticed rather than the entry being left standing.

## 8. `md-run --overwrite` could not restart a REST2/rREST2 ladder in place

> **RESOLVED 2026-09-10.** The symptom was real and the diagnosis below was wrong, in a way worth
> keeping: the flag was never dropped. `_forward` in `run/main.py` has appended BOTH `--resume`
> and `--overwrite`, for every protocol, since `2074b0d` -- 141 commits before this entry was
> written. What this entry read as the forwarding list was a REDUNDANT second `argparse.append`
> of `--resume` sitting beside it, and `--overwrite`'s absence from that redundant line was
> mistaken for its absence from the forwarding. The duplicate is gone, and the comment in its
> place says where forwarding actually happens.
>
> The defect was one layer down. `--overwrite` reached `replica_main` and became the executor's
> `--force`, which only BYPASSES the existing-output check -- and that check covers
> `_outputs(files)`: `-o`, `-x`, `-r`, `--chk`. The per-state trajectories are not among them.
> So nothing ever moved `whole_stateN_prod1.nc` or `solute_stateN_prod1.nc` aside, and
> `StateTrajectorySet.create`, which refuses to write into files it did not just create, turned
> the run away advising the flag that had just been passed. Bypassing a check was never going to
> be enough: something had to REPLACE.
>
> `_ladder_inventory` has named every one of those files -- both per-state streams, both per-state
> CV streams, the per-rank reports, the helpers, the checkpoint tree -- since the output-inventory
> pass. The ladder now runs `replace_owned_inventory` over it, once, on rank 0, BEFORE the helpers
> are published (they are owned outputs too, and replacing after writing them would delete what
> the launch had just prepared). That is the same one-transaction replacement the cMD stage path
> has always run over its own inventory. The ladder also refuses on a marker left by a
> replacement that did not finish, which the stage path already did and this path did not.
>
> Fault 3 -- the message being unreachable under MPI -- was `Coordination.fail` printing to
> `sys.stderr`, which the executor has redirected into `<protocol>.out`, and then calling
> `MPI_ABORT`, which ends the job without returning through the code that would have said "see
> `<protocol>.out`". It writes to `sys.__stderr__` as well now: the interpreter's own stream,
> which survives the redirect. Without this the NEXT failure on this path would have been just as
> opaque.
>
> **Why every existing test passed.** Each hop was checked and the destination was not.
> `test_generated_overwrite_reaches_the_executor` asserted, by source inspection, that the flag
> travels -- and it did travel, into a runtime that did nothing with it. Pinned now by
> `tests/test_runtime_contract_matrix.py::test_a_ladder_reruns_in_place_under_overwrite`, which
> runs a two-state ladder, runs it again over itself, and looks at the exit code; plus the
> inventory, ordering, marker and terminal-diagnostic tests beside it.

**What (as it was).** `md-run --overwrite` was accepted, applied to every `stage_main` stage --
minimisation and the three equilibrations all reported "--overwrite replaced N existing
output(s)" -- and then the ladder refused, advising `--overwrite`.

**Reproduction.** Any REST2 directory that has already run once:

```bash
mpirun -n 4 md-openmm md-run -ng 4 -i REST2.in -p built.pdb -s built.xml \
       -c eq_nvt_free.xml --overwrite
```

A preserved failing directory was kept until 2026-09-11 and then removed: the entry is resolved
and the reproduction is pinned by the test named above, which needs no preserved directory.

**Three faults, and the second and third are why it cost an hour rather than a minute.**

1. `--overwrite` replaced nothing on this path.
2. The advice could not be followed. The message named the right flag for the right command and
   doing what it said changed nothing.
3. The message was unreachable under MPI. It went to `REST2.out`, and rank 0's `MPI_ABORT`
   discarded it -- `mpirun` printed only "MPI_ABORT was invoked". It was recovered only by
   re-running single-rank with `-ng 1`.

The error message was separately corrected on 2026-09-09 (`--force` -> `--overwrite`, entry in
`20260909-resume-identity-and-force.md`) without testing that following the corrected advice
works -- so the wording of unfollowable advice was improved.

---

## 9. `resolved_run.yaml` was read by three code paths and written by none

> **RESOLVED 2026-09-10.** `remd/source_ensemble.py::companion_record` searched for a
> `resolved_run.yaml` beside the source trajectory and up to four directories above it, and fed
> three consumers: the frame-time map, the source tau, and the source temperature. Nothing in this
> repository has ever written that file. Only two test fixtures fabricated one, which is why the
> branch stayed green while being unreachable from any real directory.
>
> It was not merely dead. It made three refusals advise pointing at "a trajectory written by this
> repository's runtime (which records the map)" when what the runtime records is the AMBER NetCDF
> attributes `trajectory_identity` reads, not a sidecar. And it was the ONLY route
> `source_temperature` had: `trajectory_identity` already returned `temperature_k`, `source_tau`
> already preferred the file's own attribute over the sidecar, and `source_temperature` consulted
> neither -- so an rREST2 reservoir drawn from a trajectory whose own `temperature_k` stated the
> answer was refused with "no companion runtime record" while the answer sat in the file being
> read.
>
> `companion_record` and `_replica_entry` are deleted, all three call sites with them;
> `source_temperature` reads the trajectory's own attribute exactly as `source_tau` does; the
> refusals name the real evidence. `docs/scientific-defaults.md` no longer cites
> `resolved_run.yaml` as a provenance location either.
>
> Pinned by `tests/test_own_replica_exchange.py::
> test_nothing_reads_a_resolved_run_yaml_sidecar_any_more`, which plants the file and asserts it
> changes nothing, and by the two temperature tests beside it.

---

## 10. Five more documented provenance locations that nothing wrote

> **RESOLVED 2026-09-10.** The barostat row of `docs/scientific-defaults.md`'s provenance table
> cited six locations and `grep` found one of them in `src/`. Corrected by auditing where each
> fact is ACTUALLY recorded rather than by deleting the row, which would have removed the answer
> along with the wrong address.
>
> What was wrong, and what is true:
>
> | cited | reality |
> |---|---|
> | `MD/provenance.yaml → protocol.pressure_coupling` | `pressure_coupling` appears nowhere in `src/`. The requested pressure is `resolved.config → dynamics.pressure_bar` |
> | `resolved_stage.yaml → barostat_frequency_steps` | that FILE is written by nothing -- it appears only in a `STAGE_RUNTIME_OUTPUTS` tuple -- and `barostat_frequency_steps` only in comments, naming a configuration key. The requested value is `dynamics.barostat_interval_steps` |
> | `barostat_interval_ps`, `barostats_in_system`, `barostats_active` | the right facts under the wrong names. They are nested, and they are in the STAGE LOG: `<stage>.log → barostats.{in_system, active, frequency_steps, interval_ps}`, written by `md/stage.py` |
>
> The distinction the corrected row now draws is the one that matters and that the old row lost:
> `resolved.config` records what was REQUESTED, and the stage log records what was APPLIED --
> `in_system` and `active` are counted off the built System and the live Simulation, not restated
> from the request. Under implicit solvent those differ by design, and entry 5 was a case of a
> request and an outcome disagreeing with nothing to compare them against.
>
> **The whole table was then audited**, mechanically: every backtick-quoted filename and field in
> §14 checked against `src/`. Three more rows cited a `provenance.yaml` that NOTHING WRITES — it
> survives only in a comment, and `openmm/builders.py` records why: the provenance writer "belonged
> to the retired route, and registration owns that ground now". The rows had outlived the artefact
> by a refactor. One also named a `forcefield_summary` field that has never existed.
>
> Corrected to what is written today: `inputs/forcefield.json` (which IS the parameterisation
> summary, and carries `package_versions`), each run's log record `environment.packages` (what
> registration reads to bind a dataset to its engine), and each stage log's *Resolved settings*.
>
> 36 distinct tokens across 12 rows now resolve. The audit is a dozen lines of grep and is worth
> re-running whenever a writer is retired — that is how all four of these got stale.

---

## 11. `export-reference -idata .` could not find the built System -- FIXED in 0.5.2

Both exporters now resolve `-idata` and `-odir` before searching, and
`tests/test_reference_export.py` exports from inside a run directory with `-idata .`. Kept here
as the record of what was found:

Found 2026-09-11, exporting the test-systems campaign with a 0.5.2 release candidate. A run directory made
by build-md sits beside `built.xml`/`built.pdb`, and its records name them relative to the command
line (`-p ../built.pdb` is recorded as `built.pdb`), so the exporter finds them by digest "beside
the run directory or above it". Given `-idata .`, "above it" is `Path('.').parents`, which is empty,
and the export is refused with a message saying the topology is not beside or above -- true of the
path as written, false of the directory. An absolute `-idata` works.

The fix is to resolve `-idata` (and `-odir`) before any search, in `export_reference` and
`export_rest2_reference` alike, with a test that exports from inside the run directory with `.`.
Deferred rather than fixed in 0.5.2 because the release had been tagged and registered datasets
already name its commit. `docs/campaigns/test-systems-2026-09/run_tests.py` passes absolute paths.

---

## 12. `--all-in-one` writes every stage's artefacts flat in the run root

Found 2026-09-15, probing the all-in-one chain after the equilibration filing keys were fixed. The
chain itself is correct -- five stages, `status: completed` each, keys right (`eq_1.xml`, `min.xml`,
`cMD.*`, `mdout.csv`) -- but `run_generated_workflow` uses ONE `base` for every stage instead of
each stage's own `-odir`, so the artefacts land beside `md.py`:

```text
cMD-run1/  eq_1.xml eq_1.log eq_1.checkpoints/  min.xml min.log  cMD.xml  mdout_eq_1.csv
           eq/                                   <- holds only resolved.config + run.config
```

The split layout puts equilibration artefacts in `<run>/eq/` and the minimisation in the SHARED
`<system>/min/`, which is what `run.sh` does. So a split run and an all-in-one run of the same
configuration produce the same filenames in different places, and only the split one is where the
layout says to look. `build-md --all-in-one` does create `eq/` (every script-bearing directory
carries its own declaration), which makes the empty directory beside the flat files misleading in
its own right.

CLOSED 2026-09-16 by the retirement of `--all-in-one` in 0.5.4, which deleted
`run_generated_workflow` and with it the single `base` this entry is about. There is now one shape
of generated run, and its artefacts are filed by the layout that `run.sh` already followed.

It was NOT fixed while the feature existed, deliberately: it was pre-existing, unrelated to the
naming work that exposed it, and changing where a runtime writes would have been a behavioural
change for anyone already using `--all-in-one`. Retiring the flag settled it without that cost.
`--check` was suspected of leaving the stray `eq/` behind and was cleared by measurement: a
`--check` into a fresh `-odir` creates nothing, as the contract requires.

---

## 13. The omega exclusion is not enforced, and `peptide-like` fits neither classifier route

> **RESOLVED for 0.5.4, 2026-09-16. The user's reading was right, and the `peptide-like` hypothesis
> was not the cause.** `35d216c` had already retired `route` and chooses the evidence per
> candidate, and an explicit `kind: peptide-like` build of cyclo(GDR) classifies all three omegas
> as ordinary once the classifier is given its `built.sdf`. What did not fire was the ENFORCEMENT,
> on two counts:
>
> * **`build-md` scaled the rungs without the SDF.** `write_rung_systems` was called with no
>   `ligand_sdf`, so every amide of a `peptide-like` or `ligand` solute was unclassified, left out
>   of `omega_unscaled_bonds`, and scaled in the `build_state<n>.xml` files a grouped ladder
>   integrates. Measured on HEAD: `excluded_central_bonds: []`, `n_excluded_torsions: 0`, under a
>   `build_states.log` header that said `ordinary_amide_omega: unscaled`.
> * **Only one of six scaling surfaces read the unclassified list** — the ladder preflight's
>   `solute_document`. The rung writer, the fixed-tau stage and AIS preflights and the REST2
>   export scaled such candidates silently; `ScalingSelection.derive` crashed on `int('bond')`;
>   the executor's `solute.yaml` fallback ignored `omega_ambiguous_candidates`.
>
> `openmm.system.omega_exclusions` is now the one enforcing entry point, every scaling surface
> calls it, `build-md` refuses before `input/` or the run directory exists and passes the SDF
> through `preflight._ligand_sdf_beside`, and a `tau = 0` stage is still not classified at all.
> `tests/test_omega_unclassified_is_refused.py` covers each surface, an AST check forbids a
> scaling surface from calling `classify_omega_bonds` directly, and a slow test builds the
> macrocycle and asserts its rungs exclude exactly the three omegas. The refusal no longer offers
> `rest2.proline_like_residues` as a remedy: no user configuration can set it (`_legacy_cfg`
> rebuilds `rest2` from `DEFAULTS`), so it named a key `build-top.config` refuses.
>
> The text below is the entry as filed.

DEFERRED TO 0.5.4 by decision, 2026-09-16. The user's reading, held across three deferrals, is that
**the implementation is there and is not functioning** — the work is to enforce it, not to write it.

`md_tools.openmm.system.classify_omega_bonds` returns `omega_unscaled_bonds`,
`omega_proline_like_scaled_bonds` and `omega_unclassified_candidates`, and its own docstring states
the rule: "**A non-empty unclassified list must block production**" — a candidate was found that
neither rule could name, and guessing would silently change the Hamiltonian. **Nothing consumes
`omega_unclassified_candidates`.** Three callers take only the unscaled bonds:
`run/preflight.py`, `reference/rest2_export.py` and `rest2/selection.py` (grep the symbol; the line
numbers drift).

The case that matters is a solute whose `solute.kind` is `peptide` or `peptide-like`
(`build/top.py:48`, enum `("peptide", "peptide-like", "ligand")`). The classifier has exactly two
routes and `peptide-like` matches neither cleanly:

| route | evidence | fails on |
|---|---|---|
| `peptide` (default) | the amide nitrogen's residue name, against `PROTEIN_RESIDUES` and `proline_like_residues` | a whole-molecule solute with no residue evidence — every candidate falls to `unknown` |
| `ligand` | RDKit SMARTS `[CX3](=[OX1])[NX3]` over the retained SDF | needs `ligand_sdf` to be passed |

`kind: peptide-like` is built through the **whole-molecule ligand route** with a peptide-chemistry
map applied over the result — it exists for head-to-tail cyclic peptides — so it is precisely the
shape that has no residue evidence while not obviously being asked to use the SDF route. That is a
hypothesis with a clear test, not a diagnosis.

**Cost if real:** a ladder that exempts no omega scales ordinary amide omegas like any other solute
torsion, breaking the REST2 invariant. The run completes and the acceptance ratios look plausible,
so nothing surfaces it. **Rule out first:** a `tau = 0` run legitimately logs `omega bonds unscaled:
not applicable at tau = 0`, which can mask the real path.

---

## 14. The `.in` file's divergences from Amber's input conventions are not written down

DEFERRED TO 0.5.4 by decision, 2026-09-16. Not a defect — an undocumented boundary.

The surface is Amber-*like* on purpose: `docs/md-run.md` opens by promising that someone who has run
`pmemd -i mdin -p prmtop -c inpcrd -o mdout -r restrt` can read every command without a manual, and
the flag table maps each flag to its Amber counterpart. Some divergences are already stated and
reasoned — `-s` has no Amber counterpart and deliberately avoids `-x`; `mdout.csv` is the analogue
of Amber's `mden` rather than of `mdout`; the decomposition is by force group, not by Amber energy
term; there is no `ntwe` because one `info_printout` drives all three energy files.

What has NOT been done is a term-by-term audit of the `&cntrl` / `&remd` / `&AIS` sections against
Amber's own namelist variables, saying for each one whether it is matched, renamed, deliberately
absent, or means something different here. Without that, a user arriving from Amber cannot tell a
deliberate divergence from an omission, and neither can the next person to extend the parser. The
deliverable is a documented mapping, plus a refusal for any Amber variable we accept-and-ignore —
`CLAUDE.md` already requires that every accepted flag does its job or is refused.

---

## 15. AIS should become an Amber-style transformation between two topologies — IMPLEMENTED for 0.5.4

> **IMPLEMENTED 2026-09-16** on `ais/two-topology`, parameters-only V0/V1 with linear mixing.
> Design, decisions and measurements: `docs/amber-like-fix/AIS-two-topology.md`. Out of scope and
> refused: atom mapping, softcore, dummy atoms, non-linear schedules (the stated future work).
> The entry below is kept as the record of why.

DEFERRED TO 0.5.4 or later by decision, 2026-09-16. A design change, not a fix.

The user's direction: "in the future I want to just use the amber-like transformation between two
topologies." Today AIS switches ONE topology along `tau`, which is the only public persisted
coordinate; the proposal is Amber's alchemical construction, a V0/V1 topology pair with the
transformation defined between them.

This touches three `CLAUDE.md` scientific invariants at once — `tau` as the sole persisted
coordinate, the three-term identity
`U = U_non_scaled + sqrt(lambda)*U_sqrt_scaled + lambda*U_lin_scaled`, and the work convention
`dW_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)` — so it is not an increment on the present
implementation.

`docs/amber-like-fix/AIS.md` is the prior analysis and opens "**Nothing in this document is
implemented.**" Read it for what it establishes, not as the plan: Amber §27.8 Jarzynski is AIS's
real counterpart and has been in sander for years; our work convention (a finite potential
difference at frozen coordinates) diverges from Amber's `(dU/dlambda)*dlambda` and is better
conditioned; Amber is ~23% faster per step because its lambda change is a constant-block copy while
ours re-uploads parameters. Its own proposal — a parameter-offset fast path — is a SPEED change to
the current single-topology design and is a different piece of work from this entry. One unresolved
question there belongs to this one: the default schedule runs `tau` **downhill** (0.5 → 0.0) while
Amber's lambda conventionally runs 0 → 1, and direction fixes which ensemble the Jarzynski average
is taken over.

---

## 16. `build-top` accepted no supplied 3D structure for a small molecule — RESOLVED in 0.5.3

> **RESOLVED 2026-09-16.** `-i` now takes a `.sdf` beside `.pdb` and `.smi`. The entry is kept
> because of what closing it uncovered, which was not the feature.

A ligand or peptide-like solute could only be built from a SMILES string, so its conformer was
always *generated* — ETKDGv3, then MMFF94s over every embedding, lowest kept. For a molecule whose
pose is the point — a docked ligand, a crystallographic conformer, the output of another pipeline —
there was no way in that did not discard it.

`.sdf` supplies the coordinates, and they are used **as given**: no embedding, no minimisation.
That is the whole difference between the two molecular-graph routes, and it ends at one function.
`initial_structure` (SMILES) and `initial_structure_from_sdf` (SDF) write the same two files —
`solute.sdf` and `solute.pdb` — into the same place and return the same keys, so everything
downstream is identical. There is no seed to record on the SDF route because nothing was sampled;
the input file's sha256 is recorded in its place, and the log says
`coordinates: supplied by the SDF, used as given`.

**What it uncovered, which is the reason this entry exists.** `initial_structure` was called with
the staging ROOT on the explicit route and with `staging/structure` on the implicit one, while
`build/top.py` copies `built.sdf` out of the latter. So **an explicit-solvent ligand build emitted
no `built.sdf` at all** — silently, with the build reporting completion. Bond orders are not
recoverable from a topology, and three consumers need that file:
`classify_omega_bonds`'s ligand route, `peptide_map.map_from_sdf`, and
`preflight._ligand_sdf_beside`. An explicit-solvent REST2 ladder over such a solute therefore had
nothing to perceive amides from.

It survived because the assertions and the builds were disjoint sets: every test that asserted
`built.sdf` ran under `GBn2`, and the suite's explicit ligand builds asserted radii and records
instead. **This is a candidate cause of entry 13 on the explicit path** — a classifier with no SDF
produces exactly the unclassified-candidate symptom described there — and entry 13 should be
re-checked against an explicit build before its `peptide-like` hypothesis is pursued.

Also corrected while here: the kind × suffix rules were checked *below* the output `mkdir`, so
every refusal of them created the output directory first. They now run above it, with the
SDF's shape, so a refused build leaves nothing behind. `tests/test_build_top_input_formats.py`
pins the whole nine-cell matrix, the SDF refusals, and `built.sdf` under both solvents — none of
which had any test before.

---

## 17. The documentation is not readable as documentation — OPEN, scoped

Filed 2026-09-16. Not a defect: a usability and accuracy gap, with a decided direction.

76 markdown files, ~19k lines, and the three journeys a user actually needs — set up an
environment, configure a machine, run a simulation — are not separable from development
evidence. Concretely, and each verified rather than asserted:

* **Every copy-pasteable command in the method pages is one path short of working.**
  `../built.pdb` / `../built.xml` appear in the cMD, REST2, rREST2, AIS and umbrella pages and in
  `data_register/README.md`; the layout is `../build/built.pdb`. The generated-file trees in
  `openmm_methods/cMD/README.md` and `openmm_methods/REST2/README.md` still show the pre-0.5.3
  flat layout and contradict `run-layout.md`, which is the implemented authority.
* **Three version claims disagree**: `README.md` says 0.5.2, `pyproject.toml` says 0.5.3,
  `scientific-defaults.md` says `0.4.0.dev0`.
* **Four cited paths do not exist**: `docs/journal/` (twice, in `support-matrix.md`),
  `docs/examples/hmr-4fs.yaml`, `scripts/retrofit_fair_v030.py`.
* **`support-matrix.md` describes a retired layout** — `build-top` writing `inputs/` and
  `build-md` writing `MD/` — and its portability claim contradicts `run-layout.md` §5.
* **`rREST2/README.md` contradicts itself about a default**: line 48 says the velocity policy is
  `stored`/`inherit`; line 104 and the generated schema say `resample`.
* **`umbrella` is a supported protocol** with a README and shipped configs, and appears in no
  index and no protocol count.
* **`md-configuration.md` and `run-layout.md` are linked from no index**, so the generated
  configuration reference and the layout authority are findable only by someone who knows.
* **`run-layout.md` is headed "Status: IMPLEMENTED" and is not a reliable authority.** Its §6 is
  an implementation to-do list, and at least one filename in it contradicts the code: it gives the
  ladder's CV series as `remd<n>/cv_state<n>_prod<x>.dat` + `.json`, while `md_tools.remd.cv_states`
  — which `CLAUDE.md` names as the authority — writes `cv_state<i>.csv` with a `cv_state<i>.json`
  sidecar, flat in the run's output directory and with no segment in the name. Eight CV integration
  tests (`test_cv_cuda_lanes.py`, `test_cv_mpi_cuda_rrest2.py`, `test_cv_mpi_cuda_lanes.py`,
  `test_cv_definition_change_refusal.py`) read the `.csv` form, so that is what is written. This
  was found the hard way: the stale name was copied out of `run-layout.md` into the REST2 method
  page during the 0.5.3 documentation pass and had to be corrected.

  **The same disagreement exists inside the package.** `layout.py` exposes `state_cv()`,
  `state_restart()` and `state_trajectory()`, and **none of the three has a caller in `src/`** —
  only `tests/test_layout.py` and `tests/test_state_paths.py`, which pin the unwritten spelling
  (`state_cv_name(0) == "cv_state0_prod1.dat"`). `state_system()`, beside them, IS used
  (`build/rungs.py:92`, `build/md.py:1555,2127`). An unused accessor is not proof the file is
  unwritten — the per-state trajectories it names are on disk in the reference run — but for the
  CV series nothing writes the layout's name at all, and `restart_state<n>_prod<x>.*` appears in
  no reference run either. This is entry 10's defect class: a name that is declared, tested, and
  produced by nothing. Decide which module owns per-state filenames before the site publishes
  either spelling.

  **`bundles/` is the same pattern.** `layout.py` declares `BUNDLES = "bundles"` with a
  `bundles()` accessor (`:247`), `run-layout.md` gives it a section of its own, and **nothing in
  `src/` calls that accessor** — a freshly generated REST2 run has no such directory, confirmed by
  generating one. It was copied from `run-layout.md` into the REST2 method page during this pass
  and removed again. Either `export-reference` should write there, or the slot should stop being
  documented as part of the layout.

**Direction, decided by the user on 2026-09-16:** MkDocs Material published to GitHub Pages (the
repository is public), carrying the user-facing pages only; history, campaigns, integration notes,
`amber-like-fix`, `claudecode-instructions` and the dated release evidence stay in the repository
and off the site. `md-configuration.md` must be *built*, never hand-authored —
`tests/test_configuration_manual.py` fails on an edit. The README shrinks to an overview that
points at the site.

**Do the truth pass first.** Five tests pin documentation paths or content, and
`tests/test_documented_commands_run.py` extracts and *executes* every `md-run` command block from
`README.md`, `CLAUDE.md`, `docs/md-run.md` and three method pages — so the stale commands are
mechanically checkable. Publishing before fixing them would put broken commands on a website.

Entry 14's Amber-namelist mapping is a user-facing page for the same reader, so the site's
structure should leave it a place rather than be rearranged for it later.

---

## 18. `solute.residue_name` is resolved, recorded, and never applied — CLOSED in 0.5.4

Filed 2026-09-16, found while adding `.sdf` input. Small, and not urgent.

The schema promises "a deterministic name is assigned from the file and recorded, so the same
input always produces the same residue identity", and `build/top.py` does compute it
(`_assigned_residue_name`) and write it into `interpretation.residue_name`. **Nothing consumes
it.** A grep for the symbol outside `build/top.py` finds only an unrelated test name;
`reference/export.py` reads `interpretation.route` and nothing else from that dict. The residue in
the built PDB is whatever RDKit's `MolToPDBFile` writes, which is `UNL`.

So the recorded name describes nothing, and a user who sets `solute.residue_name: LIG` gets a
System whose residue is still `UNL` while the record says `LIG`. Either apply it to the written
topology or stop recording it as though it were applied; a value that is documented, accepted and
inert is worse than one refused.

**Closed (0.5.4, by applying it).** Both molecule preparers name the residue before writing the
prepared molecule, so `built.pdb`, `built.solute.pdb` and the serialised topology carry it on the
`.smi` and `.sdf` routes under either solvent, and build-top verifies the built topology carries
it before placing any output. The molecule is written as `<RESNAME>.sdf`, titled with the name,
instead of `built.sdf`; `preflight._ligand_sdf_beside` reads `<system stem>.sdf` first (older
builds) and otherwise the file named for the one non-solvent residue in `<system stem>.pdb`. A
stated name must be three letters or digits and must not already mean water, an ion or a protein
residue -- solvent selection and the omega classifier read residue names -- and is refused under
`kind: peptide`, where it would be inert. Tests: `tests/test_build_top_residue_name.py`.

---

## 19. A chemistry mismatch is a user-input error that exits 1 and leaves a directory — OPEN

Filed 2026-09-16, found while writing the `build-top` documentation page and checking its claims
against the program rather than against the source.

Two refusals of the same kind — "this input cannot build what you asked for" — behave differently:

| input | exit | leaves behind |
|---|---|---|
| `.sdf` with a 2D conformer, `kind: ligand` | 2 | nothing |
| `.sdf` of ethanol, `kind: peptide-like` | **1** | **`<outdir>/built.log`** |

The first is checked in `build/top.py` above the output `mkdir`, raises `ConfigError`, and the CLI
maps it to 2. The second is `PeptideMapError`, raised from `openmm/implicit.py` during
parameterisation — long after the log is open — and reaches `cmd_build_top`'s bare
`except Exception`, which prints it and returns 1.

**Why it matters.** 1 means "internal failure" everywhere else in this CLI, and the message is a
bare exception class name rather than a refusal naming `-i`. A script distinguishing "the user
gave me the wrong thing" from "the tool broke" gets the wrong answer, and the directory left
behind holds a `built.log` describing an attempt, which `CLAUDE.md` says a refusal must not do:
*"A `-odir` holding a `resolved.config` is indistinguishable from a run that happened."*

**The fix is not simply to raise `ConfigError` there.** The mismatch is only detectable once the
molecule has been read and mapped, which is after the point where nothing has been written. Either
the peptide-like map runs as a preflight over the input before any output is created — it needs
only the SDF, which is available — or the build becomes transactional, staging into a temporary
directory and publishing on success. The first is smaller and matches how the suffix checks were
moved above the `mkdir` in 0.5.3.

Related: entry 13, which concerns the same `peptide-like` classification route.

---

## Moved to a development branch

Requirements that were deferred rather than done, now carried by a branch with aims and acceptance
criteria of its own. They left this list because something owns them, not because they were closed.

| requirement | filed by | now |
|---|---|---|
| selected-residue REST2 — backbone and sidechain selection, chain-aware residue identity, torsion versus nonbonded membership, cross-boundary interactions | [20260917 instruction](history/claudecode-instructions/20260917_next-release-reusable-ligands-propka-cuda-ais.md) §9 | [0.6.1 aims](development/0.6.1/AIMS.md) |
| thermodynamic integration — lambda schedules, window sampling, dU/dlambda, quadrature, uncertainty | same, §9 | [0.7.0 aims](development/0.7.0/AIMS.md) |
| free-energy perturbation — supported transformations, BAR/MBAR, overlap diagnostics, atom mapping, softcore, charge-changing corrections | same, §9 | [0.7.0 aims](development/0.7.0/AIMS.md) |
| bidirectional switching and alchemical transformations | same, §9 | [0.7.0 aims](development/0.7.0/AIMS.md), with AIS's own contract unchanged |
| OpenMMTools-compatible AIS Langevin splitting | same, §9 | still deferred; no branch owns it |
| more advanced protonation and tautomer preparation | same, §9 | still deferred; no branch owns it |
| a non-periodic System with no GB force and no build record is labelled `implicit` in stage records (`run/preflight.py`: `implicit = not periodic`) | S2 finding, 2026-09-19, while adding vacuum builds | pre-existing 0.6.0 behaviour; build-top vacuum builds are identified by their record and refused on ordinary paths (0.7.0), but an unrecorded vacuum-like System still carries the wrong label |
| build-top builds an explicit box whose edge is barely above 2 × the nonbonded cutoff, which an NPT run then shrinks below the limit | S4's M2 campaign, 2026-09-19: a 1.9 nm cube at 0.9 nm stopped every 1 ns NPT window within minutes | open: a build-time warning (or refusal) below a stated margin, e.g. 2 × cutoff + 0.8 nm, recorded in built.log |
| build-top reports a residue whose atoms are present under NON-AMBER NAMES as "missing heavy atoms", and the documented fix (`input.missing_atoms: add`) would then ADD duplicates | S2, 2026-09-20, preparing TYK2: upstream renamed UNK→ACE/NME but kept Maestro atom names | message fix assigned on 0.6.1 (name the template and the unmatched names; never suggest adding atoms) |
| a cap / atom-name canonicaliser in build-top (Maestro or other conventions → Amber template names) | same finding | deferred by S0: renaming atoms in a user's structure is an edit to their file, and build-top's line is that such edits are theirs. Preparation scripts do it explicitly instead |
| a repo script that shells out to `md-openmm` can silently run the INSTALLED md-tools instead of the checkout: build-top runs with cwd inside the build directory, so a caller's relative `PYTHONPATH=src` stops resolving | S2, 2026-09-20, building the TYK2 fixture against openmm-env's md-tools 0.5.4 without noticing | fixed in that script by pinning the subprocess to the md_tools it imported; every other script that shells out has the same hazard, and it is the same shape as the installed-wheel example test that has been failing all along |
| there is no cheap way for a NEIGHBOUR to tell whether a ladder is alive: at a 10 ps exchange interval a 7-rank ladder spends ~95% of wall time in barrier, IO and coordination (555 steps/s where the box alone does >10 000), so a point-sampled `nvidia-smi` reads 0% on a perfectly healthy run — indistinguishable from a process that died without exiting | hpREST2, 2026-09-21, measured; our S3 correctly refused to judge a live run from 0% utilisation | open: a heartbeat line somewhere cheap to poll — exchange index and wall clock — so a neighbour can answer "is this alive" without reading someone else's NetCDF or guessing |
| an MPS control daemon dies SILENTLY when its pipe directory path is too long: the socket is AF_UNIX, limited to 108 bytes, and a session scratch path is ~150. It logs "Starting control daemon" and vanishes, and `nvidia-cuda-mps-control` then cannot find it | S1, 2026-09-21, starting a daemon for the TYK2 ladders; fixed by moving to `~/.cache/s1-mps/` | open: refuse a pipe directory longer than the socket limit BEFORE starting the daemon, naming the limit and the length. Also worth recording: two daemons with different pipe directories coexist fine, so one-per-user is not the constraint — path length was |
| the MPS server renumbers cards for its clients: a server started with `CUDA_VISIBLE_DEVICES=1,2,3,4` exposes them to clients as 0,1,2,3, so a client that also sets 1,2,3,4 asks for a fifth card and every rank fails with "Illegal value for DeviceIndex: 3" | S1, 2026-09-21 (and hpREST2 hit the same thing that morning) | open as documentation and a preflight message: the refusal should name the renumbering rather than the index. The upside is worth stating too — a server bound to 1–4 cannot reach card 0 at all, which is a stronger exclusion than remembering not to use it |
| `build-top`'s COMPLEX route does not write `<RESNAME>.sdf` beside the built System, so `build-top --rest2-scaler` refuses the ligand's bond orders on exactly the protein–ligand systems selective REST2 exists for | S1, 2026-09-21, on the TYK2 complex; worked around with one `cp` from the package, recorded in the tutorial rather than hidden | **STILL OPEN — the code has NOT changed.** `top.py` still sets `residue_name` only under `not peptide and not complex_build`, so `out_sdf` is None on the complex route. DECIDED 2026-09-21 (S2's argument, against S0's first preference): **build time writes `<RESNAME>.sdf` from the PACKAGE's molecule**, never re-perceived from the complex PDB. The scaler must NOT resolve packages from the catalog: `build/scaler.py` names neither `built.log` nor `LigandPackage`, and giving it the catalog would make a built tree stop being self-describing — re-deriving scaled states elsewhere would need `$MD_DATA` with the same packages registered, trading a loud missing-file refusal for a tree that builds on one machine and refuses on another. Four refusals go with it: never re-perceive bond orders from a PDB; two different packages on one residue name collide and are refused; the legacy `<stem>.sdf` refusal applies on the complex route too, since the run-time preflight reads that name first; and the file is in `built.log`'s outputs with its sha256 or the tree is not verifiable. After TYK2 |
| a session that starts MPS must set `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` explicitly, never inherit the default: stale directories in `/tmp/nvidia-mps-*` survive for days and a stale default once made 110 GPU tests SKIP while the run exited 0 | ours, 2026-09-18, still on this machine; hpREST2 binds its own under `~/.cache/` and is immune | open as documentation and a preflight check: the record should state which pipe directory a multi-rank run used |
| a run's preflight record distinguishes cores VISIBLE from cores ASSIGNED, but never says how many were CONTENDED at start, so a run degraded by an unplanned neighbour cannot be diagnosed afterwards — the first two numbers explain the plan, the third explains the throughput | hpREST2, 2026-09-20: its control degraded 1.47 → 1.79 min/ns (22%) while ~18 cores of unpinned multi-threaded work (MD-tools test lanes, ours) competed with its pinned ranks | open: record the contended figure at start. The planner cannot stop an unplanned neighbour — that is the cgroup's or the queue's job — but the record can make the loss legible instead of unexplainable |
| the planner's "cores per worker" has no measured basis, and more cores cannot help a rank that saturates one | hpREST2, 2026-09-21: 14 ranks across two jobs used 13.7 cores of real CPU — **about one core per rank**. It declined offered cores on that measurement: six ranks on six whole physical cores already carries a factor of two of headroom, and extra cores would sit idle inside its cpuset where another session could have used them | open: size blocks from measured per-rank CPU use rather than a default, and record the measurement beside the plan. A rank that is CPU-bound is then an ASK with a number behind it instead of a preference |
| a ligand package registered with WRONG or ABSENT aliases cannot be repaired in-tool, and re-registering reports success while changing nothing | S2, 2026-09-21, verified in `ligands/catalog.py:106` and confirmed by S0: the catalog path is `<compound-id>/<parameter-id>`, the parameter id digests the PARAMETERS rather than the names, so the same molecule lands on the same path; `register_package` compares `parameter_digest` and `chemical_state["digest"]` and returns the existing package with `newly_written=False` -- printed as "already registered, kept". **Aliases are in neither comparison.** The second registration is therefore ACCEPTED AND INERT: two commands report success and the name is still missing. `metadata["compound"]["aliases"]` defaults to empty and no tutorial sets it. Note the ASYMMETRY that is NOT true: "absent is permanent, wrong is fixable by parameterising again" is false, and S2 caught it in its own row -- neither is repairable, recovery means deleting the catalog entry by hand outside md-tools | open: refuse a re-registration that differs in aliases, or support an explicit alias amendment that says what it changed; and until then the TUTORIAL fix is the ONLY fix a reader has, so every tutorial that parameterises a ligand sets `aliases` and says in the text that it cannot be added afterwards. A dataset has no alias field -- `data_name` and `notes` carry its findability and are reviewable before the write -- so the trap lives at `build-top --parameterize`, not at `data-register` |
| `plan_cpus` assigns LOGICAL cpus, so a "block of 2 cores" is one physical core or two depending on where it lands, and two plans that look disjoint can share silicon | hpREST2, 2026-09-21, measured on this host: its ranks on 21,32/22,23/… and S1's TYK2 ranks on 0,24/1,25/… put one rank of each on physical cores 24, 25, 26, 28 and 30. Both planners planned correctly; neither could see the other, and the logical numbering hid the collision. S0 confirmed the sibling map directly from `thread_siblings_list`: 24 physical cores, and the sibling of logical N is uniformly **N ± 24** — NOT N+1, and not the scattered pairing it looks like when read off a handful of cpus | open: assign by PHYSICAL core, giving both siblings of a core to the same owner, and count `contended cores` as PHYSICAL contention (`/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list`) — counting logical cpus reports zero contention for the case above. **Do not key on `core_id`**: it is unique only within a package, and on this host logical {0,24} and {12,36} both report `core_id 0`, so a planner keyed on it would merge two different physical cores. Key on the sibling set, or on (`physical_package_id`, `core_id`) |
| the CPU block planner divides the CPU set exactly among ranks, so co-scheduling ladders of different widths (6 rungs and 7 rungs) forces two block sizes or a multiple of 42 | hpREST2, 2026-09-20, running a 7-rank and a 6-rank ladder on disjoint card and CPU sets | open, 0.6.1 or later: accept a REMAINDER — N ranks over C CPUs as blocks of floor(C/N) with the leftovers unassigned. Refusing rather than silently overlapping stays the rule; this only widens what can be accepted |
| a vacuum alchemical leg that applies the solvent leg's STATED 1-4 scale (so OPC-solvated hydration cycles can pair with vacuum) | S2 finding, 2026-09-19: OPC applies 0.833333 to a ligand's 1-4 pairs, a vacuum build 5/6 | deferred by S0; until then `matched_legs` refuses the pairing by name and TIP3P is the working water model for hydration cycles |

The two "still deferred" rows are listed so that archiving their instruction does not lose them.
Nothing above is implemented, and an aims page is not an implementation.

## Cross-references

- `docs/release-notes/20260907-cv-validation-final-evidence.md` — the evidence behind the
  validation work these gaps were carved out of, including its own "Open item" section for
  entry 4.
- `docs/release-notes/20260907-readiness.md` — the readiness note for the pinned commit.
- `docs/release-notes/20260909-ais-work-measurement.md` — `ais.work_measurement`, and why the
  default changed to `work`. A behaviour change for any AIS configuration that did not set it.
- `docs/release-notes/20260909-ladder-throughput-and-intervals.md` — the O(N) `rem.log` gather,
  and the three reporting intervals the ladder was discarding. Includes the one invariant that is
  now enforced without having been tested (a checkpoint must land on an exchange boundary).
- `docs/release-notes/20260909-resume-identity-and-force.md` — resume identity narrowed to what
  determines the calculation, after a defaulted AIS field made every in-flight run of every
  protocol unresumable. Records the deferred O(N) `rem.log` render and the length at which it
  starts to matter.
- `docs/release-notes/20260909-rest2-switching-cost.md` — tau by global context parameters (1172x
  on the switch), Context reuse across AIS paths, and the two places it does not work: CMAP, and
  explicit solvent's long-range dispersion correction. Corrects the cost claim in the
  work-measurement note above.
