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
`peptide-like` solute, the `.in` file's undocumented divergences from Amber's input conventions, and
rebuilding AIS as a transformation between two topologies.

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

NOT FIXED, deliberately: it is pre-existing, unrelated to the naming work that exposed it, and
changing where a runtime writes is a behavioural change for anyone already using `--all-in-one`.
`--check` was suspected of leaving the stray `eq/` behind and was cleared by measurement: a
`--check` into a fresh `-odir` creates nothing, as the contract requires.

---

## 13. The omega exclusion is not enforced, and `peptide-like` fits neither classifier route

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

## 15. AIS should become an Amber-style transformation between two topologies

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
