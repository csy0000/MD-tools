# Resume identity, a flag that did not exist, and the remaining O(N)

Second instruction from the hpREST2 campaign, after `471daac` was independently verified there.
Items 1 and 2 are fixed with tests; item 3 is a deliberate deferral with the measurement behind it.

## 1. An unrelated defaulted field made every in-flight run unresumable

**This was a regression introduced by `8fd3414`** — the AIS work-measurement commit — and it cost
the campaign a restart of three 500 ns ladders. The defect is not in that commit's fields; it is
in the gate they tripped, which has always been this strict. Adding two defaulted `ais:` settings
is a perfectly ordinary thing for an engine upgrade to do.

`_write_resolved` compared the stored `resolved.config` with the newly resolved one as whole
dictionaries:

```python
if existing != run_input.resolved:
    raise SystemExit(f"{path} already exists and describes a different run. ...")
```

So the complete diff between a running ladder's stored document and the same input re-resolved on
`471daac` was

```
   parameter_update_interval_steps: 1
+  work_measurement: work
+  verify_every_updates: 0
```

— two settings a REST2 run never reads — and that was enough to refuse `--resume`. The only exit
was `--overwrite`, which destroys the outputs, so in practice the runs were abandoned. **Any
upgrade that adds a defaulted field to any protocol made every in-flight run of every protocol
unresumable**, which is precisely backwards: the long runs are the ones that most need a fix.

### What identity means now

`src/md_tools/run/resume_identity.py`. A run's identity is what determines its CALCULATION, on
two axes:

- **Which sections.** A protocol's identity covers the sections that protocol actually reads,
  taken from `build.md._IN_SECTIONS` — the same map that decides what its `.in` file can express.
  One definition of "what this run depends on", so the input language and the resume gate cannot
  drift apart. A REST2 resume does not care what `ais:` says.
- **Which changes.** A field absent from the stored document whose new value is the schema default
  was added by an upgrade and chosen by nobody. A field present in both with different values is a
  change. A field present only in the new document with a NON-default value is also a change — it
  was set deliberately, under settings this run was never made with.

`resolved.config` now carries a `schema_version`, never compared, so that a missing field can be
read as "written before this existed" rather than "deliberately removed". Absence alone cannot
distinguish the two and the difference decides whether continuing is safe.

An unknown protocol is compared strictly — everything matters. Refusing a resume that would have
been fine costs a rerun; permitting one that changes the calculation costs the result.

### What was deliberately NOT relaxed

The reporting intervals are still part of a REST2 run's identity. After `471daac` the ladder
honours `crd_printout_solute` and `info_printout`, so a run started before it legitimately differs
in frame spacing, and refusing that is correct — the cadence would change mid-run. The
instruction warned against fixing item 1 by loosening this, and it is pinned by six parametrised
cases.

**The protection does not rest on this gate alone**, which is what makes narrowing it safe.
`ReplicaRun.compare_identity` compares the schedule stored inside the exchange record itself, and
`schedule.describe()` puts `checkpoint_interval_ps`, `solute_output_interval_ps` and
`whole_output_interval_ps` there. Two independent gates; only the outer, blunter one changed, and
a test asserts the inner one still catches all three intervals by name.

The refusal now names every differing field and both values, modelled on `IdentityError`:

```
resolved.config describes a different run:
    reporting.crd_printout_solute: was 1250, now 2500
```

"Describes a different run" is true and useless — the reader had to diff two documents by hand to
learn what the engine already knew.

## 2. `--force` was advertised by a command that does not define it

Two messages reached from `md-run` told the user to pass `--force`:

- `remd/generated.py:428` — "Delete it to regenerate, or pass --force."
- `remd/state_trajectories.py:59` — "pass --force to replace a run deliberately"

`md-run` defines no `--force`. It is real on the generated `REST2.py`, on the executor and on
`md-openmm register` — three other programs. A user following the instruction got
`unrecognized arguments: --force` and fell back to deleting files by hand, which is the more
dangerous of the two remedies the message offered.

Both messages now name **`--overwrite`**, which is valid on `md-run` AND on the generated ladder
script, and which the generated script already maps onto the executor's `--force`. `md-run` did
not gain a second destructive flag under a different name: it already has one, and two spellings
of the same intent on one command is how this confusion starts.

Each message also says explicitly why it is not `--force`, so nobody "corrects" it back.

**Every other message was audited**, as asked. `test_no_message_tells_an_md_run_user_to_pass_a_flag_md_run_does_not_define`
scans remedy instructions — `pass --x` — in the modules on the `md-run` path and asserts every
named flag is one `md-run` defines. It scans imperatives rather than every mention: a docstring
describing another command is fine, telling THIS user to pass something that does not exist is
not. The remaining `pass --x` instructions in the tree all belong to the command that defines
them (`--init` on `data-register`, `--pdb` on `build-top`, `--extend` on the executor).

## 3. `rem.log` regeneration is still O(N) — deferred, with the number

Not reopening `471daac`. Each regeneration is now two bulk reads plus a full render, both linear
in N, and after item 2 of the first instruction they run once per `checkpoint_printout` rather
than once per exchange. Measured here on the campaign's own record (N = 16,632): bulk read 0.31 s,
render 0.38 s, **0.69 s per regeneration**.

Extrapolated, at one regeneration per 100 exchanges:

| ladder length | per regeneration | per exchange | share of a 0.23 s exchange |
|---|---|---|---|
| 50k exchanges (500 ns at 10 ps) | ~2.1 s | ~0.021 s | ~9% |
| 500k exchanges (5 µs) | ~21 s | ~0.21 s | ~90% |

**Decision: deferred.** The quadratic is real but is now two orders of magnitude out, and it does
not bite below roughly a microsecond of ladder. Fixing it properly means incremental rendering
into a staging file with an atomic replace — preserving the crash-safety that made full
regeneration right in the first place — and that is a change to the write path of every ladder,
which is not something to attach to a bug-fix commit whose value is that it is provably
byte-identical.

Recorded here so the next person meets a number rather than a surprise. Two things to know before
picking it up:

- The property to preserve is that `rem.log` is a projection of `exchange.nc` and never an
  authority. An incremental writer must still be able to reconstruct the whole file, or a torn
  write leaves a log that disagrees with the record with nothing to say which half is true.
- `rem.log` still cannot be disabled on a running ladder: `rest2.rem_log` is in a section a REST2
  run reads, so changing it trips the identity gate — correctly, since turning it off mid-run
  would leave a stale log describing only a prefix. If it is ever made optional it needs the
  companion the instruction suggested: a supported way to produce the log afterwards from
  `exchange.nc`.

## Scope

Comparison, messages and reporting only. No change to the exchange rule, the acceptance
criterion, the Hamiltonian scaling, the tau ladder, or any written value.
