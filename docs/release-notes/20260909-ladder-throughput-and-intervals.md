# The ladder: a bulk `rem.log` gather, and the reporting intervals it was discarding

Three defects, reported from the hpREST2 campaign
(`projects/hpREST2/docs/claudecode-instructions/20260909_mdtools-remlog-and-ladder-intervals.md`).
All three were real and are reproduced below from this repository's own code. Nothing was written
into the hpREST2 tree; the one real file used as evidence was copied out and read from the copy.

## 1. `rem.log` regeneration replayed the history record by record

`_write_rem_log` rebuilds the log in full at every checkpoint and replaces it atomically. That
design is right and is unchanged: a full rewrite from committed rows is what makes the log
crash-safe, because a torn append would disagree with the authoritative record and nothing in the
file would say which half was true. `rem.log` remains a projection of `exchange.nc`, never an
authority, and nothing appends.

What was wrong was the READ. The gather was

```python
exchanges = [(proposed[i], accepted[i], self.reporter.reduced_potentials(i)[0])
             for i in range(last + 1)]
```

and netCDF4 charges per indexed access, so each regeneration cost O(N) in the run's own length.
`ReplicaReporter.reduced_potentials_upto` now returns the whole history in one slice.

**Measured on the campaign's own record** (a copy of `rest2_run_1/md_script/REST2.nc`, 4 states,
16,632 exchanges):

| gather | seconds |
|---|---|
| per-record, as shipped | 9.16 |
| one bulk slice | 0.31 |

**30x**, not the 70x in the report. The difference is not a disagreement about the defect: their
measurement was at N = 11,416 on a different filesystem and cache state, mine at N = 16,632 on a
copy under `/home`. Both show the same linear-in-N read with a large constant, and the direction
and magnitude of the fix are the same. I report what I measured rather than adopting their number.

**The bytes do not change.** `rem.log` rendered from those 16,632 real exchanges is sha256
`228c5481b07d4ad27c6fcbb26c7db32b3ff3fc99adcde33c54338f5ecd2665c9` from the per-record gather and
from the bulk gather alike. (Their `fc79f0d8...` is the digest at their N; a different history
renders a different log, so the digests are not comparable across the two measurements — the
comparison that matters is stock-versus-patched at one N, and it is identical.)

### What else was checked on the write path

`statistics()` was already a bulk slice, as are `state_to_walker()` and `reservoir_events()` —
`reduced_potentials` was the only per-record reader of the three, which is why the fix follows
the file's existing `upto` idiom rather than inventing one. `_reduced_potential_matrix` loops too,
but over `n_states²` per exchange, not over history, so it is not implicated and is left alone.
`reduced_potentials(index)` is kept for its single-record callers: this is an addition.

## 2. `checkpoint_printout` never reached the ladder

`ReplicaSchedule` defaults `checkpoint_interval_ps` to the exchange interval when it is `None`,
and the generated protocol passed nothing — so a configured `checkpoint_printout: 250000` (1 ns)
produced a checkpoint every exchange, 100x more often than asked. Since `rem.log` is regenerated
at every checkpoint (`driver.py:1475`), this is what turned defect 1 from a slow read into the
dominant cost of a run: the two multiply.

The configured value now reaches the schedule.

**A checkpoint interval off the exchange grid is refused, not rounded.** A ladder checkpoints at
exchange boundaries because that is where the state-to-walker mapping and the exchange RNG are
jointly defined, which is the reason the schedule defaulted to the exchange interval in the first
place. The refusal names both numbers and the two nearest legal values.

**Stated plainly: I enforced that invariant, I did not test it.** I did not establish
experimentally that a checkpoint written mid-interval breaks a resume. Refusing preserves the
property the code already documented; the alternative — honouring an arbitrary interval — would
have required demonstrating that mid-interval restarts are safe, which is a larger piece of work
than this instruction's scope. If that demonstration is ever done, the refusal is the thing to
relax.

## 3. The ladder discarded `crd_printout_solute` and `info_printout`

`protocol_file_text` wrote `whole_ps=exchange_ps, solute_ps=exchange_ps`, hard-wired. The
configured values were resolved, logged, written into `resolved.config` — and dropped without a
word. Reproduced exactly, at 4 fs:

| requested | ladder used, before |
|---|---|
| `crd_printout_solute: 1250` = 5 ps | 10 ps — half the resolution |
| `info_printout: 12500` = 50 ps | 10 ps — five times too often |
| `checkpoint_printout: 250000` = 1000 ps | 10 ps — a hundred times too often |

This is the worst of the three because it changed recorded DATA rather than runtime, and silently.
Both intervals are now honoured.

### The trap this nearly fell into

There are two ladder descriptions: the dict `build-md` assembles for its log, and
`ladder_from_resolved`, which the generated script rebuilds when it RUNS and which
`protocol_file_text` is actually handed. Patching only the first would have put the intervals in
the build log and never in the simulation — the same way `rest2.equilibration_steps` was once
added to one and not the other. Both carry the reporting block now, and
`test_the_runtime_ladder_reconstruction_carries_the_reporting_block` pins the one that matters.

### Older configurations

A `resolved.config` with no reporting block at all is an OLD description and keeps its historical
behaviour: both streams at the exchange interval, no checkpoint interval. Reading absence as "no
streams configured" would turn two working streams into one final frame each, and the run would
still look successful. An EMPTY reporting block is a different statement and is honoured as
written.

## AIS does not share the defect

Checked, because the instruction asked. AIS carries `crd_printout_solute`, `info_printout` and
`checkpoint_printout` into its generated `.in` and honours them at run time; its four reporting
cadences are independent and each is validated against `switching_steps`. The hard-wiring was
specific to the ladder's generator.

## Throughput

I did not run a production ladder to completion, so I have no measured exchanges/min to set
against the campaign's 9. What I can state is the arithmetic, from measured parts:

- the `rem.log` gather at N ≈ 16.6k falls from 9.16 s to 0.31 s per regeneration;
- regenerations fall from one per exchange to one per `checkpoint_printout`.

At the campaign's settings those compound to roughly 0.003 s of gather per exchange against
about 9 s. That is a derived figure, not a measurement of a running ladder, and it is stated as
such — the honest end-to-end number has to come from the next campaign run.

## Scope

Reading, plumbing and reporting only. No change to the exchange rule, the acceptance criterion,
the Hamiltonian scaling, the tau ladder, the event ORDER, or any written value. `rem.log` was not
made optional and cannot be disabled — the regeneration is simply cheap now.

## Note for hpREST2

hpREST2 is running a temporary overlay pinned at `records/hprest2/engine_patch.json`. Once this
commit lands the overlay can be retired and `MD_TOOLS_PIN` moved deliberately. The overlay's
`remd_bulk_read.patch` and this fix take the same approach; this one adds the checkpoint and
output-interval plumbing, which the overlay did not cover.

## Tests

`tests/test_rem_log_bulk_read.py`
- the bulk gather and a per-record reference render identical `rem.log` text;
- the bulk gather returns value-for-value what the per-record reader returns, `u_evaluated` too;
- regenerating reads `u` exactly ONCE. Counted, never timed — a timing assertion fails on a loaded
  machine for unrelated reasons and passes on a fast disk even if the loop comes back.

`tests/test_ladder_reporting_intervals.py`
- the generated protocol carries the three configured intervals, each deliberately different from
  the exchange interval and from the others so no substitution can pass by coincidence;
- a disabled interval (0) stays disabled instead of becoming the exchange interval;
- a checkpoint interval off the exchange grid is refused with both numbers named;
- the RUNTIME reconstruction carries the reporting block through to the protocol;
- a description with no reporting block keeps the historical behaviour.

All eight were confirmed FAILING on `61d5b35` before any source was changed, each for its own
reason — including `DID NOT RAISE ValueError` and `KeyError: 'reporting'`.
