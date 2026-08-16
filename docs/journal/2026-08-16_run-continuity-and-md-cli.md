# Run continuity, crash-safe restarts, lifetime statistics, and a public MD command

**2026-08-16 (second pass).** Status: **the four behavioural requirements are implemented and
wired through the real CLI**, verified end to end on CPU for both methods. 231 tests pass. Two
integration bugs were found by running the pipeline that no unit test could have reached.

This finishes the parts the first refactor deliberately left: the naming helper existed but nothing
public called it, and the durable state it needed did not exist at all.

## Files changed

| file | why |
|---|---|
| `openmm/runstate.py` | **new** -- continuity contracts, atomic writes, restart generations, commit record |
| `openmm/runner.py` | shared run-directory stager; `launch_md`; `launch_rest2` takes naming options |
| `openmm/cli.py` | public `md` command; `--run-name` / `--resume-run` on `md` and `rest2` |
| `openmm/rest2.py` | global attempt index, lifetime counters, RNG/phase persistence, generation commits, additive chunks |
| `openmm/md.py` | additive chunks, generation commits, State fallback |
| `openmm/schemas.py` | optional `md:` block in experiment manifests |
| `openmm/config.py` | maps the `md:` block into `production.md` and records that it was declared |
| `manifests/experiments/smoke.yaml` | a tiny `md:` plan, so the public MD path has a worked example |
| `README.md`, `PORTABLE_REST2.md` | the stale "runs are immutable" claim replaced with what actually happens |
| `tests/test_explicit_portable.py` | +21 tests |

## Persistent-format decisions

**One record decides what is committed.** At each chunk boundary both restart forms -- an OpenMM
binary checkpoint and a portable serialized State -- are written into `restart/gen_NNNN/` through
temporary files and renamed. Only then is `restart/committed.json` atomically replaced to name that
generation. Two independently replaced files are not an atomic pair, so nothing infers
"committed" from the presence of files: a crash part-way through writing generation N leaves N-1
committed and intact, and the previous generation is retained rather than pruned immediately.

**Everything is versioned.** `run_state.json` and `committed.json` both carry `schema_version`, and
readers refuse a version they do not know with a message saying to start a fresh run. A directory
from the immediately preceding refactor has no `run_state.json` at all, and is refused for resume
with that stated -- not silently treated as continuable.

**`n_chunks` is what the invocation adds.** Read as a lifetime total it made resuming a no-op, since
the first invocation had already reached it. Totals remain derived and reported; nothing
reconstructs a plan from them.

## Compatibility checks implemented

Compared *before* any state is loaded or any file opened, and refused with every differing field
named:

* method (`md` vs `rest2`) -- different runs even when everything else agrees;
* prepared-system fingerprint, particle count, bundle config hash;
* force fields, water model, ions, box shape and padding;
* nonbonded method, cutoff, switching, dispersion correction, PME tolerance, constraints, rigid
  water, HMR mass and scope, centre-of-mass removal, minimum-image margin;
* integrator kind, timestep, temperature, friction, ensemble, precision;
* the REST2 ladder, exchange interval, and `rest2.omega_exclusion`;
* chunk **length**.

Explicitly excluded: chunk **count**. Treating "run more" as an incompatibility would make
extension impossible, which is the point of resuming. `write_run_state` additionally refuses a
record whose declared method disagrees with its own contract, since that would defeat the check.

## Two bugs found by running, not by testing

**The resume silently restarted from chunk 0.** After a REST2 run, the driver's output was moved up
from `run_dir/rest2/` to `run_dir/`. The second invocation then looked for prior chunks where the
driver writes -- the subdirectory -- found none, and began again from zero *in the same directory*,
overwriting the exchange log. The suite was green throughout. The driver now writes directly into
the run directory and nothing is moved.

**Conventional MD silently launched 1000 ns.** The smoke experiment declares only a `rest2:` block,
so the MD path fell through to the package default of 10 x 100 ns. It was found by the run not
finishing in ten minutes. An experiment without an `md:` block is now refused for `md`, naming what
it would otherwise have launched: a REST2 ladder says nothing about how long one walker should run,
and guessing is worse than refusing.

## Tests and smoke commands, with results

```console
$ python -m pytest tests/ -q -m "not slow"
231 passed, 2 deselected, 1 warning

$ python -m compileall -q src scripts                       # clean
$ md-openmm --help | md-openmm md --help | md-openmm rest2 --help   # all exit 0
$ rg -n -i '<retired-identity pattern>' . -g '!*.pyc' -g '!build/**' -g '!dist/**' -g '!.git/**'
0 matches

# The pattern itself is deliberately not spelled here. Quoting it verbatim makes this file match
# the audit, so the gate would report one hit forever and stop being a useful signal. The literal
# command is in the refactor instruction under claudecode-instructions/, which is gitignored.
```

End-to-end on CPU, from `/tmp` with no source checkout on the path:

```console
$ md-openmm md    --bundle B --out-root O --run-name md_a       # rc=0, chunks 2/2
$ md-openmm rest2 --bundle B --out-root O --run-name r2         # rc=0, 4 attempts
$ md-openmm rest2 --bundle B --out-root O --resume-run r2       # rc=0, chunks 3/4 then 4/4
```

Observed after the resume:

| property | result |
|---|---|
| directory used | `r2` -- no sibling, no child |
| chunk directories | `chunk_0000` … `chunk_0003` across two invocations |
| CSV headers in the exchange log | 1 |
| `attempt_index` | 0 → 7, monotonic, no duplicates |
| step | 250 → 2000, continuous across the boundary |
| lifetime vs invocation attempts | 8 vs 4, separately labelled |
| committed generation | 3, with `gen_0002` retained |
| exchange RNG | `restored` (not reseeded) |
| replicas constructed by the MD path | 0 |

## Remaining limitations

**State-fallback reproducibility.** When the binary checkpoint is missing, corrupt or from another
platform, the run continues from the serialized State. That carries positions, velocities, box
vectors, time and parameters, so the continuation is physically valid and statistics are preserved
-- but the stochastic integrator's internal stream is **not** restored, so the trajectory diverges
from what an uninterrupted run would have produced. It is announced on stdout and recorded in the
summary as `restart_restored_from`, never silent.

**No end-to-end crash-injection test.** The commit protocol is tested at the unit level -- a
half-written generation cannot take over the commit record, and the previous generation survives --
but there is no test that kills a real run mid-chunk and recovers it. That is the gap most worth
closing next.

**Exchange RNG continuity is best-effort for older runs.** A run committed before the generator
state was recorded falls back to the derived seed, which is unbiased for accept/reject but restarts
the stream. Reported as `exchange_rng: reseeded` rather than claimed as continuous.

**Not scientific validation.** Everything here is control flow and bookkeeping. No ladder is
validated, the 2 fs / 4 fs equivalence gate is still open, and the smokes are picoseconds.
