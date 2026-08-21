# Extending the implicit alanine cMD run

An extension continues the existing run in its own directory, from its committed-generation record.
It is not a new run and not a new project.

```bash
cd <project>/cMD_1 && ./cMD_1.sh
```

That is the whole command. Each invocation:

* reads the committed-generation record to find where the last segment stopped -- never a log, a
  file count, or a directory listing;
* restores from the binary **checkpoint** where possible, announcing a fall back to the portable
  State if the checkpoint is unusable;
* truncates any frames past the committed watermark, since those belong to an invocation that died
  before committing;
* **appends** to the trajectories and the state log, with no repeated header and no duplicated
  boundary frame;
* preserves monotonic absolute step, time and frame indices.

Do **not** re-run `run_all.sh` to extend: that would re-run minimisation and equilibration. Use it
only for a fresh run, where `CMD_NUMBER_OF_SEGMENTS` says how many production segments to take
after equilibrating once.

## What must not change to extend

The calculation. Particle and constraint counts, periodicity, ensemble, barostat pressure,
integrator kind, timestep, temperature, friction, restraint stiffness, steps per segment, reporting
intervals and the atom selection are all compared before anything is opened for append, and a
mismatch is refused while the previous segment's outputs are still exactly as it left them.

Adding segments is the one change that is always safe, which is why the segment count lives in Bash
and never in the scientific JSON: putting it there would move the configuration hash and make a
longer run look like a different calculation.

## Verifying an extension continued rather than restarted

```
generation             grows by one per invocation
absolute_step          grows by steps_per_segment
absolute_time_ps       grows by the segment duration
restart_source         "checkpoint", or "state" with the fallback announced
```

If `absolute_step` did not grow, the run restarted. The outputs would still look plausible, which
is exactly why this is worth checking rather than assuming.
