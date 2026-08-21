# Extending an implicit alanine REST2 run

An extension is **not** a new run and not a new project. It continues the existing run, in its own
directory, from its committed-generation record.

```bash
cd <project>/REST2_1 && ./REST2_1.sh
```

That is the whole command. Each invocation:

* reads the committed-generation record to find where the last segment stopped -- never a log, a
  file count or a directory listing;
* appends to the durable exchange history rather than truncating it;
* preserves monotonic step, time, frame and exchange-attempt indices;
* accumulates **lifetime** statistics while reporting this invocation's separately.

Run it `REST2_NUMBER_OF_SEGMENTS` times, or set that variable and use `run_all.sh`. The segment count
lives in Bash and in runtime state, never in the scientific JSON: putting it there would move the
configuration hash and make a longer run look like a different calculation.

## What must NOT be changed to extend

The protocol. Changing `duration_per_segment`, the exchange count, the tau ladder or the enhanced
region makes a different calculation, and continuing an existing run with it would produce a
trajectory whose halves came from two Hamiltonians. Continuation compares the continuity-defining
fields and refuses a mismatch.

Adding segments is the one change that is always safe, which is why it is not in the JSON at all.

## Verifying an extension did what it claims

```
n_chunks                     grows by one per invocation
total_ns_per_replica         lifetime, consistent with n_chunks
invocation_ns_per_replica    this invocation only
lifetime_exchange_attempts   accumulates
invocation_exchange_attempts does not
```

If `lifetime_exchange_attempts` did not grow, the run did not continue -- it restarted, and the
result is not what the manifest says it is.
