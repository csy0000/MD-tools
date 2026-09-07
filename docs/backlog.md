# Backlog

Known gaps that are **not fixed**. None of them is a release gate: none has been shown to
corrupt, discard or misreport scientific output, or to prevent recovery of an interrupted run.
Each entry says what it is, what it costs today, and the concrete thing that should cause it to
be picked up.

Nothing here should be read as "handled". If one of these turns out to affect a result, it stops
being backlog and becomes a defect.

---

## 1. The aggregate CV-cost record is not cross-checked against the prefix costs it sums

**What.** A campaign's aggregate cost record (`aggregation`, `per_state` / `per_path`) is parsed
strictly — every scope, every count, every `wall_seconds` — and each per-state or per-path entry
is checked against the prefix it belongs to. What is *not* checked is that the aggregate's own
totals equal the sum of those entries. A record whose parts are individually valid and whose
total is wrong is accepted.

**Impact.** Reporting only. The aggregate is never read back to decide a truncation, a resume
point or any scientific value; it is metadata describing how much CV evaluation a campaign cost.
A wrong total would mislead someone comparing the cost of two campaigns.

**Do not** use aggregate CV-cost metadata for efficiency comparisons between runs until this is
closed. The per-state and per-path entries are individually verified and are the trustworthy
numbers.

**Trigger.** Pick this up when CV cost is used to make a decision — scheduling, budgeting, or a
published efficiency claim — or when a campaign's reported total is questioned.

---

## 2. Missing completion-cost metadata, and exhaustive malformed-metadata cases

**What.** Two related gaps. A completion manifest that carries no cost record at all is not
refused the way a *committed prefix* without one is (that case is closed, and refuses). And the
malformed-metadata coverage is representative rather than exhaustive: it covers the field shapes
that decide a truncation or a restoration, not every field of every record under every mutation.

**Impact.** A completed campaign whose manifest lacks cost metadata reports no cost rather than a
wrong one. The unexercised mutations are in fields that no continuation reads.

**Trigger.** Pick this up when a manifest is found in the wild without its cost record, or when
manifest metadata starts being consumed by something other than a human reader — a dashboard, an
analysis package, an automated pin.

---

## 3. Broader provenance/schema hardening and diagnostic-file transactional behaviour

**What.** The read-only continuation boundary protects trajectories, CV/work/HS tables, committed
checkpoint generations and pointers, completion manifests and authoritative configuration. It
does not make *diagnostic* files transactional: a refused attempt may append to a `.out` or a
`.log`, and a crash mid-write can leave a partial diagnostic line. Schema-version handling is
strict where a record decides scientific behaviour and permissive elsewhere.

**Impact.** Diagnostics may be untidy after a refusal or a crash. This is deliberate under the
current preservation rules — a rejected attempt is allowed to say why it refused — and no
authoritative record is involved.

**Trigger.** Pick this up if a diagnostic file is ever made authoritative for anything, or if
partial diagnostic lines start breaking a downstream parser.

---

## 4. One unexplained AIS MPI failure, with no retained diagnostic

**What.** A single run of `tests/test_cv_mpi_cuda_ais.py` reported `1 failed, 10 passed`. The
diagnostic was lost to an output filter before it could be read, and the file has passed every
run since — thirteen consecutive clean full-file runs at the time it was recorded, plus the runs
in this pass. The cause is unknown. It is recorded as unresolved historical evidence, not as a
diagnosed and fixed defect and not as an accepted limitation.

**Impact.** Unknown, which is the problem. It has never recurred and has never been reproduced,
so there is nothing to characterise.

**Trigger.** Pick this up the moment it recurs. Every launcher call in that lane now carries a
subprocess timeout and retains complete stdout and stderr, so a recurrence will arrive with the
diagnostic this one lacked. A reproducible scientific, restart or MPI failure is a release
blocker; this one is not, because it is not reproducible.

---

## Cross-references

- `docs/release-notes/20260907-cv-validation-final-evidence.md` — the evidence behind the
  validation work these gaps were carved out of, including its own "Open item" section for
  entry 4.
- `docs/release-notes/20260907-readiness.md` — the readiness note for the pinned commit.
