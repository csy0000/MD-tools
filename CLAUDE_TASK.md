# Bounded stabilization: make MD-tools ready for a research project

## Goal and authority

Work on csy0000/MD-tools, dev. Latest reviewed implementation:
ea4dffdddf8df17bf496a4f877a7570c4037a823.

The user has agreed to a narrower definition of readiness:
a new research project can install a pinned MD-tools version, build its systems,
run cMD/REST2/rREST2/AIS, resume interrupted calculations, and obtain interpretable
scientific outputs without modifying the engine.

MD-tools owns reusable simulation machinery. MD-project owns scientific aims,
systems, budgets, orchestration and project-specific methods. Analysis/model
training belongs in the project or analysis package. Do not add future methods,
a new abstraction framework, or a cross-repository redesign in this task.

This instruction supersedes the exhaustive completion gates in instruction
27a3715 and the requirement to close every metadata issue before proceeding.
Read CLAUDE.md, but apply the proportional acceptance rules below where older
validation requirements conflict. Preserve unrelated work and inspect the actual
head before starting. No reset to the reviewed SHA.

## 1. One bounded correctness pass

Investigate the remaining resume risks identified in the last review:

- Ladder checkpoint progress, extra.cv_rows, prefix rows and per-state rows are
  not fully reconciled. A coherent-looking shorter prefix must not cause valid
  committed samples to be discarded while dynamics resumes at a later step.
- Some early continuation checks skip unreadable checkpoints, missing CV prefixes
  or missing CV files. Public AIS preflight inspects completed paths but can omit
  partial paths. Determine whether a later authoritative check refuses safely
  before any scientific output is modified.

Reproduce with a small fixture and the real resume path. Fix demonstrated paths
that can truncate committed scientific data, load inconsistent progress, mix
states/paths, or silently continue with incomplete required scientific outputs.
If existing downstream checks already protect the scientific data, document that
fact and do not add another validation layer solely to preserve logs.

Use the existing checkpoint and protocol authorities. Compare committed progress
with the configured observation grid, respecting independent checkpoint/CV/frame
cadences and extension offsets. Keep legitimate uncommitted tails recoverable:
validate the committed prefix, then truncate only work beyond that prefix.

Tests must include a valid interrupted-resume control and a damaged later
state/path. Expected retained rows must come from checkpoint progress and the
schedule, not from the same row counter being tested.

## 2. Proportional preservation rules

On refused continuation, protect trajectories, CV/work/HS tables, committed
checkpoint generations and pointers, completion manifests, and authoritative
configuration/provenance needed to interpret or resume prior results.

A rejected attempt may write a separate diagnostic log or stderr. Do not overwrite
the prior run's authoritative completion state with a failed attempt's status.
There is no release requirement to preserve inode numbers, timestamps, or every
diagnostic byte. There is no requirement for an exhaustive mutation matrix over
every metadata field.

Keep existing useful checks. Do not dismantle working validation or deliberately
make the runtime less safe as a simplification exercise. Adjust tests only where
they enforce superseded diagnostic/inode rules, documenting that policy change.

## 3. Representative scientific validation

Reuse existing tests and small systems. For each supported protocol, demonstrate:

| Protocol | Required result |
|---|---|
| cMD | valid system build, production output, interrupted resume without lost/duplicated committed samples |
| REST2 | state trajectories/CVs and exchange history are consistent; interrupted resume preserves state assignment and progress |
| rREST2 | reservoir refresh actually executes, existing velocity policies and pre-refresh CV semantics remain correct; resume works |
| AIS | correct source/path identity, total/reduced work, three-group decomposition/reconstruction and HS frame alignment; partial resume and completed no-op work |

Use existing deterministic comparisons and scientifically justified tolerances.
Do not weaken them to obtain a pass. Retain the improved independent two-to-four-rank
AIS test: expected new observations are calculated before resume from committed
progress and the CV schedule. Runtime invocation IDs must not influence the science.

Run affected CUDA/MPI tests on real CUDA/MPI, with subprocess timeouts and complete
diagnostics retained. Pure parser and filesystem tests may run on CPU and must be
labelled accordingly. Do not present CPU success as GPU validation.

## 4. Finite validation and retry policy

1. Reproduce the concrete resume risks, then fix them and rerun the focused tests.
2. Run the representative protocol tests and affected CUDA/MPI lanes.
3. Once focused tests pass, commit the implementation and run ONE final full suite,
   split into the existing complementary fast and slow/GPU selections:
   python -m pytest tests -m "not slow and not gpu"
   python -m pytest tests -m "slow or gpu"
4. Build/install the wheel outside the checkout, verify its import origin, and run
   one short installed-wheel build -> run -> interrupted-resume workflow on CUDA
   through the supported public interface. Reuse an existing fixture/configuration.
5. Require green hosted CI on the implementation and final branch head. Hosted CI
   is not GPU evidence.

If a test fails, preserve the full command/output and diagnose it. Fix demonstrated
code defects and rerun affected tests. If the fix changes scientific execution or
shared resume logic after the full suite, rerun that suite; documentation-only
changes do not require another GPU sweep. Do not repeat passing tests without a
concrete remaining risk.

All required scientific checks must actually run. Report optional existing skips
by name/reason. Do not add skips, xfails or looser tolerances to obtain success.

The historical one-off AIS MPI failure has no retained diagnostic and passed
thirteen subsequent full-file runs. Record it as unresolved historical evidence,
with full capture/timeouts enabled for future runs. It is not, by itself, a
release blocker and does not require testing until it recurs. If it recurs during
this pass, retain the diagnostic and investigate; a reproducible scientific,
restart or MPI failure remains blocking.

## 5. Deferred work and handoff

Track these in a short repository backlog, with impact and a concrete trigger for
revisiting them; they are not release gates unless they affect scientific output
or recovery:

- Aggregate CV-cost metadata not cross-checked against all verified prefix costs.
- Missing completion-cost metadata and exhaustive malformed metadata cases.
- Broader provenance/schema hardening and diagnostic-file transactional behavior.
- Historical unexplained MPI failure, pending a diagnostic-bearing recurrence.

Do not claim these are fixed. Do not use inaccurate cost metadata for efficiency
comparisons; if a known reporting defect can mislead a user, document its scope.

Write a concise readiness note with:
- tested implementation SHA, environment, exact commands and results;
- any corrected data-loss/progress defect and its regression test;
- wheel import origin and the installed workflow result;
- deferred issues and limitations;
- one small copyable workflow using current supported commands/configs;
- the tested commit to pin in MD-project and the next integration experiment.

The example must be runnable outside the checkout without source edits or
machine-specific committed paths. Clarify that small regression fixtures validate
software behavior, not ensemble convergence or scientific sampling quality.

Stop when the protected scientific workflows pass, the installed example works,
and no known reproducible scientific/data-loss/MPI defect remains. Do not broaden
this task into a new audit. Keep a completed status or remove this task in a
documentation-only commit; neither choice requires another GPU run. Push the
implementation and readiness note and provide the pin and deferred backlog.
