# AIS: the work measurement becomes a choice, and its default changes

**Behaviour change.** An AIS run that does not set `ais.work_measurement` now measures its work
DIRECTLY and records no three-group decomposition. Before this commit every AIS path took the
basis probe unconditionally. Configurations that relied on the components must now ask for them:

```yaml
ais:
  work_measurement: components
```

## Why

The decomposition and the work were never separable settings, so every run paid for both. Per
parameter update the old loop performed **five** potential-energy evaluations: three basis probes
at amplitudes (0, 0.5, 1), one direct `U(tau_k)`, and one direct `U(tau_k+1)`. Two of those five
existed only to cross-check the other three.

The two modes now separate what is measured from what is merely confirmed:

| | evaluations per update | produces |
|---|---|---|
| `work` (default) | **2** | the work integral for the schedule that ran |
| `components` | **3** | the work, plus `U` as a function of tau |

`components` fell from five to three because the fit already gives `U` at both endpoints of an
update — a direct measurement is not needed to OBTAIN the work, only to CHECK the fit. That check
is now scheduled by `ais.verify_every_updates` rather than paid for every update. Its default, 0,
verifies the first update of every path: the identity is a property of the system rather than of
the step, so one verified update establishes the model the rest of the path leans on, and a force
carrying tau-dependence outside the basis is still refused. `N > 0` re-verifies every N updates,
for a tau-dependence that only appears at a geometry reached later.

## Why `work` is the right default

A `components` run is a superset in what it records and a superset in what it costs, so the
question is which one a person who did not choose deserves. It is `work`: the work integral is
what an AIS run is FOR, and reweighting onto a tau the paths never visited is a further ambition
that a run should have to state. The reverse default charges every run for a capability most
never use.

## What this does not change

Nothing already measured. Component records without a `work_measurement` key are read as
`components`, because that was the only mode any earlier build had — so existing datasets still
assemble, and `AIS_work.csv` / `AIS_hs.csv` are rebuilt from them exactly as before.

## Reading the output

Component columns are **absent** from a `work` run's tables, not zero. A reweighting script meets
a missing column and fails, rather than reading a measurement nobody took as a measured nought.
`AIS_hs.csv` is still written in `work` mode — every row is a saved coordinate with the work that
reached it and `potential_direct_kj_mol` AT it, which is what a Hummer-Szabo estimate at the
schedule that ran needs. What it lacks is the basis, so an estimate at an unvisited tau cannot be
formed from it.

The mode is recorded in three places so it never has to be inferred from which columns happen to
be present: `completed.json`, the `work_measurement` column of `AIS_paths.csv` (present in BOTH
modes), and the run-identity document — which means `require_same_run` refuses an `-odir` whose
finished paths were measured the other way before any new path runs, and a resume under the other
mode is refused by name.

## Evidence

`tests/test_ais_recovery_integration.py`, final section:

- both modes measure the same work on the same path, total and every row, to 1e-6 kJ/mol;
- `work` costs 2N evaluations and takes no probe; `components` costs 3N + 2;
- a `work` path carries no component column anywhere, and says `work_measurement: work`;
- a path cannot be resumed under the other mode;
- `verify_every_updates` fires where documented and costs exactly two evaluations each time.

`tests/test_defaults_and_ais_invariants.py` pins the runtime's default literals to the schema's,
since `ais/run.py` deliberately does not import the configuration builder.
