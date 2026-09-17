# §8 evidence specification: AIS schedules and frame selection

Companion to `20260917_next-release-reusable-ligands-propka-cuda-ais.md`, rows **AIS schedules** and
**Frame selection**. Each claim names the test that measures it, the command, the inputs, the
tolerance and what goes in the evidence table. A claim without a test is listed as a GAP, never as
passed.

Record for every row: commit SHA, environment (`python -c "import openmm, md_tools;
print(openmm.__version__, md_tools.__file__)"` and `nvidia-smi -L` for CUDA rows), the exact
command, the pytest summary line, and where the log is kept. CPU results are never CUDA evidence.

Environment for every command below: the md-stack environment activated (`openmm-env`), run from
the repository root of the commit under test. A bare venv gives fixture errors (tleap missing).

## Row: AIS schedules

| # | claim | test (node id) | inputs | tolerance | device |
|---|---|---|---|---|---|
| S1 | `linear` is the default and its λ values are unchanged | `tests/test_ais_schedules.py::test_linear_is_the_default_and_its_lambdas_are_unchanged` | 200 updates | exact list equality | CPU |
| S2 | `tau-linear` table: endpoints exactly 0 and 1, strictly monotone, mixture solute–solute prefactor `(1−λ)(1−τ₀)²+λ = (1−τ)²` at every update | `tests/test_ais_schedules.py::test_tau_linear_follows_the_solute_solute_scaling_of_a_linear_tau` | τ₀ ∈ {0.2, 0.5, 0.8}, 200 updates | abs 1e-12 | CPU |
| S3 | the formula's documented value λ(½) = 5/12 at τ₀ = ½ | `...::test_tau_linear_at_half_is_the_documented_value` | — | abs 1e-15 | CPU |
| S4 | the λ-table digest changes exactly when the λ values do | `...::test_the_schedule_digest_changes_with_the_lambdas_and_only_then` | linear, τ₀ 0.4, τ₀ 0.5 | exact | CPU |
| S5 | invalid schedules refused (linear with τ₀, tau-linear without τ₀, τ₀ = 0, unknown kind) | `...::test_a_schedule_that_cannot_define_a_path_is_refused` | 4 cases | refusal message match | CPU |
| S6 | PHYSICS: in vacuum (all atoms solute) the tau-linear mixture equals the REST2 System at τ = τ₀(1−t) | `...::test_in_vacuum_the_tau_linear_mixture_is_the_rest2_system_at_the_linear_tau` | ALA vacuum, amber14, τ₀ 0.5, updates 0/37/100/163/200, Reference | abs 1e-6 kJ/mol | CPU Reference |
| S7 | frozen-coordinate work equals the finite difference for arbitrary, non-uniform λ steps (so for any schedule) | `tests/test_ais_two_state.py::test_the_work_from_the_derivative_is_the_finite_difference_at_frozen_coordinates` and `::test_work_telescopes_over_a_schedule_at_one_coordinate` | ALA solvated/vacuum; λ pairs (0, .01), (.3, .37), (.99, 1) | rel 1e-9 / abs 1e-6 kJ/mol | CPU Reference |
| S8 | τ₀ resolution: generated source fills τ₀ from `dynamics.tau` and refuses a different one; a named source must state it; build-md writes τ₀ into `resolved.config` and `AIS.in`, which round-trip | `...::test_a_generated_source_gives_tau_linear_its_tau0_and_a_different_one_is_refused`, `...::test_a_named_source_must_state_tau0`, `...::test_build_md_writes_tau0_into_the_resolved_configuration_and_the_input` | implicit dataset root | exact | CPU |
| S9 | preflight claim check: accepts saved state → its source; refuses a non-saved V0, a wrong τ₀, a V1 that is not the recorded source; linear reads no record | `...::test_tau_linear_accepts_a_saved_state_switched_to_its_own_source`, `...refuses_a_v0_that_is_not_a_saved_state`, `...refuses_a_tau0_the_saved_state_does_not_have`, `...refuses_a_v1_that_is_not_the_scaled_source`, `...::test_linear_never_reads_the_state_record` | implicit root, saved AIS state τ 0.5 | refusal before output | CPU |
| S10 | a v2 (0.5.4) run directory is refused on continuation, with the reason | `...::test_a_v2_run_directory_is_refused_with_the_reason` | synthetic v2 identity | message match | CPU |

Command (S1–S6, S8–S10):

```bash
python -m pytest tests/test_ais_schedules.py -q -p no:cacheprovider
```

Command (S7):

```bash
python -m pytest "tests/test_ais_two_state.py::test_the_work_from_the_derivative_is_the_finite_difference_at_frozen_coordinates" "tests/test_ais_two_state.py::test_work_telescopes_over_a_schedule_at_one_coordinate" -q -p no:cacheprovider
```

Observed at 567a4df (CPU, CUDA hidden): `tests/test_ais_schedules.py` 26 passed; fast lane 1899
passed (MD-tools-main's own run).

### GAPS in this row (no test yet; not passed)

- **G1: run level, CPU.** One AIS path under tau-linear through `run_one_path`: the
  `lambda_before`/`lambda_after` columns of `AIS_work.csv` equal the λ table at the observed
  updates, `AIS_run.json` records the schedule, τ₀ and `lambda_sha256`, and every saved-frame row
  satisfies `potential_direct = (1−λ)·V0 + λ·V1`. Proposed tolerance: λ exact; potentials abs 1e-6
  kJ/mol with double-precision observation. Proposed home: `tests/test_ais_schedules.py`, marked slow.
- **G2: continuation, CPU.** A v3 directory written under `linear` refused when re-invoked under
  `tau-linear` (and vice versa, and under a different τ₀), naming `schedule`/`lambda` as the
  differing field. The generic differing-field refusal is tested
  (`tests/test_ais_directory_identity.py::test_an_incompatible_directory_is_refused_by_naming_what_differs_even_with_resume`);
  the schedule-specific case is not.
- **G3: CUDA.** A tau-linear AIS path on CUDA with the identity check at saved frames, entered in
  `docs/release-notes/cuda-coverage-matrix.md`. Needs the user's GPU go-ahead. The matrix itself is
  stale (it still names the retired `TauSwitcher.set_amplitude` and the decomposition probe) and
  must be regenerated before it can hold this row.

## Row: Frame selection

| # | claim | test (node id) | inputs | tolerance | device |
|---|---|---|---|---|---|
| F1 | `evenly_spaced` covers the whole window with distinct, sorted frames; gaps ≤ 2 for 64 of 95 | `tests/test_ais_schedules.py::test_evenly_spaced_covers_the_whole_window_with_distinct_frames` | eligible 5..99, 64 paths | exact: first 5, last 99, 64 distinct | CPU |
| F2 | exact at the edges: 1, 2, 7 and 95 (= all) paths | `...::test_evenly_spaced_is_exact_at_the_edges` | eligible 5..99 | exact | CPU |
| F3 | more paths than frames refused; with repeats allowed, each frame used equally | `...::test_more_paths_than_frames_is_still_refused_unless_repeats_are_allowed` | 3 frames, 5/6 paths | exact | CPU |
| F4 | an old-selection (0.5.4, identity v2) directory refused on continuation | S10 | — | message match | CPU |

Command: included in the `tests/test_ais_schedules.py` run above.

### GAP in this row

- **G4: run level, CPU.** `selected_source_frames.csv` of a generated-source run (for example the
  existing `tests/test_ais_generated_source.py` chain with more paths than today's 3) spans the
  eligible window from its first to its last frame. Proposed as an extra assertion in that slow
  test rather than a new run.

## Reproduction against the 0.5.4 tutorial (optional, documentation evidence)

The 0.5.4 alanine tutorial selected frames 5..68 of 95 eligible. The same `choose_frames` call at
the evidence commit must give 64 distinct frames from 5 to 99:

```bash
python -c "from md_tools.ais.run import choose_frames as c; f=c(eligible=list(range(5,100)), count=64, selection='evenly_spaced', allow_repeats=False, seed=0); print(f[0], f[-1], len(set(f)))"
# expected: 5 99 64
```
