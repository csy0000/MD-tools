# Readiness note — MD-tools for a pinned research project

**Tested implementation:** `443fc736f3a8f426414e244a50a1a438d2d8a497` (branch `dev`)
**Date:** 2026-09-07

This note says what was run, what it produced, and what is still open. The claim it supports is
narrow and deliberate: **a new project can install a pinned MD-tools, build its systems, run
cMD / REST2 / rREST2 / AIS, resume interrupted calculations, and read the outputs, without
modifying the engine.** It is not a claim about sampling quality — see *What this does not
establish*, below.

---

## 1. Environment

| | |
|---|---|
| Python | 3.12.14 |
| OpenMM | 8.6 |
| mpi4py | 4.1.2 |
| MPI | Open MPI 5.0.8 (`mpiexec`) |
| CUDA driver | 580.173.02 |
| Devices | 1 × RTX A5000, 8 × RTX 3080 (9 total) |
| Interpreter | `/path/to/software/md-stack/envs/openmm-rest2/bin/python` |

**One invocation note that cost a run.** The MPI fail-closed lane launches the `md-openmm`
console script through `mpiexec`, so the environment's `bin` must be on `PATH`. Invoking pytest
by absolute interpreter path without it produced two failures reading
`prterun was unable to find the specified executable md-openmm` — an artefact of how the suite
was invoked, not a defect. With `PATH` set, `tests/test_mpi_fail_closed.py` is 25/25. Every
command below sets it.

---

## 2. The defect that was corrected

**A ladder's committed collective-variable prefix was never reconciled with the run's own
progress**, and the result was silent loss of committed scientific samples.

A ladder checkpoint states how much CV output is durable three separate times: the absolute
`step` the dynamics reached, `extra.cv_rows`, and a per-state `cv_prefix` carrying each state's
row count, digest and cost. The validation compared the last two with each other — block count
against every per-state count, each cost against its own rows, each digest against the prefix it
describes — and never against the first.

Every one of those checks is satisfied by a prefix that is simply **too short**. Shorten it,
recompute the digests and costs to match, and the record is internally perfect.

**Reproduced on the real resume path** (`tests/test_ladder_progress_reconciliation.py`), on a
2-state REST2 ladder observing every 5 steps to step 60. A prefix shortened by two rows was
accepted; `truncate_to` cut every state's series to the shorter length; dynamics resumed from
step 10, the step the checkpoint recorded; and the observations at steps 5 and 10 were gone. The
run then recomputed the entire remaining segment on top of that gap and failed only at the
completion check at the very end:

```
DriverError: this ladder's collective-variable output is not a completed set:
  - state 0: the step grid is wrong (missing [5, 10]). It must be 0, 5, ..., 60 exactly once
  - state 1: the step grid is wrong (missing [5, 10]). It must be 0, 5, ..., 60 exactly once
```

That end-of-run check caught it here **only because the run reached its end**. An interruption
before then leaves a gapped set that no later reader can distinguish from a good one.

### The fix

The expectation now comes from somewhere the record cannot influence: the checkpoint's committed
step and the configured observation grid. A series whose first committed row is at step `first`
and whose run reached `step` holds `(step - first) / interval + 1` rows.

- `first` is read from the series rather than assumed to be zero, which is what makes it correct
  for an **out-of-place extension**, whose steps are absolute and continue from the parent's
  terminal value.
- Only the **CV cadence** is used. Checkpoint, CV and frame cadences are independent and stay so.
- The row at the committed step is itself committed, because `cv` precedes `checkpoint` in the
  ladder's event order — hence the inclusive count.
- `extra.cv_rows` is now compared with the prefix block as well: two statements of one number,
  each with its own reader, previously never compared with each other.

### Its regression tests

`tests/test_ladder_progress_reconciliation.py` — 3 tests:

| test | what it establishes |
|---|---|
| `test_a_valid_interrupted_resume_retains_every_committed_row` | **the control.** A genuine interrupted resume reproduces the uninterrupted reference row-for-row and value-for-value. Without it, a refuse-everything implementation would satisfy the file. |
| `test_a_prefix_shorter_than_committed_progress_is_refused` | the data-loss case. Refused, and every committed row still on disk afterwards. |
| `test_a_prefix_longer_than_committed_progress_is_refused` | the other direction: a prefix vouching for rows the run never reached. |

Expected retained rows are derived from `checkpoint["step"]` and the CV interval — never from
`cv_rows`, the counter under test. The shortened prefix is rewritten with **consistent** digests,
per-state counts and costs, so nothing internal disagrees; only the checkpoint's progress does.

---

## 3. The skipped early checks: what protects the data

The read-only preflight (`md_tools.run.continuation`) declines to speak in four places. The
question was whether a later authoritative check refuses before any scientific output is
modified. **For the scientific data, it does** — demonstrated rather than assumed, in
`tests/test_skipped_continuation_checks.py`, and no second validation layer was added for it.

| skipped in preflight | what actually protects the data |
|---|---|
| unreadable ladder checkpoint (`except StorageError: return`) | `_continue_cv_states` reads the same checkpoint inside the driver; a checkpoint that will not read never reaches the truncation, because the run cannot restore its configurations from it either. |
| AIS path with an unreadable committed checkpoint (`except CheckpointError: continue`) | `read_committed` in `run_one_path` raises before the first `_truncate_csv`. |
| AIS path whose committed state has no `cv_prefix` (`if entry is None: continue`) | the runtime passes `state.get("cv_prefix") or {}` to `_validate_cv_prefix`, which refuses an entry with no row count and no digest — before the first `_truncate_csv`. |
| AIS partial paths omitted from the public preflight's selection | **this one was a real gap, and it is closed** — see below. |

Each case above is exercised with a damaged **later** path, against a control
(`test_the_interrupted_campaign_resumes_cleanly`) proving the undamaged campaign resumes. The
assertion is that every scientific file of every path — `cv.csv`, `cv.json`, `observations.csv`,
`state.csv`, `completed.json`, staged trajectories — is byte-for-byte unchanged after the refused
attempt, including paths the attempt reached before the damaged one.

### The gap that was real

`validate_public_entry` built its AIS selection from `completed.json` markers alone, so a
campaign whose paths are **all partial** — interrupted, with committed generations and no markers,
precisely the state a resume exists for — was not inspected at all. `md-run` creates `-odir` and
writes `resolved.config` and the content-addressed CV definition copy **itself**, before
dispatching. So a refused resume replaced the prior run's authoritative configuration record on
its way to a refusal the runtime was always going to make.

Verified by removing the fix and re-running the test, which then reports:

```
AssertionError: a refused md-run wrote into the tree: ['cv.c34538cb271e.yaml', 'resolved.config']
```

A partial path records its source frame in its committed checkpoint exactly as a completed one
does in its manifest, so including partial paths costs one extra read and needs no new rule.

---

## 4. What was run, and what it produced

Every command below was run from the MD-tools checkout at `443fc73` with
`PATH=<env>/bin:$PATH`. GPU lanes used real CUDA devices; MPI lanes used a real `mpiexec`
launcher. Every subprocess call in these files carries a timeout, and complete stdout/stderr is
retained on failure.

### Focused tests for the corrected defect

```
python -m pytest tests/test_ladder_progress_reconciliation.py tests/test_skipped_continuation_checks.py -q
→ 10 passed
```

### Representative protocol validation

| Protocol | Required result | On real CUDA | Labelled CPU-exempt |
|---|---|---|---|
| **cMD** | valid build, production output, interrupted resume with no lost or duplicated committed samples | `test_cmd_cuda_smoke.py` (split and all-in-one layouts, every stage to completion, production continuing from the state equilibration left); `test_cv_cuda_lanes.py::test_cmd_cv_runs_on_cuda_and_writes_the_declared_grid`, `…_resume_on_cuda_reproduces_the_grid_and_cost`, `…_survives_two_interruptions_on_cuda`; fixed-τ cMD in `test_fixed_tau_phase_space_and_cv_resume_on_cuda` | — |
| **REST2** | state trajectories/CVs and exchange history consistent; resume preserves state assignment and progress | `test_cv_cuda_lanes.py::test_rest2_cv_runs_on_cuda_fresh_and_resumed`, `…_survives_two_interruptions_on_cuda`; `test_cv_mpi_cuda_lanes.py` under a real launcher | — |
| **rREST2** | reservoir refresh actually executes; velocity policy and pre-refresh CV semantics correct; resume works | `test_rrest2_cuda_smoke.py` (a refresh installs the *recorded* momentum, and the run records where its samples came from); `test_cv_cuda_lanes.py::test_rrest2_cv_on_cuda_holds_the_pre_refresh_configuration`, `…_survives_two_interruptions_on_cuda`; `test_cv_mpi_cuda_rrest2.py` across ranks | — |
| **AIS** | source/path identity, total and reduced work, three-group decomposition and reconstruction, HS frame alignment; partial resume and completed no-op | source/path identity, work, decomposition, partial resume and completed no-op: `test_cv_cuda_lanes.py::test_ais_cv_and_decomposition_on_cuda_fresh_and_resumed`, `…_survives_two_interruptions_on_cuda`; `test_cv_mpi_cuda_ais.py` (2 ranks, real launcher); 100 paths across ranks in `test_md_run_mpi_gpu.py`. **HS frame alignment** is recomputed on real CUDA for implicit and explicit PME alike in `test_cuda_coverage_matrix.py::test_hs_rows_match_recomputation_on_cuda` | `test_ais_hs_frame_alignment.py` (which frame a row's number was measured at) and `test_ais_decomposition_resume.py` (accumulator arithmetic and row alignment across a resume) carry `PLATFORM_POLICY_EXEMPTION` lines. Both assert bookkeeping that is identical on every platform; the energies and the HS recomputation they depend on are covered by the CUDA lanes named to the left. **Neither is offered as GPU evidence.** |


```
python -m pytest tests/test_cmd_cuda_smoke.py tests/test_cv_cuda_lanes.py tests/test_rrest2_cuda_smoke.py -q
→ 19 passed in 151.90s

python -m pytest tests/test_cv_mpi_cuda_lanes.py tests/test_cv_mpi_cuda_rrest2.py tests/test_cv_mpi_cuda_ais.py -q
→ 44 passed in 434.87s
```

The independent multi-rank AIS test is retained: expected new observations are computed **before**
the resume from committed progress and the CV schedule by an oracle that does not call the code
under test, and per-rank invocation IDs are checked not to influence any scientific value.

**CPU is not presented as GPU validation.** Two files added in this pass run under `--cpu` and say
so in a `PLATFORM_POLICY_EXEMPTION` line at the top: both test file arithmetic around a refusal —
which rows survive, which files are read and written, in what order — and neither exercises the
integrator. The same resume paths run on real CUDA in the lanes above.

### One final full suite

```
python -m pytest tests -m "not slow and not gpu" -q
→ 1387 passed, 438 deselected in 188.31s

python -m pytest tests -m "slow or gpu" -q
→ 437 passed, 1 skipped, 1387 deselected in 2240.80s (0:37:20)
```

**The one skip, by name and reason.** `tests/test_cuda_coverage_matrix.py:1528` —
`no --cuda-evidence=<path> given; the lanes above ran, nothing to write`. It is the function that
*writes* `docs/release-notes/cuda-coverage-matrix.md`, and it writes only when the flag is given
so that an ordinary GPU run does not rewrite a committed file as a side effect. **Every CUDA lane
in that file ran and passed**; what was skipped is the document generator, not a check. No skip,
xfail or loosened tolerance was added anywhere in this pass.

### Hosted CI

Green on the implementation commit `443fc73`
([run 34138255909](https://github.com/csy0000/MD-tools/actions/runs/34138255909), success, 2m13s).
Hosted CI has no GPU and is not GPU evidence.

---

## 5. The installed wheel

Built from the checkout, installed into a virtual environment **outside** it, and exercised
through the public interface only.

```
python -m build --wheel --outdir <out>
→ md_tools-0.5.0.dev0-py3-none-any.whl

python -m venv --system-site-packages <venv>
<venv>/bin/python -m pip install --no-deps --force-reinstall md_tools-0.5.0.dev0-py3-none-any.whl
```

**Import origin**, checked from a working directory outside the checkout:

```
md_tools     : <venv>/lib/python3.12/site-packages/md_tools/__init__.py
cv_states    : <venv>/lib/python3.12/site-packages/md_tools/remd/cv_states.py
```

Neither resolves into `MD-tools/src`, and the installed `cv_states.py` carries the reconciliation
added in this pass. `md-openmm --help` from the installed console script lists exactly the four
subcommands: `build-top`, `build-md`, `md-run`, `data-register`.

**Installed-wheel workflow on CUDA** — build → run → interrupted resume, all through
`md-openmm`, nothing reading the checkout:

- 400 000-step implicit-solvent cMD on device 0, CV reporting every 250 steps
- interrupted with SIGINT mid-production; the run had committed **356 500 steps**
- the same command rerun; `cMD.out` reports `already done 356500 (resumed from checkpoint)`
- final series: **1601 rows on the grid 0, 250, …, 400000**, no duplicated and no missing
  observation across the interruption

---

## 6. A small copyable workflow

Runnable outside the checkout. Substitute your own structure for `ALA.pdb`; nothing else is
machine-specific and no source edit is needed.

```bash
pip install md_tools-0.5.0.dev0-py3-none-any.whl      # the pinned wheel
export MD_TOOLS_CONFIG=$PWD/user.config

cat > user.config <<'YAML'
schema_version: "1.0"
user: {person_id: demo, name: Demo}
YAML

cat > sys.config <<'YAML'
solvent: {model: GBn2}
YAML

cat > cv.yaml <<'YAML'
schema_version: 1
collective_variables:
  - {name: phi, type: torsion, atom_indices: [4, 6, 8, 14]}
  - {name: psi, type: torsion, atom_indices: [6, 8, 14, 16]}
YAML

cat > cMD.config <<'YAML'
protocol: cMD
solvent: implicit
stages:
  minimization_iterations: 50
  restrained_nvt_steps: 0
  restrained_npt_steps: 0
  unrestrained_npt_steps: 0
  production_steps: 400000
reporting: {crd_printout_solute: 500, info_printout: 500, checkpoint_printout: 500}
collective_variables: {file: cv.yaml, interval_steps: 250}
dynamics: {seed: 20260907}
YAML

# 1. build the system
md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log --config sys.config

# 2. generate the run scripts and Amber-style inputs
md-openmm build-md -odir ./cMD --config cMD.config --all-in-one

# 3. run it on a GPU
md-openmm md-run -i cMD/cMD.in -p built.pdb -s built.xml -odir ./out --device 0

# 4. if it was interrupted, rerun the SAME command. It continues from its
#    committed generation; there is no separate resume flag for a stage.
md-openmm md-run -i cMD/cMD.in -p built.pdb -s built.xml -odir ./out --device 0
```

Outputs land in `./out`: `cMD.dcd` (trajectory), `cMD.cv.csv` + `cMD.cv.json` (collective
variables and their sidecar), `cMD.csv` (energies), `cMD.out` (human-readable), `cMD.log`
(provenance), `resolved.config` (the authoritative resolved configuration), and
`cMD.checkpoints/` (committed generations).

For a REST2 ladder or AIS, the shape is the same under a launcher:

```bash
mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \
       -c eq_npt_free.xml -odir ./REST2

mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \
       -source-traj ../cMD_tau0p5/tau_0p5.dcd -odir ./AIS
```

---

## 7. What this does **not** establish

The regression fixtures above are small on purpose: a few hundred to a few hundred thousand steps
of alanine dipeptide in implicit solvent, seeded, on grids chosen so the expected row counts can
be derived by hand. They validate **software behaviour** — that committed samples survive an
interruption, that a resumed series has no gap and no duplicate, that a damaged record is refused
before anything is rewritten, that identity and cost accounting survive redistribution across
ranks.

They establish **nothing** about ensemble convergence, sampling quality, free-energy accuracy, or
whether any particular schedule is long enough for a scientific question. Those are properties of
a study, not of an engine, and they belong to MD-project.

---

## 8. Deferred issues and limitations

Tracked in [`docs/backlog.md`](../backlog.md), with impact and a concrete trigger for each. In
short:

1. **Aggregate CV-cost metadata is not cross-checked against the prefix costs it sums.**
   Reporting only; never read to decide a truncation or a resume point. **Do not use aggregate CV
   cost for efficiency comparisons between runs.** The per-state and per-path entries are
   individually verified and are the trustworthy numbers.
2. **Missing completion-cost metadata, and non-exhaustive malformed-metadata cases.** A manifest
   without a cost record reports no cost rather than a wrong one. Coverage is representative, not
   exhaustive, in fields no continuation reads.
3. **Broader provenance/schema hardening and diagnostic-file transactional behaviour.** A refused
   attempt may append to a `.out` or `.log`; diagnostics are not transactional. This is deliberate
   under the current preservation rules — a rejected attempt is allowed to say why it refused —
   and no authoritative record is involved.
4. **One unexplained AIS MPI failure with no retained diagnostic.** A single
   `tests/test_cv_mpi_cuda_ais.py` run reported `1 failed, 10 passed`; the diagnostic was lost to
   an output filter. Thirteen clean full-file runs followed, plus this pass's runs. **It did not
   recur here.** The cause is unknown — recorded as unresolved historical evidence, not as fixed
   and not as an accepted limitation. Every launcher call in that lane now carries a timeout and
   retains complete output, so a recurrence will arrive with the diagnostic this one lacked.

None of these has been shown to corrupt, discard or misreport scientific output, or to prevent
recovery of an interrupted run. None is claimed to be fixed.

---

## 9. Handoff

**Pin this commit in MD-project:** `443fc736f3a8f426414e244a50a1a438d2d8a497` (`dev`).

**Next integration experiment.** Run one REST2 ladder end to end from MD-project's own
orchestration against this pin — a real system, a real budget, a real GPU allocation — and
deliberately interrupt and resume it once mid-campaign. That exercises the two things this pass
changed (progress reconciliation on resume, and the read-only refusal boundary at the public
entry) under a workload neither a regression fixture nor a CI lane can reproduce: many states,
long series, real wall-clock checkpoint cadence, and a scheduler that can kill a job at an
arbitrary point rather than at an instrumented boundary. Compare the resumed series against the
committed progress its checkpoint records; a gap or a duplicate is a defect and should come back
here with the checkpoint and the series attached.

## Cross-references

- [`docs/backlog.md`](../backlog.md) — the deferred items above, with triggers.
- [`20260907-cv-validation-final-evidence.md`](20260907-cv-validation-final-evidence.md) — the
  preceding CV validation evidence, including its own open item for the MPI failure.
- [`docs/md-run.md`](../md-run.md) — the public `md-run` surface used by the workflow above.
