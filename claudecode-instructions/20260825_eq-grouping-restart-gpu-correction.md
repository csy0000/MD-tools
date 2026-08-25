# Equilibration grouping, restart accounting, and GPU-only runtime testing

Recorded from the request. A focused correction of the `0.3.0.dev0` stage-directory
implementation. Do not redesign the six-command CLI or restore the deleted framework.

## 1. Group the common equilibration stages

```text
MD/eq/nvt_1kcal/
MD/eq/npt_1kcal/
MD/eq/npt_free/
```

Implicit solvent generates only:

```text
MD/eq/nvt_1kcal/
MD/eq/nvt_free/
```

Minimisation stays at `MD/minimization/`: it is not equilibration.

## 2. Correct common-stage restart accounting

Reset the step count after loading a PARENT state. Do not reset it after loading the stage's own
checkpoint — that count is how far this stage has come, and resetting it runs the stage again on
top of itself.

## 3. Prevent completed common stages from being silently rerun or overwritten

A stage that has written `final_state.xml` must refuse to run again by default, because downstream
stages may already have consumed it. Redoing it must be explicit.

## 4. Correct `md_config_hash`

Hash the actual resolved `MD/md.config.yaml`, not the input document. Resolution fills in stage
paths, drops the inapplicable solvent block and reconciles the ensemble with the solvent, so the
two differ.

## 5. REST2 per-tau equilibration must be concurrent

Concurrent across different GPUs, sequential for replicas sharing one — the same rule exchange
production already uses.

## 6. Complete artifact set for every common stage, minimisation included

`stage.log`, `stage.csv`, `checkpoint.chk`, `final_state.xml`, `final.pdb`, `resolved_stage.yaml`.

## 7. cMD and REST2 branch from the same final state under `MD/eq/`

## 8. Preserve existing behaviour

REST2 duration semantics, seeds, trajectories, NPT box-vector handling, common beta, and extension.

## 9-11. Testing platform policy

Every test that minimises or integrates a molecular system runs on CUDA. CPU and Reference are
permitted only for installation/platform probes and pure non-MD unit tests. The CPU-only GitHub
release workflow must not execute or claim validation of molecular simulation; GPU runtime
acceptance comes from the local GPU machine.

## 12. Keep tests focused and short

Do not repeat nanosecond production validation.

## Constraints

Keep the version at `0.3.0.dev0`. Do not merge into `main`, create another branch, or create,
delete or move a release tag. Commit and push to `dev`.

Do not return PASS based on directory names, YAML, mocks, or CPU execution.
