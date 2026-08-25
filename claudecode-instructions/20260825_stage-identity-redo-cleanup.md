# Focused cleanup: stage identity, redo safety, and accurate runtime wording

Work on the current `dev` branch of `csy0000/MD-templates`.

Start from commit `61cd91cc3e6cfcad874ea6dcf09296ac418ffeb2`. This is a small follow-up to the completed nested-stage correction. Preserve the simplified six-command architecture and the scientific behavior already validated on CUDA.

Do not merge into `main`, create a branch, create or move a tag, or change the package version from `0.3.0.dev0`.

## 1. Remove the misleading MD_REDO behavior

The current generated common-stage runner advertises:

```bash
MD_REDO=1 ./run.sh
```

but a completed dynamics stage still has its terminal `checkpoint.chk`. The runner loads that checkpoint, sees zero remaining steps, and rewrites completion outputs without actually rerunning the configured stage from its parent. For minimization it restarts from the completed checkpoint rather than the parent. This is not a valid redo operation.

Remove the `MD_REDO` shortcut from the generated common-stage runner, active documentation, and tests.

Use the simple safe behavior:

- when `final_state.xml` and `resolved_stage.yaml` are both present and consistent, always report that the stage is complete and exit successfully without modifying any file;
- when completion is ambiguous, continue to refuse;
- to redo a stage, instruct the user to either generate a new output directory or deliberately remove that stage's generated runtime outputs before running it again;
- explicitly warn that downstream stages derived from the old final state must also be regenerated or removed by the user;
- do not automatically delete, cascade-delete, quarantine, or manage downstream directories;
- do not add a transaction database, workflow manager, manifest framework, or lock manager.

The removal list in the message should cover the runtime outputs the stage writes:

```text
stage.log
stage.csv
checkpoint.chk
final_state.xml
final.pdb
resolved_stage.yaml
```

Do not remove `run.py`, `run.sh`, or `stage.yaml`.

## 2. Make completion identity cover the complete stage request

The current `COMPLETION_KEYS` list checks only a subset of scientifically relevant settings. A completed stage can currently be accepted after changing values such as its input state, pressure, random seeds, implicit/explicit mode, or other run-defining fields.

Replace the hand-maintained partial comparison with one compact canonical signature of the complete `stage.yaml` document:

1. canonicalize the loaded YAML mapping with `yaml.safe_dump(..., sort_keys=True)`;
2. hash the UTF-8 canonical text with SHA-256;
3. record the value in `resolved_stage.yaml` as `stage_config_sha256`;
4. when both completion artifacts exist, require the recorded hash to equal the hash of the current complete `stage.yaml`;
5. if the field is missing or differs, refuse clearly and say that the completion record belongs to a different stage request.

The signature must cover every key in the stage document automatically, including at least:

- stage name, kind, ensemble, implicit/explicit mode;
- parent and input-state paths;
- minimization iterations or dynamics duration;
- restraint strength;
- temperature, pressure, timestep, and friction;
- integrator, velocity, and barostat seeds;
- velocity-assignment behavior;
- output-state name and template commit.

Do not introduce a schema version or duplicate this as another hand-maintained list.

The hash implementation must travel in the generated standalone project. It must not import `md_templates`, refer to this checkout by absolute path, or make a moved `inputs/` + `MD/` project non-portable.

## 3. Correct the affected tests

Delete the tautological assertion:

```python
assert record["steps"] == record["steps"]
```

Do not replace it with another test that only proves a file was rewritten.

Add or revise focused tests that prove:

1. rerunning a completed common stage exits successfully and leaves every runtime output byte-for-byte unchanged;
2. the generated runner no longer advertises or interprets `MD_REDO`;
3. changing any representative run-defining field, parameterized across at least `input_state`, `system_pressure_bar`, and one random seed, makes the completed stage refuse;
4. an unchanged complete stage has a non-null `stage_config_sha256` that exactly reproduces the canonical hash of its current `stage.yaml`;
5. after deliberately removing all six runtime outputs in a temporary test project, running the stage starts freshly from its parent, resets its local step count, and executes the configured number of steps rather than resuming the removed checkpoint;
6. the existing interrupted-stage checkpoint test still resumes only the missing local steps and appends `stage.csv`.

All tests that minimize or integrate a molecular system must continue to run on CUDA. Do not set `MD_PLATFORM=CPU` or use Reference for those tests.

Use temporary pytest output only. Do not delete or modify real simulation results.

## 4. Correct stale wording only

Update the small number of stale descriptions that now contradict the implementation:

- the top-level tree in `src/md_templates/openmm/mdgen.py` must show `minimization/`, `eq/nvt_1kcal/`, `eq/npt_1kcal/`, `eq/npt_free/`, `cMD/`, and `REST2/`;
- `rest2_equilibrate.py` must not say “one replica at a time”; explain that replicas on different GPUs run concurrently and replicas sharing one GPU run sequentially;
- `tests/conftest.py::tiny_project` must not say the workflow runs on CPU;
- correct active pytest comments that describe the CUDA smoke workflow as a CPU workflow;
- retain intentional CPU/Reference wording for installation probes and the CPU-only packaging workflow, because those do not claim MD runtime validation.

Do not rewrite the README, journals, or historical reports unless an active statement directly contradicts the current generated behavior.

## 5. Focused validation only

Do not repeat the full 99-test run and do not run nanosecond-scale MD.

Run the smallest relevant checks:

1. non-MD unit/static tests covering the canonical signature, scheduling helper, platform policy, and generated wording;
2. the affected common-stage CUDA smoke tests for:
   - completed-stage no-modification;
   - mismatched-stage refusal;
   - deliberate manual cleanup followed by a fresh run;
   - interrupted-stage restart;
3. one generated explicit project inspection proving the nested `eq/` paths are unchanged;
4. compile/import and wheel-content checks only if files included in the wheel were changed.

Use the available CUDA devices. If the targeted molecular tests cannot obtain a working CUDA platform, report a blocker rather than running them on CPU.

Do not rerun cMD or REST2 production unless a changed shared helper requires a very small smoke check. This cleanup must not change REST2 scaling, exchange acceptance, duration accounting, device placement, trajectories, seeds, box swaps, or barostat behavior.

## 6. Scope that must remain unchanged

Preserve:

- `src/md_templates/openmm/sysgen.py` with `generate_system()`;
- `src/md_templates/openmm/mdgen.py` with `generate_md()`;
- the six public commands;
- the nested `MD/eq/` directory structure;
- OpenMM 8.6.0;
- OPC explicit and GBn2/mbondi3 implicit defaults;
- CUDA-default generated scripts;
- REST2 concurrent propagation and all validated scientific invariants;
- only `main` and `dev` as branches.

Do not restore root-level `MD_system_gen.py` or `MD_input_gen.py`.

## 7. Completion report

Commit and push the cleanup to `dev`, then report:

- commit SHA;
- changed files;
- exact targeted tests and results;
- CUDA devices used by molecular tests;
- old and new completed-stage behavior;
- canonical `stage.yaml` hash and the matching recorded value;
- evidence that a completed stage's six runtime outputs were unchanged;
- deliberate-removal fresh-run step count;
- interrupted-run before/resume/final step counts;
- confirmation that `main`, tags, REST2 scientific code, and package version were not changed;
- remaining limitations.

Do not report PASS based only on YAML generation, mocks, or documentation. Continue fixing routine failures within this narrow scope without stopping to ask questions.
