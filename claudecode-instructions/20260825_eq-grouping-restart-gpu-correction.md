# Focused correction: group equilibration stages and close restart/GPU gaps

Work on the current dev branch of csy0000/MD-templates. The starting point is the stage-directory implementation introduced by commit cb3e3b7.

This is a focused follow-up. Preserve the simplified six-command CLI and the existing OpenMM scientific behavior. Do not restore the deleted registry, bundle, schema-migration, workflow-engine, or root-level generator architecture.

Do not merge into main, create a branch, or create, delete, or move a release tag. Keep the package as the unreleased 0.3.0.dev0 line.

## 1. Group the common equilibration stages under MD/eq

The generated explicit-solvent tree must be:

~~~text
MD/
├── md.config.yaml
├── provenance.yaml
├── md_stages.py
├── run_all.sh
├── minimization/
│   ├── run.py
│   ├── run.sh
│   └── stage.yaml
├── eq/
│   ├── nvt_1kcal/
│   │   ├── run.py
│   │   ├── run.sh
│   │   └── stage.yaml
│   ├── npt_1kcal/
│   │   ├── run.py
│   │   ├── run.sh
│   │   └── stage.yaml
│   └── npt_free/
│       ├── run.py
│       ├── run.sh
│       └── stage.yaml
├── cMD/
└── REST2/
    ├── equilibrate.py
    ├── equilibrate.sh
    ├── run.py
    ├── run.sh
    ├── extend.sh
    ├── rest2_scaling.py
    └── replica_NN/
        ├── equilibration/
        └── production/
~~~

The explicit dependency is:

~~~text
inputs
→ minimization
→ eq/nvt_1kcal
→ eq/npt_1kcal
→ eq/npt_free
     ├→ cMD
     └→ REST2 per-tau equilibration → REST2 exchange production
~~~

The generated implicit-solvent tree must be:

~~~text
MD/
├── minimization/
├── eq/
│   ├── nvt_1kcal/
│   └── nvt_free/
├── cMD/
└── REST2/
~~~

There must be no implicit NPT directory and no implicit barostat.

The eq directory is organizational only. It is not a stage and must not contain another workflow abstraction. Each child stage remains independently runnable.

Use these parent paths:

- explicit nvt_1kcal reads ../../minimization/final_state.xml;
- explicit npt_1kcal reads ../nvt_1kcal/final_state.xml;
- explicit npt_free reads ../npt_1kcal/final_state.xml;
- implicit nvt_free reads ../nvt_1kcal/final_state.xml;
- cMD and REST2 both read ../eq/npt_free/final_state.xml for explicit solvent;
- cMD and REST2 both read ../eq/nvt_free/final_state.xml for implicit solvent.

If the restraint differs from 1 kcal/mol/A^2, use nvt_restrained and npt_restrained instead of filenames that incorrectly claim 1kcal.

Update generated imports so nested eq stage scripts find the one MD/md_stages.py helper without importing md_templates or embedding an absolute checkout path. Moving inputs/ and MD/ together must remain sufficient.

## 2. Fix common-stage restart step accounting

OpenMM Context.setState() copies the State step count. The current common-stage runner loads its parent State and then treats a later checkpoint step count as if it were local to the current stage. This can make an interrupted stage resume with too few or negative remaining steps.

For every fresh common dynamics stage:

1. load the parent final_state.xml;
2. preserve its positions, velocities, box vectors, time, and applicable context parameters;
3. reset Context step count to zero before beginning this stage;
4. calculate checkpoint progress only from this stage-local step count.

Do not reset the step count after loading this stage's own checkpoint.

Add a focused interrupted-stage test that creates or reaches a partial checkpoint, resumes it, and proves that:

- the resumed stage executes exactly the missing stage-local steps;
- the final stage step count equals the configured stage steps;
- the parent stage step count does not enter the remaining-step calculation;
- its trajectory/log records append rather than restart.

Use a tiny GPU run for this test.

## 3. Protect completed common stages

A completed common stage must not be silently rerun or overwritten.

If both final_state.xml and resolved_stage.yaml are present and internally consistent, the stage must report that it is already complete and exit successfully without modifying either file.

If only one completion artifact exists, or the completion record conflicts with stage.yaml, fail clearly and instruct the user to use a new output directory or deliberately remove that stage's outputs. Do not silently reuse or overwrite ambiguous results.

This must make rerunning MD/run_all.sh safe for the common preparation chain. It may preserve the existing production semantics: cMD continues only up to its configured total duration, while an explicit REST2 production invocation/extension adds its configured exchange rounds.

Do not add a transactional run database, manifest framework, schema version, lock manager, or automatic deletion.

## 4. Correct the MD-configuration provenance hash

In mdgen.py, do not reuse the input/resolved-configuration variable for individual stage documents.

The provenance field md_config_hash must equal the stable hash of the actual resolved MD/md.config.yaml content. Rename the loop variable, compute the hash from the resolved configuration written to disk, and add a test that loads MD/md.config.yaml and reproduces the recorded hash exactly.

Also ensure resolved_stage.yaml records the actual template commit for common stages and REST2 per-tau equilibration rather than null when commit provenance is available.

## 5. Run REST2 per-tau equilibration concurrently across GPUs

The current REST2/equilibrate.py processes replicas serially. Correct it using the same hardware policy as exchange production:

- use at most min(number_of_replicas, visible CUDA devices);
- assign replicas round-robin;
- replicas on different GPUs equilibrate concurrently;
- replicas sharing a GPU equilibrate sequentially;
- await every replica and propagate exceptions before declaring equilibration complete;
- CPU fallback must not be used for scientific runtime tests.

Keep distinct integrator, velocity, and barostat seeds per replica. Preserve the rule that every replica starts from the same common finalized state but integrates under its own tau-scaled Hamiltonian.

Record the replica-to-device assignment in logs and in each replica's resolved_stage.yaml.

Add focused scheduling tests and one tiny multi-GPU runtime test demonstrating overlap across different GPUs. Do not use long equilibration durations for this acceptance test.

## 6. Complete the common-stage artifact contract

Every completed common stage, including minimization, must contain:

~~~text
stage.log
stage.csv
checkpoint.chk
final_state.xml
final.pdb
resolved_stage.yaml
~~~

OpenMM minimization is not required to resume midway. Its checkpoint may represent its completed state, and stage.csv may contain a single final-state record. Do not claim that an OpenMM LocalEnergyMinimizer can resume mid-optimization.

final_state.xml remains the only normal downstream handoff. A downstream stage must never consume its parent's checkpoint.

Write completion artifacts only after successful completion, using atomic replacement for final_state.xml and resolved_stage.yaml where practical.

## 7. GPU-only rule for simulation tests

All tests that perform molecular minimization or dynamics must run with CUDA.

This includes:

- minimization;
- restrained and free equilibration stages;
- cMD production;
- REST2 per-tau equilibration;
- REST2 exchange production;
- interrupted-stage restart tests;
- generated-project portability runtime tests.

Do not set MD_PLATFORM=CPU or use the Reference platform for any scientific runtime acceptance test. Do not silently fall back when CUDA is unavailable. If the machine cannot provide a working OpenMM CUDA platform, report that as a blocker instead of returning PASS.

Pure unit tests that do not construct/integrate a molecular simulation may run on CPU. Reference/CPU one-step checks are allowed only as installation/platform probes; they are not evidence that the MD protocol works.

The GitHub-hosted release runner has no GPU. Update the single release-only workflow so it performs packaging, command generation, configuration, wheel-content, and non-MD unit checks only. It must not run minimization, equilibration, cMD, or REST2 on CPU and must not claim scientific runtime validation. GPU runtime acceptance must be recorded from the local GPU machine before release.

Keep only one release-only workflow. Do not add push, pull-request, scheduled, or additional GPU workflows.

## 8. Focused validation

Do not repeat nanosecond production validation. Use small CUDA stage durations and exchange counts sufficient to test control flow and artifacts.

Verify at minimum:

1. exact explicit and implicit directory trees;
2. exact parent final-state paths after nesting under eq/;
3. missing-parent refusal;
4. fresh common-stage step counts;
5. interrupted common-stage restart step counts;
6. completed-stage no-overwrite behavior;
7. exact md.config.yaml provenance hash;
8. non-null template commit where available;
9. explicit restraint and barostat invariants;
10. implicit absence of all barostats;
11. cMD and REST2 branching from the same eq final state;
12. concurrent REST2 equilibration on different GPUs and sequential scheduling when GPUs are shared;
13. per-replica seed uniqueness and device assignments;
14. REST2 duration_per_segment_ps × number_of_exchanges production totals;
15. REST2 extension continuity;
16. all required stage artifacts;
17. portability outside the checkout;
18. no molecular simulation test used CPU or Reference.

Keep the test suite small. Prefer direct tests of these invariants over large duplicated matrices.

## 9. Documentation and scope

Update README.md, CHANGELOG.md, the release-only workflow, and active tests to show the nested eq layout and GPU-only runtime-validation rule.

Preserve:

- src/md_templates/openmm/sysgen.py with generate_system();
- src/md_templates/openmm/mdgen.py with generate_md();
- the six public commands;
- OpenMM 8.6.0;
- OPC explicit and GBn2/mbondi3 implicit defaults;
- AMBER-style tau scaling;
- omega exclusion;
- trajectory intervals;
- REST2 box-vector swapping;
- common-beta exchange explanation;
- CUDA-default generated scripts;
- dev and main as the only branches.

Do not restore MD_system_gen.py or MD_input_gen.py at repository root.

## 10. Completion and report

Commit and push the completed work to dev. Do not merge or tag.

Return:

- commit SHA;
- generated explicit and implicit trees;
- exact parent paths;
- exact tests and results;
- CUDA platform and GPU device assignments used by every runtime test;
- interrupted-stage before/after step counts;
- proof that completed stages were not modified;
- actual and recorded md.config hashes;
- explicit and implicit restraint/barostat evidence;
- REST2 equilibration concurrency evidence;
- REST2 production and extension totals;
- required artifact inventory;
- release-workflow changes and remaining limitations.

Do not return PASS based only on generated filenames, YAML, mocks, or CPU execution. Run the small end-to-end molecular workflows on CUDA and inspect their produced states and artifacts. Continue diagnosing and fixing routine applicable failures without stopping to ask questions.
