# Correct direct-runtime preflight and AIS recovery coverage

## Baseline

Work from branch `dev` at commit:

`8746732484029384e4ec86573bb0b768e3580570`

Do not silently broaden this task into a redesign of the CLI, simulation protocols, output formats, or data-registration contract. Preserve the four public `md-openmm` subcommands and the established Amber-like flags.

## Why this correction is required

The outer `md-openmm md-run` path now performs a shared preflight, but the public/generated Python runtimes do not. A generated `min.py`, `REST2.py`, `rREST2.py`, or `AIS.py` can create an output directory, readable report, log, protocol module, group file, or solute declaration before discovering that the machine configuration is invalid or the requested OpenMM platform cannot run.

This violates the fail-closed contract. Refusing a run must leave no artefact that can be mistaken for a started run.

The previous release notes also overstate the direct-runtime coverage. Correct the code and tests rather than weakening the claim.

## 1. One shared preflight contract at every execution boundary

Refactor `md_tools.run.preflight` into reusable, mode-aware validation used by all authoritative execution entry points:

- `md_tools.run.main.md_run_main`
- `md_tools.md.stage.stage_main` (and therefore generated split stages and all-in-one workflows)
- `md_tools.remd.generated.replica_main` (and therefore generated REST2/rREST2 wrappers)
- `md_tools.ais.run.ais_main` (and therefore generated AIS wrappers)
- any executor entry point that remains public and can be invoked independently

Do not merely add calls whose results are ignored. Each runtime must consume the validated result so that machine/platform/MPI policy is not resolved again by a second implementation.

The shared result should carry, as applicable:

- normalized input and output paths;
- validated topology, System, continuation/source trajectory, and group-file declarations;
- the strictly loaded machine configuration;
- platform request, device placement, and proven OpenMM platform resolution;
- MPI coordination/rank/size for parallel modes;
- validated output-role mapping.

Mode-specific validation may be implemented by typed request/result objects or explicit helper functions. Avoid one untyped dictionary with optional fields whose meaning depends on the caller.

### Required ordering

Before the first filesystem mutation, each direct runtime must complete every validation it can complete without running dynamics:

1. parse exact flags with `allow_abbrev=False`;
2. validate contradictory flags and mandatory values;
3. open MPI coordination for REMD/AIS and fail closed if the launch requires MPI but `mpi4py` is unavailable;
4. normalize paths;
5. verify required inputs are regular files;
6. verify declared trajectory format from file content where applicable;
7. check all output-output and input-output path collisions using resolved paths;
8. strictly load and validate the existing machine/user configuration;
9. resolve device policy and rank placement;
10. prove that the requested OpenMM platform can create a Context;
11. perform cheap topology/System compatibility checks needed to refuse a mismatched input.

Only then may code call `mkdir`, construct `SimulationOutput` or `LogWriter`, write `resolved.config`, `solute.yaml`, `_protocol.py`, group files, checkpoint state, trajectories, reports, or logs.

Imports, parsing, reads, hashing, and an ephemeral CUDA probe Context are not output mutations.

For a generated all-in-one workflow, preserve the established “later parent is pending under `--check`” behavior. Do not create a log to report a pending preflight.

### Direct-runtime regression tests

Add subprocess-level tests that invoke the generated scripts directly, not only `md-openmm md-run`.

For each applicable direct entry point, snapshot the destination and assert it remains byte-for-byte unchanged—or absent—after refusal for:

- malformed YAML;
- duplicate YAML keys;
- unsupported machine platform/precision/device policy;
- an explicit missing user-config path;
- missing topology/System;
- topology/System atom-count mismatch;
- output-output collision expressed through different spellings such as `sub/../run.out` and `./run.out`;
- output path colliding with an input;
- unavailable/uninitializable CUDA, using monkeypatching or a deterministic test seam rather than requiring a GPU;
- missing or invalid AIS source trajectory;
- plural MPI launch without usable `mpi4py`.

Cover at least one split stage, the all-in-one workflow, REST2, rREST2, and AIS. A test that fails in argparse before reaching the intended validation does not count; assert a diagnostic specific to the condition under test.

## 2. Correct machine-configuration validation

An absent default config file remains valid and resolves to built-in CUDA/mixed/local_rank defaults.

However, if a configuration file exists—or is explicitly selected with `--user-config` or `MD_TOOLS_CONFIG`—validate the complete document strictly. In particular:

- `user` must be a mapping;
- `user.person_id` and `user.name` are required non-empty values;
- duplicate keys at every nesting level are errors;
- malformed YAML is an error;
- unknown or unsupported OpenMM settings are errors;
- an explicit path that does not exist is a broken reference, not “configuration absent.”

Use one strict loader and one schema-validation path for registration and simulation. Do not copy the identity rules into a second validator.

Add regression tests for missing/blank `person_id` and `name` through both the loader and a direct generated runtime. The direct runtime must refuse before creating output.

## 3. Make CPU provenance unambiguous

`explicit_cpu` must mean exactly “the user supplied `--cpu` for this invocation.”

Required records:

| Selection | name | explicit_cpu | cli_cpu_override | platform_selection |
|---|---|---:|---:|---|
| built-in CUDA | CUDA | false | false | built-in-default |
| machine CUDA | CUDA | false | false | machine-config |
| machine CPU | CPU | false | false | machine-config |
| CLI `--cpu` | CPU | true | true | cli-override |

Keep `requested_policy` consistent: `default-cuda`, `machine-cpu`, or `explicit-cpu` as appropriate. No record may describe machine CPU as an explicit CLI choice.

Update unit tests to assert every field in all four cases, including `explicit_cpu`.

## 4. Test AIS recovery through the real path runner

Keep the generation-based checkpoint transaction, digest verification, atomic pointer replacement, and retention of at least one prior generation.

Add an integration-level fault-injection test around the actual AIS path execution/resume code—not only `commit_generation()`. It may use a tiny mocked OpenMM Simulation and deterministic reporters, so CI does not need CUDA or a long molecular simulation.

Inject interruption at all meaningful boundaries:

- before and after a trajectory frame becomes durable;
- before and after a work/observation row becomes durable;
- at every checkpoint-generation boundary already exposed by `BOUNDARIES`;
- immediately before and after the commit pointer replacement.

After restart, assert:

- resume starts from the last fully committed generation only;
- no trajectory frame or CSV/work row is silently duplicated or skipped relative to committed counters;
- protocol step, source frame, cumulative work, observation count, trajectory-frame count, and checkpoint state agree;
- an uncommitted newer generation is ignored;
- a corrupt committed pair is refused with an actionable error;
- completed paths clear transactional checkpoint state and cannot be accidentally resumed as incomplete.

If trajectory and table outputs cannot be made transactional with the checkpoint, explicitly truncate them to the committed counts during recovery before appending. Never infer committed progress from whichever output file happens to be longest.

## 5. Documentation and release claims

Update implementation/release documentation only after the tests demonstrate the claims. State clearly that preflight is shared by both the CLI dispatcher and direct generated wrappers.

Do not claim successful GPU execution unless a real GPU test was run. CI may test platform refusal and policy deterministically; record any separate local CUDA/MPI run with the exact command, hardware/platform, outcome, and commit.

## Required verification

Run at minimum:

```bash
python -m pytest -q
python -m build
python -m pip install --force-reinstall dist/*.whl
md-openmm -h
md-openmm md-run -h
```

Also run focused regression selections for preflight, machine configuration, platform provenance, MPI, generated wrappers, and AIS checkpoint recovery.

Inspect the final diff for:

- no duplicate MPI or platform policy;
- no output mutation before preflight;
- no stale `explicit_cpu=True` for machine CPU;
- no tests that pass for the wrong argparse error;
- no release note claim unsupported by a test.

Commit the implementation and tests to `dev`, then remove this task file in a final cleanup commit only after all acceptance criteria pass. Report the implementation commit, cleanup commit, exact test counts, skipped tests with reasons, and any remaining limitation.
