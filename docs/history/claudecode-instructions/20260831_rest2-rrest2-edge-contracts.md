# Close the remaining REST2/rREST2 edge contracts

## Scope and starting points

Work in one Claude Code session across:

```text
git@github.com:csy0000/MD-templates.git
git@github.com:csy0000/MD-project.git
```

Before changing anything, fetch and verify these exact remote heads:

```text
MD-templates/fix/openmm-md-rest2-rrest2-correctness
6cb693f748a2b63e5a0a49a9500ae295433fc2e0

MD-project/dev6-openmm-md-rest2-rrest2-correctness
ba721e02037c66eeecd305903fed7488657904c2
```

Read and follow every applicable `CLAUDE.md`, repository instruction, and the execution journal:

```text
MD-project/docs/journals/20260830_rest2-rrest2-correctness-phase-space.md
```

Inspect the current code and tests rather than relying on this instruction alone. If either remote head differs, stop and report the observed full SHA instead of silently rebasing or working from another revision.

Create new branches from those exact heads:

```text
MD-templates: fix/openmm-md-rest2-rrest2-edge-contracts
MD-project:   dev7-openmm-md-rest2-rrest2-edge-contracts
```

Do not merge into `dev` or `main`, do not rewrite history, and do not alter the existing source branches. Make logical commits and push each new branch only after validation.

## Objective

Close three remaining interface and storage gaps without redesigning the working REST2/rREST2 runtime:

1. make the explicit `maxwell` velocity policy internally consistent;
2. resolve the contradiction between reservoir sampling with replacement and strictly ordered phase-space storage;
3. prevent `openmm-md --resume/--extend` from being silently accepted by conventional single-protocol cMD when that path does not implement append-safe continuation.

Preserve the established architecture:

```text
openmm-md --groupfile ... -ng ... [--exchange-rule ...] [--reservoir ...]
```

Keep one executor, the repository-owned replica runtime, Hamiltonian REST2 scaling at one physical temperature, independent event schedules, rank-0 writable storage, exact System fingerprints, and the current ALA defaults. Do not introduce `openmm-rest2`, OpenMMTools production execution, temperature REMD, kinetic reservoirs, non-Boltzmann acceptance, or NPT REST2.

## 1. Make `velocity_policy` one end-to-end contract

The current reservoir loader parses `velocity_policy`, but preparation calls `PreparedReservoir._prepare()` without passing the selected policy. The helper therefore uses its default `"stored"` even when the declaration explicitly requests `"maxwell"`.

Correct this so the parsed policy is passed explicitly through preparation, validation, materialization, restart identity, and provenance. Do not allow a helper default to override the declaration.

Use this narrow contract:

- The reservoir source remains the versioned phase-space NetCDF format for both policies. This preserves positions, box, absolute source step/time, Hamiltonian identity, atom identity, and completion markers in one auditable source.
- `velocity_policy: stored` remains the default. It requires finite, correctly shaped, nonzero stored velocities and installs them unchanged.
- `velocity_policy: maxwell` is an explicit alternative. It uses the source positions and box but deliberately ignores the stored velocity values and redraws momenta at the common physical temperature from a recorded deterministic seed.
- Do not describe `maxwell` as support for DCD or an arbitrary coordinate-only source. A DCD lacks the phase-space format, Hamiltonian identity, absolute-step record, and completion markers required by this reservoir contract.
- There must be no automatic fallback from `stored` to `maxwell`.
- Record the selected policy and, for every Maxwell refresh, the seed or sufficient deterministic RNG state required to reproduce the draw.
- Serial and MPI execution must install the same redrawn momenta for the same seed and reservoir event. Only the owning rank should invoke OpenMM's velocity draw; the exact resulting array must be shared and digest-checked as the current runtime intends.

Correct all misleading module docstrings, README material, examples, schemas, and error messages. Do not claim coordinate-only support unless a separately versioned, scientifically validated source contract is actually implemented; that is outside this task.

### Required tests

Add tests that fail on the current head and prove:

- `PreparedReservoir.open()` passes the requested policy into preparation;
- a phase-space source with identically zero velocities is refused under `stored`;
- the same source is accepted under explicit `maxwell`, because its velocity values are deliberately ignored;
- an unknown policy is refused;
- omitting the field selects `stored`;
- no code path silently changes policies;
- a fixed seed produces the same Maxwell velocity array in serial and the supported MPI ownership path;
- the manifest and completion record identify the policy and Maxwell seed/RNG evidence.

## 2. Forbid replacement when materializing a Boltzmann rREST2 reservoir

The current declaration exposes `allow_sampling_with_replacement`. Selection can therefore choose the same source frame more than once. Materialization then sorts the selected indices, while `PhaseSpaceReader.validate()` correctly requires strictly increasing absolute steps. A duplicated selection is consequently emitted and then rejected by the repository's own validator.

Do not weaken the strictly increasing-step invariant. Duplicate reservoir entries also give one empirical source configuration extra statistical weight while making the declared frame count overstate the effective reservoir size.

For this Boltzmann rREST2 implementation:

- reject `allow_sampling_with_replacement: true` before selecting or writing anything;
- state that the requested number of frames must not exceed the number of eligible distinct source frames;
- report the requested count, eligible distinct count, time window, and how to correct the request;
- keep AIS behavior unchanged if AIS has a legitimate, separately documented replacement contract;
- keep the shared source-window selection machinery, but apply the rREST2-specific prohibition at the reservoir boundary;
- preserve strictly increasing source frame indices, source steps, and times in the prepared file and manifest.

Do not silently deduplicate a replacement draw: that would change both the requested count and its weights without recording the change.

### Required tests

Add tests that prove:

- `allow_sampling_with_replacement: true` is refused before a prepared file is created;
- requesting more frames than eligible distinct frames fails with an actionable message;
- a valid no-replacement selection contains unique, strictly increasing source frame indices and steps;
- AIS replacement behavior, if supported, is unchanged;
- no partial reservoir directory or misleading completion manifest remains after refusal.

## 3. Refuse unsupported conventional-cMD continuation flags

The `openmm-md` parser accepts `--resume` and `--extend` in both grouped and non-grouped modes, but only the grouped replica runtime consumes those flags. The conventional protocol path calls `load_protocol(...).run(files)`; generated cMD scripts reset time and step state and create their reporters as a fresh run. Accepting continuation flags there is therefore misleading and may collide with or replace DCD, checkpoint, and phase-space outputs.

For this milestone, do not attempt a partial conventional-MD restart implementation. Add an early, side-effect-free validation rule:

- `--resume` and `--extend` are supported only with `--groupfile`;
- in non-grouped mode, refuse either flag before creating directories, opening reports, importing the protocol, or touching any output;
- the message must say that append-safe conventional cMD continuation is not implemented and that the user must either run a fresh stage in a new/clean output location or use a future dedicated continuation implementation;
- `--force` remains a deliberate fresh-run replacement control and must not be described as continuation;
- grouped REST2/rREST2 resume and repeated extension must remain unchanged.

Update the CLI help and documentation so the mode boundary is visible before execution.

### Required tests

Add tests that prove:

- non-grouped `--resume` returns the validation error and leaves every existing output byte-identical;
- non-grouped `--extend N` does the same;
- validation occurs before protocol import or output-directory creation;
- grouped `--resume` and repeated grouped `--extend` still work;
- the co-generated fixed-`tau` rREST2 source stage cannot be accidentally treated as append-safe cMD;
- the error text does not suggest deleting authoritative data automatically.

## Scientific and workflow regression checks

Retain the current defaults and rerun the existing acceptance matrix:

- ALA implicit REST2: ff14SB + GBn2 + mbondi3, no SASA;
- ALA implicit rREST2 with `velocity_policy: stored`;
- ALA explicit NVT REST2: ff14SB + TIP3P;
- ALA explicit NVT rREST2 with `velocity_policy: stored`.

Keep ff19SB/OPC as an optional compatibility case, not the default.

Also run one small implicit ALA rREST2 smoke with explicit `velocity_policy: maxwell`. This validates mechanism only; do not present it as the recommended scientific default or as convergence evidence.

For all replica smokes, verify:

- six states at `tau = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`;
- one physical thermostat temperature;
- exact independent exchange, whole, solute, and checkpoint schedules;
- correct walker/state mapping;
- reservoir events refresh only the top rung;
- stored mode preserves recorded momenta;
- Maxwell mode records and reproduces the velocity draw;
- serial and supported MPI ownership agree under a deterministic comparison platform;
- resume and two consecutive extensions preserve counters and storage;
- validation opens and checks every authoritative artifact.

Do not run production simulations.

## MD-project integration

After MD-templates is final:

1. pin `components.yaml`, `components.lock.yaml`, and every relevant system configuration to the exact new full MD-templates commit SHA;
2. update rREST2 examples and schemas to the corrected velocity-policy and no-replacement contracts;
3. keep the Snakemake workflow acyclic and dry-run capable;
4. ensure the workflow never passes conventional cMD continuation flags;
5. update documentation without reviving stale OpenMMTools or `openmm-rest2` language;
6. add an execution journal at:

```text
docs/journals/20260831_rest2-rrest2-edge-contracts.md
```

The journal must record exact starting and final SHAs, commands, environments, test counts, skips, unavailable hardware checks, actual smoke outputs, and every remaining limitation. Distinguish repository inspection, unit tests, CPU smoke, CUDA smoke, and MPI smoke; do not report one as evidence for another.

## Completion procedure

Before pushing:

1. inspect the complete diff against both starting heads;
2. run formatting and linting;
3. run the complete MD-templates test suite;
4. run the complete MD-project test suite;
5. run the required ALA smoke matrix and the explicit-Maxwell smoke;
6. exercise the non-grouped refusal paths against pre-existing sentinel outputs and verify their checksums do not change;
7. verify grouped resume and two consecutive extensions;
8. run available CUDA and six-rank MPI checks; record unavailable checks honestly;
9. dry-run the MD-project REST2 workflow;
10. validate all edited YAML and NetCDF schemas;
11. check generated projects contain no absolute machine paths;
12. inspect `git status` for generated files and unrelated changes.

Make logical commits, record the exact final SHAs in the journal, and push:

```text
MD-templates/fix/openmm-md-rest2-rrest2-edge-contracts
MD-project/dev7-openmm-md-rest2-rrest2-edge-contracts
```

Do not merge into `dev` or `main`.

Stop and report rather than inventing behavior if the current APIs, dependencies, or available hardware cannot support a requested check safely.
