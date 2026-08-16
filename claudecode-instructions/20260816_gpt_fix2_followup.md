Claude Code instruction: finish OpenMM run continuity and public MD workflow

Work on the openmm branch of csy0000/MD-templates.

Read these before changing code:

CLAUDE.md

docs/journal/2026-08-16_first-refactor.md

src/md_templates/openmm/runner.py

src/md_templates/openmm/cli.py

src/md_templates/openmm/rest2.py

src/md_templates/openmm/md.py

the experiment schemas and all current tests

The first refactor correctly completed most of the repository cleanup. Do not redesign or regress the completed chunk schema, equilibrated-box output, ion accounting, package identity, or default omega exclusion. This pass must finish the remaining behavior in the real CLI and execution paths. A standalone helper with no public caller does not count as an implementation.

Required outcomes

1. Complete named runs and same-directory continuation

Wire the existing run-directory resolution logic through every relevant public command and runner.

For new REST2 and conventional-MD runs:

Accept --run-name NAME.

When --run-name is supplied, create exactly <out-root>/<NAME>; do not append a timestamp or hash to the requested name.

When it is omitted, create a unique timestamped directory using the repository's existing naming convention.

Refuse to overwrite or silently reuse a non-empty directory. Tell the user to resume it explicitly.

For continuation:

Accept --resume-run PATH_OR_NAME.

--run-name and --resume-run must be mutually exclusive.

Resume in the existing directory. Never create a child directory or a new timestamped sibling.

Treat continuation as an extension of the same run, not as a new run.

Append to trajectory, state-data, exchange, and other history files without truncating them or duplicating CSV headers.

Preserve monotonically increasing step, time, chunk, frame, and exchange-attempt indices.

Before changing an existing run, validate continuity against its persisted provenance. At minimum compare:

simulation method (md versus rest2);

system/bundle identity and fingerprint;

topology and particle count;

force-field and water-model identity;

thermodynamic settings;

timestep, constraints, integrator, and nonbonded settings;

REST2 replica/temperature ladder and omega-exclusion setting, when applicable.

Reject incompatible continuation with a concise error that lists the differing fields. Do not offer or implement a force/unsafe bypass in this pass. Runtime extension fields such as the number of additional chunks must not make an otherwise compatible run fail validation.

The experiment input continues to mean: run exactly n_chunks additional chunks of chunk_ns during this invocation. Derive the invocation duration and lifetime duration; do not reintroduce a rounded total_ns -> n_chunks calculation.

2. Make restart persistence crash-safe and portable

At every completed chunk boundary, persist both:

an OpenMM binary checkpoint for the best same-platform continuation; and

a serialized OpenMM State containing positions, velocities, periodic box vectors, simulation time, and parameters for portable fallback.

Do not overwrite the last committed restart in place. Use generation-labelled files plus an atomically replaced progress/commit record, or an equivalently safe transaction:

write new checkpoint and State to temporary files;

flush, close, and rename them into their final generation-labelled names;

atomically replace the small progress record that identifies the committed generation;

retain at least the previous committed generation until the new one is committed.

On resume:

load the latest committed binary checkpoint when possible;

if it is missing, corrupt, or incompatible with the current OpenMM platform, warn and load the matching serialized State;

fail before appending output if neither restart is usable.

Restore the periodic box, time/step counters, run phase, current replica assignment, and exchange RNG state. A State fallback may lose bitwise stochastic-integrator continuity; document that limitation, but it must preserve a physically valid continuation and must not reset simulation time or statistics.

Do not claim that two independently replaced files are an atomic pair. The progress record must point only to a fully written, internally consistent restart generation.

3. Preserve lifetime REST2 exchange statistics

REST2 statistics must describe the entire run directory, including all previous invocations.

Persist one durable record per exchange attempt with a stable global attempt index, chunk/step, proposed pair, accepted/rejected result, and any existing diagnostic quantities.

Append new attempts on resume without repeating existing rows.

Reconstruct lifetime attempted/accepted counts from durable history when opening a run. Do not initialize the displayed lifetime counters to zero.

Persist and restore the exchange scheduler phase, replica assignment, and random-number-generator state so a normal checkpoint resume does not restart the exchange sequence.

Regenerate the final summary from durable attempt history. Summary generation must be idempotent.

Report both invocation and lifetime statistics if invocation statistics remain useful, and label them unambiguously.

Detect duplicate or non-monotonic attempt indices as run corruption instead of silently double-counting them.

Handle interruption boundaries explicitly. An attempt must not be counted as committed unless the restart/progress generation that contains its resulting state is also committed. Use either generation-scoped attempt records or a progress watermark so recovery can ignore an uncommitted tail safely.

4. Add a first-class conventional-MD CLI

Expose the existing conventional-MD engine as a supported public workflow:

md-openmm md --bundle BUNDLE --experiment EXPERIMENT --out-root RUNS [--run-name NAME | --resume-run PATH_OR_NAME]

Requirements:

The command must perform conventional explicit-water MD without constructing REST2 replicas or exchange machinery.

Use the same n_chunks plus chunk_ns contract.

Reuse the common run-directory, provenance, compatibility, restart, and reporting implementation instead of copying it.

Support the same platform/device options that are meaningful for REST2.

Produce the normal trajectory, state-data log, restart generations, resolved configuration/provenance, and a concise final summary.

Keep method-specific configuration explicit. Do not silently accept and ignore REST2-only settings for an MD run.

Refactor shared execution infrastructure when necessary, but keep the MD and REST2 algorithms separate and readable.

5. Correct public documentation

Update CLI help, examples, README content, configuration documentation, and module docstrings so they describe the implemented behavior.

In particular:

remove the stale claim that every run directory is immutable and timestamped;

document exact named-directory behavior and explicit same-directory continuation;

document n_chunks as additional work per invocation;

document checkpoint-first and State-fallback behavior;

document lifetime versus invocation REST2 statistics;

list the public md command;

retain rest2.omega_exclusion: true as the default and --omega-exclusion false as its explicit override.

Tests required

Add focused unit/integration tests. Keep expensive GPU tests optional, but the control-flow tests must run on CPU or with existing mocks.

At minimum test:

an exact --run-name creates no timestamp suffix;

omitted --run-name creates a unique timestamped directory;

--run-name and --resume-run are rejected together;

an existing directory is never reused without --resume-run;

both CLI commands actually pass naming/resume options into the execution path;

a resumed run uses the same directory and appends output without a repeated header;

steps, time, chunks, frames, and REST2 attempt indices remain monotonic across resume;

incompatible bundle/config provenance is rejected before files are appended;

a valid binary checkpoint is preferred;

a missing/corrupt/incompatible checkpoint falls back to the matching State;

a failed restart generation does not replace the previous committed generation;

an uncommitted exchange-log tail is not included after recovery;

lifetime attempted/accepted totals equal the combined pre-resume and post-resume history;

repeated summary generation produces identical results without double-counting;

restored exchange RNG state and scheduler phase continue rather than restart the sequence;

md-openmm md --help is present in the documented-command test;

the conventional-MD route runs without creating REST2/exchange objects;

default omega exclusion remains enabled and --omega-exclusion false still disables it.

Include at least one end-to-end short CPU test for new run -> resume same directory if feasible in CI. If it is too slow for the default suite, mark it clearly and provide the exact command to run it.

Implementation constraints

Preserve all currently passing tests unless a test encodes the obsolete behavior described above; update such a test deliberately.

Do not add back any project-specific terminology or code containing the old escort or ais identities. Audit paths, import names, package metadata, examples, comments, and test fixtures case-insensitively.

Do not reintroduce total-duration rounding.

Do not make omega exclusion opt-in.

Do not implement resume by copying files into a new directory.

Avoid duplicating restart/run-management code between MD and REST2.

Use atomic filesystem replacement primitives available in Python; do not rely on shell commands.

Keep persistent formats versioned and validate their schema on read. If the run-state format changes, provide a clear error or migration for directories produced by the immediately preceding refactor.

Verification and deliverables

Before finishing:

Run the full test suite and report the exact command and result.

Run CLI help for the root command, md, and rest2.

Run the repository's stale-name audit for case-insensitive escort|ais matches and explain any intentional matches; normally there should be none.

Run a minimal conventional-MD smoke test.

Run a minimal REST2 new -> resume smoke test and show that the same directory is used and lifetime counters increase.

Inspect git diff for unrelated changes.

Add a dated journal entry under docs/journal/ containing:

files changed;

persistent-format decisions;

compatibility checks implemented;

tests and smoke commands with results;

remaining limitations, especially State-fallback stochastic reproducibility.

Do not report a requirement as complete merely because its helper or test fixture exists. Trace each public CLI option through parsing, runner dispatch, filesystem behavior, restart loading, and final output.
