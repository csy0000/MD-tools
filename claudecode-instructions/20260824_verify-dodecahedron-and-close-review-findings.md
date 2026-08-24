# Claude Code instruction: verify dodecahedron geometry and close the remaining review findings

Date: 2026-08-24  
Repository: `csy0000/MD-templates`  
Target branch: `dev`  
Expected starting commit: `0d84a549a3a0fa6e4d4f81577b0991db29e4d570`

## Purpose

Perform an evidence-first audit of the remaining findings from the OpenMM 8.6.0 ALA validation work.

Do **not** assume that the dodecahedron implementation is wrong. The expectation is that the implementation is probably correct and that the current explanation is wrong. Prove that from the generated OpenMM `System`, box vectors, and numerical geometry before changing code.

Likewise, do not classify every finding as documentation-only without tracing the executable path. Separate:

1. physics/runtime defects;
2. metadata or restart-contract defects;
3. documentation-only defects;
4. already-correct behavior that only needs a regression test.

The task is complete only when every item below has an evidence-backed classification and any required fix is committed and pushed to `dev`.

## Working rules

1. Work directly on `dev`; do not create another long-lived branch. Pull/rebase safely and record the actual starting SHA.
2. Preserve existing simulation outputs and user data. Do not delete or rewrite completed trajectories.
3. Do not rerun the four 5 ns validation simulations merely to fix documentation or metadata.
4. Prefer unit tests and short CPU/Reference-platform integration tests. A very short GPU smoke test is allowed only when needed to distinguish two runtime behaviors.
5. Do not stop to ask the user a non-blocking question. Record the assumption and continue with everything that can be completed safely.
6. Do not return `PASS` while any item is merely asserted rather than demonstrated.

---

## Part 1 — Verify the dodecahedron implementation numerically

Inspect the actual code path used by `MD_system_gen.py` and the canonical generator through OpenMM 8.6.0 `Modeller.addSolvent`. Identify exactly where the dodecahedral box vectors and any cutoff-driven box growth are constructed.

Generate a small system through the same production code path, preferably ACE-ALA-NME with the current explicit peptide defaults:

- ff19SB + OPC;
- dodecahedron;
- `padding_nm = 2.0`;
- real-space cutoff = 1.0 nm;
- current cutoff-fit margin and policy.

From the resulting serialized `System` and coordinates, report and test:

1. the three periodic box vectors;
2. the shortest nonzero lattice translation,
   [
   d_{min}=min_{mathbf ninmathbb Z^3setminus{0}}
   left|n_1mathbf a+n_2mathbf b+n_3mathbf cight|,
   ]
   enumerating at least (-3le n_ile3);
3. the three perpendicular cell heights using
   [
   h_a=V/|mathbf b	imesmathbf c|,
   ]
   and cyclic permutations;
4. the OpenMM periodic-cutoff condition using the reduced-box height: `2*cutoff <= min(h_a,h_b,h_c)`, including the configured margin;
5. the minimum solute-to-periodic-image atom distance by enumerating atom pairs and nearby nonzero lattice translations;
6. which quantity OpenMM's `padding` controls in this construction.

The audit must explicitly distinguish these quantities:

- shortest lattice translation;
- perpendicular/reduced-box height;
- maximum legal real-space cutoff;
- solute-to-nearest-periodic-image atom distance.

Do not use `width/sqrt(2)` as a synonym for the shortest lattice translation. For the usual OpenMM rhombic-dodecahedral vectors, verify rather than assume whether `width/sqrt(2)` is the perpendicular cell height.

### Decision

- If the generated system satisfies the requested 2.0 nm solute-image separation and the cutoff criterion, leave the implementation unchanged. Correct only the comments/documentation and add regression tests that would detect a future geometric regression.
- If the generated system does not satisfy either invariant, fix the implementation minimally and repeat the numerical checks.
- Do not introduce a second competing definition of padding.

Update the misleading comments in `src/md_templates/openmm/config.py` and any related documentation. The explanation should state clearly that the 2.0 nm default is a repository choice expressed with OpenMM padding semantics, and that image clearance and cutoff fit are separate checks.

---

## Part 2 — Determine whether the legacy conventional-MD path really lacks an NPT barostat

Trace the public command end to end:

`md-openmm md` -> CLI -> `runner.launch_md()` -> `md.run_md()` -> loaded `bundle_system.xml` -> constructed simulation `System`.

Do not decide from the source of `run_md()` alone. Inspect whether the loaded bundle already contains a `MonteCarloBarostat`.

Add a focused test that constructs or loads a realistic prepared explicit bundle and checks the actual `System` used for propagation:

- explicit NPT: exactly one `MonteCarloBarostat`;
- implicit nonperiodic constant-temperature: zero barostats;
- no path may accidentally add a second barostat;
- any added barostat has the configured pressure, temperature, frequency, and a legal independently derived OpenMM seed.

### Decision

- If the legacy public path already propagates with exactly one correct barostat, record the evidence and add the regression test; make no physics change.
- If it labels the run NPT but propagates without a barostat, fix the path or route it through the already-correct stage implementation.
- If it is intentionally obsolete, remove it from public documentation/entry points or make it fail clearly. Do not leave an advertised command that silently samples a different ensemble.

This check concerns the reusable public runner. It does not by itself invalidate the four completed stage-based validation runs.

---

## Part 3 — Verify and fix implicit ensemble continuity metadata

Exercise `stage._cmd_continuity()` with both a periodic explicit system and a nonperiodic implicit system.

Expected canonical values:

- explicit production: `NPT`;
- implicit production: `nonperiodic-constant-temperature`.

If the current no-barostat branch records implicit simulations as `NVT`, correct it by using the canonical ensemble resolver based on the actual loaded `System`, not by replacing one hard-coded ternary with another.

Add tests that prove:

1. the implicit continuity record never contains `NVT` or `NPT`;
2. explicit cMD records `NPT`;
3. restart/extension continuity accepts an unchanged canonical ensemble and rejects a genuine ensemble change.

This is primarily a metadata/restart-contract correction, but it is code, not merely prose.

---

## Part 4 — Correct repository documentation

Update the top-level `README.md` so that it no longer claims that the repository is explicit-solvent-only or contains no implicit-solvent work.

Document the currently supported OpenMM scope accurately:

- explicit peptide/ligand MD and REST2;
- implicit GBn2 + mbondi3 MD and REST2;
- explicit peptide default ff19SB + OPC;
- Sage ligand water policy as already implemented;
- explicit production NPT;
- implicit production nonperiodic constant-temperature;
- OpenMM 8.6.0 identity policy.

Keep this concise. Do not turn the README into another implementation journal.

---

## Part 5 — Close the completed-run provenance gap without rewriting history

The existing journal states that the four completed run manifests recorded `profile: None`, while current code now records the selected profile for future runs.

If the completed run directories are accessible, inspect the actual manifests and resolved configurations. For each affected run, create an immutable provenance correction sidecar rather than silently editing the original manifest. The sidecar must contain at least:

- path and SHA-256 of the original run manifest;
- run ID;
- original code commit and bundle commit;
- the profile that supplied defaults;
- SHA-256 of the fully resolved configuration;
- reason for the correction;
- timestamp and current correcting commit;
- an explicit statement that trajectory/state data were not changed.

If those external directories are not accessible, do not fabricate sidecars. State precisely that the code fix covers future runs and list the command/check needed on the simulation machine.

Update the journal to distinguish:

- bundle-build commit;
- simulation-run commit;
- post-run audit/fix commits;
- current reviewed `dev` head.

---

## Part 6 — Verification

Run at minimum:

1. the complete fast test suite;
2. the new geometry regression tests;
3. the public conventional-MD barostat-path test;
4. the explicit/implicit continuity metadata tests;
5. the OpenMM 8.6.0 version tests;
6. the existing REST2 exchange-state tests;
7. the CPU smoke/integration workflow if feasible.

A short dynamics smoke test may be used to demonstrate volume behavior:

- explicit NPT system contains an active barostat and records volume;
- implicit system contains no barostat and does not claim a box ensemble.

Do not use “volume happened not to change” as proof that a barostat is absent; inspect the `System` forces directly.

Push all changes to `dev`. Confirm the remote head matches the local head.

## Required final report

Return one table with these rows:

| Finding | Classification | Evidence | Change made | Test |
|---|---|---|---|---|
| Dodecahedron geometry | implementation correct / implementation defect | measured vectors and distances | exact files | test name |
| Legacy cMD barostat | runtime correct / runtime defect / retired path | actual propagated System | exact files | test name |
| Implicit continuity ensemble | correct / metadata defect | emitted continuity record | exact files | test name |
| README implicit support | documentation defect | old/new wording | exact files | documentation check |
| Historical profile provenance | corrected / future-only / blocked by unavailable data | manifest/sidecar hashes | exact files | validation command |

Also report:

- starting and final SHAs;
- OpenMM identity fields;
- total tests passed/skipped/failed;
- whether any simulation was rerun and why;
- whether the four previous 5 ns results remain valid;
- remaining limitations.

Return `PASS` only if every row is supported by evidence and all applicable tests pass.
