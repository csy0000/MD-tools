# Claude Code instruction: integrate the public generators, correct their execution path, and add OpenMM GBn2/mbondi3

Date: 2026-08-20  
Repository: `csy0000/MD-templates`  
Instruction branch at creation: `feature/public-system-and-md-generators`  
Observed pre-instruction head: `47a656c2e28b3311b22f6a48fadf29d12b0297e0`  
Target working branch after Phase 0: `dev`

## Mission

First integrate the completed public-generator branch into `dev`. Then, working directly on `dev`,
make the generated OpenMM workflow truthful end to end and add an implicit-solvent mode whose
default Hamiltonian is GBn2 with mbondi3 radii, matching the pinned implementation in
`csy0000/partitioned-REST2` at commit
`537d5b6c46e9cbd6fa250c0d60b0649f616f4698`.

The repository remains OpenMM-centric. Amber `prmtop`/`rst7` files may be generated as construction
intermediates and provenance for GBn2, but this task does not add an Amber execution engine. Do not
resume the PR #3 migration campaign and do not make it a prerequisite.

The order is mandatory:

1. integrate `feature/public-system-and-md-generators` into `dev` and push `dev`;
2. add regression tests that expose the current execution gaps;
3. correct the explicit-solvent generator path;
4. add implicit GBn2/mbondi3 preparation, conventional MD, REST2, continuation, examples, and tests;
5. run the complete supported verification and push the resulting `dev` commits.

Do not create another feature branch or pull request. Do not merge `dev` into `main`; `main` remains
the stable milestone branch. Do not delete old branches in this task.

## Read before editing

Read completely before changing code:

- `CLAUDE.md`;
- `README.md`, `pyproject.toml`, and `docs/configuration.md`;
- `claudecode-instructions/20260819_openmm-ala-rgdfv-rest2-protocol.md`;
- `claudecode-instructions/20260820_public-md-generators.md`, if present;
- `docs/journal/2026-08-20_branch-integration-and-public-generators.md`;
- the latest relevant OpenMM, bundle, persistence, restart, and REST2 journals;
- `MD_system_gen.py` and `MD_input_gen.py`;
- `src/md_templates/openmm/system_prep.py`;
- `src/md_templates/openmm/input_gen.py`;
- `src/md_templates/openmm/stage.py`;
- `src/md_templates/openmm/system.py`;
- `src/md_templates/openmm/rest2.py`;
- `src/md_templates/openmm/runner.py` and restart/run-state code;
- `tests/test_public_generators.py` and all related tests/profiles/examples.

Read these pinned scientific references completely:

- `https://github.com/csy0000/partitioned-REST2/blob/537d5b6c46e9cbd6fa250c0d60b0649f616f4698/src/escort_ais/systems/topology_prep.py`
- `https://github.com/csy0000/partitioned-REST2/blob/537d5b6c46e9cbd6fa250c0d60b0649f616f4698/src/escort_ais/systems/openmm_system.py`

Treat the pinned commit, not the current default branch, as the reference. Port the scientific
behavior into this repository's existing architecture; do not copy an entire parallel framework.

## Phase 0 — branch integration first

The user explicitly wants branch integration before implementation.

1. Inspect `git status`; preserve and report unrelated changes instead of overwriting them.
2. Fetch all remotes and record exact SHAs for `main`, `dev`,
   `feature/public-system-and-md-generators`, and the current instruction commit.
3. Verify that the feature branch is a strict descendant of `dev`. At instruction creation it was
   four commits ahead and zero behind before this instruction was added. If histories have diverged,
   stop and report the exact graph; do not guess, rebase, force-push, or create a synthetic merge.
4. Fast-forward `dev` to `feature/public-system-and-md-generators`, including this instruction.
5. Run the branch's existing focused generator tests before pushing. If they fail, do not pretend the
   merge is clean: fix only an integration/mechanical failure required to make the existing branch
   testable, record it, and rerun.
6. Push `dev` and verify the remote SHA. Continue the remaining phases directly on `dev`.
7. Leave PR #3 and `migration/pr3-pr8-reusable-template-platform` untouched. Do not merge, close,
   update, rebase, or incorporate them.
8. Do not merge `dev` into `main` and do not create a new branch or PR.

If permissions, branch protection, a moving remote, or a dirty overlapping worktree prevents the
fast-forward, stop before scientific edits and report the blocker precisely.

## Scope and architecture

- Keep `MD_system_gen.py` responsible only for chemistry/system preparation.
- Keep `MD_input_gen.py` responsible only for protocol/stage generation from a prepared bundle.
- Reuse the current staged project layout and the existing REST2 runner/committed-generation
  contract. Do not create a second runner or configuration engine.
- Keep canonical public configuration in OpenMM language. Future Amber/GROMACS adapters translate
  from it; their execution is out of scope.
- Preserve current explicit-water science, including the accepted ff19SB/OPC and Sage/TIP3P routes,
  HMR/timestep choices, tau semantics, and same-directory continuation.
- Do not claim support for a molecular input route that is not exercised through the real public
  entry point. Implement it or refuse it before expensive work with a precise message.
- Do not commit trajectories, checkpoints, serialized production States, generated run trees,
  environments, caches, licensed Amber archives, or large logs.

## Phase 1 — make the existing public execution path truthful

Write regression tests that fail against the integrated pre-fix code, then fix the following.

### 1. Trajectory and stage-log reporting

The cMD stage currently records reporting intervals in JSON but does not attach the corresponding
reporters. Make every declared output real.

- Write the full-system trajectory at the configured `execution.reporting.all_atom` cadence.
- Write the solute/custom-selection trajectory at the configured
  `execution.reporting.solute` cadence, using the exact persisted resolved atom selection.
- Write the declared state-data/stage log at its configured cadence.
- Keep checkpoint and final-State persistence at committed boundaries.
- Validate each interval converts to an exact positive integer number of steps; reject rather than
  round.
- Avoid duplicate headers, truncation, or overwritten trajectories on continuation.
- Generated manifests must list only files that the execution path really writes.

Test via the public generated launcher, not only a helper. Assert files exist, contain frames at the
expected cadence in a tiny CPU run, and preserve monotonic time/step/frame indices across resume.

### 2. Velocity continuity

Do not reset velocities at every dynamic stage.

- Load the predecessor State before deciding whether velocities are needed.
- Initialize velocities once, when entering the first dynamics stage from a State with no velocities
  (normally NVT after minimization), using that stage's deterministic seed.
- Preserve predecessor velocities across NVT -> restrained NPT -> free NPT -> cMD.
- Preserve positions and periodic box vectors as already required.
- Record whether velocities were `initialized` or `inherited` and the seed used, if any.

Add tests that detect the old unconditional `setVelocitiesToTemperature` behavior and prove the
correct transition behavior.

### 3. Seed resolution

Remove the hard-coded `20260820` execution seed from stage behavior.

- Resolve `randomness.master_seed` once.
- Derive stable, nonzero, distinct seeds for conformer generation, each stage, each replica, the
  barostat, velocity initialization, and exchange RNG as applicable.
- Use one shared documented derivation algorithm independent of Python's randomized `hash()`.
- Persist the master seed, derived seeds, their purposes, and the derivation/version.
- Regenerating the same resolved input must yield the same seed map; different stages/replicas must
  not accidentally reuse a seed.

### 4. Straightforward REST2 exchange configuration

Replace the active public fields:

```json
"exchange": {
  "n_exchange_per_segment": 1000,
  "exchange_interval": "5 ps"
}
```

with:

```json
"production": {
  "duration_per_segment": "5 ns"
},
"exchange": {
  "number_of_exchanges_per_segment": 1000
}
```

Derive, do not accept as an independent input:

```text
steps_per_segment = duration_per_segment / timestep
steps_per_exchange = steps_per_segment / number_of_exchanges_per_segment
exchange_interval = steps_per_exchange * timestep
```

Both divisions must be exact integers. Reject old or contradictory fields with a migration message
that names the replacements. Segment count remains in Bash (`NUMBER_OF_SEGMENTS`) and runtime state,
never in scientific JSON. Update schemas, profiles, both worked examples, generated experiment
projection, manifests, documentation, package resources, and goldens together.

Keep the existing public tau convention:

```text
s = (1 - tau)^2
sqrt_s = 1 - tau
```

with default `tau_max = 0.5`, linear interpolation, six alanine replicas and ten RGDfV replicas.

### 5. Resolved system provenance

`system_manifest.json` must not label the raw user JSON as `resolved_system_config`.

- Persist the fully resolved system/build configuration actually used after defaults and route
  decisions.
- Persist source attribution for every value: user input, named profile/default, route-derived, or
  calculated diagnostic.
- Keep derived values clearly labelled and do not accept them back as competing user inputs.
- Record force-field versions/files, charge method, input route, radii/solvent mode, constraints,
  HMR, nonbonded method, cutoff if applicable, composition, box/absence of box, salt/absence of salt,
  package versions, input hashes, and output checksums.
- Ensure `forcefield.json`, `system.yaml`, and `system_manifest.json` describe the same Hamiltonian.

Add a regression test showing omitted values appear with their true resolved defaults and source,
and that the manifest changes when a build-defining input changes.

### 6. Honest input support

The current CLI advertises formats/types that the implementation may route through PDB loading.

- For this task, fully support and execute the current tested PDB peptide/protein route and SMI
  ligand route.
- Unless MOL/MOL2/SDF and protein-ligand complex construction are truly implemented and executed in
  tests, reject them immediately with an actionable `not implemented yet` message.
- Validate `ligand_build` values against the actual force-field route and charge method; do not only
  check that keys exist.

## Phase 2 — implicit-solvent configuration contract

Add a strict discriminated solvation configuration. A representative canonical form is:

```json
{
  "solvation": {
    "mode": "implicit",
    "implicit_model": "GBn2",
    "radii": "mbondi3"
  }
}
```

Explicit mode retains its existing water/box/salt/nonbonded fields. Implicit mode defaults exactly
to `GBn2` and `mbondi3` and has no water, box, ions, salt concentration, PME, real-space cutoff,
pressure, or barostat.

- Reject explicit-only fields in implicit mode; do not silently ignore them.
- Reject implicit-only fields in explicit mode.
- Reject implicit NPT stages and pressure settings before generation.
- Canonicalize spelling/case once while persisting the canonical `GBn2`/`mbondi3` values.
- Bump the affected schema/profile versions and provide actionable migration errors.
- Add named implicit conventional-MD and implicit REST2 profiles rather than overloading an explicit
  profile with hidden branching.

The implicit default stage graph is:

```text
min -> eq_nvt -> cMD_1 -> REST2_1
```

There is no restrained or free NPT stage. Minimization may retain the configured solute restraint;
NVT restraint behavior must be explicit in its own stage configuration. cMD and REST2 use NVT.

Do not silently inherit the explicit-water HMR/timestep profile. The pinned reference's base System
construction is the identity target and does not request hydrogen mass repartitioning. Make the
implicit profile's default mass/timestep choice explicit, documented, and tested. HMR may be an
explicit opt-in only if it is recorded as a different build configuration; it must not alter the
GBn2 energy-matching tests.

## Phase 3 — build the GBn2/mbondi3 System exactly through ParmEd

The Hamiltonian-defining path must match the pinned reference:

```python
st = parmed.load_file(str(prmtop_path), xyz=str(inpcrd_path))
parmed.tools.changeRadii(st, "mbondi3").execute()
system = st.createSystem(
    nonbondedMethod=openmm.app.NoCutoff,
    constraints=openmm.app.HBonds,
    implicitSolvent=openmm.app.GBn2,
    removeCMMotion=True,
)
```

Do not replace this with `AmberPrmtopFile.createSystem()`. The pinned project documents a measurable
CustomGB energy difference for that superficially similar route. The ParmEd construction branch is
part of the scientific identity.

Preparation requirements:

- For a protein/peptide ff19SB topology produced by tleap, set
  `set default PBRadii mbondi3` before `saveAmberParm`.
- Still execute `parmed.tools.changeRadii(..., "mbondi3")` before `createSystem`; it is harmless when
  radii are already correct and necessary for Sage/OpenFF topologies that may lack them.
- For a ligand or protein-ligand route, retain the current vetted OpenFF/Sage parameterization and
  charge provenance, serialize it through ParmEd to Amber topology/coordinate files, then construct
  the OpenMM System through the exact ParmEd path above.
- Do not add water, ions, periodic box vectors, cutoff/PME, or a barostat.
- Verify every particle/topology index mapping used for selections and REST2.

The portable implicit bundle must contain at least:

```text
system.xml
topology.pdb
topology.cif
initial_state.xml
system.prmtop
system.rst7
forcefield.json
system_manifest.json
checksums.json
```

Choose one canonical Amber coordinate filename (`system.rst7` is preferred); if the library writes
`inpcrd`, normalize or document the exact role without producing ambiguous duplicate authorities.
The Amber files are construction/provenance artifacts, not evidence of an Amber runner.

The manifest must state the exact ParmEd version, OpenMM version, topology-generation route, tleap
commands/input where used, force fields, charge method, `GBn2`, `mbondi3`, `NoCutoff`, `HBonds`,
`removeCMMotion=True`, no periodic box, no water/ions/salt, all hashes, and the resolved selection.

Keep explicit bundles backward compatible and unchanged except for corrections explicitly required
in Phase 1.

## Phase 4 — implicit conventional MD

Make the public generator and staged launcher execute a short implicit workflow end to end:

```text
prepare -> restrained minimization -> NVT equilibration -> NVT cMD
```

- No NPT stage may be emitted.
- No barostat may be present in the System or results metadata.
- Initial velocity handling and seed derivation follow Phase 1.
- Reporting produces the declared all-atom, selected-atom, state-data, checkpoint, and final-State
  outputs.
- Continuation uses the existing committed-generation authority and preserves monotonic steps,
  time, frames, and invocation history.
- A relocated prepared bundle must still generate and execute from outside the checkout with the
  installed wheel.

## Phase 5 — implicit REST2

The existing `build_rest2_scaled_system` currently refuses `CustomGBForce`. Extend the one existing
REST2 implementation rather than creating a second method.

For the first supported implicit REST2 contract:

- The entire molecular system is the solute/enhanced region.
- Require the resolved enhanced selection to include every real System particle expected by the
  GB force. Refuse partial implicit selections until their GB cross terms have an independently
  validated treatment.
- Continue to use public `tau`, deriving `s=(1-tau)^2` and `sqrt_s=1-tau`.
- Scale eligible solute bonded/torsion/CMAP terms and nonbonded terms according to the existing REST2
  rules.
- Multiply the complete `CustomGBForce` energy by `s`, including the nonpolar/dispersion term that
  charge scaling alone would miss. Implement this with a tested global parameter injected before
  Context creation, or an equivalently exact method that scales every GB energy term.
- Keep omega exclusion torsion-only and apply it after resolving the enhanced region. It must not
  remove or alter GB/nonbonded scaling.
- Persist the original selection, resolved indices, atom-order fingerprint, excluded omega bonds,
  tau ladder, derived s values, seeds, replica/device map, and exact Hamiltonian identifiers.
- Preserve alternating nearest-neighbor exchange, durable global attempt indices, phase, RNG state,
  walker mapping, committed watermark, checkpoint preference, State fallback, and same-directory
  continuation.
- The exchange interval remains derived from `duration_per_segment` and
  `number_of_exchanges_per_segment`; no implicit-specific duplicate fields.

At `tau=0` / `s=1`, the implicit REST2 System must be energetically identical to the unscaled GBn2
System within a justified numerical tolerance.

## Phase 6 — examples

Keep the existing explicit examples working and add small, reviewable implicit examples without
copying molecular inputs unnecessarily. Use:

```text
test/ala/implicit/
test/ala/implicit/extension/
test/rgd/implicit/
test/rgd/implicit/extension/
```

Each example must contain or reference:

- a system configuration selecting GBn2/mbondi3;
- conventional-MD and REST2 protocol configurations;
- exact generator and launcher commands;
- expected stage graph showing that NPT is absent;
- tau ladder and exchange derivation;
- output layout and status meanings;
- a same-directory extension command using existing checkpoints;
- a clear note that a smoke run is engineering validation, not convergence or scientific
  validation.

Alanine uses the complete ACE-ALA-NME solute and six replicas. RGDfV uses the already vetted
cyclo-RGDfV input/chemistry and ten replicas; do not reinvent stereochemistry or charge. Keep
generated outputs gitignored.

## Phase 7 — required tests and validation

Add tests that would fail against the old behavior. At minimum cover:

1. strict explicit/implicit solvation schema separation;
2. GBn2 and mbondi3 default resolution with source attribution;
3. rejection of implicit box/water/salt/PME/cutoff/pressure/NPT/barostat fields;
4. exact ParmEd construction path, including a regression that distinguishes it from
   `AmberPrmtopFile.createSystem()`;
5. mbondi3 radii after `changeRadii` for ff19SB alanine and Sage/OpenFF RGDfV;
6. absence of periodic box, waters, ions, cutoff/PME, and barostat;
7. total and per-force energies for alanine and RGDfV against systems independently constructed by
   the pinned ParmEd reference path;
8. `tau=0` identity and several nonzero tau values;
9. scaling of the entire CustomGB energy, including the charge-independent term;
10. refusal of partial enhanced-region selections in implicit REST2;
11. omega exclusion affects only the intended torsions;
12. public-generator implicit min/NVT/cMD execution with real output reporters;
13. velocity initialization once and preservation across dynamic stages;
14. deterministic master/derived seed use;
15. exact exchange-step derivation from segment duration and exchange count;
16. implicit REST2 short CPU run, restart, extension, checkpoint preference, State fallback, and
    monotonically continued exchange state;
17. bundle relocation and installed-wheel execution outside the checkout;
18. both alanine and RGDfV public preparation/generation paths;
19. explicit-water unit/integration behavior remains passing;
20. unsupported advertised formats fail early and honestly.

Use tiny CPU systems/segments for normal CI. Mark expensive CUDA or scientific-duration runs
explicitly and do not run 10 ns merely to satisfy a software test. Where the machine has a validated
CUDA/OpenMM installation, run a very short optional GPU smoke for implicit conventional MD and
REST2 and record device, precision, command, and result. Never label CPU/mocked tests as CUDA
validation.

For energy comparisons, assign force groups or otherwise report independently calculated
per-force components. State units and justified tolerances. Compare both total and relevant
Nonbonded/CustomGB/torsion/CMAP components so compensating errors cannot pass.

Run, in order:

1. focused new regression tests;
2. all public-generator tests;
3. all OpenMM conventional-MD/REST2/restart tests;
4. complete non-slow suite;
5. configured CPU integration scripts;
6. build wheel and sdist, install the wheel into a clean environment/path outside the checkout,
   verify packaged profiles/examples, and run tiny implicit public commands;
7. committed golden checks;
8. optional real-CUDA smoke if the validated environment is available.

Report exact commands, versions, pass/fail/skip counts, and unrun checks. Fix real failures within
scope; do not weaken scientific assertions merely to make tests green.

## Phase 8 — documentation and journal

Update `README.md`, `docs/configuration.md`, example READMEs, CLI help, schemas/profiles, and package
data documentation. Explain:

- when explicit versus implicit solvent is appropriate;
- the exact GBn2/mbondi3 ParmEd construction path;
- why no box/salt/cutoff/NPT exists in implicit mode;
- that Amber topology files are provenance/intermediates for OpenMM, not an Amber engine;
- implicit REST2's initial full-solute-only restriction;
- tau-to-s scaling and full CustomGB scaling;
- segment duration versus Bash repetition versus committed runtime state;
- continuation and extension behavior;
- tested, conditionally tested, scientifically unvalidated, and deferred capabilities.

Add `docs/journal/2026-08-20_openmm-implicit-gbn2.md`. Record the request, branch integration SHAs,
reference commit, exact equations/construction path, files and persistent formats changed, test
commands/results, energy comparisons, any hardware smoke, limitations, and deferred work. Correct
stale contradictory claims in earlier active documentation, but do not rewrite historical evidence.

## Acceptance criteria

The task is complete only when all of the following are true:

- `dev` contains the prior public-generator branch before these implementation changes.
- Work was performed and pushed directly on `dev`; no new branch/PR was created.
- PR #3 and its branch were not used or modified.
- Both public generators have an honest, executed PDB and SMI path and refuse unimplemented routes.
- Declared cMD trajectories/logs are actually written.
- Velocities are initialized once and inherited thereafter.
- All runtime randomness comes from a persisted master/derived seed map.
- REST2 uses `duration_per_segment` plus `number_of_exchanges_per_segment`; exchange interval is
  derived exactly and segment count remains shell/runtime state.
- System manifests contain the actual resolved configuration and value sources.
- Implicit preparation uses the exact ParmEd GBn2/mbondi3/NoCutoff/HBonds path.
- Implicit stage graphs contain no NPT or barostat.
- Implicit conventional MD and full-solute REST2 execute and continue from committed state.
- Complete CustomGB energy scales by `s=(1-tau)^2` and `tau=0` is an identity.
- Alanine and RGDfV implicit examples generate successfully; short execution evidence is labelled
  honestly.
- Explicit-water behavior and the supported full test suite remain passing.
- Wheel/sdist contain the needed profiles, schemas, examples, and public entry points.
- Documentation and the new journal agree with executed evidence.
- No generated scientific output, secrets, environment files, or unrelated changes are committed.

## Final delivery

Use focused commits on `dev`, inspect every diff, push `dev`, and verify the remote SHA. Do not merge
to `main`.

The final Claude report must lead with the outcome and include:

- Phase 0 branch SHAs and confirmation that the generator branch was integrated first;
- commit list;
- explicit-generator defects fixed;
- implicit GBn2/mbondi3 design and exact pinned-reference match;
- configuration/persistent-format compatibility;
- exact tests and results, including energy-component comparisons;
- wheel/sdist and relocated-bundle evidence;
- CPU and any real-CUDA smoke results, clearly classified;
- known limitations, especially full-solute-only implicit REST2 and unimplemented formats;
- journal path;
- final remote `dev` SHA.

If an invariant cannot be preserved or the pinned ParmEd energy identity cannot be matched, stop
before publishing a misleading implementation. Report the smallest reproducible discrepancy with
the exact files, commands, versions, force components, and numerical differences.
