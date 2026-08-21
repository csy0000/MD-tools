# Closing the remaining cMD acceptance gaps

**2026-08-21.** Four items raised in review of
[`2026-08-21_cmd-review-fixes-and-opc-default.md`](2026-08-21_cmd-review-fixes-and-opc-default.md),
plus the validation and exact-SHA CI evidence for them.

Starting SHA `ace55b1` (= `1bc29cd` plus the instruction). Final SHA recorded under **Remote CI**.

## 1. The water default follows the force field, not the input label

**User-approved supersession.** The earlier instruction required OPC for every explicit route. That
requirement is superseded on this one decision, and only this one; nothing else in
`20260821_cmd-review-fixes-and-opc-default.md` is reopened. The older instruction file is left as
written and cross-linked from here.

A protein force field and a small-molecule force field are each fitted against water, and not the
same water, so one default mispairs one of them by construction. The choice is now derived from the
**resolved force-field family** rather than from `peptide`/`ligand`: the label says which reader
parsed the input, the force-field fields say which parameters get assigned, and it is the parameters
that were fitted. A complex carries both, so a label-driven rule has no answer for it at all.

| family | fitting water | source |
|---|---|---|
| ff19SB | **OPC** | CMAPs trained against QM surfaces *in solution*; with TIP3P the helical propensities they exist to reproduce come out wrong. Tian *et al.*, *JCTC* 2020, 16, 528–552, [doi:10.1021/acs.jctc.9b00591](https://doi.org/10.1021/acs.jctc.9b00591) |
| ff14SB | TIP3P | developed and validated in TIP3P. Maier *et al.*, *JCTC* 2015, 11, 3696–3713, [doi:10.1021/acs.jctc.5b00255](https://doi.org/10.1021/acs.jctc.5b00255) |
| OpenFF Sage (openff-2.x) | **plain TIP3P** | LJ refit trained against condensed-phase properties in TIP3P; openff-forcefields ships `tip3p.offxml` with each release. Boothroyd *et al.*, *JCTC* 2023, 19, 3251–3275, [doi:10.1021/acs.jctc.3c00039](https://doi.org/10.1021/acs.jctc.3c00039) |

giving `explicit-*-peptide-v2` → OPC, `explicit-*-ligand-v2` → plain TIP3P (**not** TIP3P-FB, which
is a separate ForceBalance refit and is what the `-v1` profiles kept), and an ff19SB + Sage complex
→ OPC, recorded as a **mixed-force-field compatibility choice** rather than a validated pairing.

Matching on the family rather than an exact resource string means `leaprc.protein.ff19SB` and
`amber19/protein.ff19SB.xml` — the tleap and OpenMM spellings of one choice — agree.

An **unrecognised** force field is refused, naming `forcefield.water` and `solvation.water_model` as
the way to state the pairing intended; a half-recognised complex still resolves from the recognised
half, so an unenumerated small molecule does not block a build whose protein force field is known.
Explicit overrides win and are recorded as `user input`.

**Ions.** Joung–Cheatham sets are fitted per water model and OpenMM ships them *inside* each water
force-field file, so they follow the water with no separate choice and no way to drift. Measured
through `createSystem`, Na⁺ ε is 0.124 kJ/mol with OPC against 0.366 with TIP3P — a factor of three,
which is why `water_policy.ion_parameters.source` is worth recording.

## 2. The continuity identity is complete; on-disk schema is 3

Version 2 hashed `system.xml` and the topology but recorded the predecessor State by **pathname**
and the bundle by a partial parsed projection. A pathname can be repointed at a different
equilibration, and a projection cannot notice a change to a field it does not name — and every field
it does not name is still part of the Hamiltonian.

Version 3 binds by exact bytes: the predecessor State (with declared path and producer), the
`system_manifest.json` and its recorded configuration hash, and `forcefield.json`. Added alongside:
ordered atom identity per output stream, and restraint active/selection beside the convention. One
deterministic hash covers the canonical whole. A deleted predecessor records as `"absent"` rather
than null, so deletion is a visible change rather than something that compares equal to a run that
never had one.

**Schema compatibility.** `CMD_RUN_STATE_VERSION` is 3. A version-2 record is **refused** with
regeneration guidance, not reinterpreted: it never carried the new identities, so treating their
absence as "unchanged" would silently skip the checks the bump adds. The message states that the
committed physics is not lost — continue under the build that wrote it, or regenerate and start
fresh, with existing trajectories still valid as the record of what was committed.

## 3. The restart is verified before anything about it is written

The continuation did this:

```python
sim.currentStep = int(continuation["absolute_step"])
verify_restart_matches_commit(sim, ...)
```

`Simulation.currentStep` is a **property** over `Context.getStepCount()` in OpenMM 8.5.2, so the
assignment called `Context.setStepCount` and overwrote the value the next line was about to check.
The verification compared the commit against itself and could never fail; a checkpoint from the
wrong generation loaded cleanly and was accepted. The position is now captured first, verified, and
only then restored.

**A correction to the instruction's premise, measured rather than argued.** The instruction states
that an OpenMM State does not carry `currentStep`. That was true of older OpenMM, where
`Simulation.currentStep` was a plain attribute. In 8.5.2 `State` exposes `getStepCount()` and
`Context.setState` restores it — asserted directly in
`test_both_restart_forms_carry_step_and_time_in_this_openmm`. The step is therefore verified on both
paths. The commit is consulted for it only when a restart genuinely reports none, and that origin is
recorded explicitly rather than assumed. Time is always authoritative, which is what makes trusting
the commit for a missing step safe. State fallback is announced and recorded as a non-bitwise
stochastic continuation; checkpoint preference is unchanged.

## 4. The five commit boundaries are interrupted deliberately

A timed `kill -9` proves some interruption is survivable but cannot say which boundary it hit.
`faults.crash_point` makes the boundary selectable from `MD_TEMPLATES_CRASH_AT` and is inert
otherwise — unset, it is one `os.environ.get`. It uses `os._exit`, so no `finally` block, `atexit`
handler or buffer flush runs: the conditions a real kill produces.

| # | boundary | what it leaves |
|---|---|---|
| 1 | before reporters close | trajectory frames past the last commit |
| 2 | after reporters close, before restart save | streams ending exactly at the boundary, nothing saved |
| 3 | after the checkpoint member, before the State | half a restart pair |
| 4 | after both members, before atomic commit | a complete restart pair no commit points at |
| 5 | after atomic commit, before cache reconciliation | a commit whose `run_state.json` is stale |

Each test asserts the boundary's *distinguishing* on-disk state — the tail exists, the pair is half
written, the cache is behind — so a hook that failed to fire would fail the test rather than pass it
vacuously. Boundaries 1–4 keep generation 1 authoritative; retry adds exactly one generation with
monotonic steps, frames, log rows and invocation history. Boundary 5 retains the new generation and
rebuilds the cache from `committed.json`. The exclusive run-directory lock and the existing
timing-based kill test are kept as additional coverage.

## Requirement → implementation → test → evidence

| # | requirement | implementation | test | evidence |
|---|---|---|---|---|
| 1 | force-field-dependent water | `water_policy.resolve_default_water`, wired after the forcefield merge in `system_prep` | `test_water_policy.py` (28) | four v2 defaults, complex rule, overrides, v1 compatibility, implicit absence, ion provenance |
| 2 | complete continuity identity | `_cmd_continuity` v3, `assert_supported_cmd_schema` | `test_continuity_identity.py` (24) | all eight refusals, each asserting outputs byte-unchanged and no invocation counted |
| 3 | verify before mutate | `read_restart_position` / `verify_restart_matches_commit` / `restore_restart_step` | `test_restart_verification.py` (9) | stale checkpoint refused, wrong-time State refused, fallback provenance exact |
| 4 | deterministic crash boundaries | `faults.crash_point` at five sites | `test_crash_boundaries.py` (15) | each boundary's distinguishing state asserted, then recovery |

## Gates

| gate | command | result |
|---|---|---|
| focused + subsystem | `pytest` over water/continuity/restart/crash/DCD/cMD/implicit/REST2/profile/generator tests | **373 passed** |
| complete non-slow | `pytest -m "not slow"` | **840 passed, 163 deselected**, 90.07 s |
| complete suite | `pytest` | **1003 passed, 0 failed, 0 error, 0 skipped**, 3 warnings, 1632.65 s (27:12) |
| fast CI | `scripts/ci/fast_checks.sh` | **PASSED** — wheel + sdist, packaged resources, installed-wheel commands outside the checkout with no source `PYTHONPATH` |
| CPU integration | `scripts/ci/integration_cpu.sh` | **PASSED**, 13 steps — relocation with sources deleted, offline validation, chunked and staged resume, State fallback, tail recovery |
| artifact scan | `git ls-files` | clean: **2.2 MiB across 215 files**, no trajectory, checkpoint, State, environment, secret or large log |

## Two pre-existing defects fixed to reach those results

**A flaky hand-off test.** `test_a_stage_consumes_its_predecessors_endpoint` compared min's final
potential against eq_nvt's starting potential at 1e-4 relative tolerance (~3.7 kJ/mol). The two
cannot be equal: min's potential includes its restraint term and the solute has moved off the
reference positions by then, while eq_nvt restrains to the coordinates it just loaded, so its
restraint term starts at zero. The gap *is* min's final restraint energy, and how far minimisation
travels is not reproducible because L-BFGS on the threaded CPU platform sums in a nondeterministic
order. Measured across runs the gap ranged 0.8–4.8 kJ/mol against a 3.7 threshold.

Confirmed pre-existing rather than caused here: the bundle this test builds is **byte-identical**
to the one `ace55b1` builds (`system.xml` and `topology.pdb`), and nothing in these changes touches
the min or eq_nvt path. The tolerance is now absolute and bounds the restraint term, leaving the
real failure two orders of magnitude outside it — a dropped hand-off restarts from pre-minimisation
coordinates, ~4,000 kJ/mol away — with the direction pinned, since releasing a restraint can only
lower the reported potential. Stable over six consecutive runs.

**A schema literal in CI.** `integration_cpu.sh` step 11 asserted `cmd_schema_version == 2` as a
literal, so a deliberate bump broke the gate meant to be checking the schema. It now compares
against `CMD_RUN_STATE_VERSION`.

## Final CUDA validation

Regenerated from scratch — bundles, projects and runs — under the final code, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` throughout. All GPUs idle beforehand; the two runs used separate
devices.

| | explicit | implicit |
|---|---|---|
| GPU | index 0, RTX A5000, `GPU-7a14ba65-b0a3-66bd-536d-881e08b55da1` | index 1, RTX 3080, `GPU-96ce533d-9d42-acf6-8384-5e27150e9a85` |
| force field / water | ff19SB / **OPC**, basis recorded as `ff19SB` | ff19SB / GBn2 / mbondi3, no water |
| stages | min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1 | min, eq, cMD_1 |
| production | 2 × 125,000 steps | 2 × 250,000 steps |
| absolute step / time | 250,000 / 1000.0 ps | 500,000 / 1000.0 ps |
| committed generations | 2 | 2 |
| cmd schema / contract version | 3 / 3 | 3 / 3 |
| restart source, segment 2 | checkpoint, bitwise | checkpoint, bitwise |
| step origin | `restart (checkpoint)` | `restart (checkpoint)` |
| predecessor / manifest / forcefield bound by bytes | yes | yes |
| periodic | true | **false** |
| all-atom / selected frames | 10 (2438 atoms) / 100 (22 atoms) | 10 (22 atoms) / 100 (22 atoms) |
| log header / rows | 1 / 10, monotonic | 1 / 10, monotonic |
| NaN or Inf | none | none |

The `step origin` row is the item-3 fix observed end to end: a healthy checkpoint supplies its own
step, and nothing writes over it before the check.

```
explicit  cMD_1_all_atoms.dcd       61622e004f8d88ce571967c3fa02718d494d8e920017019d64e3a993858f03c2
explicit  cMD_1_selected_atoms.dcd  15c729e37ffa76b86d6e29595142f68a0b169a8df3e9c2385ba330cc5acf5e31
explicit  cMD_1.log                 8fee9432676f08ec4ace49b28f275cb39abed33a39c85a6ad3dc50a89d2a8ff3
implicit  cMD_1_all_atoms.dcd       ce700cb1a3f5be4e126971ddf98710ab33bd33edf41a828c12bb220efcc487b0
implicit  cMD_1_selected_atoms.dcd  b4b4fdc35ac4500593563ae09d2f64597542dac5253c044892e6d3af2df11b3d
implicit  cMD_1.log                 be8ff48f7145ea9002cc356d6fad4eed23dbb281b488aad0410eac2216213a2b
```

Driver 580.173.02, OpenMM 8.5.2, ParmEd 4.3.1, Python 3.12.13, CUDA mixed precision. Runs live under
`MD-test/cmd-validation-20260821-gaps/`, outside the repository; nothing generated is committed.

## Limitations

* Bundle reproducibility is *same machine, same library versions*. A different OpenMM release may
  place hydrogens differently or ship a different water box.
* The fault-injection hooks model a killed process. They do not model a torn write inside a single
  `os.replace`, which the filesystem is relied on to make atomic.
* `ff14SB → TIP3P` is implemented and tested but is not exercised by any shipped profile; no run in
  this repository uses ff14SB.
* The complex rule is tested at the policy level and through provenance. No protein–ligand complex
  is built end to end, because the `complex` construction route is still unimplemented.
* Item 4's boundaries cover the segment commit. Interruption *during* equilibration stages is
  covered only by the existing single-shot rerun behaviour.
