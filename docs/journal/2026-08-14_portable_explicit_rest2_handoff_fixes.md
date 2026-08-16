# Closing the portable REST2 handoff defects

**2026-08-14 (executed 2026-08-15).** Status: **all five defects closed and mechanically verified
on this machine.** Every command in the required verification sequence exited zero before this
journal was written.

Starting state: `40f222abb1dc81bffcc72fe2b14e841fe541130e` on `flatbottom-basin-cft-mixture`,
working tree dirty with 119 entries (unrelated talk/report work from earlier sessions, preserved).
Ending state: the commit that introduces this journal; 132 dirty entries before it, of which 13
belong to this task.

## Changed files

| file | why |
|---|---|
| `explicit/manifests/systems/cyclo_rgdfv.yaml` | box shape corrected to `dodecahedron`, with the evidence path and hash recorded inline |
| `explicit/manifests/experiments/rgd_rest2_10rung.yaml` | exact 12-digit ladder; `ladder_status: pilot_supported` |
| `explicit/manifests/systems/small_macrocycle_smoke.yaml` | reworded: the small cutoff is a speed choice, not the box-growth fix |
| `explicit/schemas.py` | `LADDER_STATUSES` vocabulary; `validated` no longer accepted |
| `explicit/fingerprint.py` *(new)* | build-defining projection and prepared-system fingerprint |
| `explicit/bundle.py` | records the projection + fingerprint and the box geometry; `validate-bundle` re-hashes the projection |
| `explicit/runner.py` | rejects an incompatible `--experiment` before any directory or context; exit code 8; records both fingerprints |
| `explicit/cli.py` | `EXIT_INCOMPATIBLE`; `validate-system` reports box shape and (with `--experiment`) ladder status |
| `systems/explicit_baseline.py` | `system_build.minimum_image_margin_nm`, the growth rule, and the recorded geometry fields |
| `scripts/config_defaults.json` | regenerated from the runtime tree |
| `tests/test_explicit_portable.py` | +54 tests (101 non-slow, 2 slow) |
| `docs/.../PORTABLE_REST2.md` | ladder vocabulary, box-shape evidence, override policy, margin rule |
| `reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/` *(new)* | the pilot evidence, preserved from a job temp directory |

## RGD box shape: proven, not assumed

The manifest said `cube`. The pilots did not run in a cube.

`raw_exchange/pilots6_command.sh` shows all six matched pilots copying `rgd_simbox.json`,
`rgd_system.xml` and `rgd_topology.pdb` from one Phase-A prepared system, so that system's geometry
*is* the geometry the ten-rung evidence was measured in. That file lived only in
`~/.claude/jobs/29daa285/tmp/20260814T1500_phaseA_cyclo_rgdfv/`, which is not durable
storage, and is now preserved in-repo:

```
reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/rgd_simbox.json
sha256 ea1c14edb7c9fc949d892fb41fb733750179555c235b3d6a0a54c7bda821ac61

  geometry.box_shape              dodecahedron
  geometry.box_width_nm           3.66182        grown_for_cutoff: false
  geometry.min_image_distance_nm  2.5893         at a 1.0 nm cutoff
  geometry.padding_nm_requested   1.2            solute_image_gap_nm 1.2
  realised_ionic_strength_molar   0.15903        Na+ 3 / Cl- 3, 1047 waters, 3226 particles
```

The accompanying `simbox-config.json` (sha256 `522cd046…`) records `box_shape: dodecahedron`
independently. **Resolved to `dodecahedron`** — this is direct evidence, not the fallback to the
baseline default the instruction allowed. `test_rgd_box_shape_matches_the_pilot_evidence` pins the
manifest to the artifact and fails if either drifts.

Everything else about the RGD identity is unchanged and still enforced: canonical hash
`59d4422635f7…`, formal charge 0, Sage 2.2 + AM1-BCC, TIP3P-FB water, `protein_forcefield: null`.

## Exact ten-rung ladder

Was truncated to six decimals. Now stored at full precision:

```
1.000000000000  0.891975308642  0.790123456790  0.694444444444  0.604938271605
0.521604938272  0.444444444444  0.373456790123  0.308641975309  0.250000000000
```

There is one definition — `rest2_ladder(1.0, 0.25, 10, "sqrt")` — and the manifest is a
serialisation of it, not a second hand-written list. Tests compare all ten values to that rule at
1e-12, assert the values are *not* representable at 6 dp, check YAML round-trip and wheel-resource
loading, and confirm that perturbing **any single** scale factor by 1e-9 changes the config hash.
Historical CSVs were not touched and no claim is made that the pilots ran the truncated numbers.

## Ladder-status vocabulary

`validated` was too broad — the predeclared conjunctive rule was not formally satisfied, because
condition 5 was under-specified. Two values now, and `validated` is not one of them:

* `unvalidated` — no system-specific pair-resolved evidence. **Absent resolves here.**
* `pilot_supported` — adopted as the working ladder on declared, system-specific pilot evidence.
  Explicitly **not** convergence, production readiness, or a formal pass of every predeclared
  condition.

RGD is `pilot_supported`; `macrocycle_pilot_8rung` and `smoke` stay `unvalidated`. Unknown values
fail validation — `validated`, `production_validated`, `converged` and `yes` are all tested to
raise. Run and bundle manifests carry the status verbatim, and `validate-system --experiment`
prints it with the caveat attached.

## Prepared-system fingerprint

`rest2 --bundle B --experiment E` could previously run a new experiment declaring a different force
field or box against an already-built System. The setting cannot take effect — `system.xml` and
`equilibrated_state.xml` were built under the old one — so the run would be *described* by one
configuration and *performed* under another.

`fingerprint.py` defines a **build-defining projection**: 66 dotted paths covering identity and
route, force field and charges, structure/protonation, solvation, `system_build` (PME, cutoff,
constraints, rigid water, HMR, the new margin), REST2 omega selection, the equilibration that
produced the stored state, and the integrator it ran under. 23 paths are explicitly runtime-only,
each with its reason recorded; a test asserts the two sets are disjoint.

The bundle stores the projection **and** its SHA-256. An override recomputes the projection and
compares **values**, not just the hash — so rehashing an edited manifest does not bypass it, and a
manifest whose fingerprint disagrees with its own projection is reported as tampered. Rejection
happens before the run directory or any OpenMM context exists, names the differing dotted paths
with both values, and exits `8`.

Accepted (tested): duration, chunk length, exchange interval, precision, production timestep,
reporting intervals, platform, device. Rejected (tested): force field, charge method, water model,
box shape, padding, salt, cutoff, PME method, constraints, rigid water, HMR, box margin, omega
selection and classification, equilibration protocol and length.

## Minimum-image margin

The growth policy enlarged an undersized box to *exactly* `2 * cutoff`, leaving it on OpenMM's hard
limit; the first NPT contraction then crossed it. Reducing the smoke's cutoff hid the symptom but
left every other preparation exposed.

`system_build.minimum_image_margin_nm` (default **0.10 nm**) now requires
`min_image >= 2 * cutoff + margin` **when growth is necessary**. A box already above the threshold
is left exactly as requested — the rule only ever grows. A negative margin is rejected before any
box is built; `0.0` restores the old behaviour deliberately; the `refuse` policy reports the
padding that satisfies the same margin. The cutoff, requested margin, required and actual image
distances, and whether growth occurred are recorded in the simbox and bundle manifests.

The margin is load-bearing, measured directly on the same geometry:

```
margin 0.00:  min image 1.800 nm -> ABORT: The periodic box size has decreased to less than
                                            twice the nonbonded cutoff
margin 0.10:  min image 1.900 nm -> 2 ps NPT OK
```

`test_grown_box_survives_npt_without_the_box_size_abort` is a real slow integration test: it forces
growth, solvates, and integrates 2 ps under a barostat on the CPU platform, then asserts the final
box still exceeds twice the cutoff. Its first version used a bare uncapped ALA and failed on a
missing ff19SB template — a defect in the fixture, not in the margin; the solute is now a single
water so the test is about box geometry and nothing else.

## Verification

Every command below was run in this order and exited zero **before** this journal was created.

| step | command | result | exit |
|---|---|---|---|
| A | `pytest test_rest2_scaler test_rest2_lambda_system test_explicit_baseline test_explicit_portable -v` | **179 passed** | 0 |
| B | `pytest -m "not slow" -q` | **873 passed, 3 skipped, 3 deselected** | 0 |
| C | `pytest tests/test_explicit_portable.py -m slow -v` | **2 passed** (CPU REST2 smoke; grown-box NPT) | 0 |
| D | `python -m build` + `zipfile -l` + `sha256sum` | wheel contains the CLI, all modules, all 5 manifests | 0 |
| E | wheel installed outside the checkout, `--help`, `validate-env`, `validate-system`, `smoke` | `status: completed`, **4/4 exchange rounds** | 0 |
| F | `git diff --check` | clean | 0 |

Wheel: `escort_ais-0.1.0-py3-none-any.whl`
sha256 `50e9a1a51ecbd3680d43985786813727186d534c486a9501ebd4895fe639a183`

Step E ran from `/tmp/escort-fix8-RSIhpz` with the package resolving to a venv site-packages
directory, `PYTHONPATH` unset, so nothing was importable from the source tree. `validate-system`
reported the exact identity, formal charge 0, `box_shape=dodecahedron`, and
`ladder_status=pilot_supported`. `validate-bundle` passed on the produced bundle, and the bundle
manifest carries the new fields: `minimum_image_margin_nm 0.1`, `minimum_image_required_nm 1.5`,
`min_image_distance_nm 1.87025`, `grown_for_cutoff false`, plus a 66-path prepared-system
projection whose fingerprint appears identically in `run_manifest.json`.

## Handoff package

```
dist/portable_rest2_handoff/
  escort_ais-0.1.0-py3-none-any.whl      50e9a1a51ecbd3680d43985786813727186d534c486a9501ebd4895fe639a183
  environment.yml                        593bac40be510105e1986f760289cdc1e8ef1e7228fee62f14b7e24f35b5c2c4
  cyclo_rgdfv.yaml                       ab6bd85ebf2cb1907c44b0a3449061b6a3fbe0107e01a4d313d10bcde195e789
  rgd_rest2_10rung.yaml                  a41a73be487d3f59b95b4d88f08016ab8488e2d8bc62b73668110ea6eb6201e7
  macrocycle_pilot_8rung.yaml            c9097bfb7fe94bcbb1800d5526c2882b444b034846be4c1ed7e6f177a0928be5
  small_macrocycle_smoke.yaml            91f8b59a86380eac67cf8e0a3bbf7a0cd076dc0bb49281c32d6999fdd925a04a
  smoke.yaml                             06dcdd5ee0e713625075668c9c5e3d80348785bb173e39773bcbec6aa7cfaf92
  HANDOFF_README.md                      6d0a749aa111795488a9b255eced6845701bfeb595c772b4e7f01f535be3db28
  SHA256SUMS
```

`SHA256SUMS` was generated from the copied files and verifies with `sha256sum -c`. `dist/` is
gitignored, so the wheel is not committed. No prepared RGD bundle is included: the only defensible
one is the historical Phase-A system, and rebuilding a bundle and presenting it as the pilots' own
would be a fabrication.

## What is and is not established

* **Source-machine portability is mechanically verified.** A wheel built here, installed where the
  source tree is unreachable, prepares and runs from a temporary directory and produces a complete
  run directory.
* **Target-machine validation remains to be run there.** Nothing in this task touched another
  machine. `HANDOFF_README.md` makes the CPU and CUDA `validate-env` and the CPU smoke a required
  first step on arrival.
* **Not established by this task:** restart safety; the 2 fs / 4 fs hydrogen-mass-repartitioning
  equivalence; convergence of anything; readiness for long production. The RGD ladder is
  `pilot_supported`, which is deliberately weaker than validated, and no production run was
  started.
