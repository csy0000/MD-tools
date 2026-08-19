# 2026-08-19 — OpenMM peptide MD/REST2: segment contract, tau ladder, worked examples

Instruction: `claudecode-instructions/20260819_openmm-ala-rgdfv-rest2-protocol.md`
Base commit: `547147133ba5e42adc8d2d4ce041bce6abd1f09d` (verified to contain the instruction, and to
be a descendant of the expected `9e08dffd`).

## Status summary, read this first

| item | state |
|---|---|
| Phase 0 branch migration | **BLOCKED** — needs GitHub API access I do not have |
| segment contract replacing `n_chunks`/`chunk_ns` | implemented, tested |
| tau parameterisation of the REST2 ladder | implemented, tested |
| exact step / exchange arithmetic | implemented, tested |
| deterministic replica→device mapping | implemented, tested |
| alanine worked example (6 replicas) | generated, configuration validated, **not executed** |
| RGDfV worked example (10 replicas) | generated, configuration validated, **not executed** |
| extension examples | generated, **not executed** |
| runner/runstate wiring of the new contract | **deferred** — see "Not done" |

Nothing in this entry is a scientific result. No simulation was run.

## Phase 0: branch migration is blocked

The migration could not be performed and was **not forced or simulated**.

Verified state:

```
origin/openmm HEAD = 547147133ba5e42adc8d2d4ce041bce6abd1f09d   (matches the instruction)
9e08dffd is an ancestor of HEAD                                  yes
worktree clean                                                    yes (one stray fort.116 removed;
                                                                  it was my own artifact from an
                                                                  earlier `sander -h` probe)
```

What blocks it: GitHub's branch-rename is an API operation. On this machine `gh` is not installed,
the remote is SSH, no token is present in the environment or in a credential helper, and
`api.github.com/repos/csy0000/MD-templates` returns 404 unauthenticated because the repository is
private. Open pull requests and branch-protection settings could not be inspected for the same
reason, so item 3 of Phase 0 is also unverified.

Plain SSH pushes *could* create `main` and `dev` at the right commit, but that is not the instructed
operation: it would not retarget open pull requests, would not leave redirects, and cannot move the
default branch. The instruction says never to force the migration, so it was left undone.

**The action needed:** in the GitHub UI, Settings → Branches → rename `openmm` to `main`. That
performs the true rename, including moving the default branch. `dev` can then be created at the same
commit, and this feature branch retargeted.

Implementation continued regardless, on `feature/openmm-peptide-rest2-examples` created at exactly
the migration base `5471471`, so it is already correct for a `dev` created at that commit.

## Segment count leaves the scientific configuration

`protocol.production` now declares `duration_per_segment` and nothing about how many segments to
run. Three things that used to be conflated are now separate:

| | lives in | example |
|---|---|---|
| length of ONE segment | scientific JSON | `duration_per_segment: 5 ns` |
| how many segments to run | driver script | `NUMBER_OF_SEGMENTS=2` |
| how many committed | run manifest | `committed.json` |

Why it matters: when segment count was an input, asking for a longer run changed the configuration
hash, so an extended run looked like a different calculation.

A consequence worth recording: **`canonical.EXTENSION_ONLY` is now empty**. The only field that was
ever extension-only was `n_chunks`. Nothing remains in the canonical configuration that a user can
change and still resume — every remaining field is bundle-, continuity-, or execution-defining.
`duration_per_segment` *is* hashed, because it is the restart granularity and a silently changed
segment length would put two different segment lengths inside one run.

### Migration, not silent reinterpretation

`n_chunks`, `chunk_ns`, `chunk`, `scale_factors` and `exchange_interval` are each refused with the
exact replacement. The `scale_factors` message carries the conversion formula, because a user
holding an existing ladder needs `tau = 1 - sqrt(s)`, not just a new field name.

Legacy YAML manifests migrate through `spec/migrate.py`, which **refuses a ladder that is not linear
in tau** rather than respacing it — respacing changes exchange acceptance and therefore the run.

## The tau parameterisation reproduces the existing Hamiltonian exactly

`tau` is the source parameter and what gets persisted:

```
sqrt(s) = 1 - tau         solute-environment coupling
s       = (1 - tau)^2     solute-solute terms
          1               environment terms
```

`s` is computed as the square of the *same float* as `sqrt_s`, so `sqrt_s * sqrt_s == s` holds
exactly rather than approximately. `s` reaches the existing `build_rest2_scaled_system` unchanged,
so **no physics moved** — this is a reparameterisation of the input, not a new Hamiltonian.

### The existing validated ladder already was a tau ladder

Converting the shipped profiles showed the ten-rung ladder the project validated is exactly
`tau ∈ [0.0, 0.5]` with `count = 10`:

```
profile s        1.000000000000  0.891975308642  0.790123456790  ...  0.250000000000
tau-derived s    1.000000000000  0.891975308642  0.790123456790  ...  0.250000000000
worst absolute difference: 4.569e-13     (the residual is the profile's 12-decimal storage)
```

So the RGDfV ten-replica ladder is the already-validated ladder, not a new one. The alanine
six-replica ladder is `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]` → `s = [1.0, 0.81, 0.64, 0.49, 0.36, 0.25]`.

`s`, `sqrt(s)` and effective temperatures are recorded as **labelled derived diagnostics**, never
accepted back as input.

## Exact integer steps, or refusal

`segments.py` converts durations to whole steps and refuses to round, naming the nearest durations
that would work:

```
duration_per_segment            5 ns
timestep                        2 fs
steps per segment               2,500,000
number_of_exchanges_per_segment 100
steps per exchange round        25,000        (= 50 ps)
```

An exchange count that does not divide the segment is refused: a round landing mid-step drops or
duplicates an attempt across a boundary, and the committed watermark stops agreeing with the
exchange history.

## Scientific decisions and one documented conflict

### Water models are matched to their force fields

| system | solute route | water | box |
|---|---|---|---|
| ACE-ALA-NME | ff19SB | **OPC** (`amber19/opc.xml`) | dodecahedron |
| cyclo-RGDfV | **openff-2.2.0 + AM1-BCC** | **TIP3P** | dodecahedron |

ff19SB's backbone parameters were fit with OPC. Sage's vdW parameters were trained against
condensed-phase properties in TIP3P, and AM1-BCC charges were derived to be consistent with
TIP3P-era additive force fields. The pairings are not interchangeable.

OPC is a four-site model, so the alanine system's OpenMM particle count exceeds its topology atom
count. Anything indexing particles must use the resolved indices from the bundle.

### Conflict with the instruction, recorded rather than reconciled

The instruction's shared-preparation block specifies ff19SB, OPC water, and a truncated-octahedral
box. For cyclo-RGDfV the vetted manifest
(`src/md_templates/openmm/manifests/systems/cyclo_rgdfv.yaml`) specifies otherwise, and the vetted
definition was followed:

| | instruction | used | reason |
|---|---|---|---|
| solute FF | ff19SB | openff-2.2.0 + AM1-BCC | the manifest sets `protein_forcefield: null` so an accidental ff19SB load is an error, and states that an ff19SB run "would be a different calculation with the same name". The instruction scopes ff19SB to "where supported by the vetted topology route"; this route is a ligand parameterisation. |
| water | OPC | TIP3P | matched to Sage/AM1-BCC as above |
| box | truncated octahedron | dodecahedron | the manifest records that the ten-rung ladder was selected on a dodecahedron-prepared system; changing the box would detach the ladder from the evidence it was measured in |

The alanine box is a dodecahedron at the user's explicit direction (the instruction said
truncated octahedron).

This conflict was raised with the user before implementation and the resolution above was their
decision.

## Files changed

```
new   src/md_templates/openmm/tau.py                 tau -> s/sqrt(s), ladders, device mapping
new   src/md_templates/openmm/segments.py            exact step and exchange arithmetic
new   tests/test_tau_and_segments.py                 48 tests
new   tests/test_worked_examples.py                  46 tests
new   test/ala/REST2/{README.md,alanine_rest2.json,run_all.sh,.gitignore}
new   test/ala/REST2/extension/{README.md,extend.sh}
new   test/rgd/REST2/{README.md,rgdfv_rest2.json,run_all.sh,.gitignore}
new   test/rgd/REST2/extension/{README.md,extend.sh}
edit  src/md_templates/openmm/spec/models.py         SegmentedProduction, TauLadderSpec,
                                                     ExchangeSpec, SelectionSpec, OmegaExclusionSpec
edit  src/md_templates/openmm/spec/resolve.py        retired-input refusals with migrations
edit  src/md_templates/openmm/spec/adapter.py        segment/tau projection onto the runtime tree
edit  src/md_templates/openmm/spec/migrate.py        legacy manifests -> tau ladder, or refusal
edit  src/md_templates/openmm/spec/canonical.py      EXTENSION_ONLY emptied
edit  src/md_templates/openmm/spec/diffs.py          field explanations for the new contract
edit  src/md_templates/openmm/spec/profiles/*.json   5 profiles, schema 2
edit  scripts/capture_goldens.py                     capture tau and derived s
edit  tests/goldens/*.json                           4 regenerated (3 unchanged)
edit  tests/test_spec_config.py, test_compat_goldens.py, test_template_catalog.py
edit  CLAUDE.md, README.md, docs/configuration.md
```

### Schema and profile versions

`PROTOCOL_SCHEMA_VERSION` 1 → **2**. All five profiles `profile_schema_version` 1 → **2**.
`SYSTEM`, `BUILD`, `EXECUTION` and `RANDOMNESS` versions are unchanged — the change is confined to
the protocol.

Every profile keeps its previous segment length and exchange cadence **exactly**; only the count
moved out. `cpu-smoke-v1` and both `explicit-rest2-*` profiles converted with ladder residuals of
0.0 and 3.7e-13 respectively.

## Validation actually run

Environment: `openfftools` conda env — OpenMM 8.5.2, Python 3.12.13, pytest 9.1.1, pydantic 2.13.4,
numpy 2.5.2. (The repository's documented CI env pins OpenMM 8.5.1; this work used 8.5.2 as the
instruction requires. `pytest`, `python-build` and `setuptools` were added to that environment.)

```
baseline before this branch    527 passed, 17 deselected
after tau/segments module      575 passed
after the full spec change     577 passed
final                          623 passed, 17 deselected, 1 warning   (39.6 s)

python scripts/capture_goldens.py --check     exit 0, all 7 goldens ok
python -m build --wheel --no-isolation        Successfully built md_templates-0.1.0-py3-none-any.whl
```

Goldens: `configuration_hashes`, `format_equivalence`, `profiles` and `rest2_defaults` moved by
design. `bundle_contract`, `runstate_contract` and `seed_derivation` are **byte-identical**, which
is the intended blast radius — the bundle format and the restart contract did not change.

Example configurations resolve to exactly what their READMEs claim:

```
alanine  6 replicas   tau [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   s [1.0, 0.81, 0.64, 0.49, 0.36, 0.25]
RGDfV   10 replicas   tau [0.0 .. 0.5]                     s matches the validated ladder to 1e-12
both     5 ns segments, 100 exchanges, 2 fs, no HMR, dodecahedron, 0.15 M, PME 1.0 nm
```

## GPU mapping

Replicas are dealt round-robin over an ordered device list. Replicas may share a device when they
outnumber the devices — refusing that would make a ten-replica ladder impossible on a four-GPU
machine. The mapping is a pure function of `(n_replicas, device_indices)` and is persisted so that a
changed device list is visible rather than silently re-dealt. Duplicate device entries are refused.

The workstation has 9 GPUs, of which 8 are identical RTX 3080s and one is an RTX A5000. The examples
document selecting a homogeneous set and pinning `CUDA_DEVICE_ORDER=PCI_BUS_ID`; no run was
performed, so no mapping was exercised on real hardware.

## Not done, and why

**The runner and runstate layers were not rewired to the new contract.** The canonical spec layer —
which is the public scientific configuration — is fully converted, and `spec/adapter.py` projects it
onto the existing runtime dictionary (`n_chunks: 1` per invocation, derived exchange interval, tau
diagnostics). The runtime tree's internal `chunk_ns` naming is unchanged, which the adapter's
docstring already described as runtime convention. What this means concretely: **the segment/tau
contract is enforced at configuration time, but a full fresh-run-then-extend cycle through
`runner.py`/`runstate.py` has not been exercised against it.** Instruction test items 13–16
(fresh REST2 run then same-directory extension, checkpoint preference and State fallback, lifetime
exchange statistics across extension, incompatibility rejection before append) are therefore **not
implemented**. They need the runner work first, and asserting them against the adapter alone would
be testing the wrong layer.

**No simulation was executed.** Neither worked example was run — not the 1 ns cMD, not the 10 ns
REST2, not the extension. The instruction permitted execution "if the estimated runtime is
acceptable"; the full protocol is 6 and 10 replicas × 10 ns at 2 fs in explicit water, which is not
a session-length job, and the runner rewiring above is a prerequisite for the extension
demonstration in any case.

**Distinct minimisation/NVT/NPT schema blocks** were achieved through the existing
`EquilibrationSpec` (`minimize_max_iterations`, `nvt`, `npt`, `simple` protocol) rather than by
adding separate typed blocks with their own restraint sub-objects. The stages are validated and
distinct, and the examples exercise all three, but the restraint force constant and reference
coordinates are not yet first-class schema fields.

**Reporting field names** remain `all_atom`/`solute` rather than the instruction's
`full_system_interval_ps`/`selected_atoms_interval_ps`. The semantics are the required ones — two
streams at 100 ps and 10 ps — and the instruction permits schema improvement "retaining these
semantics"; renaming would have rippled through profiles, goldens and the bundle format for no
behavioural gain.

**Amber-mask selection through ParmEd** is accepted and validated in the schema
(`enhanced_region.type: amber_mask`) but is **not resolved to indices** — the resolver is not
implemented. A configuration using it will validate and then fail later. `solute` and explicit
`atom_indices` work.

**CI workflow files were not updated.** The new tests run in the default suite, but no workflow
change was made and no remote CI run was observed.

## Deferred

- rewire `runner.py`/`runstate.py` onto the segment contract; then implement instruction test items
  13–16 (extension continuity, checkpoint/State fallback, lifetime statistics, pre-append rejection)
- implement the ParmEd Amber-mask → index resolver, or reject the option until it exists
- first-class minimisation restraint block (force constant, reference coordinates) in the schema
- execute the worked protocols on the workstation and record concise summaries and hashes
- CPU smoke variants of each example for the default CI suite
