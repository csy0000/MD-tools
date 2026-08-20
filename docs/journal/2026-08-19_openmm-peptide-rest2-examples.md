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

---

# Addendum — HMR, CUDA execution, and measured throughput

Added after the entry above, at the user's direction. Everything here was measured on the
workstation's RTX 3080s, not estimated.

## Hydrogen-mass repartitioning at 4 fs

Both systems now repartition hydrogen mass to **3.024 amu** (solute scope) and integrate at **4 fs**,
across minimisation, NVT, NPT, cMD, REST2 and the extension. HMR did not need implementing —
`repartition_hydrogen_mass` already conserved mass with an assertion, refused to leave a heavy atom
under 1 amu, and never touched water. The examples had explicitly *disabled* it to follow the
instruction's "no HMR"; this re-enables it.

Every stage remains an exact whole number of steps, so nothing is rounded:

```
NVT / NPT   10 ps  ->     2,500 steps      cMD          1 ns  ->   250,000 steps
REST2 segment 5 ns -> 1,250,000 steps      exchange round     ->    12,500 steps (50 ps)
report full 100 ps ->    25,000 steps      report solute 10 ps->     2,500 steps
```

Confirmed on the real path: alanine repartitioned **12** hydrogens (it has exactly 12), RGDfV **38**,
both solute-scope with total mass conserved.

HMR is applied to the base System **before** tau scaling, so every replica has identical masses.
That matters for exchange validity: the acceptance criterion here is potential-energy-only, so
masses must not differ across the ladder. HMR changes the equations of motion, not the potential
energy surface — thermodynamic averages are unaffected, kinetic quantities are not comparable to an
unrepartitioned run.

## Three defects that only appeared when the code RAN

Configuration-level tests passed on all three. This is the entry's main lesson.

**1. OPC was accepted by the schema and refused at runtime.** OpenMM's `Modeller.addSolvent` ships
pre-equilibrated boxes only for tip3p/spce/tip4pew/tip5p/swm4ndp, so `model=opc` raised
`Unknown water model`. The alanine example validated and would have failed the moment anyone ran it.

Fixed properly rather than by retreating to TIP3P: the simulated water model is decided by the
**force field**, and OpenMM documents the route — "a box of TIP4P-Ew water can be used for most four
site water models". `solvation.resolve_packing_model` maps a model without its own box onto a
same-site-count stand-in (`opc -> tip4pew`, `opc3 -> tip3p`) and the bundle records both the
simulated and the packing model. Only same-site-count substitutions are declared; packing a
four-site model into a three-site box would leave its virtual sites unplaced, so anything undeclared
is refused rather than approximated.

**2. The multi-GPU device mapping was correct and unreachable.** `map_replicas_to_devices` had seven
passing tests and appeared nowhere in `src/` outside its own module. The CLI accepted only
`--device` (singular) and `_platform_and_properties` applied one `DeviceIndex` to every replica, so
every replica would have run on one GPU whatever the mapping said. The example scripts compounded it
by passing `--devices`, a flag that did not exist. Now wired through
`_platform_and_properties` -> `_make_simulation` -> `rest2` -> `runner` -> CLI, with `--device`
alone still covering every replica so existing behaviour is unchanged.

**3. Both example scripts would have failed.** `run_all.sh` passed `--run-name` on every segment,
but a fresh run correctly refuses to overwrite an existing directory, so segment 2 would have
stopped. `extend.sh` passed `--run-name` *and* `--resume-run`, which are mutually exclusive.

## The segment contract needed no rewiring

`runner`/`md`/`rest2` already treated `n_chunks` as "what THIS invocation adds", with the start taken
from the committed record, so the adapter's `n_chunks: 1` already means one segment per invocation.
Demonstrated on CPU rather than assumed — a fresh run plus two extensions produced:

```
attempt_index  0 1 2 3 4 5      no duplicates
step           500 .. 3000      monotonic
time_ps        2 .. 12          continuous across both boundaries
phase          0 1 0 1 0 1      alternation preserved
                                one CSV header, two committed generations
```

`tests/test_segment_extension.py` locks this in (instruction test items 13 and 15, plus the
fresh-run refusal).

## Measured REST2 throughput

4 fs with HMR, mixed precision, one timed 1 ns segment per replica, wall clock including setup and
exchange overhead.

| system | replicas | GPUs | particles | ns/day per replica | aggregate |
|---|---|---|---|---|---|
| alanine (ACE-ALA-NME, OPC) | 6 | 6 | 2,442 | **398.2** | 2,389 |
| cyclo-RGDfV (Sage/AM1-BCC, TIP3P) | 10 | 8 | 4,221 | **143.4** | 1,434 |
| cyclo-RGDfV | 10 | **5** | 4,221 | **143.4** | 1,434 |

**The 8-GPU and 5-GPU RGDfV results are identical to 0.04 %** (602.4 s vs 602.6 s wall). With 10
replicas on 8 devices, two devices carry two replicas and take ~2x as long per exchange round; every
replica waits at the round barrier, so the round costs whatever the slowest device costs — the same
as if every device were doubled. **Three of the eight GPUs contribute nothing to an RGDfV run.**

Use **5 GPUs for RGDfV**, not 8. Speeding it up genuinely would need one replica per device, i.e.
ten identical GPUs; the machine has eight 3080s plus a differently-architected A5000. Changing the
ladder to 8 rungs would be a scientific change to exchange spacing, not a scheduling decision.

Alanine cannot use eight GPUs either: its ladder is six replicas, so six devices are occupied and
two idle. REST2 parallelises across replicas, not within one.

Derived wall clock for the worked protocols, all replicas in parallel:

| | 10 ns protocol | + one 5 ns extension |
|---|---|---|
| alanine | 36 min | 54 min |
| RGDfV | 1 h 40 min | 2 h 30 min |

This corrects the earlier entry's expectation of a multi-day job. It is not one.

## Preparation cost

RGDfV's prepare took **27 m 41 s wall / 1300 CPU-minutes**, almost entirely AM1-BCC charge
derivation through `sqm`. That is a one-time per-system cost paid before any GPU work, and it is
why the SMILES route needed testing separately from alanine's PDB route.

The prepared RGDfV bundle independently confirms the vetted chemistry: **79 solute atoms** (matching
C26H38N8O7), net charge **4.6e-15** against a declared formal charge of 0, `protein_forcefield: None`.

## Environment note

The slow suite requires **`sqm` on PATH** (AmberTools), or the SMILES route cannot compute AM1-BCC
charges and `validate-env` fails. `environment-ci.yml` lists `ambertools` for exactly this reason.
In this session `sqm` came from the `ambertools26` conda environment while the tests ran under
`openfftools`.

Full suite after all of the above: **667 passed** (both markers, 10 min).
