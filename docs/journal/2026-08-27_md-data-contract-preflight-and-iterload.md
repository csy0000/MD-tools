# 2026-08-27 — the MD-data contract, automatic preflight, and streamed AIS sources

Instruction: `claudecode-instructions/20260827_md-data-contract-preflight-and-iterload.md`.
Branch `dev`. No merge to `main`, no tag.

## What this closed

Three separate gaps, which share a shape: each one let a run get further than it should have
before anything noticed.

1. **Nothing connected a generated project to MD-data's dataset contract.** A project could be
   written into `$MD_DATA` in a layout MD-data would later reject, and nobody would find out until
   registration.
2. **Nothing was checked before a Context existed.** A run could allocate a GPU, build a System,
   fork REST2 workers and start writing a trajectory, and only then discover that its parent stage
   was inconsistent or its prepared inputs had changed underneath it.
3. **AIS read its source trajectory with `mdtraj.load`.** Peak memory scaled with the length of
   the source. And a finished path was judged complete from a JSON record alone, so a truncated
   `observations.dcd` beside a healthy JSON and CSV was silently skipped as done.

## 1. The dataset contract, from MD-data and not from here

`src/md_templates/openmm/md_data_contract.py` assembles a manifest and hands it to **MD-data's own
validator** — `md_data.validate_dataset` and `md_data.storage.check_dataset_tree`, imported. There
is no copy of the schema in this repository, no second implementation of the rules, and no field
this repository invented. That was the whole design decision: two repositories that each believe
they implement the same contract will drift, and the drift is invisible until someone's data is
already on disk.

Off by default. `dataset.enabled: true` in `sys.config.yaml` turns it on.

**Layout.** `{namespace}/{yyyy-mm}/{dataset_name}` relative to `$MD_DATA`. Components are top level
and a component's `path` equals its `name`. `eq/nvt_1kcal` is a **stage inside the `eq`
component**, not a component — the stage chain and the component list are different things and
conflating them would have produced a manifest MD-data rejects.

**The two variables.** `MD_DATA` is the storage root; `MD_DATA_LOCAL` is the one dataset directory
an invocation may write into. Refused: a symlink (aliases are MD-data's to make), a path escaping
`MD_DATA`, a path that is not three segments, a middle segment that is not `yyyy-mm`, and a status
that is not `active`.

**The identity is declared once**, in `sys.config.yaml`. `md-gen` reads it back out of
`common/resolved_sys.config.yaml`. Nothing is retyped, so nothing can disagree.

**What the user must supply.** `dataset_id`, `namespace`, `dataset_name`, `role`, `system`,
`created_by.person_id`/`.name`, `origin.repository`/`.commit`, `templates.commit`. Commits must
match `^[0-9a-f]{40}$` — never fabricated, never abbreviated, never resolved from `HEAD` on the
user's behalf. A dataset ID this package derived from a path stops being stable the moment the path
changes; a person is a scientific identity, not the account the job ran under. `md-openmm
show-default dataset` prints the block with every required field `null`.

Missing fields stop `sys-gen` **before the system is built**, not after — `_plan_dataset` runs
first, so a configuration error does not cost a solvation.

**`$MD_DATA`'s value is never written into a generated file**, so moving the tree and re-pointing
the variable is enough. `provenance.yaml` is the single exception, and only because it records the
command line verbatim: the `-of` the user typed was absolute, and rewriting it to look relative
would falsify the one record whose job is to say what happened.

### A documented rule that was wrong

`CLAUDE.md`, `README.md` and `docs/FAIR_HANDOFF.md` all said this repository never writes into
`$MD_DATA`. That is not true and cannot be, because a run's outputs have to land where the next
stage reads them — the stage chain's dependency is on disk. The accurate boundary is now written
in all three:

> MD-templates may write generated files and a contract-valid `dataset.yaml` into the **one**
> active dataset directory the user explicitly selected. MD-data owns the schema, the validator,
> identity, the `active -> complete -> archived` lifecycle, the catalogue, aliases, extensions,
> archival checksums and retention.

Everything that was true stays true: no dataset ID is minted here, `$MD_DATA` is never walked or
hashed, no second dataset is touched, no lifecycle status is changed, and a `complete` or
`archived` dataset is refused rather than quietly reopened.

## 2. Preflight

`src/md_templates/openmm/templates/preflight.py` is copied into every generated project and run by
every launcher — `run.sh`, `run_all.sh`, the REST2 workers, the AIS paths — **before any OpenMM
Context, integrator, worker process, checkpoint or trajectory exists**. There is no flag to skip it.

`./run_all.sh --check` and `./run.sh --check` run the identical gate and stop.

Checks: `MD_DATA`/`MD_DATA_LOCAL`; the running project is the selected dataset; MD-data validation
with the on-disk tree; status `active` and writable; this component declared, owned, active and not
linked; `common/` present with matching `SHA256SUMS`; `forcefield.json` agreeing with
`resolved_sys.config.yaml`; the platform; the parent stage; this stage's own completion record.

Three decisions worth recording:

**A CUDA platform with zero visible devices now FAILS.** With `CUDA_VISIBLE_DEVICES=""` OpenMM
still *lists* a CUDA platform, and the previous check accepted that — the run then died at Context
creation, after doing all the setup work. Listing a platform is not having a device.

**`--check` writes nothing at all, including `run.log`.** `stage_run.sh` now `exec`s past the
`tee` when `--check` is present. A check that leaves a trace in the record of what ran is not the
non-destructive thing it claims to be.

**A parent that has not run yet is `skip` under `--check`, `FAIL` for a real run.** Otherwise
`--check` on a fresh project fails everything downstream of stage 0 for the wrong reason. An
*inconsistent* parent — a `final_state.xml` with no `resolved_stage.yaml` — fails in both modes.

**Bounded by construction.** Preflight reads a fixed list of named files. It never walks
`$MD_DATA`, never enumerates datasets, never opens a trajectory and never hashes one. A preflight
whose cost grows with the size of the archive is a preflight people learn to disable, so a test
greps the shipped `preflight.py` for `rglob`, `os.walk`, `glob.glob`, `iterdir` and `.dcd` and
fails if any appear.

## 3. AIS sources

**Streamed.** `mdtraj.iterload` in bounded 50-frame chunks, in two passes: survey (count and locate
the frames the time window selects, retaining none) and collect (only the selected frames). Peak
memory is set by the chunk size, not by the length of the source, so a nanosecond source and a
microsecond source cost the same. `mdtraj.load` is never called on a source trajectory.

**Tau is evidence, not an assumption.** A source produced here carries a companion record and its
tau is read from there. A source from anywhere else must state `AIS.source.source_tau`. Refused:
absence; a value conflicting with a companion record (both values named, and the explicit one is
never silently preferred); and a value that is not `path.tau_start` — annealing from a state
sampled at a different Hamiltonian is not the calculation the work values would be read as.

**Completion is agreement, not a flag.** A finished path is skipped on rerun only if the completion
record says so, its trajectory index matches, the observation count matches, the CSV exists with
the right number of rows, `observations.dcd` exists, and its frame count equals the CSV row count
one-to-one. The DCD frame count comes from a 100-byte header read, not from a reader — OpenMM has
no DCD reader and `openmm.app.DCDFile` is write-only.

## Evidence

Environment: `/path/to/software/md-stack/envs/openmm-8.6.0` — conda-forge `openmm
8.6.0=py312hdfcc665_0` (the release; its Python string reads `8.6.0.dev-c6173db`, and the package
identity is what is authoritative), `cuda-version 13.0`, `mdtraj 1.11.1`, `md-data 0.2.0`,
`openff-toolkit 0.19.0`, Python 3.12.14.

MD-data contract: `csy0000/MD-data` at `48628f9a5d3ace6c6398a63bc3905cd58d542de3`,
`docs/contracts/dataset-v1.md`. Validator: `md-data 0.2.0`, contract v1.0, supported schema
versions `["1.0"]`.

CUDA: 9 devices — device 0 NVIDIA RTX A5000 (24564 MiB), devices 1–8 NVIDIA GeForce RTX 3080
(10240 MiB each), driver 580.173.02.

### The generated tree

```text
$MD_DATA/adenosine/2026-08/ala-ais-demo/
├── dataset.yaml         5 components: common, minimization, eq, cMD, AIS
├── common/              eq/nvt_1kcal, eq/npt_1kcal, eq/npt_free are STAGES inside eq
├── minimization/  eq/  cMD/  AIS/
```

`sys-gen` reported `manifest : dataset.yaml validated by md-data 0.2.0 (contract v1.0)`.

### `--check` created nothing

`./run_all.sh --check` over all five stages: `11/11 ok` each, `[stage] --check: preflight only. No
dynamics ran and nothing was written.` The file set before and after was byte-identical, and a find
for `*.dcd`, `*.chk`, `final_state.xml`, `resolved_stage.yaml`, `resolved_run.yaml`, `*.csv` and
`run.log` returned **0**.

### Preflight refusals, all before a Context

| what was broken | what preflight said |
|---|---|
| `owner: nobody` added to the manifest | `[FAIL] md-data validator  $.owner: unknown key; this contract forbids undeclared fields so that a typo cannot silently become metadata` |
| `status: complete` (consistently) | `[FAIL] dataset status  'complete': a complete or archived dataset is read-only, and new output belongs in a new dated dataset` |
| `cMD` removed from `components` | `[FAIL] component  'cMD' is not declared in dataset.yaml (declared: AIS, common, eq, minimization)` |
| `CUDA_VISIBLE_DEVICES=""` | `[FAIL] platform  OpenMM offers a CUDA platform but this process can see no CUDA device` |
| one byte appended to `common/solute.yaml` | `[FAIL] input checksums  solute.yaml changed since it was prepared` |
| `resolved_stage.yaml` deleted after a run | `[FAIL] completion record  final_state.xml exists but resolved_stage.yaml does not, so this stage did not finish cleanly` |

Every one ended `preflight failed: N check(s). Nothing was run, no Context was created, and no
output was written.`

### `iterload`

From the AIS preflight on a 300-frame source:

```text
[PASS] AIS source     whole_system.dcd: 300 frame(s), 193 atoms, read with mdtraj.iterload in 6 chunk(s) of 50
[PASS] AIS frame timing   companion_record: 0.002 + k * 0.002 ps
[PASS] AIS source tau     tau = 0.5 from companion record, matches path.tau_start
[PASS] AIS selection      3 path(s) from 300 eligible frame(s) in 0.002-0.6 ps inclusive; frames [226, 26, 109]
```

Six chunks for 300 frames, and 3 frames retained out of 300. The test fixture uses a 120-frame
source read in 3 chunks, so a one-chunk implementation cannot pass it by accident. The test also
replaces `mdtraj.load` with a function that raises, via a `sitecustomize` on `PYTHONPATH`, and
asserts a sentinel file proving the guard actually installed — otherwise a guard that silently
failed to install would look exactly like a guard that was never triggered.

### Force-field records

Peptide (default, no `--solvent`), from `forcefield.json`:

```json
"protein": {"openmm_resource": "amber14-all.xml",
            "openmm_resource_includes": ["amber14/protein.ff14SB.xml", ...]},
"water":   {"openmm_resource": "amber14/tip3p.xml", "requested_label": "TIP3P"},
"ligand":  null
```

`ligand.openff_resource` is `null` and is asserted to be — **Sage did not participate in a
peptide-only system**, and the record says so rather than listing a force field that touched
nothing.

Ligand (ethanol, `CCO`): `ligand.openff_resource == "openff-2.2.1"`, `requested_label ==
"sage-2.2.1"`, `water.openmm_resource == "amber14/tip3p.xml"`, `protein.openmm_resource` `null`.
A test asserts `"amber19" not in json.dumps(record)` for both, so no CUDA acceptance in this task
used the optional ff19SB + OPC combination. That combination remains available via `--solvent OPC`
and is exercised by `test_scientific_defaults.py`, which builds no System.

### AIS frame and work counts

```text
trajectory_0000: dcd=21 csv=21  tau 0.5->0.0  W0=0.0  Wend=-91.2282 kJ/mol
trajectory_0001: dcd=21 csv=21  tau 0.5->0.0  W0=0.0  Wend=-94.8654 kJ/mol
trajectory_0002: dcd=21 csv=21  tau 0.5->0.0  W0=0.0  Wend=-110.4757 kJ/mol
```

21 DCD frames to 21 work rows, one-to-one, on every path. 40 switching updates over 40 steps, 0
barostats in the switching System (fixed volume, so no pV work), distinct integrator and velocity
seeds per path.

Rerun with everything intact:

```text
[AIS 0000] already complete: 21 observations, 21 DCD frame(s), endpoints 0.5 -> 0.0. Nothing was run.
```

Rerun after rewriting path 0000's DCD header to claim 7 frames:

```text
[AIS 0000] replacing an incomplete directory (observations.dcd holds 7 frame(s), 21 expected -- truncated, not short)
[AIS 0000] complete: 21 observations, 40 integration steps
[AIS 0001] already complete: ...
```

The truncated path was rerun; the two intact paths were not. The same holds for a deleted DCD
(`observations.dcd is missing`). Both cases have a healthy JSON and a healthy 21-row CSV, which is
exactly the state that was previously accepted as complete.

## Tests

`tests/test_md_data_contract.py`, 47 tests. Full suite: **301 tests**, run on CUDA.

Two pre-existing tests were corrected rather than worked around:

* `test_the_forcefield_record_is_written` still asserted `amber19-all.xml` + `amber19/opc.xml`.
  That expectation went stale when the default became ff14SB + TIP3P; it now asserts the current
  default pairing and that `"amber19"` appears nowhere in the record.
* `test_a_stage_with_only_half_its_completion_artifacts_refuses_to_run` asserted the phrase
  "refusing to run". The refusal now comes from preflight — earlier, and before a Context — so the
  test asserts the invariant instead: it stops, it names `resolved_stage.yaml`, it says no Context
  was created, and it wrote no record.

## Limitations

* **`iterload` bounds retention, not RSS.** The two-pass design guarantees that at most one chunk
  plus the selected frames is *held*, and the chunk size is fixed at 50. It does not measure peak
  process memory, and mdtraj's own buffering inside a chunk is not instrumented. The claim is
  bounded retention, which is what the code controls.
* **The DCD frame count is a header read.** It detects a truncated or absent file and a
  frame-count disagreement. It does not detect a DCD whose header is right and whose coordinates
  are corrupt; nothing short of reading and checking every frame would, and that would put a
  trajectory read into preflight, which is exactly what was ruled out.
* **`md-data` is an optional dependency.** Without it installed, `dataset.enabled: true` fails with
  install instructions rather than falling back — a manifest written without the authoritative
  validator would be the one thing this design exists to prevent. Unregistered projects
  (`enabled: false`, the default) do not need it.
* **The demonstration run is a smoke.** Picoseconds, a 0.5 nm box, 40 switching steps. It exercises
  the plumbing, the records and the refusals. It is not a converged free-energy estimate and the
  work values above should not be read as one.
* **Only `origin` and `templates` commits are pinned.** The contract has no field for the
  environment, so the exact OpenMM/CUDA build that produced a dataset lives in `provenance.yaml`,
  not in `dataset.yaml`. Anyone reading the manifest alone cannot tell which OpenMM ran.
