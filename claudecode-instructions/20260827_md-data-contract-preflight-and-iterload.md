# MD-data contract integration, execution preflight, and streamed AIS sources

## Purpose

Make newly generated and executed OpenMM projects conform to the published MD-data dataset v1
contract, add a real preflight before any dynamics begins, stream AIS source trajectories with
`mdtraj.iterload`, and correct the acceptance fixture so dynamics tests use the current
method-development default rather than a crossed or obsolete force-field/water combination.

Work on the current `dev` branch. Read `CLAUDE.md`, the current implementation, the most recent
journal, and the authoritative MD-data contract before editing:

- https://github.com/csy0000/MD-data/blob/dev/docs/contracts/dataset-v1.md
- https://github.com/csy0000/MD-data/blob/dev/docs/storage-layout.md
- https://github.com/csy0000/MD-data/blob/dev/README.md
- MD-data contract commit inspected for this task:
  `2c013482728d77abe792da8761ae193e23fa99ff`

MD-data owns the contract and validator. MD-templates owns simulation construction, execution, and
generation provenance. Do not copy the Pydantic models or invent a competing dataset schema in this
repository.

Keep the six public commands and the two canonical generators:

- `src/md_templates/openmm/sysgen.py::generate_system()`
- `src/md_templates/openmm/mdgen.py::generate_md()`

Do not restore the deleted framework, add a seventh public command, add a registry, create a data
mover, scan `$MD_DATA`, hash production trajectories, merge to `main`, or create/move a tag.

Write the execution report to:

    docs/journal/2026-08-27_md-data-contract-preflight-and-iterload.md

Commit and push the completed correction to `dev`.

## Required outcome

A new contract-managed project has this shape:

    $MD_DATA/{namespace}/{yyyy-mm}/{dataset_name}/
    ├── dataset.yaml
    ├── common/
    │   ├── system.xml
    │   ├── topology.pdb
    │   ├── forcefield.json
    │   └── ...
    ├── minimization/
    ├── eq/
    │   ├── nvt_1kcal/
    │   ├── npt_1kcal/
    │   └── npt_free/
    ├── cMD/
    ├── REST2/
    │   └── replica_*/
    └── AIS/
        └── trajectory_*/

`common`, `minimization`, `eq`, and each selected production method are top-level components
of one dataset. The nested equilibration stages, REST2 replicas, and AIS paths are not separate
datasets. There is exactly one authoritative `dataset.yaml` at the dataset root.

A normal contract-managed generation should be possible with the existing commands, for example:

    export MD_DATA=/managed/storage
    export MD_DATA_LOCAL="$MD_DATA/my-project/2026-08/ALA-explicit"

    md-openmm sys-gen -i ./ALA.pdb --config sys.config.yaml \
        -of "$MD_DATA_LOCAL/common/"
    md-openmm md-gen -if "$MD_DATA_LOCAL/common/" --config md.config.yaml \
        -of "$MD_DATA_LOCAL/"

The generated scripts remain standalone: they must not import `md_templates`, depend on this
checkout, or embed the absolute value of `MD_DATA`. Moving the complete `$MD_DATA` tree and
changing the environment variables must be sufficient.

## 1. Use the MD-data v1 contract, not an imitation

Use the MD-data package's published validator through its public CLI or Python API. Pin and record
the contract/package identity used by acceptance. Do not vendor its schema or silently accept a
locally invented subset.

The dataset metadata needed to create `dataset.yaml` must be explicit editable user input. Add the
smallest coherent block to the existing generated YAML; do not add another configuration layer.
It must provide or resolve every required MD-data field, including:

- `schema_version: "1.0"`;
- stable `dataset_id`;
- `namespace`, `dataset_name`, and `role`;
- the canonical path relative to `MD_DATA`;
- scientific creator identity;
- originating project repository and exact 40-hex commit;
- MD-templates repository, version, and exact 40-hex commit;
- system description;
- `status: active`;
- components for `common`, `minimization`, `eq`, and selected methods.

Do not guess a person, repository, commit, dataset ID, or role. A required value that cannot be
established must remain visibly unset and fail generation/preflight with the exact dotted field and
a short explanation of how to supply it. If an installed package has no Git checkout, a version or
installed fingerprint is useful generation provenance but does not satisfy MD-data's required
40-hex template commit; require the commit explicitly rather than fabricating one.

The canonical path must be exactly
`{namespace}/{yyyy-mm}/{dataset_name}` relative to `MD_DATA`, and the resolved
`MD_DATA_LOCAL` must be that dataset root. Refuse:

- an absolute path in `dataset.yaml`;
- a role/namespace/path disagreement;
- output outside the explicitly supplied `MD_DATA` root;
- an alias or symlink as a writable output dataset;
- a `complete` or `archived` dataset;
- a writable component marked `linked: true`;
- nested component declarations such as one component per `eq/*` stage;
- overwriting a pre-existing manifest with different identity.

Do not make MD-templates register, copy, move, archive, delete, catalogue, or recursively inspect
data. The scope is generation into the selected dataset plus validation of that dataset's manifest
and declared paths. Do not mutate a completed parent dataset. Extension behavior remains owned by
MD-data's `extension.yaml` contract and is outside this task.

If a legacy local `inputs/ + MD/` generation mode is retained for template inspection, label it
unregistered and do not call its outputs MD-data compliant. Production execution must not silently
proceed as though such a directory were a registered dataset.

## 2. Mandatory preflight before execution

Add one small shared, generated preflight helper. This is not a new public command and it must
travel with the generated project.

Every generated launcher must support a no-dynamics check, for example:

    ./run.sh --check
    ./run_all.sh --check

A normal `./run.sh` or `./run_all.sh` must invoke the same check automatically before importing
or constructing the OpenMM `Context`. If preflight fails, exit nonzero before minimization,
integration, worker spawning, checkpoint creation, or trajectory creation.

The preflight must be bounded and non-destructive. It may inspect the named manifest, declared
component directories, small configuration/provenance files, and specifically named source/output
artifacts. It must never walk `$MD_DATA`, hash a production trajectory, or open unrelated
datasets.

At minimum check:

1. `MD_DATA` and `MD_DATA_LOCAL` are set, exist as appropriate, and resolve to the exact current
   dataset root without escaping `MD_DATA`;
2. authoritative MD-data validation succeeds for `dataset.yaml` with the explicit root;
3. the dataset is `active`, project/baseline role agrees with its namespace, and the component
   being run is declared, owned, active, and not linked;
4. `common/` contains the prepared inputs required by the generated scripts, and the existing
   small checksum/provenance checks still pass;
5. the configured method and current directory map to the correct declared component;
6. the force-field identity in `common/forcefield.json` agrees with the resolved system config;
7. CUDA is available, the requested device assignment is valid, and there is no CPU/OpenCL
   fallback for a dynamics run;
8. the parent stage is complete and its recorded configuration identity matches before a
   downstream stage begins;
9. an existing completion record is internally consistent with all artifacts required to skip the
   run.

`--check` must print a compact PASS/FAIL table or list and perform zero integration steps. Test
that an invalid contract, wrong dataset root, read-only dataset, undeclared component, mismatched
force field, absent CUDA, missing parent state, or corrupt completion artifact fails before a
Context exists.

Keep normal completion semantics simple. Do not add locks, transactions, a run database, or an
automatic dataset lifecycle manager. The authoritative dataset may remain `active` until its
owner deliberately marks it complete.

## 3. Stream AIS source trajectories with `mdtraj.iterload`

Remove whole-trajectory loading from the generated AIS runtime. The source DCD may be very large;
`mdtraj.load(source_trajectory, ...)` is not acceptable.

Use `mdtraj.iterload` with bounded chunks. A sound implementation may use two bounded passes:

1. count/validate frames and determine the eligible frame indices from trustworthy timing;
2. stream again and retain only the deterministically selected frames.

Do not keep all source coordinates in memory, and do not serialize the full source trajectory into
`_ais_plan.json`. Memory must scale with the chunk size plus the selected frames, not with total
trajectory length. Record the chunk size and selected frame indices. Preserve periodic box vectors
for every selected explicit-solvent frame.

Add a test that monkeypatches or otherwise makes `mdtraj.load` fail if called on the source DCD,
then proves the generated runtime succeeds through `iterload`. Add a multi-chunk source fixture so
a one-chunk implementation cannot pass accidentally. The test may use a tiny synthetic DCD; do not
commit production trajectories.

## 4. Complete the AIS source and completion checks

Add an explicit editable `AIS.source.source_tau` field. It may be null when a trustworthy
companion `resolved_run.yaml` records tau. For an external trajectory without that record, it is
required. Record whether tau came from the companion record or explicit user input. Refuse a
conflict and refuse any source tau different from `AIS.path.tau_start`.

Before any AIS workers start, preflight must validate:

- source trajectory and topology exist;
- trustworthy first-frame time and frame interval exist;
- the inclusive time window contains enough frames;
- atom count and atom identity/order agree with `common/topology.pdb`;
- selected explicit frames contain valid periodic box vectors;
- source tau is established and matches the path start;
- deterministic frame selection and distinct seeds are completely resolved.

A finished AIS path is skippable only when all of these agree:

- `completed.json` says completed;
- `observations.csv` has the expected rows and endpoint/work mapping;
- `observations.dcd` exists and has exactly the expected number of frames;
- every CSV coordinate frame index maps one-to-one to the DCD;
- the recorded path/configuration identity and trajectory index match the current request.

A missing or truncated DCD must never be skipped merely because the JSON and CSV look complete.
An incomplete path may be safely replaced according to the existing documented behavior; never
append a second path to an old DCD.

## 5. Correct force-field combinations and acceptance fixtures

The method-development explicit default is:

- peptide/protein: ff14SB (`amber14-all.xml`);
- ligand: OpenFF Sage 2.2.1 (`openff-2.2.1`) with the configured charge method;
- water: TIP3P (`amber14/tip3p.xml`).

Keep ff19SB + OPC as the documented optional explicit selection. Keep ff14SB + GBn2/mbondi3 with
no SASA as the implicit selection. Do not cross these selections.

All system-building and CUDA dynamics acceptance added or rerun for this task must use the default
ff14SB/Sage/TIP3P selection. Do not use ff19SB with TIP3P, ff14SB with OPC, Sage with OPC in a
claimed default-combination test, or an obsolete TIP3P-FB fixture.

Be scientifically exact about the current route scope: the generator currently builds a peptide
route or a ligand route, not a protein-ligand complex containing both parameter sets. Therefore:

- the peptide acceptance must prove ff14SB + TIP3P actually loaded;
- a small ligand acceptance must prove Sage 2.2.1 + TIP3P actually loaded;
- the generated default configuration must name Sage 2.2.1;
- do not claim that Sage participated in a peptide-only System;
- do not expand this focused correction into protein-ligand-complex support.

Together these fixtures verify every member of the default selection without making the false claim
that all three were loaded into the same current-route System. Force-field assertions must inspect
`forcefield.json` or the built System's preparation record, not just `sys.config.yaml`.

Pure configuration tests may still exercise the OPC option and mismatch rejection without running
dynamics. No CUDA acceptance in this task should use OPC.

## 6. Small focused test plan

Keep tests short. Every test that minimizes, evaluates forces/energies, or integrates a molecular
System is marked `gpu`, explicitly selects CUDA, and is deselected rather than passed when CUDA is
unavailable. No nanosecond run is permitted.

Required non-dynamics tests:

1. a synthetic `$MD_DATA` tree and generated `dataset.yaml` pass the authoritative v1 validator;
2. missing/invalid dataset fields and path/role/root mismatches fail clearly;
3. preflight `--check` executes zero OpenMM steps and produces no runtime trajectory/checkpoint;
4. corrupt or incomplete AIS completion artifacts are not accepted as complete;
5. source tau from companion and explicit input resolve correctly; absent/conflicting tau fails;
6. a multi-chunk AIS source uses `iterload` and never `mdtraj.load`;
7. generated scripts contain no absolute storage path and do not import `md_templates`;
8. default config is ff14SB/Sage 2.2.1/TIP3P; crossed pairs remain rejected.

Required focused CUDA acceptance:

1. a tiny peptide build/run using ff14SB + TIP3P;
2. a tiny ligand build/run using Sage 2.2.1 + TIP3P;
3. the smallest AIS run needed to show automatic preflight occurs before Context creation, streamed
   source selection works across chunks, 21 DCD frames map to 21 work rows, and a truncated DCD is
   rerun/refused rather than skipped;
4. the smallest cMD/REST2 regression needed to prove the shared launch/preflight change did not
   alter their scientific runtime.

Use temporary synthetic dataset roots only. Never point tests at or walk real `$MD_DATA`. Do not
commit DCDs, checkpoints, environments, or GPU output.

## 7. Documentation and durable rules

Update `README.md`, `CLAUDE.md`, `docs/FAIR_HANDOFF.md`, `CHANGELOG.md`, configuration help,
and the instruction index.

Document:

- the exact MD-data v1 layout and ownership boundary;
- how to set `MD_DATA` and `MD_DATA_LOCAL`;
- the contract-managed `sys-gen` and `md-gen` commands;
- which metadata the user must provide and why it cannot be guessed;
- `run.sh --check` and automatic preflight;
- that validation is bounded and does not scan or hash `$MD_DATA`;
- that active-to-complete lifecycle changes remain deliberate MD-data operations;
- AIS source tau evidence and streamed loading;
- the exact default ff14SB/Sage/TIP3P scope and the route limitation.

Remove or correct statements saying MD-templates can never write into `$MD_DATA`. Replace them with
the accurate boundary: MD-templates may generate simulation files and a contract-valid manifest
inside the one explicitly selected active dataset; MD-data owns the schema, validation, identity
rules, lifecycle, catalogue, aliases, extensions, and archival policy.

## 8. Acceptance and report

Do not return PASS because YAML generation or documentation looks correct.

Before PASS, report evidence that:

- the generated dataset manifest passes the authoritative MD-data v1 validator;
- all execution launchers run the same automatic preflight before a Context exists;
- `--check` performs no dynamics;
- no code scans `$MD_DATA` or hashes a production trajectory;
- AIS source selection uses `iterload` across multiple chunks and not `mdtraj.load`;
- external source tau is explicit, recorded, and checked;
- a missing/truncated AIS DCD is not accepted as complete;
- CUDA peptide acceptance loaded ff14SB + TIP3P;
- CUDA ligand acceptance loaded Sage 2.2.1 + TIP3P;
- no CUDA acceptance used an incorrect or optional force-field/water combination;
- focused cMD, REST2, and AIS regressions pass;
- generated files remain portable and contain no machine-specific storage root.

The journal and final response must include:

- commit SHA pushed to `dev`;
- MD-data contract commit and validator version used;
- exact commands, markers, test counts, and runtimes;
- CUDA device(s);
- generated synthetic dataset tree and manifest validation command;
- preflight failure evidence from at least one invalid contract;
- proof that `--check` created no runtime outputs;
- peak or bounded loading evidence for the multi-chunk `iterload` test;
- exact force-field records from the peptide and ligand fixtures;
- AIS frame/CSV counts and completion-integrity evidence;
- remaining limitations.

If MD-data is unavailable or its validator cannot be installed, continue every independent code and
test change, record the exact blocker, and do not claim contract acceptance passed.
