# Implement fixed-state Amber REST2 trajectories, CMAP duplication, and REM log compatibility

Work in one Claude Code session across:

```text
git@github.com:csy0000/MD-templates.git
git@github.com:csy0000/MD-project.git
```

## Starting points and branches

Verify these exact remote heads before editing:

```text
MD-templates/fix/openmm-md-rest2-pending-validation
1b0d866c83762117ce437da15b589c0691170253

MD-project/dev11-openmm-md-rest2-pending-validation
c1d36f93ef4acdc7cf3838ff6ef1778998093099
```

Work only on:

```text
MD-templates: feat/rest2-state-trajectories
MD-project:   dev12-rest2-state-trajectories
```

Read and follow every applicable `CLAUDE.md` and repository instruction. Preserve the completed migration-transaction, restart, rREST2, provenance, and validation contracts from the starting branches. Do not merge into `dev` or `main`, do not rewrite history, and do not force-push.

## Objective

Change the owned `openmm-md` REST2/rREST2 implementation so that NVT replica-exchange coordinates are written like Amber Hamiltonian REMD: one Amber-compatible NetCDF trajectory per fixed thermodynamic state, named by stable state index:

```text
remd0.nc
remd1.nc
...
remdN.nc
```

A file must never encode tau in its name. `remd2.nc` means state index 2 only; the exact tau belongs in validated metadata. A successful exchange moves the complete configuration between fixed-state contexts, so subsequent frames are written to the trajectory of the state the configuration now occupies.

Also:

- call the Hamiltonian convention `rest2-no-bond-angle-omega/v1`;
- make tau the only public and persisted scaling coordinate;
- duplicate shared CMAP maps and attach solute torsions to scaled copies;
- persist the resolved omega classification and affected torsion indices;
- report tau and derived effective temperature in resolved YAML;
- accept `--rem rem.log` and write the exact Amber26 H-REMD replica log grammar required by cpptraj;
- print only cumulative neighbouring-pair acceptance at normal completion;
- keep deeper mixing analysis outside the executor;
- restrict this milestone to NVT REST2/rREST2;
- update MD-project to pin and exercise the completed MD-templates commit.

## 1. Hamiltonian identity: tau only

The public and persistent Hamiltonian definition is:

```text
U_tau =
    (1 - tau)^2 * U_solute-solute,nonbonded
  + (1 - tau)   * U_solute-environment,nonbonded
  +                 U_environment
  + (1 - tau)^2 * U_eligible-solute-torsions
  +                 U_bonds,angles,ordinary-amide-omega
```

Do not expose or persist a secondary variable named `s`. Remove it from YAML, manifests, protocol records, CLI output, documentation, stored state descriptions, and public helper names. Internally calculate the factors once with descriptive names such as:

```python
solute_solute_scale = (1.0 - tau) ** 2
solute_environment_scale = 1.0 - tau
```

Do not copy the formulas independently into multiple force handlers.

Persist:

```yaml
rest2_implementation:
  name: rest2-no-bond-angle-omega
  version: 1
  state_coordinate: tau
  solute_solute_nonbonded_scale: "(1-tau)^2"
  solute_environment_nonbonded_scale: "1-tau"
  eligible_solute_torsion_scale: "(1-tau)^2"
  bonds: unscaled
  angles: unscaled
  ordinary_amide_omega: unscaled
```

This complete convention, including its version and resolved exclusions, participates in Hamiltonian identity and continuation checks.

## 2. CMAP duplication

Correct shared-map handling. For each scaled System and each original CMAP map used by both wholly-solute and other torsions:

1. retain the original map unchanged;
2. create exactly one duplicate of that map;
3. multiply every energy in the duplicate by `(1-tau)^2`;
4. redirect every wholly-solute torsion using that original map to the duplicated map;
5. leave every other torsion attached to the original.

For an exclusively solute-owned map, scaling in place is acceptable on the cloned System. For an exclusively non-solute map, do nothing. At tau zero the cold System must remain byte/energy equivalent to the source System and must not gain unnecessary duplicate maps. Implement equivalent correct behaviour for live tau switching without accumulating duplicate maps or compounding scaling.

Add focused tests for exclusive-solute, exclusive-environment, shared, and multiple-solute-torsions-sharing-one-map cases.

## 3. Resolved omega record

Keep the existing two-route classifier:

- peptide/protein: topology plus residue-aware classification;
- ligand: retained SDF plus RDKit SMARTS `[CX3](=[OX1])[NX3]` and bounded small-ring nitrogen detection.

Do not replace it with atom-name matching. Persist, version, and validate:

- detection route and detector version;
- actual unscaled central C-N bonds;
- proline-like bonds left eligible for scaling;
- ambiguous candidates and their evidence;
- the exact PeriodicTorsionForce torsion indices affected by every excluded central bond.

Production must continue to refuse unresolved ambiguous candidates. Add cases for ACE-ALA-NME, X-PRO, N-methyl amide, ligand amide, macrocyclic amide, and an unrecognized modified residue.

## 4. Resolved YAML

For every state, record stable state index, trajectory basename, exact tau, and derived effective temperature:

```yaml
states:
  - index: 0
    trajectory: remd0.nc
    tau: 0.0
    effective_temperature_k: 300.0
```

Compute only for reporting:

```text
effective_temperature_k = physical_temperature_k / (1 - tau)^2
```

Never infer tau from the filename and never sort trajectory files lexicographically. State order comes from the validated numeric indices.

## 5. Output ownership and file layout

Replace the bundled coordinate layout `positions[frame,walker,atom,spatial]` as the primary REST2 trajectory representation.

For N states, write:

```text
remd0.nc ... remdN.nc       one full-system trajectory per fixed thermodynamic state
exchange.nc                 authoritative exchange/provenance record, no bundled coordinates
rem.log                     Amber26-compatible H-REMD log
checkpoint.nc               complete restart state
restart.json                completion evidence
```

If the dense solute stream remains supported, also write one state-indexed trajectory per state under a documented location, for example `solute/remd0.nc ... solute/remdN.nc`. It may have an independent output interval.

At a frame event after exchange:

```python
for state_index in range(n_states):
    walker = state_to_walker[state_index]
    write remd{state_index}.nc from configurations[walker]
```

The event order remains:

```text
propagate -> exchange -> full-state frames -> solute-state frames -> checkpoint
```

Thus a coincident trajectory frame is post-exchange.

## 6. Amber NetCDF trajectory compatibility

Each `remd{index}.nc` must be a genuine Amber NetCDF trajectory readable by AmberTools26 cpptraj, not merely a NetCDF file with similar arrays. Use Amber coordinate variable names, dimensions, attributes, units, and conventions. Coordinates are written in angstrom. This milestone is NVT only; do not add NPT exchange or variable-cell semantics. Fixed-cell metadata may be handled exactly as the Amber NVT convention requires.

The groupfile supplies one explicit `-x remd{index}.nc` for every state. Reject duplicate, missing, non-contiguous, or state-mismatched outputs before creating files.

Keep `exchange.nc` as the authoritative exchange record, with tau, steps/times, mappings, proposals, decisions, independently evaluated reduced potentials, rule state, reservoir events, identities, migrations, and committed markers. Version the new schema. Do not silently reinterpret older bundled-coordinate files. Provide an explicit read-only compatibility/refusal or migration boundary and document it.

## 7. Amber H-REMD rem.log

Add the coordinated CLI flag exactly:

```text
--rem rem.log
```

The generated launcher obtains this path from `paths.sh` and `$MD_DATA`.

Implement the exact Amber26 section 27.4.4.2 Hamiltonian-REMD log grammar accepted by cpptraj. Inspect the installed Amber26 source/parser and manual; do not infer field meanings from column names or from old temperature-REMD examples. The expected H-REMD header includes:

```text
# Rep#, Neibr#, Temp0, PotE(x_1), PotE(x_2), left_fe, right_fe, Success, Success rate (i,i+1)
```

Establish and test exact:

- one-based `Rep#` and `Neibr#` meaning;
- no-partner sentinel;
- row ordering and one row per state/replica per exchange block;
- kcal/mol conversion from OpenMM kJ/mol;
- signs and orientation of both potential and free-energy columns;
- use of the common physical thermostat temperature in `Temp0`, never effective temperature;
- alternating-pair behaviour;
- cumulative neighbouring success-rate convention;
- relationship between rem.log mappings and state-indexed trajectory files.

The existing full reduced-potential information must be the source; do not approximate cross energies.

`rem.log` is a deterministic compatibility projection of committed `exchange.nc` rows, not an independent authority. Make restart crash-safe: regenerate atomically from committed exchange history, or validate/truncate complete blocks against the authoritative marker before append. `--verify-only` remains read-only and compares the log when supplied.

Reservoir refreshes remain in exchange/provenance records unless Amber26 explicitly defines a compatible rREST2 row representation. Do not overload unrelated Amber columns.

## 8. Completion report

At normal REST2/rREST2 completion, print and persist only neighbouring-pair acceptance:

```text
state 0 <-> state 1   accepted/proposed   ratio
...
overall               accepted/proposed   ratio
```

Use cumulative counts from authoritative committed rows. Report reservoir attempts separately for rREST2. Round trips, transition matrices, first-passage times, and convergence diagnostics belong to downstream analysis and must not be added here.

## 9. Multi-file crash consistency

One frame event writes N state trajectories. Treat the set as one logical commit:

1. write the row to every state trajectory;
2. sync all state trajectories;
3. advance one authoritative global committed-frame marker;
4. sync the authoritative exchange record.

On continuation, validate equal committed times/steps and state identities across all trajectories. Rows past the global marker are uncommitted and must be deterministically removed or ignored according to one documented policy. A missing or shorter state file is corruption and must be refused; never fabricate or pad frames.

Keep the existing rule that validation and continuation eligibility are established read-only before mutation.

## 10. Tests

Tests must include:

- tau-only public/persisted contract; no public or persisted `s`;
- implementation identity `rest2-no-bond-angle-omega/v1`;
- CMAP duplication and routing;
- omega classification and exact affected torsion indices;
- resolved YAML state-index-to-tau/effective-temperature mapping;
- `remd0.nc ... remdN.nc` numeric state naming independent of tau representation;
- genuine Amber NetCDF convention checks;
- fixed-state output after accepted and rejected swaps;
- groupfile per-state `-x` validation;
- exact Amber H-REMD rem.log golden cases;
- an actual AmberTools26 cpptraj parse/reconstruction test when installed;
- differing exchange and trajectory output intervals;
- restart/extension and every relevant crash point across multiple trajectory files;
- serial versus one-rank-per-state MPI equivalence of exchange schedule, mappings, and records;
- preservation of rREST2 reservoir provenance;
- all previous migration-transaction and pending-validation tests.

Use short CPU/unit tests for CI. Hardware validation is limited to alanine dipeptide:

- ALA implicit NVT REST2;
- ALA explicit NVT REST2;
- serial and MPI where available;
- no validation run longer than 5 ns;
- prefer a much shorter smoke unless a longer run is needed to establish the stated contract.

Do not require bitwise-identical long trajectories across serial/MPI or hardware. Require identities, schedules, mappings, complete records, parser compatibility, and numerical agreement within explicit tolerances.

## 11. MD-project integration

After MD-templates is committed:

- pin its exact full commit SHA and branch in MD-project;
- update workflow/config examples for per-state `remd{index}.nc`, `exchange.nc`, and `--rem rem.log`;
- update provenance/staleness inputs to include the implementation identity, tau ladder, state trajectories, exchange record, and rem.log;
- keep generated simulation outputs ignored;
- validate dry-run behaviour without production MD.

## 12. Documentation and execution record

Update the user documentation and architecture description. State clearly:

- each `remd{index}.nc` is a fixed thermodynamic-state trajectory;
- configurations/walkers move between these files after accepted exchanges;
- filenames never encode tau;
- `exchange.nc` is authoritative;
- `rem.log` and Amber trajectories provide cpptraj compatibility;
- only NVT is supported in this milestone;
- the Hamiltonian is `rest2-no-bond-angle-omega/v1`.

Create an execution journal in each repository recording exact commits, environment, available AmberTools/cpptraj/CUDA/MPI capabilities, commands, test results, unavailable checks, limitations, and any deviation. Do not claim cpptraj compatibility from text inspection alone: it requires a real parser test.

## Completion

Run all applicable unit, integration, formatting, linting, generation, restart, verify-only, cpptraj, CUDA, and MPI checks. Inspect generated outputs and Git status. Make logical commits. Push each branch only after its local suite passes. Do not merge into `dev` or `main`.

Stop and report rather than inventing Amber field semantics, silently migrating incompatible trajectory files, weakening restart validation, or claiming hardware/parser validation that was unavailable.


## 13. Replace old machine test data with one human-readable validation dataset

Before running the new hardware validation, inventory the old generated test data under the immediate entries matching:

```text
/path/to/MD_DATA*
```

The wildcard is for discovery only. Never pass `/path/to/MD_DATA*` or another unresolved glob to `rm`, `mv`, `find -delete`, or any recursive destructive command.

Resolve every candidate to an explicit absolute path and classify it using its manifests, journals, completion records, ownership, and repository references. Remove only obsolete generated REST2/rREST2 validation datasets made by the preceding implementation attempts. Preserve:

- source structures and configuration inputs;
- accepted or archived scientific datasets;
- unrelated projects;
- any directory whose ownership or purpose is uncertain;
- the new validation dataset;
- repository fixtures such as the committed small ALA PDB.

Before removal, record an inventory in the execution journal containing each exact path, its classification, why it is obsolete, and whether another committed file references it. Prefer a recoverable same-filesystem move into one explicitly named quarantine/trash directory when practical. If the old data cannot be distinguished safely, stop and report the candidates rather than guessing. After cleanup, confirm the old confusing `MD_DATA*` validation roots are gone or quarantined and report exactly what happened and how it can be recovered.

Use one human-readable canonical validation root for the new run. Unless the installed MD-data contract requires a different exact structure, use the semantic shape:

```text
/path/to/MD_DATA/validation/2026-08/alanine-dipeptide-nvt-rest2/
├── inputs/
├── common/
├── REST2/
└── REST2_ext1/
```

Do not create numbered roots such as `MD_DATA2`, `MD_DATA_test3`, or names encoding implementation attempts. Record the actual chosen canonical path and why it satisfies the current MD-data contract. Paths in committed configuration remain portable; machine-specific absolute paths belong only in ignored local configuration and the execution journal.

## 14. ALA 5 ns run plus a separate 5 ns extension example

After all unit, short smoke, parser, restart, and dry-run checks pass, perform one human-readable ALA NVT REST2 validation example:

```text
REST2/       initial 5 ns per state
REST2_ext1/  one genuine additional 5 ns continuation per state
```

The initial `REST2/` run must use the completed fixed-state output design:

```text
REST2/
├── remd0.nc ... remdN.nc
├── exchange.nc
├── rem.log
├── checkpoint.nc
├── restart.json
└── resolved protocol/provenance records
```

`REST2_ext1/` is not an independent rerun and not a copy relabelled as an extension. It must consume the terminal coordinated checkpoint and authoritative completion state of `REST2/`, continue all states for exactly another 5 ns, and write a new output set under `REST2_ext1/`:

```text
REST2_ext1/
├── remd0.nc ... remdN.nc
├── exchange.nc
├── rem.log
├── checkpoint.nc
├── restart.json
└── extension provenance
```

The extension contract is:

- the original `REST2/` directory is a completed, immutable parent and remains byte-for-byte unchanged;
- state indices, tau ladder, Hamiltonian identity, topology, atom order, timestep, physical temperature, exchange rule, and implementation identity are identical;
- positions, velocities, fixed box, walker/state mapping, integrator state, exchange-rule RNG/state, reservoir state where applicable, absolute step, physical time, exchange index, and cumulative budget continue from the parent;
- the extension contains 5 ns of new dynamics, so the chain represents 10 ns total;
- extension-local trajectories contain only the new segment, but their time and step coordinates remain absolute and begin after the parent terminal values;
- extension provenance records the parent dataset/path, parent project/component commits, parent completion-manifest identity/checksum, parent terminal checkpoint identity/checksum, inherited exchange count, segment duration, and cumulative duration;
- cumulative neighbouring acceptance may be reported for the complete 10 ns chain, but segment-local counts must remain distinguishable;
- cpptraj can concatenate `REST2/remd{index}.nc` followed by `REST2_ext1/remd{index}.nc` without duplicated or missing boundary frames;
- rem.log semantics and exchange numbering across the boundary are explicit and tested;
- a mismatched or incomplete parent is refused read-only before `REST2_ext1/` is created.

If the current `--extend` interface only appends in place, redesign it narrowly so the documented example produces a new extension directory without mutating its parent. Do not fake the requested structure by copying the parent or restarting from coordinates alone. The groupfile, `paths.sh`, README example, MD-project workflow, validation command, and execution journal must show the exact reproducible commands.

The two 5 ns segments are the requested hardware validation exception. No other production-like test should duplicate them. If CUDA/MPI/AmberTools or wall time makes the full 5 ns + 5 ns validation unavailable, complete the implementation with short tests, record the exact limitation, and provide the exact command without claiming the long run passed.


## 15. Superseding correction: linear generalized-Born scaling

This section supersedes the generalized-Born rule in section 1 and the contrary decision recorded in
the 2026-08-31 journals. The supported implicit-solvent GBn2 REST2 route must scale the complete
generalized-Born energy contribution linearly:

```text
U_tau,implicit =
    (1 - tau)^2 * U_solute-solute,ordinary-nonbonded
  + (1 - tau)   * U_GB
  + (1 - tau)^2 * U_eligible-solute-torsions
  +                 U_bonds,angles,ordinary-amide-omega
```

For the supported whole-system implicit-solvent route, multiply the entire GB force contribution by
`1 - tau`. This includes every term implemented by that GB force; do not obtain this result only by
scaling charges, because charge scaling can leave non-charge-dependent terms unchanged. Do not
silently apply the old solute-solute factor `(1-tau)^2` to GB. If a future partial-solute implicit
system cannot partition GB contributions without changing the requested Hamiltonian, refuse it with
a clear message rather than inventing a decomposition.

Thread the already-derived linear factor into cloned-system generation and live `TauSwitcher`
updates. Repeated switching must always restore from the unscaled reference and must not compound
the factor.

This physics change requires a new Hamiltonian identity:

```yaml
rest2_implementation:
  name: rest2-no-bond-angle-omega
  version: 2
  state_coordinate: tau
  solute_solute_nonbonded_scale: "(1-tau)^2"
  solute_environment_nonbonded_scale: "1-tau"
  generalized_born_scale: "1-tau"
  eligible_solute_torsion_scale: "(1-tau)^2"
  bonds: unscaled
  angles: unscaled
  ordinary_amide_omega: unscaled
```

Replace references to `rest2-no-bond-angle-omega/v1` in active code, schemas, generated records,
tests, examples, and documentation with version 2. Preserve version 1 only as a historical identity
that continuation and verification must refuse as Hamiltonian-incompatible. Do not rewrite old
manifests or journals to pretend they used version 2.

Add force-level and end-to-end tests at tau 0, at least one intermediate tau, and tau 0.5. Separate
the ordinary nonbonded and GB energy groups and verify their ratios independently, for both cloned
states and live switching. Retain the existing GBn2/mbondi3/no-SASA default checks. Record the
resolved GB force class, scaling expression, and implementation identity in YAML and completion
provenance.

## 16. Superseding data lifecycle: finish locally, then register

This section supersedes the direct-to-`$MD_DATA` generation path in sections 13 and 14 and in the
current journals. The cleanup already recorded remains historical evidence, but the new validation
must be generated first in the ignored project-local staging area:

```text
{MD-project checkout}/
├── config/
│   └── machine/
│       └── machine.config       ignored, machine-local
├── configs/
│   └── data/
│       └── ALA.yaml             tracked registration intent
└── data/
    └── ALA/                     ignored while active
        ├── dataset.draft.yaml
        ├── common/
        ├── REST2/
        └── REST2_ext1/
```

The initial 5 ns and genuine 5 ns extension from section 14 run under
`./data/ALA/REST2/` and `./data/ALA/REST2_ext1/`. Generated launchers and records must remain
valid before and after registration: use paths relative to the generated dataset root and do not
embed the checkout path or the final `$MD_DATA` path.

### Machine configuration

Implement and document the ignored file exactly at:

```text
config/machine/machine.config
```

with a machine-local value such as:

```yaml
schema_version: 1
paths:
  md_data: /path/to/MD_DATA
```

`md-data-register` must discover the project root and read this file automatically. It must resolve
and validate `paths.md_data`, require an existing writable managed root, reject a relative path,
reject the filesystem root, and print the configuration file and resolved destination during
preflight. Do not commit this file or copy its absolute path into portable project configuration.
An explicit `--machine-config` may select another file, but normal use requires no machine-path
argument. If the file or key is missing, stop with an actionable error before touching the dataset.

### Registration ownership and interface

Implement `md-data-register` in MD-project, not MD-templates. MD-templates emits
`dataset.draft.yaml`, simulation manifests, and checksums it authoritatively knows; MD-project owns
completion policy, common/project assignment, movement, symlink creation, and the `$MD_DATA`
catalogue boundary. Reuse and migrate compatible schemas from MD-data rather than creating a
competing manifest. Do not delete or archive the MD-data repository in this milestone.

The normal interface is:

```bash
md-data-register \
  --idata ./data/ALA/ \
  --iconfig ./configs/data/ALA.yaml
```

and the same command with `--dry-run` must perform every read-only validation and print the exact
plan without changing source, destination, registry, or symlinks.

The tracked registration configuration owns human intent, including dataset ID/title, access and
licensing status, and `common` versus project ownership. The generated draft owns generator facts.
Merge them into one resolved authoritative dataset manifest during registration, retaining both
source identities and checksums. A disagreement in an overlapping authoritative field is an error,
not a precedence guess.

### Finish gate

Registration is forbidden until the dataset is finished. For this ALA acceptance dataset, the gate
must establish read-only that:

- `REST2/` completed exactly 5 ns per state;
- `REST2_ext1/` is a genuine chained additional 5 ns per state;
- the parent is unchanged and all parent/extension identities and checkpoint hashes agree;
- the Hamiltonian is `rest2-no-bond-angle-omega/v2`;
- all `remd{index}.nc` files, `exchange.nc`, `rem.log`, checkpoints, resolved YAML, completion
  manifests, and extension provenance exist and validate;
- state indices, tau ladder, topology, atom order, absolute steps/times, exchange numbering, and
  committed-frame markers are coherent;
- cpptraj parses every state trajectory and concatenates parent plus extension without a duplicate
  or missing boundary frame;
- required checksums are complete and no output is actively open or being written;
- the source is a real directory inside this project checkout, is ignored by Git, and is not already
  a symlink or registered destination.

A failed gate leaves everything unchanged and reports every detected error where safe.

### Transaction

Registration must be restartable and failure-safe:

1. inventory the source and calculate required checksums;
2. resolve the common/project destination from the tracked configuration;
3. refuse an existing conflicting dataset ID or destination;
4. write a durable transaction plan and registration ID;
5. stage the complete dataset at the destination;
6. verify destination size, inventory, manifests, and checksums;
7. atomically commit the resolved dataset manifest and catalogue entry;
8. only then remove the original source directory;
9. create `./data/ALA` as a symlink to the registered destination;
10. verify the symlink and registered dataset read-only and mark the transaction complete.

For different filesystems use copy, sync, checksum verification, destination commit, source removal,
then symlink creation. Never delete the source after an incomplete copy. For the same filesystem an
atomic rename is allowed only after all preflight checks and with a recoverable transaction record.
On interruption, rerunning must resume or report the exact recoverable state; it must not duplicate,
silently overwrite, or register a partial dataset.

The symlink target should be relative when safely representable; otherwise document why an absolute
machine-local symlink is required. The registered data and symlink remain ignored by Git, while
`configs/data/ALA.yaml`, schemas, tests, and portable provenance references remain tracked.

### Tests and acceptance

Use temporary project and managed-data roots for unit/integration tests. Cover missing or malformed
machine configuration, unsafe roots, dry-run purity, unfinished data refusal, active-writer refusal,
common/project assignment, destination collision, same- and cross-filesystem transaction paths,
checksum failure, interruption at every mutation boundary, idempotent resume, symlink creation, and
post-registration verification.

For the hardware example, capture inventories and checksums before and after registration and prove
that the local source became a valid link to one complete registered dataset. Update MD-project
documentation and its journal with the exact generation, finish-check, dry-run, registration, and
verification commands. The long simulation must finish before registration begins; do not register
partial output merely to demonstrate the CLI.
