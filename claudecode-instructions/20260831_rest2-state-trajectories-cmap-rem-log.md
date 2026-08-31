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
