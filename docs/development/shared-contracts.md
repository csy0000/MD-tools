# Shared contracts

Records that 0.6.1 and 0.7.0 must agree on. They are written down here **before** the branches
diverge, because the alternative is two private versions of the same record that both look right
and cannot be read by the same code.

A change to anything on this page is **one commit** that updates this document and the fixtures
together, landed by the coordinator and propagated to every branch. A worker that needs a change
asks for it; it does not make a local variant.

Prefer the existing module for a job. Do not build a second framework beside one that already
works — the package already has one authority each for configuration resolution, preflight,
platform policy, MPI, checkpoints, cost accounting and registration, and every one of them is
named in [`CLAUDE.md`](../../CLAUDE.md).

## 1. Identities, kept apart

Five different things are routinely confused. They are not interchangeable and no two of them may
share a field name:

All five already exist in `md_tools.ligands`. **Reuse them; do not introduce a parallel vocabulary.**

| identity | what it answers | where it lives |
|---|---|---|
| **compound** | which chemical species | `metadata["compound"] = {id, aliases, residue_name}`; the id is `CHEMBL<n>` or `LOCAL-<14 letters>`, validated by `ligands.identity.check_compound_id`. A name such as `paracetamol` is an **alias**, never an id |
| **chemical state** | which tautomer, protomer, stereochemistry, formal charges | `ligands.identity.chemical_state(mol)`, scheme `md-tools-chemical-state/1`, digested from the canonical isomeric explicit-H SMILES |
| **parameter package** | which immutable saved parameter set — what reuse is authorized against | `ligands.package.LigandPackage`, directory `param_<12 hex>`, where the id is `sha256(canonical_json({chemical_state: <state digest>, parameters: <table digest>}))[:12]` (`IDENTITY_SCHEME`). The table digest is over the OpenMM-read parameter table, not the ffxml bytes |
| **molecular instance** | which copy in *this* structure | `ligands.mapping.LigandSelector` (`chain`, `resid`, `insertion_code`, `resname`) and `LigandInstance.residue_key = (chain, resid, insertion_code)`; the structure-wide record is `MappedStructure`, persisted as `ligand_mapping.json`, schema `md-tools-ligand-mapping/1` |
| **resolved topology atom** | which atom index in *this* built System | the built topology plus `selection.topology_digest`; inside a residue, `LigandInstance.heavy_atom_map` gives the deposited-name ↔ package-name correspondence |

A compound ID never authorizes parameter reuse — `ligands.match.matches()` decides that, and it
refuses a package with no recognized charge `backend_id`. A residue name never identifies an
instance: two copies of one compound share the residue name, the package and the residue template
`MDT_param_<id>`, and are distinguished **only** by `residue_key`. An atom index never survives a
rebuild without its topology digest beside it. Catalog lookup keeps going through
`ligands.catalog.resolve_package` and the existing `$MD_DATA` configuration — no new catalog.

**Two gaps both branches will hit, named here so neither invents a private answer:**

1. **There is no instance alias today.** `aliases` in a package manifest are *compound* aliases.
   The `L01:` keys in the 0.6.1 ligand selection syntax therefore need either a recorded
   instance name added to the mapping record, or resolution against `residue_key`. Whichever S1
   chooses, an alias that does not resolve to exactly one instance is refused, never guessed from
   a residue name. S0 lands the mapping-record change if one is needed.
2. **Aliases are metadata, not identity, and today they are write-once and incomplete.**
   `parameter_id` hashes the chemical state and the parameter table; `compound.aliases` sits
   beside it. Stated `solute.aliases` reach `create_package` only when a build *creates* a
   package — both reuse paths load an existing one and never see them — `package.summary()` omits
   aliases altogether, and `register_package` takes no aliases argument, so there is no supported
   way to add an alias to an already-registered package. A registered package whose aliases were
   never written declares `[]` permanently. Neither branch may treat an alias as a reliable
   handle for anything, and neither may make one part of an identity.
3. **There is no A→B atom map between two different compounds.** `heavy_atom_map` maps deposited
   atom names to package atom names *within one compound*, and `ais.two_state` assumes an identity
   map by refusing any pair whose particles differ. The alchemical map of section 4 is new work,
   not an extension of either.

## 2. The selection record (0.6.1 owns it, 0.7.0 consumes it)

Today's record is `ScalingSelection` in `src/md_tools/rest2/selection.py`, format
`md-tools-solute-selection/1.0`, written as `solute.yaml`:

~~~yaml
format: md-tools-solute-selection/1.0
topology_sha256: ...        # selection.topology_digest(topology)
solute_atoms: [...]
unscaled_torsion_central_bonds: [[i, j], ...]
labels: [{bond: [i, j], residues: "ALA12", reason: "..."}, ...]
detection_method: ...
~~~

It records *which atoms are hot* and *which central bonds are unscaled*, and nothing about how
either was chosen — because until now the answer was always "every non-solvent residue"
(`src/md_tools/md/stage.py:64-74`).

Selective REST2 makes the *how* part of the identity, so the record grows and its format version
moves to `md-tools-solute-selection/2.0`. The added fields:

| field | meaning |
|---|---|
| `selection_mode` | `legacy-full-solute` or `explicit`. Never inferred at read time |
| `masks` | the original strings exactly as the user wrote them, per category, plus `numbering: topology-residue-index-1based` |
| `residue_map` | one-based topology residue index → `{chain, residue_id, insertion_code, residue_name, ligand_instance}` for every residue a mask matched |
| `selected_nonbonded_atoms` | the atom set that carries the `(1-tau)` / `(1-tau)^2` factors |
| `selected_torsion_central_bonds` | eligible central bonds, each with its owning residue |
| `cmap_decisions` | per CMAP term: scaled, or left unchanged with the reason (a mixed selected/unselected torsion pair) |
| `excluded_central_bonds` | unchanged in meaning from 1.0 |
| `improper_policy` | the flag that today is `unscaled_impropers` |
| `detector_policy_version` | the classifier version (`detector_version`, currently 2) |
| `ligand_instances` | per selected instance: the instance identity, its parameter package identity, the package-local ↔ topology atom correspondence, and the **resolved contents** and digest of its torsion-exclusion file |

Two rules that are the point of the record:

- **The exclusion file's contents are saved, not just its path.** A path is not provenance: the
  file can change after the record is written, and then the record describes a run that never
  happened.
- **Ligand exclusions are expressed in package-local atom identities**, never in global force
  indices, which are not transferable between builds.

One seam to be aware of: the torsion classifier
(`md_tools.openmm.system.unscaled_torsions`, the one enforcing entry point) still resolves bond
orders by reading an SDF **from disk** — `sdf_filelist`, then `<RESNAME>.sdf` beside the System,
then `built.sdf` when it is the only non-standard residue — and maps bond evidence onto topology
indices positionally. It does not read `LigandPackage.mol`, even on the package path. With several
instances and per-instance exclusions that resolution has to become package-aware; whoever changes
it changes one function, not two.

**One parser, one resolver, used by preparation and validation alike.** The runtime consumes and
verifies a resolved record; it never re-selects atoms from inputs that may have changed since.

### Mask grammar, version 1

A documented subset of AMBER's, not the whole grammar:

~~~text
":45"          one residue
":45,46,59"    a list
":45-50"       an inclusive range
":45,48-52"    a mixture
~~~

Numbers are one-based topology residue indices. Anything else — chain qualifiers, atom selectors,
operators, wildcards — is refused by name with the supported subset printed. Out-of-range and
empty matches are refused. No per-chain index reset is ever invented.

## 3. Hamiltonian identity

`FINGERPRINT_FORMAT` in `src/md_tools/rest2/identity.py` is `md-tools-hamiltonian-identity/v2`,
and `selection_sha256` there currently hashes only `{solute, unscaled_central_bonds,
unscaled_impropers}`. Under selective REST2 two different selections could produce the same solute
atom set with different torsion and CMAP treatment, so the fingerprint moves to **v3** and hashes
the full version-2 selection record.

`REST2_IMPLEMENTATION` (`src/md_tools/rest2/hamiltonian.py`), name `rest2-unscaled-torsions`,
currently version 3, gains the selection semantics it now depends on and its version moves with
them. `require_same_hamiltonian` keeps refusing a mismatch; the point of the version bump is that
a saved state built under other semantics is **refused**, not silently mixed into a ladder.

## 4. The alchemical topology plan (0.7.0)

A record independent of the MD runner. It is produced by `combine-topology` and consumed by the
Hamiltonian builder, the executor and the analysis, so all three read the same file rather than
re-deriving a mapping each.

It contains: endpoint package and System identities; the atom map **in both directions**; the
common, A-only and B-only particle sets; environment identity; coordinates; constraints, masses
and virtual sites; force-field conventions; exclusions; and a digest of every source.

It also records **the topology numbering used to resolve any pre-combination mask**, and carries
those mapped identities through the combined topology. Residue numbers change when topologies are
combined; a mask resolved before combination and reinterpreted afterwards silently selects
different atoms.

The existing `md_tools.ais.two_state` is **not** this. It mixes two Systems with identical
particles, masses and constraints — `_structural_differences` refuses anything else — which is an
implicit identity atom map. Reuse its constraints and its honesty about limitations; do not reuse
it as a softcore engine.

## 5. Hamiltonian state coordinates

State is a set of **named** coordinates:

~~~yaml
state:
  lambda_sterics: 0.4
  lambda_electrostatics: 0.0
  tau: 0.0            # only where REST2 is part of the Hamiltonian
~~~

- **A name means one thing.** `tau` is REST2 solute scaling. `lambda_*` are alchemical components.
  They never share a parameter name, and a numerical coincidence between them means nothing.
- A **path** maps a scalar progress coordinate `s` to those components and states the endpoint
  orientation — which endpoint is A, which is B, and which direction "appearing" means.
- `md_tools.ais.two_state` already owns the name `lambda` as a Context parameter for its
  two-System mixing, with `LAMBDA_PARAMETER` and `TWO_STATE_SCHEMA`. The alchemical
  implementation uses its own distinct names and does not overload that one.

**A serialized System, its parameter values, the topology plan, the schedule and the provenance
must reconstruct the same state energies without a live original Context.** Changing parameters in
a Context is allowed only when the complete state is recorded well enough to rebuild it.

Agree, before S2/S3/S4 diverge, the callable interfaces for: construction, state setting,
energy and cross-state evaluation, and complete derivatives. Provide real miniature fixtures, not
only mocks — a mock cannot notice a missing PME reciprocal-space term.

## 6. Data and execution

Reuse the existing authorities: preflight, platform policy, MPI, checkpoints, registration and
cost accounting. Specifically:

- Every record uses **named state and sample IDs with units**. Sample provenance is the state a
  sample came from — never the rank of the worker that happened to process it.
- Collect the reduced potentials the ensemble actually requires, including pressure–volume terms
  under NPT, plus sample counts, derivative components, and uncertainty and overlap information.
- Nothing is written until the whole preflight passes, and a refused continuation stays read-only,
  including its own logs.
- A checkpoint commit is a generation transaction. `md_tools.openmm.checkpoint` is the one
  implementation, for alchemical windows as for everything else.
- Alchemical support is **optional** for existing cMD installations: an install without the
  alchemy extra keeps working unchanged.
- Reused OpenFE / OpenMMTools components are pinned, with their license, version and commit
  recorded. A standard OpenFE workflow must never silently reparameterize a registered MD-tools
  ligand.

## 7. Fixtures

A small versioned fixture set is frozen with this document so that S2, S3 and S4 can work in
parallel against the same numbers instead of against each other's code:

| fixture | what it is | consumed by |
|---|---|---|
| a two-residue peptide plus one small ligand, explicit solvent | selection boundaries: phi/psi across a residue boundary, chi1 reaching backbone, two copies of one compound | S1 |
| a miniature endpoint pair, neutral, one atom substituted | a topology plan small enough to check by hand in both directions | S2, S3 |
| frozen state energies and derivatives for that pair, at lambda 0, 0.25, 0.5, 0.75, 1 | lets estimator work start before a real Hamiltonian exists | S4 |

A fixture is versioned. When a fixture changes, its version changes, and every result citing the
old version is re-derived rather than reinterpreted.

## Validation gates

Every gate below applies to both branches. An acceptance matrix is written **before** any long
campaign, one row per check: fixture, expected quantity, independent reference, tolerance,
platform, command, evidence path, and a verdict of PASS / FAIL / BLOCKED / NOT RUN.

1. **Documentation** — no broken internal links after cleanup; one active instruction index per
   branch; examples parse through the real resolver; generated references regenerated from their
   authorities, never hand-edited.
2. **Deterministic physics** — analytic pair energies, forces and derivatives; physical endpoints;
   mask boundaries; exceptions; constraints; dummy contributions; serialization and relocation.
3. **Derivatives** — central finite differences in the interior, appropriate one-sided checks at
   the endpoints, several step sizes, with convergence and both absolute and per-component errors
   reported so a solvent background energy cannot hide a defect.
4. **Cross-engine** — fixed-coordinate AMBER comparison for the claimed Amber18 path at lambda 0,
   0.25, 0.5, 0.75 and 1, with matched parameters, cutoffs, PME, exclusions, restraints and
   corrections; intentional differences reported. The OpenFE adapter is compared separately
   against an OpenFE-matched path — different paths need not agree in the interior.
5. **Sampling** — independent repetitions with uncertainty, overlap and convergence diagnostics.
   Thresholds are defined before sampling starts.
6. **CUDA and runtime** — real propagation, correct exchange state mapping, interruption and
   resume, changed-state rejection, installed-wheel execution outside the checkout, export
   reconstruction, full dataset registration.
7. **Regression** — existing cMD, whole-solute REST2 and two-state AIS keep their contracts, on
   the candidate integration commit.

Tolerances are calibrated against documented precision and reference error **before** the
comparison, and recorded. A tolerance is never weakened after a failure, and a failing test is
never deleted to close a gate: the implementation is fixed and the affected checks rerun.

**Missing CUDA, AMBER or license access is BLOCKED evidence.** A skip is not a pass, and a
successful CPU run is not CUDA evidence — a stale MPS pipe directory once made an entire 110-test
GPU lane skip and exit 0.
