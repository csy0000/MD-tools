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

**Three gaps both branches will hit, named here so neither invents a private answer:**

1. **There is no instance alias today.** `aliases` in a package manifest are *compound* aliases.
   The `L01:` keys in the 0.6.1 ligand selection syntax therefore need either a recorded
   instance name added to the mapping record, or resolution against `residue_key`. **Decided
   2026-09-19:** until S0 adds an instance name to the mapping record, the compact `L01: <path>`
   form is refused by name — see section 3. Nothing is ever guessed from a residue name.
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

### The topology digest is canonical from 2.0 on

**Decided 2026-09-19, on S2's reproduction, verified by S0.** `selection.topology_digest` hashes
bonds in the order the Topology iterates them. That order comes from the file: OpenMM's PDB
writer emits CONECT records for a non-standard residue in its own order, so reading and rewriting
a file with a cross-residue CONECT bond — S2's hybrid `combined.pdb`, ethane in TIP3P plus a CLE
residue bonded by CONECT — makes the digest alternate between two values forever
(`28aba0b9…` / `40f51187…`, the last three bonds swapping). Its docstring promises the opposite.
Standard protein bonds come from residue templates and are stable: ALA explicit and implicit and
the ethane-in-TIP3P fixture round-trip unchanged. The exposure is therefore ligands, covalent
links and cyclic peptides — whatever carries its bonds in CONECT.

- **One function, fixed in place** (S1 owns `rest2/selection.py`): bonds are hashed as a sorted
  list of sorted pairs. No second digest anywhere.
- **2.0 records** carry `topology_digest_scheme: atoms-in-index-order+sorted-bond-set/1` and only
  the canonical digest.
- **1.0 records** (every 0.6.0 `solute.yaml`) keep validating: `validate_against` accepts a 1.0
  record whose stored digest equals EITHER the canonical digest OR the legacy iteration-order
  digest of the current topology. It is a named compatibility branch with its own tests, not a
  loosened comparison, and nothing new is ever written in the legacy scheme.

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
**stays at version 3** (decided 2026-09-19, on S1's deviation). It describes the per-term scaling
rule, which selective REST2 does not change; the convention dict is embedded in every record and
checked by `require_compatible_implementation`, so bumping it would refuse every 0.6.0
continuation and defeat the legacy-resume ruling below. Selective semantics are versioned where
they live instead: the selection policy `md-tools-selective-rest2/1` inside the
`md-tools-solute-selection/2.0` record, and fingerprint v3. `scaler.yaml` carries `selection`,
`selection_sha256` and `scaler_arguments` — exactly the `build_scaled_system` keyword arguments —
and every consumer that rebuilds or identifies a state reads them from there rather than
reconstructing them from `solute.atom_indices`. `require_same_hamiltonian` keeps refusing a mismatch; the point of the version bump is that
a saved state built under other semantics is **refused**, not silently mixed into a ladder.

**Decided 2026-09-19 (S0, on S1's proposal): a legacy run stays resumable.** A v3 fingerprint
would otherwise make every 0.6.0 ladder and phase-space stage unresumable under 0.6.1 even when
nothing about its Hamiltonian changed — the same defect the 20260909 resume-identity note records,
where a defaulted field made every in-flight run unresumable. So `require_same_hamiltonian`
accepts a recorded **v2** identity if and only if:

- the current selection is `selection_mode: legacy-full-solute`, and
- every v2 field (`system_sha256`, `selection_sha256` recomputed the v2 way, `tau`,
  `temperature_k`, `ensemble`, `n_solute_atoms`, `n_unscaled_central_bonds`,
  `unscaled_impropers`) matches.

A v2 record meeting an **explicit** selection is refused by name, and so is any v2 field mismatch.
The acceptance is a named compatibility branch with three tests — legacy resumes, explicit
refused, one-field mismatch refused — not a relaxed comparison. Anything written by 0.6.1 is v3.

**Decided 2026-09-19: the v3 identity hashes the Hamiltonian, not the provenance.** An identity
that changed when a mask was respelled (`":2,3"` / `":2-3"`) or a ligand instance relabelled
would refuse to resume an identical Hamiltonian -- the 20260909 defect again.
`rest2.identity.hamiltonian_selection_projection` is the ONE definition of what determines the
Hamiltonian: selection mode, hot nonbonded atoms, scaled and protected torsion central bonds, CMAP
decisions, improper and detector policy, and per instance its `residue_key`, package
`parameter_id` and RESOLVED exclusions (package plus named bonds; comments and file bytes are
provenance). `selection_sha256` is the sha256 of that projection and is the identity;
`selection_provenance_sha256` covers the whole document and is never compared.
`claimed_region_differences` compares through the same projection. v3 records written on the
worker branch before `323779f` hashed the whole document; none left a test.

### The REST2 selection configuration keys

**Decided 2026-09-19.** Optional top-level keys of the scaler configuration (`SCALER_SCHEMA`,
`src/md_tools/build/scaler.py`), named as the user wrote them:

~~~yaml
backbone_scaling_list: ":45,46,59"     # mask string, grammar v1
sidechain_scaling_list: ":45-50"       # mask string, grammar v1
ligand_scaling_dict:
  L01:                                 # a user label; NOT an identity
    mask: ":201"
    torsion_exclusions: L01-exclusions.yaml   # or: auto
~~~

None present → legacy full solute. In the REST2 workflow configuration the same three keys, under
`rest2:`, are **claims** about the saved states: `build-md` resolves a claim through the one
resolver and refuses it unless it is the region `scaler.yaml` records, as it does for
`number_of_replicas` and `tau_max`; no claim accepts the record and `build-md.log` prints it. A
claim is a build-md instruction like `cv_generate`: it is not carried into `resolved.config` or a
generated input, and `md-run` refuses it by name in an input. Adding the keys moved the resume
gate's `SCHEMA_VERSION` to 4, without which every 0.6.0 ladder would have been refused on resume.

**The compact form `L01: <path>` is refused by name in 0.6.1**, with the explicit form printed as
the fix. The user asked for it when `L01` is an existing, unambiguous recorded instance alias, and
`md-tools-ligand-mapping/1` records no instance alias, so no key can qualify yet. This is a
deferred requirement, not a dropped one: adding an instance name to the mapping record is an S0
item, and when it lands the compact form is accepted for names that resolve to exactly one
instance. A `ligand_scaling_dict` key is a label; instance identity is `residue_key` plus the
one-based topology residue index.

### Torsion-exclusion files, `md-tools-torsion-exclusions/1`

~~~yaml
format: md-tools-torsion-exclusions/1
parameters: CHEMBL112/param_0123456789ab    # the package the atom names belong to
residue_name: TYL
central_bonds:
  - [C4, N1]
~~~

- Atom names are **package-local** (`LigandPackage.atom_names`), never global indices.
- The file is bound to a **parameter package**, not to a residue name: two packages can share a
  residue name, and names from one mean nothing in the other. A `parameters` that does not match
  the selected instance's package is refused.
- It **adds** protected central bonds. It never un-protects a bond the classifier protects —
  amide, aromatic, double bond, improper — and a listed bond that is not a central bond of any
  torsion in the package is refused rather than ignored.
- Its resolved contents and sha256 are copied into the selection record; the path alone is not
  provenance.

### The ladder's direct Python API: caller-supplied rungs are declared, recorded, never re-derived

**Decided by the user, 2026-09-19.** `md-run` and the generated scripts already read every rung from
a saved state named in the group file and verify it against `scaler.yaml`. The direct Python path
did not: `ReplicaRun._rung_systems` accepted a hand-built `LadderPreflight.rung_systems` after
checking only their count, and with no rungs at all it re-derived scaled rungs at run time. Both
contradict "a scaled Hamiltonian is built once, as a file, never re-derived".

- **Run-time re-derivation is refused outright**, on every path. No plan, no ladder.
- **Caller-supplied rungs require an explicit declaration**: `rung_source="caller-supplied"` and a
  non-empty `rung_source_reason`. Without it, supplied `rung_systems` are refused. The saved-state
  preflight sets `rung_source="saved-states"`; a caller cannot set that value.
- **Every rung's origin is recorded, in `restart.json`**: `rung_source`, `rung_source_reason`, and
  per rung its canonical System digest (`rest2.identity.system_fingerprint`, stable under
  XmlSerializer round trip), its tau, and — compared against the `scaler.yaml` states — either
  `saved-state <i> (verified)` or `caller-modified`. Origin is verified on the rung BEFORE ladder
  restraints are added, and a restrained rung says so. A caller-modified rung is allowed — a biased
  auxiliary rung is a legitimate method — but it is never presented as a saved state.
  `solute.yaml` is content-addressed and already records every saved state's sha256; on the
  saved-states path it stays byte-identical, so no existing ladder becomes a stale helper.
- **Finding the saved states for a direct caller**, in order: an optional
  `LadderPreflight.saved_states_record`; else the `scaler.yaml` of `files.system` when that is a
  saved state; else `<dir of files.system>/REST2/scaler.yaml`, accepted only when its
  `source.system_sha256` equals `files.system`'s. None found: caller-supplied rungs are refused by
  name, because they cannot be checked.
- Count, particle number, masses and constraints are checked against the saved states' System.
- The declaration and each rung's origin are part of the ladder identity (`rungs`), so resuming
  with a different set of rungs, a different declaration or a different reason is refused. A stored
  identity with no `rungs` entry — every pre-0.6.1 ladder — agrees through a named compatibility
  branch if and only if the current ladder is `saved-states` with every rung verified, which is
  exactly what such a ladder ran.

The first user of this path is hpREST2's OPES-REST2 (six saved-state tau rungs plus one hot rung
carrying a bias force). It must keep working with the declaration added and nothing else changed.

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

### Endpoint conventions (decided 2026-09-19)

- **A unique group keeps its internal nonbonded interactions at every lambda.** Pairs and
  exceptions wholly inside an A-only or B-only group stay at full physical strength, dummy end
  included — the Amber convention, and the one that keeps a decoupled group a physical fragment.
  Like the retained bonded terms, the internal term is separable and cancels between legs; the
  plan's `endpoint_accounting` names it (`internal_nonbonded`), and `factorization_check`
  demonstrates that it factorizes. There is ONE definition of an endpoint System: the plan's,
  including those terms. The Hamiltonian's U(0) and U(1) equal it to <1e-8 kJ/mol.
  "Internal" is defined per **connected** unique group — the atoms of one side's unique set that
  are bonded to each other and hang off one anchor — not over a side's whole unique set. Pairs
  between two groups on different anchors depend on the core's conformation, so they cannot
  factor out; they are zero at the dummy end like every other dummy interaction. At the dummy end
  a group's internal non-excluded pairs are carried by a `CustomBondForce`
  (`UniqueGroupInternalNonbonded`: vacuum Coulomb plus LJ, no cutoff, no PME), identical in every
  leg of a cycle, which is what lets them cancel; its internal exceptions stay in the
  `NonbondedForce`. S2's plan is the definition, and S3's Hamiltonian reproduces it.
- **Junction bonded terms are retained at both ends: `junction_policy: retain-all` is the default**
  (decided 2026-09-20, jointly by S2 and S3, after the M2 campaign failed its gates). A dummy's
  junction bonded terms stay at full physical strength in BOTH end states, as pmemd
  (`gti_bat_sc = 0`) and OpenFE do, so they contribute exactly nothing to `dU/dλ` and cancel
  between the legs of a cycle.
  Why it changed: removing them at the dummy end put their stiff energy ON the λ path — measured
  `dU/dλ_bonded` of +611 kJ/mol at one end and −238 at the other, 100% of it junction terms and
  0.00 from the core, of which +611 was ONE angle at 610.585 kJ/mol. Sixteen of eighteen windows
  could not resolve it, and four of nine legs fell below the overlap floor. With the terms retained,
  every λ-dependent bonded slot goes to zero (verified independently by both sessions; endpoints
  still recover to 2e-10 kJ/mol).
  What it costs: OpenFE's documented dummy-group limitation returns — `Z_dummy` depends on the core
  conformation, a spread of 1.757 kJ/mol between two deliberately distant conformations. That
  spread is NOT a ΔΔG bias: the bias is the difference of the MEAN of `W = −kT ln Z_dummy` between
  two legs. From `dW/dq = 0.073 kJ/mol` per degree on an angle whose thermal σ is 3.8°, it bounds
  at 0.02–0.05 kJ/mol against a 2.09 kJ/mol gate. **That is a sensitivity bound from a curvature
  measurement, not a sampled ΔΔG**, and it is confirmed or refuted by running the same edge and
  legs under both policies — the test that must pass before the next campaign.
  `separable` (the old behaviour) stays available and the policy is RECORDED in the plan, so every
  result states which construction produced it. Separability, if it is ever needed, is recovered by
  correcting the dummy end in post-processing from saved core frames — never by switching a term
  along λ, because present-at-one-end-and-absent-at-the-other IS the λ path.
- **Exceptions across the softcore/core boundary: `sc_boundary_14: scaled` is the default,
  confirmed by the user.** It is pmemd 20+ `gti_add_sc=1` behaviour and is consistent with the
  plan's dummy factorization. `unscaled` — the literal Amber18 manual 21.1.5 rule
  (`gti_add_sc=0`) — stays implemented, tested and selectable. The softcore functional form
  (Amber18 equations 21.5–21.7) and the boundary rule are two separately recorded facts; no
  record calls the combination simply "amber18".
- **Long-range dispersion** is each end state's own OpenMM correction, mixed linearly in
  `lambda_sterics`, with its derivative among the TI components. OpenMM's correction is not
  additive over particle subsets — grouped `CustomNonbondedForce` corrections come out as
  N/(N+1) of the pair tail — so a per-group correction is not a substitute.

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

### The alchemy package and the sample record

**Decided 2026-09-19.** All 0.7.0 alchemical code lives in `src/md_tools/alchemy/`. One owner per
file:

| file | owner |
|---|---|
| `alchemy/__init__.py` | S0 — a docstring only until integration; no re-exports |
| `alchemy/paths.py` | S0 as a **contract file** — the named state coordinates and `AlchemicalPath`, as S4 wrote them at `9d465a5`; changes are requested, not made |
| `alchemy/samples.py`, `estimators.py`, `restraints.py`, `cycles.py`, `windows.py` | S4 |
| `alchemy/topology*.py` | S2 |
| `alchemy/hamiltonian*.py`, `alchemy/softcore*.py` | S3 |

**`md-tools-alchemical-samples/1`** (`alchemy/samples.py`, S4) is adopted as THE shared sample
record: `potential_kj_mol[n, k]` evaluated at every state, `volume_nm3[n]` required under NPT,
`derivatives[component][n]` keyed by the component names of `paths.py`, `origin_state` as the only
provenance, and any rank/worker/GPU field refused on read. The reduced potential is computed from
it, never stored. S3's cross-state evaluation produces exactly this shape; S4's estimators consume
nothing else.

Agree, before S2/S3/S4 diverge, the callable interfaces for: construction, state setting,
energy and cross-state evaluation, and complete derivatives. Provide real miniature fixtures, not
only mocks — a mock cannot notice a missing PME reciprocal-space term.

### Exchange ladders: what a rung IS (decided 2026-09-21, S3 and S4, for 0.7.0 A3b and 0.7.1)

A REST2 rung and a lambda window are both "rungs" and they are not the same object. The table is
in `docs/development/0.7.0/lambda-exchange-design.md`; the part that binds both branches:

- **A rung is addressed by its INDEX `j`, with `(lambda_j, tau_j)` as its CONTENT.** Never by its
  lambda. Addressing by lambda is the obvious shortcut while tau is absent and it forces 0.7.1
  either to re-index every rung or to carry two addressing schemes — and two addressing schemes is
  how a walker is filed under the wrong state. One field now, a migration later.
- **tau is BAKED into a serialised System; lambda is a Context parameter of ONE System.** So a
  REST2 ladder reads a saved state per group-file line, and **a lambda ladder has NO group file**:
  a per-rung `-s` would be the same path repeated K times, and a column that can only ever hold one
  value will eventually hold a wrong one. The rung's lambda belongs in the resolved configuration.
  0.7.1's tent path needs BOTH mechanisms at once, since its neighbours differ by a Context
  parameter AND by a serialised System.
- **The invariant is satisfied genuinely, not by analogy.** "A scaled Hamiltonian is built once and
  never re-derived at run time" holds for a lambda window because there is nothing to derive:
  `set_state` sets parameters, with no reinitialise, no second System, no second Context and no
  coordinate copy.
- **Identity splits in two.** The LADDER's recorded identity answers "are these rungs the same
  experiment?" — end-state digests, plan digest, softcore settings including `sc_boundary_14`,
  kappa, PME grid, force groups — and every rung shares it by construction. The RUNG's identity is
  `context_parameters(state)` plus its index, which is already the one definition of a state and
  the dict the forces actually read. No lambda-aware second authority on what a state is.
- **Configurations are exchanged, not states**, as 0.6.1's ladder does, so STATE ↔ CONTEXT stays
  fixed and every per-state output is correct by construction. Swapping lambda between Contexts
  would make each Context follow the WALKER, and every per-state file would need re-routing at each
  accepted swap — bookkeeping that is invisible when it is wrong.
- **Ownership of A3b** (assigned 2026-09-21, after S3 asked): the lambda-ladder RUNTIME is S3's —
  it designed the shape and owns the per-rung Hamiltonian, and S4's hands are full with the TYK2
  campaigns. S4 owns the acceptance test's consumption, the per-state outputs and the tutorial
  evidence, and reviews the runtime. S0 owns the `md-run` / `build-md` surface and wires it once
  the runtime shape is settled. One module, one writer.
- **Two refusals, and both are about the SHAPE rather than the contents** (S3):
  a per-rung `-s` on a lambda ladder is refused EVEN WHEN the K paths are identical — the column
  cannot express a true statement about a lambda ladder, so its presence is the error, not its
  contents, exactly as `-s` on the command line is refused for a REST2 ladder even when it names
  the right file. And a rung whose recorded state disagrees with its Context parameters is refused:
  the lambda analogue of "tau 0 on a hot state", compared against `context_parameters(state)`, the
  one definition.
- **The ladder coordinate is ONE accessor, and the record carries BOTH coordinates** (S0 ruling,
  2026-09-21, on S3's A3b report). `driver.py` reads `protocol.tau` in seventeen places; for a
  lambda ladder every one of them wants "the ladder coordinate of rung i", which tau is not. The
  ruling is a generic `protocol.ladder_coordinates()` with a record that names its coordinate and
  can hold more than one -- NOT a second writer (a second storage path is the second-policy shape),
  and NOT a `tau` property returning zeros. **A lambda ladder recording `tau = 0` at every rung is
  indistinguishable from a REST2 ladder that never heated**, which is the flattening this contract
  exists to prevent; it is refused even though it would run today and produce structurally valid
  files. S3 raised it rather than doing it, which is what the freeze is for.
  **Design once, with 0.7.1 in view**: a FEP-REST2 rung has BOTH a tau and a lambda, so a field
  designed now to hold one number would be designed again in 0.7.1. The record change is **v3 -> v4**
  -- `md_tools.remd.storage.SCHEMA_VERSION` is already `md-tools-replica-exchange/v3`, and S3
  caught S0 and itself both writing v2 -> v3. Harmless in a message, not in a migration note:
  `SUPERSEDED_SCHEMAS` keys on the exact string, so an entry written for the wrong version would
  never match a real file and the refusal it was meant to produce would never fire. It lands
  AFTER the TYK2 campaign. Until it does, a lambda ladder is not launchable end to end,
  and it fails LOUDLY at that line rather than being made to run.
- **A lambda ladder's `-s` is a hybrid System that something must WRITE** (open, assigned to S0's
  surface with S2's plan). The invariant is that every Hamiltonian a run integrates is written as
  a file before the run and never re-derived at run time -- as `build-top --rest2-scaler` writes
  `system_state<i>.xml`. Today nothing writes the hybrid: `combine-topology` writes a PLAN and
  `from_plan` builds the System in memory, so a runtime that built it from the plan would be
  exactly the re-derivation the invariant forbids. Ruled: `combine-topology` grows a System output
  and records its sha256 beside the plan's, `from_plan` becomes that writer's implementation rather
  than a run-time path, and ONE hybrid System serves every rung -- the rungs differ in Context
  parameters, not in file. It is a plan-schema change, so it lands with the coordinate record,
  after the campaign.
- **$MD_DATA is open for TUTORIAL datasets only** (the user, 2026-09-21, lifting part of the
  2026-09-19 sandbox). A simulation that SUCCEEDED and is CITED BY A TUTORIAL may be registered.
  Everything else about the sandbox stands: no other dataset is read, retrieved, altered or
  deleted, and no session reaches outside its own new dataset.
  The path is the CONTRACT's, not a new namespace: `$MD_DATA/{year}/tutorials/{data_name}/`, filed
  under the year the run completed, through `md-openmm data-register`. `$MD_DATA/dev/tutorials/...`
  as first written cannot be registered -- v2 has no `dev` segment and no month segment -- so
  registering there would have failed, or worse, written files that no `dataset.yaml` describes.
  **Registration is WRITE-ONCE, and the `data_name` goes to the user for approval before it
  happens** -- one approval per dataset, with the notes it will carry, since neither can be
  corrected afterwards.
  **A DATASET HAS NO ALIAS FIELD** (S1's correction to S0, 2026-09-21): v2 `Dataset` is
  dataset_id, path, year, project_name, data_name, role, system, created_at, created_by, status,
  origin, software, components, derived_from, completed_at, archived_at, notes. `solute.aliases`
  is a LIGAND PACKAGE field, set when `build-top --parameterize` creates the package, and that is
  where the write-once findability hazard lives -- a package registered without aliases is
  permanently unfindable by name, and no tutorial sets the field. So a dataset's findability rests
  on its `data_name` and `notes`; a ligand package's rests on aliases decided before the build.
  Two different write-once traps, and conflating them hides the real one.
- **A RELAYED approval is not an approval, for anything write-once or outside the worktree** (S2,
  2026-09-21, and adopted). The coordinator relays what the user decided in good faith, and that is
  enough for ordinary work; it is NOT enough to register a dataset, which is irreversible and
  leaves the repository. The owning session drafts the `data_name` and the full alias list, sends
  it up for the user, and registers only after the USER tells it directly. A rule that is set aside
  the first time the relay is probably right was never a rule.
- **Reproduce, do not register, what git already carries.** S2's TYK2 fixture commits the inputs,
  the prepared structures, the parameter packages and the nine build records (~2.8 MB) and
  deliberately does NOT commit the built Systems (56 MB), which `build_tyk2_fixture.py` rebuilds in
  about two minutes each, refusing if any input has moved. Registering those Systems would put a
  second, unverifiable copy of committed evidence in `$MD_DATA`, leaving a reader two sources for
  one artefact and no rule saying which is authoritative. A tutorial cites the committed fixture
  and its rebuild script. **An opened sandbox is not a reason to find something to register.**
- **The estimator must not be able to tell that a ladder produced the data** (S4's A3b acceptance
  plan, 2026-09-22, adopted). With exchange disabled a lambda ladder must reproduce independent
  fixed-lambda windows, and MBAR must consume its rows through NO ladder-specific branch. If the
  estimator needs to know how the samples were generated, the record is not the contract -- the
  sampler and the analysis have agreed privately about something the file does not say, and the
  next reader of that file cannot reproduce the result.
- **A comparison is published either way.** A3b.6 asks whether exchange over lambda is worth its
  cost: the same edge with and without exchange, at the SAME total sampling, reported whichever
  wins. A campaign that reports only the configuration that won is not a comparison, and it is the
  shape that makes a feature look justified forever after.
- **An exchange attempt uses `energy` only.** `derivative_components` is TI's consumer and is not
  part of an attempt: pairing a derivative at one state with energies at two is the class of error
  the AIS two-probe separation exists to prevent.

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
- No session reads or writes the machine's `$MD_DATA` (the user, 2026-09-19). Package and
  catalog code is exercised against a temporary root with `MD_DATA` set to it; a registered
  package is never a fixture.
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

**Status 2026-09-19:** none of the three shared fixtures is built yet. S4 is working against its
own analytic fixture, `tests/data/alchemy_s4/harmonic_staged_v1.json` — a harmonic model with a
closed-form dG — named so it cannot be mistaken for the shared one. That is the right call and it
stays S4's. The shared frozen-energy fixture will be written in the sample-record shape above once
S3 has a molecular Hamiltonian that can produce it; until then it is BLOCKED on A2, not missing.

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
   *Registration is BLOCKED (sandbox) for this wave:* no session may touch the machine's
   `$MD_DATA`. Registration code paths are tested against a temporary `MD_DATA` root only, and
   the real registration row stays BLOCKED until the user lifts the restriction.
7. **Regression** — existing cMD, whole-solute REST2 and two-state AIS keep their contracts, on
   the candidate integration commit.

Tolerances are calibrated against documented precision and reference error **before** the
comparison, and recorded. A tolerance is never weakened after a failure, and a failing test is
never deleted to close a gate: the implementation is fixed and the affected checks rerun.

**Missing CUDA, AMBER or license access is BLOCKED evidence.** A skip is not a pass, and a
successful CPU run is not CUDA evidence — a stale MPS pipe directory once made an entire 110-test
GPU lane skip and exit 0.
