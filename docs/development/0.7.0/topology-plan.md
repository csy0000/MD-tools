# The alchemical topology plan (A1, callable layer)

**Under construction, not in any release.** This describes the callable layer on
0.7.0 line. `md-openmm combine-topology` (S0's surface, `md_tools.build.combine`) runs it; this
page describes the callable layer underneath.

## What it is

```python
from md_tools.alchemy.topology import Environment, build_topology_plan, load_plan
from md_tools.alchemy.topology_mapping import AtomMap

plan = build_topology_plan(package_a, package_b, AtomMap.from_pairs(package_a, package_b, pairs),
                           Environment.from_files("built.xml", "built.pdb", selector,
                                                  record="built.log"),
                           mode="hybrid")          # or "single", "dual"
plan.write("plan/")                                # plan.json, system_{a,b}.xml, combined.pdb,
                                                   # positions.npy -- a new directory, never over
plan = load_plan("plan/", package_roots=[catalog]) # every digest re-verified
```

`plan.system_a` and `plan.system_b` are the two endpoint Systems; `plan.common`, `plan.a_only`
and `plan.b_only` are the particle sets (frozensets of plan indices); `plan.record` is the JSON
record, schema `md-tools-topology-plan/1`, and `plan.sha256` its digest.

The environment's build record is required: the 1-4 scales a System applies to its ligand are
read from its `nonbonded_compatibility` (an OPC build applies 0.833333, not 5/6), never inferred,
and both endpoints' 1-4 exceptions are given that applied scale (`scaled_table`).

Inputs are two ligand parameter packages (`md_tools.ligands.LigandPackage`, loaded and verified),
an explicit atom map in package-local indices or names, and ONE environment -- a built System and
topology holding endpoint A, with a `LigandSelector` naming that one residue. Endpoint B never
has its own environment: two independently solvated boxes are not atom-matched.

## The index space

Every environment particle keeps its index. Endpoint-B-only atoms are appended as one new
residue in a new chain. A one-based residue index or an atom index resolved against the
environment before combination selects the same atoms after it; `record.numbering` states this,
with the source and combined topology digests.

## The three representations

| mode | particles | map |
|---|---|---|
| single | one evolving representation: every atom of one endpoint is mapped | element changes allowed, a hydrogen may map to a heavy atom where no constraint forbids it |
| hybrid | mapped core + A-only + B-only, all separate | no hydrogen <-> heavy atom |
| dual | no shared particle; B appended whole | used only to superpose B and to define the restrained centroid groups |

Separated topology is refused by name.

## Endpoint Systems

Identical particles, masses, constraints and force layout, term slot for term slot; only
parameter values differ. System A is the environment System unchanged, plus appended particles
and terms. In System B the core carries package B's charges and Lennard-Jones, the A-only atoms
are dummies.

- **Dummies** have charge 0 and epsilon 0 (sigma kept) and every exception touching them is zero,
  EXCEPT inside their own unique group: no nonbonded interaction with the physical system.
- **A unique group's internal nonbonded interactions stay physical at its dummy end** (S0 ruling,
  2026-09-19, the Amber convention: interactions among the disappearing atoms are not changed).
  Its own exceptions keep their physical values in the NonbondedForce; its non-excluded internal
  pairs are carried by one `CustomBondForce` named `UniqueGroupInternalNonbonded` (vacuum
  Coulomb, constant 138.93545764438198, plus Lennard-Jones, Lorentz-Berthelot), present in both
  Systems and zero at the physical end, where the NonbondedForce computes them. Per connected
  group, never across two groups: two groups on different anchors are separated by physical
  coordinates, and a pair between them would not separate. In dual topology each whole ligand is
  one group. `record.nonbonded.unique_group_internal` lists the groups and pairs.
- **A-only x B-only** pairs are explicit zero exceptions in both Systems (`record.nonbonded.
  exclusions`): they never interact at any lambda. In dual mode that is every A-B pair.
- **Bonded terms** are laid out once: A's terms in their environment slots, B's extras appended.
  A term present at one endpoint only has force constant 0 at the other. A torsion slot is
  shared only when periodicity AND phase agree. Each slot is in `record.terms` with its role
  (`core`, `a-only`, `b-only`), both parameter sets and `at_dummy_end`.
- **Dual** adds one `CustomCentroidBondForce` (`0.5*k_restraint*distance(g1,g2)^2`, geometric
  centroids, default 1000 kJ/mol/nm^2), identical at both endpoints.

### The dummy junction: two policies, and why the default changed

A dummy group's bonded terms to the core can be kept at the end where the group is a dummy, or
removed there. Both choices are wrong in a way. `junction_policy` in the plan record says which
one produced it, because they are different physics and a result has to state its construction.

`retain-all` is the DEFAULT. Every bonded term touching the group keeps its physical parameters
at BOTH ends. No bonded term is then lambda-dependent, `dU/dlambda` has no bonded part at all,
and the dummy stays in the geometry its own force field gives it. This is also pmemd's default
(`gti_bat_sc = 0`).

`separable` is the single-anchor rule of Fleck, Wieder and Boresch (JCTC 2021, 17, 4403), which
addresses the limitation OpenFE's hybrid factory documents -- that keeping every dummy-core term
can bias results, because the dummy partition function then need not separate:

- each unique group G hangs from ONE bond D1-P1 to the core (a group with two attachments --
  a ring growing or a partially mapped ring -- is refused by `validate_map`);
- P2 is a core neighbour of P1 and P3 a core neighbour of P2 other than P1, heavy atoms first,
  then lowest package-local index -- a property of the package, so every leg of a cycle picks
  the same frame;
- where G is a dummy, a term touching G is RETAINED only if its atoms are within G + {P1, P2}, or
  it is the torsion D1-P1-P2-P3. Every other term touching G is REMOVED there (force constant 0)
  and stays at full strength where G is physical.

Every retained term is then a function of G's coordinates in the (P1, P2, P3) frame only, so the
dummy integral is independent of the physical coordinates. `factorization_check` demonstrates it
at construction by arithmetic: for ethane -> chloroethane the retained dummy energy moves by
~1e-13 kJ/mol when the physical atoms move, against [411.7, 471.5] kJ/mol for the keep-everything
set.

**The rule shipped first, and it was wrong on the lambda path.** It protected a real thing --
measured below -- at a cost that had not been measured. Once a bonded term is removed at one end,
its energy IS the bonded integrand; and with the term off, the dummy is free to explore exactly
the geometries that term forbade, so the integrand grows without bound. That is not a subtlety in
the third decimal. On the M2 campaign it was `dU/dlambda_bonded` = +611 kJ/mol at s=0 and -238 at
s=1, 16 of 18 vacuum windows unable to resolve their endpoints, repeats scattering 0.80 kcal/mol
against a claimed sigma of 0.09, and a cycle that did not close. The +611 was ONE term: the
removed junction angle H4-C2-Cl_dummy, worth 610.585 kJ/mol at the built coordinates.

**What retaining costs, quantified.** Under `retain-all` OpenFE's documented limitation returns,
and the honest question is how large it is in a ddG rather than whether it exists. Those are
different quantities:

| measured | what it is |
|---|---|
| 1.757 kJ/mol | the SPREAD of `-kT ln Z_dummy` between two core conformations differing by an 8% C-H stretch and ~0.027 nm displacements -- a deliberately large distortion, chosen to put the dependence above noise |
| 0.073 kJ/mol per degree | `dW/dq` for the H4-C2-C1 bend the retained junction angle couples to |
| 0.146 kJ/mol per degree^2 | `d2W/dq2` for the same coordinate |

A ddG bias is neither of the first two: it is the difference of the MEAN of `W = -kT ln Z_dummy`
between the two legs of the cycle. That angle has k = 559 kJ/mol/rad^2, so its thermal sigma is
3.8 degrees, and a difference in its mean between a solvated and a complexed core of a whole
degree -- generous for an angle that stiff -- costs 0.07 kJ/mol. The convexity term, at a 5%
difference in its variance between legs, costs 0.05 kJ/mol. The agreement gate is 0.5 kcal/mol =
2.09 kJ/mol.

**That is a sensitivity bound, not a measured ddG.** It assumes the core's internal distribution
near the anchor is nearly environment-independent, which is what a stiff angle buys and not what
has been sampled. What converts it into evidence is the two-policy comparison: the same edge and
the same legs, ddG computed twice, so the construction is the only difference. If the two agree
within their combined error the bound is confirmed; if they do not, the disagreement IS the bias,
measured rather than bounded. Amber's manual reaches the same conclusion from the other
direction, calling the alternative "not ... a significant effect in most cases".

`factorization_check` therefore REFUSES above its tolerance under `separable`, and REPORTS the
same number under `retain-all`. Separability is not lost for good either way: it can be recovered
at the dummy end in post-processing, as `<W>` over the saved core frames by the same quadrature --
no Hamiltonian change, no lambda component and no extra sampling. Not implemented in A1; recorded
because `retain-all` does not close that door.

## The automatic map

`md_tools.alchemy.topology_mapping.propose_map(package_a, package_b, mode)` returns an `AtomMap`
and a report, method `rdkit-fmcs-heavy/1`:

1. RDKit FMCS over the heavy atoms: elements equal, bond orders exact, rings match only rings and
   only complete rings, 10 s timeout (a timeout is a refusal, never a partial map).
2. Every placement of that substructure in A and in B is enumerated. For each, B's conformer is
   superposed on A's mapped heavy atoms and the hydrogens of each mapped heavy pair are paired by
   distance (Hungarian assignment).
3. Every candidate must pass `validate_map`; the rejected ones and why are in the report.
4. The largest survivors are compared by canonical atom ranks with ties unbroken. If they are not
   all related by a symmetry of A or B, the choice changes the transformation, and the map is
   REFUSED as chemically ambiguous, listing the alternatives. Example: ethanolamine -> propane,
   where A's N-side carbon can sit on propane's end or middle carbon.

Single topology is explicit-map only. An automatic map is as reviewable as an explicit one: it
is an ordinary `AtomMap`, stored in the plan's `atom_map` record with both directions.

## Two legs of one cycle

`record.ligand_hamiltonian_sha256` digests everything a plan says about the ligands'
Hamiltonian, in package-local atom identities so it does not depend on the environment's
numbering: packages, map, mode, applied 1-4 scales, constraint policy, dummy groups and frames,
every ligand term and exception with both parameter sets, the internal pairs, the exclusions and
the dual restraint. `matched_legs(plan_1, plan_2)` refuses two plans whose digests differ and
names what differs; S4's cycles refuse legs whose digests differ or are missing. A vacuum leg and a
TIP3P leg built from the same packages, map and mode (and the same constraints) match.

The vacuum leg comes from `build-top` with `solvent.model: vacuum` (ligand only, NoCutoff, no
box, constraints as configured, the usual record), so it goes through `Environment.from_files`
exactly as a solvated leg does; `tests/data/alchemy/vacuum-v1/` is the ethane one, and it is the
matched vacuum leg of the ethane TIP3P cycle.

The record carries its schema, `md-tools-topology-plan/<n>`, and it moves whenever a field is
added or removed — because `plan_sha256` covers the whole record, so a field addition changes the
digest of plans already written. `check_plan_digest(stored_digest, stored_record, plan)` tells a
caller which case a mismatch is: the schema moved and the plan's identity (ligand Hamiltonian,
endpoints, map) did not, so rebuilding changes no physics; or it is a different plan, and the
first differing field is named. `load_plan` refuses another schema with the same explanation.

`record.environment.solvation` ("explicit", "implicit" or "vacuum") comes from the build
record's `solvent.treatment`, or is stated for an in-memory environment; it is never inferred
from periodicity, and a stated value that contradicts the System's periodicity is refused. It is
part of `plan_sha256` (a vacuum leg and a solvent leg are different plans) and not of
`ligand_hamiltonian_sha256` (they are legs of one cycle). The window runner derives
`vacuum_leg` from it.

Hydration cycles with OPC water are refused until the vacuum leg can apply the solvent leg's
scale: an OPC leg applies 0.833333 to the ligand's 1-4 pairs and a vacuum leg the package's 5/6,
which makes the ligand's intramolecular Hamiltonian a different function in the two legs.

## Constraints and masses

The environment's constraint policy is read from its ligand (HBonds, AllBonds or None) and
applied to endpoint B. A constraint that would appear or vanish along the path, or change length,
is refused -- a constraint has one length. An environment whose ligand has only X-H bonds cannot
say whether it was built with HBonds or AllBonds; it is refused when endpoint B has a heavy-atom
bond (tested with methane, `tests/data/alchemy/xh-only-v1/`).

Masses are the environment's for every existing particle (endpoint A's for the core) and package
B's for appended ones; core mass changes are recorded. Masses do not enter the configurational
free energy. An environment with repartitioned hydrogen masses is refused: appended atoms would
need the same repartitioning, which is not implemented.

## Pressure coupling

A barostat in the environment is CARRIED THROUGH, not refused: it is a passive force, copied
identically into both endpoint Systems (`test_a_barostat_environment_is_carried_unchanged_to_both_
endpoints`). This is a construction statement only. Contract section 6 requires ensemble-correct
state energies for alchemical NPT -- the reduced potential needs the pressure-volume term -- and
that is the Hamiltonian's and the executor's to supply. AIS refuses a barostat; that rule is
AIS's and does not transfer here.

## Refused

Net charge change; different force field, charge method or charge backend between the packages;
core bond-graph change (ring opening/closure, rearrangement); ring-membership change; an inverted
stereocentre, or one with fewer than three mapped shared neighbours; hydrogen <-> heavy mapping
outside single topology; single topology with unmapped atoms on both sides; a dummy group with
more than one attachment; an environment whose ligand is not package A parameter for parameter;
HMR; forces other than the four standard ones on the ligand (GB, custom forces); CMAP on the
ligand; a NonbondedForce with global parameters or offsets (a scaled System); ligand virtual
sites; a constraint changing across the path.

## Endpoint recovery

`md_tools.alchemy.topology_recovery`:

- `audit_plan` (run by the builder): at each endpoint, the System restricted to that endpoint's
  physical ligand atoms reproduces the package table exactly -- every nonzero bonded term once,
  every exception, every charge and LJ pair; environment terms untouched; dummies inert; the two
  Systems agree in layout.
- `factorization_check` (run by the builder): above.
- `endpoint_accounting(plan, endpoint, reference_system, reference_to_hybrid)`: per force class,

  ```text
  E_endpoint(x) = E_reference(x_phys) + E_dummy(x) + dE_dispersion + E_restraint
  ```

  with E_dummy (retained bonded terms, plus each unique group's internal exceptions and pairs)
  and E_restraint computed in numpy from the record and dE_dispersion the change in
  OpenMM's dispersion correction. **That correction averages over every particle, zero-epsilon
  dummies included**, so a dummy shifts it (-1.2e-4 kJ/mol for one dummy in the fixture box). It
  is a real term, not noise, and a Hamiltonian that moves LJ out of `NonbondedForce` changes it.

The tests build the reference endpoint from scratch with `ForceField("amber14/tip3p.xml")` plus
the package's ffxml, and hold every force class to 1e-7 kJ/mol on the Reference platform, after
asserting the raw difference is large enough that the check cannot pass by accident.

## The command

`md-openmm combine-topology` is S0's surface over this layer: `md_tools.build.combine`, input
format `md-tools-combine-topology/1`, output directory given as `-odir` on the command line,
`--check` creating nothing. Its module docstring and schema are the authority for the input; the
map key takes exactly one of `file` (explicit pairs, or a stored `AtomMap` record) or
`automatic: true` (`propose_map`, whose proposal is written for review beside the plan as
`<odir>.map.yaml` and, given back as `map: {file: ...}`, reproduces the same `plan_sha256`).

## Decoupling plans for absolute binding

**Implemented**, to the design S0 and S3 reviewed. `build_decoupling_plan(package, environment,
*, restraint=None)`. ABFE had no construction to run: every other plan here is A -> B with both
endpoints real, and S3 softens only the unique particles, so decoupling a ligand COMMON to both
endpoints would be linear in lambda and diverge at the endpoint.

Measured on the fixtures (Reference platform): at lambda 0 the plan reproduces the environment
(residual < 1e-7 kJ/mol); at lambda 1 `E_B = E(environment without the ligand) + E(the ligand's
own Hamiltonian) + dE_dispersion` closes to 1.8e-12 kJ/mol on the TYK2 solvated leg, and moving
the decoupled ligand anywhere -- 1 nm, 4 nm, outside the box -- changes the energy by exactly
zero. Every ligand term has the same parameters at both ends, which is what makes its
intramolecular Hamiltonian lambda-independent.

### 1. Representation: an explicit `mode: "decoupling"`, with no endpoint B

The alternative -- the whole ligand as an A-only unique group of an otherwise empty map -- does
not work and should not be made to. An atom map with no pairs is refused (`AtomMap.from_pairs`:
"the map is empty"), a unique group must hang from exactly ONE bond to the core, and here there
is no core to hang from. Forcing it would mean special cases in the map validator, the junction
rule and the term layout, each of which exists to describe a transformation between two real
molecules.

So: `mode: "decoupling"`, and the record says what it is rather than encoding it as a degenerate
map. One package, not two. Proposed callable:

```python
plan = build_decoupling_plan(package, environment, *, restraint=None)
```

The particle sets stay exactly what S3 already consumes: `common` empty, `a_only` the ligand's
particles, `b_only` empty. System A is the environment System unchanged. In System B every ligand
particle carries charge 0 and epsilon 0 and every exception to the environment is zero, which is
the same "dummy" treatment as everywhere else -- so S3 softens `a_only` over the whole path with
no new rule, and `U(0)`/`U(1)` must equal System A / System B as for every other mode.

`endpoints.B` in the record becomes `{"absent": true, "reference": null}`; `mode` and the schema
version say why, and nothing else in the record changes shape.

### 2. "Decoupled" means the ligand keeps its own Hamiltonian

Its intramolecular terms stay PHYSICAL at both endpoints: every bonded term, and its own internal
nonbonded pairs and exceptions, exactly as S0 ruled for a unique group's internal terms. Only the
ligand-environment interactions vanish. That is the standard choice, and the standard-state
correction assumes it: at lambda 1 the ligand is the same molecule it is in the gas phase, so the
cycle's other leg cancels it. Annihilation (removing the intramolecular nonbonded too) is
REFUSED by name in v1, because it changes what the solvent leg must cancel and needs its own
derivation; it goes to the backlog.

The junction rule does not apply: there is no anchor and no physical neighbour, so no bonded term
couples the ligand to the environment at all. Nothing is dropped, and the whole ligand's
configurational integral separates -- as the dual-topology dummy ligand's already does.

### 3. Endpoint checks, both measurable

- **lambda 0 is the complex, exactly.** System A restricted to the environment's particles is
  the environment System, term for term, and the ligand carries its package's parameters -- the
  existing audit, unchanged.
- **lambda 1 is the ligand absent, and that is a measurement, not a claim.** Two independent
  checks, both against Systems built by OpenMM's force field rather than from the plan:

  ```text
  E_B(x) = E_environment_without_ligand(x_env) + E_ligand_in_vacuum(x_ligand) + dE_dispersion
  ```

  where the first is the environment built with the ligand residue deleted and the second is the
  ligand alone from its package; and, separately, **E_B is invariant under any rigid displacement
  of the ligand** -- translate it into the protein, into the solvent, out of the box, and the
  energy does not move. The second is what "non-interacting with the environment" means, stated
  so it can fail.
- **Net charge.** A charged ligand changes the box's net charge between the endpoints, which PME
  handles with a neutralising background whose free-energy contribution needs a finite-size
  correction this release does not implement. A decoupling plan for a ligand with non-zero net
  formal charge is REFUSED by name (the campaign's three TYK2 ligands are neutral).

### 4. Pairing the legs of the cycle (S0's ruling, 2026-09-20: by ROLE, not by mode)

Two different objects were being called a restraint, and the rule follows the difference:

| role | what it is | in a cycle |
|---|---|---|
| `alchemical-coupling` | part of the construction: dual topology's centroid restraint, which shapes the path | must be present and IDENTICAL in both legs; `matched_legs` refuses otherwise |
| `standard-state` | an external term whose free energy is computed and corrected: ABFE's Boresch restraint | may differ; at most one leg of a pair may carry one; both records are reported side by side for S4, never silently ignored |

`ligand_hamiltonian_sha256` therefore covers the ligand's OWN Hamiltonian only -- bonded terms,
internal nonbonded, parameters -- and NO restraint, in any mode: a restraint is not part of what
the ligand is. The digest keeps one meaning, dual topology keeps its guarantee, and a correct ABFE
cycle stops refusing itself. `record.restraints` is a list and every entry carries its role, the
atoms, the functional form and the constants. One field is deliberately not compared between legs,
`periodic`: it is a property of the box (a solvated leg takes the minimum image, a vacuum leg has
no images), both evaluate the same centroid separation for a molecule that does not straddle a
boundary, and requiring it to match would make every vacuum/solvent dual cycle impossible.

### 5. Restraints are S4's, and the plan records that they exist

The plan does not build a Boresch force -- that, its free energy and the standard-state
correction are S4's. What the plan records, because a decoupled ligand with no restraint wanders
off and the free energy diverges:

```yaml
decoupling:
  intramolecular: retained          # annihilation refused in v1
  restraint:
    required: true                  # an environment holding a protein requires one
    kind: boresch
    ligand_atoms: [...]             # plan indices, and their package-local identities
    environment_atoms: [...]        # plan indices, with chain/resid/name for review
    built_by: md_tools.alchemy.restraints
```

A decoupling plan in an environment that holds a protein and names no restraint atoms is
REFUSED: the construction cannot supply the restraint, but it can refuse to pretend the leg is
complete without one. In a ligand-only solvated or vacuum environment `required` is false, with
the reason recorded ("no binding site to leave").

### 6. What S3's Hamiltonian expects (confirmed, 2026-09-20)

S3 confirms that nothing new is needed on the Hamiltonian side: its softcore forces use
interaction groups, so a pair softens only when exactly one end is in the unique region, and
pairs INSIDE the region are excluded from the weighted Ewald sum and carried at full strength at
every lambda by its own internal force. The ligand's intramolecular Hamiltonian is therefore
lambda-independent by construction, which is what this design needs, and 32-38 atoms with
internal 1-4 and 1-5+ pairs is what the tail and n-pentane fixtures already exercise. Nothing
assumes `b_only` is non-empty.

**One thing the plan must NOT assert.** S3's builder does not read `plan.common`; it computes
common = all particles - `a_only` - `b_only`. In decoupling that is the whole environment, which
is exactly the region that must soften against the ligand -- while `plan.common`, which means
"ligand particles physical at both endpoints", is empty. The two sets are different by design and
no check may require them to agree.

### 7. What this does not change

Single, dual and hybrid are untouched. The refusals, the junction rule, the internal-nonbonded
convention, the applied 1-4 scales, `plan_sha256` and `ligand_hamiltonian_sha256` all keep their
current meaning; `mode: "decoupling"` adds a fourth value and the schema version moves with the
`decoupling` block, batched before a campaign as the contract requires.

## Fixtures

`tests/data/alchemy/v1/` (see its README): ethane, chloroethane and ethanol packages (AM1-BCC,
openff-2.2.1) and ethane in TIP3P built by `build-top`. `tests/data/alchemy/internal-v1/`:
n-pentane, whose unmapped propyl group has internal 1-4 and 1-5 pairs. `xh-only-v1/`: methane.
`complex-v1/`: capped alanine and ethane in TIP3P (ff14SB). `complex-cmap-v1/`: the same with
ff19SB + OPC (CMAP; OPC's 0.833333 applied to every 1-4 pair).
`charged-v1/`: acetate and propanoate (both -1) and acetate + Na+ in TIP3P. `vacuum-v1/`: ethane
built by `build-top` with `solvent.model: vacuum`. Loaders in `tests/alchemy_fixtures.py`.
