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

### The dummy junction rule

OpenFE's hybrid factory keeps every bonded term between the dummy region and the core, and
documents that this can bias results because the dummy partition function need not separate.
This construction uses the single-anchor rule of Fleck, Wieder and Boresch (JCTC 2021, 17, 4403):

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
~1e-13 kJ/mol when the physical atoms move, and the keep-everything set moves by 412 kJ/mol.

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

## Fixtures

`tests/data/alchemy/v1/` (see its README): ethane, chloroethane and ethanol packages (AM1-BCC,
openff-2.2.1) and ethane in TIP3P built by `build-top`. `tests/data/alchemy/internal-v1/`:
n-pentane, whose unmapped propyl group has internal 1-4 and 1-5 pairs. `xh-only-v1/`: methane.
`complex-v1/`: capped alanine and ethane in TIP3P (ff14SB). `complex-cmap-v1/`: the same with
ff19SB + OPC (CMAP; OPC's 0.833333 applied to every 1-4 pair).
`charged-v1/`: acetate and propanoate (both -1) and acetate + Na+ in TIP3P. Loaders in `tests/alchemy_fixtures.py`.
